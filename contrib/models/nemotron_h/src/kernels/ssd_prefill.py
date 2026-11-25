"""Mamba-2 prefill kernels for NemotronH (custom NKI kernel 2, A06).

Two kernels share one chunked-SSD core:

* ``ssd_prefill_kernel`` (v1, "scan"): replaces ``NemotronHMamba2Mixer.ssd_chunk_scan``
  plus the ``D x`` term; input is the conv output.
* ``mamba_prefill_fused_kernel`` (v2, "fused", the default of ``NEMOTRON_H_KERNEL2=1``):
  everything between ``in_proj`` and ``out_proj`` of one prefill chunk: causal
  depthwise conv (kernel 4, continuing from the cached window) + bias + SiLU,
  ``dt = softplus(dt + dt_bias)`` masked to real tokens, the SSD scan, ``D x``
  and the gated RMSNorm. Input is the ``in_proj`` output, output is the
  ``out_proj`` input.

SSD math per head (chunk L = 128):

    a      = dt * A
    acum   = in-chunk inclusive cumsum of a
    y_i    = sum_{j<=i} (C_i . B_j) exp(acum_i - acum_j) dt_j x_j     (intra-chunk)
           + exp(acum_i) C_i . h_in                                   (state entering the chunk)
           + D x_i
    h_out  = exp(acum_last) h_in + sum_j exp(acum_last - acum_j) dt_j x_j B_j^T

Why kernels. (1) The torch scan works on ``[heads * chunks, L, d]`` tensors and
broadcasts B/C to heads; neuronx-cc lowers the layout changes to ~96 MB of small
DMA descriptors per layer per 2,048-token chunk (results/profile/baseline-analysis.md).
(2) At full depth (26+ layers) the compiler also spills and reloads ~1.3 GB per
Mamba block around the torch conv (A06 26-layer profile), which v1 leaves in
place; v2 moves the conv into the kernel. The NKI Library ``ssd`` kernel factors
``exp(acum_i - acum_j)`` as ``exp(acum_i) / exp(acum_j)``, which under/overflows
for this checkpoint (``-A`` up to 22,026). Here the exponent is
``min(acum_i - acum_j, 0)`` and the upper triangle is masked afterwards.

Layout: tokens on partitions, token-major inputs/outputs as the model has them.
``B^T``/``C^T`` come from TensorE identity matmuls (exact), ``C B^T`` is one
matmul per (chunk, group). The state is kept transposed, ``h^T [d_state, (head,
head_dim)]``, so ``y_off = C h^T`` and ``h^T += B^T (w dt x)`` are one matmul
each for the 8 heads of a group. ``acum`` is broadcast along partitions by a
one-hot matmul (exact). Matmuls feeding the recurrence are float32 (TensorE
native FP32); cumsums run on the Vector engine in float32. The v2 conv reads the
four tap windows straight from HBM (row-shifted DMAs) and rounds to bf16 after
every multiply/add, as the torch bf16 path does.

LNC=2: one SSM group per physical core (2 groups per rank at TP=4).
"""

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

L = 128  # SSD chunk (config chunk_size)
# R2-03: pass the conv weight to the split kernel as [K, conv_dim] so each tap's per-channel weights are one contiguous
# row (A06 read [conv_dim, K] with a stride-K 2-byte gather broadcast to 128 partitions). "0" = A06 layout.
import os as _os
CONV_KMAJOR = _os.environ.get("NEMOTRON_H_MAMBA_PREFILL_CONV_KMAJOR", "1") == "1"
# R2-03m scan-phase options (comma list, default none; each keeps the arithmetic of every element):
#   "convpool" the conv's bf16 multiply/add tensor_tensor ops on the GpSimd engine instead of Vector (Vector is the
#              busiest engine of the scan phase, GpSimd is idle there).
#   "sp1"      softplus(dt + dt_bias) for all chunks of the group in one activation before the chunk loop (one
#              activation-table load instead of one per chunk).
#   "zearly"   silu(z) of the gated norm right after the conv's silu (same activation table) instead of after the scan.
# R2-03n options (fewer Vector instructions per 128-token chunk; the device profile showed ~0.24 us of fixed cost per
# Vector instruction, so the scan phase is bound by the instruction count, ~77 per chunk):
#   "bc"   per-head broadcasts batched over the group's heads with stride-0 access patterns: x*dt and x*dt*w (16 -> 2
#          ops), the y tail exp(acum)*C.h_in + D*x + intra (16 Vector + 8 Scalar -> 4 Vector), the state update
#          (8 -> 2), and B^T / C^T copied out of one PSUM tile (2 -> 1). Same fp32 operations in the same order.
#   "eb"   the per-head min(acum_i - acum_j, 0) Vector op folded into the exp: a -1e30 mask over i < j is put into the
#          PSUM tile by a bf16 identity matmul before the one-hot broadcast of acum_i accumulates onto it, and
#          -acum_j is the activation's per-partition bias (acum is non-increasing, so min(., 0) was a no-op for i >= j);
#          exp(.) * (C B^T masked) for 4 heads in one op (8 -> 0 + 2 ops).
#   "cv"   the conv's four tap multiplies in one [L, K, 768] op (8 -> 5 ops), same bf16 roundings and order.
#   "dn"   the gated norm's rsqrt deferred: y * silu(z) and its sum of squares per chunk, then var / rsqrt / scale for
#          DN_GROUP chunks at once (one rsqrt activation-table load per DN_GROUP chunks instead of one per chunk).
#   "xtb"  the merged kernel's in_proj copies its x^T PSUM tiles out 4 at a time (21 -> 6 copies). (Was part of "bc" in
#          cfa2504..f3de1b0.)
#   "rb"   (with eb) acum_i broadcast for 4 heads by one matmul: constant all-ones stationary, block-diagonal moving
#          bd[k, (h, i)] = (k == h) * acum[k, i] built by one GpSimd op (8 per-head one-hot matmuls + LDWEIGHTS -> 2).
#   "se"   (with bc) the state update h^T = exp(acum_last) h^T + B^T(w dt x) emitted right after the two matmuls that read
#          h^T, before the per-head decay loop, so chunk c+1's y_off matmul is not queued behind chunk c's y tail.
#   "cp"   software pipeline: chunk c+1's conv / silu / dt are emitted before chunk c's scan (the merged kernel's in_proj
#          runs two chunks ahead), so the conv's Vector work is not queued behind chunk c's gated norm.
# "none" = the R2-03m kernel. Default on this branch: bc,eb,cv (SCAN_DEFAULT).
SCAN_DEFAULT = "bc,eb,cv"  # R2-03n default; "none" = the R2-03m kernel
_scan_env = _os.environ.get("NEMOTRON_H_MAMBA_PREFILL_SCAN", SCAN_DEFAULT)
SCAN_OPTS = set() if _scan_env == "none" else set(filter(None, _scan_env.split(",")))
if not SCAN_OPTS <= {"convpool", "sp1", "zearly", "bc", "eb", "cv", "dn", "cp", "rb", "se", "xtb"}:
    raise ValueError(f"NEMOTRON_H_MAMBA_PREFILL_SCAN: unknown options {sorted(SCAN_OPTS)}")
NEG_BIG = -1.0e30
DN_GROUP = int(_os.environ.get("NEMOTRON_H_MAMBA_PREFILL_DN_GROUP", "8"))


def _hv(t, F, off, n, w):
    """``[L, n, w]`` view of columns ``off : off + n * w`` of a 2-D tile ``t`` with ``F`` free elements per partition."""
    return t.ap(pattern=[[F, L], [w, n], [1, w]], offset=off)


def _bcv(t, F, off, n, w):
    """``[L, n, w]`` view of columns ``off : off + n`` of ``t``, each repeated ``w`` times (stride-0 broadcast)."""
    return t.ap(pattern=[[F, L], [1, n], [0, w]], offset=off)


def _consts(hpg, tri):
    ident = nl.shared_identity_matrix(128, dtype=nl.float32)
    identb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=identb, src=ident)
    tri_sb = nl.ndarray((L, L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tri_sb, src=tri[0:L, 0:L])
    ones = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    # sel_last[k, p] = 1 iff k == L-1: matmul broadcasts row L-1 to all partitions.
    sel_last = nl.ndarray((L, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sel_last, data=ones[0:L, 0:128], op0=nl.multiply, operand0=ident[0:L, L - 1 : L])
    # onehot[k, h, p] = 1 iff k == h: matmul broadcasts head h's row to all partitions.
    onehot = nl.ndarray((hpg, hpg, 128), dtype=nl.float32, buffer=nl.sbuf)
    for h in range(hpg):
        nisa.tensor_scalar(
            dst=onehot[0:hpg, h, 0:128], data=ones[0:hpg, 0:128], op0=nl.multiply, operand0=ident[0:hpg, h : h + 1]
        )
    k = dict(identb=identb, tri_sb=tri_sb, ones=ones, sel_last=sel_last, onehot=onehot)
    if "eb" in SCAN_OPTS:
        # R2-03n: negm4[j, (q, i)] = NEG_BIG if i < j else 0, for 4 heads side by side (one PSUM bank of fp32).
        negm = nl.ndarray((L, L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negm, data=tri_sb, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=-NEG_BIG)
        negm4 = nl.ndarray((L, 4 * L), dtype=nl.bfloat16, buffer=nl.sbuf)
        for q in range(4):
            nisa.tensor_copy(dst=negm4[0:L, q * L : (q + 1) * L], src=negm)
        k["negm4"] = negm4
    return k


def _group_params(a_neg, d_skip, h0, hpg):
    a_col = nl.ndarray((hpg, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_col, src=a_neg.ap(pattern=[[1, hpg], [1, 1]], offset=h0))
    d_sb = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=d_sb, src=d_skip.ap(pattern=[[0, L], [1, hpg]], offset=h0))
    return a_col, d_sb


def _load_state(init_state, h0, hpg, P, N, gi):
    """Transposed state h^T [N, (head, P)] of one group."""
    stT = nl.ndarray((N, gi), dtype=nl.float32, buffer=nl.sbuf)
    for h in range(hpg):
        s0 = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s0, src=init_state[h0 + h, 0:P, 0:N])
        s_ps = nl.ndarray((N, P), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=s_ps, data=s0, engine=nisa.tensor_engine)
        nisa.tensor_copy(dst=stT[0:N, h * P : (h + 1) * P], src=s_ps)
    return stT


def _store_state(stT, st_out, h0, hpg, P, N):
    for h in range(hpg):
        f_ps = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=f_ps, data=stT[0:N, h * P : (h + 1) * P], engine=nisa.tensor_engine)
        f_sb = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=f_sb, src=f_ps)
        nisa.dma_copy(dst=st_out[h0 + h, 0:P, 0:N], src=f_sb)


def _scan_chunk(k, xs, xo, bs, bo, cs, co, dts, stT, a_col, d_sb, hpg, P, N, gi):
    """One 128-token chunk of one group: returns ``y + D x`` [L, gi] fp32; updates ``stT`` in place.

    x, B, C are columns ``xo:xo+gi``, ``bo:bo+N``, ``co:co+N`` of the bf16 SBUF tiles ``xs``,
    ``bs``, ``cs`` (tokens on partitions); ``dts`` [L, hpg] fp32.
    """
    identb, tri_sb, ones, sel_last, onehot = k["identb"], k["tri_sb"], k["ones"], k["sel_last"], k["onehot"]
    BC, EB = "bc" in SCAN_OPTS, "eb" in SCAN_OPTS
    xF = xs.shape[1]
    # B (fp32, tokens on partitions) and B^T, C^T (d_state on partitions).
    bf = nl.ndarray((L, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bf, src=bs[0:L, bo : bo + N])
    if BC:  # R2-03n: B^T and C^T in one PSUM tile, one copy
        bct_ps = nl.ndarray((N, 2 * L), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=bct_ps[0:N, 0:L], stationary=bs[0:L, bo : bo + N], moving=identb)
        nisa.nc_matmul(dst=bct_ps[0:N, L : 2 * L], stationary=cs[0:L, co : co + N], moving=identb)
        bcT = nl.ndarray((N, 2 * L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=bcT, src=bct_ps)
        bT = bcT[0:N, 0:L]
        cT = bcT[0:N, L : 2 * L]
    else:
        bt_ps = nl.ndarray((N, L), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=bt_ps, stationary=bs[0:L, bo : bo + N], moving=identb)
        bT = nl.ndarray((N, L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=bT, src=bt_ps)
        ct_ps = nl.ndarray((N, L), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ct_ps, stationary=cs[0:L, co : co + N], moving=identb)
        cT = nl.ndarray((N, L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cT, src=ct_ps)
    # (C B^T)^T [j, i] = B_j . C_i, masked to i >= j.
    cb_ps = nl.ndarray((L, L), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=cb_ps, stationary=bT, moving=cT)
    cbm = nl.ndarray((L, L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=cbm, data1=cb_ps, data2=tri_sb, op=nl.multiply)

    # a = dt * A with heads on partitions, in-chunk cumsum along the free axis.
    dtT_ps = nl.ndarray((hpg, L), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=dtT_ps, data=dts, engine=nisa.tensor_engine)
    aT = nl.ndarray((hpg, L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=aT, data=dtT_ps, op0=nl.multiply, operand0=a_col)
    acT = nl.ndarray((hpg, L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor_scan(dst=acT, data0=ones[0:hpg, 0:L], data1=aT, initial=0.0, op0=nl.multiply, op1=nl.add)
    # acum with tokens on partitions, and acum_last broadcast to all partitions.
    acol_ps = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=acol_ps, data=acT, engine=nisa.tensor_engine)
    acol = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=acol, src=acol_ps)
    if EB:  # -acum_j: the per-partition bias of the decay exp
        nacol = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nacol, data=acol, op0=nl.multiply, operand0=-1.0, engine=nisa.gpsimd_engine)
    last_ps = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=last_ps, stationary=sel_last, moving=acol)
    lastb = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lastb, src=last_ps)
    expcol = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)  # exp(acum_i)
    nisa.activation(dst=expcol, op=nl.exp, data=acol)
    wd = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=wd, data1=lastb, data2=acol, op=nl.subtract)
    w = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)  # exp(acum_last - acum_j)
    nisa.activation(dst=w, op=nl.exp, data=wd)
    dtw = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dtw, data1=dts, data2=w, op=nl.multiply)
    cd = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)  # exp(acum_last), whole-chunk decay
    nisa.activation(dst=cd, op=nl.exp, data=lastb)

    xdt = nl.ndarray((L, gi), dtype=nl.float32, buffer=nl.sbuf)
    xdtw = nl.ndarray((L, gi), dtype=nl.float32, buffer=nl.sbuf)
    if BC:  # R2-03n: all heads in one op each (dt / dt*w broadcast along head_dim)
        xv = _hv(xs, xF, xo, hpg, P)
        nisa.tensor_tensor(dst=_hv(xdt, gi, 0, hpg, P), data1=xv, data2=_bcv(dts, hpg, 0, hpg, P), op=nl.multiply)
        nisa.tensor_tensor(dst=_hv(xdtw, gi, 0, hpg, P), data1=xv, data2=_bcv(dtw, hpg, 0, hpg, P), op=nl.multiply)
    else:
        for h in range(hpg):
            hs = slice(h * P, (h + 1) * P)
            xh = slice(xo + h * P, xo + (h + 1) * P)
            nisa.tensor_scalar(dst=xdt[0:L, hs], data=xs[0:L, xh], op0=nl.multiply, operand0=dts[0:L, h : h + 1])
            nisa.tensor_scalar(dst=xdtw[0:L, hs], data=xs[0:L, xh], op0=nl.multiply, operand0=dtw[0:L, h : h + 1])

    # Contribution of the entering state: y_off[i, (h, p)] = C_i . h_in[h, p, :].
    yoff_ps = nl.ndarray((L, gi), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=yoff_ps, stationary=cT, moving=stT)
    # This chunk's state contribution: sum_j B_j^T (w_j dt_j x_j).
    st_ps = nl.ndarray((N, gi), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=st_ps, stationary=bf, moving=xdtw)
    SE = BC and "se" in SCAN_OPTS
    if SE:  # R2-03n: state update early (same two ops as at the end of the bc path)
        nisa.tensor_tensor(dst=_hv(stT, gi, 0, hpg, P), data1=_hv(stT, gi, 0, hpg, P),
                           data2=_bcv(cd, hpg, 0, hpg, P), op=nl.multiply)
        nisa.tensor_tensor(dst=stT, data1=stT, data2=st_ps, op=nl.add)

    ych = nl.ndarray((L, gi), dtype=nl.float32, buffer=nl.sbuf)
    if BC:
        yd_all = nl.ndarray((L, gi), dtype=nl.float32, buffer=nl.psum)  # intra-chunk term of all heads, one bank
    if EB:
        HB = 4  # heads per PSUM bank of fp32 [L, 4 * L]
        if "rb" in SCAN_OPTS:  # bd[k, (h, i)] = onehot[k, h] * acum[k, i]
            bd = nl.ndarray((hpg, hpg * L), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=bd.ap(pattern=[[hpg * L, hpg], [L, hpg], [1, L]], offset=0),
                               data1=onehot.ap(pattern=[[hpg * 128, hpg], [128, hpg], [1, L]], offset=0),
                               data2=acT.ap(pattern=[[L, hpg], [0, hpg], [1, L]], offset=0), op=nl.multiply,
                               engine=nisa.gpsimd_engine)
        for hb0 in range(0, hpg, HB):
            nb = min(HB, hpg - hb0)
            # rowb4[j, (q, i)] = acum_i of head hb0+q, plus NEG_BIG where i < j
            rowb4 = nl.ndarray((L, HB * L), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=rowb4[0:L, 0 : nb * L], stationary=identb, moving=k["negm4"][0:L, 0 : nb * L],
                           accumulate=False)
            if "rb" in SCAN_OPTS:
                nisa.nc_matmul(dst=rowb4[0:L, 0 : nb * L], stationary=ones[0:hpg, 0:L],
                               moving=bd[0:hpg, hb0 * L : (hb0 + nb) * L], accumulate=True)
            else:
                for q in range(nb):
                    h = hb0 + q
                    nisa.nc_matmul(dst=rowb4[0:L, q * L : (q + 1) * L], stationary=onehot[0:hpg, h, 0:L], moving=acT,
                                   accumulate=True)
            # e[j, (q, i)] = exp(acum_i - acum_j) for i >= j, 0 for i < j
            e4 = nl.ndarray((L, HB * L), dtype=nl.float32, buffer=nl.sbuf)
            for q in range(nb):
                h = hb0 + q
                nisa.activation(dst=e4[0:L, q * L : (q + 1) * L], op=nl.exp, data=rowb4[0:L, q * L : (q + 1) * L],
                                bias=nacol[0:L, h : h + 1])
            mt4 = nl.ndarray((L, HB * L), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=_hv(mt4, HB * L, 0, nb, L), data1=_hv(e4, HB * L, 0, nb, L),
                               data2=cbm.ap(pattern=[[L, L], [0, nb], [1, L]], offset=0), op=nl.multiply)
            for q in range(nb):
                h = hb0 + q
                hs = slice(h * P, (h + 1) * P)
                if BC:
                    nisa.nc_matmul(dst=yd_all[0:L, hs], stationary=mt4[0:L, q * L : (q + 1) * L], moving=xdt[0:L, hs])
                else:
                    xh = slice(xo + h * P, xo + (h + 1) * P)
                    yd_ps = nl.ndarray((L, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=yd_ps, stationary=mt4[0:L, q * L : (q + 1) * L], moving=xdt[0:L, hs])
                    t1 = nl.ndarray((L, P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=t1, op=nl.copy, data=yoff_ps[0:L, hs], scale=expcol[0:L, h : h + 1])
                    t2 = nl.ndarray((L, P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(
                        dst=t2, data=xs[0:L, xh], op0=nl.multiply, operand0=d_sb[0:L, h : h + 1], op1=nl.add, operand1=t1
                    )
                    nisa.tensor_tensor(dst=ych[0:L, hs], data1=t2, data2=yd_ps, op=nl.add)
    else:
        for h in range(hpg):
            hs = slice(h * P, (h + 1) * P)
            xh = slice(xo + h * P, xo + (h + 1) * P)
            rowb_ps = nl.ndarray((L, L), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=rowb_ps, stationary=onehot[0:hpg, h, 0:L], moving=acT)
            # decay^T[j, i] = exp(min(acum_i - acum_j, 0)) * (i >= j) * (B_j . C_i)
            ein = nl.ndarray((L, L), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=ein, data=rowb_ps, op0=nl.subtract, operand0=acol[0:L, h : h + 1], op1=nl.minimum, operand1=0.0
            )
            e = nl.ndarray((L, L), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=e, op=nl.exp, data=ein)
            mt = nl.ndarray((L, L), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=mt, data1=e, data2=cbm, op=nl.multiply)
            if BC:
                nisa.nc_matmul(dst=yd_all[0:L, hs], stationary=mt, moving=xdt[0:L, hs])
                continue
            yd_ps = nl.ndarray((L, P), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=yd_ps, stationary=mt, moving=xdt[0:L, hs])
            t1 = nl.ndarray((L, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=t1, op=nl.copy, data=yoff_ps[0:L, hs], scale=expcol[0:L, h : h + 1])
            t2 = nl.ndarray((L, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=t2, data=xs[0:L, xh], op0=nl.multiply, operand0=d_sb[0:L, h : h + 1], op1=nl.add, operand1=t1
            )
            nisa.tensor_tensor(dst=ych[0:L, hs], data1=t2, data2=yd_ps, op=nl.add)
    if BC:  # R2-03n: ((exp(acum_i) * y_off) and D x for all heads, then + intra; same order as the per-head ops
        t1 = nl.ndarray((L, gi), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=_hv(t1, gi, 0, hpg, P), data1=_hv(yoff_ps, gi, 0, hpg, P),
                           data2=_bcv(expcol, hpg, 0, hpg, P), op=nl.multiply)
        t2 = nl.ndarray((L, gi), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=_hv(t2, gi, 0, hpg, P), data1=_hv(xs, xF, xo, hpg, P),
                           data2=_bcv(d_sb, hpg, 0, hpg, P), op=nl.multiply)
        nisa.tensor_tensor(dst=t2, data1=t2, data2=t1, op=nl.add)
        nisa.tensor_tensor(dst=ych, data1=t2, data2=yd_all, op=nl.add)
        if not SE:
            # h_out^T = exp(acum_last) h_in^T + this chunk's contribution (cd is the same on every partition).
            nisa.tensor_tensor(dst=_hv(stT, gi, 0, hpg, P), data1=_hv(stT, gi, 0, hpg, P),
                               data2=_bcv(cd, hpg, 0, hpg, P), op=nl.multiply)
            nisa.tensor_tensor(dst=stT, data1=stT, data2=st_ps, op=nl.add)
        return ych

    # h_out^T = exp(acum_last) h_in^T + this chunk's contribution.
    for h in range(hpg):
        hs = slice(h * P, (h + 1) * P)
        nisa.scalar_tensor_tensor(
            dst=stT[0:N, hs], data=stT[0:N, hs], op0=nl.multiply, operand0=cd[0:N, h : h + 1],
            op1=nl.add, operand1=st_ps[0:N, hs],
        )
    return ych


@nki.jit
def ssd_prefill_kernel(
    xbc: Tensor,
    dt: Tensor,
    a_neg: Tensor,
    d_skip: Tensor,
    init_state: Tensor,
    tri: Tensor,
    n_groups: int,
    head_dim: int,
    d_state: int,
):
    """v1: chunked SSD scan of one sequence on the conv output.

    Args:
        xbc: ``[T, conv_dim]`` bf16, ``silu(conv(xBC))`` = ``[x | B | C]``, T a multiple of 128.
        dt: ``[T, H]`` float32, softplus'd dt (0 at padding rows).
        a_neg: ``[H]`` float32, ``A = -exp(A_log)``.
        d_skip: ``[H]`` float32, ``D``.
        init_state: ``[H, head_dim, d_state]`` float32, state before the first token.
        tri: ``[128, 128]`` float32, ``tri[j, i] = 1 if i >= j else 0``.

    Returns:
        ``(y [T, H * head_dim] float32 incl. D x, final_state [H, head_dim, d_state] float32)``.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("ssd_prefill", (0, 1), 2)
    T, conv_dim = xbc.shape
    nh = dt.shape[1]
    P, N, G = head_dim, d_state, n_groups
    I = nh * P
    gi = I // G  # x channels per group
    hpg = nh // G
    nc = T // L
    assert T % L == 0 and N == 128 and conv_dim == I + 2 * G * N
    assert gi <= 512 and hpg <= 128 and P <= 128

    y_out = nl.ndarray((T, I), dtype=nl.float32, buffer=nl.shared_hbm)
    st_out = nl.ndarray((nh, P, N), dtype=nl.float32, buffer=nl.shared_hbm)
    k = _consts(hpg, tri)

    for g in range(G):
        if n_prgs > 1 and g % n_prgs != prg_id:
            continue
        h0 = g * hpg
        b0 = I + g * N
        c0 = I + G * N + g * N
        a_col, d_sb = _group_params(a_neg, d_skip, h0, hpg)
        stT = _load_state(init_state, h0, hpg, P, N, gi)
        for c in range(nc):
            r0 = c * L
            xs = nl.ndarray((L, gi), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=xs, src=xbc[r0 : r0 + L, g * gi : (g + 1) * gi])
            bs = nl.ndarray((L, N), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=bs, src=xbc[r0 : r0 + L, b0 : b0 + N])
            cs = nl.ndarray((L, N), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=cs, src=xbc[r0 : r0 + L, c0 : c0 + N])
            dts = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=dts, src=dt[r0 : r0 + L, h0 : h0 + hpg])
            ych = _scan_chunk(k, xs, 0, bs, 0, cs, 0, dts, stT, a_col, d_sb, hpg, P, N, gi)
            nisa.dma_copy(dst=y_out[r0 : r0 + L, g * gi : (g + 1) * gi], src=ych)
        _store_state(stT, st_out, h0, hpg, P, N)

    return y_out, st_out


def _fused_body(
    srcs,
    T,
    conv_prev: Tensor,
    real: Tensor,
    conv_w: Tensor,
    conv_b: Tensor,
    dt_bias: Tensor,
    a_neg: Tensor,
    d_skip: Tensor,
    norm_w: Tensor,
    init_state: Tensor,
    tri: Tensor,
    n_groups: int,
    head_dim: int,
    d_state: int,
    eps: float,
    conv_kmajor: bool = False,
    inproj=None,
):
    """Body of the fused kernels; ``srcs = ((z, col0), (xBC, col0), (dt, col0))`` give where z, xBC and the raw dt
    columns of each token row are in HBM.

    R2-03m: ``inproj = (h [T, H], w [H, Pp], proj [T, Pp])``: compute ``proj = h @ w`` inside the kernel, one 128-token
    chunk ahead of the scan (this core's group columns only: z_g, x_g, B_g, C_g, dt_g), writing it to the ``proj`` output
    that ``srcs`` point at, so the in_proj TensorE work can run under the previous chunk's Vector/Scalar work."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("mamba_prefill_fused", (0, 1), 2)
    (z_t, z_c), (x_t, x_c), (d_t, d_c) = srcs
    nh = dt_bias.shape[0]
    P, N, G = head_dim, d_state, n_groups
    I = nh * P
    K, conv_dim = conv_w.shape if conv_kmajor else conv_w.shape[::-1]  # R2-03: [K, conv_dim] (kmajor) or [conv_dim, K]
    KW = K - 1
    gi = I // G
    hpg = nh // G
    gw = gi + 2 * N
    nc = T // L
    assert T % L == 0 and N == 128 and conv_dim == I + 2 * G * N
    assert gi <= 512 and hpg <= 128 and P <= 128 and KW < L

    y_out = nl.ndarray((T, I), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    st_out = nl.ndarray((nh, P, N), dtype=nl.float32, buffer=nl.shared_hbm)
    k = _consts(hpg, tri)

    for g in range(G):
        if n_prgs > 1 and g % n_prgs != prg_id:
            continue
        h0 = g * hpg
        # This group's conv channel segments: (channel offset in xBC, width, offset in the group tile).
        segs = ((g * gi, gi, 0), (I + g * N, N, gi), (I + G * N + g * N, N, gi + N))
        a_col, d_sb = _group_params(a_neg, d_skip, h0, hpg)
        # Per-channel parameters broadcast to all partitions (partition stride 0).
        w_g = nl.ndarray((L, K, gw), dtype=nl.bfloat16, buffer=nl.sbuf)
        cb_g = nl.ndarray((L, gw), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c0, w, so in segs:
            for tap in range(K):
                if conv_kmajor:  # R2-03: one contiguous row per tap, broadcast to the partitions
                    nisa.dma_copy(dst=w_g[0:L, tap, so : so + w], src=conv_w.ap(pattern=[[0, L], [1, w]], offset=tap * conv_dim + c0))
                else:  # A06: [conv_dim, K], every element its own 2-byte strided read
                    nisa.dma_copy(dst=w_g[0:L, tap, so : so + w], src=conv_w.ap(pattern=[[0, L], [K, w]], offset=c0 * K + tap))
            nisa.dma_copy(dst=cb_g[0:L, so : so + w], src=conv_b.ap(pattern=[[0, L], [1, w]], offset=c0))
        dtb = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=dtb, src=dt_bias.ap(pattern=[[0, L], [1, hpg]], offset=h0))
        nw_g = nl.ndarray((L, gi), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=nw_g, src=norm_w.ap(pattern=[[0, L], [1, gi]], offset=g * gi))
        stT = _load_state(init_state, h0, hpg, P, N, gi)
        ceng = nisa.gpsimd_engine if "convpool" in SCAN_OPTS else nisa.unknown_engine
        if inproj is not None:
            h_in, w_in, proj_o = inproj
            H = h_in.shape[1]
            nk = H // 128
            # this group's proj columns (start, width), concatenated into the local weight slice
            pr = ((z_c + g * gi, gi), (x_c + g * gi, gi), (x_c + I + g * N, N), (x_c + I + G * N + g * N, N),
                  (d_c + h0, hpg))
            lo_of = []
            acc_w = 0
            for c0_, w_ in pr:
                lo_of.append(acc_w)
                acc_w += w_
            pw = acc_w
            w_sb = nl.ndarray((128, nk, pw), dtype=w_in.dtype, buffer=nl.sbuf)
            for kk in range(nk):
                for (c0_, w_), lo_ in zip(pr, lo_of):
                    nisa.dma_copy(dst=w_sb[0:128, kk, lo_ : lo_ + w_], src=w_in[kk * 128 : (kk + 1) * 128, c0_ : c0_ + w_])
            ptiles = [(n0, min(512, pw - n0)) for n0 in range(0, pw, 512)]

            def inproj_chunk(cc):
                q0 = cc * L
                xh = nl.ndarray((L, H), dtype=h_in.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=xh, src=h_in[q0 : q0 + L, 0:H])
                xT = nl.ndarray((128, nk, L), dtype=h_in.dtype, buffer=nl.sbuf)
                for g0 in range(0, nk, 4):
                    g1 = min(nk, g0 + 4)
                    tx = nl.ndarray((128, (g1 - g0) * L), dtype=h_in.dtype, buffer=nl.psum)
                    for kk in range(g0, g1):
                        nisa.nc_transpose(dst=tx[0:128, (kk - g0) * L : (kk - g0 + 1) * L], data=xh[0:L, kk * 128 : (kk + 1) * 128])
                    if "xtb" in SCAN_OPTS:  # R2-03n: one copy per PSUM tile of 4 transposes
                        nisa.tensor_copy(dst=xT.ap(pattern=[[nk * L, 128], [1, (g1 - g0) * L]], offset=g0 * L),
                                         src=tx[0:128, 0 : (g1 - g0) * L])
                    else:
                        for kk in range(g0, g1):
                            nisa.tensor_copy(dst=xT[0:128, kk, 0:L], src=tx[0:128, (kk - g0) * L : (kk - g0 + 1) * L])
                accs = [nl.ndarray((L, nw), dtype=nl.float32, buffer=nl.psum) for _, nw in ptiles]
                for kk in range(nk):
                    for j, (n0, nw) in enumerate(ptiles):
                        nisa.nc_matmul(dst=accs[j], stationary=xT[0:128, kk, 0:L], moving=w_sb[0:128, kk, n0 : n0 + nw],
                                       accumulate=kk > 0)
                ob = nl.ndarray((L, pw), dtype=h_in.dtype, buffer=nl.sbuf)
                for j, (n0, nw) in enumerate(ptiles):
                    nisa.tensor_copy(dst=ob[0:L, n0 : n0 + nw], src=accs[j])
                for (c0_, w_), lo_ in zip(pr, lo_of):
                    nisa.dma_copy(dst=proj_o[q0 : q0 + L, c0_ : c0_ + w_], src=ob[0:L, lo_ : lo_ + w_])

            inproj_chunk(0)
        sp1 = "sp1" in SCAN_OPTS and inproj is None  # R2-03n: sp1 needs the proj input up front (ignored when merged)
        if sp1:  # R2-03m: softplus for all chunks at once
            dtt_all = nl.ndarray((L, nc, hpg), dtype=nl.float32, buffer=nl.sbuf)
            for c in range(nc):
                dtr = nl.ndarray((L, hpg), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=dtr, src=d_t[c * L : (c + 1) * L, d_c + h0 : d_c + h0 + hpg])
                nisa.tensor_tensor(dst=dtt_all[0:L, c, 0:hpg], data1=dtr, data2=dtb, op=nl.add)
            dsp_all = nl.ndarray((L, nc, hpg), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=dsp_all, op=nl.softplus, data=dtt_all)

        def front(c):
            """Conv + silu (+ silu(z) with zearly) and dt of chunk c."""
            r0 = c * L
            zs = None
            # ── Conv: tap windows straight from HBM, rows r0-KW+tap .. r0+L-1-KW+tap ──
            win = nl.ndarray((L, K, gw), dtype=nl.bfloat16, buffer=nl.sbuf)
            for tap in range(K):
                lo = r0 - KW + tap
                for c0, w, so in segs:
                    col = x_c + c0  # column of channel c0 in the xBC source
                    if lo >= 0:
                        nisa.dma_copy(dst=win[0:L, tap, so : so + w], src=x_t[lo : lo + L, col : col + w])
                    else:
                        npv = -lo  # leading rows from the cached window: conv_prev[tap : KW]
                        nisa.dma_copy(dst=win[0:npv, tap, so : so + w], src=conv_prev[tap:KW, c0 : c0 + w])
                        nisa.dma_copy(dst=win[npv:L, tap, so : so + w], src=x_t[0 : L - npv, col : col + w])
            if "cv" in SCAN_OPTS:  # R2-03n: the four tap products in one op, then the same bf16 adds in order
                prod4 = nl.ndarray((L, K, gw), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=prod4, data1=win, data2=w_g, op=nl.multiply, engine=ceng)
                acc = prod4[0:L, 0, 0:gw]
                for tap in range(1, K):
                    acc2 = nl.ndarray((L, gw), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=acc2, data1=acc, data2=prod4[0:L, tap, 0:gw], op=nl.add, engine=ceng)
                    acc = acc2
            else:
                acc = nl.ndarray((L, gw), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=acc, data1=win[0:L, 0, 0:gw], data2=w_g[0:L, 0, 0:gw], op=nl.multiply, engine=ceng)
                for tap in range(1, K):
                    prod = nl.ndarray((L, gw), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=prod, data1=win[0:L, tap, 0:gw], data2=w_g[0:L, tap, 0:gw], op=nl.multiply, engine=ceng)
                    acc2 = nl.ndarray((L, gw), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=acc2, data1=acc, data2=prod, op=nl.add, engine=ceng)
                    acc = acc2
            accb = nl.ndarray((L, gw), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=accb, data1=acc, data2=cb_g, op=nl.add, engine=ceng)
            xc = nl.ndarray((L, gw), dtype=nl.bfloat16, buffer=nl.sbuf)  # [x | B | C] of this group
            nisa.activation(dst=xc, op=nl.silu, data=accb)
            if "zearly" in SCAN_OPTS:  # R2-03m: gate silu(z) next to the conv silu (one table load)
                zt = nl.ndarray((L, gi), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=zt, src=z_t[r0 : r0 + L, z_c + g * gi : z_c + (g + 1) * gi])
                zs = nl.ndarray((L, gi), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=zs, op=nl.silu, data=zt)

            # ── dt = softplus(dt + dt_bias) * real ──
            rl = nl.ndarray((L, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rl, src=real[r0 : r0 + L, 0:1])
            if sp1:
                dsp = dsp_all[0:L, c, 0:hpg]
            else:
                dtr = nl.ndarray((L, hpg), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=dtr, src=d_t[r0 : r0 + L, d_c + h0 : d_c + h0 + hpg])
                dtt = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=dtt, data1=dtr, data2=dtb, op=nl.add)
                dsp = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=dsp, op=nl.softplus, data=dtt)
            dts = nl.ndarray((L, hpg), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=dts, data=dsp, op0=nl.multiply, operand0=rl)


            return xc, zs, dts

        CP = "cp" in SCAN_OPTS  # R2-03n: chunk c+1's front emitted before chunk c's scan (software pipeline)
        ahead = 2 if CP else 1
        if inproj is not None:
            for cc in range(1, min(ahead, nc)):
                inproj_chunk(cc)
        nxt = front(0) if CP else None
        for c in range(nc):
            r0 = c * L
            if inproj is not None and c + ahead < nc:
                inproj_chunk(c + ahead)  # R2-03m: one chunk ahead of the scan (two with cp)
            if CP:
                xc, zs, dts = nxt
                if c + 1 < nc:
                    nxt = front(c + 1)
            else:
                xc, zs, dts = front(c)
            ych = _scan_chunk(k, xc, 0, xc, gi, xc, gi + N, dts, stT, a_col, d_sb, hpg, P, N, gi)

            # ── Gated RMSNorm over this group's 512 channels: (y * silu(z)) * rstd * w ──
            if "zearly" not in SCAN_OPTS:
                zt = nl.ndarray((L, gi), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=zt, src=z_t[r0 : r0 + L, z_c + g * gi : z_c + (g + 1) * gi])
                zs = nl.ndarray((L, gi), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=zs, op=nl.silu, data=zt)
            if "dn" in SCAN_OPTS:  # R2-03n: y * silu(z) and its sum of squares now, rsqrt and scale per DN_GROUP chunks
                cg = c % DN_GROUP
                if cg == 0:
                    ng = min(DN_GROUP, nc - c)
                    gt_g = nl.ndarray((L, ng, gi), dtype=nl.float32, buffer=nl.sbuf)
                    ss_g = nl.ndarray((L, ng), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=gt_g[0:L, cg, 0:gi], data1=ych, data2=zs, op=nl.multiply)
                sq = nl.ndarray((L, gi), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=sq, op=nl.square, data=gt_g[0:L, cg, 0:gi], reduce_op=nl.add,
                                       reduce_res=ss_g[0:L, cg : cg + 1])
                if cg == ng - 1:
                    var_g = nl.ndarray((L, ng), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=var_g, data=ss_g, op0=nl.multiply, operand0=1.0 / gi, op1=nl.add, operand1=eps)
                    rstd_g = nl.ndarray((L, ng), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=rstd_g, op=nl.rsqrt, data=var_g)
                    for q in range(ng):
                        q0 = (c - cg + q) * L
                        out_bf = nl.ndarray((L, gi), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.scalar_tensor_tensor(dst=out_bf, data=gt_g[0:L, q, 0:gi], op0=nl.multiply,
                                                  operand0=rstd_g[0:L, q : q + 1], op1=nl.multiply, operand1=nw_g)
                        nisa.dma_copy(dst=y_out[q0 : q0 + L, g * gi : (g + 1) * gi], src=out_bf)
                continue
            gt = nl.ndarray((L, gi), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gt, data1=ych, data2=zs, op=nl.multiply)
            sq = nl.ndarray((L, gi), dtype=nl.float32, buffer=nl.sbuf)
            ss = nl.ndarray((L, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=sq, op=nl.square, data=gt, reduce_op=nl.add, reduce_res=ss)
            var = nl.ndarray((L, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=var, data=ss, op0=nl.multiply, operand0=1.0 / gi, op1=nl.add, operand1=eps)
            rstd = nl.ndarray((L, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=var)
            out_bf = nl.ndarray((L, gi), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=out_bf, data=gt, op0=nl.multiply, operand0=rstd, op1=nl.multiply, operand1=nw_g)
            nisa.dma_copy(dst=y_out[r0 : r0 + L, g * gi : (g + 1) * gi], src=out_bf)
        _store_state(stT, st_out, h0, hpg, P, N)

    return y_out, st_out



@nki.jit
def mamba_prefill_fused_kernel(
    proj: Tensor,
    conv_prev: Tensor,
    real: Tensor,
    conv_w: Tensor,
    conv_b: Tensor,
    dt_bias: Tensor,
    a_neg: Tensor,
    d_skip: Tensor,
    norm_w: Tensor,
    init_state: Tensor,
    tri: Tensor,
    n_groups: int,
    head_dim: int,
    d_state: int,
    eps: float,
    conv_kmajor: bool = False,
):
    """v2: one Mamba-2 prefill chunk of one sequence, ``in_proj`` output -> ``out_proj`` input.

    Args:
        proj: ``[T, I + conv_dim + H]`` bf16, ``in_proj`` output ``[z | xBC | dt]``, T a multiple of 128.
        conv_prev: ``[K-1, conv_dim]`` bf16, conv window before the first token (zeros on a cold start).
        real: ``[T, 1]`` float32, 1 for real tokens, 0 for bucket padding (padding only at the end).
        conv_w: ``[conv_dim, K]`` bf16. conv_b: ``[conv_dim]`` bf16.
        dt_bias, a_neg (``-exp(A_log)``), d_skip: ``[H]`` float32. norm_w: ``[I]`` bf16.
        init_state: ``[H, head_dim, d_state]`` float32. tri: as ``ssd_prefill_kernel``.

    Returns:
        ``(y [T, I] bf16 = RMSNormGated(y + D x, z), final_state [H, head_dim, d_state] float32)``.

    Padding rows are not zeroed before the conv here (the torch path zeroes them): the conv is
    causal and padding only follows the real tokens, so this changes padding rows only,
    and their ``dt = 0`` keeps them out of the state.
    """
    I = dt_bias.shape[0] * head_dim
    conv_dim = conv_w.shape[1] if conv_kmajor else conv_w.shape[0]  # R2-03m: [K, conv_dim] with conv_kmajor
    return _fused_body(((proj, 0), (proj, I), (proj, I + conv_dim)), proj.shape[0], conv_prev, real, conv_w, conv_b, dt_bias, a_neg, d_skip, norm_w, init_state, tri, n_groups, head_dim, d_state, eps, conv_kmajor)


@nki.jit
def mamba_prefill_inproj_fused_kernel(
    h: Tensor,
    w_in: Tensor,
    conv_prev: Tensor,
    real: Tensor,
    conv_w: Tensor,
    conv_b: Tensor,
    dt_bias: Tensor,
    a_neg: Tensor,
    d_skip: Tensor,
    norm_w: Tensor,
    init_state: Tensor,
    tri: Tensor,
    n_groups: int,
    head_dim: int,
    d_state: int,
    eps: float,
):
    """R2-03m: ``in_proj`` + :func:`mamba_prefill_fused_kernel` in one kernel. ``h [T, H]`` bf16 (hidden states),
    ``w_in [H, I + conv_dim + nh]`` bf16, conv weight ``[K, conv_dim]``. Returns ``(y [T, I] bf16, final_state,
    proj [T, I + conv_dim + nh] bf16)``; only the columns of the groups (all of them over the cores) are written."""
    T = h.shape[0]
    Pp = w_in.shape[1]
    I = dt_bias.shape[0] * head_dim
    conv_dim = conv_w.shape[1]
    proj = nl.ndarray((T, Pp), dtype=h.dtype, buffer=nl.shared_hbm)
    y, st = _fused_body(((proj, 0), (proj, I), (proj, I + conv_dim)), T, conv_prev, real, conv_w, conv_b, dt_bias, a_neg,
                        d_skip, norm_w, init_state, tri, n_groups, head_dim, d_state, eps, True, inproj=(h, w_in, proj))
    return y, st, proj


@nki.jit
def mamba_prefill_fused_split_kernel(
    z: Tensor,
    xbc: Tensor,
    dtr: Tensor,
    conv_prev: Tensor,
    real: Tensor,
    conv_w: Tensor,
    conv_b: Tensor,
    dt_bias: Tensor,
    a_neg: Tensor,
    d_skip: Tensor,
    norm_w: Tensor,
    init_state: Tensor,
    tri: Tensor,
    n_groups: int,
    head_dim: int,
    d_state: int,
    eps: float,
    conv_kmajor: bool = False,
):
    """v2 with the three ``in_proj`` outputs of ``NEMOTRON_H_MAMBA_PREFILL=split`` (T12): ``z [T, I]``,
    ``xbc [T, conv_dim]`` (already masked to real tokens) and the raw ``dtr [T, H]``, all bf16.
    Otherwise as :func:`mamba_prefill_fused_kernel`."""
    return _fused_body(((z, 0), (xbc, 0), (dtr, 0)), xbc.shape[0], conv_prev, real, conv_w, conv_b, dt_bias, a_neg, d_skip, norm_w, init_state, tri, n_groups, head_dim, d_state, eps, conv_kmajor)


@nki.jit
def mamba_in_proj_kernel(h: Tensor, w: Tensor):
    """R2-03m: Mamba-2 prefill ``in_proj``: ``proj [T, P] = h [T, H] @ w [H, P]`` (bf16 in/out, fp32 accumulation).

    Why a kernel. In the compiler's repeated module (layers 9-50 of the prefill graph) the torch ``in_proj`` matmul is
    lowered with strided moving operands and transposes, and its TensorE time runs under the hardware activity throttle
    (k/n 4/8): 1.5-1.8 ms per layer against 0.55 ms in module 0 (results/kernels/mamba-prefill-r2/README.md). A kernel's
    instruction stream is the same in every module.

    LNC=2: the two physical cores split the output columns (core 0 ``[0, P0)``, core 1 ``[P0, P)``, ``P0`` a multiple of
    128); each keeps its ``[H, P/2]`` weight slice resident in SBUF and streams all token tiles: x tile -> TensorE
    transposes -> ``x^T`` stationary, weight rows moving, K accumulated in PSUM over the H/128 tiles.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("mamba_in_proj", (0, 1), 2)
    T, H = h.shape
    _, P = w.shape
    assert T % 128 == 0 and H % 128 == 0
    nk = H // 128
    if n_prgs > 1:
        p0 = (P // 2) // 128 * 128
        c_lo, c_hi = (0, p0) if prg_id == 0 else (p0, P)
    else:
        c_lo, c_hi = 0, P
    cw = c_hi - c_lo
    NT = 512
    ntiles = [(n0, min(NT, cw - n0)) for n0 in range(0, cw, NT)]
    out = nl.ndarray((T, P), dtype=h.dtype, buffer=nl.shared_hbm)

    # Resident weight slice: w_sb[p, k, c] = w[k * 128 + p, c_lo + c] (contiguous rows of cw columns).
    w_sb = nl.ndarray((128, nk, cw), dtype=w.dtype, buffer=nl.sbuf)
    for k in range(nk):
        nisa.dma_copy(dst=w_sb[0:128, k, 0:cw], src=w[k * 128 : (k + 1) * 128, c_lo:c_hi])

    for t in range(T // 128):
        r0 = t * 128
        x_sb = nl.ndarray((128, H), dtype=h.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb, src=h[r0 : r0 + 128, 0:H])
        # x^T [128 (h within tile k), nk, 128 (tokens)], four TensorE transposes per PSUM tile
        xT = nl.ndarray((128, nk, 128), dtype=h.dtype, buffer=nl.sbuf)
        for g0 in range(0, nk, 4):
            g1 = min(nk, g0 + 4)
            tx = nl.ndarray((128, (g1 - g0) * 128), dtype=h.dtype, buffer=nl.psum)
            for k in range(g0, g1):
                nisa.nc_transpose(dst=tx[0:128, (k - g0) * 128 : (k - g0 + 1) * 128], data=x_sb[0:128, k * 128 : (k + 1) * 128])
            for k in range(g0, g1):
                nisa.tensor_copy(dst=xT[0:128, k, 0:128], src=tx[0:128, (k - g0) * 128 : (k - g0 + 1) * 128])
        # proj tile [128 tokens, cw]: x^T tile stationary (one load per k for all column tiles), weight rows moving
        acc = [nl.ndarray((128, nw), dtype=nl.float32, buffer=nl.psum) for _, nw in ntiles]
        for k in range(nk):
            for j, (n0, nw) in enumerate(ntiles):
                nisa.nc_matmul(dst=acc[j], stationary=xT[0:128, k, 0:128], moving=w_sb[0:128, k, n0 : n0 + nw],
                               accumulate=k > 0)
        for j, (n0, nw) in enumerate(ntiles):
            o_sb = nl.ndarray((128, nw), dtype=h.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_sb, src=acc[j])
            nisa.dma_copy(dst=out[r0 : r0 + 128, c_lo + n0 : c_lo + n0 + nw], src=o_sb)
    return out


def mamba_in_proj(h: Tensor, w: Tensor) -> Tensor:
    """``h @ w`` with :func:`mamba_in_proj_kernel` (LNC=2 grid)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(mamba_in_proj_kernel)[2](h=h.contiguous(), w=w)


def tri_mask(device) -> Tensor:
    """``tri[j, i] = 1 if i >= j`` (float32 [128, 128])."""
    return torch.triu(torch.ones(L, L, dtype=torch.float32, device=device))


def kernel_args(mixer, xbc: Tensor, dt: Tensor, init_state: Tensor) -> dict:
    """Keyword arguments of :func:`ssd_prefill_kernel` for a ``NemotronHMamba2Mixer``."""
    return dict(
        xbc=xbc.contiguous(),
        dt=dt.contiguous(),
        a_neg=-torch.exp(mixer.A_log.float()),
        d_skip=mixer.D.float(),
        init_state=init_state.contiguous(),
        tri=tri_mask(xbc.device),
        n_groups=mixer.n_groups,
        head_dim=mixer.head_dim,
        d_state=mixer.d_state,
    )


def ssd_prefill(mixer, xbc: Tensor, dt: Tensor, init_state: Tensor) -> tuple[Tensor, Tensor]:
    """v1 on device (LNC=2 grid, one SSM group per core).

    Returns ``(y [T, H, head_dim] float32 incl. D x, final_state [H, head_dim, d_state])``.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    y, st = wrap_nki(ssd_prefill_kernel)[2](**kernel_args(mixer, xbc, dt, init_state))
    return y.view(xbc.shape[0], mixer.num_heads, mixer.head_dim), st


def fused_kernel_args(mixer, proj: Tensor, conv_prev: Tensor, real: Tensor, init_state: Tensor) -> dict:
    """Keyword arguments of :func:`mamba_prefill_fused_kernel` for a ``NemotronHMamba2Mixer``."""
    return dict(
        proj=proj.contiguous(),
        conv_prev=conv_prev.contiguous(),
        real=real.contiguous(),
        conv_w=mixer.conv1d_weight,
        conv_b=mixer.conv1d_bias,
        dt_bias=mixer.dt_bias,
        a_neg=-torch.exp(mixer.A_log.float()),
        d_skip=mixer.D.float(),
        norm_w=mixer.norm_weight,
        init_state=init_state.contiguous(),
        tri=tri_mask(proj.device),
        n_groups=mixer.n_groups,
        head_dim=mixer.head_dim,
        d_state=mixer.d_state,
        eps=float(mixer.eps),
    )


def mamba_prefill_fused_split(mixer, z: Tensor, xbc: Tensor, dtr: Tensor, conv_prev: Tensor, real: Tensor,
                              init_state: Tensor) -> tuple[Tensor, Tensor]:
    """v2 on the split in_proj outputs (LNC=2 grid). Returns ``(y [T, I] bf16, final_state)``."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    kw = fused_kernel_args(mixer, z, conv_prev, real, init_state)
    del kw["proj"]
    if CONV_KMAJOR:  # R2-03: conv weight as [K, conv_dim] (12 KB per layer, transposed in the graph)
        kw["conv_w"] = mixer.conv1d_weight.t().contiguous()
        kw["conv_kmajor"] = True
    return wrap_nki(mamba_prefill_fused_split_kernel)[2](z=z.contiguous(), xbc=xbc.contiguous(), dtr=dtr.contiguous(), **kw)


def mamba_prefill_inproj_fused(mixer, h: Tensor, conv_prev: Tensor, real: Tensor, init_state: Tensor):
    """R2-03m: :func:`mamba_prefill_inproj_fused_kernel` (LNC=2 grid). Returns ``(y [T, I] bf16, final_state, proj)``."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    kw = fused_kernel_args(mixer, h, conv_prev, real, init_state)
    del kw["proj"]
    kw["conv_w"] = mixer.conv1d_weight.t().contiguous()
    return wrap_nki(mamba_prefill_inproj_fused_kernel)[2](h=h.contiguous(), w_in=mixer.in_proj_weight, **kw)


def mamba_prefill_fused(mixer, proj: Tensor, conv_prev: Tensor, real: Tensor, init_state: Tensor) -> tuple[Tensor, Tensor]:
    """v2 on device (LNC=2 grid). Returns ``(y [T, I] bf16 out_proj input, final_state)``.

    R2-03m: with ``CONV_KMAJOR`` the conv weight goes in as ``[K, conv_dim]`` (as in :func:`mamba_prefill_fused_split`)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    kw = fused_kernel_args(mixer, proj, conv_prev, real, init_state)
    if CONV_KMAJOR:
        kw["conv_w"] = mixer.conv1d_weight.t().contiguous()
        kw["conv_kmajor"] = True
    return wrap_nki(mamba_prefill_fused_kernel)[2](**kw)
