"""Fused top-k MoE decode for non-gated squared-ReLU experts (T14, optimization step "opt4").

For each (token, selected expert) pair the kernel streams the expert's up and down matrices from
the stacked weights straight into SBUF (dynamic expert offset, ``[116, 2688]`` row tiles, software DGE
by default so the tiles spread over all 16 DMA engines, see ``DMA_MODE``), and computes

    u   = up_e @ x_t                      # Vector Engine dot products (fused multiply + add-reduce)
    a   = relu(u)^2 * w_(t,e)             # routing weight folded in before the down projection
    y_t += a @ down_e                     # Tensor Engine, accumulated in PSUM over all pairs

so no gathered copy is written back to HBM (the T10 gather wrote and re-read every selected
expert). The pairs are split over the two physical cores of the logical core; each core returns
a float32 partial ``[T, H]`` and the caller sums them.

Layouts (both ``[E * I, H]``, rows of H contiguous): the up weight stored ``[out, in]``
(``NEMOTRON_H_MOE_WT=up``) and the down weight ``[in, out]``, the same as ``expert_gather.py``.
The up projection contracts over H, which is the free axis of the natural row tiles, hence the
Vector Engine dot products instead of a transposed weight.
"""

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

_PSUM_COLS = 512

# T10b: descriptor generation for the expert row tiles (trace-time constant, read at import).
#   hw : hardware DGE triggered from the Sync and Scalar queues (T14). A 4-layer device profile
#        shows these tiles on 4 of the 16 DMA engines of each physical core (~24 GB/s each).
#   sw : software DGE (GpSimd writes the descriptors; same scalar expert offset, same tiles). The
#        tiles spread over all 16 engines. Unit, 8 MoE layers at batch 1: 0.307 -> 0.199 ms/layer;
#        full model, batch-1 decode step 9.38 -> 7.37 ms, outputs bit-identical, gate passes (default since T10b).
# Compare variants with NKI_ENABLE_TRACE_CACHE=0: the persisted NKI trace cache does not key on this global.
import os as _os

DMA_MODE = _os.environ.get("NEMOTRON_H_MOE_TOPK_DMA", "sw")
#   T14c: mix  : up tiles on hardware DGE (Sync queue), down tiles on software DGE, so two DGE queues
#                stream at the same time (nkilib moe_tkg does this: up on qSyncDynamicHW, down on qGpSimdDynamic).
#         mix2 : as mix, up tiles alternate Sync / Scalar.
if DMA_MODE not in ("hw", "sw", "mix", "mix2"):
    raise ValueError(f"NEMOTRON_H_MOE_TOPK_DMA must be hw, sw, mix or mix2, got {DMA_MODE!r}")
# T14c: row tile height. "exact" = largest divisor of I <= 128 (116 for I=464, T14); "128" = 128-row tiles
# with a shorter last tile (464 = 3 * 128 + 80). A unit profile shows hardware DGE spreading a
# [116, H] tile over only 4 of the 16 DMA engines (116 = 4 * 29), and nkilib's [128, ...] tiles over 16;
# software DGE also loads faster with 128 partitions (~235 -> ~280 GB/s per physical core). Unit, 4 MoE layers:
# 131.3 -> 118.3 us per layer of device time; full model, batch-1 decode step 7.34 -> 7.04 ms, c=1 TPOT
# -2.2..-3.9% on the Must rows, gate passes (default since T14c).
TILE = _os.environ.get("NEMOTRON_H_MOE_TOPK_TILE", "128")
# R2-10: EARLY=1 emits every pair's expert-tile loads before the broadcast loads of the routing weights and x (a
# batch-1 device profile shows the first expert load ~8 us after the kernel's index load, behind those loads).
# BQ=act puts the two broadcast loads on the Scalar (Activation) hardware-DGE queue instead of the default queue.
# Both default on since R2-10 for the BF16 kernel (the FP8 kernel keeps the old order); EARLY=0 / BQ=default restore it.
EARLY = _os.environ.get("NEMOTRON_H_MOE_TOPK_EARLY", "1") == "1"
_BQ = _os.environ.get("NEMOTRON_H_MOE_TOPK_BQ", "act")
EARLY_KW = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar) if _BQ == "act" else {}
if TILE not in ("exact", "128"):
    raise ValueError(f"NEMOTRON_H_MOE_TOPK_TILE must be exact or 128, got {TILE!r}")


@nki.jit
def _moe_topk_kernel(x: Tensor, up: Tensor, down: Tensor, experts: Tensor, w: Tensor, geom: Tensor):
    """BF16 weights (see ``_moe_topk_body``)."""
    return _moe_topk_body(x, up, down, experts, w, geom, None)


@nki.jit
def _moe_topk_kernel_fp8(x: Tensor, up: Tensor, down: Tensor, experts: Tensor, w: Tensor, geom: Tensor, c: Tensor):
    """FP8 weights (``NEMOTRON_H_FP8=1``, R2-05): ``up``/``down`` float8_e4m3, ``c`` ``[E * I + 128, 1]`` float32
    per-intermediate-row scale ``s_up^2 * s_down`` (see ``moe.py``)."""
    return _moe_topk_body(x, up, down, experts, w, geom, cscale=c)


def _moe_topk_body(x, up, down, experts, w, geom, cscale):
    """Routed-expert output of ``T`` tokens, float32 partial sums per physical core.

    Args:
        x: ``[T, H]`` bf16 hidden states.
        up, down: ``[E * I, H]`` bf16 stacked expert weights (up ``[out, in]``, down ``[in, out]``).
        experts: ``[n, 1]`` int32, row offset of each pair's expert (= expert id * I), pair-major
            (pair ``p`` belongs to token ``p // (n // T)``).
        w: ``[1, n]`` float32 routing weight of each pair.
        geom: dummy ``[I, PT]`` tensor; its shape gives the expert rows I and the row tile PT
            (the last tile has ``I - (nj - 1) * PT`` rows when PT does not divide I).
    Returns:
        ``[n_prgs, T, H]`` float32; the routed output is the sum over the first axis.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("moe_topk", (0, 1), 2)
    t_n, h = x.shape
    n = experts.shape[0]
    k = n // t_n
    rows, pt = geom.shape
    nj = (rows + pt - 1) // pt
    tsz = [min(pt, rows - j * pt) for j in range(nj)]  # rows of tile j (last one shorter if pt does not divide I)
    out = nl.ndarray((n_prgs, t_n, h), dtype=nl.float32, buffer=nl.shared_hbm)

    per = (n + n_prgs - 1) // n_prgs
    p_lo = prg_id * per
    p_hi = min(n, p_lo + per)

    idx_sb = nl.ndarray((n, 1), dtype=experts.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_sb, src=experts)

    def _issue(p, q):
        """The expert up / down row-tile loads of pair ``p`` (they need only ``idx_sb``)."""
        up_sb = nl.ndarray((pt, nj, h), dtype=up.dtype, buffer=nl.sbuf)
        dn_sb = nl.ndarray((pt, nj, h), dtype=down.dtype, buffer=nl.sbuf)
        for j in range(nj):
            sz = tsz[j]
            for si, (src, dst) in enumerate(((up, up_sb), (down, dn_sb))):
                if DMA_MODE == "sw" or (DMA_MODE in ("mix", "mix2") and si == 1):
                    kw = dict(dge_mode=nisa.dge_mode.swdge)
                elif DMA_MODE == "mix":
                    kw = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
                else:
                    kw = dict(dge_mode=nisa.dge_mode.hwdge,
                              engine=nisa.engine.sync if q % 2 == 0 else nisa.engine.scalar)
                    q += 1
                nisa.dma_copy(
                    dst=dst[0:sz, j, 0:h],
                    src=src.ap(pattern=[[h, sz], [1, h]], offset=j * pt * h,
                               scalar_offset=idx_sb[p : p + 1, 0:1], indirect_dim=0),
                    **kw,
                )
        return up_sb, dn_sb, q

    q = 0
    loads = {}
    early = EARLY and cscale is None  # R2-10 defaults apply to the BF16 kernel only (FP8 variant unchanged)
    bq_kw = EARLY_KW if cscale is None else {}
    if early:
        # R2-10: every pair's expert loads are emitted before the broadcast loads of w and x, so the expert loads
        # (the critical path) do not queue behind them.
        for p in range(p_lo, p_hi):
            up_sb, dn_sb, q = _issue(p, q)
            loads[p] = (up_sb, dn_sb)
    # routing weights broadcast to every partition: [pt, n]
    w_b = nl.ndarray((pt, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_b, src=w.ap(pattern=[[0, pt], [1, n]], offset=0), **bq_kw)
    # each token broadcast to every partition: [pt, T, H]
    x_b = nl.ndarray((pt, t_n, h), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_b, src=x.ap(pattern=[[0, pt], [h, t_n], [1, h]], offset=0), **bq_kw)

    n_chunks = (h + _PSUM_COLS - 1) // _PSUM_COLS
    ps = []
    for c in range(n_chunks):
        csz = min(_PSUM_COLS, h - c * _PSUM_COLS)
        ps.append(nl.ndarray((t_n, csz), dtype=nl.float32, buffer=nl.psum))

    for p in range(p_lo, p_hi):
        tok = p // k
        if p in loads:
            up_sb, dn_sb = loads.pop(p)
        else:
            up_sb, dn_sb, q = _issue(p, q)
        # u[i] = sum_h up[i, h] * x[h], one fused multiply + add-reduce per row tile
        u = nl.ndarray((pt, nj), dtype=nl.float32, buffer=nl.sbuf)
        if tsz[-1] < pt:  # rows past the short last tile stay 0 (they never reach the down matmul)
            nisa.memset(dst=u, value=0.0)
        junk = nl.ndarray((pt, h), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(nj):
            sz = tsz[j]
            nisa.scalar_tensor_tensor(
                dst=junk[0:sz, 0:h], data=up_sb[0:sz, j, 0:h], op0=nl.multiply, operand0=1.0,
                op1=nl.multiply, operand1=x_b[0:sz, tok, 0:h],
                reduce_op=nl.add, reduce_res=u[0:sz, j : j + 1],
            )
        # a = relu(u)^2 * w
        r = nl.ndarray((pt, nj), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r, data=u, op0=nl.maximum, operand0=0.0)
        a_f = nl.ndarray((pt, nj), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=a_f, data=r, op0=nl.multiply, operand0=w_b[0:pt, p : p + 1],
                                  op1=nl.multiply, operand1=r)
        if cscale is not None:  # FP8: a *= c[e * I + j * pt + row] (rows past a short last tile are 0 in r)
            c_sb = nl.ndarray((pt, nj), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=c_sb, src=cscale.ap(pattern=[[1, pt], [pt, nj]], offset=0,
                                             scalar_offset=idx_sb[p : p + 1, 0:1], indirect_dim=0),
                          dge_mode=nisa.dge_mode.swdge)
            nisa.tensor_tensor(dst=a_f, data1=a_f, data2=c_sb, op=nl.multiply)
        # stationary [pt, nj * T] bf16, column j * T + tok holds a[:, j], other columns 0
        a_st = nl.ndarray((pt, nj * t_n), dtype=nl.bfloat16, buffer=nl.sbuf)
        if t_n > 1:
            nisa.memset(dst=a_st, value=0.0)
        for j in range(nj):
            nisa.tensor_copy(dst=a_st[0:pt, j * t_n + tok : j * t_n + tok + 1], src=a_f[0:pt, j : j + 1])
        for c in range(n_chunks):
            c0 = c * _PSUM_COLS
            csz = min(_PSUM_COLS, h - c0)
            for j in range(nj):
                sz = tsz[j]
                nisa.nc_matmul(
                    dst=ps[c][0:t_n, 0:csz],
                    stationary=a_st[0:sz, j * t_n : (j + 1) * t_n],
                    moving=dn_sb[0:sz, j, c0 : c0 + csz],
                    accumulate=not (p == p_lo and j == 0),
                )

    o_sb = nl.ndarray((t_n, h), dtype=nl.float32, buffer=nl.sbuf)
    if p_hi > p_lo:
        for c in range(n_chunks):
            c0 = c * _PSUM_COLS
            csz = min(_PSUM_COLS, h - c0)
            nisa.tensor_copy(dst=o_sb[0:t_n, c0 : c0 + csz], src=ps[c][0:t_n, 0:csz])
    else:
        nisa.memset(dst=o_sb, value=0.0)
    nisa.dma_copy(dst=out[prg_id, 0:t_n, 0:h], src=o_sb)
    return out


def _row_tile(rows: int) -> int:
    for pt in range(min(rows, 128), 0, -1):
        if rows % pt == 0:
            return pt
    return 1


def moe_topk_decode_off(x: Tensor, up: Tensor, down: Tensor, off: Tensor, w: Tensor, rows: int) -> Tensor:
    """:func:`moe_topk_decode` with the row offsets ``off`` ``[n, 1]`` int32 (expert id * rows) and the weights
    ``w`` ``[1, n]`` float32 already in the kernel's layout (R2-10 ``router_fused``)."""
    pt = _row_tile(rows) if TILE == "exact" else min(rows, 128)
    out = wrap_nki(_moe_topk_kernel)[2](
        x=x, up=up, down=down, experts=off, w=w,
        geom=torch.zeros((rows, pt), dtype=torch.int32, device=x.device))
    return out.sum(0).to(x.dtype)


def moe_topk_decode(x: Tensor, up: Tensor, down: Tensor, experts: Tensor, w: Tensor, rows: int,
                    c: Tensor | None = None) -> Tensor:
    """Routed-expert sum for ``x`` ``[T, H]``.

    ``up``/``down`` ``[E * rows, H]``; ``experts`` ``[T, k]`` expert ids; ``w`` ``[T, k]`` routing
    weights (float32). Returns ``[T, H]`` in ``x.dtype``.
    """
    pt = _row_tile(rows) if TILE == "exact" else min(rows, 128)
    kw = dict(
        x=x,
        up=up,
        down=down,
        experts=(experts.to(torch.int32) * rows).reshape(-1, 1),
        w=w.to(torch.float32).reshape(1, -1),
        geom=torch.zeros((rows, pt), dtype=torch.int32, device=x.device),
    )
    if c is not None:
        out = wrap_nki(_moe_topk_kernel_fp8)[2](c=c, **kw)
    else:
        out = wrap_nki(_moe_topk_kernel)[2](**kw)
    return out.sum(0).to(x.dtype)
