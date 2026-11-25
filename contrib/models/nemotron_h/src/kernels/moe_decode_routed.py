"""Routed MoE for batched decode (R2-01): each selected expert's weights are loaded once per step and applied to
all tokens of the batch.

The dense decode path (``F.linear`` on the ``[E * I, H]`` up weight) reads all 128 experts per MoE layer per step.
A decode batch of ``T`` tokens selects ``T * k`` (token, expert) pairs, but only ``n_u <= min(T * k, E)`` distinct
experts. The torch side builds a compact list of the distinct experts (static shapes: an exclusive cumsum over the
``[E]`` selection mask and one 128-element scatter) and the kernel walks that list with a dynamic loop:

    for each distinct expert e (this core's half of the list, two per loop iteration, double-buffered loads):
        load up_e, down_e  ([I, H] row tiles at a runtime row offset, software DGE)
        upT  = transpose(up_e)                   # Tensor Engine, [H, I]
        u    = x @ upT                           # [T, I], all T tokens (the batch is one tile)
        a    = relu(u)^2 * w[:, e]               # w[t, e] = routing weight, 0 if t did not pick e
        y   += a @ down_e                        # [T, H] float32

Tokens that did not pick ``e`` get weight 0, so ``y`` is the exact routed sum (same as the dense masked path).
Expert ``e`` runs on physical core ``e % 2`` (``NEMOTRON_H_MOE_ROUTED_SPLIT=parity``, same dynamic trip count on both
cores, NCC_IXGM002); each core returns a float32 partial ``[T, H]`` and the caller adds them.

Layouts as ``moe_topk.py``: up ``[E * I, H]`` (``[out, in]``, ``NEMOTRON_H_MOE_WT=up``), down ``[E * I, H]``
(``[in, out]``).
"""

import os

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

_PSUM_COLS = 512
# list slots per core: up to 64 experts per core, plus the prefetch of one entry past the last iteration
_S2 = 68
# Engines for the PSUM -> SBUF copies of the transposed up weight (21 per expert): "va" alternates Vector / Scalar
# (Activation) engines, "v" uses the Vector engine only.
COPY_ENG = os.environ.get("NEMOTRON_H_MOE_ROUTED_COPY", "va")
# Load pipelining: "v1" issues the next entry's up and down loads together before / after a whole compute; "v2" issues
# the next up load right after this entry's up transposes (up buffer free) and the next down load after its down
# matmuls, so the loads start earlier.
PIPE = os.environ.get("NEMOTRON_H_MOE_ROUTED_PIPE", "v1")
# How the distinct experts are split over the two physical cores:
#   "parity" (default): expert e always runs on core e % 2, in ascending id order. A token's result then does not
#            depend on which experts the other tokens of the step picked (unpicked experts add exact zeros), so a
#            request's output is the same in any batch of the same bucket. Trip count = ceil(max(n_even, n_odd) / 2).
#   "pos": alternate list positions (balanced, trip count ceil(n_u / 4)), but the per-core partial sums, and so the
#          fp32 rounding of a token's output, depend on the other tokens' experts (R2-01 gate: batch-invariance
#          leakage rule failed on 3 prompts).
SPLIT = os.environ.get("NEMOTRON_H_MOE_ROUTED_SPLIT", "parity")
if SPLIT not in ("parity", "pos"):
    raise ValueError(f"NEMOTRON_H_MOE_ROUTED_SPLIT must be parity or pos, got {SPLIT!r}")


@nki.jit
def moe_routed_kernel(x: Tensor, up: Tensor, down: Tensor, wt: Tensor, elist: Tensor, erow: Tensor, niter: Tensor,
                      geom: Tensor):
    """Args:
        x: ``[T, H]`` bf16 (T <= 128).
        up, down: ``[E * I, H]`` bf16.
        wt: ``[(E + 1) * T, 1]`` float32 routing weights, transposed and flattened (row ``E`` = 0, for pads).
        elist: ``[2 * S2, 1]`` int32 ``e * T`` (offset of the entry's weight row in ``wt``) of each list entry
            (core ``c`` owns ``[c * S2, (c + 1) * S2)``; pad entries hold ``E * T``). erow: same layout, first weight row of the entry's expert (pads: 0).
        niter: ``[1, 1]`` int32 loop iterations (2 entries per core each).
        geom: dummy ``[I, PT]``; its shape gives I and the weight row tile PT.
    Returns:
        ``[n_prgs, T, H]`` float32 partial sums.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("moe_routed", (0, 1), 2)
    t_n, h = x.shape
    rows, pt = geom.shape
    nj = (rows + pt - 1) // pt
    pts = [min(pt, rows - j * pt) for j in range(nj)]
    nh = h // 128
    assert nh * 128 == h and t_n <= 128 and rows <= _PSUM_COLS
    s2 = elist.shape[0] // n_prgs
    base = prg_id * s2
    n_chunks = (h + _PSUM_COLS - 1) // _PSUM_COLS
    out = nl.ndarray((n_prgs, t_n, h), dtype=nl.float32, buffer=nl.shared_hbm)

    # x^T [128 (h), nh, T] once
    x_sb = nl.ndarray((t_n, h), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    xT = nl.ndarray((128, nh, t_n), dtype=x.dtype, buffer=nl.sbuf)
    for hh in range(nh):
        tx = nl.ndarray((128, t_n), dtype=x.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=tx, data=x_sb[0:t_n, hh * 128 : (hh + 1) * 128])
        nisa.tensor_copy(dst=xT[0:128, hh, 0:t_n], src=tx)

    y_acc = nl.ndarray((t_n, h), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=y_acc, value=0.0)

    # list cursor (entry index within this core's range), loop-carried in SBUF
    cur = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cur, src=niter.ap(pattern=[[1, 1], [1, 1]], offset=1))  # niter[1] holds 0

    def _fetch(e_sb, r_sb):
        """Load entry ``cur`` of this core's list: expert id -> e_sb, weight row -> r_sb."""
        nisa.dma_copy(dst=e_sb, src=elist.ap(pattern=[[1, 1], [1, 1]], offset=base, scalar_offset=cur,
                                             indirect_dim=0),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        nisa.dma_copy(dst=r_sb, src=erow.ap(pattern=[[1, 1], [1, 1]], offset=base, scalar_offset=cur,
                                            indirect_dim=0),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)

    def _load(r_sb, up_sb, dn_sb, which="ud"):
        for j in range(nj):
            for src, dst in (((up, up_sb),) if which == "u" else ((down, dn_sb),) if which == "d"
                             else ((up, up_sb), (down, dn_sb))):
                nisa.dma_copy(dst=dst[0:pts[j], j, 0:h],
                              src=src.ap(pattern=[[h, pts[j]], [1, h]], offset=j * pt * h, scalar_offset=r_sb,
                                         indirect_dim=0),
                              dge_mode=nisa.dge_mode.swdge)

    def _compute(e_sb, up_sb, dn_sb, after_t=None, after_d=None):
        # routing weight of every token for this expert: [T, 1]
        w_sb = nl.ndarray((t_n, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=wt.ap(pattern=[[1, t_n], [1, 1]], offset=0, scalar_offset=e_sb, indirect_dim=0),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
        upT = nl.ndarray((128, nh, rows), dtype=up.dtype, buffer=nl.sbuf)
        for hh in range(nh):
            tp = nl.ndarray((128, rows), dtype=up.dtype, buffer=nl.psum)
            for j in range(nj):
                nisa.nc_transpose(dst=tp[0:128, j * pt : j * pt + pts[j]],
                                  data=up_sb[0:pts[j], j, hh * 128 : (hh + 1) * 128])
            eng = nisa.engine.scalar if (COPY_ENG == "va" and hh % 2 == 1) else nisa.engine.vector
            nisa.tensor_copy(dst=upT[0:128, hh, 0:rows], src=tp[0:128, 0:rows], engine=eng)
        pu = nl.ndarray((t_n, rows), dtype=nl.float32, buffer=nl.psum)
        for hh in range(nh):
            nisa.nc_matmul(dst=pu, stationary=xT[0:128, hh, 0:t_n], moving=upT[0:128, hh, 0:rows],
                           accumulate=hh > 0)
        r = nl.ndarray((t_n, rows), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r, data=pu, op0=nl.maximum, operand0=0.0)
        a_sb = nl.ndarray((t_n, rows), dtype=x.dtype, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=a_sb, data=r, op0=nl.multiply, operand0=w_sb[0:t_n, 0:1],
                                  op1=nl.multiply, operand1=r)
        aT = nl.ndarray((pt, nj, t_n), dtype=x.dtype, buffer=nl.sbuf)
        for j in range(nj):
            ta = nl.ndarray((pts[j], t_n), dtype=x.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=ta, data=a_sb[0:t_n, j * pt : j * pt + pts[j]])
            nisa.tensor_copy(dst=aT[0:pts[j], j, 0:t_n], src=ta)
        for c in range(n_chunks):
            c0 = c * _PSUM_COLS
            csz = min(_PSUM_COLS, h - c0)
            py = nl.ndarray((t_n, csz), dtype=nl.float32, buffer=nl.psum)
            for j in range(nj):
                nisa.nc_matmul(dst=py, stationary=aT[0:pts[j], j, 0:t_n], moving=dn_sb[0:pts[j], j, c0 : c0 + csz],
                               accumulate=j > 0)
            nisa.tensor_tensor(dst=y_acc[0:t_n, c0 : c0 + csz], data1=y_acc[0:t_n, c0 : c0 + csz], data2=py,
                               op=nl.add)
        if after_d is not None:
            after_d()

    # double buffers (A: even entries, B: odd entries)
    ea = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    ra = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    eb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    rb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    up_a = nl.ndarray((pt, nj, h), dtype=up.dtype, buffer=nl.sbuf)
    dn_a = nl.ndarray((pt, nj, h), dtype=down.dtype, buffer=nl.sbuf)
    up_b = nl.ndarray((pt, nj, h), dtype=up.dtype, buffer=nl.sbuf)
    dn_b = nl.ndarray((pt, nj, h), dtype=down.dtype, buffer=nl.sbuf)

    # prologue: entry 0 -> A
    _fetch(ea, ra)
    _load(ra, up_a, dn_a)
    nisa.tensor_scalar(dst=cur, data=cur, op0=nl.add, operand0=1)

    n_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=n_sb, src=niter.ap(pattern=[[1, 1], [1, 1]], offset=0))
    n_reg = nisa.register_alloc()
    nisa.register_load(n_reg, n_sb)

    def _body(_i):
        if PIPE == "v2":
            def _next(e_sb, r_sb, up_sb, dn_sb):
                _fetch(e_sb, r_sb)
                nisa.tensor_scalar(dst=cur, data=cur, op0=nl.add, operand0=1)
                _load(r_sb, up_sb, dn_sb, "u")
            # entry 2i (A); entry 2i + 1 -> B while A computes (up after A's transposes, down after A's down matmuls)
            _compute(ea, up_a, dn_a, after_t=lambda: _next(eb, rb, up_b, dn_b),
                     after_d=lambda: _load(rb, up_b, dn_b, "d"))
            # entry 2i + 1 (B); entry 2i + 2 -> A
            _compute(eb, up_b, dn_b, after_t=lambda: _next(ea, ra, up_a, dn_a),
                     after_d=lambda: _load(ra, up_a, dn_a, "d"))
            return
        _fetch(eb, rb)                # entry 2i + 1 -> B
        _load(rb, up_b, dn_b)
        nisa.tensor_scalar(dst=cur, data=cur, op0=nl.add, operand0=1)
        _compute(ea, up_a, dn_a)      # entry 2i
        _fetch(ea, ra)                # entry 2i + 2 -> A (the last one is a pad or the next iteration's)
        _load(ra, up_a, dn_a)
        nisa.tensor_scalar(dst=cur, data=cur, op0=nl.add, operand0=1)
        _compute(eb, up_b, dn_b)      # entry 2i + 1

    nl.fori_loop(0, n_reg, _body)
    nisa.dma_copy(dst=out[prg_id, 0:t_n, 0:h], src=y_acc)
    return out


# R2-01b options for the BF16 kernel (``NEMOTRON_H_MOE_ROUTED_V2``, comma list; default V2_DEFAULT below; "none" = the
# R2-01 kernel above):
#   "skip"    : pad entries past list position 2 of each core get an out-of-range weight row and the weight loads use
#               ``oob_mode=skip`` (no transfer; the slot keeps the previous finite expert, the pad's weight is 0).
#   "bulk"    : the static prefix's list entries are read with two DMAs (all positions at once) instead of two per entry.
#   "wsync"   : the per-entry routing-weight load goes on the Sync HWDGE queue instead of the Scalar (Activation) queue,
#               whose engine also does half of the PSUM -> SBUF copies.
#   "wfirst"  : entry 0's weight loads are emitted before the x load / transposes (kernel prologue).
#   "one"     : row tiles of I / 4 rows (116 for I = 464), so each expert's up and down weights are one DMA each
#               (instead of 4 + 4); the down projection then accumulates over 4 x 116 rows instead of 3 x 128 + 80, so
#               the fp32 rounding differs from the R2-01 kernel (not bit-identical).
#   "dsync"   : the down-weight loads go on the Sync hardware DGE queue (up stays on the GpSimd software DGE queue).
#   "pN"      : N entries per dynamic-loop iteration (even, default 2), still two load buffers. Every loop iteration
#               ends with a wait for all outstanding DMAs (~9 us of DMA idle at T=32, R2-01b profile), so more
#               entries per iteration amortize it; the price is up to N - 1 pad computes at the end.
#   "preT:K"  : for a T-token graph, the first K entries (even) of each core run as straight-line code before the
#               loop (static list positions, no loop barrier); the loop then covers the rest.
# Default (R2-01b, adopted): skip + bulk + 2 entries per loop iteration + static prefixes per bucket. Bit-identical to
# the R2-01 kernel on device (T = 8/16/32) and in the simulator; late-decode MoE per layer b8 0.571 -> 0.534 ms,
# b16 0.871 -> 0.744 ms. "none" = the R2-01 kernel.
V2_DEFAULT = "skip,bulk,p2,pre8:16,pre16:24,pre32:40"
_V2_RAW = os.environ.get("NEMOTRON_H_MOE_ROUTED_V2", V2_DEFAULT).strip()
V2 = {} if _V2_RAW in ("", "none") else {"skip": False, "per_iter": 2, "pre": {}}
for _o in ([] if not V2 else _V2_RAW.split(",")):
    _o = _o.strip()
    if _o in ("skip", "bulk", "wsync", "wfirst", "one", "dsync"):
        V2[_o] = True
    elif _o.startswith("p") and _o[1:].isdigit():
        V2["per_iter"] = int(_o[1:])
    elif _o.startswith("pre") and ":" in _o:
        _t, _k = _o[3:].split(":")
        V2["pre"][int(_t)] = int(_k)
    elif _o:
        raise ValueError(f"unknown NEMOTRON_H_MOE_ROUTED_V2 option {_o!r}")
if V2 and (V2["per_iter"] % 2 or any(k % 2 for k in V2["pre"].values())):
    raise ValueError("NEMOTRON_H_MOE_ROUTED_V2: pN and preT:K need even N / K")
_S2_V2 = 64 + 8 + 2 if not V2 else max([64 + V2["per_iter"]] + list(V2["pre"].values())) + 2


def _v2_cfg(t_n):
    """(static prefix entries, entries per loop iteration) for a T-token graph."""
    return V2["pre"].get(t_n, 0), V2["per_iter"]


@nki.jit
def moe_routed_kernel_v2(x: Tensor, up: Tensor, down: Tensor, wt: Tensor, elist: Tensor, erow: Tensor, niter: Tensor,
                         geom: Tensor):
    """``moe_routed_kernel`` with the R2-01b options (same arguments; ``niter`` = loop iterations after the static
    prefix). Same per-entry compute and per-core expert order, so the outputs are bit-identical."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("moe_routed", (0, 1), 2)
    t_n, h = x.shape
    rows, pt = geom.shape
    nj = (rows + pt - 1) // pt
    pts = [min(pt, rows - j * pt) for j in range(nj)]
    nh = h // 128
    assert nh * 128 == h and t_n <= 128 and rows <= _PSUM_COLS
    s2 = elist.shape[0] // n_prgs
    base = prg_id * s2
    n_chunks = (h + _PSUM_COLS - 1) // _PSUM_COLS
    pre, per_iter = _v2_cfg(t_n)
    out = nl.ndarray((n_prgs, t_n, h), dtype=nl.float32, buffer=nl.shared_hbm)
    sw = dict(dge_mode=nisa.dge_mode.swdge)
    if V2["skip"]:
        sw["oob_mode"] = nisa.oob_mode.skip

    x_sb = nl.ndarray((t_n, h), dtype=x.dtype, buffer=nl.sbuf)
    xT = nl.ndarray((128, nh, t_n), dtype=x.dtype, buffer=nl.sbuf)
    y_acc = nl.ndarray((t_n, h), dtype=nl.float32, buffer=nl.sbuf)

    def _xprep():
        nisa.dma_copy(dst=x_sb, src=x)
        for hh in range(nh):
            tx = nl.ndarray((128, t_n), dtype=x.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=tx, data=x_sb[0:t_n, hh * 128 : (hh + 1) * 128])
            nisa.tensor_copy(dst=xT[0:128, hh, 0:t_n], src=tx)
        nisa.memset(dst=y_acc, value=0.0)

    if not V2.get("wfirst"):
        _xprep()

    if V2.get("bulk"):
        e_all = nl.ndarray((1, pre + 1), dtype=nl.int32, buffer=nl.sbuf)
        r_all = nl.ndarray((1, pre + 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=r_all, src=erow.ap(pattern=[[1, 1], [1, pre + 1]], offset=base),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        nisa.dma_copy(dst=e_all, src=elist.ap(pattern=[[1, 1], [1, pre + 1]], offset=base),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)

    def _fetch_static(pos):
        if V2.get("bulk"):
            return e_all[0:1, pos:pos + 1], r_all[0:1, pos:pos + 1]
        e_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        r_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e_sb, src=elist.ap(pattern=[[1, 1], [1, 1]], offset=base + pos),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        nisa.dma_copy(dst=r_sb, src=erow.ap(pattern=[[1, 1], [1, 1]], offset=base + pos),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        return e_sb, r_sb

    cur = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cur, src=niter.ap(pattern=[[1, 1], [1, 1]], offset=1))  # niter[1] holds 0
    nisa.tensor_scalar(dst=cur, data=cur, op0=nl.add, operand0=pre + 1)  # first entry the loop fetches

    def _fetch_dyn(e_sb, r_sb):
        nisa.dma_copy(dst=e_sb, src=elist.ap(pattern=[[1, 1], [1, 1]], offset=base, scalar_offset=cur,
                                             indirect_dim=0),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        nisa.dma_copy(dst=r_sb, src=erow.ap(pattern=[[1, 1], [1, 1]], offset=base, scalar_offset=cur,
                                            indirect_dim=0),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        nisa.tensor_scalar(dst=cur, data=cur, op0=nl.add, operand0=1)

    hw = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    if V2["skip"]:
        hw["oob_mode"] = nisa.oob_mode.skip

    def _load(r_sb, up_sb, dn_sb):
        if V2.get("one"):
            assert pt * nj == rows
            for src, dst in ((up, up_sb), (down, dn_sb)):
                nisa.dma_copy(dst=dst[0:pt, 0:nj, 0:h],
                              src=src.ap(pattern=[[h, pt], [pt * h, nj], [1, h]], offset=0, scalar_offset=r_sb,
                                         indirect_dim=0), **(hw if (V2.get("dsync") and src is down) else sw))
            return
        for j in range(nj):
            for src, dst in ((up, up_sb), (down, dn_sb)):
                nisa.dma_copy(dst=dst[0:pts[j], j, 0:h],
                              src=src.ap(pattern=[[h, pts[j]], [1, h]], offset=j * pt * h, scalar_offset=r_sb,
                                         indirect_dim=0), **(hw if (V2.get("dsync") and src is down) else sw))

    def _compute(e_sb, up_sb, dn_sb):
        w_sb = nl.ndarray((t_n, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=wt.ap(pattern=[[1, t_n], [1, 1]], offset=0, scalar_offset=e_sb, indirect_dim=0),
                      dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.sync if V2.get("wsync") else nisa.engine.scalar)
        upT = nl.ndarray((128, nh, rows), dtype=up.dtype, buffer=nl.sbuf)
        for hh in range(nh):
            tp = nl.ndarray((128, rows), dtype=up.dtype, buffer=nl.psum)
            for j in range(nj):
                nisa.nc_transpose(dst=tp[0:128, j * pt : j * pt + pts[j]],
                                  data=up_sb[0:pts[j], j, hh * 128 : (hh + 1) * 128])
            eng = nisa.engine.scalar if (COPY_ENG == "va" and hh % 2 == 1) else nisa.engine.vector
            nisa.tensor_copy(dst=upT[0:128, hh, 0:rows], src=tp[0:128, 0:rows], engine=eng)
        pu = nl.ndarray((t_n, rows), dtype=nl.float32, buffer=nl.psum)
        for hh in range(nh):
            nisa.nc_matmul(dst=pu, stationary=xT[0:128, hh, 0:t_n], moving=upT[0:128, hh, 0:rows],
                           accumulate=hh > 0)
        r = nl.ndarray((t_n, rows), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r, data=pu, op0=nl.maximum, operand0=0.0)
        a_sb = nl.ndarray((t_n, rows), dtype=x.dtype, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=a_sb, data=r, op0=nl.multiply, operand0=w_sb[0:t_n, 0:1],
                                  op1=nl.multiply, operand1=r)
        aT = nl.ndarray((pt, nj, t_n), dtype=x.dtype, buffer=nl.sbuf)
        for j in range(nj):
            ta = nl.ndarray((pts[j], t_n), dtype=x.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=ta, data=a_sb[0:t_n, j * pt : j * pt + pts[j]])
            nisa.tensor_copy(dst=aT[0:pts[j], j, 0:t_n], src=ta)
        for c in range(n_chunks):
            c0 = c * _PSUM_COLS
            csz = min(_PSUM_COLS, h - c0)
            py = nl.ndarray((t_n, csz), dtype=nl.float32, buffer=nl.psum)
            for j in range(nj):
                nisa.nc_matmul(dst=py, stationary=aT[0:pts[j], j, 0:t_n], moving=dn_sb[0:pts[j], j, c0 : c0 + csz],
                               accumulate=j > 0)
            nisa.tensor_tensor(dst=y_acc[0:t_n, c0 : c0 + csz], data1=y_acc[0:t_n, c0 : c0 + csz], data2=py,
                               op=nl.add)

    bufs = [(nl.ndarray((pt, nj, h), dtype=up.dtype, buffer=nl.sbuf),
             nl.ndarray((pt, nj, h), dtype=down.dtype, buffer=nl.sbuf)) for _ in range(2)]
    # prologue + static prefix: entry j uses buffer j % 2; entry j + 1 is loaded before entry j computes
    regs = {0: _fetch_static(0)}
    _load(regs[0][1], *bufs[0])
    if V2.get("wfirst"):
        _xprep()
    for j in range(pre):
        regs[j + 1] = _fetch_static(j + 1)
        _load(regs[j + 1][1], *bufs[(j + 1) % 2])
        _compute(regs[j][0], *bufs[j % 2])
    # loop: entry registers of the two buffers (entry pre is already fetched into regs[pre])
    lr = [regs[pre], (nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf),
                      nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf))]

    n_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=n_sb, src=niter.ap(pattern=[[1, 1], [1, 1]], offset=0))
    n_reg = nisa.register_alloc()
    nisa.register_load(n_reg, n_sb)

    def _body(_i):
        for s in range(per_iter):
            nxt = lr[(s + 1) % 2]
            _fetch_dyn(*nxt)
            _load(nxt[1], *bufs[(s + 1) % 2])
            _compute(lr[s % 2][0], *bufs[s % 2])

    nl.fori_loop(0, n_reg, _body)
    nisa.dma_copy(dst=out[prg_id, 0:t_n, 0:h], src=y_acc)
    return out


def _routed_fp8_body(xt, up_t, down, c, wt, elist, erow, eup, niter, geom, nbuf, feats=frozenset()):
    """FP8 variant of ``moe_routed_kernel`` (R2-05b, ``NEMOTRON_H_FP8=1``): same compact expert list, parity split
    and double-buffered software-DGE loads, but on the FP8 tiles of ``kernels/moe_fp8.py``:

        uT[i, t] = sum_hh upT[:, hh, i]^T @ xT[:, hh, t]     # FP8 up tile stationary, I on partitions (no transposes)
        aT[i, t] = relu(uT)^2 * c_i                          # c = s_up^2 s_down (dequantization), bf16
        y[t, :] += w[t, e] * (aT^T @ down)                   # FP8 down tiles moving; w applied per PSUM row

    Args:
        xt: ``[H, T]`` bf16 (x transposed by the caller). up_t: ``[E * 128, nh * I]`` FP8 (``moe_fp8.up_t_layout``).
        down: ``[E * I, H]`` FP8. c: ``[E * I + 128, 1]`` float32.
        wt, elist, erow, niter, geom: as ``moe_routed_kernel``, but ``niter`` = loop iterations of ``nbuf`` entries
            per core. eup: same layout as erow, ``e * 128`` (pads: 0).
        nbuf: load buffers (2, 3 or 4). Each loop iteration computes ``nbuf`` entries; slot ``s`` is refilled with
            entry ``(i + 1) * nbuf + s`` right after its compute, so ``nbuf - 1`` expert loads are in flight while one
            expert computes (2 = the R2-01 A/B scheme; 4 was 1-13% faster on device in R2-05b, see results).
        feats: R2-05c options (``NEMOTRON_H_FP8_ROUTED_V2``; empty = the R2-05b kernel; the module default is skip,peel):
            "skip": the weight loads use ``oob_mode=skip``; ``routed_dispatch(skip_from=nbuf)`` gives pad entries
                    past the first ``nbuf`` list positions an out-of-range row, so their loads are skipped (the slot
                    keeps the previous, finite expert weights; the pad's routing weight is 0).
            "peel": the first loop iteration is unrolled before the loop (its computes overlap the prologue loads,
                    which the loop entry otherwise waits for); ``niter[0]`` then holds the remaining iterations.
            "psum": routing weight folded into the activation (``aT[i, t] = bf16(relu(uT)^2 c_i * w_t)``, w
                    broadcast over partitions by a DMA) and the down projection accumulated in PSUM over all entries
                    (no Vector Engine accumulation into SBUF per 512-column chunk); implies "peel" (the first entry
                    starts the accumulation). Same op sequence and per-core expert order as the FP8 dense kernel
                    (``moe_fp8._expert``, experts of parity ``prg_id`` ascending, unpicked experts add exact zeros), so
                    the outputs should equal ``moe_fp8_dense``'s.
            "hw": weight loads on hardware DGE (Sync queue) instead of software DGE (GpSimd).
            "rot": (nbuf 3/4) the last slot is refilled at the start of the next iteration instead of after its
                    compute, so its load overlaps the other slots' computes and the loop's end-of-iteration DMA wait
                    covers only the other slots' loads.
            "actrelu": relu(uT) on the Scalar (Activation) engine instead of the Vector engine.
            "uhw": only the up tile on hardware DGE (Sync); "dhw": only the down tiles, alternating Sync / Scalar
                    (two DMA queues in flight with the software-DGE up load).
            "wfold": numerics of the FP8 dense kernel: the routing weight is applied to the fp32 activation before
                    its bf16 rounding (``aT = bf16(relu(uT)^2 c * w)``, w broadcast over partitions by a DMA) and
                    each expert's down product is added to the SBUF accumulator (R2-05b: ``bf16(relu^2 c)``, w on
                    the PSUM rows). Not bit-identical to ``moe_fp8_dense`` (that kernel accumulates in PSUM).
            "psum": NOT COMPILABLE (neuronx-cc NCC_IBIR092: a PSUM accumulator cannot be live across the dynamic
                    loop); kept for the record.
            "early": (nbuf 3/4) a slot's next entry is fetched and its up tile load issued right after the slot's up
                    matmuls (the up buffer is free then), the down and c loads after its down matmuls.
    Returns:
        ``[n_prgs, T, H]`` float32 partial sums.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("moe_routed_fp8", (0, 1), 2)
    h, t_n = xt.shape
    rows, pt = geom.shape
    nj = (rows + pt - 1) // pt
    pts = [min(pt, rows - j * pt) for j in range(nj)]
    nh = h // 128
    assert nh * 128 == h and t_n <= 128 and rows <= _PSUM_COLS
    s2 = elist.shape[0] // n_prgs
    base = prg_id * s2
    n_chunks = (h + _PSUM_COLS - 1) // _PSUM_COLS
    out = nl.ndarray((n_prgs, t_n, h), dtype=nl.float32, buffer=nl.shared_hbm)

    xT = nl.ndarray((128, nh, t_n), dtype=xt.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xT, src=xt.ap(pattern=[[t_n, 128], [128 * t_n, nh], [1, t_n]], offset=0))
    # y_acc and w_sb span all 128 partitions so both start at partition 0 (T=32: NCC_IBIR297, "base partition for
    # access is expected to be equal if both inputs are in SB", when w_sb [32, 1] was placed at another base)
    y_acc = nl.ndarray((128, h), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=y_acc[0:t_n, 0:h], value=0.0)

    cur = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cur, src=niter.ap(pattern=[[1, 1], [1, 1]], offset=1))  # niter[1] holds 0
    sw = dict(dge_mode=nisa.dge_mode.swdge)
    if "hw" in feats:
        sw = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    if "skip" in feats:
        sw["oob_mode"] = nisa.oob_mode.skip
    psum_acc = "psum" in feats
    wfold = "wfold" in feats
    peel = "peel" in feats or psum_acc
    if psum_acc:
        y_ps = [nl.ndarray((t_n, min(_PSUM_COLS, h - ci * _PSUM_COLS)), dtype=nl.float32, buffer=nl.psum)
                for ci in range(n_chunks)]
    first = [True]  # the first (peeled, static) compute starts the PSUM accumulation

    def _fetch(e_sb, r_sb, u_sb):
        for src, dst in ((elist, e_sb), (erow, r_sb), (eup, u_sb)):
            nisa.dma_copy(dst=dst, src=src.ap(pattern=[[1, 1], [1, 1]], offset=base, scalar_offset=cur,
                                              indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)

    def _hw(eng):
        d = dict(dge_mode=nisa.dge_mode.hwdge, engine=eng)
        if "skip" in feats:
            d["oob_mode"] = nisa.oob_mode.skip
        return d

    def _load(r_sb, u_sb, up_sb, dn_sb, c_sb, which="ud"):
        if "u" in which:
            nisa.dma_copy(dst=up_sb, src=up_t.ap(pattern=[[nh * rows, 128], [rows, nh], [1, rows]], offset=0,
                                                 scalar_offset=u_sb, indirect_dim=0),
                          **(_hw(nisa.engine.sync) if "uhw" in feats else sw))
        if "d" not in which:
            return
        for j in range(nj):
            dsw = _hw(nisa.engine.sync if j % 2 == 0 else nisa.engine.scalar) if "dhw" in feats else sw
            nisa.dma_copy(dst=dn_sb[0:pts[j], j, 0:h],
                          src=down.ap(pattern=[[h, pts[j]], [1, h]], offset=j * pt * h, scalar_offset=r_sb,
                                      indirect_dim=0), **dsw)
        nisa.dma_copy(dst=c_sb, src=c.ap(pattern=[[1, pt], [pt, nj]], offset=0, scalar_offset=r_sb,
                                         indirect_dim=0), **sw)

    def _load_w(S):
        """wfold: the slot's routing weights w_t for every partition ([128, T]), loaded with the slot's weights."""
        if wfold:
            nisa.dma_copy(dst=S["w"], src=wt.ap(pattern=[[0, 128], [1, t_n]], offset=0, scalar_offset=S["e"],
                                                indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)

    def _compute(e_sb, up_sb, dn_sb, c_sb, after_up=None, w_bc=None):
        if psum_acc:
            return _compute_psum(e_sb, up_sb, dn_sb, c_sb)
        w_sb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        if wfold:  # w_bc: the slot's [128, T] tile (_load_w)
            pass
        else:
            nisa.dma_copy(dst=w_sb[0:t_n, 0:1], src=wt.ap(pattern=[[1, t_n], [1, 1]], offset=0, scalar_offset=e_sb, indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
        aT = nl.ndarray((pt, nj, t_n), dtype=xt.dtype, buffer=nl.sbuf)
        for j in range(nj):
            sz = pts[j]
            pu = nl.ndarray((pt, t_n), dtype=nl.float32, buffer=nl.psum)
            for hh in range(nh):
                nisa.nc_matmul(dst=pu[0:sz, 0:t_n], stationary=up_sb[0:128, hh, j * pt : j * pt + sz],
                               moving=xT[0:128, hh, 0:t_n], accumulate=hh > 0)
            r = nl.ndarray((pt, t_n), dtype=nl.float32, buffer=nl.sbuf)
            if "actrelu" in feats:  # relu on the Scalar (Activation) engine: frees the Vector engine, exact
                nisa.activation(dst=r[0:sz, 0:t_n], op=nl.relu, data=pu[0:sz, 0:t_n])
            else:
                nisa.tensor_scalar(dst=r[0:sz, 0:t_n], data=pu[0:sz, 0:t_n], op0=nl.maximum, operand0=0.0)
            if wfold:  # as moe_fp8._expert ("col"): a_f = relu^2 c (fp32), aT = bf16(a_f * w)
                a_f = nl.ndarray((pt, t_n), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=a_f[0:sz, 0:t_n], data=r[0:sz, 0:t_n], op0=nl.multiply,
                                          operand0=c_sb[0:sz, j : j + 1], op1=nl.multiply, operand1=r[0:sz, 0:t_n])
                nisa.tensor_tensor(dst=aT[0:sz, j, 0:t_n], data1=a_f[0:sz, 0:t_n], data2=w_bc[0:sz, 0:t_n],
                                   op=nl.multiply)
                continue
            nisa.scalar_tensor_tensor(dst=aT[0:sz, j, 0:t_n], data=r[0:sz, 0:t_n], op0=nl.multiply,
                                      operand0=c_sb[0:sz, j : j + 1], op1=nl.multiply, operand1=r[0:sz, 0:t_n])
        if after_up is not None:  # "early": the up buffer is free once the up matmuls are done
            after_up()
        for ci in range(n_chunks):
            c0 = ci * _PSUM_COLS
            csz = min(_PSUM_COLS, h - c0)
            py = nl.ndarray((t_n, csz), dtype=nl.float32, buffer=nl.psum)
            for j in range(nj):
                nisa.nc_matmul(dst=py, stationary=aT[0:pts[j], j, 0:t_n], moving=dn_sb[0:pts[j], j, c0 : c0 + csz],
                               accumulate=j > 0)
            if wfold:  # y += py (as py * 1.0 + y: the same instruction type as the R2-05b accumulation; exact)
                nisa.scalar_tensor_tensor(dst=y_acc[0:t_n, c0 : c0 + csz], data=py, op0=nl.multiply, operand0=1.0,
                                          op1=nl.add, operand1=y_acc[0:t_n, c0 : c0 + csz])
                continue
            nisa.scalar_tensor_tensor(dst=y_acc[0:t_n, c0 : c0 + csz], data=py, op0=nl.multiply,
                                      operand0=w_sb[0:t_n, 0:1], op1=nl.add, operand1=y_acc[0:t_n, c0 : c0 + csz])


    def _compute_psum(e_sb, up_sb, dn_sb, c_sb):
        # w_t for every partition: [128, T]
        w_bc = nl.ndarray((128, t_n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_bc, src=wt.ap(pattern=[[0, 128], [1, t_n]], offset=0, scalar_offset=e_sb, indirect_dim=0),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
        aT = nl.ndarray((pt, nj, t_n), dtype=xt.dtype, buffer=nl.sbuf)
        for j in range(nj):
            sz = pts[j]
            pu = nl.ndarray((pt, t_n), dtype=nl.float32, buffer=nl.psum)
            for hh in range(nh):
                nisa.nc_matmul(dst=pu[0:sz, 0:t_n], stationary=up_sb[0:128, hh, j * pt : j * pt + sz],
                               moving=xT[0:128, hh, 0:t_n], accumulate=hh > 0)
            r = nl.ndarray((pt, t_n), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=r[0:sz, 0:t_n], data=pu[0:sz, 0:t_n], op0=nl.maximum, operand0=0.0)
            # same op sequence as moe_fp8._expert ("col"): a_f = relu^2 c (fp32), aT = bf16(a_f * w)
            a_f = nl.ndarray((pt, t_n), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=a_f[0:sz, 0:t_n], data=r[0:sz, 0:t_n], op0=nl.multiply,
                                      operand0=c_sb[0:sz, j : j + 1], op1=nl.multiply, operand1=r[0:sz, 0:t_n])
            nisa.tensor_tensor(dst=aT[0:sz, j, 0:t_n], data1=a_f[0:sz, 0:t_n], data2=w_bc[0:sz, 0:t_n],
                               op=nl.multiply)
        for ci in range(n_chunks):
            c0 = ci * _PSUM_COLS
            csz = min(_PSUM_COLS, h - c0)
            for j in range(nj):
                nisa.nc_matmul(dst=y_ps[ci], stationary=aT[0:pts[j], j, 0:t_n],
                               moving=dn_sb[0:pts[j], j, c0 : c0 + csz], accumulate=not (first[0] and j == 0))
        first[0] = False

    regs = []
    for _ in range(nbuf):
        regs.append(dict(e=nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf),
                         r=nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf),
                         u=nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf),
                         up=nl.ndarray((128, nh, rows), dtype=up_t.dtype, buffer=nl.sbuf),
                         dn=nl.ndarray((pt, nj, h), dtype=down.dtype, buffer=nl.sbuf),
                         c=nl.ndarray((pt, nj), dtype=nl.float32, buffer=nl.sbuf),
                         w=nl.ndarray((128, t_n), dtype=nl.float32, buffer=nl.sbuf) if wfold else None))

    def _next(S):
        _fetch(S["e"], S["r"], S["u"])
        _load_w(S)
        _load(S["r"], S["u"], S["up"], S["dn"], S["c"])
        nisa.tensor_scalar(dst=cur, data=cur, op0=nl.add, operand0=1)

    rot = "rot" in feats and nbuf != 2
    if nbuf == 2:
        A, B = regs
        _next(A)  # prologue: entry 0 -> A
    else:
        for S in (regs[:-1] if rot else regs):  # prologue: entries 0 .. nbuf - 1 ("rot": 0 .. nbuf - 2)
            _next(S)
    n_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=n_sb, src=niter.ap(pattern=[[1, 1], [1, 1]], offset=0))
    n_reg = nisa.register_alloc()
    nisa.register_load(n_reg, n_sb)

    def _next_up(S):
        _fetch(S["e"], S["r"], S["u"])
        nisa.tensor_scalar(dst=cur, data=cur, op0=nl.add, operand0=1)
        _load_w(S)
        _load(S["r"], S["u"], S["up"], S["dn"], S["c"], "u")

    def _body(_i):
        if rot:  # the last slot's entry is loaded at the start of its own iteration (overlaps the other computes),
            # so the iteration-end DMA wait only covers the loads of slots 0 .. nbuf - 2
            last = regs[-1]
            _next(last)
            for S in regs[:-1]:
                _compute(S["e"], S["up"], S["dn"], S["c"], w_bc=S["w"])
                _next(S)
            _compute(last["e"], last["up"], last["dn"], last["c"], w_bc=last["w"])
            return
        if nbuf != 2 and "early" in feats:  # next up load right after the up matmuls, down + c after the down
            for S in regs:
                _compute(S["e"], S["up"], S["dn"], S["c"], after_up=lambda S=S: _next_up(S), w_bc=S["w"])
                _load(S["r"], S["u"], S["up"], S["dn"], S["c"], "d")
            return
        if nbuf != 2:  # entries i * nbuf + s
            for S in regs:
                _compute(S["e"], S["up"], S["dn"], S["c"], w_bc=S["w"])
                _next(S)
            return
        _next(B)                                   # entry 2i + 1 -> B
        _compute(A["e"], A["up"], A["dn"], A["c"], w_bc=A["w"])  # entry 2i
        _next(A)                                   # entry 2i + 2 -> A
        _compute(B["e"], B["up"], B["dn"], B["c"], w_bc=B["w"])  # entry 2i + 1

    if peel:  # iteration 0 before the loop; niter[0] = remaining iterations
        _body(0)
    nl.fori_loop(0, n_reg, _body)
    if psum_acc:
        for ci in range(n_chunks):
            c0 = ci * _PSUM_COLS
            csz = min(_PSUM_COLS, h - c0)
            nisa.tensor_copy(dst=y_acc[0:t_n, c0 : c0 + csz], src=y_ps[ci])
    nisa.dma_copy(dst=out[prg_id, 0:t_n, 0:h], src=y_acc[0:t_n, 0:h])
    return out


# One @nki.jit function per buffer count (a module-global switch read inside one jit function did not take effect
# in R2-05); the torch wrapper picks the function at trace time.
@nki.jit
def moe_routed_fp8_kernel(xt: Tensor, up_t: Tensor, down: Tensor, c: Tensor, wt: Tensor, elist: Tensor,
                          erow: Tensor, eup: Tensor, niter: Tensor, geom: Tensor):
    return _routed_fp8_body(xt, up_t, down, c, wt, elist, erow, eup, niter, geom, 2)


@nki.jit
def moe_routed_fp8_b3_kernel(xt: Tensor, up_t: Tensor, down: Tensor, c: Tensor, wt: Tensor, elist: Tensor,
                             erow: Tensor, eup: Tensor, niter: Tensor, geom: Tensor):
    return _routed_fp8_body(xt, up_t, down, c, wt, elist, erow, eup, niter, geom, 3)


@nki.jit
def moe_routed_fp8_b4_kernel(xt: Tensor, up_t: Tensor, down: Tensor, c: Tensor, wt: Tensor, elist: Tensor,
                             erow: Tensor, eup: Tensor, niter: Tensor, geom: Tensor):
    return _routed_fp8_body(xt, up_t, down, c, wt, elist, erow, eup, niter, geom, 4)


_FP8_KERNELS = {2: moe_routed_fp8_kernel, 3: moe_routed_fp8_b3_kernel, 4: moe_routed_fp8_b4_kernel}
# Load buffers of the FP8 routed kernel (R2-05b): 4 (default) or 2 / 3.
FP8_BUFS = int(os.environ.get("NEMOTRON_H_FP8_ROUTED_BUFS", "4"))
# R2-05c kernel options (comma list of skip, peel, psum, hw, early, wfold, uhw, dhw, actrelu, rot; see _routed_fp8_body).
# Default "skip,peel" (R2-05c: bit-identical outputs to the R2-05b kernel on device, 4-8% less device time per MoE
# layer at T=8/16); "none" = the R2-05b kernel. "skip,peel,wfold" = the FP8 dense kernel's rounding (needed to pass the
# batch-invariance gate with NEMOTRON_H_MOE_ROUTED_BUCKETS=8,16,32).
_V2_RAW = os.environ.get("NEMOTRON_H_FP8_ROUTED_V2", "skip,peel").strip()
FP8_V2 = frozenset() if _V2_RAW == "none" else frozenset(v.strip() for v in _V2_RAW.split(",") if v.strip())
if FP8_BUFS not in _FP8_KERNELS and not (FP8_V2 and FP8_BUFS in (5, 6, 8)):  # 5/6/8 only with R2-05c options
    raise ValueError(f"NEMOTRON_H_FP8_ROUTED_BUFS must be 2, 3 or 4 (5, 6, 8 with NEMOTRON_H_FP8_ROUTED_V2), got {FP8_BUFS}")
_V2_OPTS = {"skip", "peel", "psum", "hw", "early", "wfold", "uhw", "dhw", "actrelu", "rot"}
if not FP8_V2 <= _V2_OPTS:
    raise ValueError(f"NEMOTRON_H_FP8_ROUTED_V2: unknown options {sorted(FP8_V2 - _V2_OPTS)}")
_V2_KERNELS = {}


def _fp8_kernel(nbuf: int, feats: frozenset):
    """The @nki.jit kernel for (nbuf, feats); one separately named function per combination (trace/NEFF caches)."""
    if not feats:
        return _FP8_KERNELS[nbuf]
    key = (nbuf, feats)
    if key not in _V2_KERNELS:
        def _k(xt: Tensor, up_t: Tensor, down: Tensor, c: Tensor, wt: Tensor, elist: Tensor, erow: Tensor,
               eup: Tensor, niter: Tensor, geom: Tensor):
            return _routed_fp8_body(xt, up_t, down, c, wt, elist, erow, eup, niter, geom, nbuf, feats)

        name = f"moe_routed_fp8_b{nbuf}_{'_'.join(sorted(feats))}_kernel"
        _k.__name__ = _k.__qualname__ = name
        _V2_KERNELS[key] = nki.jit(_k)
    return _V2_KERNELS[key]


if FP8_V2:  # build the kernel before torch.compile traces the wrapper (Dynamo cannot trace its creation)
    _fp8_kernel(FP8_BUFS, FP8_V2)
# list slots per core for the FP8 kernel: 64 experts + the prefetch of up to 4 entries past the last iteration
_S2_FP8 = 72


def _f2i(v: Tensor) -> Tensor:
    """Exact float32 -> int32 for integer values in [0, 2^23) by a bit cast."""
    return (v + 8388608.0).view(torch.int32) - 0x4B000000


def routed_dispatch(onehot_sel: Tensor, aff: Tensor, rows: int, s2: int = _S2, with_eup: bool = False,
                    per_iter: int = 2, skip_from: int = 0, pre: int = 0, peel: bool = False):
    """Static-shape compaction of the selected experts.

    ``onehot_sel``: ``[E]`` float32 1/0 (expert picked by any token); ``aff``: ``[T, E]`` float32 routing weights
    (0 for unpicked). Returns ``(wt [E + 1, T], elist [2 * s2, 1], erow [2 * s2, 1], niter [2, 1])`` (int32 except wt).
    ``skip_from > 0`` (R2-05c "skip"): pad entries at list positions >= skip_from (per core) get the out-of-range rows
    ``E * rows`` / ``E * 128``, so the kernel's ``oob_mode=skip`` loads skip them. ``peel``: ``niter[0]`` = iterations - 1.
    """
    E = onehot_sel.shape[0]
    t = aff.shape[0]
    dev = aff.device
    sel = onehot_sel
    ids = torch.arange(E, device=dev, dtype=torch.float32)
    dummy = float(2 * s2)
    if SPLIT == "parity":
        par = ids - 2.0 * torch.floor(ids * 0.5)  # e % 2 (exact)
        sel1 = sel * par
        sel0 = sel - sel1
        pos0 = torch.cumsum(sel0, dim=0) - sel0  # exclusive rank among the selected experts of the same parity
        pos1 = torch.cumsum(sel1, dim=0) - sel1
        dst = torch.where(sel > 0.5, par * float(s2) + torch.where(par > 0.5, pos1, pos0),
                          torch.full_like(ids, dummy))
    else:
        pos = torch.cumsum(sel, dim=0) - sel  # exclusive rank among the selected experts (exact in float32)
        half = torch.floor(pos * 0.5)
        core = pos - 2.0 * half
        dst = torch.where(sel > 0.5, core * float(s2) + half, torch.full_like(pos, dummy))
    elist_f = torch.full((2 * s2 + 1,), float(E), dtype=torch.float32, device=dev).scatter(
        0, _f2i(dst).to(torch.int64), ids)[: 2 * s2]
    real = elist_f < float(E) - 0.5
    pad_row = torch.zeros_like(elist_f)
    if skip_from > 0:  # R2-01b "skip": pads past the first skip_from positions of each core: out-of-range rows
        lpos = torch.remainder(torch.arange(2 * s2, device=dev, dtype=torch.float32), float(s2))
        pad_row = torch.where(lpos >= float(skip_from), torch.ones_like(elist_f), pad_row)
    erow_f = torch.where(real, elist_f * float(rows), pad_row * float(E * rows))
    if SPLIT == "parity" and pre > 0:  # R2-01b static prefix: loop iterations after the first `pre` entries
        rest = torch.clamp(torch.maximum(sel0.sum(), sel1.sum()) - float(pre), min=0.0)
        niter_f = torch.floor((rest + (per_iter - 0.5)) * (1.0 / per_iter))
    elif SPLIT == "parity":
        if per_iter == 2:
            niter_f = torch.floor((torch.maximum(sel0.sum(), sel1.sum()) + 1.0) * 0.5)  # ceil(max(n0, n1) / 2)
        else:  # ceil(max(n0, n1) / per_iter); the +0.5 keeps float32 rounding away from the integer boundaries
            niter_f = torch.floor((torch.maximum(sel0.sum(), sel1.sum()) + (per_iter - 0.5)) * (1.0 / per_iter))
    else:
        niter_f = torch.floor((sel.sum() + 3.0) * 0.25)  # ceil(ceil(n_u / 2) / 2)
    if peel:  # the kernel runs iteration 0 before the loop
        niter_f = torch.clamp(niter_f - 1.0, min=0.0)
    niter = torch.stack([niter_f, torch.zeros_like(niter_f)]).view(2, 1)
    wt = torch.cat([aff.t(), torch.zeros((1, t), dtype=aff.dtype, device=dev)], dim=0).reshape((E + 1) * t, 1)
    if with_eup:  # FP8 kernel: first up_t row of the entry's expert (e * 128; pads: 0)
        eup_f = torch.where(real, elist_f * 128.0, pad_row * float(E * 128))
        return (wt, _f2i(elist_f * float(t)).view(-1, 1), _f2i(erow_f).view(-1, 1), _f2i(niter),
                _f2i(eup_f).view(-1, 1))
    return wt, _f2i(elist_f * float(t)).view(-1, 1), _f2i(erow_f).view(-1, 1), _f2i(niter)


def moe_routed_decode(x: Tensor, up: Tensor, down: Tensor, aff: Tensor, sel: Tensor, rows: int, run=None) -> Tensor:
    """Routed-expert sum for ``x`` ``[T, H]``; ``aff`` ``[T, E]`` float32 (0 for unpicked experts), ``sel`` ``[E]``
    float32 0/1. Returns ``[T, H]`` in ``x.dtype``. ``run`` replaces the device launch (gets kwargs and kernel)."""
    if V2:  # R2-01b kernel options (NEMOTRON_H_MOE_ROUTED_V2)
        if SPLIT != "parity":
            raise ValueError("NEMOTRON_H_MOE_ROUTED_V2 needs NEMOTRON_H_MOE_ROUTED_SPLIT=parity")
        pre, per_iter = _v2_cfg(x.shape[0])
        wt, elist, erow, niter = routed_dispatch(sel, aff.to(torch.float32), rows, s2=_S2_V2, per_iter=per_iter,
                                                 skip_from=2 if V2["skip"] else 0, pre=pre)
        kern = moe_routed_kernel_v2
        if V2.get("one") and rows % 4 == 0 and rows // 4 <= 128:
            pt = rows // 4
            kw = dict(x=x, up=up, down=down, wt=wt, elist=elist, erow=erow, niter=niter,
                      geom=torch.zeros((rows, pt), dtype=torch.int32, device=x.device))
            out = run(kw, kern) if run is not None else wrap_nki(kern)[2](**kw)
            return out.sum(0).to(x.dtype)
    else:
        wt, elist, erow, niter = routed_dispatch(sel, aff.to(torch.float32), rows)
        kern = moe_routed_kernel
    pt = min(rows, 128)
    kw = dict(x=x, up=up, down=down, wt=wt, elist=elist, erow=erow, niter=niter,
              geom=torch.zeros((rows, pt), dtype=torch.int32, device=x.device))
    out = run(kw, kern) if run is not None else wrap_nki(kern)[2](**kw)
    return out.sum(0).to(x.dtype)


def moe_routed_decode_fp8(x: Tensor, up_t: Tensor, down: Tensor, c: Tensor, aff: Tensor, sel: Tensor, rows: int,
                          run=None) -> Tensor:
    """FP8 routed-expert sum (``moe_routed_fp8_kernel``): ``up_t`` / ``down`` / ``c`` as ``kernels/moe_fp8.py``,
    ``aff`` / ``sel`` as ``moe_routed_decode``. Returns ``[T, H]`` in ``x.dtype``."""
    k = FP8_BUFS if SPLIT == "parity" else 2  # the "pos" split's trip count assumes 2 entries per iteration
    feats = FP8_V2
    wt, elist, erow, niter, eup = routed_dispatch(sel, aff.to(torch.float32), rows, s2=_S2_FP8, with_eup=True,
                                                  per_iter=k, skip_from=k if "skip" in feats else 0,
                                                  peel="peel" in feats or "psum" in feats)
    pt = min(rows, 128)
    kw = dict(xt=x.t().contiguous(), up_t=up_t, down=down, c=c, wt=wt, elist=elist, erow=erow, eup=eup, niter=niter,
              geom=torch.zeros((rows, pt), dtype=torch.int32, device=x.device))
    kern = _fp8_kernel(k, feats)
    out = run(kw, kern) if run is not None else wrap_nki(kern)[2](**kw)
    return out.sum(0).to(x.dtype)
