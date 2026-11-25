"""Gather the selected experts' stacked weights, one DMA per expert (T10, optimization step 2).

The dense MoE decode path reads all 128 experts' weights every step (0.64 GB per layer per
rank). At batch 1 only 6 experts contribute. This kernel copies the 6 selected experts' up and
down matrices HBM -> HBM with one DMA each; the expert id is a runtime scalar offset on the
expert dimension (``dge_mode.hwdge``), the same pattern as ``state_cache.py``'s page gather.
The matmuls then run in torch on the gathered ``[k * I, H]`` matrices.

Both stacked weights must be ``[E, I, H]`` (rows of H contiguous): the up weight stored
``[out, in]`` (``NEMOTRON_H_MOE_WT=up``) and the down weight ``[in, out]``.
"""

import os

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki


# Each expert copy is split into SPLIT row blocks issued alternately from the Sync and Scalar
# (Activation) engine queues, so more DMA rings run at once (trace-time constant).
SPLIT = int(os.environ.get("NEMOTRON_H_EXPERT_GATHER_SPLIT", "1"))
# DGE mode of the copies: "hw" (hardware DGE from Sync/Scalar), "sw" (software DGE, GpSimd), or
# "mix" (half the rows each way). Profile (T10, 4-layer, batch 1): every hwdge copy of a core runs
# on ONE DMA engine (E96 / E112 of 16), ~27 GB/s, so the 15 MB per core take ~0.55 ms.
# 4-layer c1 TPOT (1 MoE layer): hw 2.75, sw 2.57, mix 2.54 ms (default), sw in 4 blocks 3.39,
# mix with the sw half in 4 blocks 2.79; hw in 2/4 blocks alternating Sync/Scalar 3.61/2.65-2.80.
# T14 (opt4): option "r116": each expert goes HBM -> SBUF in [116, 2688] row tiles (dynamic offset,
# hardware DGE), then SBUF -> HBM at a static address. Unit test, 8 MoE layers at batch 1, TP=4
# rank-0 shapes (results/trn2/opt4/unit/): dense 1.13, mix 0.755, r116 0.409 ms per layer. A DMA
# with a real 116-partition SBUF side is spread over the DMA engines; the T10 HBM -> HBM copies
# ran on one. Leading-128 patterns with 19.5 KB rows (p128 HBM -> HBM, s128 via SBUF) are slower
# (5.7-5.9 ms per layer). The default stays "mix" (the gated T10 path): with the fused top-k kernel
# (moe_topk.py, NEMOTRON_H_MOE_TOPK_FUSED) the gather is not used, and r116 was not gated.
MODE = os.environ.get("NEMOTRON_H_EXPERT_GATHER_MODE", "mix")


def _expert(w, idx_sb, i, row0=0, nrows=None):
    """Rows ``[row0, row0 + nrows)`` of ``w[idx_sb[i]]``, ``w`` ``[E, rows, R]`` in HBM (runtime scalar offset)."""
    _, rows, r = w.shape
    nrows = rows if nrows is None else nrows
    return w.ap(
        pattern=[[rows * r, 1], [r, nrows], [1, r]],
        offset=row0 * r,
        scalar_offset=idx_sb[i : i + 1, 0:1],
        indirect_dim=0,
    )


def _dense(t, i, row0=0, nrows=None):
    """Rows ``[row0, row0 + nrows)`` of item ``i`` of a dense ``[n, rows, R]`` HBM tensor."""
    _, rows, r = t.shape
    nrows = rows if nrows is None else nrows
    return t.ap(pattern=[[nrows * r, 1], [r, nrows], [1, r]], offset=(i * rows + row0) * r)


def _copy(dst, src, idx_sb, i, q):
    """Copy expert ``idx_sb[i]`` of ``src`` to item ``i`` of ``dst`` in SPLIT row blocks."""
    _, rows, _ = src.shape
    if MODE in ("sw", "mix"):
        # sw: all through software DGE (GpSimd); mix: first half hardware DGE, second half software.
        half = rows // 2 if MODE == "mix" else 0
        if half:
            nisa.dma_copy(dst=_dense(dst, i, 0, half), src=_expert(src, idx_sb, i, 0, half),
                          dge_mode=nisa.dge_mode.hwdge)
        rest = rows - half
        step = (rest + SPLIT - 1) // max(SPLIT, 1)
        for row0 in range(half, rows, step):
            n_r = min(step, rows - row0)
            nisa.dma_copy(dst=_dense(dst, i, row0, n_r), src=_expert(src, idx_sb, i, row0, n_r),
                          dge_mode=nisa.dge_mode.swdge)
        return
    if SPLIT <= 1:
        nisa.dma_copy(dst=_dense(dst, i), src=_expert(src, idx_sb, i), dge_mode=nisa.dge_mode.hwdge)
        return
    step = (rows + SPLIT - 1) // SPLIT
    for b in range(SPLIT):
        row0 = b * step
        nrows = min(step, rows - row0)
        if nrows <= 0:
            continue
        eng = nisa.engine.sync if (q + b) % 2 == 0 else nisa.engine.scalar
        nisa.dma_copy(
            dst=_dense(dst, i, row0, nrows),
            src=_expert(src, idx_sb, i, row0, nrows),
            dge_mode=nisa.dge_mode.hwdge,
            engine=eng,
        )


@nki.jit
def _expert_gather_kernel(up: Tensor, down: Tensor, experts: Tensor):
    """``(up[experts], down[experts])``, each ``[n, rows, R]``.

    Args:
        up, down: ``[E, rows, R]`` (same shape), the stacked expert weights.
        experts: ``[n, 1]`` int32 expert ids.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("expert_gather", (0, 1), 2)
    _, rows, r = up.shape
    n = experts.shape[0]
    up_out = nl.ndarray((n, rows, r), dtype=up.dtype, buffer=nl.shared_hbm)
    down_out = nl.ndarray((n, rows, r), dtype=down.dtype, buffer=nl.shared_hbm)
    idx_sb = nl.ndarray((n, 1), dtype=experts.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_sb, src=experts)
    for i in range(n):
        # up on one physical core, down on the other, so the copies overlap.
        if n_prgs == 1 or prg_id == 0:
            _copy(up_out, up, idx_sb, i, i)
        if n_prgs == 1 or prg_id == 1:
            _copy(down_out, down, idx_sb, i, i)
    return up_out, down_out


# T14 (optimization step "opt4"): ``NEMOTRON_H_EXPERT_GATHER_MODE=p<P>`` (e.g. ``p128``) issues each
# expert copy as ONE DMA whose access pattern is ``[P, rows * R / P]`` instead of ``[1, rows, R]``.
# A DMA transfer is spread over up to 16 DMA engines by its leading (partition-like) dimension, so
# the T10 pattern with a leading 1 ran on one engine (~27 GB/s). The weights are passed as
# ``[E * P, rows * R / P]`` views and the expert ids pre-multiplied by P, so the runtime scalar
# offset (applied on pattern dim 0) lands on row ``idx * P`` = expert ``idx``.
P_SPLIT = int(MODE[1:]) if MODE.startswith("p") else 0
# ``s<P>``: same, but through an SBUF tile ``[P, F]`` (HBM -> SBUF with the dynamic offset, then a
# static SBUF -> HBM store).
S_SPLIT = int(MODE[1:]) if MODE.startswith("s") and MODE[1:].isdigit() else 0
# Engines that trigger the copies in p-mode: "sync", "scalar" or "alt" (alternate per expert).
P_ENGINE = os.environ.get("NEMOTRON_H_EXPERT_GATHER_ENGINE", "alt")


@nki.jit
def _expert_gather_p_kernel(up: Tensor, down: Tensor, experts: Tensor):
    """``up``/``down`` ``[E * P, F]``; ``experts`` ``[n, 1]`` int32 = expert id * P. Returns ``[n * P, F]`` each."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("expert_gather", (0, 1), 2)
    _, f = up.shape
    p = P_SPLIT
    n = experts.shape[0]
    up_out = nl.ndarray((n * p, f), dtype=up.dtype, buffer=nl.shared_hbm)
    down_out = nl.ndarray((n * p, f), dtype=down.dtype, buffer=nl.shared_hbm)
    idx_sb = nl.ndarray((n, 1), dtype=experts.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_sb, src=experts)
    for i in range(n):
        if P_ENGINE == "sync":
            eng = nisa.engine.sync
        elif P_ENGINE == "scalar":
            eng = nisa.engine.scalar
        else:
            eng = nisa.engine.sync if i % 2 == 0 else nisa.engine.scalar
        for which, (src, dst) in enumerate(((up, up_out), (down, down_out))):
            if n_prgs > 1 and prg_id != which:
                continue
            nisa.dma_copy(
                dst=dst.ap(pattern=[[f, p], [1, f]], offset=i * p * f),
                src=src.ap(pattern=[[f, p], [1, f]], offset=0,
                           scalar_offset=idx_sb[i : i + 1, 0:1], indirect_dim=0),
                dge_mode=nisa.dge_mode.hwdge,
                engine=eng,
            )
    return up_out, down_out


# s-mode geometry (trace-time): the weights are viewed ``[E * S_R2, F]`` with ``S_R2 * F = rows * R``,
# and each expert is copied in ``S_R2 / S_PT`` tiles of ``[S_PT, F]``.
#   s128 : S_R2 = 128, F = rows * R / 128, one [128, F] tile per expert
#   r<P> : S_R2 = rows, F = R (natural rows), tiles of P rows (e.g. r116: 4 x [116, 2688])
S_ROWS = int(MODE[1:]) if MODE.startswith("r") and MODE[1:].isdigit() else 0
S_DGE = os.environ.get("NEMOTRON_H_EXPERT_GATHER_DGE", "hw")


@nki.jit
def _expert_gather_s_kernel(up: Tensor, down: Tensor, experts: Tensor, geom: Tensor):
    """``up``/``down`` ``[E * R2, F]``, ``experts`` ``[n, 1]`` = expert id * R2. Every expert goes
    HBM -> SBUF in ``[pt, F]`` tiles (dynamic offset) -> HBM (static address)."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("expert_gather", (0, 1), 2)
    _, f = up.shape
    r2, pt = geom.shape  # geometry passed as the shape of a dummy [R2, PT] tensor
    n = experts.shape[0]
    up_out = nl.ndarray((n * r2, f), dtype=up.dtype, buffer=nl.shared_hbm)
    down_out = nl.ndarray((n * r2, f), dtype=down.dtype, buffer=nl.shared_hbm)
    idx_sb = nl.ndarray((n, 1), dtype=experts.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_sb, src=experts)
    q = 0
    for i in range(n):
        for which, (src, dst) in enumerate(((up, up_out), (down, down_out))):
            if n_prgs > 1 and prg_id != which:
                continue
            for j in range(r2 // pt):
                if P_ENGINE == "sync":
                    eng = nisa.engine.sync
                elif P_ENGINE == "scalar":
                    eng = nisa.engine.scalar
                else:
                    eng = nisa.engine.sync if q % 2 == 0 else nisa.engine.scalar
                q += 1
                t = nl.ndarray((pt, f), dtype=src.dtype, buffer=nl.sbuf)
                src_ap = src.ap(pattern=[[f, pt], [1, f]], offset=j * pt * f,
                                scalar_offset=idx_sb[i : i + 1, 0:1], indirect_dim=0)
                if S_DGE == "sw":
                    nisa.dma_copy(dst=t, src=src_ap, dge_mode=nisa.dge_mode.swdge)
                else:
                    nisa.dma_copy(dst=t, src=src_ap, dge_mode=nisa.dge_mode.hwdge, engine=eng)
                nisa.dma_copy(dst=dst[i * r2 + j * pt : i * r2 + (j + 1) * pt, 0:f], src=t)
    return up_out, down_out


S_R2 = 0  # set by gather_experts before tracing
S_PT = 0


def gather_experts(up: Tensor, down: Tensor, experts: Tensor) -> tuple[Tensor, Tensor]:
    """``(up[experts], down[experts])`` for ``[E, rows, R]`` weights and ``[n]`` expert ids."""
    global S_R2, S_PT
    e, rows, r = up.shape
    n = experts.numel()
    if P_SPLIT:
        ps = P_SPLIT
        if (rows * r) % ps:
            raise ValueError(f"expert size {rows}x{r} not divisible by P={ps}")
        f = rows * r // ps
        u, d = wrap_nki(_expert_gather_p_kernel)[2](
            up=up.reshape(e * ps, f), down=down.reshape(e * ps, f),
            experts=(experts.to(torch.int32) * ps).view(-1, 1),
        )
        return u.view(n, rows, r), d.view(n, rows, r)
    if S_SPLIT or S_ROWS:
        if S_SPLIT:
            S_R2, S_PT = S_SPLIT, S_SPLIT
        else:
            S_R2, S_PT = rows, S_ROWS
        pt = S_PT
        if (rows * r) % S_R2 or S_R2 % pt:
            raise ValueError(f"expert size {rows}x{r}: bad s-mode geometry R2={S_R2} pt={pt}")
        f = rows * r // S_R2
        u, d = wrap_nki(_expert_gather_s_kernel)[2](
            up=up.reshape(e * S_R2, f), down=down.reshape(e * S_R2, f),
            experts=(experts.to(torch.int32) * S_R2).view(-1, 1),
            geom=torch.zeros((S_R2, pt), dtype=torch.int32, device=experts.device),
        )
        return u.view(n, rows, r), d.view(n, rows, r)
    return wrap_nki(_expert_gather_kernel)[2](
        up=up, down=down, experts=experts.to(torch.int32).view(-1, 1)
    )
