"""Routed top-k MoE for prefill chunks (A07): expert-major grouped GEMM over token-sorted slots.

The dense prefill path multiplies every token by all 128 experts (21x the routed FLOPs). Here the
caller sorts the ``T * k`` (token, expert) pairs by expert into a slot buffer whose per-expert
ranges are padded to whole tiles of ``B = 128`` slots, gathers the token rows into it (``xs``),
and the kernel runs, for every expert ``e`` (static loop, so the weight loads have static
addresses and use the normal DMA queues):

    load up_e, down_e (this core's half of the intermediate dim I)
    transpose up_e on the Tensor Engine to [H, I/2] (the stored layout is [out, in])
    for each of the expert's n_e tiles (dynamic loop, n_e = ceil(count_e / B) from HBM):
        xT  = transpose(xs[tile])                      # [H, B]
        aT  = relu(up_e @ xT)^2                        # [I/2, B], fp32 PSUM -> bf16
        y   = (aT.T @ down_e) * w[slot]                # [B, H], fp32 PSUM, routing weight
        ys[core, tile] = y                             # bf16

LNC=2: both physical cores walk the same experts and tiles (dynamic loops must have the same trip
counts on both cores) and each takes half of I, so each returns a partial sum; the caller adds
the two partials and sums the k slots of each token.

Layouts: up ``[E * I, H]`` (``[out, in]``, ``NEMOTRON_H_MOE_WT=up``) and down ``[E * I, H]``
(``[in, out]``), the same as ``moe_topk.py``.
"""

import os

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

B = 128  # slots per tile (tokens on the PSUM partition axis of the down projection)
_PSUM_COLS = 512
_DGE = nisa.dge_mode.hwdge
# Experts per iteration of the dynamic expert loop; 0 = all experts unrolled (static weight
# addresses, long compile). ``NEMOTRON_H_MOE_PREFILL_UNROLL``.
EXPERT_UNROLL = int(os.environ.get("NEMOTRON_H_MOE_PREFILL_UNROLL", "0"))
# Debug ablations (timing only, wrong results): comma list of "tiles" (no tile loop), "wt" (no up transposes).
# Up projection with the token tile as the stationary operand (21 matmuls of N = I/2 per tile instead of
# 42 of N = 128) plus two small transposes of the activation. ``NEMOTRON_H_MOE_PREFILL_UP_XSTAT``.
# Static expert loop (UNROLL=0): weight loads with hardware DGE on two queues (1) or compiler descriptors (0).
WEIGHT_HWDGE = os.environ.get("NEMOTRON_H_MOE_PREFILL_WHWDGE", "0") == "1"
WSPLIT = int(os.environ.get("NEMOTRON_H_MOE_PREFILL_WSPLIT", "1"))
# Static path: transpose the up weight with DMA transposes from HBM instead of on the Tensor Engine.
WT_DMA = os.environ.get("NEMOTRON_H_MOE_PREFILL_WT_DMA", "0") == "1"
WT_DMA_N = int(os.environ.get("NEMOTRON_H_MOE_PREFILL_WT_DMA_N", "0"))
PREFETCH = os.environ.get("NEMOTRON_H_MOE_PREFILL_PREFETCH", "1") == "1"
UP_X_STATIONARY = os.environ.get("NEMOTRON_H_MOE_PREFILL_UP_XSTAT", "1") == "1"
COMBINE = os.environ.get("NEMOTRON_H_MOE_PREFILL_COMBINE", "sum")
# torch slot dispatch variant: "fast" (default: fused slot reduction, bit-cast float -> int), "" (attempt 1),
# "mm" (exclusive cumsums as triangular matmuls; no faster)
DISPATCH_MODE = os.environ.get("NEMOTRON_H_MOE_PREFILL_DISPATCH", "fast")
# How the two cores of a logical core share the work: "i" (both walk all experts, half of I each; attempt 1)
# or "e" (default: expert split, each core takes every other expert with all of I; attempt 2; LNC=2 only).
SPLIT = os.environ.get("NEMOTRON_H_MOE_PREFILL_SPLIT", "e")
ABLATE = set(filter(None, os.environ.get("NEMOTRON_H_MOE_PREFILL_ABLATE", "").split(",")))
# Expert split: how experts are paired on the two cores (T12b). "static" (A07 a2): core c takes experts 2g + c, and
# the pair's shared trip count is the larger of the two tile counts. "sorted": the dispatch ranks the experts by
# tile count and pairs neighbours in that order (the two largest, the next two, ...), so the max-of-pair padding is
# minimal; the kernel then loads each slot's expert weights at a runtime row offset (``erow``).
# Default "sorted" (T12b step 8b: outputs bit-identical to "static", prefill chunk 267.5 -> 258.3 ms, gate passes).
PAIR = os.environ.get("NEMOTRON_H_MOE_PREFILL_PAIR", "sorted")
if PAIR not in ("static", "sorted"):
    raise ValueError(f"NEMOTRON_H_MOE_PREFILL_PAIR must be static or sorted, got {PAIR!r}")
# DGE mode of the runtime-offset weight loads of PAIR=sorted: "sw" (software DGE, all 16 queues; T10b) or "hw".
PAIR_DGE = os.environ.get("NEMOTRON_H_MOE_PREFILL_PAIR_DGE", "sw")
# PAIR=sorted only: sum the k expert rows of each token inside the kernel (T12b). Each core accumulates its experts'
# rows into its own float32 [T, H] plane with DMA scatter read-modify-write (``nisa.dma_compute``); torch adds the
# two planes. The kernel output shrinks from [T * k + 1, H] bf16 (66 MB at T = 2048) to [2 T, H] float32 (44 MB).
# Default 1 (T12b step 8c: gate passes, prefill chunk 258.3 -> 248.8 ms, TTFT -1..-4% on every Must row). Outputs differ
# from PAIR=sorted only by fp32 summation order (the planes accumulate in expert processing order): unit with expert
# outputs spread over 2^+-8 differs in 1 element of 5.5M (rel 7e-8), same error vs the fp32 reference.
ACC = os.environ.get("NEMOTRON_H_MOE_PREFILL_ACC", "1") == "1" and PAIR == "sorted"
# ACC only: build the slot layout with one integer scatter instead of three (T12d). The T12d device profile of the
# 14-layer checkpoint puts ~0.9 ms of every prefill MoE block before the kernel's first tile in the three
# ``[NPAD]`` int32 scatters of the dispatch (slot_pair, slot_tok, slot_dst: 96 GpSimd indirect DMAs each, run one
# after the other). With ``SCATTER1=1`` only slot_pair is scattered; the token of each slot is ``pair // k``
# (exact float arithmetic on the [NPAD] vector) and the kernel adds its core's plane offset ``prg_id * T`` to the
# destination row (pads stay out of bounds). The slot layout and every kernel input value used are unchanged, so
# outputs should be bit-identical to SCATTER1=0. Default 1 (T12d optimization step 12, with TILE0: unit 6.52 -> 6.28 ms
# per layer alone, bit-identical; full-model servers with SCATTER1 only give outputs identical to SCATTER1=0).
SCATTER1 = os.environ.get("NEMOTRON_H_MOE_PREFILL_SCATTER1", "1") == "1"
# ACC only: run the first tile of every expert as straight-line code and only the remaining tiles in the dynamic
# loop (T12d). Every dynamic-loop iteration ends with a barrier over all engines that waits for the tile's
# read-modify-write DMA (~10 us of the ~39 us of a tile in the T12d profile), and most experts have one tile at
# T = 2048. The dispatch gives every expert group at least one tile (empty groups get one all-pad tile) and passes
# ``ntile - 1`` to the kernel. Only for chunks of at least ``TILE0_MIN_T`` tokens (small chunks have many empty experts).
# Default 1 (T12d optimization step 12: unit 6.28 -> 5.05 ms per layer on top of SCATTER1, bit-identical, also with
# adversarial routing; full-model prefill chunk 248.8 -> 219.3 ms; gate passes; TTFT -6..-12% on the Must rows).
TILE0 = os.environ.get("NEMOTRON_H_MOE_PREFILL_TILE0", "1") == "1"
TILE0_MIN_T = int(os.environ.get("NEMOTRON_H_MOE_PREFILL_TILE0_MIN_T", "1024"))
# R2-03b (TILE0 + DEFER only): run the first ``TILES_SL`` tiles of every expert group as straight-line code (TILE0 = 1).
# The dispatch gives every group at least that many tiles (pad tiles: token 0, out-of-bounds plane row, weight 0) and
# passes ``ntile - TILES_SL`` to the kernel, so the experts with up to TILES_SL tiles run no dynamic-loop iteration and
# no all-engine barrier. Only for chunks of at least ``TILES_SL_MIN_T`` tokens. Not kept (R2-03b): faster with uniform
# random routing (unit 6.87 -> 6.54 ms) but the bench prompts' routing leaves 38 of 64 expert pairs per core with one tile,
# and the forced pad tiles made the full-model 4,096-token chunk 286.7 -> 340.9 ms.
TILES_SL = int(os.environ.get("NEMOTRON_H_MOE_PREFILL_TILES_SL", "1"))
TILES_SL_MIN_T = int(os.environ.get("NEMOTRON_H_MOE_PREFILL_TILES_SL_MIN_T", "4096"))
if TILES_SL < 1:
    raise ValueError("NEMOTRON_H_MOE_PREFILL_TILES_SL must be >= 1")


# R2-03b (ACC + SCATTER1 only): pad slots carry the out-of-bounds token row T and the kernel's x gather skips them
# (``oob_mode=skip``), so pad rows move no bytes; their x rows keep stale SBUF data, but every pad row's output stays in
# its own row (all ops are row-local) and its read-modify-write is skipped. Outputs of real rows are unchanged.
# BF16 only: with NEMOTRON_H_FP8=1 the FP8 kernel and its dispatch are unchanged.
# Default 1 since R2-03b (optimization step R2-H: with the bench prompts' skewed routing ~31% of a 4,096-token chunk's slots
# are pads; unit -2.3% (4,096) / -2.4% (2,048) per MoE layer, bit-identical; full-model chunk 287.1 -> 284.5 ms (4,096) and
# 156.8 -> 153.6 ms (2,048); gate details identical record for record to the installed tree; device-0 TTFT p50 -0.7..-3.0%
# in 14 of 14 same-session pairs, TPOT unchanged). "0" = previous dispatch and gather.
PADSKIP = os.environ.get("NEMOTRON_H_MOE_PREFILL_PADSKIP", "1") == "1"

# R2-03d (ACC + SCATTER1 + PADSKIP path, BF16 only): the first pair of each (core, token) in the core's processing order
# (lowest sorted expert rank of that parity) writes its row of the float32 plane (``dma_copy`` bf16 -> fp32) instead of a
# read-modify-write, and the plane is not zeroed; rows a core never touches are masked to 0 in the combine. 0 + y == y in
# float32, so the planes hold the same values (up to the sign of an all-zero row) with ~2 x T x H x 4 bytes less DMA traffic
# per core (the zeroing and the first read). ``slot_dst`` gets two columns: [write row, RMW row] (the other one out of
# bounds). "0" = previous zero + RMW.
FIRSTW = os.environ.get("NEMOTRON_H_MOE_PREFILL_FIRSTW", "0") == "1"
# R2-03d (sorted ACC dispatch, BF16 only, chunks that use TILE_PREFETCH, i.e. >= TILE_PREFETCH_MIN_T tokens): lay each
# core's slot ranges out contiguously in processing order (core 0's experts g = 0, 1, ... then core 1's) instead of by
# expert id. The last tile of an expert then prefetches the next expert's first tile (it read unrelated slots before), and
# the kernel carries the slot pointer and the prefetched rows across experts: no per-expert tstart load and no index ->
# gather chain after the loop barrier. Same experts, order and arithmetic per core: outputs bit-identical. Default 1 since
# R2-03d (optimization step R2-I: unit, real routing, 5.95 -> 5.65 ms per MoE layer at 4,096 tokens, random routing 6.83 ->
# 6.34; full-model 4,096-token chunk 283.9 -> 278.9 ms; gate details identical record for record; device-0 TTFT p50
# agent_8k c1 -1.9%, c32 -2.1%, long_32k c1 -1.7%, c16 -1.8% in 8 of 8 same-session pairs, TPOT unchanged). Shorter chunks
# keep the previous layout (the measured candidate also changed their layout: ref_1k_1k / reasoning c1 TTFT +0.3% / +0.9%).
# "0" = slot ranges by expert id everywhere.
# R2-03c5: the default follows NEMOTRON_H_MAMBA_PREFILL_INPROJ (same default, "fused", as mamba.py). With the in_proj
# kernel ("fused" or "nki") the carried slot layout makes neuronx-cc spill 2.7-3.6 GB more per core in the MoE blocks
# after a Mamba block in its repeated module (+29 ms per 4,096-token chunk, R2-03c4 device profiles), and turning the carry
# off there makes the chunk 20 ms faster than 7549710. So the default is "0" unless INPROJ=torch; an explicit
# NEMOTRON_H_MOE_PREFILL_XCARRY still wins.
_XCARRY_DEFAULT = "1" if os.environ.get("NEMOTRON_H_MAMBA_PREFILL_INPROJ", "fused") == "torch" else "0"
XCARRY = os.environ.get("NEMOTRON_H_MOE_PREFILL_XCARRY", _XCARRY_DEFAULT) == "1"
# R2-03d: load all groups' trip counts once at kernel start (one DMA) instead of one small DMA per expert group.
NPRE = os.environ.get("NEMOTRON_H_MOE_PREFILL_NPRE", "0") == "1"


def _xcarry_on(t: int) -> bool:
    """R2-03d: XCARRY applies to a chunk of ``t`` tokens (the kernel's TILE_PREFETCH conditions; the dispatch lays out the
    slots for the carry exactly when the BF16 kernel carries)."""
    return (XCARRY and TILE_PREFETCH and t >= TILE_PREFETCH_MIN_T and DEFER and TILE0 and t >= TILE0_MIN_T
            and not TILE0_LOOPED)


def _tiles_sl(t: int) -> int:
    """Straight-line tiles per expert group for a chunk of ``t`` tokens (TILE0 on)."""
    return TILES_SL if t >= TILES_SL_MIN_T else 1
# Debug (T12e): run the TILE0 first tile inside a one-trip dynamic loop (so it ends with the loop's all-engine
# barrier) for the first expert group only ("first") or for every group ("all"); the dispatch is unchanged.
TILE0_LOOPED = os.environ.get("NEMOTRON_H_MOE_PREFILL_TILE0_LOOPED", "")
# ACC + TILE0 only (T12e): defer each tile's read-modify-write to the start of the next tile of the same expert (or to
# after the expert's dynamic loop), so the all-engine barrier at the end of a loop iteration no longer waits for that
# iteration's RMW DMA. One kernel-level y / destination buffer carries the pending tile; the compiler orders the next
# write of that buffer after the pending RMW has read it. Within one expert every token appears at most once, so the
# RMW order per plane row is unchanged. Default 1 (T12e optimization step 16: unit 5.03 -> 4.29 ms per layer at T = 2048 and
# 8.72 -> 7.83 at T = 4096, bit-identical also with skewed and adversarial routing; 14-layer device profile MoE blocks
# -9..-14%; full-model 4,096-token chunk 390.3 -> 374.4 ms; full-model outputs identical to DEFER=0; gate passes).
DEFER = os.environ.get("NEMOTRON_H_MOE_PREFILL_DEFER", "1") == "1"
# Debug (T12e): with TILE0, pass the full (clamped) tile count to the kernel instead of ``ntile - 1`` and start the
# dynamic loop at 1 (the torch dispatch graph is then the one of TILE0=0 + CLAMP1=1).
TILE0_LB1 = os.environ.get("NEMOTRON_H_MOE_PREFILL_TILE0_LB1", "0") == "1"
# Debug (T12e): give every expert group at least one tile in the dispatch (as TILE0 does) also with TILE0=0.
CLAMP1 = os.environ.get("NEMOTRON_H_MOE_PREFILL_CLAMP1", "0") == "1"
if TILE0_LOOPED not in ("", "first", "all"):
    raise ValueError(f"NEMOTRON_H_MOE_PREFILL_TILE0_LOOPED must be '', first or all, got {TILE0_LOOPED!r}")
# ACC only: row tile of the expert weight loads (T14d). "128" (default): 128-row tiles plus one short last tile
# (464 = 3 x 128 + 80). "exact": the largest divisor of I = 464 that is at most 128 (116, 4 tiles). The software-DGE queue
# splits a DMA over the 16 DMA engines of a physical core by partition; with 116 rows, engines 14 and 15 carried 19 and
# 4.6 MB of a layer's weight loads and the others ~30 MB (T14d unit profile). Default 128 since T14d (optimization step 13:
# kernel window -1.5..-2.1%, 2,048-token chunk 219.0 -> 215.7 ms, gate passes, TTFT -1..-2%). Outputs are close but not
# bit-identical to "exact" (the down projection's PSUM accumulation runs over 3 x 128 + 80 instead of 4 x 116 rows).
WTILE = os.environ.get("NEMOTRON_H_MOE_PREFILL_WTILE", "128")
# ACC + DEFER only (R2-03, NEMOTRON_H_MOE_PREFILL_TILE_PREFETCH): prefetch the next tile's slot indices, routing weights and gathered x rows during the current
# tile. Each dynamic-loop iteration ends with an all-engine barrier, so a tile's index loads -> x gather -> x transposes
# chain started only after the previous tile had finished. With PREFETCH the loop body first transposes the kept x
# rows, then issues the next tile's loads into the kept buffers, so they run under this tile's matmuls; the first tile
# of every expert (TILE0, straight-line) loads its own rows and prefetches tile 1. The last prefetch of an expert reads
# the next slots (unused); the dispatch adds one pad tile at the end so it stays in bounds. Same data, same arithmetic:
# outputs should be bit-identical to TILE_PREFETCH=0. Default 1 since R2-03 (optimization step R2-C2: unit 7.26 -> 6.67 ms per
# layer at 4,096 tokens, bit-identical; full-model 4,096-token chunk 297.9 -> 287.7 ms with the Mamba prefill kernel; gate
# details identical; device-0 agent_8k TTFT -4.3% (c1) / -3.8% (c32) over 2 same-session pairs). "0" = previous kernel.
TILE_PREFETCH = os.environ.get("NEMOTRON_H_MOE_PREFILL_TILE_PREFETCH", "1") == "1"
# Chunks of at least this many tokens use TILE_PREFETCH (R2-03 unit: -8% per layer at 4,096 tokens, +8% at 2,048, where
# most experts have a single tile and the first tile's prefetch is not used).
TILE_PREFETCH_MIN_T = int(os.environ.get("NEMOTRON_H_MOE_PREFILL_TILE_PREFETCH_MIN_T", "4096"))
if WTILE not in ("exact", "128"):
    raise ValueError(f"NEMOTRON_H_MOE_PREFILL_WTILE must be exact or 128, got {WTILE!r}")


def _ndma(name: str) -> int:
    """R2-03b: "te" -> 0 (all 21 blocks on the Tensor Engine), "dma" -> all, "dmaN" -> the first N of the 128-column
    blocks by DMA transpose, the rest on the Tensor Engine."""
    v = os.environ.get(name, "te")
    if v == "te":
        return 0
    if v == "dma":
        return 1 << 30
    if v.startswith("dma") and v[3:].isdigit():
        return int(v[3:])
    raise ValueError(f"{name} must be te, dma or dmaN, got {v!r}")


# ACC kernel, BF16 only (R2-03b, NEMOTRON_H_MOE_PREFILL_UPT): the expert up weight (stored [out, in] = [I, H]) reaches
# SBUF transposed ([128 (h), nh, I]) by DMA transposes from HBM instead of a plain load + 84 Tensor-Engine transposes per
# expert. Same bits (a transpose moves values only). UPT_DGE: "auto" (compiler picks), "hw" (hardware DGE), "sw".
# UPT_FORM: "2d" (one [I, 128] -> [128, I] transpose per column block) or "3d" ([16, I/16, 128] -> [128, I/16, 16], the
# hardware-DGE shape rule; "2d" and "2dt" do not compile, NCC_INLA001 in SplitLargeDMAs::splitTranspose).
# Not kept (R2-03b): slower for every N (unit, T=4096: 6.82 -> 7.08 (dma2) .. 8.34 ms (all) per MoE layer). The kernel is
# DMA-bound and the transposes go onto the same queues; each loop barrier waits for them.
UPT_ND = _ndma("NEMOTRON_H_MOE_PREFILL_UPT")
UPT_DGE = os.environ.get("NEMOTRON_H_MOE_PREFILL_UPT_DGE", "auto")
UPT_FORM = os.environ.get("NEMOTRON_H_MOE_PREFILL_UPT_FORM", "2d")
# ACC kernel (R2-03b, NEMOTRON_H_MOE_PREFILL_XT): the gathered token tile x [B, H] is transposed to [128 (h), nh, B] by
# SBUF -> SBUF DMA transposes instead of Tensor-Engine transposes (first N blocks, "dmaN"; "dma" = all). XT_FORM "2d"
# (one per block) or "3d" (one [B, nh, 128] -> [128, nh, B] transpose per tile). XT_DGE as UPT_DGE.
# R2-03b, BF16 up stored [E * 128, nh * I] (unit experiment): load each partition's nh * I row as one 2-D access pattern.
UPT_LOAD2D = os.environ.get("NEMOTRON_H_MOE_PREFILL_UPT_LOAD2D", "0") == "1"
# Not kept (R2-03b): unit T=4096 7.31 (dma4) / 7.87 ms (all) vs 6.82 ms per MoE layer.
XT_ND = _ndma("NEMOTRON_H_MOE_PREFILL_XT")
XT_DGE = os.environ.get("NEMOTRON_H_MOE_PREFILL_XT_DGE", "auto")
XT_FORM = os.environ.get("NEMOTRON_H_MOE_PREFILL_XT_FORM", "2d")
_DGE_MODES = {"auto": nisa.dge_mode.unknown, "hw": nisa.dge_mode.hwdge, "sw": nisa.dge_mode.swdge}
for _n, _v in (("UPT_DGE", UPT_DGE), ("XT_DGE", XT_DGE)):
    if _v not in _DGE_MODES:
        raise ValueError(f"NEMOTRON_H_MOE_PREFILL_{_n} must be auto, hw or sw, got {_v!r}")
for _n, _v in (("UPT_FORM", UPT_FORM), ("XT_FORM", XT_FORM)):
    if _v not in ("2d", "2dt", "3d"):
        raise ValueError(f"NEMOTRON_H_MOE_PREFILL_{_n} must be 2d, 2dt or 3d, got {_v!r}")


@nki.jit
def moe_prefill_kernel(x: Tensor, up: Tensor, down: Tensor, ntile: Tensor, tstart: Tensor, slot_tok: Tensor,
                       slot_pair: Tensor, ws: Tensor, geom: Tensor):
    """Routed-expert outputs per slot.

    Args:
        x: ``[T, H]`` bf16 tokens. slot_tok / slot_pair: ``[NPAD, 1]`` int32 token row and output row
            (pair ``t * k + j``) of each slot; pad slots: token 0, output row ``T * k``, weight 0.
        up, down: ``[E * I, H]`` bf16 stacked expert weights.
        ntile: ``[E, 1]`` int32 tiles per expert. tstart: ``[E, 1]`` int32 first slot of each expert.
        ws: ``[T * k + 1, 1]`` float32 routing weight per pair (last row 0, for the pad slots).
        geom: dummy ``[I, PT]``; its shape gives I and the row tile PT (I / n_prgs / PT tiles per core).
    Returns:
        ``[n_prgs * (T * k + 1), H]`` bf16; row ``c * (T * k + 1) + p`` holds core ``c``'s partial of pair ``p``.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("moe_prefill", (0, 1), 2)
    npad = slot_tok.shape[0]
    h = x.shape[1]
    e_n = ntile.shape[0]
    rows, pt = geom.shape
    half = rows // n_prgs
    nj = half // pt
    nh = h // 128
    assert nh * 128 == h and nj * pt == half
    nrow = ws.shape[0]  # T * k pairs + 1 pad row
    out = nl.ndarray((n_prgs * nrow, h), dtype=x.dtype, buffer=nl.shared_hbm)
    n_chunks = (h + _PSUM_COLS - 1) // _PSUM_COLS
    unroll = EXPERT_UNROLL if EXPERT_UNROLL > 0 else e_n

    # static path: the first ``nd`` of the nh 128-column blocks of up are transposed by DMA (HBM -> SBUF),
    # the rest on the Tensor Engine (balances the TE and the DMA queues; WT_DMA = all blocks)
    nd = nh if WT_DMA else max(0, min(nh, WT_DMA_N))
    halfp = (half + 15) // 16 * 16 if nd else half  # DMA transpose outputs must start 32-byte aligned

    def _load(w_src, static_w=False):
        """DMA this core's half of one expert's up/down weights; ``w_src(t, j)`` is the source of row tile ``j``.
        Returns ``(up_sb, dn_sb, upT)``; ``upT`` holds the DMA-transposed blocks (static path, nd > 0) or is None."""
        ndl = nd if static_w else 0
        c_up = ndl * 128  # up columns still needed in natural layout (TE transposes)
        up_sb = nl.ndarray((pt, nj, h), dtype=up.dtype, buffer=nl.sbuf)
        dn_sb = nl.ndarray((pt, nj, h), dtype=down.dtype, buffer=nl.sbuf)
        for j in range(nj):
            if "wdma" in ABLATE:
                continue
            if static_w and not WEIGHT_HWDGE:  # compiler-generated descriptors, spread over the DMA queues
                # split each [pt, H] row tile into WSPLIT column chunks so that more DMA engines work at once
                cw = (h + WSPLIT - 1) // WSPLIT
                for c0 in range(0, h, cw):
                    c1 = min(h, c0 + cw)
                    if c1 > c_up:
                        u0 = max(c0, c_up)
                        nisa.dma_copy(dst=up_sb[0:pt, j, u0:c1], src=w_src(up, j)[0:pt, u0:c1])
                    nisa.dma_copy(dst=dn_sb[0:pt, j, c0:c1], src=w_src(down, j)[0:pt, c0:c1])
            else:
                nisa.dma_copy(dst=up_sb[0:pt, j, 0:h], src=w_src(up, j), dge_mode=_DGE, engine=nisa.engine.sync)
                nisa.dma_copy(dst=dn_sb[0:pt, j, 0:h], src=w_src(down, j), dge_mode=_DGE, engine=nisa.engine.scalar)
        upT = None
        if ndl:
            # up weight blocks transposed by the DMA engines straight from HBM: upT [128 (h), nh, halfp (i)]
            upT = nl.ndarray((128, nh, halfp), dtype=up.dtype, buffer=nl.sbuf)
            for hh in range(ndl):
                nisa.dma_transpose(dst=upT[0:128, hh, 0:half], src=w_src(up, None)[0:half, hh * 128 : (hh + 1) * 128])
        return up_sb, dn_sb, upT

    def _expert(w_src, n_src, off_src, static_w=False, bufs=None):
        """One expert. ``bufs``: weights already loaded by ``_load`` (prefetch), else loaded here.
        ``n_src`` / ``off_src`` are [1, 1] DMA sources of the tile count and the first slot."""
        up_sb, dn_sb, upT = bufs if bufs is not None else _load(w_src, static_w)
        h0 = nd if upT is not None else 0
        if upT is None:
            upT = nl.ndarray((128, nh, halfp), dtype=up.dtype, buffer=nl.sbuf)
        for hh in (range(h0, nh) if "wt" not in ABLATE else range(0)):
            tp = nl.ndarray((128, half), dtype=up.dtype, buffer=nl.psum)
            for j in range(nj):
                nisa.nc_transpose(dst=tp[0:128, j * pt : (j + 1) * pt],
                                  data=up_sb[0:pt, j, hh * 128 : (hh + 1) * 128])
            nisa.tensor_copy(dst=upT[0:128, hh, 0:half], src=tp[0:128, 0:half])

        n_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=n_sb, src=n_src, dge_mode=_DGE, engine=nisa.engine.sync)
        n_reg = nisa.register_alloc()
        nisa.register_load(n_reg, n_sb)
        off_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=off_sb, src=off_src, dge_mode=_DGE, engine=nisa.engine.scalar)

        def _tile(_i, off_sb=off_sb, up_sb=up_sb, dn_sb=dn_sb, upT=upT):
            tok_sb = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tok_sb, src=slot_tok.ap(pattern=[[1, B], [1, 1]], offset=0, scalar_offset=off_sb,
                                                      indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            pair_sb = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pair_sb, src=slot_pair.ap(pattern=[[1, B], [1, 1]], offset=0, scalar_offset=off_sb,
                                                        indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
            w_sb = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_sb, src=ws.ap(pattern=[[1, B], [1, 1]], offset=0, vector_offset=pair_sb, indirect_dim=0))
            # gather the tile's token rows (one row per partition)
            x_sb = nl.ndarray((B, h), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sb, src=x.ap(pattern=[[h, B], [1, h]], offset=0, vector_offset=tok_sb, indirect_dim=0))
            # xT [128 (h), nh, B]
            xT = nl.ndarray((128, nh, B), dtype=x.dtype, buffer=nl.sbuf)
            for g0 in range(0, nh, 4):
                g1 = min(nh, g0 + 4)
                tx = nl.ndarray((128, (g1 - g0) * B), dtype=x.dtype, buffer=nl.psum)
                for hh in range(g0, g1):
                    nisa.nc_transpose(dst=tx[0:128, (hh - g0) * B : (hh - g0 + 1) * B],
                                      data=x_sb[0:B, hh * 128 : (hh + 1) * 128])
                for hh in range(g0, g1):
                    nisa.tensor_copy(dst=xT[0:128, hh, 0:B], src=tx[0:128, (hh - g0) * B : (hh - g0 + 1) * B])
            aT = nl.ndarray((pt, nj, B), dtype=x.dtype, buffer=nl.sbuf)
            if UP_X_STATIONARY:
                # a [B, half] = relu(x @ up.T)^2: token tile stationary, all of this core's I as moving
                pu = nl.ndarray((B, half), dtype=nl.float32, buffer=nl.psum)
                for hh in range(nh):
                    nisa.nc_matmul(dst=pu, stationary=xT[0:128, hh, 0:B], moving=upT[0:128, hh, 0:half],
                                   accumulate=hh > 0)
                r = nl.ndarray((B, half), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=r, data=pu, op0=nl.maximum, operand0=0.0)
                a_sb = nl.ndarray((B, half), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a_sb, data1=r, data2=r, op=nl.multiply)
                for j in range(nj):
                    ta = nl.ndarray((pt, B), dtype=x.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=ta, data=a_sb[0:B, j * pt : (j + 1) * pt])
                    nisa.tensor_copy(dst=aT[0:pt, j, 0:B], src=ta)
            else:
                # aT [pt, nj, B] = relu(up @ xT)^2
                for j in range(nj):
                    pu = nl.ndarray((pt, B), dtype=nl.float32, buffer=nl.psum)
                    for hh in range(nh):
                        nisa.nc_matmul(dst=pu, stationary=upT[0:128, hh, j * pt : (j + 1) * pt],
                                       moving=xT[0:128, hh, 0:B], accumulate=hh > 0)
                    r = nl.ndarray((pt, B), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=r, data=pu, op0=nl.maximum, operand0=0.0)
                    nisa.tensor_tensor(dst=aT[0:pt, j, 0:B], data1=r, data2=r, op=nl.multiply)
            # y [B, H] = aT.T @ down, scaled by the slot's routing weight
            y_sb = nl.ndarray((B, h), dtype=x.dtype, buffer=nl.sbuf)
            for c in range(n_chunks):
                c0 = c * _PSUM_COLS
                csz = min(_PSUM_COLS, h - c0)
                py = nl.ndarray((B, csz), dtype=nl.float32, buffer=nl.psum)
                for j in range(nj):
                    nisa.nc_matmul(dst=py, stationary=aT[0:pt, j, 0:B], moving=dn_sb[0:pt, j, c0 : c0 + csz],
                                   accumulate=j > 0)
                nisa.tensor_scalar(dst=y_sb[0:B, c0 : c0 + csz], data=py, op0=nl.multiply, operand0=w_sb[0:B, 0:1])
            # scatter the rows to their (token, k) pair rows
            nisa.dma_copy(dst=out.ap(pattern=[[h, B], [1, h]], offset=prg_id * nrow * h, vector_offset=pair_sb,
                                     indirect_dim=0), src=y_sb)
            nisa.tensor_scalar(dst=off_sb, data=off_sb, op0=nl.add, operand0=B)

        if "tiles" not in ABLATE:
            nl.fori_loop(0, n_reg, _tile)

    if unroll >= e_n:
        def _src(e):
            r0 = e * rows + prg_id * half
            # j=None: this core's whole half [half, H]
            return lambda t, j: t[r0 : r0 + half, 0:h] if j is None else t[r0 + j * pt : r0 + (j + 1) * pt, 0:h]

        nxt = _load(_src(0), static_w=True) if PREFETCH else None
        for e in range(e_n):
            cur = nxt
            # software pipelining: issue the next expert's weight DMAs before this expert's dynamic tile
            # loop (instructions after a fori_loop cannot start before it ends on each engine)
            if PREFETCH and e + 1 < e_n:
                nxt = _load(_src(e + 1), static_w=True)
            _expert(_src(e), ntile[e : e + 1, 0:1], tstart[e : e + 1, 0:1], static_w=True, bufs=cur)
    else:
        # dynamic expert loop, ``unroll`` experts per iteration (body emitted once: short compile)
        assert e_n % unroll == 0
        e_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)  # first expert of the group
        nisa.memset(dst=e_sb, value=0)
        wrow_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)  # its weight row (this core's half)
        nisa.memset(dst=wrow_sb, value=prg_id * half)

        def _group(_g):
            for u in range(unroll):
                _expert(lambda t, j, u=u: t.ap(pattern=[[h, pt], [1, h]], offset=(u * rows + j * pt) * h,
                                               scalar_offset=wrow_sb, indirect_dim=0),
                        ntile.ap(pattern=[[1, 1], [1, 1]], offset=u, scalar_offset=e_sb, indirect_dim=0),
                        tstart.ap(pattern=[[1, 1], [1, 1]], offset=u, scalar_offset=e_sb, indirect_dim=0))
            nisa.tensor_scalar(dst=e_sb, data=e_sb, op0=nl.add, operand0=unroll)
            nisa.tensor_scalar(dst=wrow_sb, data=wrow_sb, op0=nl.add, operand0=unroll * rows)

        nl.fori_loop(0, e_n // unroll, _group)
    return out


@nki.jit
def moe_prefill_kernel_esplit(x: Tensor, up: Tensor, down: Tensor, ntile: Tensor, tstart: Tensor, slot_tok: Tensor,
                              slot_pair: Tensor, ws: Tensor, geom: Tensor):
    """Expert split (``NEMOTRON_H_MOE_PREFILL_SPLIT=e``): core ``c`` computes experts ``e = g * n_prgs + c`` with
    all of I, so a token tile is gathered, transposed and multiplied by one core only (the I split does that
    work on both cores). LNC=2 dynamic loops must have the same trip count on both cores (NCC_IXGM002), so
    ``ntile[g]`` is the larger tile count of the group's experts and the dispatch pads both experts' slot ranges
    to it (pad slots: token 0, weight 0, output row ``T * k``).

    Args as ``moe_prefill_kernel`` except ``ntile``: ``[E / n_prgs, 1]`` int32 tiles per expert group.
    Returns ``[T * k + 1, H]`` bf16 (one row per (token, k) pair; row ``T * k`` is scratch).
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("moe_prefill_e", (0, 1), 2)
    h = x.shape[1]
    e_n = tstart.shape[0]
    g_n = ntile.shape[0]
    assert g_n * n_prgs == e_n, "dispatch group size must equal the number of programs"
    rows, pt = geom.shape
    nj = rows // pt
    nh = h // 128
    assert nh * 128 == h and nj * pt == rows and rows <= _PSUM_COLS
    nrow = ws.shape[0]
    out = nl.ndarray((nrow, h), dtype=x.dtype, buffer=nl.shared_hbm)
    n_chunks = (h + _PSUM_COLS - 1) // _PSUM_COLS
    nd = max(0, min(nh, WT_DMA_N))
    rowsp = (rows + 15) // 16 * 16 if nd else rows

    def _load(r0):
        up_sb = nl.ndarray((pt, nj, h), dtype=up.dtype, buffer=nl.sbuf)
        dn_sb = nl.ndarray((pt, nj, h), dtype=down.dtype, buffer=nl.sbuf)
        c_up = nd * 128
        for j in range(nj):
            if "wdma" in ABLATE:
                continue
            if c_up < h:
                nisa.dma_copy(dst=up_sb[0:pt, j, c_up:h], src=up[r0 + j * pt : r0 + (j + 1) * pt, c_up:h])
            nisa.dma_copy(dst=dn_sb[0:pt, j, 0:h], src=down[r0 + j * pt : r0 + (j + 1) * pt, 0:h])
        upT = nl.ndarray((128, nh, rowsp), dtype=up.dtype, buffer=nl.sbuf)
        for hh in range(nd):
            nisa.dma_transpose(dst=upT[0:128, hh, 0:rows], src=up[r0 : r0 + rows, hh * 128 : (hh + 1) * 128])
        return up_sb, dn_sb, upT

    def _expert(g, bufs):
        up_sb, dn_sb, upT = bufs
        for hh in (range(nd, nh) if "wt" not in ABLATE else range(0)):
            tp = nl.ndarray((128, rows), dtype=up.dtype, buffer=nl.psum)
            for j in range(nj):
                nisa.nc_transpose(dst=tp[0:128, j * pt : (j + 1) * pt], data=up_sb[0:pt, j, hh * 128 : (hh + 1) * 128])
            nisa.tensor_copy(dst=upT[0:128, hh, 0:rows], src=tp[0:128, 0:rows])
        e = g * n_prgs + prg_id
        n_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=n_sb, src=ntile[g : g + 1, 0:1], dge_mode=_DGE, engine=nisa.engine.sync)
        n_reg = nisa.register_alloc()
        nisa.register_load(n_reg, n_sb)
        off_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=off_sb, src=tstart[e : e + 1, 0:1], dge_mode=_DGE, engine=nisa.engine.scalar)

        def _tile(_i, off_sb=off_sb, dn_sb=dn_sb, upT=upT):
            tok_sb = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tok_sb, src=slot_tok.ap(pattern=[[1, B], [1, 1]], offset=0, scalar_offset=off_sb,
                                                      indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            pair_sb = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pair_sb, src=slot_pair.ap(pattern=[[1, B], [1, 1]], offset=0, scalar_offset=off_sb,
                                                        indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
            w_sb = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_sb, src=ws.ap(pattern=[[1, B], [1, 1]], offset=0, vector_offset=pair_sb, indirect_dim=0))
            x_sb = nl.ndarray((B, h), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sb, src=x.ap(pattern=[[h, B], [1, h]], offset=0, vector_offset=tok_sb, indirect_dim=0))
            xT = nl.ndarray((128, nh, B), dtype=x.dtype, buffer=nl.sbuf)
            for g0 in range(0, nh, 4):
                g1 = min(nh, g0 + 4)
                tx = nl.ndarray((128, (g1 - g0) * B), dtype=x.dtype, buffer=nl.psum)
                for hh in range(g0, g1):
                    nisa.nc_transpose(dst=tx[0:128, (hh - g0) * B : (hh - g0 + 1) * B],
                                      data=x_sb[0:B, hh * 128 : (hh + 1) * 128])
                for hh in range(g0, g1):
                    nisa.tensor_copy(dst=xT[0:128, hh, 0:B], src=tx[0:128, (hh - g0) * B : (hh - g0 + 1) * B])
            # a [B, rows] = relu(x @ up.T)^2: token tile stationary, the expert's whole I as moving
            pu = nl.ndarray((B, rows), dtype=nl.float32, buffer=nl.psum)
            for hh in range(nh):
                nisa.nc_matmul(dst=pu, stationary=xT[0:128, hh, 0:B], moving=upT[0:128, hh, 0:rows], accumulate=hh > 0)
            r = nl.ndarray((B, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=r, data=pu, op0=nl.maximum, operand0=0.0)
            a_sb = nl.ndarray((B, rows), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=a_sb, data1=r, data2=r, op=nl.multiply)
            aT = nl.ndarray((pt, nj, B), dtype=x.dtype, buffer=nl.sbuf)
            for j in range(nj):
                ta = nl.ndarray((pt, B), dtype=x.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=ta, data=a_sb[0:B, j * pt : (j + 1) * pt])
                nisa.tensor_copy(dst=aT[0:pt, j, 0:B], src=ta)
            y_sb = nl.ndarray((B, h), dtype=x.dtype, buffer=nl.sbuf)
            for c in range(n_chunks):
                c0 = c * _PSUM_COLS
                csz = min(_PSUM_COLS, h - c0)
                py = nl.ndarray((B, csz), dtype=nl.float32, buffer=nl.psum)
                for j in range(nj):
                    nisa.nc_matmul(dst=py, stationary=aT[0:pt, j, 0:B], moving=dn_sb[0:pt, j, c0 : c0 + csz],
                                   accumulate=j > 0)
                nisa.tensor_scalar(dst=y_sb[0:B, c0 : c0 + csz], data=py, op0=nl.multiply, operand0=w_sb[0:B, 0:1])
            nisa.dma_copy(dst=out.ap(pattern=[[h, B], [1, h]], offset=0, vector_offset=pair_sb, indirect_dim=0),
                          src=y_sb)
            nisa.tensor_scalar(dst=off_sb, data=off_sb, op0=nl.add, operand0=B)

        if "tiles" not in ABLATE:
            nl.fori_loop(0, n_reg, _tile)

    # static group loop; the next group's weight DMAs are issued before this group's dynamic tile loop
    nxt = _load(prg_id * rows)
    for g in range(g_n):
        cur = nxt
        if g + 1 < g_n:
            nxt = _load(((g + 1) * n_prgs + prg_id) * rows)
        _expert(g, cur)
    return out


@nki.jit
def moe_prefill_kernel_esplit_sorted(x: Tensor, up: Tensor, down: Tensor, ntile: Tensor, tstart: Tensor,
                                     slot_tok: Tensor, slot_pair: Tensor, ws: Tensor, geom: Tensor, erow: Tensor):
    """``NEMOTRON_H_MOE_PREFILL_PAIR=sorted`` (T12b): as ``moe_prefill_kernel_esplit``, but slot ``s = g * n_prgs + c``
    computes the expert whose first weight row is ``erow[s]`` (runtime offset), and ``tstart`` / ``ntile`` are in slot
    order. The dispatch fills the slots with the experts sorted by tile count.

    Expert split (``NEMOTRON_H_MOE_PREFILL_SPLIT=e``): core ``c`` computes experts ``e = g * n_prgs + c`` with
    all of I, so a token tile is gathered, transposed and multiplied by one core only (the I split does that
    work on both cores). LNC=2 dynamic loops must have the same trip count on both cores (NCC_IXGM002), so
    ``ntile[g]`` is the larger tile count of the group's experts and the dispatch pads both experts' slot ranges
    to it (pad slots: token 0, weight 0, output row ``T * k``).

    Args as ``moe_prefill_kernel`` except ``ntile``: ``[E / n_prgs, 1]`` int32 tiles per expert group.
    Returns ``[T * k + 1, H]`` bf16 (one row per (token, k) pair; row ``T * k`` is scratch).
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("moe_prefill_e", (0, 1), 2)
    h = x.shape[1]
    e_n = tstart.shape[0]
    g_n = ntile.shape[0]
    assert g_n * n_prgs == e_n, "dispatch group size must equal the number of programs"
    rows, pt = geom.shape
    nj = rows // pt
    nh = h // 128
    assert nh * 128 == h and nj * pt == rows and rows <= _PSUM_COLS
    nrow = ws.shape[0]
    out = nl.ndarray((nrow, h), dtype=x.dtype, buffer=nl.shared_hbm)
    n_chunks = (h + _PSUM_COLS - 1) // _PSUM_COLS
    nd = 0  # no DMA weight transposes in this variant
    rowsp = (rows + 15) // 16 * 16 if nd else rows

    er_sb = nl.ndarray((e_n, 1), dtype=erow.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=er_sb, src=erow)
    qn = [0]

    def _load(s_idx):
        up_sb = None if eup is not None else nl.ndarray((pt, nj, h), dtype=up.dtype, buffer=nl.sbuf)
        dn_sb = nl.ndarray((pt, nj, h), dtype=down.dtype, buffer=nl.sbuf)
        for j in range(nj):
            for src, dst in (((down, dn_sb),) if eup is not None else ((up, up_sb), (down, dn_sb))):
                if PAIR_DGE == "sw":
                    kw = dict(dge_mode=nisa.dge_mode.swdge)
                else:
                    kw = dict(dge_mode=nisa.dge_mode.hwdge,
                              engine=nisa.engine.sync if qn[0] % 2 == 0 else nisa.engine.scalar)
                    qn[0] += 1
                nisa.dma_copy(dst=dst[0:pt, j, 0:h],
                              src=src.ap(pattern=[[h, pt], [1, h]], offset=j * pt * h,
                                         scalar_offset=er_sb[s_idx : s_idx + 1, 0:1], indirect_dim=0),
                              **kw)
        upT = nl.ndarray((128, nh, rowsp), dtype=up.dtype, buffer=nl.sbuf)
        return up_sb, dn_sb, upT

    def _expert(g, bufs):
        up_sb, dn_sb, upT = bufs
        for hh in (range(nd, nh) if "wt" not in ABLATE else range(0)):
            tp = nl.ndarray((128, rows), dtype=up.dtype, buffer=nl.psum)
            for j in range(nj):
                nisa.nc_transpose(dst=tp[0:128, j * pt : (j + 1) * pt], data=up_sb[0:pt, j, hh * 128 : (hh + 1) * 128])
            nisa.tensor_copy(dst=upT[0:128, hh, 0:rows], src=tp[0:128, 0:rows])
        e = g * n_prgs + prg_id
        n_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=n_sb, src=ntile[g : g + 1, 0:1], dge_mode=_DGE, engine=nisa.engine.sync)
        n_reg = nisa.register_alloc()
        nisa.register_load(n_reg, n_sb)
        off_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=off_sb, src=tstart[e : e + 1, 0:1], dge_mode=_DGE, engine=nisa.engine.scalar)

        def _tile(_i, off_sb=off_sb, dn_sb=dn_sb, upT=upT):
            tok_sb = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tok_sb, src=slot_tok.ap(pattern=[[1, B], [1, 1]], offset=0, scalar_offset=off_sb,
                                                      indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            pair_sb = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pair_sb, src=slot_pair.ap(pattern=[[1, B], [1, 1]], offset=0, scalar_offset=off_sb,
                                                        indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
            w_sb = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_sb, src=ws.ap(pattern=[[1, B], [1, 1]], offset=0, vector_offset=pair_sb, indirect_dim=0))
            x_sb = nl.ndarray((B, h), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sb, src=x.ap(pattern=[[h, B], [1, h]], offset=0, vector_offset=tok_sb, indirect_dim=0))
            xT = nl.ndarray((128, nh, B), dtype=x.dtype, buffer=nl.sbuf)
            for g0 in range(0, nh, 4):
                g1 = min(nh, g0 + 4)
                tx = nl.ndarray((128, (g1 - g0) * B), dtype=x.dtype, buffer=nl.psum)
                for hh in range(g0, g1):
                    nisa.nc_transpose(dst=tx[0:128, (hh - g0) * B : (hh - g0 + 1) * B],
                                      data=x_sb[0:B, hh * 128 : (hh + 1) * 128])
                for hh in range(g0, g1):
                    nisa.tensor_copy(dst=xT[0:128, hh, 0:B], src=tx[0:128, (hh - g0) * B : (hh - g0 + 1) * B])
            # a [B, rows] = relu(x @ up.T)^2: token tile stationary, the expert's whole I as moving
            pu = nl.ndarray((B, rows), dtype=nl.float32, buffer=nl.psum)
            for hh in range(nh):
                nisa.nc_matmul(dst=pu, stationary=xT[0:128, hh, 0:B], moving=upT[0:128, hh, 0:rows], accumulate=hh > 0)
            r = nl.ndarray((B, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=r, data=pu, op0=nl.maximum, operand0=0.0)
            a_sb = nl.ndarray((B, rows), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=a_sb, data1=r, data2=r, op=nl.multiply)
            aT = nl.ndarray((pt, nj, B), dtype=x.dtype, buffer=nl.sbuf)
            for j in range(nj):
                ta = nl.ndarray((pt, B), dtype=x.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=ta, data=a_sb[0:B, j * pt : (j + 1) * pt])
                nisa.tensor_copy(dst=aT[0:pt, j, 0:B], src=ta)
            y_sb = nl.ndarray((B, h), dtype=x.dtype, buffer=nl.sbuf)
            for c in range(n_chunks):
                c0 = c * _PSUM_COLS
                csz = min(_PSUM_COLS, h - c0)
                py = nl.ndarray((B, csz), dtype=nl.float32, buffer=nl.psum)
                for j in range(nj):
                    nisa.nc_matmul(dst=py, stationary=aT[0:pt, j, 0:B], moving=dn_sb[0:pt, j, c0 : c0 + csz],
                                   accumulate=j > 0)
                nisa.tensor_scalar(dst=y_sb[0:B, c0 : c0 + csz], data=py, op0=nl.multiply, operand0=w_sb[0:B, 0:1])
            nisa.dma_copy(dst=out.ap(pattern=[[h, B], [1, h]], offset=0, vector_offset=pair_sb, indirect_dim=0),
                          src=y_sb)
            nisa.tensor_scalar(dst=off_sb, data=off_sb, op0=nl.add, operand0=B)

        if "tiles" not in ABLATE:
            nl.fori_loop(0, n_reg, _tile)

    # static group loop; the next group's weight DMAs are issued before this group's dynamic tile loop
    nxt = _load(prg_id)
    for g in range(g_n):
        cur = nxt
        if g + 1 < g_n:
            nxt = _load((g + 1) * n_prgs + prg_id)
        _expert(g, cur)
    return out


@nki.jit
def moe_prefill_kernel_esplit_acc(x: Tensor, up: Tensor, down: Tensor, ntile: Tensor, tstart: Tensor,
                                  slot_tok: Tensor, slot_pair: Tensor, ws: Tensor, geom: Tensor, erow: Tensor,
                                  slot_dst: Tensor):
    """BF16 weights (see ``_esplit_acc_body``)."""
    return _esplit_acc_body(x, up, down, ntile, tstart, slot_tok, slot_pair, ws, geom, erow, slot_dst, None)


@nki.jit
def moe_prefill_kernel_esplit_acc_fp8(x: Tensor, up: Tensor, down: Tensor, ntile: Tensor, tstart: Tensor,
                                      slot_tok: Tensor, slot_pair: Tensor, ws: Tensor, geom: Tensor, erow: Tensor,
                                      slot_dst: Tensor, c: Tensor, eup: Tensor):
    """FP8 weights (``NEMOTRON_H_FP8=1``, R2-05): ``up``/``down`` float8_e4m3; ``c`` ``[E * I + 128, 1]`` float32
    per-intermediate-row scale ``s_up^2 * s_down``, applied to the transposed activation ``aT`` (I on partitions)
    when it is copied out of PSUM. ``up`` is stored transposed, ``[E * 128, nh * I]`` (``moe_fp8.up_t_layout``), and
    ``eup`` ``[E, 1]`` int32 holds ``e * 128`` in slot order, so each expert's ``upT`` is one DMA (no Tensor-Engine
    transposes). The up and down projections read the FP8 tiles directly (bf16 x fp8 matmuls)."""
    return _esplit_acc_body(x, up, down, ntile, tstart, slot_tok, slot_pair, ws, geom, erow, slot_dst, cscale=c,
                            eup=eup)


@nki.jit
def moe_prefill_kernel_esplit_acc_upt(x: Tensor, up: Tensor, down: Tensor, ntile: Tensor, tstart: Tensor,
                                      slot_tok: Tensor, slot_pair: Tensor, ws: Tensor, geom: Tensor, erow: Tensor,
                                      slot_dst: Tensor, eup: Tensor):
    """R2-03b: BF16 weights with ``up`` stored transposed, ``[E * 128, nh * I]`` (``moe_fp8.up_t_layout``, any dtype);
    ``eup`` ``[E, 1]`` int32 holds ``e * 128`` in slot order, so each expert's ``upT`` is one DMA and the Tensor Engine
    does no up-weight transposes (the FP8 kernel's up path with bf16 tiles). Unit experiment only (R2-03b: the upper bound
    of a stored-layout change; real routing 6.12 -> 6.10 ms at T=4096 and 4.31 -> 4.61 ms at T=2048 per MoE layer, so the
    layout change was not built; a second up copy does not fit in HBM)."""
    return _esplit_acc_body(x, up, down, ntile, tstart, slot_tok, slot_pair, ws, geom, erow, slot_dst, None, eup=eup)


def _esplit_acc_body(x, up, down, ntile, tstart, slot_tok, slot_pair, ws, geom, erow, slot_dst, cscale, eup=None):
    """``NEMOTRON_H_MOE_PREFILL_ACC=1`` (T12b): as ``moe_prefill_kernel_esplit_sorted``, but each tile's rows are added
    into ``out[slot_dst[s]]`` (row ``c * T + token`` of core ``c``'s float32 plane; pad slots carry the out-of-bounds
    row ``2 T`` and are skipped). Returns ``[n_prgs * T, H]`` float32.

    ``NEMOTRON_H_MOE_PREFILL_PAIR=sorted`` (T12b): as ``moe_prefill_kernel_esplit``, but slot ``s = g * n_prgs + c``
    computes the expert whose first weight row is ``erow[s]`` (runtime offset), and ``tstart`` / ``ntile`` are in slot
    order. The dispatch fills the slots with the experts sorted by tile count.

    Expert split (``NEMOTRON_H_MOE_PREFILL_SPLIT=e``): core ``c`` computes experts ``e = g * n_prgs + c`` with
    all of I, so a token tile is gathered, transposed and multiplied by one core only (the I split does that
    work on both cores). LNC=2 dynamic loops must have the same trip count on both cores (NCC_IXGM002), so
    ``ntile[g]`` is the larger tile count of the group's experts and the dispatch pads both experts' slot ranges
    to it (pad slots: token 0, weight 0, output row ``T * k``).

    Args as ``moe_prefill_kernel`` except ``ntile``: ``[E / n_prgs, 1]`` int32 tiles per expert group.
    Returns ``[T * k + 1, H]`` bf16 (one row per (token, k) pair; row ``T * k`` is scratch).
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("moe_prefill_e", (0, 1), 2)
    h = x.shape[1]
    e_n = tstart.shape[0]
    g_n = ntile.shape[0]
    assert g_n * n_prgs == e_n, "dispatch group size must equal the number of programs"
    rows, pt = geom.shape
    nj = (rows + pt - 1) // pt
    pts = [min(pt, rows - j * pt) for j in range(nj)]  # row count of tile j (the last one may be short)
    nh = h // 128
    assert nh * 128 == h and pt <= 128 and rows <= _PSUM_COLS
    nrow = ws.shape[0]
    t_n = x.shape[0]
    out = nl.ndarray((n_prgs * t_n, h), dtype=nl.float32, buffer=nl.shared_hbm)
    nd2 = slot_dst.shape[1]  # R2-03d FIRSTW: 2 columns [first-touch write row, RMW row]
    if nd2 == 1:  # zero this core's plane before any read-modify-write
        z_sb = nl.ndarray((128, h), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=z_sb, value=0.0)
        for r0 in range(0, t_n, 128):
            rn = min(128, t_n - r0)
            nisa.dma_copy(dst=out[prg_id * t_n + r0 : prg_id * t_n + r0 + rn, 0:h], src=z_sb[0:rn, 0:h])
    n_chunks = (h + _PSUM_COLS - 1) // _PSUM_COLS
    nd = 0 if eup is not None else min(nh, UPT_ND)  # R2-03b: up column blocks transposed by DMA (BF16 only)
    if nd and UPT_FORM == "3d":
        assert rows % 16 == 0, "UPT_FORM=3d needs I % 16 == 0"
    xnd = min(nh, XT_ND)  # R2-03b: x column blocks transposed by SBUF -> SBUF DMA
    rowsp = rows
    up_t_dt = up.dtype  # FP8: the transpose keeps float8 (PE transpose dst must match on gen3+); upT is the fp8 moving operand

    er_sb = nl.ndarray((e_n, 1), dtype=erow.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=er_sb, src=erow)
    if eup is not None:
        eu_sb = nl.ndarray((e_n, 1), dtype=eup.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=eu_sb, src=eup)
    qn = [0]
    tile0_on = TILE0 and t_n >= TILE0_MIN_T
    defer = DEFER and tile0_on and not TILE0_LOOPED
    lb = 1 if (tile0_on and TILE0_LB1) else 0  # first dynamic-loop index
    nsl = _tiles_sl(t_n) if (tile0_on and defer and not TILE0_LB1) else 1
    pskip = PADSKIP and cscale is None  # R2-03b: BF16 only (the FP8 graphs stay unchanged)
    if defer:  # the pending tile of DEFER (y rows and their plane rows)
        y_keep = nl.ndarray((B, h), dtype=x.dtype, buffer=nl.sbuf)
        d_keep = nl.ndarray((B, nd2), dtype=nl.int32, buffer=nl.sbuf)
    tpref = TILE_PREFETCH and defer and t_n >= TILE_PREFETCH_MIN_T
    carry = _xcarry_on(t_n) and tpref and cscale is None and eup is None  # R2-03d: slot pointer and rows across experts
    if NPRE:
        nt_all = nl.ndarray((1, g_n), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=nt_all, src=ntile.ap(pattern=[[g_n, 1], [1, g_n]], offset=0), dge_mode=_DGE,
                      engine=nisa.engine.sync)
    if carry:
        off_run = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=off_run, src=tstart[prg_id : prg_id + 1, 0:1], dge_mode=_DGE, engine=nisa.engine.scalar)
    if tpref:  # R2-03: the next tile's plane rows, routing weights and x rows
        k_dst = nl.ndarray((B, nd2), dtype=nl.int32, buffer=nl.sbuf)
        k_w = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
        k_x = nl.ndarray((B, h), dtype=x.dtype, buffer=nl.sbuf)

    def _rmw(y_sb, dst_sb):
        if nd2 == 2:  # R2-03d FIRSTW: first touches are plain (bf16 -> fp32) scatters, the rest RMW
            w_ap = out.ap(pattern=[[h, B], [1, h]], offset=0, vector_offset=dst_sb[0:B, 0:1], indirect_dim=0)
            nisa.dma_copy(dst=w_ap, src=y_sb, oob_mode=nisa.oob_mode.skip, dge_mode=nisa.dge_mode.swdge)
            o_ap = out.ap(pattern=[[h, B], [1, h]], offset=0, vector_offset=dst_sb[0:B, 1:2], indirect_dim=0)
        else:
            o_ap = out.ap(pattern=[[h, B], [1, h]], offset=0, vector_offset=dst_sb, indirect_dim=0)
        nisa.dma_compute(dst=o_ap, srcs=[o_ap, y_sb], reduce_op=nl.add, oob_mode=nisa.oob_mode.skip)

    def _load(s_idx):
        c_up = nd * 128  # R2-03b: up columns [0, c_up) arrive transposed by DMA, the rest is loaded as stored
        up_sb = None if (eup is not None or c_up >= h) else nl.ndarray((pt, nj, h), dtype=up.dtype, buffer=nl.sbuf)
        dn_sb = nl.ndarray((pt, nj, h), dtype=down.dtype, buffer=nl.sbuf)
        for j in range(nj):
            for src, dst in (((down, dn_sb),) if up_sb is None else ((up, up_sb), (down, dn_sb))):
                if PAIR_DGE == "sw":
                    kw = dict(dge_mode=nisa.dge_mode.swdge)
                else:
                    kw = dict(dge_mode=nisa.dge_mode.hwdge,
                              engine=nisa.engine.sync if qn[0] % 2 == 0 else nisa.engine.scalar)
                    qn[0] += 1
                c0 = c_up if src is up else 0
                nisa.dma_copy(dst=dst[0:pts[j], j, c0:h],
                              src=src.ap(pattern=[[h, pts[j]], [1, h - c0]], offset=j * pt * h + c0,
                                         scalar_offset=er_sb[s_idx : s_idx + 1, 0:1], indirect_dim=0),
                              **kw)
        upT = nl.ndarray((128, nh, rowsp), dtype=up_t_dt, buffer=nl.sbuf)
        if nd:  # R2-03b: up column blocks hh < nd transposed by the DMA engines on the way from HBM
            ukw = dict(dge_mode=_DGE_MODES[UPT_DGE])
            for hh in range(nd):
                if UPT_FORM == "2d":
                    nisa.dma_transpose(dst=upT[0:128, hh, 0:rows],
                                       src=up.ap(pattern=[[h, rows], [1, 128]], offset=hh * 128,
                                                 scalar_offset=er_sb[s_idx : s_idx + 1, 0:1], indirect_dim=0),
                                       axes=(1, 0), **ukw)
                elif UPT_FORM == "2dt":  # one [pts, 128] -> [128, pts] transpose per row tile
                    for j in range(nj):
                        nisa.dma_transpose(dst=upT[0:128, hh, j * pt : j * pt + pts[j]],
                                           src=up.ap(pattern=[[h, pts[j]], [1, 128]], offset=j * pt * h + hh * 128,
                                                     scalar_offset=er_sb[s_idx : s_idx + 1, 0:1], indirect_dim=0),
                                           axes=(1, 0), **ukw)
                else:  # [16, I/16, 128] -> [128, I/16, 16]: dst[c, b, a] = up[erow + 16 b + a, 128 hh + c]
                    up4 = upT.reshape((128, nh, rows // 16, 16))
                    nisa.dma_transpose(dst=up4[0:128, hh, 0 : rows // 16, 0:16],
                                       src=up.ap(pattern=[[h, 16], [16 * h, rows // 16], [1, 128]], offset=hh * 128,
                                                 scalar_offset=er_sb[s_idx : s_idx + 1, 0:1], indirect_dim=0),
                                       axes=(2, 1, 0), **ukw)
        if eup is not None and cscale is None and UPT_LOAD2D:  # R2-03b: one contiguous nh * I row per partition
            nisa.dma_copy(dst=upT.reshape((128, nh * rowsp))[0:128, 0 : nh * rows],
                          src=up.ap(pattern=[[nh * rows, 128], [1, nh * rows]], offset=0,
                                    scalar_offset=eu_sb[s_idx : s_idx + 1, 0:1], indirect_dim=0),
                          dge_mode=nisa.dge_mode.swdge)
        elif eup is not None:  # FP8: the transposed up tiles straight from HBM
            nisa.dma_copy(dst=upT[0:128, 0:nh, 0:rows],
                          src=up.ap(pattern=[[nh * rows, 128], [rows, nh], [1, rows]], offset=0,
                                    scalar_offset=eu_sb[s_idx : s_idx + 1, 0:1], indirect_dim=0),
                          dge_mode=nisa.dge_mode.swdge)
        c_sb = None
        if cscale is not None:  # FP8 scales of this expert's rows: c_sb[p, j] = c[erow + j * pt + p]
            c_sb = nl.ndarray((pt, nj), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=c_sb, src=cscale.ap(pattern=[[1, pt], [pt, nj]], offset=0,
                                             scalar_offset=er_sb[s_idx : s_idx + 1, 0:1], indirect_dim=0),
                          dge_mode=nisa.dge_mode.swdge)
        return up_sb, dn_sb, upT, c_sb

    def _expert(g, bufs):
        up_sb, dn_sb, upT, c_sb = bufs
        for hh in (range(nd, nh) if ("wt" not in ABLATE and up_sb is not None) else range(0)):
            tp = nl.ndarray((128, rows), dtype=up_t_dt, buffer=nl.psum)
            for j in range(nj):
                nisa.nc_transpose(dst=tp[0:128, j * pt : j * pt + pts[j]],
                                  data=up_sb[0:pts[j], j, hh * 128 : (hh + 1) * 128])
            nisa.tensor_copy(dst=upT[0:128, hh, 0:rows], src=tp[0:128, 0:rows])
        e = g * n_prgs + prg_id
        if NPRE:
            n_sb = nt_all[0:1, g : g + 1]
        else:
            n_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=n_sb, src=ntile[g : g + 1, 0:1], dge_mode=_DGE, engine=nisa.engine.sync)
        n_reg = nisa.register_alloc()
        nisa.register_load(n_reg, n_sb)
        if carry:
            off_sb = off_run
        else:
            off_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=off_sb, src=tstart[e : e + 1, 0:1], dge_mode=_DGE, engine=nisa.engine.scalar)

        def _fetch(dst_sb, w_sb, x_sb):
            """The slot indices, plane rows, routing weights and x rows of the tile at slot offset ``off_sb``."""
            tok_sb = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tok_sb, src=slot_tok.ap(pattern=[[1, B], [1, 1]], offset=0, scalar_offset=off_sb,
                                                      indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            nisa.dma_copy(dst=dst_sb, src=slot_dst.ap(pattern=[[nd2, B], [1, nd2]], offset=0, scalar_offset=off_sb,
                                                       indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            if SCATTER1:  # slot_dst holds the token (pads: 2 T); add this core's plane offset
                nisa.tensor_scalar(dst=dst_sb, data=dst_sb, op0=nl.add, operand0=prg_id * t_n)
            pair_sb = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pair_sb, src=slot_pair.ap(pattern=[[1, B], [1, 1]], offset=0, scalar_offset=off_sb,
                                                        indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
            nisa.dma_copy(dst=w_sb, src=ws.ap(pattern=[[1, B], [1, 1]], offset=0, vector_offset=pair_sb, indirect_dim=0))
            nisa.dma_copy(dst=x_sb, src=x.ap(pattern=[[h, B], [1, h]], offset=0, vector_offset=tok_sb, indirect_dim=0),
                          **(dict(oob_mode=nisa.oob_mode.skip) if pskip else {}))

        def _tile(_i, off_sb=off_sb, dn_sb=dn_sb, upT=upT, c_sb=c_sb, pend=False, kept=False):
            if pend:  # DEFER: the previous tile's RMW first
                _rmw(y_keep, d_keep)
            dst_sb = nl.ndarray((B, nd2), dtype=nl.int32, buffer=nl.sbuf)
            w_sb = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
            if kept:  # TILE_PREFETCH: this tile's rows were loaded by the previous tile
                nisa.tensor_copy(dst=dst_sb, src=k_dst)
                nisa.tensor_copy(dst=w_sb, src=k_w)
                x_sb = k_x
            else:
                x_sb = nl.ndarray((B, h), dtype=x.dtype, buffer=nl.sbuf)
                _fetch(dst_sb, w_sb, x_sb)
            xT = nl.ndarray((128, nh, B), dtype=x.dtype, buffer=nl.sbuf)
            if xnd:  # R2-03b: x column blocks hh < xnd transposed SBUF -> SBUF by DMA
                xkw = dict(dge_mode=_DGE_MODES[XT_DGE])
                if XT_FORM == "3d":
                    nisa.dma_transpose(dst=xT[0:128, 0:xnd, 0:B],
                                       src=x_sb.reshape((B, nh, 128))[0:B, 0:xnd, 0:128], axes=(2, 1, 0), **xkw)
                else:
                    for hh in range(xnd):
                        nisa.dma_transpose(dst=xT[0:128, hh, 0:B], src=x_sb[0:B, hh * 128 : (hh + 1) * 128],
                                           axes=(1, 0), **xkw)
            for g0 in range(xnd, nh, 4):
                g1 = min(nh, g0 + 4)
                tx = nl.ndarray((128, (g1 - g0) * B), dtype=x.dtype, buffer=nl.psum)
                for hh in range(g0, g1):
                    nisa.nc_transpose(dst=tx[0:128, (hh - g0) * B : (hh - g0 + 1) * B],
                                      data=x_sb[0:B, hh * 128 : (hh + 1) * 128])
                for hh in range(g0, g1):
                    nisa.tensor_copy(dst=xT[0:128, hh, 0:B], src=tx[0:128, (hh - g0) * B : (hh - g0 + 1) * B])
            if tpref:  # the next tile's loads, under this tile's matmuls (k_x is free once xT is built)
                nisa.tensor_scalar(dst=off_sb, data=off_sb, op0=nl.add, operand0=B)
                _fetch(k_dst, k_w, k_x)
            # a [B, rows] = relu(x @ up.T)^2: token tile stationary, the expert's whole I as moving
            pu = nl.ndarray((B, rows), dtype=nl.float32, buffer=nl.psum)
            for hh in range(nh):
                nisa.nc_matmul(dst=pu, stationary=xT[0:128, hh, 0:B], moving=upT[0:128, hh, 0:rows], accumulate=hh > 0)
            r = nl.ndarray((B, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=r, data=pu, op0=nl.maximum, operand0=0.0)
            a_sb = nl.ndarray((B, rows), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=a_sb, data1=r, data2=r, op=nl.multiply)
            aT = nl.ndarray((pt, nj, B), dtype=x.dtype, buffer=nl.sbuf)
            for j in range(nj):
                ta = nl.ndarray((pts[j], B), dtype=x.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=ta, data=a_sb[0:B, j * pt : j * pt + pts[j]])
                if c_sb is not None:  # FP8: per-intermediate-row scale on the way out of PSUM
                    nisa.tensor_scalar(dst=aT[0:pts[j], j, 0:B], data=ta, op0=nl.multiply,
                                       operand0=c_sb[0:pts[j], j : j + 1])
                else:
                    nisa.tensor_copy(dst=aT[0:pts[j], j, 0:B], src=ta)
            y_sb = y_keep if defer else nl.ndarray((B, h), dtype=x.dtype, buffer=nl.sbuf)
            for c in range(n_chunks):
                c0 = c * _PSUM_COLS
                csz = min(_PSUM_COLS, h - c0)
                py = nl.ndarray((B, csz), dtype=nl.float32, buffer=nl.psum)
                for j in range(nj):
                    nisa.nc_matmul(dst=py, stationary=aT[0:pts[j], j, 0:B], moving=dn_sb[0:pts[j], j, c0 : c0 + csz],
                                   accumulate=j > 0)
                nisa.tensor_scalar(dst=y_sb[0:B, c0 : c0 + csz], data=py, op0=nl.multiply, operand0=w_sb[0:B, 0:1])
            if defer:
                nisa.tensor_copy(dst=d_keep, src=dst_sb)
            else:
                _rmw(y_sb, dst_sb)
            if not tpref:
                nisa.tensor_scalar(dst=off_sb, data=off_sb, op0=nl.add, operand0=B)

        if "tiles" not in ABLATE:
            if tile0_on:  # first tile as straight-line code; ntile holds the remaining tiles
                if TILE0_LOOPED == "all" or (TILE0_LOOPED == "first" and g == 0):
                    one = nisa.register_alloc(1)
                    nl.fori_loop(0, one, _tile)
                else:
                    _tile(0, kept=carry and g > 0)
                for i_sl in range(1, nsl):  # R2-03b TILES_SL: more straight-line tiles (no loop barrier)
                    _tile(i_sl, pend=defer, kept=tpref)
            if defer:
                nl.fori_loop(lb, n_reg, lambda _i: _tile(_i, pend=True, kept=tpref))
                _rmw(y_keep, d_keep)  # the expert's last tile
            else:
                nl.fori_loop(lb, n_reg, _tile)

    # static group loop; the next group's weight DMAs are issued before this group's dynamic tile loop
    nxt = _load(prg_id)
    for g in range(g_n):
        cur = nxt
        if g + 1 < g_n:
            nxt = _load((g + 1) * n_prgs + prg_id)
        _expert(g, cur)
    return out


def _row_tile(rows: int) -> int:
    for pt in range(min(rows, 128), 0, -1):
        if rows % pt == 0:
            return pt
    return 1


def _f2i(v: Tensor) -> Tensor:
    """Exact float32 -> int32 for integer values in [0, 2^23) by a bit cast."""
    return (v + 8388608.0).view(torch.int32) - 0x4B000000


def _excl_cumsum0(v: Tensor, mode: str) -> Tensor:
    """Exclusive cumsum over dim 0 of a small non-negative integer-valued float32 tensor."""
    if "mm" not in mode:
        return torch.cumsum(v, dim=0) - v
    m = v.shape[0]
    ar = torch.arange(m, device=v.device)
    tri = (ar.view(m, 1) > ar.view(1, m)).to(torch.float32)
    return torch.matmul(tri, v.view(m, -1)).view(v.shape)


def dispatch(idx: Tensor, w: Tensor, num_experts: int, mode: str = DISPATCH_MODE, grp: int = 1, pair: str = "static",
             rows: int = 0, acc: bool = False, padskip: bool = False, firstw: bool = False, xcarry: bool = False):
    """Slot layout for ``idx`` / ``w`` ``[T, k]`` (static shapes, no sort; float32 counting, exact < 2^24).

    Returns ``(slot_tok [NPAD, 1], slot_pair [NPAD, 1], ntile [E, 1], tstart [E, 1], wp [T * k + 1, 1])``
    (int32 except ``wp``, the routing weights in pair order plus a 0); pad slots: token 0, pair ``T * k``.
    ``grp > 1`` (expert-split kernel): ``ntile`` is ``[E / grp, 1]``, the largest tile count in each group of
    ``grp`` consecutive experts, and every expert's slot range is padded to its group's count.
    """
    t, k = idx.shape
    n = t * k
    nb = (n + B - 1) // B
    npad = grp * (nb + num_experts) * B  # sum over groups of grp * max(tiles) <= grp * sum(tiles)
    if acc and TILE_PREFETCH and t >= TILE_PREFETCH_MIN_T:  # R2-03: one pad tile for the last prefetch of the kernel
        npad += B
    if acc and grp > 1 and TILE0 and t >= TILE0_MIN_T and _tiles_sl(t) > 1:  # R2-03b: groups padded to TILES_SL tiles
        npad += grp * (_tiles_sl(t) - 1) * (num_experts // grp) * B
    dev = idx.device
    f = idx.reshape(n).to(torch.float32)
    if nb * B > n:
        f = torch.cat([f, torch.full((nb * B - n,), float(num_experts), dtype=torch.float32, device=dev)])
    experts = torch.arange(num_experts, dtype=torch.float32, device=dev)
    oh = (f.view(nb * B, 1) == experts.view(1, -1)).to(torch.float32)  # [nb*B, E]
    oh_b = oh.view(nb, B, num_experts)
    # exclusive rank inside each block of B pairs: strictly lower-triangular 0/1 matmul (exact, <= 127)
    ar = torch.arange(B, device=dev)
    tri = (ar.view(B, 1) > ar.view(1, B)).to(torch.bfloat16)
    in_blk = torch.matmul(tri, oh_b.to(torch.bfloat16)).to(torch.float32)  # [nb, B, E]
    bsum = oh_b.sum(1)  # [nb, E]
    boff = _excl_cumsum0(bsum, mode)  # [nb, E], exclusive
    counts = bsum.sum(0)  # [E]
    ntile = torch.floor((counts + (B - 1)) / B)
    sort_oh = None
    if grp > 1 and pair == "sorted":
        # rank experts by tile count (descending, ties by id) without a sort: rank[e] = #{e' before e}
        te = ntile
        ar_e = torch.arange(num_experts, device=dev)
        before = (te.view(1, -1) > te.view(-1, 1)) | ((te.view(1, -1) == te.view(-1, 1)) & (ar_e.view(1, -1) < ar_e.view(-1, 1)))
        rank = before.to(torch.float32).sum(1)  # [E], a permutation of 0..E-1
        sort_oh = (rank.view(-1, 1) == ar_e.to(torch.float32).view(1, -1)).to(torch.float32)  # [E, S]: e -> slot
        sv = (sort_oh * te.view(-1, 1)).sum(0)  # [S] tile counts in slot order (descending)
        ng = num_experts // grp
        ntile = sv.view(ng, grp).amax(dim=1)  # [E / grp]
        tile0 = acc and TILE0 and t >= TILE0_MIN_T
        nsl = _tiles_sl(t) if (tile0 and DEFER and not TILE0_LOOPED and not TILE0_LB1) else 1
        if tile0 or CLAMP1:  # T12d: at least one tile per group (the kernel runs it unconditionally)
            ntile = torch.clamp(ntile, min=float(nsl))
        reg_s = (ntile.view(ng, 1) + torch.zeros((ng, grp), dtype=torch.float32, device=dev)).view(num_experts)
        reg = (sort_oh * reg_s.view(1, -1)).sum(1)  # [E] padded tile count per expert
        if xcarry and acc:  # R2-03d: slot s = g * grp + c starts at c * (all of a core's tiles) + core-local offset
            cs = _excl_cumsum0(ntile.view(ng, 1), mode).view(ng)
            cst = cs.view(ng, 1) + ntile.sum() * torch.arange(grp, dtype=torch.float32, device=dev).view(1, grp)
            tstart_x = (sort_oh * (cst.view(1, num_experts) * B)).sum(1)  # [E] in expert-id order
    elif grp > 1:
        ng = num_experts // grp
        ntile = ntile.view(ng, grp).amax(dim=1)  # [E / grp] shared trip counts
        reg = (ntile.view(ng, 1) + torch.zeros((ng, grp), dtype=torch.float32, device=dev)).view(num_experts)
    else:
        reg = ntile
    if xcarry and acc and sort_oh is not None:
        tstart = tstart_x
    else:
        tstart = _excl_cumsum0(reg, mode) * B  # first slot per expert
    if "fast" in mode:
        # one fused reduction (slot = expert start + block offset + rank in block), and float -> int32 by a
        # bit cast (x + 2^23 has the integer x in its low mantissa bits; exact for 0 <= x < 2^23). A plain
        # ``.to(torch.int32)`` of these 12k floats cost ~0.5 ms on device.
        slotf = ((in_blk + (boff + tstart.view(1, num_experts)).view(nb, 1, num_experts)) * oh_b).sum(-1)
        slot = _f2i(slotf.view(nb * B)[:n]).to(torch.int64)
    else:
        rank = ((in_blk + boff.view(nb, 1, num_experts)) * oh_b).sum(-1).view(nb * B)  # [nb*B]
        base = (oh * tstart.view(1, -1)).sum(-1)
        slot = (base + rank)[:n].to(torch.int64)  # [n]
    pair = torch.arange(n, dtype=torch.int32, device=dev)
    firstw = firstw and acc and SCATTER1 and sort_oh is not None and padskip and grp == 2
    if firstw:  # R2-03d: flag the first pair of each (core, token) in processing order (lowest rank of that parity)
        fl = 1 << max(1, n.bit_length())  # power of two > n (the pad value)
        rk = (oh[:n] * rank.view(1, -1)).sum(-1).view(t, k)  # sorted rank of each pair's expert
        core_k = rk - 2.0 * torch.floor(rk * 0.5)  # [t, k] core of each pair
        same = core_k.view(t, k, 1) == core_k.view(t, 1, k)
        earlier = (same & (rk.view(t, 1, k) < rk.view(t, k, 1))).to(torch.float32).sum(-1)  # [t, k]
        first = (earlier < 0.5).to(torch.int32).view(n)
        touched = torch.stack([(core_k < 0.5).to(torch.float32).amax(-1), (core_k > 0.5).to(torch.float32).amax(-1)]) > 0.5
        code = pair + first * fl
        slot_code = torch.full((npad,), n, dtype=torch.int32, device=dev).scatter(0, slot, code)
        scf = (slot_code + 0x4B000000).view(torch.float32) - 8388608.0
        isf = scf >= float(fl)
        spf = scf - isf.to(torch.float32) * float(fl)
        slot_pair = _f2i(spf)
    else:
        slot_pair = torch.full((npad,), n, dtype=torch.int32, device=dev).scatter(0, slot, pair)
    if acc and SCATTER1 and sort_oh is not None:  # T12d: one scatter; token and plane row from slot_pair
        if not firstw:
            spf = (slot_pair + 0x4B000000).view(torch.float32) - 8388608.0  # exact int32 -> float32 (0 <= v < 2^23)
        valid = spf < float(n)
        tokf = torch.floor((spf + 0.5) * (1.0 / k))  # pair // k, exact for pair < 2^20
        slot_tok = _f2i(torch.where(valid, tokf, torch.full_like(tokf, float(t)) if padskip else torch.zeros_like(tokf))).view(npad, 1)
        oob = torch.full_like(tokf, float(grp * t))
        if firstw:  # [write row, RMW row]
            slot_dst = _f2i(torch.stack([torch.where(valid & isf, tokf, oob), torch.where(valid & ~isf, tokf, oob)], 1))
        else:
            slot_dst = _f2i(torch.where(valid, tokf, oob)).view(npad, 1)
        wp = torch.cat([w.reshape(n).to(torch.float32), torch.zeros(1, dtype=torch.float32, device=dev)])
        tstart_s = (sort_oh * tstart.view(-1, 1)).sum(0)
        erow_s = (sort_oh * (torch.arange(num_experts, device=dev, dtype=torch.float32) * rows).view(-1, 1)).sum(0)
        nk = ntile - float(nsl) if (tile0 and not TILE0_LB1) else ntile
        r = (slot_tok, slot_pair.view(npad, 1), _f2i(nk).view(-1, 1), _f2i(tstart_s).view(-1, 1),
             wp.view(n + 1, 1), _f2i(erow_s).view(-1, 1), slot_dst)
        return r + ((touched,) if firstw else ())
    if "tokx" in mode:  # token id of pair t*k+j without an integer division
        ptok = torch.arange(t, dtype=torch.int32, device=dev).view(t, 1).expand(t, k).reshape(n)
    else:
        ptok = pair // k
    slot_tok = torch.zeros((npad,), dtype=torch.int32, device=dev).scatter(0, slot, ptok)
    # routing weights stay in pair order (+ a 0 for the pad row); the kernel gathers them by pair id.
    # (A float32 scatter here lowered to ~74k serialized GpSimd instructions: 36 ms per layer.)
    wp = torch.cat([w.reshape(n).to(torch.float32), torch.zeros(1, dtype=torch.float32, device=dev)])
    cast = _f2i if "fast" in mode else (lambda v: v.to(torch.int32))
    if sort_oh is not None and acc:  # destination row of each slot: core * T + token (pads: 2 T, out of bounds)
        core_e = rank - 2.0 * torch.floor(rank * 0.5)  # [E] core of each expert (slot parity)
        core_p = (oh_b * core_e.view(1, 1, num_experts)).sum(-1).view(nb * B)[:n]  # [n]
        dst_p = _f2i(core_p * float(t)).to(torch.int32) + ptok
        slot_dst = torch.full((npad,), grp * t, dtype=torch.int32, device=dev).scatter(0, slot, dst_p)
    if sort_oh is not None:  # slot order for the kernel: start slot and first weight row of the expert in slot s
        tstart_s = (sort_oh * tstart.view(-1, 1)).sum(0)
        erow_s = (sort_oh * (torch.arange(num_experts, device=dev, dtype=torch.float32) * rows).view(-1, 1)).sum(0)
        nk = ntile - float(nsl) if (tile0 and not TILE0_LB1) else ntile
        r = (slot_tok.view(npad, 1), slot_pair.view(npad, 1), cast(nk).view(-1, 1),
             cast(tstart_s).view(-1, 1), wp.view(n + 1, 1), cast(erow_s).view(-1, 1))
        return r + ((slot_dst.view(npad, 1),) if acc else ())
    return (slot_tok.view(npad, 1), slot_pair.view(npad, 1), cast(ntile).view(-1, 1),
            cast(tstart).view(-1, 1), wp.view(n + 1, 1))


def moe_prefill_routed(x: Tensor, up: Tensor, down: Tensor, idx: Tensor, w: Tensor, rows: int,
                       num_experts: int, run=None, c: Tensor | None = None, up_t: bool = False) -> Tensor:
    """Routed-expert sum for ``x`` ``[T, H]`` (``idx``/``w`` ``[T, k]``). Returns ``[T, H]`` in ``x.dtype``.

    ``run`` replaces the device launch (e.g. the CPU simulator); it gets the kernel kwargs.
    """
    t, h = x.shape
    k = idx.shape[1]
    n = t * k
    esplit = SPLIT == "e"
    if "nodispatch" in ABLATE:  # timing only: constant slot layout
        npad = ((n + B - 1) // B + num_experts) * B
        z = (idx[0, 0] * 0).to(torch.int32)
        slot_tok = torch.zeros((npad, 1), dtype=torch.int32, device=x.device) + z
        slot_pair = slot_tok + n
        ntile = torch.ones((num_experts // (2 if esplit else 1), 1), dtype=torch.int32, device=x.device) + z
        tstart = torch.arange(num_experts, dtype=torch.int32, device=x.device).view(-1, 1) * B + z
        ws = torch.zeros((n + 1, 1), dtype=torch.float32, device=x.device) + w[0, 0] * 0
    else:
        esplit = SPLIT == "e"
    sorted_pairs = esplit and PAIR == "sorted"
    acc = sorted_pairs and ACC
    d = dispatch(idx, w, num_experts, grp=2 if esplit else 1, pair="sorted" if sorted_pairs else "static", rows=rows,
                 acc=acc, padskip=PADSKIP and acc and SCATTER1 and c is None, firstw=FIRSTW and c is None and not up_t,
                 xcarry=_xcarry_on(t) and c is None and not up_t)
    slot_tok, slot_pair, ntile, tstart, ws = d[:5]
    npad = slot_tok.shape[0]
    pt = _row_tile(rows if esplit else rows // 2)
    if acc and WTILE == "128":  # T14d: 128-row weight tiles, short last tile
        pt = min(rows, 128)
    kw = dict(x=x, up=up, down=down, ntile=ntile, tstart=tstart, slot_tok=slot_tok, slot_pair=slot_pair, ws=ws,
              geom=torch.zeros((rows, pt), dtype=torch.int32, device=x.device))
    if sorted_pairs:
        kw["erow"] = d[5]
    if acc:
        kw["slot_dst"] = d[6]
    if "nokernel" in ABLATE:  # timing of the torch dispatch/combine only
        ys = torch.zeros((2 * (n + 1), h), dtype=x.dtype, device=x.device) + ws[0, 0].to(x.dtype) + slot_tok[0, 0]
    else:
        kern = (moe_prefill_kernel_esplit_acc if acc else moe_prefill_kernel_esplit_sorted if sorted_pairs
                else moe_prefill_kernel_esplit) if esplit else moe_prefill_kernel
        if c is not None:  # FP8 (R2-05): only the default ACC kernel
            if kern is not moe_prefill_kernel_esplit_acc:
                raise ValueError("NEMOTRON_H_FP8=1 needs the default prefill kernel (SPLIT=e, PAIR=sorted, ACC=1)")
            kern = moe_prefill_kernel_esplit_acc_fp8
            kw["c"] = c
            kw["eup"] = _f2i(torch.floor(d[5].to(torch.float32) * (128.0 / rows) + 0.5)).view(-1, 1)
        elif up_t:  # R2-03b: BF16 up stored [E * 128, nh * I]
            if kern is not moe_prefill_kernel_esplit_acc:
                raise ValueError("up_t needs the default prefill kernel (SPLIT=e, PAIR=sorted, ACC=1)")
            kern = moe_prefill_kernel_esplit_acc_upt
            kw["eup"] = _f2i(torch.floor(d[5].to(torch.float32) * (128.0 / rows) + 0.5)).view(-1, 1)
        ys = run(kw, kern) if run is not None else wrap_nki(kern)[2](**kw)
    if "nocombine" in ABLATE:  # timing only
        return x * ys[0:1, :].to(x.dtype)
    if acc and len(d) > 7:  # R2-03d FIRSTW: planes are not zeroed; rows a core never touched are masked
        yv = ys.view(2, t, h)
        return torch.where(d[7].view(2, t, 1), yv, torch.zeros_like(yv)).sum(dim=0).to(x.dtype)
    if acc:  # the two cores' float32 planes
        return ys.view(2, t, h).sum(dim=0).to(x.dtype)
    if esplit:  # one row per pair: sum the k rows of each token
        return ys[:n].view(t, k, h).sum(dim=1, dtype=torch.float32).to(x.dtype)
    ys = ys.view(-1, n + 1, h)[:, :n].reshape(-1, t, k, h)
    if COMBINE == "fp32":
        return ys.float().sum(0).sum(1).to(x.dtype)
    return ys.sum(dim=(0, 2), dtype=torch.float32).to(x.dtype)
