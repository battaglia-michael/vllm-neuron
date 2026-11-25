"""Decode GEMV: ``out = x @ w`` for a few tokens with the activation as the stationary matmul operand (R2-10).

For ``x`` ``[M, K]`` with small ``M`` (batch-1 decode) and a weight stored ``[K, N]`` (in, out), the compiler lowers
``x @ w`` with the weight as the stationary operand: every 128 x 128 weight tile costs one LDWEIGHTS plus one MATMUL
that moves a single column. A batch-1 device profile of the full model (R2-10, results/trn2/r2-next/) shows the Mamba
``in_proj`` (``[1, 2688] @ [2688, 2576]``) at ~30 us of Tensor Engine time per layer for 6.9 MB of weights per physical
core (~230 GB/s), and the ``lm_head`` (``[1, 2688] @ [32768, 2688]^T``, stored ``[out, in]``) at ~314 us with 22 GFLOP
of PE transposes of the weight.

This kernel puts ``x^T`` (``[128, M]`` per K tile) in the stationary slot and streams the weight as the moving
operand in ``[128, 512]`` column chunks, accumulating ``[M, 512]`` in PSUM over the K tiles. One MATMUL then moves
128 KB of weights (~0.38 us on trn2, ~340 GB/s per physical core, about the HBM share of one core), independent of
``M <= 128``. The weight tiles are ``[128, NB]`` row blocks (contiguous ``NB * 2`` bytes per partition) on the
hardware DGE queues, alternating Sync / Scalar (``DMA``). The columns are split over the two physical cores of the
logical core (LNC=2 launch grid); ``N`` must be a multiple of the grid size and ``K`` of 128.

Numerics: fp32 PSUM accumulation over K tiles in order, rounded once to bf16 (the compiler's matmul also accumulates
in fp32; the summation order may differ, so outputs can differ in the last bf16 bit).
"""

import os

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

_PSUM_COLS = 512
# R2-10 switch: decode graphs with at most this many tokens run the Mamba in_proj and the batch-1 MoE shared expert
# through this kernel (0 = off, the compiler's matmuls).
MAX_T = int(os.environ.get("NEMOTRON_H_DECODE_GEMV_MAX_T", "0"))
# Call sites that use it: inproj (Mamba in_proj), shup / shdn (shared-expert up / down in the batch-1 MoE).
SITES = frozenset(s for s in os.environ.get("NEMOTRON_H_DECODE_GEMV_SITES", "inproj,shup,shdn").split(",") if s)
# Columns per weight row block (one DMA per (block, K tile)); 2048 bf16 = 4 KB per partition.
NB = int(os.environ.get("NEMOTRON_H_GEMV_NB", "2048"))
# Weight-tile DMA: hw = hardware DGE alternating Sync / Scalar, sw = software DGE, auto = compiler default.
DMA = os.environ.get("NEMOTRON_H_GEMV_DMA", "hw")
if DMA not in ("hw", "sw", "auto"):
    raise ValueError(f"NEMOTRON_H_GEMV_DMA must be hw, sw or auto, got {DMA!r}")
# Weight loads at a runtime offset of 0 derived from x, so they stay inside the kernel's own segment (see the kernel).
PIN = os.environ.get("NEMOTRON_H_GEMV_PIN", "1") == "1"


def _wdma(dst, src, k):
    if DMA == "sw":
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.swdge)
    elif DMA == "hw":
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.sync if k % 2 == 0 else nisa.engine.scalar)
    else:
        nisa.dma_copy(dst=dst, src=src)


@nki.jit
def decode_gemv_kernel(x: Tensor, w: Tensor):
    """``x`` ``[M, K]`` bf16, ``w`` ``[K, N]`` bf16 -> ``[M, N]`` bf16 (see module docstring)."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("decode_gemv", (0, 1), 2)
    m, k = x.shape
    k2, n = w.shape
    assert k == k2 and n % n_prgs == 0 and m <= 128
    kt_n = (k + 127) // 128
    ksz = [min(128, k - kt * 128) for kt in range(kt_n)]  # rows of K tile kt (the last one may be short)
    ncore = n // n_prgs
    col0 = prg_id * ncore
    out = nl.ndarray((m, n), dtype=x.dtype, buffer=nl.shared_hbm)

    # x^T: xT[p, kt, i] = x[i, kt * 128 + p], through PE transposes (exact).
    xT = nl.ndarray((128, kt_n, m), dtype=x.dtype, buffer=nl.sbuf)
    if m == 1 and k % 128 == 0:
        # x as [KT, 128] rows (one contiguous DMA), one transpose -> [128, KT].
        xr = nl.ndarray((kt_n, 128), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xr, src=x.ap(pattern=[[128, kt_n], [1, 128]], offset=0))
        xf = nl.ndarray((kt_n, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xf, src=xr)
        t_ps = nl.ndarray((128, kt_n), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=t_ps, data=xf, engine=nisa.tensor_engine)
        nisa.tensor_copy(dst=xT.ap(pattern=[[kt_n * m, 128], [m, kt_n]], offset=0), src=t_ps)
    else:
        xr = nl.ndarray((m, k), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xr, src=x[0:m, 0:k])
        xf = nl.ndarray((m, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xf, src=xr)
        for kt in range(kt_n):
            kr = ksz[kt]
            t_ps = nl.ndarray((kr, m), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=t_ps, data=xf[0:m, kt * 128 : kt * 128 + kr], engine=nisa.tensor_engine)
            nisa.tensor_copy(dst=xT[0:kr, kt, 0:m], src=t_ps)

    q = 0
    zi = None
    if PIN:
        # z = 0 computed from x (int32(x[0, 0]) * 0, exact for any int), used as a runtime row offset of every weight
        # load so the compiler cannot hoist the loads into earlier layers (where they compete with the routed-expert
        # loads of the MoE segment; first R2-10 trace).
        z0 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=z0, src=xT[0:1, 0, 0:1])
        zi = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=zi, data=z0, op0=nl.multiply, operand0=0)
    for b0 in range(0, ncore, NB):
        bw = min(NB, ncore - b0)
        nch = (bw + _PSUM_COLS - 1) // _PSUM_COLS
        ps = []
        for c in range(nch):
            cw = min(_PSUM_COLS, bw - c * _PSUM_COLS)
            ps.append(nl.ndarray((m, cw), dtype=nl.float32, buffer=nl.psum))
        for kt in range(kt_n):
            kr = ksz[kt]
            w_sb = nl.ndarray((kr, bw), dtype=w.dtype, buffer=nl.sbuf)
            if zi is None:
                src = w[kt * 128 : kt * 128 + kr, col0 + b0 : col0 + b0 + bw]
            else:
                src = w.ap(pattern=[[n, kr], [1, bw]], offset=kt * 128 * n + col0 + b0,
                           scalar_offset=zi[0:1, 0:1], indirect_dim=0)
            _wdma(w_sb, src, q)
            q += 1
            for c in range(nch):
                c0 = c * _PSUM_COLS
                cw = min(_PSUM_COLS, bw - c0)
                nisa.nc_matmul(dst=ps[c][0:m, 0:cw], stationary=xT[0:kr, kt, 0:m],
                               moving=w_sb[0:kr, c0 : c0 + cw], accumulate=kt > 0)
        o_sb = nl.ndarray((m, bw), dtype=x.dtype, buffer=nl.sbuf)
        for c in range(nch):
            c0 = c * _PSUM_COLS
            cw = min(_PSUM_COLS, bw - c0)
            nisa.tensor_copy(dst=o_sb[0:m, c0 : c0 + cw], src=ps[c][0:m, 0:cw])
        nisa.dma_copy(dst=out[0:m, col0 + b0 : col0 + b0 + bw], src=o_sb)
    return out


def decode_gemv(x: Tensor, w: Tensor) -> Tensor:
    """``x @ w`` for ``x`` ``[M, K]`` and ``w`` ``[K, N]`` bf16 on device (LNC=2 grid)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(decode_gemv_kernel)[2](x=x.contiguous(), w=w)


def maybe_gemv(x: Tensor, w: Tensor, site: str) -> Tensor:
    """``x @ w``, through :func:`decode_gemv` when ``site`` is enabled, ``0 < M <= MAX_T`` and the tensors are on a
    Neuron device."""
    if site in SITES and 0 < x.shape[0] <= MAX_T and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        if can_run_kernel(x):
            return decode_gemv(x, w)
    return x @ w
