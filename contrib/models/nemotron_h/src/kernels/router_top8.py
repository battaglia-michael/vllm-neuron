"""Router top-k for up to 8 experts per token with ``max8`` + ``nc_find_index8`` (R2-10).

The MoE router picks the top 6 of 128 ``choice = sigmoid(logits) + bias`` values per token. ``moe.py`` called
``vllm_neuron.functional.topk.topk``, which runs the vendored rotational NKI top-k kernel (built for vocab-size rows)
for every bucket. A batch-1 device profile (results/trn2/r2-next/) puts it at ~15 us per MoE layer between the
router's sigmoid/bias and the first instruction of the top-k MoE kernel, with several HBM round trips, on the
critical path of every MoE layer (23 per decode step).

This kernel loads the ``[T, E]`` fp32 rows (tokens on partitions, 128-row tiles split over the two physical cores),
runs ``nisa.max8`` (the 8 largest values per row, descending) and ``nisa.nc_find_index8`` (their indices), and
writes the first ``k`` indices as int32. This is the selection nkilib's ``router_topk`` uses. Tie behaviour:
``nc_find_index8`` returns the first occurrence of each value, so two experts with bit-identical fp32 ``choice``
inside the top k would both map to the lower index (``torch.topk`` returns both). With fp32 sigmoid outputs plus
fp32 biases this needs an exact fp32 tie.
"""

import os

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

# Router graphs with at most this many tokens use this kernel (0 = off: vllm_neuron.functional.topk).
MAX_T = int(os.environ.get("NEMOTRON_H_ROUTER_TOP8_MAX_T", "0"))


@nki.jit
def router_top8_kernel(choice: Tensor):
    """``choice`` ``[T, E]`` float32 (8 <= E <= 16384) -> ``[T, 8]`` int32 indices of the 8 largest values per row,
    in descending value order."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("router_top8", (0, 1), 2)
    t, e = choice.shape
    out = nl.ndarray((t, 8), dtype=nl.int32, buffer=nl.shared_hbm)
    n_tiles = (t + 127) // 128
    for ti in range(n_tiles):
        if ti % n_prgs != prg_id:
            continue
        r0 = ti * 128
        r = min(128, t - r0)
        c_sb = nl.ndarray((r, e), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=c_sb, src=choice[r0 : r0 + r, 0:e])
        v = nl.ndarray((r, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=v, src=c_sb)
        iu = nl.ndarray((r, 8), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=iu, data=c_sb, vals=v)
        ii = nl.ndarray((r, 8), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ii, src=iu)
        nisa.dma_copy(dst=out[r0 : r0 + r, 0:8], src=ii)
    return out


def maybe_router_topk(choice: Tensor, k: int) -> Tensor | None:
    """``[T, k]`` int64 expert ids through :func:`router_top8_kernel` when ``0 < T <= MAX_T``, ``k <= 8`` and the
    tensor is on a Neuron device; ``None`` otherwise (the caller keeps its top-k)."""
    if not (0 < choice.shape[0] <= MAX_T and k <= 8 and choice.shape[-1] >= 8 and choice.dim() == 2):
        return None
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    if not can_run_kernel(choice):
        return None
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    idx = wrap_nki(router_top8_kernel)[2](choice=choice.to(torch.float32).contiguous())
    return idx[:, :k].to(torch.int64)


# R2-10 v2: the whole batch-1 router in one kernel, from the MoE input x, so no small vector crosses a torch / kernel
# boundary between the router matmul and the top-k MoE kernel (the device profile shows each such hand-off of a
# partition-major [1, 128] vector costing ~9 us). Batch-1 decode only (NEMOTRON_H_ROUTER_FUSED=1 with
# NEMOTRON_H_MOE_TOPK_MAX_T=1, BF16 path).
# Default on since R2-10 (NEMOTRON_H_ROUTER_FUSED=0 restores the torch router + rotational top-k at batch 1).
FUSED = os.environ.get("NEMOTRON_H_ROUTER_FUSED", "1") == "1"


@nki.jit
def router_fused_kernel(x: Tensor, wr: Tensor, bias: Tensor, geom: Tensor, k: int, scale: float, norm: bool):
    """One token's router: ``x`` ``[1, H]`` bf16, ``wr`` ``[H, E]`` float32, ``bias`` ``[E]`` float32, ``geom`` dummy
    ``[rows, 1]`` (expert rows I).

    ``logits = x.float() @ wr`` (Tensor Engine, fp32), ``scores = sigmoid(logits)``, top-``k`` of ``scores + bias``
    (``max8`` + ``nc_find_index8``), ``w = scores[idx] / (sum + 1e-20) * scale`` (``norm``) or ``scores[idx] * scale``.
    Returns ``off`` ``[1, k]`` int32 (``idx * rows``, the top-k MoE kernel's row offsets) and ``w`` ``[1, k]`` float32.
    Both physical cores compute it; program 0 writes the outputs.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("router_fused", (0, 1), 2)
    t, h = x.shape
    h2, e = wr.shape
    rows = geom.shape[0]
    assert t == 1 and h == h2 and h % 128 == 0 and 8 <= e <= 512 and k <= 8
    kt_n = h // 128
    off = nl.ndarray((1, k), dtype=nl.int32, buffer=nl.shared_hbm)
    w_out = nl.ndarray((1, k), dtype=nl.float32, buffer=nl.shared_hbm)
    # x^T as [128, KT] fp32 (one contiguous load of x as [KT, 128] rows, one PE transpose)
    xr = nl.ndarray((kt_n, 128), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xr, src=x.ap(pattern=[[128, kt_n], [1, 128]], offset=0))
    xf = nl.ndarray((kt_n, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=xf, src=xr)
    t_ps = nl.ndarray((128, kt_n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=t_ps, data=xf, engine=nisa.tensor_engine)
    xT = nl.ndarray((128, kt_n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=xT, src=t_ps)
    # router weight [H, E] as KT row tiles [128, E] (one DMA), logits [1, E] accumulated in PSUM over the K tiles
    w_sb = nl.ndarray((128, kt_n, e), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_sb, src=wr.ap(pattern=[[e, 128], [128 * e, kt_n], [1, e]], offset=0))
    lg_ps = nl.ndarray((1, e), dtype=nl.float32, buffer=nl.psum)
    for kt in range(kt_n):
        nisa.nc_matmul(dst=lg_ps, stationary=xT[0:128, kt : kt + 1], moving=w_sb[0:128, kt, 0:e], accumulate=kt > 0)
    b_sb = nl.ndarray((1, e), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_sb, src=bias.ap(pattern=[[e, 1], [1, e]], offset=0))
    sc = nl.ndarray((1, e), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sc, op=nl.sigmoid, data=lg_ps)
    ch = nl.ndarray((1, e), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ch, data1=sc, data2=b_sb, op=nl.add)
    v8 = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=v8, src=ch)
    i8 = nl.ndarray((1, 8), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=i8, data=ch, vals=v8)
    # weights: the selected sigmoid scores, normalized, scaled
    sw = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=sw, value=0.0)
    nisa.nc_n_gather(dst=sw[0:1, 0:k], data=sc, indices=i8[0:1, 0:k])
    wk = nl.ndarray((1, k), dtype=nl.float32, buffer=nl.sbuf)
    if norm:
        ssum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ssum, data=sw[0:1, 0:k], op=nl.add, axis=(1,))
        nisa.tensor_scalar(dst=ssum, data=ssum, op0=nl.add, operand0=1e-20)
        # w = s * (1 / sum) * scale (the Vector engine has no divide; 1 ulp from torch's s / sum)
        rs = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=ssum)
        nisa.tensor_scalar(dst=wk, data=sw[0:1, 0:k], op0=nl.multiply, operand0=rs[0:1, 0:1],
                           op1=nl.multiply, operand1=float(scale))
    else:
        nisa.tensor_scalar(dst=wk, data=sw[0:1, 0:k], op0=nl.multiply, operand0=float(scale))
    # row offsets idx * rows (exact in fp32 for idx * rows < 2^24)
    i_f = nl.ndarray((1, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=i_f, src=i8[0:1, 0:k])
    nisa.tensor_scalar(dst=i_f, data=i_f, op0=nl.multiply, operand0=float(rows))
    o_i = nl.ndarray((1, k), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o_i, src=i_f)
    if prg_id == 0:
        nisa.dma_copy(dst=off[0:1, 0:k], src=o_i)
        nisa.dma_copy(dst=w_out[0:1, 0:k], src=wk)
    return off, w_out


def router_fused(x: Tensor, wr: Tensor, bias: Tensor, rows: int, k: int, scale: float, norm: bool):
    """``(off [k, 1] int32, w [1, k] float32)`` for one token, through :func:`router_fused_kernel`."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    geom = torch.zeros((rows, 1), dtype=torch.int32, device=x.device)
    off, w = wrap_nki(router_fused_kernel)[2](x=x.contiguous(), wr=wr, bias=bias, geom=geom, k=k, scale=float(scale),
                                              norm=bool(norm))
    return off.view(k, 1), w
