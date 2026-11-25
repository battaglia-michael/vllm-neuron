# SPDX-License-Identifier: Apache-2.0
"""NemotronH MoE block (23 of the 30B-A3B's 52 layers), plain PyTorch baseline.

Math copied from ``transformers``' ``NemotronHMoE`` / ``NemotronHTopkRouter``:

    logits  = x.float() @ W_router.float().T           # [T, 128], float32
    scores  = sigmoid(logits)
    choice  = scores + e_score_correction_bias         # bias picks experts only
    idx     = topk(choice, 6)                          # n_group = 1: no group step
    w       = scores[idx]; w /= sum(w) + 1e-20         # norm_topk_prob
    w      *= routed_scaling_factor                     # 2.5
    routed  = sum_k w_k * down_k(relu(up_k(x))^2)      # NON-gated squared-ReLU experts
    out     = routed + down_s(relu(up_s(x))^2)         # 1 shared expert, unweighted

Expert compute is written as two dense matmuls over ALL experts with a masked
affinity in between:

    h   = relu(x @ W_up_all)^2          # W_up_all  [H, E * I/tp]
    h  *= affinity broadcast per expert  # zero for the 122 unselected experts
    out = h @ W_down_all                 # W_down_all [E * I/tp, H]

``down_k(h) * w_k == down_k(h * w_k)`` (linearity), so this equals the routed
sum exactly in real arithmetic. It computes all 128 experts (21x the FLOPs of
top-6) but has static shapes, no gather/scatter and no per-expert Python loop,
which is the compile-safe baseline. Cost: every decode step reads ALL expert
weights (about 14.7 GB per rank over 23 layers at TP=4, so roughly 20 ms/step at
~0.7 TB/s per logical core), where a top-k kernel at batch 1 would read the 6
selected experts only (about 9x less). The gap shrinks with batch size (at batch
32 most experts are hit). A kernel replaces it later (see README, "MoE kernels").

TP: every expert's intermediate dim is split across ranks (1856 / 4 = 464), as
is the shared expert's (3712 / 4 = 928); the router is replicated.

Router: hand-rolled. ``NF.router``'s ``router_bias`` is added to the logits
before the activation, and its ``activation`` result is used for both the
selection and the weights; it has no selection-only bias
(``RouterComputationOrder`` offers only pre-norm -> linear -> {topk, act}
orderings). Top-k uses ``vllm_neuron.functional.topk.topk`` (NKI where it fits,
else ``torch.topk``).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.functional.topk import topk
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import NemotronHConfig
from .mamba import SEQUENCE_PARALLEL


# Stored layout of the two stacked expert weights (T09, optimization step 1).
# The T08 decode profile shows a compiler transpose (private_nkl/transpose.py) that reads and
# writes one whole stacked expert weight (319 MB per rank) per MoE layer per decode step.
# ``NEMOTRON_H_MOE_WT`` picks which stacked weights are kept in the checkpoint's own
# ``[out, in]`` orientation (``F.linear`` style, matmul against ``.T``) instead of ``[in, out]``:
#   ``none`` (baseline-v1): up ``[H, E*I]``, down ``[E*I, H]``
#   ``up`` (default since T09): up ``[E*I, H]`` (out, in)    ``down``: down ``[H, E*I]`` (out, in)
#   ``both``: both in ``[out, in]``
# The math is identical; only the HBM layout the compiler sees changes. Measured (T09, full model,
# TP=4): ``up`` removes the transpose, decode at batch 1 71.6 -> 32.3 ms/token; ``down`` and
# ``both`` are slower than ``up`` on the 4-layer checkpoint (c1 TPOT 9.48 / 8.25 vs 6.60 ms).
MOE_WT = os.environ.get("NEMOTRON_H_MOE_WT", "up")
if MOE_WT not in ("none", "up", "down", "both"):
    raise ValueError(f"NEMOTRON_H_MOE_WT must be 'none', 'up', 'down' or 'both', got {MOE_WT!r}")
UP_OUT_IN = MOE_WT in ("up", "both")
DOWN_OUT_IN = MOE_WT in ("down", "both")

# Top-k expert decode path (T10, optimization step 2).
# ``NEMOTRON_H_MOE_TOPK_MAX_T``: decode graphs with at most this many tokens compute only the
# top-k experts per token. The selected experts' up and down matrices are copied out of the
# stacked weights with one DMA each (``kernels/expert_gather.py``) and multiplied in torch.
# Graphs with more tokens (and prefill) keep the dense all-experts path. ``0`` turns it off.
# Default 1 (bucket 1 only) with MOE_WT=up, else 0. 4-layer c1 TPOT 3.26 -> 2.54 ms; c8 TPOT
# with it at bucket 8 is 8.18 vs 6.65 ms dense, so bucket 8 stays dense (T10).
# Per token this moves 3x the selected weights (gather read + write, matmul read: ~90 MB per
# layer at k=6), against 0.64 GB for the dense path, so it pays only at small batch; at batch 8
# (48 expert copies) it would already be ~0.7 GB. Needs MOE_WT=up (both weights [E, I, H]).
MOE_TOPK_MAX_T = int(os.environ.get("NEMOTRON_H_MOE_TOPK_MAX_T", "1" if UP_OUT_IN and not DOWN_OUT_IN else "0"))
if MOE_TOPK_MAX_T and (not UP_OUT_IN or DOWN_OUT_IN):
    raise ValueError("NEMOTRON_H_MOE_TOPK_MAX_T > 0 needs NEMOTRON_H_MOE_WT=up")

# Fused top-k decode kernel (T14, opt4): ``NEMOTRON_H_MOE_TOPK_FUSED=1`` replaces gather + torch
# matmuls in the top-k decode path with ``kernels/moe_topk.py`` (selected experts streamed into
# SBUF, up projection as Vector Engine dot products, down projection on the Tensor Engine).
# Unit test, 8 MoE layers at batch 1, TP=4 rank-0 shapes (results/trn2/opt4/unit/): dense 1.13,
# T10 gather 0.755, gather via SBUF row tiles 0.409, fused 0.289 ms per layer. Full model (T14):
# batch-1 decode step 19.5 -> 9.4 ms (MoE 0.68 -> 0.23 ms per layer), c1 TPOT -51..-53% on one
# trn2 device, gate passes. Default 1; ``0`` gives the T10 gather path (``NEMOTRON_H_EXPERT_GATHER_MODE``).
MOE_TOPK_FUSED = os.environ.get("NEMOTRON_H_MOE_TOPK_FUSED", "1") == "1"

# NKI Library MoE TKG for the top-k decode path (T09b): ``NEMOTRON_H_MOE_TKG=1`` runs graphs with
# ``T <= MOE_TOPK_MAX_T`` through nkilib ``moe_tkg`` (selective-expert mode, skip_gate_proj +
# SquaredReLU, see ``kernels/moe_tkg_sq.py``) instead of ``kernels/moe_topk.py``. It needs the
# routed up weights a second time, ``[E, H, 1, I]`` (in, out): +319 MB per MoE layer per rank
# (weights 14.74 -> 21.58 GiB per rank), which does not fit next to the S/L profiles' KV caches
# and graphs; it fits in the one-request latency profile (bench/bringup/profiles/C1.sh, KV cap 0.01).
# Unit, 8 layers at batch 1: 0.309 (moe_topk) -> 0.231 ms per layer. Full model, C1 profile on one
# trn2 device: c1 TPOT 9.0-9.4 -> 6.6-7.1 ms on the four c=1 Must rows, gate passes (T09b,
# results/trn2/opt-tkg/). Default 0.
MOE_TKG = os.environ.get("NEMOTRON_H_MOE_TKG", "0") == "1"
if MOE_TKG and not MOE_TOPK_MAX_T:
    raise ValueError("NEMOTRON_H_MOE_TKG=1 needs NEMOTRON_H_MOE_TOPK_MAX_T > 0")

# Routed prefill MoE (A07): ``NEMOTRON_H_MOE_PREFILL_KERNEL=1`` computes prefill graphs with the
# top-k experts only. The (token, expert) pairs are sorted into expert tiles of 128 slots
# (static-shape dispatch in torch) and ``kernels/moe_prefill.py`` runs an expert-major grouped GEMM
# with a dynamic tile count per expert; the two cores of a logical core take alternate experts
# (``NEMOTRON_H_MOE_PREFILL_SPLIT=e``). Unit (T=2048, TP=4 rank shapes): 9.99 -> 6.74 ms per MoE layer.
# Full model, profile S on one trn2 device: gate passes, TTFT p50 -16..-19% (agent_8k c1 1,386 ->
# 1,116 ms) (A07, results/kernels/moe-prefill/README.md). Main lane (T12b, optimization step 8, device 0):
# gate passes, prefill chunk 330 -> 267 ms, TTFT -15..-19% on every Must row. Default 1 with the default
# NEMOTRON_H_MOE_WT=up layout (0 otherwise, since the kernel needs it).
_MOE_PREFILL_DEFAULT = "1" if (UP_OUT_IN and not DOWN_OUT_IN) else "0"
MOE_PREFILL_KERNEL = os.environ.get("NEMOTRON_H_MOE_PREFILL_KERNEL", _MOE_PREFILL_DEFAULT) == "1"
if MOE_PREFILL_KERNEL and (not UP_OUT_IN or DOWN_OUT_IN):
    raise ValueError("NEMOTRON_H_MOE_PREFILL_KERNEL=1 needs NEMOTRON_H_MOE_WT=up")

# FP8 routed-expert weights (R2-05 variant, default off): ``NEMOTRON_H_FP8=1`` quantizes the routed experts'
# up and down weights at load time from the BF16 checkpoint. Activations stay bf16, accumulation fp32.
# Format: torch ``float8_e4m3fn`` storage with every value inside +-240, which is bit-identical to the trn2
# Tensor/Vector Engine format ``nl.float8_e4m3`` (IEEE-style e4m3, bias 7, max 240; NKI maps
# torch.float8_e4m3fn to it on trn2 when UNSAFE_FP8FNCAST=1, which this flag sets).
# Scales: one fp32 scale per row of the stored ``[E * I, H]`` matrices (amax / 240, round to nearest even):
#   up   ``[E * I, H]`` (out, in): per output channel of up_proj;
#   down ``[E * I, H]`` (in, out): per input channel of down_proj (= the same intermediate channel i).
# Both scales index the intermediate channel i, and relu(s * u)^2 = s^2 relu(u)^2 for s > 0, so the expert is
#   down_e(relu(up_e x)^2) = sum_i relu((Wq_up x)_i)^2 * c_i * Wq_down[i, :],  c_i = s_up_i^2 * s_down_i,
# and one fp32 vector ``experts_c`` [E * I + 128] (128 pad rows for the kernels' tile over-reads) carries all scaling.
# It is applied to the activation, never to a weight tile, in all three paths (batch-1 kernel, prefill kernel,
# dense decode). Per-output-channel scales on down would not fold this way: the batch-1 kernel accumulates the 6
# experts in one PSUM tile and the dense path sums all experts in one matmul. Router, norms, embeddings, LM head,
# attention, Mamba and the shared expert stay bf16 (fp32 where they were).
FP8 = os.environ.get("NEMOTRON_H_FP8", "0") == "1"
# Batch-1 FP8 kernel: "dve" (default, kernels/moe_topk.py with fp8 tiles) or "t" (kernels/moe_fp8.moe_fp8_topk).
FP8_TOPK = os.environ.get("NEMOTRON_H_FP8_TOPK", "dve")
# FP8 shared expert in decode (R2-05 step 2): ``NEMOTRON_H_FP8_SHARED=1`` appends the shared expert to the stacked FP8
# weights as ``shared_inter / I`` = 2 pseudo-experts (rows E*I.., routing weight 1; exact, since relu^2 is per
# channel), so the decode kernels read it in FP8 and the bf16 torch matmuls go away. Prefill keeps the bf16 shared
# expert (dense torch matmul; the bf16 copy stays in memory, 10 MB per layer per rank).
FP8_SHARED = FP8 and os.environ.get("NEMOTRON_H_FP8_SHARED", "0") == "1"
# FP8 routed batched decode (R2-05b): with NEMOTRON_H_FP8=1, decode buckets in NEMOTRON_H_MOE_ROUTED_BUCKETS use the FP8
# routed kernel (kernels/moe_decode_routed.moe_routed_fp8_kernel) when NEMOTRON_H_FP8_ROUTED=1, else the FP8 dense
# kernel (kernels/moe_fp8.py, the R2-05 merge behaviour).
FP8_ROUTED = os.environ.get("NEMOTRON_H_FP8_ROUTED", "1") == "1"
FP8_MAX = 240.0
FP8_DTYPE = torch.float8_e4m3fn
FP8_CPAD = 128
if FP8:
    os.environ.setdefault("UNSAFE_FP8FNCAST", "1")
    if not (UP_OUT_IN and not DOWN_OUT_IN):
        raise ValueError("NEMOTRON_H_FP8=1 needs NEMOTRON_H_MOE_WT=up")
    if MOE_TKG:
        raise ValueError("NEMOTRON_H_FP8=1 does not support NEMOTRON_H_MOE_TKG=1")
    if MOE_TOPK_MAX_T and not MOE_TOPK_FUSED:
        raise ValueError("NEMOTRON_H_FP8=1 needs NEMOTRON_H_MOE_TOPK_FUSED=1 (the T10 gather path is bf16 only)")


def fp8_quant_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[R, C]`` -> (``[R, C]`` float8_e4m3fn within +-240, ``[R]`` float32 scale = row amax / 240)."""
    wf = w.float()
    s = (wf.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-30)
    q = (wf / s.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, s


def _stack_rows(slices: list, rank: int, shard: int, kind: str) -> torch.Tensor:
    """Stacked ``[E * shard, H]`` bf16/f32 rows as the bf16 loader builds them (up out-in, down in-out)."""
    start, stop = rank * shard, (rank + 1) * shard
    if kind == "up":
        return torch.cat([s[start:stop, :] for s in slices], dim=0)
    return torch.cat([s[:, start:stop] for s in slices], dim=1).T


def _shared_rows(sl, rank: int, sshard: int, kind: str) -> torch.Tensor:
    """Shared expert rows ``[Is, H]`` of this rank: up (out, in) rows, down (in, out) = checkpoint columns."""
    a, b = rank * sshard, (rank + 1) * sshard
    return sl[a:b, :] if kind == "up" else sl[:, a:b].T


def _rows_with_shared(slices: list, rank: int, shard: int, kind: str, num_experts: int, sshard: int):
    w = _stack_rows(slices[:num_experts], rank, shard, kind)
    if len(slices) > num_experts:  # FP8_SHARED: + the shared expert's rows (the pseudo-experts E, E+1, ...)
        w = torch.cat([w, _shared_rows(slices[num_experts], rank, sshard, kind)], dim=0)
    return w


def _experts_fp8_loader(shard: int, kind: str, num_experts: int, sshard: int = 0) -> SafetensorsWeightLoader:
    """``down``: ``[E * I, H]`` (in, out). ``up``: ``[E * I, H]`` (out, in). ``up_t``: the same quantized up,
    stored transposed as ``[E * 128, nh * I]`` (``kernels/moe_fp8.up_t_layout``)."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        k2 = "down" if kind == "down" else "up"
        q = fp8_quant_rows(_rows_with_shared(slices, rank, shard, k2, num_experts, sshard))[0].contiguous()
        if kind == "up_t":
            from .kernels.moe_fp8 import up_t_layout

            q = up_t_layout(q, q.shape[0] // shard, shard)
        return q

    return SafetensorsWeightLoader(transform=transform)


def _experts_c_loader(shard: int, num_experts: int, sshard: int = 0) -> SafetensorsWeightLoader:
    """``c = s_up^2 * s_down`` per intermediate row, from the up keys followed by the down keys (each list with the
    shared expert's key last when FP8_SHARED)."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        half = len(slices) // 2
        s_up = fp8_quant_rows(_rows_with_shared(slices[:half], rank, shard, "up", num_experts, sshard))[1]
        s_dn = fp8_quant_rows(_rows_with_shared(slices[half:], rank, shard, "down", num_experts, sshard))[1]
        c = s_up * s_up * s_dn
        return torch.cat([c, torch.zeros(FP8_CPAD, dtype=torch.float32)]).view(-1, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


# Routed batched decode MoE (R2-01): decode graphs whose token count is in ``NEMOTRON_H_MOE_ROUTED_BUCKETS`` (comma
# list, e.g. "8,16") load only the experts some token of the step picked, each once, and apply it to all tokens
# (``kernels/moe_decode_routed.py``). Other decode graphs keep the dense all-experts path (or the batch-1 top-k kernel).
# Needs NEMOTRON_H_MOE_WT=up. Default "8,16" with MOE_WT=up (R2-01: gate passes, device-0 long_32k c16 +4.0% tok/s; "" = off).
MOE_ROUTED_BUCKETS = frozenset(int(v) for v in os.environ.get("NEMOTRON_H_MOE_ROUTED_BUCKETS", "8,16" if (UP_OUT_IN and not DOWN_OUT_IN) else "").split(",") if v.strip())
if MOE_ROUTED_BUCKETS and (not UP_OUT_IN or DOWN_OUT_IN):
    raise ValueError("NEMOTRON_H_MOE_ROUTED_BUCKETS needs NEMOTRON_H_MOE_WT=up")


def _experts_loader(
    shard: int, num_shards: int, *, kind: str, out_in: bool = False
) -> SafetensorsWeightLoader:
    """Stack E per-expert checkpoint tensors into one matrix.

    ``up``: each ``[I, H]`` -> rows ``[r*shard, (r+1)*shard)`` -> concat -> ``[E*shard, H]``.
    ``down``: each ``[H, I]`` -> cols ``[r*shard, (r+1)*shard)`` -> concat -> ``[H, E*shard]``.
    That is the ``[out, in]`` orientation; with ``out_in=False`` it is transposed to ``[in, out]``.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        start, stop = rank * shard, (rank + 1) * shard
        if kind == "up":
            w = torch.cat([s[start:stop, :] for s in slices], dim=0)
        else:
            w = torch.cat([s[:, start:stop] for s in slices], dim=1)
        return w.contiguous() if out_in else w.T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _experts_up_tkg_loader(shard: int) -> SafetensorsWeightLoader:
    """Routed up weights as ``[E, H, 1, I]`` (in, out) for nkilib ``moe_tkg`` (T09b)."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        start, stop = rank * shard, (rank + 1) * shard
        return torch.stack([s[start:stop, :].T for s in slices], dim=0).unsqueeze(2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def relu2(x: torch.Tensor) -> torch.Tensor:
    """HF ``ACT2FN['relu2']``: ``relu(x) ** 2``."""
    r = F.relu(x)
    return r * r


class NemotronHMoE(nn.Module):
    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.tp_group = get_tp_group()
        self.world_size = ws = self.tp_group.world_size
        self.hidden_size = H = config.hidden_size
        self.num_experts = E = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        for name, value in (
            ("moe_intermediate_size", config.moe_intermediate_size),
            ("moe_shared_expert_intermediate_size", config.moe_shared_expert_intermediate_size),
        ):
            if value % ws:
                raise ValueError(f"{name}={value} must be divisible by tp_size={ws}")
        self.inter = config.moe_intermediate_size // ws
        self.shared_inter = config.moe_shared_expert_intermediate_size // ws

        # Router replicated, float32 (HF computes it in float32).
        self.router_weight = nn.Parameter(torch.empty(H, E, dtype=torch.float32))
        self.e_score_correction_bias = nn.Parameter(torch.empty(E, dtype=torch.float32))
        EI = E * self.inter
        up_shape = (EI, H) if UP_OUT_IN else (H, EI)
        down_shape = (H, EI) if DOWN_OUT_IN else (EI, H)
        self.fp8 = FP8
        wdt = FP8_DTYPE if self.fp8 else self.dtype
        if self.fp8 and H % 128:
            raise ValueError(f"NEMOTRON_H_FP8=1 needs hidden_size % 128 == 0, got {H}")
        self.fp8_shared = FP8_SHARED and self.fp8
        self.n_shared_pe = 0  # pseudo-experts holding the shared expert in the FP8 stacks
        if self.fp8_shared:
            if self.shared_inter % self.inter:
                raise ValueError("NEMOTRON_H_FP8_SHARED=1 needs shared intermediate % expert intermediate == 0")
            self.n_shared_pe = self.shared_inter // self.inter
            EI = (E + self.n_shared_pe) * self.inter
            up_shape = down_shape = (EI, H)
        self.experts_up_weight = nn.Parameter(torch.empty(*up_shape, dtype=wdt), requires_grad=not self.fp8)
        self.experts_down_weight = nn.Parameter(torch.empty(*down_shape, dtype=wdt), requires_grad=not self.fp8)
        self.shared_up_weight = nn.Parameter(torch.empty(H, self.shared_inter, dtype=self.dtype))
        self.shared_down_weight = nn.Parameter(torch.empty(self.shared_inter, H, dtype=self.dtype))

        set_weight_loader(
            self.router_weight, SafetensorsWeightLoader(transform=lambda s, r: s[0][:].T.contiguous())
        )
        if self.fp8:
            # Two FP8 copies of up (same values): [E * I, H] for the batch-1 kernel (Vector Engine dot products
            # over contiguous rows) and transposed [E * 128, nh * I] for the dense decode and prefill kernels
            # (one DMA per expert, no on-chip transposes). Each decode step reads one of them.
            E2 = E + self.n_shared_pe
            self.experts_up_t = nn.Parameter(torch.empty(E2 * 128, (H // 128) * self.inter, dtype=wdt),
                                             requires_grad=False)
            self.experts_c = nn.Parameter(torch.empty(EI + FP8_CPAD, 1, dtype=torch.float32), requires_grad=False)
            ssh = self.shared_inter
            set_weight_loader(self.experts_up_weight, _experts_fp8_loader(self.inter, "up", E, ssh))
            set_weight_loader(self.experts_up_t, _experts_fp8_loader(self.inter, "up_t", E, ssh))
            set_weight_loader(self.experts_down_weight, _experts_fp8_loader(self.inter, "down", E, ssh))
            set_weight_loader(self.experts_c, _experts_c_loader(self.inter, E, ssh))
        else:
            set_weight_loader(
                self.experts_up_weight, _experts_loader(self.inter, ws, kind="up", out_in=UP_OUT_IN)
            )
            set_weight_loader(
                self.experts_down_weight,
                _experts_loader(self.inter, ws, kind="down", out_in=DOWN_OUT_IN),
            )
        if MOE_TKG:
            self.experts_up_tkg = nn.Parameter(torch.empty(E, H, 1, self.inter, dtype=self.dtype))
            set_weight_loader(self.experts_up_tkg, _experts_up_tkg_loader(self.inter))
        set_weight_loader(
            self.shared_up_weight,
            sharding_weight_loader(
                shard_dim=1, shard_size=self.shared_inter, num_shards=ws, is_storage_transposed=True
            ),
        )
        set_weight_loader(
            self.shared_down_weight,
            sharding_weight_loader(
                shard_dim=0, shard_size=self.shared_inter, num_shards=ws, is_storage_transposed=True
            ),
        )

    def route(self, x: torch.Tensor, return_topk: bool = False):
        """``[T, H] -> [T, E]`` float32 affinities, zero except the top-k experts.

        With ``return_topk`` also returns ``(idx [T, k], w [T, k])``: the selected expert ids and
        their affinities (the same values as in the dense ``[T, E]`` row, picked by a one-hot
        reduction rather than a data-dependent gather).
        """
        logits = x.float() @ self.router_weight.float()
        scores = torch.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.float()
        from .kernels.router_top8 import maybe_router_topk

        idx = maybe_router_topk(choice, self.top_k)
        if idx is None:
            _, idx = topk(choice, k=self.top_k, dim=-1, gather_dim=-1)
        experts = torch.arange(self.num_experts, device=x.device, dtype=idx.dtype)
        onehot = idx.unsqueeze(-1) == experts.view(1, 1, -1)  # [T, k, E]
        selected = onehot.any(dim=1)
        weights = scores * selected.to(scores.dtype)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        if not return_topk:
            return weights
        w_k = (onehot.to(weights.dtype) * weights.unsqueeze(1)).sum(dim=-1)  # [T, k]
        return weights, idx, w_k

    def _routed_topk(self, x: torch.Tensor) -> torch.Tensor:
        """Routed experts on the top-k experts only (decode, ``T <= MOE_TOPK_MAX_T``)."""
        t, k, inter, H = x.shape[0], self.top_k, self.inter, self.hidden_size
        from vllm_neuron.utils.neuron_utils import can_run_kernel
        from .kernels import router_top8 as _rt

        if (_rt.FUSED and t == 1 and MOE_TOPK_FUSED and not MOE_TKG and not self.fp8 and k <= 8
                and can_run_kernel(x)):
            # R2-10: router + top-k + weights in one kernel, its outputs go straight into the top-k MoE kernel
            from .kernels.moe_topk import moe_topk_decode_off

            off, w = _rt.router_fused(x, self.router_weight, self.e_score_correction_bias, inter, k,
                                      self.routed_scaling_factor, self.norm_topk_prob)
            return moe_topk_decode_off(x, self.experts_up_weight, self.experts_down_weight, off, w, inter)
        aff, idx, w_k = self.route(x, return_topk=True)
        flat = idx.reshape(t * k)
        n_pe = self.num_experts + self.n_shared_pe
        up3 = self.experts_up_weight.view(n_pe, inter, H)  # [E, I, H] (out, in)
        down3 = self.experts_down_weight.view(n_pe, inter, H)  # [E, I, H] (in, out)
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        if MOE_TKG and can_run_kernel(x):
            from .kernels.moe_tkg_sq import moe_tkg_sq_decode

            return moe_tkg_sq_decode(x, self.experts_up_tkg, self.experts_down_weight, aff, idx)
        if self.fp8:
            if self.fp8_shared:  # the shared expert = pseudo-experts E.., weight 1, for every token
                ns = self.n_shared_pe
                sh_idx = torch.arange(self.num_experts, self.num_experts + ns, device=x.device, dtype=idx.dtype)
                idx = torch.cat([idx, sh_idx.view(1, ns).expand(t, ns)], dim=1)
                w_k = torch.cat([w_k, torch.ones((t, ns), dtype=w_k.dtype, device=x.device)], dim=1)
                aff = torch.cat([aff, torch.ones((t, ns), dtype=aff.dtype, device=x.device)], dim=1)
            if can_run_kernel(x):
                if FP8_TOPK == "t":  # weights-stationary kernel on the transposed up (unit: slower at batch 1)
                    from .kernels.moe_fp8 import moe_fp8_topk

                    return moe_fp8_topk(x, self.experts_up_t, self.experts_down_weight, self.experts_c, idx, w_k,
                                        inter)
                from .kernels.moe_topk import moe_topk_decode

                return moe_topk_decode(x, self.experts_up_weight, self.experts_down_weight, idx, w_k, inter,
                                       c=self.experts_c)
            return self._routed_dense_fp8(x, aff)  # CPU: dense FP8 math with the routing mask
        if MOE_TOPK_FUSED and can_run_kernel(x):
            from .kernels.moe_topk import moe_topk_decode

            return moe_topk_decode(x, self.experts_up_weight, self.experts_down_weight, idx, w_k, inter)
        if can_run_kernel(x):
            from .kernels.expert_gather import gather_experts

            up_sel, down_sel = gather_experts(up3, down3, flat)
        else:
            up_sel, down_sel = up3.index_select(0, flat), down3.index_select(0, flat)
        if t == 1:
            h = relu2(F.linear(x, up_sel.reshape(k * inter, H)))  # [1, k * I]
            h = (h.view(k, inter) * w_k.view(k, 1).to(h.dtype)).view(1, k * inter)
            return h @ down_sel.reshape(k * inter, H)
        up_sel = up_sel.view(t, k * inter, H)
        h = relu2(torch.bmm(up_sel, x.unsqueeze(-1)).squeeze(-1))  # [t, k * I]
        h = (h.view(t, k, inter) * w_k.unsqueeze(-1).to(h.dtype)).view(t, 1, k * inter)
        return torch.bmm(h, down_sel.view(t, k * inter, H)).squeeze(1)

    def _routed_dense_fp8(self, x: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
        """Dense all-experts routed sum from the FP8 weights (``affinity`` ``[T, E]``, zero off the top-k).

        On device: ``kernels/moe_fp8.moe_fp8_dense`` (decode, T <= 128). Otherwise (CPU reference) torch:
        ``relu(Wq_up x)^2 * (affinity * c)`` then ``@ Wq_down`` with the weights cast to bf16.
        """
        t, inter = x.shape[0], self.inter
        E = affinity.shape[1]  # num_experts (+ shared pseudo-experts with weight 1 when FP8_SHARED)
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        from .kernels.moe_fp8 import moe_fp8_dense

        if can_run_kernel(x) and t <= 128:
            return moe_fp8_dense(x, self.experts_up_t, self.experts_down_weight, self.experts_c, affinity, inter)
        u = F.relu(F.linear(x, self.experts_up_weight[: E * inter].to(x.dtype)))  # [T, E * I], unscaled
        coef = affinity.view(t, E, 1) * self.experts_c[: E * inter, 0].view(1, E, inter)  # fp32
        h = (u * u) * coef.view(t, E * inter).to(x.dtype)
        return h @ self.experts_down_weight[: E * inter].to(x.dtype)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        x = hidden_states.to(self.dtype)
        if is_prefill and self.world_size > 1 and SEQUENCE_PARALLEL:
            x = self.tp_group.all_gather(x, dim=0)
        t = x.shape[0]

        if not is_prefill and 0 < t <= MOE_TOPK_MAX_T:
            out = self._routed_topk(x)
            if not self.fp8_shared:
                from .kernels.decode_gemv import maybe_gemv

                out = out + maybe_gemv(relu2(maybe_gemv(x, self.shared_up_weight, "shup")), self.shared_down_weight, "shdn")
            if self.world_size > 1:
                out = self.tp_group.all_reduce(out)
            return out

        from vllm_neuron.utils.neuron_utils import can_run_kernel

        # NEMOTRON_H_FP8: the routed buckets use the FP8 routed kernel (NEMOTRON_H_FP8_ROUTED=1, R2-05b) or the FP8
        # dense kernel below (NEMOTRON_H_FP8_ROUTED=0). FP8_SHARED keeps the dense kernel (shared pseudo-experts).
        if (not is_prefill and t in MOE_ROUTED_BUCKETS and t > MOE_TOPK_MAX_T and self.fp8 and FP8_ROUTED
                and not self.fp8_shared and can_run_kernel(x)):
            from .kernels.moe_decode_routed import moe_routed_decode_fp8

            affinity = self.route(x)
            sel = (affinity > 0).any(dim=0).to(torch.float32)
            out = moe_routed_decode_fp8(x, self.experts_up_t, self.experts_down_weight, self.experts_c, affinity, sel,
                                        self.inter)
            out = out + relu2(x @ self.shared_up_weight) @ self.shared_down_weight
            if self.world_size > 1:
                out = self.tp_group.all_reduce(out)
            return out
        if (not is_prefill and t in MOE_ROUTED_BUCKETS and t > MOE_TOPK_MAX_T and not self.fp8
                and can_run_kernel(x)):
            from .kernels.moe_decode_routed import moe_routed_decode

            affinity = self.route(x)  # [T, E] float32, 0 for unpicked experts
            sel = (affinity > 0).any(dim=0).to(torch.float32)  # [E] experts with a nonzero weight in this step
            out = moe_routed_decode(x, self.experts_up_weight, self.experts_down_weight, affinity, sel, self.inter)
            out = out + relu2(x @ self.shared_up_weight) @ self.shared_down_weight
            if self.world_size > 1:
                out = self.tp_group.all_reduce(out)
            return out

        if is_prefill and MOE_PREFILL_KERNEL and can_run_kernel(x):
            from .kernels.moe_prefill import moe_prefill_routed
            from .kernels import moe_prefill as _mp

            # Debug (T12e): NEMOTRON_H_MOE_PREFILL_TILE0_LAYERS="6,8" = TILE0 only in these layers (read at trace time)
            _tl = os.environ.get("NEMOTRON_H_MOE_PREFILL_TILE0_LAYERS", "")
            if _tl:
                _mp.TILE0 = str(self.layer_idx) in _tl.split(",")

            # shared expert first in program order (the scheduler still runs it after the kernel; forcing it
            # earlier through a data dependency did not change the time, A07 attempt 2)
            shared = relu2(x @ self.shared_up_weight) @ self.shared_down_weight
            _, idx, w_k = self.route(x, return_topk=True)
            bf16_up_t = getattr(self, "bf16_up_t", False)  # R2-03b unit experiments: BF16 up stored [E * 128, nh * I]
            out = moe_prefill_routed(x, self.experts_up_t if (self.fp8 or bf16_up_t) else self.experts_up_weight,
                                     self.experts_down_weight, idx, w_k, self.inter, self.num_experts,
                                     **(dict(c=self.experts_c) if self.fp8 else dict(up_t=True) if bf16_up_t else {}))
            out = out + shared
            if self.world_size > 1:
                if SEQUENCE_PARALLEL:
                    out = self.tp_group.reduce_scatter(out, dim=0).contiguous()
                else:
                    out = self.tp_group.all_reduce(out)
            return out

        affinity = self.route(x)  # [T, E] float32
        if self.fp8:
            if self.fp8_shared and not is_prefill:
                ns = self.n_shared_pe
                aff2 = torch.cat([affinity, torch.ones((t, ns), dtype=affinity.dtype, device=x.device)], dim=1)
                out = self._routed_dense_fp8(x, aff2)  # includes the shared expert
            else:
                out = self._routed_dense_fp8(x, affinity)
        else:
            w_up = self.experts_up_weight
            h = relu2(F.linear(x, w_up) if UP_OUT_IN else x @ w_up)  # [T, E * I]
            h = (h.view(t, self.num_experts, self.inter) * affinity.unsqueeze(-1).to(h.dtype)).view(
                t, self.num_experts * self.inter
            )
            w_down = self.experts_down_weight
            out = F.linear(h, w_down) if DOWN_OUT_IN else h @ w_down
        if not (self.fp8_shared and not is_prefill):
            out = out + relu2(x @ self.shared_up_weight) @ self.shared_down_weight

        if self.world_size > 1:
            if is_prefill and SEQUENCE_PARALLEL:
                out = self.tp_group.reduce_scatter(out, dim=0).contiguous()
            else:
                out = self.tp_group.all_reduce(out)
        return out
