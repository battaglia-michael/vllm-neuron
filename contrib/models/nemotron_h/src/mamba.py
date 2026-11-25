# SPDX-License-Identifier: Apache-2.0
"""Mamba-2 mixer for NemotronH (23 of the 30B-A3B's 52 layers).

Math copied from ``transformers``' ``NemotronHMamba2Mixer`` (torch path):

    zxbcdt = in_proj(x)                     # [z | xBC | dt]
    xBC    = silu(causal_conv1d(xBC) + b)   # depthwise, kernel 4
    x, B, C = split(xBC)                    # B, C: n_groups x d_state
    dt     = softplus(dt + dt_bias)         # no clamp by default (DT_CLAMP below)
    A      = -exp(A_log)
    h_t    = exp(dt_t A) h_{t-1} + dt_t B_t x_t      (per head, state [head_dim, d_state])
    y_t    = C_t . h_t + D x_t
    y      = RMSNormGated(y, z)             # y * silu(z), then RMS over groups of
                                            # intermediate / n_groups channels, * weight
    out    = out_proj(y)

Prefill runs the chunked SSD scan (chunk 128, HF ``mamba2_chunk_scan``); decode
runs one recurrence step (HF ``mamba2_selective_state_update``). Stock HF
clamps ``dt`` to ``time_step_min`` in the chunked scan but not in its cached
single step; NVIDIA's reference code and vLLM GPU never clamp, and neither does
this port by default (``DT_CLAMP`` below; ``NEMOTRON_H_DT_CLAMP=hf`` clamps in
both paths).

TP: heads, groups and the gated-norm groups are split across ranks (64 heads /
8 groups -> 16 heads / 2 groups per rank at TP=4). Every quantity the recurrence
touches is per head, B/C are per group and a group's heads stay on one rank, and
the gated norm's groups (512 channels = 8 heads) do too, so the only collective
is ``out_proj``'s row-parallel reduction.

State: one page per request per layer, bound by ``bind_state`` as two float32
halves (``vllm_neuron.model.state_cache``), holding
``[conv window (K-1) x conv_dim | ssm state heads x head_dim x d_state | pad]``
(``NEMOTRON_H_STATE_ALIGN=1``: pad between the SSM groups so none straddles the halves).
State plumbing is the Qwen3.5 contrib's (``deltanet.py``), including its
NCC_IDLO901 rule: the prefill state write uses the raw block id.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import NemotronHConfig

logger = logging.getLogger(__name__)

SEQUENCE_PARALLEL = os.environ.get("VLLM_NEURON_NEMOTRON_H_DISABLE_SP") != "1"

# dt lower clamp (decided in T03b, see README "dt clamp").
#   "none" (default) never clamp softplus(dt + dt_bias). This is the model's own
#          config (time_step_limit absent -> (0, inf)), NVIDIA's remote modeling
#          code, Megatron-LM's MambaMixer, mamba_ssm's Mamba2 default and vLLM GPU
#          (the H100 baseline: mamba_mixer2.py dt_limit=(0.0, inf)).
#   "hf"   clamp to time_step_min (1e-3) in prefill AND decode. Stock transformers'
#          NemotronHMamba2Mixer hard-codes time_step_limit = (time_step_min, inf)
#          in its chunked scan (every full-sequence forward) but not in its cached
#          single step; "hf" makes decode equal that full-sequence forward.
DT_CLAMP = os.environ.get("NEMOTRON_H_DT_CLAMP", "none")
if DT_CLAMP not in ("hf", "none"):
    raise ValueError(f"NEMOTRON_H_DT_CLAMP must be 'hf' or 'none', got {DT_CLAMP!r}")

# Mamba prefill formulation (T12). Default "split" (opt3: full-model prefill chunk
# 897 -> 330 ms on one trn2 device, accuracy gate unchanged).
#   "base" one in_proj matmul + column slices of its output (baseline-v1..opt2), the
#          rank-3 chunked scan (heads x chunks in the batch dim, B/C broadcast from
#          groups to heads, inter-chunk state carried by a python loop over chunks).
#   "grp"  B/C stay per group (C.B^T and the state matmuls are batched per group
#          with the group's 8 heads side by side), and the inter-chunk recurrence
#          is one masked decay matmul over chunks instead of the loop.
#   "nomask" do not multiply xBC by the real-token mask before the conv. Bucket
#          padding is appended after the last real token and the conv is causal,
#          so pad rows never reach a real row; dt stays 0 at pads, so the scan
#          and the written state are unchanged.
#   "split" three in_proj matmuls (z, xBC, dt) instead of one plus slices.
# Comma-separated, e.g. "nomask,split".
PREFILL_IMPL = os.environ.get("NEMOTRON_H_MAMBA_PREFILL", "split")
_PREFILL_OPTS = {"base", "grp", "nomask", "split"}


def prefill_opts() -> set[str]:
    opts = set(PREFILL_IMPL.split(","))
    if not opts <= _PREFILL_OPTS:
        raise ValueError(f"NEMOTRON_H_MAMBA_PREFILL must be a comma list of {sorted(_PREFILL_OPTS)}, got {PREFILL_IMPL!r}")
    return opts


prefill_opts()

# T-chunk4096: run the prefill body (in_proj .. out_proj) on sub-chunks of this many tokens when the prefill chunk is a
# larger multiple of it (e.g. 2048 with --max-num-batched-tokens 4096), carrying the conv window and SSM state between them.
# The torch Mamba prefill graph is 11x slower per layer at T = 4096 than at 2048 (30.95 vs 2.69 ms, full-model trace).
# 0 = off (one pass). Default 1024 (T-chunk4096 optimization step 15, with --max-num-batched-tokens 4096 and prefill buckets
# [2048, 4096] in the S/L profiles): 4,096-token chunk 935 -> 392 ms (Mamba 5.42 ms/layer; 2 x 2,048 chunks: 432 ms),
# 2,048-token chunk 216 -> 213 ms; gate passes; device-0 TTFT -9..-10% and c32/c128 TPOT -4..-6% on agent_8k.
PREFILL_SUBCHUNK = int(os.environ.get("NEMOTRON_H_MAMBA_PREFILL_SUBCHUNK", "1024"))
# What runs per sub-chunk: "body" (in_proj .. out_proj), "mid" (conv .. gated norm; in_proj and out_proj on the whole
# chunk), "scan" (only the SSD scan).
PREFILL_SUBCHUNK_SCOPE = os.environ.get("NEMOTRON_H_MAMBA_PREFILL_SUBCHUNK_SCOPE", "body")

# R2-03: fused Mamba-2 prefill kernel (kernels/ssd_prefill.py, A06's kernel 2 v2, split-input variant): conv + bias +
# SiLU, softplus(dt + dt_bias), chunked SSD scan, D x and the gated RMSNorm in one NKI kernel between the three split
# in_proj matmuls and out_proj, on the whole prefill chunk (no 1,024-token sub-chunks). Default "1" since R2-03 (optimization
# step R2-C: 4,096-token chunk 372.9 -> 297.9 ms, Mamba 125.1 -> 57.6 ms; gate passes); "0" = torch on 1,024-token sub-chunks.
# Applies only with the "split" in_proj, DT_CLAMP "none", a conv bias and token counts that are multiples of 128.
PREFILL_KERNEL = os.environ.get("NEMOTRON_H_MAMBA_PREFILL_KERNEL", "1")
if PREFILL_KERNEL not in ("0", "1"):
    raise ValueError(f"NEMOTRON_H_MAMBA_PREFILL_KERNEL must be '0' or '1', got {PREFILL_KERNEL!r}")
# Token counts at or above this use the kernel (smaller chunks keep the torch path). Default 1 = every chunk.
# R2-03p2: the fused prefill kernel also runs with the KV-aliased shared pool (prefix caching, profile P). R2-03 kept P on
# the torch path because the server hung in its batch-8 decode warmup; the trigger was the warmup prefill's out-of-range
# (dropped) state write, see NEMOTRON_H_PREFILL_NULL_WRITE. "0" = the R2-03 guard (torch Mamba prefill in the shared pool).
PREFILL_KERNEL_SHARED = os.environ.get("NEMOTRON_H_MAMBA_PREFILL_KERNEL_SHARED", "1")
# R2-03p2: shared pool, a prefill of the null page (only the warmup's dummy block table; a request's state page is never 0):
# "page0" (default) writes page 0, which the runner clears before the decode warmup (VLLM_NEURON_WARMUP_CLEAR_SHARED) and
# again after warmup; "oor" = the T11d out-of-range page (write dropped by oob_mode=skip). With the fused prefill kernel
# in layers 0-9, the oor write made the next batch-8 decode graph hang on device (execution timeout, SEQUENCER_ERROR
# BAD_OPCODE on the second physical core); results/kernels/prefix-cache-mamba-kernel/.
PREFILL_NULL_WRITE = os.environ.get("NEMOTRON_H_PREFILL_NULL_WRITE", "page0")
if PREFILL_NULL_WRITE not in ("page0", "oor"):
    raise ValueError(f"NEMOTRON_H_PREFILL_NULL_WRITE must be 'page0' or 'oor', got {PREFILL_NULL_WRITE!r}")
# R2-03p2 diagnostic: NEMOTRON_H_MAMBA_PREFILL_KERNEL_LAYERS="lo-hi" limits the prefill kernel to layers lo..hi (unset = all).
_pkl = os.environ.get("NEMOTRON_H_MAMBA_PREFILL_KERNEL_LAYERS", "")
PREFILL_KERNEL_LAYERS = frozenset(range(int(_pkl.split("-")[0]), int(_pkl.split("-")[1]) + 1)) if _pkl else frozenset()
PREFILL_KERNEL_MIN_T = int(os.environ.get("NEMOTRON_H_MAMBA_PREFILL_KERNEL_MIN_T", "1"))
# R2-03m: in_proj of the kernel path. "nki" (default R2-03m .. R2-03n) = kernels/ssd_prefill.py mamba_in_proj_kernel
# ([T, H] @ [H, proj] in one NKI matmul kernel, LNC=2 column split, weight slice resident in SBUF), then the proj-input
# fused kernel (no xBC mask: pads follow the real tokens and the conv is causal, so only pad rows change). "torch" = three
# split torch matmuls + the xBC mask, then the split-input kernel (R2-03). In the compiler's repeated module the torch
# in_proj ran 1.5-1.8 ms per layer under the TensorE activity throttle (0.55 ms in module 0); with the kernel every Mamba
# block is 1.5-1.9 ms and the Mamba layers take 38.2 instead of 58.0 ms per 4,096-token chunk (aux-device profile,
# results/kernels/mamba-prefill-r2/).
# "fused" (default since R2-03n) = in_proj inside the fused kernel (kernels/ssd_prefill.py
# mamba_prefill_inproj_fused_kernel), one 128-token chunk ahead of the scan, so its TensorE work can overlap the scan's
# Vector/Scalar work. With the R2-03n scan options (NEMOTRON_H_MAMBA_PREFILL_SCAN=bc,eb,cv) the Mamba layers take 34.3
# instead of 38.2 ms per 4,096-token chunk (aux-device profiles, results/kernels/mamba-prefill-r2b/). "nki" = R2-03m.
PREFILL_INPROJ = os.environ.get("NEMOTRON_H_MAMBA_PREFILL_INPROJ", "fused")
if PREFILL_INPROJ not in ("torch", "nki", "fused"):
    raise ValueError(f"NEMOTRON_H_MAMBA_PREFILL_INPROJ must be 'torch', 'nki' or 'fused', got {PREFILL_INPROJ!r}")
# R2-03m: with INPROJ=nki, run the chunk as this many pieces (in_proj kernel + fused kernel per piece, conv window and
# fp32 state carried), so the next piece's in_proj could overlap the current piece's scan. 1 = one pass (default). 2 did
# not overlap on device (the kernels run back to back; Mamba 39.3 vs 38.2 ms per chunk), so it is kept opt-in only.
PREFILL_KSPLIT = int(os.environ.get("NEMOTRON_H_MAMBA_PREFILL_KSPLIT", "1"))

# Decode graphs with at least this many requests compute z / xBC / dt with three
# in_proj matmuls, like NEMOTRON_H_MAMBA_PREFILL=split (T12). 0 = never.
# One layer on device: batch 1 0.66 -> 1.06 ms (worse), 32: 3.77 -> 2.68, 128: 9.16 -> 3.51.
DECODE_SPLIT_MIN_BATCH = int(os.environ.get("NEMOTRON_H_MAMBA_DECODE_SPLIT_MIN_BATCH", "0"))
# The full-model batch-128 decode graph with the split exceeds neuronx-cc's 5M-instruction
# limit (NCC_ELUR015, 6.19M), so graphs above this size keep the single matmul.
DECODE_SPLIT_MAX_BATCH = int(os.environ.get("NEMOTRON_H_MAMBA_DECODE_SPLIT_MAX_BATCH", "32"))


def align_hf_reference(hf_model) -> int:
    """Make a stock transformers NemotronH model use this port's dt clamp.

    For ``DT_CLAMP == "none"`` sets every mixer's ``time_step_limit`` to (0, inf),
    so HF's chunked scan stops clamping (its single-step decode never clamps).
    For ``"hf"`` leaves the model as is. Returns the number of mixers changed.
    Used by the tests and reference scripts that compare against HF.
    """
    if DT_CLAMP == "hf":
        return 0
    n = 0
    for m in hf_model.modules():
        if hasattr(m, "time_step_limit"):
            m.time_step_limit = (0.0, float("inf"))
            n += 1
    return n

# Decode SSM update + readout y = ssm @ C:
#   "rows"   (default) the whole update on 2-D [reqs*heads*head_dim, d_state] rows,
#            y by broadcast multiply + sum over d_state;
#   "mulsum" 4-D update, y by broadcast multiply + sum;
#   "bmm"    4-D update, y as reqs*heads batched [head_dim, d_state] @ [d_state, 1]
#            matmuls (the original; 9.5x slower than "rows" at batch 128, 4 Mamba
#            layers 349 vs 37 ms, results/bringup/decode-timing/ in the work tree).
DECODE_Y = os.environ.get("NEMOTRON_H_DECODE_Y", "rows")
if DECODE_Y not in ("bmm", "mulsum", "rows"):
    raise ValueError(f"NEMOTRON_H_DECODE_Y must be 'bmm', 'mulsum' or 'rows', got {DECODE_Y!r}")

# Fused NKI decode step (kernels/mamba_decode.py): everything between in_proj and
# out_proj, including the state page read and the in-place write, in one kernel.
#   "1" (default since T15) the kernel on device when it applies (DT_CLAMP "none",
#       <= 128 requests, the [pages, rows, 128] state view); the torch path otherwise.
#       One trn2 device, opt3 -> opt-kernel1: batch-32 decode step 123.5 -> 42.4 ms,
#       batch 1 23.4 -> 19.5 ms; accuracy gate passes.
#   "0" the torch path below (baseline-v1..opt3 decode graphs).
MAMBA_DECODE_KERNEL = os.environ.get("NEMOTRON_H_MAMBA_DECODE_KERNEL", "1")
if MAMBA_DECODE_KERNEL not in ("0", "1"):
    raise ValueError(
        f"NEMOTRON_H_MAMBA_DECODE_KERNEL must be '0' or '1', got {MAMBA_DECODE_KERNEL!r}"
    )


# State page layout (T-mamba-state-align).
#   "0" [conv window | SSM state], filling half A and spilling into half B:
#       at TP=4 the second SSM group (512 rows) has 92 rows in A and 420 in B, so the
#       decode kernel's second core moves it with 2 DMAs per request and direction.
#   "1" aligned: half A = [conv window | as many whole SSM groups as fit | pad],
#       half B = [the remaining groups | pad] (TP=4: A = conv 36 + group 0 512 + pad 92
#       rows, B = group 1 512 + pad 128), so every group is one DMA. Same page size.
#       Default since T-mamba-state-align (falls back to "0" with a warning where the
#       aligned layout does not fit the page, e.g. TP=1).
STATE_ALIGN = os.environ.get("NEMOTRON_H_STATE_ALIGN", "1")
if STATE_ALIGN not in ("0", "1"):
    raise ValueError(f"NEMOTRON_H_STATE_ALIGN must be '0' or '1', got {STATE_ALIGN!r}")
# T11e: whether the aligned layout also applies to the KV-aliased shared pool (prefix caching).
# "1" (default) = follow NEMOTRON_H_STATE_ALIGN; "0" = default layout in the shared pool (T11d).
STATE_ALIGN_SHARED = os.environ.get("NEMOTRON_H_STATE_ALIGN_SHARED", "1")
if STATE_ALIGN_SHARED not in ("0", "1"):
    raise ValueError(f"NEMOTRON_H_STATE_ALIGN_SHARED must be '0' or '1', got {STATE_ALIGN_SHARED!r}")


def segment_shard_loader(
    segment_sizes: tuple[int, ...],
    num_shards: int,
    *,
    is_storage_transposed: bool,
    squeeze_dim: int | None = None,
) -> SafetensorsWeightLoader:
    """Shard each segment of a concatenated checkpoint dim 0, then re-concatenate.

    ``in_proj.weight`` is ``[z | x | B | C | dt]`` along dim 0 and ``conv1d``
    is ``[x | B | C]``; cutting dim 0 contiguously would mix segments.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, "segment_shard_loader() takes a single tensor"
        slice_obj = slices[0]
        ndim = len(slice_obj.get_shape())
        parts, offset = [], 0
        for size in segment_sizes:
            per_rank = size // num_shards
            start = offset + rank * per_rank
            sel = [slice(None)] * ndim
            sel[0] = slice(start, start + per_rank)
            parts.append(slice_obj[tuple(sel)])
            offset += size
        result = torch.cat(parts, dim=0)
        if squeeze_dim is not None:
            result = result.squeeze(squeeze_dim)
        if is_storage_transposed:
            result = result.T
        return result

    return SafetensorsWeightLoader(transform=transform)


class NemotronHMamba2Mixer(nn.Module):
    _STATE_ROW = 128

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.mixer"
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.eps = config.norm_eps
        self.conv_kernel_size = config.conv_kernel
        self.chunk_size = config.chunk_size
        self.time_step_min = config.time_step_min
        self.use_conv_bias = config.use_conv_bias

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        ws = self.world_size
        for name, value in (("mamba_num_heads", config.mamba_num_heads), ("n_groups", config.n_groups)):
            if value % ws:
                raise NotImplementedError(
                    f"{name}={value} is not divisible by tp_size={ws}; group replication "
                    "is not implemented"
                )
        self.num_heads = config.mamba_num_heads // ws
        self.head_dim = config.mamba_head_dim
        self.n_groups = config.n_groups // ws
        self.d_state = config.ssm_state_size
        self.heads_per_group = self.num_heads // self.n_groups
        self.intermediate = self.num_heads * self.head_dim
        self.group_size = config.mamba_intermediate_size // config.n_groups  # gated-norm group
        self.bc_size = self.n_groups * self.d_state
        self.conv_dim = self.intermediate + 2 * self.bc_size
        proj_size = self.intermediate + self.conv_dim + self.num_heads

        full_i = config.mamba_intermediate_size
        full_bc = config.n_groups * config.ssm_state_size
        H = config.hidden_size

        self.in_proj_weight = nn.Parameter(torch.empty(H, proj_size, dtype=self.dtype))
        self.conv1d_weight = nn.Parameter(
            torch.empty(self.conv_dim, self.conv_kernel_size, dtype=self.dtype)
        )
        self.conv1d_bias = (
            nn.Parameter(torch.empty(self.conv_dim, dtype=self.dtype))
            if self.use_conv_bias
            else None
        )
        # float32: they feed softplus/exp that set the decay rate.
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.D = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.norm_weight = nn.Parameter(torch.empty(self.intermediate, dtype=self.dtype))
        self.out_proj_weight = nn.Parameter(torch.empty(self.intermediate, H, dtype=self.dtype))

        set_weight_loader(
            self.in_proj_weight,
            segment_shard_loader(
                (full_i, full_i, full_bc, full_bc, config.mamba_num_heads),
                ws,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.conv1d_weight,
            segment_shard_loader((full_i, full_bc, full_bc), ws, is_storage_transposed=False, squeeze_dim=1),
        )
        if self.conv1d_bias is not None:
            set_weight_loader(
                self.conv1d_bias,
                segment_shard_loader((full_i, full_bc, full_bc), ws, is_storage_transposed=False),
            )
        for param in (self.dt_bias, self.A_log, self.D):
            set_weight_loader(
                param, sharding_weight_loader(shard_dim=0, shard_size=self.num_heads, num_shards=ws)
            )
        set_weight_loader(
            self.norm_weight,
            sharding_weight_loader(shard_dim=0, shard_size=self.intermediate, num_shards=ws),
        )
        set_weight_loader(
            self.out_proj_weight,
            sharding_weight_loader(
                shard_dim=0, shard_size=self.intermediate, num_shards=ws, is_storage_transposed=True
            ),
        )

        self.state_a: torch.Tensor | None = None
        self.state_b: torch.Tensor | None = None
        # KV-aliased binding (prefix caching, runner VLLM_NEURON_STATE_KV_ALIAS): state_a /
        # state_b are the attention layer's bf16 K / V tensors of the shared buffer, and
        # state_shape is the float32 page view [pages, rows, R] the kernels bit-cast to.
        self.state_kv = False
        self.state_shape: tuple[int, ...] | None = None
        self.conv_numel = (self.conv_kernel_size - 1) * self.conv_dim
        self.ssm_numel = self.num_heads * self.head_dim * self.d_state
        # SSM state rows in half A when the page layout is aligned (STATE_ALIGN), else None.
        self.ssm_rows_a: int | None = None

    # ── State plumbing (from contrib/models/qwen3_5/src/deltanet.py) ──────

    def bind_state(
        self, state_a: torch.Tensor, state_b: torch.Tensor, num_pages: int | None = None
    ) -> None:
        """Bind this layer's two state halves ``[pages, half]`` (float32).

        With ``num_pages`` (KV-aliased shared pool), the halves are the attention layer's
        2-byte K / V tensors of the shared buffer, kept as they are so the graph sees each
        buffer once; page ``i`` of half A is bytes ``[i * P/2, (i+1) * P/2)`` of K. The
        kernels bit-cast them to float32 ``[pages, rows, R]`` (``state_shape``).
        """
        if num_pages is not None:
            R = self._STATE_ROW
            if (
                state_a.shape != state_b.shape
                or state_a.dtype != state_b.dtype
                or state_a.element_size() != 2
                or state_a.numel() % (num_pages * 2 * R)
            ):
                raise ValueError(
                    f"{self.layer_name}: KV-aliased state needs two 2-byte tensors of one shape "
                    f"holding {num_pages} pages of whole {R}-float rows, got "
                    f"{tuple(state_a.shape)} {state_a.dtype} / {tuple(state_b.shape)}"
                )
            rows = state_a.numel() // (num_pages * 2 * R)
            if 2 * rows * R < self.conv_numel + self.ssm_numel or self.conv_numel % R or self.ssm_numel % R:
                raise ValueError(f"{self.layer_name}: state page of 2 x {rows} x {R} floats does not fit")
            self.state_a, self.state_b = state_a, state_b
            self.state_kv, self.state_shape = True, (num_pages, rows, R)
            # T11e: the shared pool follows NEMOTRON_H_STATE_ALIGN too (halves of rows * R floats;
            # TP=4, 2,048-token blocks: 1,024 rows, A = conv 36 + group 0 512, B = group 1 512).
            # NEMOTRON_H_STATE_ALIGN_SHARED=0 keeps the default layout here only (the T11d layout).
            self.ssm_rows_a = None
            if STATE_ALIGN == "1":
                if STATE_ALIGN_SHARED == "1":
                    self.ssm_rows_a = self.aligned_ssm_rows_a(rows * R)
                    if self.ssm_rows_a is not None:
                        logger.info(
                            "%s: shared (KV-aliased) state pool uses the aligned page layout "
                            "(NEMOTRON_H_STATE_ALIGN=1): %d rows per half, %d SSM rows in half A",
                            self.layer_name, rows, self.ssm_rows_a,
                        )
                else:
                    logger.warning(
                        "%s: NEMOTRON_H_STATE_ALIGN_SHARED=0: shared (KV-aliased) state pool keeps the default "
                        "page layout", self.layer_name,
                    )
            return
        if state_a.shape != state_b.shape or state_a.dtype != torch.float32:
            raise ValueError(
                f"{self.layer_name}: expected two float32 state halves of one shape, "
                f"got {tuple(state_a.shape)} {state_a.dtype} / {tuple(state_b.shape)}"
            )
        half = state_a.shape[1]
        if 2 * half < self.conv_numel + self.ssm_numel:
            raise ValueError(
                f"{self.layer_name}: a state page of 2 x {half} floats cannot hold "
                f"{self.conv_numel} conv + {self.ssm_numel} ssm floats"
            )
        R = self._STATE_ROW
        if half % R == 0 and self.conv_numel % R == 0 and self.ssm_numel % R == 0:
            state_a = state_a.view(state_a.shape[0], half // R, R)
            state_b = state_b.view(state_b.shape[0], half // R, R)
        self.state_a, self.state_b = state_a, state_b
        self.state_kv, self.state_shape = False, tuple(state_a.shape)
        self.ssm_rows_a = self.aligned_ssm_rows_a(half) if STATE_ALIGN == "1" else None

    def aligned_ssm_rows_a(self, half: int) -> int | None:
        """SSM rows in half A of the aligned layout for halves of ``half`` floats.

        ``None`` (default layout) when the default layout is already aligned (the
        whole SSM state fits in A), or when the aligned layout does not fit the page
        or the page is not whole rows (logged; e.g. TP=1: 8 groups of 512 rows).
        """
        R = self._STATE_ROW
        if half % R or self.conv_numel % R or self.ssm_numel % R:
            logger.warning("%s: NEMOTRON_H_STATE_ALIGN=1 needs whole %d-float rows; default layout", self.layer_name, R)
            return None
        rows, cr, rr = half // R, self.conv_numel // R, self.ssm_numel // R
        gr = rr // self.n_groups
        if cr + rr <= rows:
            return None
        in_a = (rows - cr) // gr * gr
        if in_a == 0 or rr - in_a > rows:
            logger.warning(
                "%s: NEMOTRON_H_STATE_ALIGN=1: aligned layout does not fit (%d conv rows + %d groups of %d rows "
                "in two halves of %d rows); default layout", self.layer_name, cr, self.n_groups, gr, rows,
            )
            return None
        return in_a

    def _ssm_from_page(self, page: torch.Tensor) -> torch.Tensor:
        """The flat SSM state ``[n, ssm_numel]`` of flat pages ``[n, 2 * half]``."""
        if self.ssm_rows_a is None:
            return page[:, self.conv_numel : self.conv_numel + self.ssm_numel]
        R, half = self._STATE_ROW, page.shape[1] // 2
        in_a = self.ssm_rows_a * R
        return torch.cat(
            [
                page[:, self.conv_numel : self.conv_numel + in_a],
                page[:, half : half + self.ssm_numel - in_a],
            ],
            dim=1,
        )

    def _page_from_states(self, conv: torch.Tensor, ssm: torch.Tensor, half: int) -> torch.Tensor:
        """Flat pages ``[n, 2 * half]`` float32 holding ``conv`` and ``ssm`` (pad rows zero)."""
        n = conv.shape[0]
        conv, ssm = conv.reshape(n, -1).float(), ssm.reshape(n, -1).float()
        if self.ssm_rows_a is None:
            page = torch.cat([conv, ssm], dim=1)
            return F.pad(page, (0, 2 * half - page.shape[1]))
        in_a = self.ssm_rows_a * self._STATE_ROW
        a = F.pad(torch.cat([conv, ssm[:, :in_a]], dim=1), (0, half - self.conv_numel - in_a))
        b = F.pad(ssm[:, in_a:], (0, half - (self.ssm_numel - in_a)))
        return torch.cat([a, b], dim=1)

    def _f32_state(self, t: torch.Tensor) -> torch.Tensor:
        """``t`` (a bound half) as float32 ``state_shape`` (a bit-cast view when KV-aliased)."""
        if not self.state_kv:
            return t
        n, rows, r = self.state_shape
        return t.view(n, rows, 2 * r).view(torch.float32)

    def _kernel_state_args(self) -> dict:
        """State keyword arguments of the NKI state kernels."""
        rows = self.state_shape[1] if self.state_kv else 0
        return dict(state_rows=rows)

    def _use_state_kernel(self, x: torch.Tensor, side: str) -> bool:
        """Move state pages with ``kernels/state_cache.py`` on device.

        ``NEMOTRON_H_STATE_KERNEL``: ``1`` (default: read and write), ``read``,
        ``write``, ``0`` (torch ``index_select``/``index_copy_``). The read
        kernel avoids the per-element DMAs an ``index_select`` over state pages
        lowers to in the decode graph (PR #68).
        """
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        mode = os.environ.get("NEMOTRON_H_STATE_KERNEL", "1")
        wanted = mode in ("1", side)
        return wanted and len(self.state_shape) == 3 and can_run_kernel(x)

    def _read_states(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather ``(conv_window [n, K-1, conv_dim], ssm [n, heads, head_dim, d_state])``."""
        n = indices.shape[0]
        if self.ssm_rows_a is not None and self._use_state_kernel(indices, "read"):
            # Aligned layout: read [conv | ssm] rows in order (A head, then B head), no concatenate.
            from .kernels.state_cache import gather_state_rows

            R = self._STATE_ROW
            cr, rr = self.conv_numel // R, self.ssm_numel // R
            page = gather_state_rows(
                self.state_a, self.state_b, indices, cr + self.ssm_rows_a, rr - self.ssm_rows_a,
                **self._kernel_state_args(),
            ).view(n, -1)
            conv = page[:, : self.conv_numel].reshape(n, self.conv_kernel_size - 1, self.conv_dim)
            ssm = page[:, self.conv_numel : self.conv_numel + self.ssm_numel].reshape(
                n, self.num_heads, self.head_dim, self.d_state
            )
            return conv, ssm
        if self._use_state_kernel(indices, "read"):
            from .kernels.state_cache import gather_state_pages

            page = gather_state_pages(
                self.state_a, self.state_b, indices, **self._kernel_state_args()
            ).view(n, -1)
        else:
            sa, sb = self._f32_state(self.state_a), self._f32_state(self.state_b)
            page = torch.cat(
                [sa.index_select(0, indices), sb.index_select(0, indices)],
                dim=1,
            ).reshape(n, -1)
        conv = page[:, : self.conv_numel].reshape(n, self.conv_kernel_size - 1, self.conv_dim)
        ssm = self._ssm_from_page(page).reshape(n, self.num_heads, self.head_dim, self.d_state)
        return conv, ssm

    def _write_states(
        self,
        indices: torch.Tensor,
        conv: torch.Tensor,
        ssm: torch.Tensor,
        live: torch.Tensor | None = None,
    ) -> None:
        """Scatter both states back. Non-``live`` rows (padding) write zeros."""
        n = indices.shape[0]
        if self._use_state_kernel(indices, "write"):
            from .kernels.state_cache import can_scatter_split, scatter_state_split

            if can_scatter_split(self.state_shape, self.conv_numel, self.ssm_numel, self.ssm_rows_a):
                r = self.state_shape[-1]
                conv_rows = conv.reshape(n, -1, r).float()
                ssm_rows = ssm.reshape(n, -1, r).float()
                if live is not None:
                    keep = live.view(n, 1, 1).to(torch.float32)
                    conv_rows, ssm_rows = conv_rows * keep, ssm_rows * keep
                scatter_state_split(
                    self.state_a, self.state_b, conv_rows, ssm_rows, indices, self.ssm_rows_a,
                    **self._kernel_state_args(),
                )
                return
        shape = self.state_shape[1:]
        half = 1
        for d in shape:
            half *= d
        if self.ssm_rows_a is None:
            page = torch.cat([conv.reshape(n, -1).float(), ssm.reshape(n, -1).float()], dim=1)
            if live is not None:
                page = page * live.view(n, 1).to(page.dtype)
            page = F.pad(page, (0, 2 * half - page.shape[1]))
        else:
            page = self._page_from_states(conv, ssm, half)
            if live is not None:
                page = page * live.view(n, 1).to(page.dtype)
        if self._use_state_kernel(page, "write"):
            from .kernels.state_cache import scatter_state_pages

            scatter_state_pages(
                self.state_a,
                self.state_b,
                page[:, :half].reshape((n,) + shape),
                page[:, half:].reshape((n,) + shape),
                indices,
                **self._kernel_state_args(),
            )
            return
        sa, sb = self._f32_state(self.state_a), self._f32_state(self.state_b)
        sa.index_copy_(0, indices, page[:, :half].reshape((n,) + shape))
        sb.index_copy_(0, indices, page[:, half:].reshape((n,) + shape))

    def state_indices(self, metadata: dict, num_reqs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-request state page and liveness. Padded rows use the null block 0."""
        block_table = metadata["block_table_tensor"]
        indices = block_table[:num_reqs, 0].to(torch.long)
        slot_mapping = metadata["slot_mapping"].view(num_reqs, -1)[:, 0]
        num_pages = self.state_shape[0]
        live = (slot_mapping > 0) & (indices > 0) & (indices < num_pages)
        return torch.where(live, indices, torch.zeros_like(indices)), live

    # ── Math ─────────────────────────────────────────────────────────────

    def _split_proj(self, proj: torch.Tensor):
        """``[T, proj] -> (z, xBC, dt)``. Plain slices: ``Tensor.split`` miscompiles on Neuron."""
        i, c = self.intermediate, self.conv_dim
        return proj[:, :i], proj[:, i : i + c], proj[:, i + c :]

    def _split_xbc(self, xbc: torch.Tensor):
        """``[T, conv_dim] -> x [T, heads, head_dim], B [T, heads, N], C [T, heads, N]``.

        B and C are per group; broadcast them to the group's heads.
        """
        t = xbc.shape[0]
        i, bc = self.intermediate, self.bc_size
        x = xbc[:, :i].reshape(t, self.num_heads, self.head_dim)

        def to_heads(m: torch.Tensor) -> torch.Tensor:
            m = m.reshape(t, self.n_groups, 1, self.d_state)
            return m.expand(t, self.n_groups, self.heads_per_group, self.d_state).reshape(
                t, self.num_heads, self.d_state
            )

        return x, to_heads(xbc[:, i : i + bc]), to_heads(xbc[:, i + bc : i + 2 * bc])

    def _causal_conv(self, xbc: torch.Tensor, num_tokens: int, prev: torch.Tensor | None) -> torch.Tensor:
        """Depthwise causal conv as a sum of shifted taps, plus bias (no activation).

        ``out[t, c] = b[c] + sum_j w[c, j] * x[t - (K-1) + j, c]`` ==
        ``F.conv1d(x, w, b, groups=conv_dim, padding=K-1)[..., :T]``.
        """
        k = self.conv_kernel_size
        if prev is None:
            padded = F.pad(xbc, (0, 0, k - 1, 0))
        else:
            padded = torch.cat([prev.to(xbc.dtype), xbc], dim=0)
        w = self.conv1d_weight
        out = padded[:num_tokens] * w[:, 0]
        for tap in range(1, k):
            out = out + padded[tap : tap + num_tokens] * w[:, tap]
        if self.conv1d_bias is not None:
            out = out + self.conv1d_bias
        return out

    @staticmethod
    def _tail_rows(rows_in: torch.Tensor, is_real: torch.Tensor, count: int) -> torch.Tensor:
        """The last ``count`` real rows, selected by a one-hot matmul (no dynamic index).

        From the Qwen3.5 contrib: a data-dependent ``index_select`` here silently
        miscompiles on Neuron. Rows before the sequence start come out zero.
        """
        real = is_real.to(rows_in.dtype)
        onehot_last = real - F.pad(real[1:], (0, 1))
        selector = torch.stack(
            [
                F.pad(onehot_last[count - 1 - j :], (0, count - 1 - j)) if count - 1 - j else onehot_last
                for j in range(count)
            ]
        )
        return selector @ rows_in

    def _gated_norm_out(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """HF ``Zamba2RMSNormGated`` (norm_before_gate=False), then ``out_proj``.

        ``y`` is float32 ``[T, intermediate]``: gate by ``silu(z)``, RMS-normalise
        each group of ``group_size`` channels, scale, project.
        """
        t = y.shape[0]
        y = y.float() * F.silu(z.float())
        g = y.reshape(t, self.intermediate // self.group_size, self.group_size)
        g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + self.eps)
        y = self.norm_weight.float() * g.reshape(t, self.intermediate)
        return y.to(self.dtype) @ self.out_proj_weight

    def _gated_norm(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``_gated_norm_out`` without ``out_proj`` (returns ``[T, intermediate]`` in the model dtype)."""
        t = y.shape[0]
        y = y.float() * F.silu(z.float())
        g = y.reshape(t, self.intermediate // self.group_size, self.group_size)
        g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + self.eps)
        y = self.norm_weight.float() * g.reshape(t, self.intermediate)
        return y.to(self.dtype)

    def ssd_chunk_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked SSD scan for one sequence (HF ``mamba2_chunk_scan``, without D).

        Args:
            x: ``[T, heads, head_dim]`` float32.
            dt: ``[T, heads]`` float32 (already softplus'd and clamped; 0 at pads).
            A: ``[heads]`` (negative).
            B, C: ``[T, heads, d_state]`` float32 (group values broadcast to heads).
            initial_state: ``[heads, head_dim, d_state]`` float32.

        Returns:
            ``(y [T, heads, head_dim], final_state [heads, head_dim, d_state])``.

        Kept at rank 3 (heads x chunks folded into the batch dim), as the
        Qwen3.5 delta rule is, for neuronx-cc. Zero-``dt`` positions (padding)
        are exact no-ops: no decay and no update.
        """
        t, h, p = x.shape
        n = B.shape[-1]
        L = self.chunk_size
        pad = (L - t % L) % L
        if pad:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            B = F.pad(B, (0, 0, 0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, 0, 0, pad))
            dt = F.pad(dt, (0, 0, 0, pad))
        tp = t + pad
        nc = tp // L
        device, dtype = x.device, torch.float32

        def chunked(v: torch.Tensor) -> torch.Tensor:
            """``[T, H, d] -> [H * chunks, L, d]``."""
            return v.transpose(0, 1).reshape(h * nc, L, v.shape[-1])

        xdt = chunked(x * dt.unsqueeze(-1))
        Bc = chunked(B)
        Cc = chunked(C)
        a = (dt * A.view(1, h)).transpose(0, 1).reshape(h * nc, L)

        rows = torch.arange(L, device=device).reshape(L, 1)
        cols = torch.arange(L, device=device).reshape(1, L)
        lower = (rows >= cols).to(dtype)
        # In-chunk inclusive cumsum as a matmul with an upper-triangular ones matrix.
        a_cum = a @ (cols >= rows).to(dtype)  # [H*nc, L]

        # L[i, j] = exp(a_cum_i - a_cum_j) for i >= j, else 0 (HF exp(segment_sum)).
        decay = ((a_cum.unsqueeze(-1) - a_cum.unsqueeze(-2)) * lower).exp() * lower
        # 1. intra-chunk (diagonal blocks)
        y_diag = ((Cc @ Bc.transpose(-1, -2)) * decay) @ xdt  # [H*nc, L, P]
        # 2. each chunk's own contribution to the state at its end: [H*nc, P, N]
        decay_to_end = (a_cum[:, -1:] - a_cum).exp()
        chunk_states = xdt.transpose(-1, -2) @ (Bc * decay_to_end.unsqueeze(-1))
        decay_in = a_cum.exp()  # decay from the chunk start to each position
        chunk_decay = a_cum[:, -1].exp()  # whole-chunk decay

        def per_chunk(v: torch.Tensor) -> list[torch.Tensor]:
            return list(v.reshape(h, nc, *v.shape[1:]).unbind(1))

        Cl, sl, dil, cdl = per_chunk(Cc), per_chunk(chunk_states), per_chunk(decay_in), per_chunk(chunk_decay)
        state = initial_state.to(dtype)  # [H, P, N]
        outs = []
        for k in range(nc):
            # 3./4. state entering the chunk -> outputs
            outs.append((Cl[k] @ state.transpose(-1, -2)) * dil[k].unsqueeze(-1))  # [H, L, P]
            state = state * cdl[k].view(h, 1, 1) + sl[k]
        y_off = torch.stack(outs, dim=1).reshape(h * nc, L, p)
        y = (y_diag + y_off).reshape(h, tp, p).transpose(0, 1)[:t]
        return y, state

    def ssd_chunk_scan_grouped(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        Bg: torch.Tensor,
        Cg: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same result as ``ssd_chunk_scan``, with ``Bg``/``Cg`` per group ``[T, G, N]``.

        Every exponent is a masked difference of cumulative sums (``<= 0``), as
        in the base path, so nothing overflows for large ``|A|``.
        """
        t, h, p = x.shape
        G, n = Bg.shape[1], Bg.shape[2]
        hg = h // G
        L = self.chunk_size
        pad = (L - t % L) % L
        if pad:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            Bg = F.pad(Bg, (0, 0, 0, 0, 0, pad))
            Cg = F.pad(Cg, (0, 0, 0, 0, 0, pad))
            dt = F.pad(dt, (0, 0, 0, pad))
        tp = t + pad
        nc = tp // L
        device, dtype = x.device, torch.float32

        rows = torch.arange(L, device=device).reshape(L, 1)
        cols = torch.arange(L, device=device).reshape(1, L)
        lower = (rows >= cols).to(dtype)
        a = (dt * A.view(1, h)).reshape(nc, L, h).permute(2, 0, 1)  # [h, nc, L]
        a_cum = a @ (cols >= rows).to(dtype)  # in-chunk inclusive cumsum
        decay = ((a_cum.unsqueeze(-1) - a_cum.unsqueeze(-2)) * lower).exp() * lower  # [h, nc, L, L]

        xdt = (x * dt.unsqueeze(-1)).reshape(nc, L, h, p).permute(2, 0, 1, 3)  # [h, nc, L, p]
        Bc = Bg.reshape(nc, L, G, n).permute(2, 0, 1, 3)  # [G, nc, L, N]
        Cc = Cg.reshape(nc, L, G, n).permute(2, 0, 1, 3)
        CB = Cc @ Bc.transpose(-1, -2)  # [G, nc, L, L], once per group
        scores = decay.reshape(G, hg, nc, L, L) * CB.unsqueeze(1)
        y_diag = scores.reshape(h, nc, L, L) @ xdt  # [h, nc, L, p]

        # Each chunk's contribution to the state at its end, batched per group:
        # [G, nc, hg*p, L] @ [G, nc, L, N] -> [G, nc, hg*p, N].
        decay_to_end = (a_cum[..., -1:] - a_cum).exp()  # [h, nc, L]
        xs = (xdt * decay_to_end.unsqueeze(-1)).reshape(G, hg, nc, L, p)
        xs = xs.permute(0, 2, 1, 4, 3).reshape(G, nc, hg * p, L)
        cs = (xs @ Bc).reshape(G, nc, hg, p, n).permute(0, 2, 1, 3, 4).reshape(h, nc, p * n)

        # Inter-chunk recurrence as one matmul: the state entering chunk c is
        # exp(Acs_ex[c]) S0 + sum_{c' < c} exp(Acs_ex[c] - Acs[c']) cs[c'].
        a_end = a_cum[..., -1]  # [h, nc] whole-chunk log decay
        crow = torch.arange(nc, device=device).reshape(nc, 1)
        ccol = torch.arange(nc, device=device).reshape(1, nc)
        acs = a_end @ (ccol >= crow).to(dtype)  # inclusive cumsum over chunks
        acs_ex = acs - a_end
        strict = (crow > ccol).to(dtype)
        w = ((acs_ex.unsqueeze(-1) - acs.unsqueeze(-2)) * strict).exp() * strict  # [h, nc, nc]
        s0 = initial_state.to(dtype).reshape(h, 1, p * n)
        s_in = w @ cs + acs_ex.exp().unsqueeze(-1) * s0  # [h, nc, p*n]
        w_fin = (acs[:, -1:] - acs).exp().unsqueeze(1)  # [h, 1, nc]
        final_state = (w_fin @ cs + acs[:, -1:].exp().unsqueeze(-1) * s0).reshape(h, p, n)

        # Outputs from the entering state, batched per group:
        # [G, nc, L, N] @ [G, nc, N, hg*p] -> [G, nc, L, hg*p].
        s_t = s_in.reshape(G, hg, nc, p, n).permute(0, 2, 4, 1, 3).reshape(G, nc, n, hg * p)
        y_off = (Cc @ s_t).reshape(G, nc, L, hg, p)
        decay_in = a_cum.exp().reshape(G, hg, nc, L).permute(0, 2, 3, 1).unsqueeze(-1)  # [G, nc, L, hg, 1]
        y_off = (y_off * decay_in).permute(1, 2, 0, 3, 4).reshape(tp, h, p)
        y = y_diag.permute(1, 2, 0, 3).reshape(tp, h, p) + y_off
        return y[:t], final_state

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, attn_metadata: dict) -> torch.Tensor:
        metadata = attn_metadata[self.layer_name]
        if metadata["max_query_len"] <= metadata["decode_token_threshold"]:
            return self.forward_decode(hidden_states, metadata)
        if self.world_size > 1 and SEQUENCE_PARALLEL:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(hidden_states, positions, metadata)

    def forward_prefill(self, hidden_states: torch.Tensor, positions: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Prefill (or one chunk of a chunked prefill) of ONE sequence.

        Bucket padding appends token 0 and repeats the last position, so a token
        is real iff ``positions[i] - positions[0] == i``. Pads get ``xBC = 0``
        (so the conv does not smear them into real rows) and ``dt = 0`` (so the
        scan treats them as no-ops), and the written state is the state after the
        last real token.
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        offsets = torch.arange(num_tokens, device=device, dtype=positions.dtype)
        is_real = (positions - positions[0]) == offsets
        real = is_real.to(torch.float32).unsqueeze(-1)

        # Raw block id, no arithmetic on it before the state write (NCC_IDLO901).
        page_idx = metadata["block_table_tensor"][:1, 0].to(torch.long)

        conv_prev = None
        initial_state = torch.zeros(
            self.num_heads, self.head_dim, self.d_state, dtype=torch.float32, device=device
        )
        prior = metadata.get("cached_seq_len")
        if prior is not None:
            # Chunked prefill: continue from the state the earlier chunks left. The
            # first chunk starts cold (its page holds a previous request's state),
            # selected away rather than multiplied by zero.
            cached_conv, cached_ssm = self._read_states(page_idx)
            warm = prior.reshape(1) > 0
            conv_prev = torch.where(
                warm.view(1, 1), cached_conv[0].float(), torch.zeros_like(cached_conv[0].float())
            )
            initial_state = torch.where(warm.view(1, 1, 1), cached_ssm[0].float(), initial_state)

        sub = PREFILL_SUBCHUNK
        if self._use_prefill_kernel(hidden_states):
            # R2-03: fused kernel on the whole chunk (conv .. gated norm), torch in_proj (split) and out_proj.
            from .kernels.ssd_prefill import mamba_prefill_fused_split

            k1 = self.conv_kernel_size - 1
            h_in = hidden_states.to(self.dtype)
            i, c = self.intermediate, self.conv_dim
            w = self.in_proj_weight
            prev_bf = (
                torch.zeros(k1, self.conv_dim, dtype=self.dtype, device=device)
                if conv_prev is None
                else conv_prev.to(self.dtype)
            )
            if PREFILL_INPROJ == "fused":
                from .kernels.ssd_prefill import mamba_prefill_inproj_fused

                y_n, final_state, proj = mamba_prefill_inproj_fused(self, h_in, prev_bf, real, initial_state)
                xbc = proj[:, i : i + c]
            elif PREFILL_INPROJ == "nki":
                # R2-03m: in_proj as one NKI kernel, then the proj-input fused kernel; xBC unmasked (pad rows only).
                from .kernels.ssd_prefill import mamba_in_proj, mamba_prefill_fused

                ns = PREFILL_KSPLIT
                if ns > 1 and num_tokens % (ns * self.chunk_size) == 0:
                    # R2-03m: the chunk in ns pieces (in_proj kernel, fused kernel, ...) carrying the conv window and the
                    # fp32 state, so piece p+1's in_proj (TensorE) can run under piece p's scan (Vector/Scalar).
                    tp = num_tokens // ns
                    ys, st, prev, new_conv = [], initial_state, prev_bf, None
                    ones = torch.ones(k1, dtype=is_real.dtype, device=device)
                    sel = self._tail_selector(torch.cat([ones, is_real]), k1, self.dtype)
                    new_conv = sel[:, :k1] @ prev_bf
                    for s0 in range(0, num_tokens, tp):
                        pj = mamba_in_proj(h_in[s0 : s0 + tp], w)
                        yk, st = mamba_prefill_fused(self, pj, prev, real[s0 : s0 + tp], st)
                        ys.append(yk)
                        xk = pj[:, i : i + c]
                        new_conv = new_conv + sel[:, k1 + s0 : k1 + s0 + tp] @ xk
                        prev = xk[tp - k1 :]
                    y_n, final_state = torch.cat(ys, dim=0), st
                    xbc = None
                else:
                    proj = mamba_in_proj(h_in, w)
                    y_n, final_state = mamba_prefill_fused(self, proj, prev_bf, real, initial_state)
                    xbc = proj[:, i : i + c]
            else:
                z, xbc, dt = h_in @ w[:, :i], h_in @ w[:, i : i + c], h_in @ w[:, i + c :]
                xbc = xbc * real.to(xbc.dtype)
                y_n, final_state = mamba_prefill_fused_split(self, z, xbc, dt, prev_bf, real, initial_state)
            output = y_n @ self.out_proj_weight
            # last K-1 real rows of [prev | xbc]: one-hot selector applied per piece (no concatenate); on a cold
            # start prev is zeros (A06: the cold-only selector graph wrote non-finite values in a unit graph).
            if xbc is not None:  # (R2-03m KSPLIT pieces computed new_conv above)
                ones = torch.ones(k1, dtype=is_real.dtype, device=device)
                sel = self._tail_selector(torch.cat([ones, is_real]), k1, xbc.dtype)
                new_conv = sel[:, :k1] @ prev_bf + sel[:, k1:] @ xbc
        elif sub and num_tokens > sub and num_tokens % sub == 0 and PREFILL_SUBCHUNK_SCOPE == "body":
            # T-chunk4096: run the per-token body on sub-chunks of ``sub`` tokens in order, carrying the
            # conv window and the SSM state; same arithmetic as one pass (sub is a multiple of chunk_size).
            k1 = self.conv_kernel_size - 1
            outs, xbcs = [], []
            state, prev = initial_state, conv_prev
            for s0 in range(0, num_tokens, sub):
                o, xbc, state = self._prefill_body(hidden_states[s0 : s0 + sub], real[s0 : s0 + sub], prev, state, sub)
                outs.append(o)
                xbcs.append(xbc)
                prev = xbc[sub - k1 :]
            output = torch.cat(outs, dim=0)
            final_state = state
            # last K-1 real rows of [conv_prev | xbc], one-hot selector applied per piece (no concatenate)
            if conv_prev is None:
                sel = self._tail_selector(is_real, k1, xbcs[0].dtype)
                new_conv = sum(sel[:, s0 : s0 + sub] @ x for s0, x in zip(range(0, num_tokens, sub), xbcs))
            else:
                ones = torch.ones(k1, dtype=is_real.dtype, device=device)
                sel = self._tail_selector(torch.cat([ones, is_real]), k1, xbcs[0].dtype)
                new_conv = sel[:, :k1] @ conv_prev.to(xbcs[0].dtype)
                for s0, x in zip(range(0, num_tokens, sub), xbcs):
                    new_conv = new_conv + sel[:, k1 + s0 : k1 + s0 + sub] @ x
        else:
            output, xbc, final_state = self._prefill_body(hidden_states, real, conv_prev, initial_state, num_tokens,
                                                          sub if PREFILL_SUBCHUNK_SCOPE != "body" else 0)
            if conv_prev is None:
                new_conv = self._tail_rows(xbc, is_real, self.conv_kernel_size - 1)
            else:
                ones = torch.ones(conv_prev.shape[0], dtype=is_real.dtype, device=device)
                new_conv = self._tail_rows(
                    torch.cat([conv_prev.to(xbc.dtype), xbc], dim=0),
                    torch.cat([ones, is_real]),
                    self.conv_kernel_size - 1,
                )
        wpage = page_idx
        if self.state_kv and self._use_state_kernel(page_idx, "write") and PREFILL_NULL_WRITE == "oor":
            # Shared pool: page 0 is also the attention layers' null block, which padded
            # attention positions read (0 * NaN = NaN). A prefill with the null page (the
            # warmup's dummy block table) must not leave float32 state bits there, so its
            # write goes to an out-of-range page, which the kernel drops. (Default "page0" instead
            # writes page 0 and relies on the runner clearing the shared pool, R2-03p2.)
            wpage = torch.where(page_idx > 0, page_idx, torch.full_like(page_idx, self.state_shape[0]))
        self._write_states(wpage, new_conv.unsqueeze(0), final_state.unsqueeze(0))
        if self.world_size > 1:
            if SEQUENCE_PARALLEL:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                output = self.tp_group.all_reduce(output)
        return output.contiguous()

    def _prefill_body(self, hidden_states: torch.Tensor, real: torch.Tensor, conv_prev, initial_state: torch.Tensor,
                      num_tokens: int, sub: int = 0):
        """in_proj .. out_proj of ``forward_prefill`` for ``num_tokens`` rows (no state read/write, no collective).

        Returns ``(output [T, H] before the TP reduction, masked xBC [T, conv_dim], final SSM state)``.
        """
        opts = prefill_opts()
        h_in = hidden_states.to(self.dtype)
        if "split" in opts:
            i, c = self.intermediate, self.conv_dim
            w = self.in_proj_weight
            z, xbc, dt = h_in @ w[:, :i], h_in @ w[:, i : i + c], h_in @ w[:, i + c :]
        else:
            proj = h_in @ self.in_proj_weight
            z, xbc, dt = self._split_proj(proj)
        if "nomask" not in opts:
            xbc = xbc * real.to(xbc.dtype)

        if sub and num_tokens > sub and num_tokens % sub == 0 and PREFILL_SUBCHUNK_SCOPE == "mid":
            k1 = self.conv_kernel_size - 1
            ys, state, prev = [], initial_state, conv_prev
            for s0 in range(0, num_tokens, sub):
                xs = xbc[s0 : s0 + sub]
                yg, state = self._prefill_mid(xs, z[s0 : s0 + sub], dt[s0 : s0 + sub], real[s0 : s0 + sub], prev, state,
                                              sub, 0)
                ys.append(yg)
                prev = xs[sub - k1 :]
            yg = torch.cat(ys, dim=0)
            return yg @ self.out_proj_weight, xbc, state
        yg, final_state = self._prefill_mid(xbc, z, dt, real, conv_prev, initial_state, num_tokens,
                                            sub if PREFILL_SUBCHUNK_SCOPE == "scan" else 0)
        return yg @ self.out_proj_weight, xbc, final_state

    def _prefill_mid(self, xbc, z, dt, real, conv_prev, initial_state, num_tokens: int, scan_sub: int):
        """conv .. gated norm (bf16, before out_proj) for ``num_tokens`` rows; ``scan_sub``: SSD scan in pieces."""
        opts = prefill_opts()
        conv_out = F.silu(self._causal_conv(xbc, num_tokens, conv_prev))
        if "grp" in opts:
            i, bc = self.intermediate, self.bc_size
            x = conv_out[:, :i].reshape(num_tokens, self.num_heads, self.head_dim).float()
            B = conv_out[:, i : i + bc].reshape(num_tokens, self.n_groups, self.d_state).float()
            C = conv_out[:, i + bc : i + 2 * bc].reshape(num_tokens, self.n_groups, self.d_state).float()
        else:
            x, B, C = self._split_xbc(conv_out)
            x, B, C = x.float(), B.float(), C.float()

        dt = F.softplus(dt.float() + self.dt_bias)
        if DT_CLAMP == "hf":
            dt = torch.clamp(dt, min=self.time_step_min)
        dt = dt * real
        A = -torch.exp(self.A_log.float())

        scan = self.ssd_chunk_scan_grouped if "grp" in opts else self.ssd_chunk_scan
        if scan_sub and num_tokens > scan_sub and num_tokens % scan_sub == 0:
            ys, final_state = [], initial_state
            for s0 in range(0, num_tokens, scan_sub):
                sl = slice(s0, s0 + scan_sub)
                yp, final_state = scan(x[sl], dt[sl], A, B[sl], C[sl], final_state)
                ys.append(yp)
            y = torch.cat(ys, dim=0)
        else:
            y, final_state = scan(x, dt, A, B, C, initial_state)
        y = y + self.D.float().view(1, -1, 1) * x
        return self._gated_norm(y.reshape(num_tokens, self.intermediate), z), final_state

    @staticmethod
    def _tail_selector(is_real: torch.Tensor, count: int, dtype) -> torch.Tensor:
        """The ``[count, T]`` one-hot selector of ``_tail_rows``."""
        real = is_real.to(dtype)
        onehot_last = real - F.pad(real[1:], (0, 1))
        return torch.stack(
            [
                F.pad(onehot_last[count - 1 - j :], (0, count - 1 - j)) if count - 1 - j else onehot_last
                for j in range(count)
            ]
        )

    def _use_prefill_kernel(self, x: torch.Tensor) -> bool:
        """Whether the fused prefill kernel (``NEMOTRON_H_MAMBA_PREFILL_KERNEL=1``, R2-03) applies."""
        if PREFILL_KERNEL != "1":
            return False
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        t = x.shape[0]
        return (
            # Shared pool (prefix caching): on by default since R2-03p2 (NEMOTRON_H_MAMBA_PREFILL_KERNEL_SHARED).
            (not self.state_kv or PREFILL_KERNEL_SHARED == "1")
            and (not PREFILL_KERNEL_LAYERS or self.layer_idx in PREFILL_KERNEL_LAYERS)
            and t >= PREFILL_KERNEL_MIN_T
            and t % self.chunk_size == 0
            and "split" in prefill_opts()
            and DT_CLAMP == "none"
            and self.conv1d_bias is not None
            and self.chunk_size == 128
            and self.d_state == 128
            and self.group_size == self.heads_per_group * self.head_dim <= 512
            and self.n_groups % 2 == 0
            and can_run_kernel(x)
        )

    def _use_decode_kernel(self, proj: torch.Tensor) -> bool:
        """Whether the fused decode kernel (``NEMOTRON_H_MAMBA_DECODE_KERNEL=1``) applies."""
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        return (
            DT_CLAMP == "none"
            and self.conv1d_bias is not None
            and proj.shape[0] <= 128
            and self.state_a is not None
            and len(self.state_shape) == 3
            and self.state_shape[-1] == self._STATE_ROW == self.d_state
            and self.group_size == self.heads_per_group * self.head_dim == 4 * 128
            and self.conv_numel % self._STATE_ROW == 0
            and can_run_kernel(proj)
        )

    def _forward_decode_kernel(self, proj: torch.Tensor, metadata: dict, num_reqs: int) -> torch.Tensor:
        """``forward_decode`` through ``kernels/mamba_decode.py`` (state updated in place)."""
        from .kernels.mamba_decode import fused_decode

        if self.state_kv and os.environ.get("NEMOTRON_H_T11B_DBG", "") == "skip":
            # T11b bisect only: no state access at all in decode (output = 0).
            return torch.zeros(proj.shape[0], self.hidden_size, dtype=self.dtype, device=proj.device)
        state_idx, live = self.state_indices(metadata, num_reqs)
        # Padding rows read the null page 0 and skip their write (out-of-range id).
        wpages = torch.where(live, state_idx, torch.full_like(state_idx, self.state_shape[0]))
        dbg = os.environ.get("NEMOTRON_H_T11B_DBG", "")
        if self.state_kv and dbg == "nowrite":  # T11b bisect only: drop the decode state write
            wpages = torch.full_like(state_idx, self.state_shape[0])
        if self.state_kv and dbg == "noread":  # T11b bisect only: read the null page
            state_idx = torch.zeros_like(state_idx)
        y = fused_decode(self, proj, state_idx, wpages)
        output = y @ self.out_proj_weight
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output

    def forward_decode(self, hidden_states: torch.Tensor, metadata: dict) -> torch.Tensor:
        """One recurrence step per request."""
        num_reqs = metadata["block_table_tensor"].shape[0]
        if hidden_states.shape[0] != num_reqs:
            raise NotImplementedError(
                f"Mamba decode expects one token per request, got {hidden_states.shape[0]} "
                f"tokens for {num_reqs} requests (speculative decoding is not wired up)"
            )
        # The fused kernel takes the single in_proj output; it is checked before the
        # T12 decode split (which stays off by default).
        if MAMBA_DECODE_KERNEL == "1":
            from .kernels.decode_gemv import maybe_gemv

            proj = maybe_gemv(hidden_states.to(self.dtype), self.in_proj_weight, "inproj")
            if self._use_decode_kernel(proj):
                return self._forward_decode_kernel(proj, metadata, num_reqs)
        if 0 < DECODE_SPLIT_MIN_BATCH <= num_reqs <= DECODE_SPLIT_MAX_BATCH:
            h_in = hidden_states.to(self.dtype)
            i, c = self.intermediate, self.conv_dim
            w_in = self.in_proj_weight
            z, xbc, dt = h_in @ w_in[:, :i], h_in @ w_in[:, i : i + c], h_in @ w_in[:, i + c :]
        else:
            proj = hidden_states.to(self.dtype) @ self.in_proj_weight
            z, xbc, dt = self._split_proj(proj)
        state_idx, live = self.state_indices(metadata, num_reqs)
        conv_prev, ssm = self._read_states(state_idx)
        conv_prev = conv_prev.to(xbc.dtype)

        # Conv over the cached window plus this token, tap by tap (see qwen3_5
        # deltanet.forward_decode for why not a concatenate + broadcast multiply).
        w = self.conv1d_weight
        last = self.conv_kernel_size - 1
        mixed = conv_prev[:, 0] * w[:, 0]
        for tap in range(1, last):
            mixed = mixed + conv_prev[:, tap] * w[:, tap]
        mixed = mixed + xbc * w[:, last]
        if self.conv1d_bias is not None:
            mixed = mixed + self.conv1d_bias
        x, B, C = self._split_xbc(F.silu(mixed))
        x, B, C = x.float(), B.float(), C.float()

        dt = F.softplus(dt.float() + self.dt_bias)  # [n, heads]
        if DT_CLAMP == "hf":
            dt = torch.clamp(dt, min=self.time_step_min)
        A = -torch.exp(self.A_log.float())
        dA = torch.exp(dt * A.view(1, -1))
        if DECODE_Y == "rows":
            # Same math on [num_reqs * heads * head_dim, d_state] rows: one row per
            # (request, head, channel), d_state along the free axis.
            nh, P, N = num_reqs * self.num_heads, self.head_dim, self.d_state
            rows = nh * P
            ssm2 = ssm.float().reshape(rows, N)
            dA_r = dA.reshape(nh, 1).expand(nh, P).reshape(rows, 1)
            dtx_r = (dt.unsqueeze(-1) * x).reshape(rows, 1)
            B_r = B.reshape(nh, 1, N).expand(nh, P, N).reshape(rows, N)
            C_r = C.reshape(nh, 1, N).expand(nh, P, N).reshape(rows, N)
            ssm2 = ssm2 * dA_r + dtx_r * B_r
            y = (ssm2 * C_r).sum(dim=-1).reshape(num_reqs, self.num_heads, P)
            ssm = ssm2.reshape(num_reqs, self.num_heads, P, N)
        else:
            ssm = ssm.float() * dA.view(num_reqs, -1, 1, 1) + (
                (dt.unsqueeze(-1) * x).unsqueeze(-1) * B.unsqueeze(-2)
            )
        if DECODE_Y == "rows":
            pass
        elif DECODE_Y == "mulsum":
            # Broadcast multiply + reduce over d_state (vector engine), instead of
            # num_reqs * heads tiny [head_dim, d_state] @ [d_state, 1] matmuls.
            y = (ssm * C.unsqueeze(-2)).sum(dim=-1)
        else:
            nh = num_reqs * self.num_heads
            y = (ssm.reshape(nh, self.head_dim, self.d_state) @ C.reshape(nh, self.d_state, 1)).reshape(
                num_reqs, self.num_heads, self.head_dim
            )
        y = y + self.D.float().view(1, -1, 1) * x

        self._write_states(
            state_idx,
            torch.cat([conv_prev[:, 1:], xbc.unsqueeze(1)], dim=1),
            ssm,
            live,
        )
        output = self._gated_norm_out(y.reshape(num_reqs, self.intermediate), z)
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output
