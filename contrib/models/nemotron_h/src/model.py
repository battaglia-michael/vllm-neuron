# SPDX-License-Identifier: Apache-2.0
"""NemotronH decoder for Neuron (vLLM Neuron plugin, V1 runner, recurrent-state core).

Each of the 52 layers of NVIDIA-Nemotron-3.5-Lightning-30B-A3B is one mixer
with a pre-norm and a residual, ``x + mixer(RMSNorm(x))``:

* 23 Mamba-2 layers (``mamba.py``): fixed-size per-request state, declared as
  ``KVSpec.state_layers`` and bound through ``bind_state_cache``.
* 23 MoE layers (``moe.py``): sigmoid router with selection bias, non-gated
  squared-ReLU experts, one shared expert.
* 6 attention layers (here): GQA 32 Q / 2 KV heads, head_dim 128, no bias and
  **no positional encoding**. HF ``NemotronHAttention`` (and vLLM's GPU model)
  never apply RoPE, so ``NF.qkv_proj`` / ``NF.attention_decode`` are called
  without cos/sin. ``rope_theta`` in config.json is unused.

Weights follow the plugin's convention (``[in_features, out_features]``,
``hidden @ weight``). The ``mtp.*`` tensors are not loaded.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, StateLayerSpec
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

from .config import ATTENTION, MAMBA, MLP, MOE, NemotronHConfig
from .mamba import SEQUENCE_PARALLEL, NemotronHMamba2Mixer
from .moe import NemotronHMoE

_DECODE_MASK_CACHE_KEY = "_nemotron_h_decode_mask"

# The decode attention kernel (nkilib attention_block_tkg) splits the hidden dim
# into H/128 tiles and shards them over the LNC=2 cores; H=2688 is 21 tiles, so
# it fails validation (NCC_INKI016 "H1 must be divisible by num_shards"). The
# attention layers therefore carry the hidden dim zero-padded to a multiple of
# 128 * 2 (2688 -> 2816): extra zero rows in W_qkv and zero columns in W_o, the
# input zero-padded and the output sliced back. Exact in real arithmetic.
_ATTN_HIDDEN_MULTIPLE = 256

# Chunked-prefill KV write. "torch" (default): index_put_ into the paged cache,
# then NF.segmented_attention. "kernel": NF.qkv_proj's in-kernel write
# (use_block_kv=True). On vllm-neuron 0.29 the kernel path reshapes the 4D cache
# to 3D and copy_()s the whole cache back afterwards, so the prefill graph holds
# several full copies of every K/V cache (4-layer model with a 6.6 GiB KV
# budget: 34 GB peak, NCC_EOOM002).
_SEG_KV_WRITE = os.environ.get("NEMOTRON_H_SEG_KV_WRITE", "torch")
# Debug (T12e): return the hidden states after layer L (0-based) to the final norm and LM head, while every later layer
# still runs (its output is added times 0.0, so the graph keeps all layers). -1 (default) = off.
_DEBUG_EXIT_LAYER = int(os.environ.get("NEMOTRON_H_DEBUG_EXIT_LAYER", "-1"))
if _SEG_KV_WRITE not in ("torch", "kernel"):
    raise ValueError(f"NEMOTRON_H_SEG_KV_WRITE must be 'torch' or 'kernel', got {_SEG_KV_WRITE!r}")
# R2-02: decode attention context alignment (tokens). The nkilib decode kernel (attention_tkg, block KV)
# needs blocks-per-sequence to be a multiple of 128 (256 when s_prior is sharded over the 2 cores) and
# shrinks its block length until that holds: resized block_len = gcd(32, ctx / 128 [, FA tile terms]).
# The S profile's 8,960-token decode context gives block_len 1-2 (one 256-512 B DMA element per 1-2
# tokens). Padding the kernel's block table (with -1 / the null block, masked) up to a multiple of
# this many tokens keeps the element size up: 1024 -> 9,216 (block_len 4-8), 2048 -> 10,240 (8-16).
# Default 1024 (R2-02, optimization step R2-B: S decode step b32 40.0 -> 35.4 ms, b128 82.8 -> 63.9 ms; L and P
# contexts are already multiples of 1024, so their graphs are unchanged). 0 = off (the context the runner gives).
_DECODE_CTX_ALIGN = int(os.environ.get("NEMOTRON_H_DECODE_CTX_ALIGN", "1024"))


class NemotronHRMSNorm(nn.Module):
    """HF ``NemotronHRMSNorm``: normalise in float32, cast back, then ``weight *``."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * h.to(input_dtype)


class NemotronHAttention(nn.Module):
    """GQA without positional encoding, on the plugin's NF attention ops.

    Prefill: ``NF.qkv_proj`` -> paged KV write -> ``NF.flash_attention`` ->
    ``NF.o_proj``. Chunked prefill (``kv_segment_size`` set): ``NF.qkv_proj``
    with the in-kernel KV write, then ``NF.segmented_attention`` over the cache.
    Decode: ``NF.attention_decode`` (fused QKV + attention + o_proj, in-kernel KV
    write). Same structure as ``vllm_neuron/model/llama3`` minus RoPE, FP8, DCP
    and attention-DP.

    With 2 KV heads and TP=4 each KV head is replicated on 2 ranks
    (``num_kv_replicas``), each rank holding 8 query heads.
    """

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.mixer"
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.tp_group = get_tp_group()
        self.world_size = ws = self.tp_group.world_size
        nh, nkv = config.num_attention_heads, config.num_key_value_heads
        if nh % ws:
            raise ValueError(f"num_attention_heads={nh} must be divisible by tp_size={ws}")
        self.num_heads_per_rank = nh // ws
        if ws >= nkv:
            if ws % nkv:
                raise ValueError(f"tp_size={ws} must be a multiple of num_key_value_heads={nkv}")
            self.num_kv_heads_per_rank, self.num_kv_replicas = 1, ws // nkv
        else:
            if nkv % ws:
                raise ValueError(f"num_key_value_heads={nkv} must be divisible by tp_size={ws}")
            self.num_kv_heads_per_rank, self.num_kv_replicas = nkv // ws, 1
        self.num_kv_groups = self.num_heads_per_rank // self.num_kv_heads_per_rank

        self.q_size = self.num_heads_per_rank * self.head_dim
        self.kv_size = self.num_kv_heads_per_rank * self.head_dim
        self.qkv_split = [self.q_size, self.q_size + self.kv_size]
        m = _ATTN_HIDDEN_MULTIPLE
        self.hidden_padded = -(-self.hidden_size // m) * m
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_padded, self.q_size + 2 * self.kv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(torch.empty(self.q_size, self.hidden_padded, dtype=self.dtype))
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=ws,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
                padded_hidden_size=self.hidden_padded,
                unpadded_hidden_size=self.hidden_size,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=self.q_size,
                num_shards=ws,
                is_storage_transposed=True,
                pad_dim=1,
                padded_size=self.hidden_padded,
                unpadded_size=self.hidden_size,
            ),
        )
        # Bound by bind_kv_cache: [blocks, kv_heads, block_size, head_dim].
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        # Set by NemotronHForCausalLM.bind_state_cache when Mamba state shares this pool.
        self.decode_pad_to_null = False

    def forward(self, hidden_states, positions, attn_metadata: dict) -> torch.Tensor:
        metadata = attn_metadata[self.layer_name]
        if metadata["max_query_len"] <= metadata["decode_token_threshold"]:
            return self.forward_decode(hidden_states, positions, attn_metadata)
        if self.world_size > 1 and SEQUENCE_PARALLEL:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(hidden_states, metadata)

    def _all_reduce(self, output: torch.Tensor) -> torch.Tensor:
        """TP all-reduce of the ``[:, :hidden_size]`` slice of the padded output.

        The slice is a non-contiguous view once there is more than one row, and
        the gloo all-reduce behind the CPU TP simulation (``test/unit/tp_sim.py``)
        silently returns wrong rows for a non-contiguous input (row 0 right,
        later rows read from the wrong storage offsets). Copy it on CPU only, so
        the traced Neuron graphs (and their compile-cache keys) do not change;
        on device the batched decode and batch-invariance checks pass with the
        slice as is (T03c).
        """
        if output.device.type == "cpu":
            output = output.contiguous()
        return self.tp_group.all_reduce(output)

    def _pad_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-pad the last (hidden) dim to ``hidden_padded``."""
        pad = self.hidden_padded - self.hidden_size
        return F.pad(x, (0, pad)) if pad else x

    def _write_kv(self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor, block_size: int):
        """Scatter ``[kv_heads, T, d]`` into the paged cache.

        Padded tokens (slot <= 0) write zeros into the null block 0, which dead
        decode rows read (Qwen3.5 contrib ``_write_kv``).
        """
        nkh = self.num_kv_heads_per_rank
        num_tokens = slot_mapping.shape[0]
        live = slot_mapping > 0
        blocks = torch.where(live, slot_mapping // block_size, torch.zeros_like(slot_mapping))
        offsets = torch.where(live, slot_mapping % block_size, torch.zeros_like(slot_mapping))
        heads = torch.arange(nkh, dtype=torch.long, device=k.device).repeat_interleave(num_tokens)
        blocks, offsets = blocks.repeat(nkh).to(torch.long), offsets.repeat(nkh).to(torch.long)
        keep = live.repeat(nkh).view(-1, 1).to(self.k_cache.dtype)
        self.k_cache.index_put_((blocks, heads, offsets), k.reshape(-1, self.head_dim).to(self.k_cache.dtype) * keep)
        self.v_cache.index_put_((blocks, heads, offsets), v.reshape(-1, self.head_dim).to(self.v_cache.dtype) * keep)

    def forward_prefill(self, hidden_states: torch.Tensor, metadata: dict) -> torch.Tensor:
        hidden_states = self._pad_hidden(hidden_states.to(self.dtype))
        tokens = hidden_states.shape[0]
        slot_mapping = metadata["slot_mapping"]
        block_size = metadata["block_size"]
        kv_segment_size = metadata.get("kv_segment_size")

        if kv_segment_size and _SEG_KV_WRITE == "torch":
            # Chunked prefill: project, scatter this chunk's K/V into the paged
            # cache, then segmented attention reads prefix + chunk from the cache.
            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
                d_head=self.head_dim,
                num_q_heads=self.num_heads_per_rank,
                num_kv_heads=self.num_kv_heads_per_rank,
            ).squeeze(0)
            q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)
            q = q.reshape(tokens, self.num_heads_per_rank, self.head_dim).transpose(0, 1)
            k = k.reshape(tokens, self.num_kv_heads_per_rank, self.head_dim).transpose(0, 1)
            v = v.reshape(tokens, self.num_kv_heads_per_rank, self.head_dim).transpose(0, 1)
            self._write_kv(k, v, slot_mapping, block_size)
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=metadata["block_table_tensor"],
                prior_tokens=metadata["cached_seq_len"],
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
                fp8_packed=False,
                allow_vector_exp=False,
            )  # [Nh, Dh, T]
        elif kv_segment_size:
            # Chunked prefill: the KV write is folded into the qkv kernel, and
            # segmented attention reads prefix + chunk from the cache.
            max_slot = self.k_cache.shape[0] * block_size
            slots = torch.where(
                (slot_mapping < 0) | (slot_mapping >= max_slot),
                torch.zeros_like(slot_mapping),
                slot_mapping,
            ).to(torch.int32)
            q_hbm, _, _ = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
                d_head=self.head_dim,
                num_q_heads=self.num_heads_per_rank,
                num_kv_heads=self.num_kv_heads_per_rank,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                use_block_kv=True,
                block_size=block_size,
                slot_mapping=slots.unsqueeze(0),
            )
            q = q_hbm.squeeze(0).view(tokens, self.num_heads_per_rank, self.head_dim).transpose(0, 1)
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=metadata["block_table_tensor"],
                prior_tokens=metadata["cached_seq_len"],
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
                fp8_packed=False,
                allow_vector_exp=False,
            )  # [Nh, Dh, T]
        else:
            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
                d_head=self.head_dim,
                num_q_heads=self.num_heads_per_rank,
                num_kv_heads=self.num_kv_heads_per_rank,
            ).squeeze(0)
            q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)
            q = q.reshape(tokens, self.num_heads_per_rank, self.head_dim).transpose(0, 1)
            k = k.reshape(tokens, self.num_kv_heads_per_rank, self.head_dim).transpose(0, 1)
            v = v.reshape(tokens, self.num_kv_heads_per_rank, self.head_dim).transpose(0, 1)
            if metadata.get("cached_seq_len") is not None:
                # Chunked prefill without segmented attention (kv_segment_size_buckets
                # unset): attend to the cached prefix in torch.
                attn_output = self._chunk_attention_torch(q, k, v, metadata)
                self._write_kv(k, v, slot_mapping, block_size)
                return self._finish_prefill(attn_output)
            self._write_kv(k, v, slot_mapping, block_size)
            k = k.repeat_interleave(self.num_kv_groups, dim=0)
            v = v.repeat_interleave(self.num_kv_groups, dim=0)
            attn_output = NF.flash_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
                allow_vector_exp=False,
            )  # [Nh, Dh, T]
        return self._finish_prefill(attn_output)

    def _chunk_attention_torch(self, q, k, v, metadata: dict) -> torch.Tensor:
        """One chunk of one sequence against ``cached_seq_len`` cached tokens + itself.

        ``q`` ``[Nh, T, d]``, ``k``/``v`` ``[Nkv, T, d]`` -> ``[Nh, d, T]``. From the
        Qwen3.5 contrib's ``forward_prefill_chunk``: the prefix is gathered from
        the paged cache as it was before this step's write; unwritten page tails
        are selected away (not multiplied) because they may hold non-finite bits.
        """
        tokens = q.shape[1]
        nkh, groups, d = self.num_kv_heads_per_rank, self.num_kv_groups, self.head_dim
        block_table = metadata["block_table_tensor"][:1].to(torch.long)
        num_blocks, _, block_size = self.k_cache.shape[0], self.k_cache.shape[1], self.k_cache.shape[2]
        blocks = torch.where(
            (block_table > 0) & (block_table < num_blocks), block_table, torch.zeros_like(block_table)
        )
        ctx_len = blocks.shape[1] * block_size
        rows = (blocks.unsqueeze(1) * nkh + torch.arange(nkh, device=q.device).view(1, nkh, 1)).reshape(-1)
        k_ctx = self.k_cache.reshape(num_blocks * nkh, block_size, d)[rows].reshape(nkh, ctx_len, d)
        v_ctx = self.v_cache.reshape(num_blocks * nkh, block_size, d)[rows].reshape(nkh, ctx_len, d)
        prior = metadata["cached_seq_len"].reshape(1, 1).to(torch.long)
        in_prefix = torch.arange(ctx_len, device=q.device).view(1, ctx_len) < prior
        v_ctx = torch.where(in_prefix.view(1, ctx_len, 1), v_ctx, torch.zeros_like(v_ctx))

        qg = q.reshape(nkh, groups * tokens, d).float()
        keys = torch.cat([k_ctx.float(), k.float()], dim=1)
        values = torch.cat([v_ctx.float(), v.float()], dim=1)
        scores = (qg @ keys.transpose(-1, -2)) * self.scaling
        i = torch.arange(tokens, device=q.device).view(tokens, 1)
        j = torch.arange(tokens, device=q.device).view(1, tokens)
        visible = torch.cat([in_prefix.expand(tokens, ctx_len), j <= i], dim=1).repeat(groups, 1).unsqueeze(0)
        scores = scores.masked_fill(~visible, float("-inf"))
        out = (torch.softmax(scores, dim=-1) @ values).to(q.dtype)  # [nkh, groups*T, d]
        return out.reshape(self.num_heads_per_rank, tokens, d).transpose(1, 2)

    def _finish_prefill(self, attn_output: torch.Tensor) -> torch.Tensor:
        output = NF.o_proj(attn_output.unsqueeze(0), self.o_proj_weight, None).squeeze(0)  # [T, Hp]
        output = output[:, : self.hidden_size]
        if self.world_size > 1:
            if SEQUENCE_PARALLEL:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                output = self._all_reduce(output)
        return output.contiguous()

    def forward_decode(self, hidden_states, positions, attn_metadata: dict) -> torch.Tensor:
        metadata = attn_metadata[self.layer_name]
        slot_mapping = metadata["slot_mapping"]
        block_size = metadata["block_size"]
        block_table = metadata["block_table_tensor"]
        B = block_table.shape[0]
        tokens = hidden_states.shape[0]
        S = tokens // B
        S_ctx = metadata["max_blocks_per_seq"] * block_size
        if _DECODE_CTX_ALIGN > 0 and S_ctx % _DECODE_CTX_ALIGN:
            # R2-02: pad the kernel's table with -1 (inactive) columns to the aligned context; the
            # mask (built for the padded S_ctx from the positions) hides the extra columns.
            S_pad = -(-S_ctx // _DECODE_CTX_ALIGN) * _DECODE_CTX_ALIGN
            if S_pad % block_size == 0:
                block_table = F.pad(block_table, (0, S_pad // block_size - block_table.shape[1]), value=-1)
                S_ctx = S_pad
        X = self._pad_hidden(hidden_states.to(self.dtype).view(B, S, self.hidden_size))

        # Same for every attention layer of the step: build once, reuse.
        attention_mask = attn_metadata.get(_DECODE_MASK_CACHE_KEY)
        if attention_mask is None:
            attention_mask = NF.gen_attention_decode_mask(
                pos_ids=positions.view(1, B * S).to(torch.float32),
                bs=B,
                q_head=self.num_heads_per_rank,
                s_active=S,
                s_prior=S_ctx,
                start_pos=None,
                block_len=block_size,
            )
            attn_metadata[_DECODE_MASK_CACHE_KEY] = attention_mask

        # NEMOTRON_H_DECODE_KV_WRITE=torch (T11b): write this step's K/V with a torch scatter
        # and run the decode kernel with update_cache=False.
        torch_kv = os.environ.get("NEMOTRON_H_DECODE_KV_WRITE", "kernel") == "torch"
        if self.decode_pad_to_null:
            # Shared pool (T11c): the NKI decode kernel rewrites -1 table slots to real blocks
            # spread over the pool (nkilib _fill_inactive_block_slots_with_spread) and gathers
            # V there; a masked weight 0 times Mamba-state bits read as bf16 NaN is NaN. Point
            # padding at the null block 0 instead (zeroed by the runner, never a state page),
            # which is what the torch fallback reads for -1.
            block_table = torch.where(block_table < 0, torch.zeros_like(block_table), block_table)
        output = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=None,  # no RoPE in NemotronH
            sin=None,
            K_cache_transposed=False,
            active_blocks_table=block_table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling,
            sink=None,
            update_cache=not torch_kv,
            kv_cache_update_idx=slot_mapping.view(B, S).to(torch.uint32),
            fp8_packed=False,
            W_out=self.o_proj_weight,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
        )
        if torch_kv:
            # update_cache=False returns (output, K_out [d, B, S], V_out [B, 1, S, d]).
            output, k_out, v_out = output
            assert self.num_kv_heads_per_rank == 1
            k = k_out.permute(1, 2, 0).reshape(1, B * S, self.head_dim)
            v = v_out.reshape(1, B * S, self.head_dim)
            self._write_kv(k, v, slot_mapping.reshape(-1), block_size)
        output = output.reshape(tokens, self.hidden_padded)[:, : self.hidden_size]
        if self.world_size > 1:
            output = self._all_reduce(output)
        return output


class NemotronHBlock(nn.Module):
    """``x + mixer(norm(x))`` (HF ``NemotronHBlock``)."""

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.block_type = config.block_types[layer_idx]
        self.norm = NemotronHRMSNorm(config.hidden_size, config.norm_eps, config.torch_dtype)
        if self.block_type == MAMBA:
            self.mixer = NemotronHMamba2Mixer(config, layer_idx)
        elif self.block_type == ATTENTION:
            self.mixer = NemotronHAttention(config, layer_idx)
        elif self.block_type == MOE:
            self.mixer = NemotronHMoE(config, layer_idx)
        else:
            raise NotImplementedError(f"block type {self.block_type!r} ({MLP!r} layers are not ported)")

    def forward(self, hidden_states, positions, attn_metadata: dict, is_prefill: bool):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.block_type == MOE:
            hidden_states = self.mixer(hidden_states, is_prefill=is_prefill)
        else:
            hidden_states = self.mixer(hidden_states, positions, attn_metadata)
        return residual + hidden_states


class NemotronHBackbone(nn.Module):
    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.embeddings = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.embeddings.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embeddings.vocab_size_per_rank,
                num_shards=self.embeddings.tp_size,
                is_storage_transposed=False,
            ),
        )
        self.layers = nn.ModuleList(NemotronHBlock(config, i) for i in range(config.num_hidden_layers))
        self.norm_f = NemotronHRMSNorm(config.hidden_size, config.norm_eps, config.torch_dtype)
        # Any layer with metadata decides prefill vs decode for the step.
        self._probe_layer = next(
            layer.mixer.layer_name for layer in self.layers if layer.block_type in (MAMBA, ATTENTION)
        )

    def is_prefill(self, attn_metadata: dict) -> bool:
        m = attn_metadata[self._probe_layer]
        return m["max_query_len"] > m["decode_token_threshold"]

    def forward(self, input_ids, positions, attn_metadata: dict, rank=None) -> torch.Tensor:
        is_prefill = self.is_prefill(attn_metadata)
        hidden_states = self.embeddings(input_ids, scatter_tokens=is_prefill and SEQUENCE_PARALLEL, rank=rank)
        keep = None
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, positions, attn_metadata, is_prefill)
            if i == _DEBUG_EXIT_LAYER:
                keep = hidden_states
        if keep is not None:
            hidden_states = keep + hidden_states * 0.0
        hidden_states = self.norm_f(hidden_states)
        if is_prefill and self.world_size > 1 and SEQUENCE_PARALLEL:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


class NemotronHForCausalLM(nn.Module):
    """Text-only NemotronH. ``mtp.*`` tensors are ignored (no MTP head)."""

    # The Mamba state accepts the shared-pool K/V binding (runner VLLM_NEURON_STATE_KV_ALIAS,
    # used with prefix caching only; see NemotronHMamba2Mixer.bind_state).
    supports_state_kv_alias = True

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.backbone = NemotronHBackbone(config)

        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self._gather_logits = nc is not None and (nc.max_logprobs != 0 or nc.debug_logits_dir is not None)
        if config.vocab_size % self.world_size:
            raise ValueError(f"vocab_size={config.vocab_size} must be divisible by tp_size={self.world_size}")
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0, shard_size=config.vocab_size // self.world_size, num_shards=self.world_size
            ),
        )
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(self.on_device_sampling_config, process_group=self.tp_group.device_group)

    # ── KV / state cache ─────────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        layers: list[LayerSpec] = []
        state_layers: list[StateLayerSpec] = []
        conv_shape, ssm_shape = self.config.mamba_state_shapes(self.world_size)
        for layer in self.backbone.layers:
            mixer = layer.mixer
            if layer.block_type == MAMBA:
                if conv_shape != (mixer.conv_kernel_size - 1, mixer.conv_dim) or ssm_shape != (
                    mixer.num_heads,
                    mixer.head_dim,
                    mixer.d_state,
                ):
                    raise RuntimeError(
                        f"vLLM state shapes {conv_shape}/{ssm_shape} disagree with the mixer's"
                    )
                state_layers.append(
                    StateLayerSpec(
                        name=mixer.layer_name,
                        shapes=(conv_shape, ssm_shape),
                        # The runner's state pages are float32 throughout.
                        dtypes=(torch.float32, torch.float32),
                        mamba_type="MAMBA2",
                    )
                )
            elif layer.block_type == ATTENTION:
                layers.append(
                    LayerSpec(
                        name=mixer.layer_name,
                        num_kv_heads=mixer.num_kv_heads_per_rank,
                        head_size=mixer.head_dim,
                        dtype=mixer.dtype,
                        sliding_window_size=None,
                        chunk_size=None,
                    )
                )
        return KVSpec(layers=layers, state_layers=state_layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for layer in self.backbone.layers:
            if layer.block_type != ATTENTION:
                continue
            name = layer.mixer.layer_name
            if name not in kv_caches:
                raise KeyError(f"KV cache for layer {name} not initialized")
            layer.mixer.k_cache, layer.mixer.v_cache = kv_caches[name][:2]

    def bind_state_cache(self, state_caches: dict[str, list[torch.Tensor]]) -> None:
        for layer in self.backbone.layers:
            if layer.block_type != MAMBA:
                continue
            name = layer.mixer.layer_name
            if name not in state_caches:
                raise KeyError(f"State cache for layer {name} not initialized")
            layer.mixer.bind_state(*state_caches[name])
        # T11c: with the state in the attention K/V pool (shared pool, prefix caching), the decode
        # attention reads padding slots from the null block. NEMOTRON_H_DECODE_PAD_NULL:
        # auto (default: on for the shared pool only), 1, 0. The dedicated pool is unchanged.
        mode = os.environ.get("NEMOTRON_H_DECODE_PAD_NULL", "auto")
        shared = any(l.mixer.state_kv for l in self.backbone.layers if l.block_type == MAMBA)
        pad = shared if mode == "auto" else mode == "1"
        for layer in self.backbone.layers:
            if layer.block_type == ATTENTION:
                layer.mixer.decode_pad_to_null = pad

    def set_runner_clears_kv_blocks(self, value: bool) -> None:
        """The runner zeroes newly allocated attention blocks (and the null block)."""
        self.runner_clears_kv_blocks = value

    # ── Forward ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            raise NotImplementedError("NemotronH does not support prompt embeddings")
        positions = positions.to(torch.int32)
        num_tokens = input_ids.shape[0]
        if self.backbone.is_prefill(attn_metadata) and self.world_size > 1 and SEQUENCE_PARALLEL and (
            num_tokens % self.world_size
        ):
            raise ValueError(
                f"prefill length ({num_tokens}) must be a multiple of world_size "
                f"({self.world_size}) for sequence parallelism"
            )
        hidden_states = self.backbone(input_ids, positions, attn_metadata, rank=rank)
        hidden_states = torch.index_select(hidden_states, 0, sampling_positions)
        logits = self.lm_head(hidden_states)

        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)
        if self.on_device_sampling_config is None:
            return logits
        sampled_tokens = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled_tokens)
        return sampled_tokens, gathered_logits

    # ── Construction and weights ─────────────────────────────────────────

    @classmethod
    def from_configs(cls, hf_config, text_neuron_config=None, encoder_neuron_config=None):
        return cls(NemotronHConfig.from_configs(hf_config, text_neuron_config=text_neuron_config))

    def checkpoint_mappings(self) -> dict[str, object]:
        """Parameter name -> checkpoint key (or list of keys to fuse)."""
        m: dict[str, object] = {}
        for i, layer in enumerate(self.backbone.layers):
            ours = hf = f"backbone.layers.{i}"
            m[f"{ours}.norm.weight"] = f"{hf}.norm.weight"
            o, h = f"{ours}.mixer", f"{hf}.mixer"
            if layer.block_type == MAMBA:
                m[f"{o}.in_proj_weight"] = f"{h}.in_proj.weight"
                m[f"{o}.conv1d_weight"] = f"{h}.conv1d.weight"
                if layer.mixer.conv1d_bias is not None:
                    m[f"{o}.conv1d_bias"] = f"{h}.conv1d.bias"
                for name in ("dt_bias", "A_log", "D"):
                    m[f"{o}.{name}"] = f"{h}.{name}"
                m[f"{o}.norm_weight"] = f"{h}.norm.weight"
                m[f"{o}.out_proj_weight"] = f"{h}.out_proj.weight"
            elif layer.block_type == ATTENTION:
                m[f"{o}.qkv_proj_weight"] = [f"{h}.q_proj.weight", f"{h}.k_proj.weight", f"{h}.v_proj.weight"]
                m[f"{o}.o_proj_weight"] = f"{h}.o_proj.weight"
            elif layer.block_type == MOE:
                E = layer.mixer.num_experts
                m[f"{o}.router_weight"] = f"{h}.gate.weight"
                m[f"{o}.e_score_correction_bias"] = f"{h}.gate.e_score_correction_bias"
                m[f"{o}.experts_up_weight"] = [f"{h}.experts.{e}.up_proj.weight" for e in range(E)]
                m[f"{o}.experts_down_weight"] = [f"{h}.experts.{e}.down_proj.weight" for e in range(E)]
                if hasattr(layer.mixer, "experts_c"):  # NEMOTRON_H_FP8: scales from the up then down keys
                    if layer.mixer.fp8_shared:  # NEMOTRON_H_FP8_SHARED: the shared expert's key last in each list
                        m[f"{o}.experts_up_weight"] = m[f"{o}.experts_up_weight"] + [f"{h}.shared_experts.up_proj.weight"]
                        m[f"{o}.experts_down_weight"] = m[f"{o}.experts_down_weight"] + [
                            f"{h}.shared_experts.down_proj.weight"]
                    m[f"{o}.experts_c"] = m[f"{o}.experts_up_weight"] + m[f"{o}.experts_down_weight"]
                    m[f"{o}.experts_up_t"] = list(m[f"{o}.experts_up_weight"])
                if hasattr(layer.mixer, "experts_up_tkg"):
                    m[f"{o}.experts_up_tkg"] = [f"{h}.experts.{e}.up_proj.weight" for e in range(E)]
                m[f"{o}.shared_up_weight"] = f"{h}.shared_experts.up_proj.weight"
                m[f"{o}.shared_down_weight"] = f"{h}.shared_experts.down_proj.weight"
        m["backbone.embeddings.weight"] = "backbone.embeddings.weight"
        m["backbone.norm_f.weight"] = "backbone.norm_f.weight"
        m["lm_head.weight"] = (
            "backbone.embeddings.weight" if self.config.tie_word_embeddings else "lm_head.weight"
        )
        return m

    _KEEP_FP32 = {"dt_bias", "A_log", "D", "router_weight", "e_score_correction_bias", "experts_c"}

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self.checkpoint_mappings(), device, strict=False
        ).state_dict
        # strict=False leaves a mis-mapped parameter at torch.empty garbage; check.
        unfilled = sorted({n for n, _ in self.named_parameters()} - set(rank_sharded))
        if unfilled:
            raise RuntimeError(
                f"{len(unfilled)} parameter(s) got no checkpoint tensor: {unfilled[:8]}"
                + (" ..." if len(unfilled) > 8 else "")
            )
        target = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype == torch.float8_e4m3fn:  # NEMOTRON_H_FP8 expert weights (moe.py)
                continue
            want = torch.float32 if name.rsplit(".", 1)[-1] in self._KEEP_FP32 else target
            if tensor.dtype != want:
                rank_sharded[name] = tensor.to(want)
        self.load_state_dict(rank_sharded, strict=False, assign=True)


__all__ = ["NemotronHForCausalLM", "NemotronHBackbone"]
