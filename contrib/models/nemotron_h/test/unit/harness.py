# SPDX-License-Identifier: Apache-2.0
"""Direct (no vLLM engine) CPU harness for the NemotronH port.

Builds ``NemotronHForCausalLM`` at world size 1 on CPU, binds hand-made KV and
state caches, and drives prefill / decode with the same metadata dicts the V1
runner passes. Used by the unit tests to compare individual layers and steps
against HF ``transformers`` without the engine in the way.
"""

from __future__ import annotations

import os

import torch

_DIST_READY = False


def init_dist() -> None:
    global _DIST_READY
    if _DIST_READY:
        return
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    port = int(os.environ.get("NEMOTRON_H_TEST_PORT", "29571"))
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
        initialize_model_parallel(1)
    _DIST_READY = True


class Harness:
    """One sequence at a time (or a decode batch), with caches it owns."""

    def __init__(self, ckpt: str, block_size: int = 128, num_blocks: int = 64, num_pages: int = 16):
        from transformers import AutoConfig

        from nemotron_h.model import NemotronHForCausalLM

        init_dist()
        self.hf_config = AutoConfig.from_pretrained(ckpt)
        self.model = NemotronHForCausalLM.from_configs(self.hf_config)
        self.model.load_weights(ckpt, torch.device("cpu"), None)
        self.model.eval()
        self.block_size = block_size
        spec = self.model.get_kv_spec()
        kv = {}
        for layer in spec.layers:
            shape = (num_blocks, layer.num_kv_heads, block_size, layer.head_size)
            kv[layer.name] = [torch.zeros(shape, dtype=layer.dtype), torch.zeros(shape, dtype=layer.dtype)]
        self.model.bind_kv_cache(kv)
        states = {}
        for s in spec.state_layers:
            numel = sum(int(torch.tensor(sh).prod()) for sh in s.shapes)
            half = -(-numel // 2)
            half = -(-half // 128) * 128
            # Optional larger halves (rows of 128 floats), e.g. 640 = the server's TP=4 page,
            # which leaves room for the aligned layout (NEMOTRON_H_STATE_ALIGN=1).
            half = max(half, 128 * int(os.environ.get("NEMOTRON_H_TEST_STATE_HALF_ROWS", "0")))
            states[s.name] = [torch.zeros(num_pages, half), torch.zeros(num_pages, half)]
        self.model.bind_state_cache(states)
        self.attn_names = [l.name for l in spec.layers]
        self.state_names = [s.name for s in spec.state_layers]

    def _metadata(self, *, max_query_len, slot_mapping, attn_block_table, state_block_table, cached_seq_len=None, kv_segment_size=None):
        md = {}
        for name in self.attn_names:
            d = {
                "max_query_len": max_query_len,
                "decode_token_threshold": 1,
                "slot_mapping": slot_mapping,
                "block_size": self.block_size,
                "block_table_tensor": attn_block_table,
                "max_blocks_per_seq": attn_block_table.shape[1],
            }
            if cached_seq_len is not None:
                d["cached_seq_len"] = cached_seq_len
            if kv_segment_size is not None:
                d["kv_segment_size"] = kv_segment_size
            md[name] = d
        for name in self.state_names:
            d = {
                "max_query_len": max_query_len,
                "decode_token_threshold": 1,
                "slot_mapping": slot_mapping,
                "block_table_tensor": state_block_table,
            }
            if cached_seq_len is not None:
                d["cached_seq_len"] = cached_seq_len
            md[name] = d
        return md

    def prefill(self, ids: list[int], seq_slot: int, start: int = 0, pad_to: int | None = None, chunked: bool = False, kv_segment_size=None):
        """Prefill ``ids`` (tokens ``start..start+len``) of the sequence in ``seq_slot``.

        Sequence ``s`` owns attention blocks ``1 + s*8 ..`` and state page ``1 + s``.
        Returns the logits of every real token.
        """
        n = len(ids)
        t = pad_to or n
        pos = torch.arange(start, start + n, dtype=torch.int32)
        input_ids = torch.tensor(ids + [0] * (t - n), dtype=torch.long)
        positions = torch.cat([pos, pos[-1:].repeat(t - n)])
        blocks = torch.arange(1 + seq_slot * 8, 1 + seq_slot * 8 + 8, dtype=torch.int32).view(1, -1)
        slots = blocks[0].long()[positions.long() // self.block_size] * self.block_size + positions.long() % self.block_size
        slots[n:] = -1
        md = self._metadata(
            max_query_len=t,
            slot_mapping=slots,
            attn_block_table=blocks,
            state_block_table=torch.tensor([[1 + seq_slot]]),
            cached_seq_len=torch.tensor([start], dtype=torch.int32) if chunked else None,
            kv_segment_size=kv_segment_size,
        )
        logits = self.model(input_ids, positions, attn_metadata=md, sampling_positions=torch.arange(n))
        return logits

    def decode(self, tokens: list[int], positions: list[int], seq_slots: list[int]):
        """One decode step for a batch of sequences. Returns ``[B, V]`` logits."""
        blocks = torch.stack([torch.arange(1 + s * 8, 1 + s * 8 + 8) for s in seq_slots]).to(torch.int32)
        pos = torch.tensor(positions, dtype=torch.int32)
        slots = blocks.long()[torch.arange(len(seq_slots)), pos.long() // self.block_size] * self.block_size + pos.long() % self.block_size
        md = self._metadata(
            max_query_len=1,
            slot_mapping=slots,
            attn_block_table=blocks,
            state_block_table=torch.tensor([[1 + s] for s in seq_slots]),
        )
        return self.model(torch.tensor(tokens), pos, attn_metadata=md, sampling_positions=torch.arange(len(tokens)))
