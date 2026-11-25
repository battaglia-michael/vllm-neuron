# SPDX-License-Identifier: Apache-2.0
"""TP=N on CPU: N gloo processes run the sharded model and compare with HF.

The vLLM CPU mode is single-process (world size 1), so it never exercises the
per-rank weight loaders (Mamba segment sharding, KV-head replication at TP=4,
expert intermediate sharding), the sequence-parallel collectives, or the
per-rank state shapes. This script does: each rank builds the model with its
own TP rank, loads its shards, and runs prefill (padded, sequence-parallel),
chunked prefill and decode; rank 0 compares the logits with HF.

    VLLM_NEURON_CPU_MODE=1 python tp_sim.py --ckpt /mnt/nvme/models/nemotron-h-4l-fp32 \
        --tp 4 --out results/cpu-numerics/tp4-sim.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))


def worker(rank: int, args) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    # The neuron PrivateUse1 backend is registered but has no device here, and
    # gloo's barrier asks for the current accelerator: answer CPU.
    torch._C._get_accelerator = lambda: torch.device("cpu")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=args.tp,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{args.port}",
            backend="gloo",
        )
        initialize_model_parallel(args.tp)
    import harness

    harness._DIST_READY = True
    h = harness.Harness(args.ckpt)
    m = h.model
    report = {"tp": args.tp, "rank": rank}
    mixers = {l.block_type: l.mixer for l in m.backbone.layers}
    report["per_rank"] = {
        "mamba_heads": mixers["mamba"].num_heads,
        "mamba_groups": mixers["mamba"].n_groups,
        "attn_q_heads": mixers["attention"].num_heads_per_rank,
        "attn_kv_heads": mixers["attention"].num_kv_heads_per_rank,
        "attn_kv_replicas": mixers["attention"].num_kv_replicas,
        "moe_inter": mixers["moe"].inter,
        "state_shapes": [list(map(list, s.shapes)) for s in m.get_kv_spec().state_layers][:1],
    }

    prompt = [1, 1784, 6129, 1307, 6233, 1395]
    long = list(range(1000, 1300))
    out = {}
    out["prefill_pad128"] = h.prefill(prompt, 0, pad_to=128)
    seq = list(prompt)
    nxt = int(out["prefill_pad128"][-1].argmax())
    dec = []
    for _ in range(3):
        seq.append(nxt)
        d = h.decode([nxt], [len(seq) - 1], [0])[0]
        dec.append(d)
        nxt = int(d.argmax())
    out["decode"] = torch.stack(dec)
    out["seq"] = seq
    out["chunk_a"] = h.prefill(long[:128], 1, start=0, chunked=True)
    out["chunk_b"] = h.prefill(long[128:], 1, start=128, chunked=True, pad_to=256)
    out["seg_a"] = h.prefill(long[:128], 2, start=0, chunked=True, kv_segment_size=512)
    out["seg_b"] = h.prefill(long[128:], 2, start=128, chunked=True, pad_to=256, kv_segment_size=512)
    # Batched decode: two sequences in one step (slots 3 and 4).
    p2 = [1, 422, 5544, 1278, 4200, 1044, 1278]
    t1 = int(h.prefill(prompt, 3, pad_to=128)[-1].argmax())
    t2 = int(h.prefill(p2, 4, pad_to=128)[-1].argmax())
    out["batch_decode"] = h.decode([t1, t2], [len(prompt), len(p2)], [3, 4])
    out["batch_seqs"] = (prompt + [t1], p2 + [t2])

    if rank != 0:
        return
    from transformers import AutoModelForCausalLM

    from nemotron_h.mamba import align_hf_reference

    hf = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=m.config.torch_dtype).eval()
    align_hf_reference(hf)

    def ref(ids):
        with torch.no_grad():
            return hf(torch.tensor([ids]), use_cache=False).logits[0].float()

    r_long = ref(long)
    r_seq = ref(seq)
    diffs = {
        "prefill_pad128": (out["prefill_pad128"] - ref(prompt)).abs().max().item(),
        "decode": (out["decode"] - r_seq[len(prompt) : len(prompt) + 3]).abs().max().item(),
        "decode_tokens_match": [int(x) for x in out["decode"].argmax(-1)]
        == [int(x) for x in r_seq[len(prompt) : len(prompt) + 3].argmax(-1)],
        "chunked_torch": max(
            (out["chunk_a"] - r_long[:128]).abs().max().item(), (out["chunk_b"] - r_long[128:]).abs().max().item()
        ),
        "chunked_segmented": max(
            (out["seg_a"] - r_long[:128]).abs().max().item(), (out["seg_b"] - r_long[128:]).abs().max().item()
        ),
        "batch_decode": max(
            (out["batch_decode"][i] - ref(list(out["batch_seqs"][i]))[-1]).abs().max().item() for i in range(2)
        ),
        "ref_logit_absmax": r_long.abs().max().item(),
    }
    report["sequence_parallel"] = __import__("nemotron_h.mamba", fromlist=["x"]).SEQUENCE_PARALLEL
    report["max_abs_logit_diff"] = diffs
    report["pass"] = all(v < 1e-3 for k, v in diffs.items() if k not in ("ref_logit_absmax", "decode_tokens_match")) and diffs["decode_tokens_match"]
    print(json.dumps(report, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)
    if not report["pass"]:
        raise SystemExit("TP simulation FAILED (see max_abs_logit_diff)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--port", type=int, default=29581)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=args.tp, join=True)


if __name__ == "__main__":
    main()
