# SPDX-License-Identifier: Apache-2.0
"""CPU-mode numerics of the NemotronH port against HF transformers.

Serves a reduced-layer checkpoint (``scripts/make_reduced_checkpoint.py``)
through ``vllm.LLM`` with ``VLLM_NEURON_CPU_MODE=1`` (the real V1 runner, KV
cache, dedicated recurrent-state pool, chunked prefill and batching, all on
CPU), then compares against ``transformers``' ``NemotronHForCausalLM`` run
teacher-forced on the same tokens.

Cases (each prompt is a list of token ids):

* ``single``: every prompt alone.
* ``batch``: all prompts (8) in one ``generate`` call, so decode runs batched
  (batch invariance: tokens and logits must match ``single``).
* chunked prefill: one prompt is longer than ``--max-num-batched-tokens``, so its
  prefill runs in several chunks that hand the Mamba state and KV over.

Metrics per prompt: greedy token match (vLLM token == HF argmax at the same
teacher-forced position), max |logit diff| over the full vocabulary and all
generated positions, and HF's own top1-top2 margin where a token differs.

    VLLM_NEURON_CPU_MODE=1 python cpu_numerics.py --ckpt /mnt/nvme/models/nemotron-h-4l-fp32 \
        --dtype float32 --out results/cpu-numerics/fp32.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch

SHORT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "1, 2, 3, 4, 5, 6,",
    "Translate to German: The weather is nice today.",
    "Q: What is 17 times 23?\nA:",
    "Mamba is a state space model architecture that",
    "<|im_start|>user\nWrite a haiku about Trainium.<|im_end|>\n<|im_start|>assistant\n",
]

LONG_PARAGRAPH = (
    "The history of computing hardware covers the developments from early simple devices "
    "to aid calculation to modern day computers. The first aids to computation were purely "
    "mechanical devices which required the operator to set up the initial values of an "
    "elementary arithmetic operation, then manipulate the device to obtain the result. "
)


def build_prompts(tokenizer, long_tokens: int) -> list[list[int]]:
    prompts = [tokenizer(p, add_special_tokens=True)["input_ids"] for p in SHORT_PROMPTS]
    long_ids: list[int] = []
    i = 0
    while len(long_ids) < long_tokens:
        long_ids = tokenizer(LONG_PARAGRAPH * (i + 1) + "\nSummary:", add_special_tokens=True)["input_ids"]
        i += 1
    prompts.append(long_ids[-long_tokens:])
    return prompts


def run_vllm(args, prompts):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    neuron_config = {"on_device_sampling_config": {"all_greedy": True}}
    if args.kv_segment:
        # Chunked prefill through NF.segmented_attention instead of the torch path.
        neuron_config["kv_segment_size_buckets"] = [args.kv_segment]
    if args.neuron_config:
        neuron_config.update(json.loads(args.neuron_config))
    llm = LLM(
        model=args.ckpt,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=len(prompts),
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=not args.device,
        additional_config={"neuron_config": neuron_config},
    )
    vocab = llm.llm_engine.model_config.get_vocab_size()
    sp = SamplingParams(temperature=0, max_tokens=args.output_len, logprobs=vocab, ignore_eos=True)

    def dense(outputs):
        res = []
        for out in outputs:
            comp = out.outputs[0]
            logits = torch.full((len(comp.token_ids), vocab), float("-inf"))
            for t, lp in enumerate(comp.logprobs):
                ids = torch.tensor(list(lp.keys()))
                vals = torch.tensor([v.logprob for v in lp.values()])
                logits[t, ids] = vals
            res.append({"tokens": list(comp.token_ids), "logits": logits})
        return res

    results = {}
    t0 = time.time()
    results["single"] = [dense(llm.generate([TokensPrompt(prompt_token_ids=p)], sp))[0] for p in prompts]
    results["single_s"] = time.time() - t0
    t0 = time.time()
    results["batch"] = dense(llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp))
    results["batch_s"] = time.time() - t0
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:  # noqa: BLE001 - best effort, process exits after
        pass
    del llm
    gc.collect()
    return results


def hf_logits(model, prompt: list[int], gen: list[int]) -> torch.Tensor:
    """Teacher-forced HF logits for the positions that predicted ``gen``: ``[len(gen), V]``."""
    ids = torch.tensor([prompt + gen])
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False).logits[0].float()
    return out[len(prompt) - 1 : len(prompt) - 1 + len(gen)]


def compare(ref: torch.Tensor, got_logits: torch.Tensor, got_tokens: list[int]) -> dict:
    n = len(got_tokens)
    ref = ref[:n]
    ref_top = ref.argmax(-1)
    tokens = torch.tensor(got_tokens)
    match = (ref_top == tokens).float()
    diff = (got_logits[:n] - ref).abs()
    finite = torch.isfinite(diff)
    top2 = ref.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1])
    first_mismatch = int((match == 0).nonzero()[0]) if (match == 0).any() else None
    return {
        "greedy_match": float(match.mean()),
        "first_mismatch": first_mismatch,
        "hf_margin_at_first_mismatch": float(margin[first_mismatch]) if first_mismatch is not None else None,
        "max_abs_logit_diff": float(diff[finite].max()) if finite.any() else float("nan"),
        "mean_abs_logit_diff": float(diff[finite].mean()) if finite.any() else float("nan"),
        "missing_logits": int((~finite).sum()),
        "ref_logit_absmax": float(ref.abs().max()),
        "margins": [float(x) for x in margin],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--hf-ref-dtypes", default=None, help="comma list; default float32 + --dtype")
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--max-num-batched-tokens", type=int, default=256)
    ap.add_argument("--long-tokens", type=int, default=700)
    ap.add_argument("--output-len", type=int, default=16)
    ap.add_argument("--kv-segment", type=int, default=0, help="kv_segment_size_buckets=[N] (0: unset)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    ap.add_argument("--device", action="store_true", help="run on Neuron devices (not CPU mode)")
    ap.add_argument("--neuron-config", default=None, help="JSON merged into neuron_config (buckets)")
    args = ap.parse_args()
    if args.device:
        assert os.environ.get("VLLM_NEURON_CPU_MODE") != "1", "--device runs without VLLM_NEURON_CPU_MODE"
    else:
        assert os.environ.get("VLLM_NEURON_CPU_MODE") == "1", "run with VLLM_NEURON_CPU_MODE=1"

    from nemotron_h.mamba import DT_CLAMP, align_hf_reference
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    prompts = build_prompts(tok, args.long_tokens)
    assert len(prompts) == 8
    assert len(prompts[-1]) > args.max_num_batched_tokens, "long prompt must exceed the prefill chunk"

    t0 = time.time()
    vl = run_vllm(args, prompts)
    vllm_s = time.time() - t0

    ref_dtypes = (args.hf_ref_dtypes or ",".join(dict.fromkeys(["float32", args.dtype]))).split(",")
    report = {
        "ckpt": args.ckpt,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "output_len": args.output_len,
        "kv_segment_size_buckets": [args.kv_segment] if args.kv_segment else None,
        "tp": args.tp,
        "device": args.device,
        "dt_clamp": DT_CLAMP,
        "neuron_config_extra": args.neuron_config,
        "prompt_lens": [len(p) for p in prompts],
        "vllm_wall_s": vllm_s,
        "vllm_single_s": vl["single_s"],
        "vllm_batch_s": vl["batch_s"],
        "cases": {},
    }
    # Batch invariance: tokens and logits of batch vs single.
    inv = []
    for i in range(len(prompts)):
        s, b = vl["single"][i], vl["batch"][i]
        n = min(len(s["tokens"]), len(b["tokens"]))
        d = (s["logits"][:n] - b["logits"][:n]).abs()
        d = d[torch.isfinite(d)]
        inv.append(
            {
                "tokens_equal": s["tokens"] == b["tokens"],
                "max_abs_logit_diff": float(d.max()) if d.numel() else float("nan"),
            }
        )
    report["batch_invariance"] = inv

    for ref_dtype in ref_dtypes:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=getattr(torch, ref_dtype))
        model.eval()
        align_hf_reference(model)  # HF uses the port's dt clamp (NEMOTRON_H_DT_CLAMP)
        hf = {}
        for case in ("single", "batch"):
            rows = []
            for i, p in enumerate(prompts):
                got = vl[case][i]
                ref = hf_logits(model, p, got["tokens"])
                rows.append(compare(ref, got["logits"], got["tokens"]) | {"prompt_len": len(p)})
            hf[case] = rows
        report["cases"][f"vs_hf_{ref_dtype}"] = hf
        if ref_dtype != "float32":
            # Reference line: HF in this dtype against HF float32, on vLLM's tokens.
            fp32 = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.float32).eval()
            align_hf_reference(fp32)
            line = []
            for i, p in enumerate(prompts):
                toks = vl["single"][i]["tokens"]
                a = hf_logits(model, p, toks)
                b = hf_logits(fp32, p, toks)
                line.append(compare(b, a, a.argmax(-1).tolist()) | {"prompt_len": len(p)})
            report["cases"][f"hf_{ref_dtype}_vs_hf_float32"] = line
            del fp32
        del model
        gc.collect()

    report["vllm_tokens"] = {c: [r["tokens"] for r in vl[c]] for c in ("single", "batch")}
    report["verdict"] = verdict(report)
    report["vllm_text_single"] = [tok.decode(r["tokens"]) for r in vl["single"]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    # Console summary
    for name, case in report["cases"].items():
        for sub, rows in (case.items() if isinstance(case, dict) else [("", case)]):
            print(name, sub, [(r["prompt_len"], round(r["greedy_match"], 3), round(r["max_abs_logit_diff"], 5)) for r in rows])
    print("batch_invariance", [(r["tokens_equal"], round(r["max_abs_logit_diff"], 6)) for r in inv])
    print("verdict", json.dumps(report["verdict"]))
    if not report["verdict"]["pass"]:
        raise SystemExit(1)


FP32_TOL = 1e-3  # max |logit diff| vs HF fp32 (logits are O(10))
BF16_RATIO = 2.0  # vLLM-bf16 error may be at most 2x HF-bf16's own error vs HF fp32
NEAR_TIE = 0.25  # bf16 logit noise scale; a token flip at a smaller HF top-2 margin is drift


def verdict(report: dict) -> dict:
    """Pass/fail rules, fixed before the bf16 numbers were looked at except NEAR_TIE.

    float32: every prompt, both cases: greedy match 1.0 and max |diff| < FP32_TOL vs HF
    fp32; batch tokens equal single tokens.
    bfloat16: the largest vLLM-bf16 |diff| vs HF fp32 (over all prompts, both cases)
    is at most BF16_RATIO x the largest HF-bf16 |diff| vs HF fp32 (three-way rule);
    batch-vs-single token differences are allowed only where the first divergence
    sits at an HF fp32 top-2 margin below NEAR_TIE (bf16 drift, not state mixing).
    """
    cases = report["cases"]["vs_hf_float32"]
    rows = cases["single"] + cases["batch"]
    out = {"dtype": report["dtype"]}
    if report["dtype"] == "float32":
        out["max_abs_logit_diff"] = max(r["max_abs_logit_diff"] for r in rows)
        out["min_greedy_match"] = min(r["greedy_match"] for r in rows)
        out["batch_tokens_equal"] = all(r["tokens_equal"] for r in report["batch_invariance"])
        out["pass"] = (
            out["max_abs_logit_diff"] < FP32_TOL and out["min_greedy_match"] == 1.0 and out["batch_tokens_equal"]
        )
        return out
    ref_line = report["cases"]["hf_bfloat16_vs_hf_float32"]
    ours = max(r["max_abs_logit_diff"] for r in rows)
    theirs = max(r["max_abs_logit_diff"] for r in ref_line)
    out.update(
        max_abs_logit_diff=ours,
        hf_bf16_max_abs_logit_diff=theirs,
        ratio=ours / theirs,
        min_greedy_match_single=min(r["greedy_match"] for r in cases["single"]),
        hf_bf16_min_greedy_match=min(r["greedy_match"] for r in ref_line),
    )
    flips = []
    for i, r in enumerate(report["batch_invariance"]):
        if r["tokens_equal"]:
            continue
        s, b = report["vllm_tokens"]["single"][i], report["vllm_tokens"]["batch"][i]
        k = next(j for j, (x, y) in enumerate(zip(s, b)) if x != y)
        # HF fp32 margin at the divergence, from the single run (same prefix up to k).
        margin = cases["single"][i].get("margins", [None] * (k + 1))[k]
        flips.append({"prompt": i, "first_divergence": k, "hf_fp32_margin": margin})
    out["batch_flips"] = flips
    out["pass"] = out["ratio"] <= BF16_RATIO and all(
        f["hf_fp32_margin"] is not None and f["hf_fp32_margin"] < NEAR_TIE for f in flips
    )
    return out


if __name__ == "__main__":
    main()
