# SPDX-License-Identifier: Apache-2.0
"""Decode logits vs stock HF's full-sequence forward (which clamps dt in its chunked
scan), with and without the dt clamp in the port. --align-hf makes HF use the port's
setting instead (time_step_limit (0, inf) for "none").

    NEMOTRON_H_DT_CLAMP=none VLLM_NEURON_CPU_MODE=1 python dt_clamp_ablation.py --ckpt ... --out x.json [--align-hf]
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--align-hf", action="store_true", help="patch HF to the port's dt clamp")
    a = ap.parse_args()
    from harness import Harness
    from transformers import AutoModelForCausalLM
    from nemotron_h.mamba import DT_CLAMP, align_hf_reference

    h = Harness(a.ckpt)
    hf = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.float32).eval()
    if a.align_hf:
        align_hf_reference(hf)
    prompt = [1, 1784, 6129, 1307, 6233, 1395]
    seq = list(prompt)
    nxt = int(h.prefill(prompt, 0)[-1].argmax())
    rows = []
    for _ in range(8):
        seq.append(nxt)
        d = h.decode([nxt], [len(seq) - 1], [0])[0]
        with torch.no_grad():
            ref = hf(torch.tensor([seq]), use_cache=False).logits[0, -1].float()
        rows.append({"max_abs_logit_diff": (d - ref).abs().max().item(), "token_match": int(d.argmax()) == int(ref.argmax())})
        nxt = int(ref.argmax())  # teacher-force along HF
    out = {"NEMOTRON_H_DT_CLAMP": DT_CLAMP, "hf_aligned": a.align_hf, "ckpt": a.ckpt, "decode_steps": rows,
           "max_abs_logit_diff": max(r["max_abs_logit_diff"] for r in rows)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
