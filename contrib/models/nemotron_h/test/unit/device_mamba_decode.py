# SPDX-License-Identifier: Apache-2.0
"""On-device unit test and timing: one Mamba layer's ``forward_decode``, kernel on vs off.

Builds ``NemotronHMamba2Mixer`` with the TP=4 rank-0 shapes (16 heads, 2 groups)
at world size 1, loads the real layer-0 weights of that shard, binds random
state pages on the Neuron device, and compiles ``forward_decode`` with the
native backend twice: ``NEMOTRON_H_MAMBA_DECODE_KERNEL`` off (the port's torch
path incl. the state gather/scatter kernels) and on (the fused kernel). Both
include ``in_proj`` and ``out_proj``. Checks each against a CPU float32
reference (``test_mamba_decode_kernel.torch_decode`` + ``out_proj``), including
the in-place state write, then times steps.

    NEURON_RT_VISIBLE_CORES=4 python device_mamba_decode.py --batches 1,8,32,128 --out x.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_mamba_decode_kernel import Params, make_case, read_pages, torch_decode  # noqa: E402


def build_mixer(p: Params, device):
    from harness import init_dist

    init_dist()
    from nemotron_h import mamba as M

    cfg = SimpleNamespace(
        torch_dtype=torch.bfloat16, hidden_size=p.hidden, norm_eps=p.eps, conv_kernel=p.K,
        chunk_size=128, time_step_min=1e-3, use_conv_bias=True, mamba_num_heads=p.nh,
        n_groups=p.G, mamba_head_dim=p.P, ssm_state_size=p.N, mamba_intermediate_size=p.I,
    )
    m = M.NemotronHMamba2Mixer(cfg, 0)
    with torch.no_grad():
        m.in_proj_weight.copy_(p.in_proj)
        m.conv1d_weight.copy_(p.conv_w)
        m.conv1d_bias.copy_(p.conv_b)
        m.dt_bias.copy_(p.dt_bias)
        m.A_log.copy_(p.A_log)
        m.D.copy_(p.D)
        m.norm_weight.copy_(p.norm_w)
        m.out_proj_weight.copy_(p.out_proj)
    m.requires_grad_(False)
    return M, m.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--half-rows", type=int, default=530)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--modes", default="off,on")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    import libtorch_neuronx_lite  # noqa: F401  (registers the neuron device)
    from libtorch_neuronx_lite.compile.native_backend import register

    backend = register()
    dev = torch.device("neuron:0")
    p = Params()
    from safetensors import safe_open

    # out_proj shard of rank 0: [I_full, H] rows of this rank, stored transposed.
    idx = json.load(open(os.path.join(os.environ.get("MODEL_DIR", "/mnt/nvme/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"), "model.safetensors.index.json")))["weight_map"]
    key = "backbone.layers.0.mixer.out_proj.weight"
    md = os.environ.get("MODEL_DIR", "/mnt/nvme/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    with safe_open(os.path.join(md, idx[key]), "pt") as f:
        w = f.get_tensor(key)  # [H, I_full]
    p.out_proj = w[:, : p.I].T.contiguous()  # [I, H]

    M, mixer = build_mixer(p, dev)
    results = []
    for n in [int(x) for x in args.batches.split(",")]:
        num_pages = n + 4
        n_pad = 1 if n > 2 else 0
        proj_unused, sa, sb, pages, wpages, live = make_case(p, n, args.half_rows, num_pages, n_pad, seed=1000 + n)
        g = torch.Generator().manual_seed(n)
        hidden = (torch.randn(n, p.hidden, generator=g)).to(torch.bfloat16)
        proj = hidden @ p.in_proj
        conv0, ssm0 = read_pages(p, sa, sb, pages)
        y_ref, conv_ref, ssm_ref = torch_decode(p, proj, conv0, ssm0, fp32=True)
        out_ref = (y_ref @ p.out_proj).float()
        y16, _, ssm16 = torch_decode(p, proj, conv0, ssm0, fp32=False)
        out16 = (y16 @ p.out_proj).float()
        slot = torch.where(live, torch.arange(1, n + 1), torch.zeros(n, dtype=torch.long))
        bt = pages.view(n, 1).to(torch.int32)
        row = {"n": n, "half_rows": args.half_rows, "n_pad": n_pad,
               "torch_bf16_vs_fp32_out_max_abs": (out16 - out_ref)[live].abs().max().item(),
               "out_ref_max": out_ref[live].abs().max().item()}
        for mode in args.modes.split(","):
            M.MAMBA_DECODE_KERNEL = "1" if mode == "on" else "0"
            torch._dynamo.reset()
            st_a, st_b = sa.clone().reshape(num_pages, -1).to(dev), sb.clone().reshape(num_pages, -1).to(dev)
            mixer.bind_state(st_a, st_b)

            def step(h, bt, slot):
                return mixer.forward_decode(h, {"block_table_tensor": bt, "slot_mapping": slot})

            f = torch.compile(step, backend=backend, fullgraph=True)
            h_d, bt_d, slot_d = hidden.to(dev), bt.to(dev), slot.to(dev)
            t0 = time.time()
            out = f(h_d, bt_d, slot_d)
            torch.neuron.synchronize()
            compile_s = time.time() - t0
            out_c = out.cpu().float()
            sa_c, sb_c = mixer.state_a.cpu().reshape(sa.shape), mixer.state_b.cpu().reshape(sb.shape)
            conv_k, ssm_k = read_pages(p, sa_c, sb_c, torch.where(live, pages, torch.zeros_like(pages)))
            d_out = (out_c - out_ref)[live].abs()
            r = {
                "compile_and_first_s": round(compile_s, 1),
                "out_max_abs_vs_fp32": d_out.max().item(),
                "out_mean_abs_vs_fp32": d_out.mean().item(),
                "out_max_abs_vs_torch_bf16": (out_c - out16)[live].abs().max().item(),
                "ssm_max_abs_vs_fp32": (ssm_k - ssm_ref)[live].abs().max().item(),
                "ssm_max_abs_vs_torch_bf16": (ssm_k - ssm16)[live].abs().max().item(),
                "conv_max_abs_vs_ref": (conv_k - conv_ref)[live].abs().max().item(),
                "finite": bool(torch.isfinite(out_c).all()),
            }
            # Timing (state keeps evolving in place; that does not change the work).
            for _ in range(4):
                f(h_d, bt_d, slot_d).cpu()
            t0 = time.time()
            print("CHECK", n, mode, json.dumps(r), flush=True)
            # .cpu() blocks until the step ran (torch.neuron.synchronize() does not wait
            # for the async queue: Execution Queue Full); the copy is the same in both modes.
            for it in range(args.iters):
                f(h_d, bt_d, slot_d).cpu()
            r["step_ms"] = 1000 * (time.time() - t0) / args.iters
            row[mode] = r
            print("ROW", n, mode, json.dumps(r), flush=True)
        # Fixed cost of one compiled call + .cpu() of an output of the same size.
        torch._dynamo.reset()
        null = torch.compile(lambda h: (h[:, :1] * 2).expand(n, p.hidden).contiguous(), backend=backend, fullgraph=True)
        for _ in range(4):
            null(h_d).cpu()
        t0 = time.time()
        for _ in range(args.iters):
            null(h_d).cpu()
        row["null_call_ms"] = 1000 * (time.time() - t0) / args.iters
        if "on" in row and "off" in row:
            row["speedup_off_over_on"] = row["off"]["step_ms"] / row["on"]["step_ms"]
        results.append(row)
        json.dump({"results": results, "iters": args.iters,
                   "env": {k: os.environ.get(k) for k in ("NEURON_RT_VISIBLE_CORES", "NEMOTRON_H_STATE_KERNEL", "NEMOTRON_H_DECODE_Y", "NEMOTRON_H_MAMBA_DECODE_DMA")}},
                  open(args.out, "w"), indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
