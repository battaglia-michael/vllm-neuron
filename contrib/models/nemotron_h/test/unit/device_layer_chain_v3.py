# SPDX-License-Identifier: Apache-2.0
"""Per-layer device time of one Mamba layer's ``forward_decode`` (in_proj + fused kernel + out_proj), by state-DMA mode (A05b).

Same layer as ``device_mamba_decode.py`` (TP=4 rank-0 shapes, real layer-0 weights,
state half 640 rows). For each ``NEMOTRON_H_MAMBA_DECODE_DMA`` value two graphs are
compiled: one layer step and ``--chain`` steps in sequence (``h = h + layer(h)``, the
state is updated in place each step). Per-layer time = (t_chain - t_1) / (chain - 1),
which removes the dispatch and ``.cpu()`` overhead; graphs are timed in interleaved
rounds and the median is reported. The 1-step outputs and states of all modes are
compared with the first mode's (bit-identical expected) and with the CPU fp32
reference.

    NEMOTRON_H_MAMBA_DECODE_DMA=v3 NEURON_RT_VISIBLE_CORES=4 python device_layer_chain_v3.py --modes v3 --out x.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types

import torch

sys.path.insert(0, os.path.dirname(__file__))
from device_mamba_decode import build_mixer  # noqa: E402
from test_mamba_decode_kernel import Params, make_case, read_pages, torch_decode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--modes", default="v2,v3")
    ap.add_argument("--chain", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--half-rows", type=int, default=640)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    import libtorch_neuronx_lite  # noqa: F401
    from libtorch_neuronx_lite.compile.native_backend import register

    from nemotron_h.kernels import mamba_decode as MD

    backend = register()
    torch._dynamo.config.recompile_limit = 10_000
    torch._dynamo.config.accumulated_recompile_limit = 100_000
    dev = torch.device("neuron:0")
    p = Params()
    from safetensors import safe_open

    md = os.environ.get("MODEL_DIR", "/mnt/nvme/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    idx = json.load(open(os.path.join(md, "model.safetensors.index.json")))["weight_map"]
    key = "backbone.layers.0.mixer.out_proj.weight"
    with safe_open(os.path.join(md, idx[key]), "pt") as f:
        w = f.get_tensor(key)
    p.out_proj = w[:, : p.I].T.contiguous()
    M, mixer = build_mixer(p, dev)
    M.MAMBA_DECODE_KERNEL = "1"
    modes = args.modes.split(",")
    # One mode per process: in one process the second mode's graph reused the first
    # mode's kernel (same compile, same time, A05b), so the mode comes from the env.
    if len(modes) != 1 or modes[0] != MD.DECODE_DMA:
        raise SystemExit("run one mode per process with NEMOTRON_H_MAMBA_DECODE_DMA=<mode> --modes <mode>")
    out = {"chain": args.chain, "iters": args.iters, "rounds": args.rounds, "half_rows": args.half_rows,
           "cores": os.environ.get("NEURON_RT_VISIBLE_CORES"), "results": []}
    for n in [int(x) for x in args.batches.split(",")]:
        num_pages = n + 4
        n_pad = 1 if n > 2 else 0
        _, sa, sb, pages, wpages, live = make_case(p, n, args.half_rows, num_pages, n_pad, seed=1000 + n)
        g = torch.Generator().manual_seed(n)
        hidden = torch.randn(n, p.hidden, generator=g).to(torch.bfloat16)
        proj = hidden @ p.in_proj
        conv0, ssm0 = read_pages(p, sa, sb, pages)
        y_ref, _, ssm_ref = torch_decode(p, proj, conv0, ssm0, fp32=True)
        out_ref = (y_ref @ p.out_proj).float()
        slot = torch.where(live, torch.arange(1, n + 1), torch.zeros(n, dtype=torch.long))
        bt = pages.view(n, 1).to(torch.int32)
        h_d, bt_d, slot_d = hidden.to(dev), bt.to(dev), slot.to(dev)
        st_a, st_b = sa.clone().reshape(num_pages, -1).to(dev), sb.clone().reshape(num_pages, -1).to(dev)
        mixer.bind_state(st_a, st_b)
        torch._dynamo.reset()
        fns, checks, first = {}, {}, None
        for mi, mode in enumerate(modes):
            MD.DECODE_DMA = mode  # read by kernel_args at trace time
            for L in (1, args.chain):

                def step(h, bt, slot, _L=L):
                    meta = {"block_table_tensor": bt, "slot_mapping": slot}
                    if _L == 1:
                        return mixer.forward_decode(h, meta)
                    for _ in range(_L):
                        h = h + mixer.forward_decode(h, meta)
                    return h

                step = types.FunctionType(step.__code__.replace(co_name=f"step_{mi}_{L}"), step.__globals__,
                                          None, step.__defaults__, step.__closure__)
                f = torch.compile(step, backend=backend, fullgraph=True)
                if L == 1:
                    mixer.state_a.copy_(sa.to(dev).reshape(mixer.state_a.shape))
                    mixer.state_b.copy_(sb.to(dev).reshape(mixer.state_b.shape))
                t0 = time.time()
                o = f(h_d, bt_d, slot_d).cpu().float()
                comp = round(time.time() - t0, 1)
                if L == 1:
                    sa_c, sb_c = mixer.state_a.cpu().reshape(sa.shape), mixer.state_b.cpu().reshape(sb.shape)
                    _, ssm_k = read_pages(p, sa_c, sb_c, torch.where(live, pages, torch.zeros_like(pages)))
                    if first is None:
                        first = (o, sa_c, sb_c)
                    checks[mode] = {
                        "compile_s": comp,
                        "out_max_abs_vs_fp32": (o - out_ref)[live].abs().max().item(),
                        "ssm_max_abs_vs_fp32": (ssm_k - ssm_ref)[live].abs().max().item(),
                        "out_max_abs_vs_first_mode": (o - first[0]).abs().max().item(),
                        "state_max_abs_vs_first_mode": max((sa_c - first[1]).abs().max().item(), (sb_c - first[2]).abs().max().item()),
                    }
                    print("CHECK", n, mode, json.dumps(checks[mode]), flush=True)
                for _ in range(3):
                    f(h_d, bt_d, slot_d).cpu()
                fns[(mode, L)] = f
        times = {k: [] for k in fns}
        for _ in range(args.rounds):
            for k, f in fns.items():
                f(h_d, bt_d, slot_d).cpu()
                t0 = time.time()
                for _ in range(args.iters):
                    f(h_d, bt_d, slot_d).cpu()
                times[k].append(1000 * (time.time() - t0) / args.iters)
        for mode in modes:
            per = [(b - a) / (args.chain - 1) for a, b in zip(times[(mode, 1)], times[(mode, args.chain)])]
            t1, tL = statistics.median(times[(mode, 1)]), statistics.median(times[(mode, args.chain)])
            r = {"n": n, "mode": mode, "layer_ms": (tL - t1) / (args.chain - 1), "layer_ms_per_round": per,
                 "call1_ms_median": t1, "chain_ms_median": tL, "check": checks[mode]}
            print("RES", json.dumps(r), flush=True)
            out["results"].append(r)
        json.dump(out, open(args.out, "w"), indent=1)
        del fns


if __name__ == "__main__":
    main()
