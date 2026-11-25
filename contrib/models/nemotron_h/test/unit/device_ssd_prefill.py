# SPDX-License-Identifier: Apache-2.0
"""On-device unit test and timing: one Mamba layer's ``forward_prefill``, kernel 2 on vs off (A06).

Builds ``NemotronHMamba2Mixer`` with the TP=4 rank-0 shapes (16 heads, 2 groups)
at world size 1, loads the real weights of one layer's rank-0 shard, binds
random state pages on the Neuron device, and compiles ``forward_prefill`` (one
2,048-token chunk incl. in_proj, conv, scan, gated norm, out_proj and the state
write) with the native backend twice: ``NEMOTRON_H_KERNEL2`` off (torch
``ssd_chunk_scan``) and on (``kernels/ssd_prefill.py``). Both are compared with
the same module run on CPU (torch path, bf16 weights, what flag-off computes)
and with a float64 sequential recurrence of the scan (``reference64``). Cases:
cold first chunk (no cached state) and warm (continuing from the state page).

    NEURON_RT_VISIBLE_CORES=4 python device_ssd_prefill.py --layers 0,48 --out x.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from test_mamba_decode_kernel import MODEL, Params  # noqa: E402
from test_ssd_prefill_kernel import reference64  # noqa: E402


def load_out_proj(p: Params, layer: int):
    from safetensors import safe_open

    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    key = f"backbone.layers.{layer}.mixer.out_proj.weight"
    with safe_open(os.path.join(MODEL, idx[key]), "pt") as f:
        w = f.get_tensor(key)  # [H, I_full]
    return w[:, : p.I].T.contiguous()


def build_mixer(M, p: Params):
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
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,48")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--n-pad", type=int, default=100)
    ap.add_argument("--half-rows", type=int, default=640)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--modes", default="off,on")
    ap.add_argument("--cases", default="cold,warm")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    import libtorch_neuronx_lite  # noqa: F401  (registers the neuron device)
    from libtorch_neuronx_lite.compile.native_backend import register

    from harness import init_dist

    init_dist()
    from nemotron_h import mamba as M

    backend = register()
    dev = torch.device("neuron:0")
    T, R = args.T, 128
    results = []
    for layer in [int(x) for x in args.layers.split(",")]:
        p = Params(layer=layer)
        p.out_proj = load_out_proj(p, layer)
        cpu_mixer = build_mixer(M, p)
        dev_mixer = build_mixer(M, p).to(dev)
        num_pages = 6
        page_floats = 2 * args.half_rows * R
        g = torch.Generator().manual_seed(layer)
        base_a = torch.randn(num_pages, args.half_rows * R, generator=g) * 0.05
        base_b = torch.randn(num_pages, args.half_rows * R, generator=g) * 0.05
        hidden = torch.randn(T, p.hidden, generator=g).to(torch.bfloat16)
        page = 3
        pos0 = {"cold": 0, "warm": 4096}
        for case in args.cases.split(","):
            n_real = T - args.n_pad
            positions = torch.arange(T, dtype=torch.long) + pos0[case]
            positions[n_real:] = positions[n_real - 1]  # bucket padding repeats the last position
            meta = {"block_table_tensor": torch.tensor([[page]], dtype=torch.int32)}
            if case == "warm":
                meta["cached_seq_len"] = torch.tensor([pos0[case]], dtype=torch.int32)
            # CPU reference (torch path, bf16 weights). The scan inputs are captured to also
            # build a float64 sequential reference of the final SSM state.
            M.KERNEL2 = "0"
            sa, sb = base_a.clone(), base_b.clone()
            cpu_mixer.bind_state(sa, sb)
            cap = {}
            orig_scan = M.NemotronHMamba2Mixer.ssd_chunk_scan

            def capture(self_, x, dt_, A, B, C, init):
                cap.update(x=x, dt=dt_, A=A, B=B, C=C, init=init)
                return orig_scan(self_, x, dt_, A, B, C, init)

            cpu_mixer.ssd_chunk_scan = capture.__get__(cpu_mixer)
            out_cpu = cpu_mixer.forward_prefill(hidden, positions, meta).float()
            del cpu_mixer.ssd_chunk_scan
            h64 = cap["init"].double()
            dA = torch.exp(cap["dt"].double() * cap["A"].double().view(1, -1))
            for t in range(T):
                h64 = h64 * dA[t].view(-1, 1, 1) + (cap["dt"][t].double().view(-1, 1) * cap["x"][t].double()).unsqueeze(-1) * cap["B"][t].double().unsqueeze(1)
            st64 = h64.reshape(-1)
            st_cpu = torch.cat([cpu_mixer.state_a.reshape(num_pages, -1), cpu_mixer.state_b.reshape(num_pages, -1)], 1)[page]
            row = {"layer": layer, "case": case, "T": T, "n_pad": args.n_pad, "neg_A_max": torch.exp(p.A_log).max().item(),
                   "out_cpu_max": out_cpu[:n_real].abs().max().item(),
                   "ssm_ref_max": st64.abs().max().item(),
                   "ssm_cpu_torch_vs_fp64_max_abs": (st_cpu[p.conv_numel : p.conv_numel + p.ssm_numel].double() - st64).abs().max().item()}
            for mode in args.modes.split(","):
                M.KERNEL2 = {"off": "0", "scan": "scan", "on": "1"}[mode]
                torch._dynamo.reset()
                da, db = base_a.clone().to(dev), base_b.clone().to(dev)
                dev_mixer.bind_state(da, db)

                def step(h, pos, bt, csl=None):
                    md = {"block_table_tensor": bt}
                    if csl is not None:
                        md["cached_seq_len"] = csl
                    return dev_mixer.forward_prefill(h, pos, md)

                f = torch.compile(step, backend=backend, fullgraph=True)
                ins = [hidden.to(dev), positions.to(dev), meta["block_table_tensor"].to(dev)]
                if case == "warm":
                    ins.append(meta["cached_seq_len"].to(dev))
                t0 = time.time()
                out = f(*ins).cpu().float()
                compile_s = time.time() - t0
                st_dev = torch.cat([dev_mixer.state_a.cpu().reshape(num_pages, -1), dev_mixer.state_b.cpu().reshape(num_pages, -1)], 1)[page]
                d = (out - out_cpu)[:n_real].abs()
                ssm_sl = slice(p.conv_numel, p.conv_numel + p.ssm_numel)
                conv_d = st_dev[: p.conv_numel]
                if not torch.isfinite(conv_d).all():
                    bad = (~torch.isfinite(conv_d)).nonzero().flatten()
                    print("NONFINITE conv window", mode, case, "count", bad.numel(), "first", bad[:5].tolist(),
                          "rows", sorted(set((bad // p.conv_dim).tolist()))[:5], flush=True)
                r = {
                    "compile_and_first_s": round(compile_s, 1),
                    "out_max_abs_vs_cpu_torch": d.max().item(),
                    "out_mean_abs_vs_cpu_torch": d.mean().item(),
                    "ssm_max_abs_vs_cpu_torch": (st_dev[ssm_sl] - st_cpu[ssm_sl]).abs().max().item(),
                    "ssm_max_abs_vs_fp64": (st_dev[ssm_sl].double() - st64).abs().max().item(),
                    "conv_max_abs_vs_cpu_torch": (st_dev[: p.conv_numel] - st_cpu[: p.conv_numel]).abs().max().item(),
                    "finite": bool(torch.isfinite(out).all()),
                }
                row.setdefault("outs", {})[mode] = out
                # Timing: the state page is rewritten every call (same work each time).
                for _ in range(3):
                    f(*ins).cpu()
                t0 = time.time()
                for _ in range(args.iters):
                    f(*ins).cpu()
                r["step_ms"] = 1000 * (time.time() - t0) / args.iters
                row[mode] = r
                print("ROW", layer, case, mode, json.dumps(r), flush=True)
            if "on" in row and "off" in row:
                o_on, o_off = row["outs"]["on"], row["outs"]["off"]
                row["out_max_abs_on_vs_off"] = (o_on - o_off)[:n_real].abs().max().item()
                row["speedup_off_over_on"] = row["off"]["step_ms"] / row["on"]["step_ms"]
            row.pop("outs", None)
            torch._dynamo.reset()
            null = torch.compile(lambda h: (h[:, :1] * 2).expand(T, p.hidden).contiguous(), backend=backend, fullgraph=True)
            hd = hidden.to(dev)
            for _ in range(3):
                null(hd).cpu()
            t0 = time.time()
            for _ in range(args.iters):
                null(hd).cpu()
            row["null_call_ms"] = 1000 * (time.time() - t0) / args.iters
            for mode in ("off", "scan", "on"):
                if mode in row:
                    row[mode]["net_ms"] = row[mode]["step_ms"] - row["null_call_ms"]
            results.append(row)
            print("RESULT", json.dumps(row), flush=True)
            json.dump({"results": results, "iters": args.iters,
                       "env": {k: os.environ.get(k) for k in ("NEURON_RT_VISIBLE_CORES", "NEMOTRON_H_STATE_KERNEL")}},
                      open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
