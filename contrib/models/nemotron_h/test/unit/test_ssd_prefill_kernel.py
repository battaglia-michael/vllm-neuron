# SPDX-License-Identifier: Apache-2.0
"""SSD prefill kernel (``kernels/ssd_prefill.py``) vs the port's torch ``ssd_chunk_scan`` (A06).

Real Mamba weights of one layer (TP=4 rank-0 shard: 16 heads, 2 groups), random
hidden states through the real in_proj and conv, trailing padding rows (xBC = 0,
dt = 0) as the bucket padding produces, and a random initial state (chunked
prefill continues from a cached state).

References, all on CPU:
  * ``torch_fp32``: ``NemotronHMamba2Mixer.ssd_chunk_scan`` + ``D x`` in float32,
    i.e. exactly the port's flag-off math.
  * ``torch_fp64``: the same function in float64 (truth for both).

    python test_ssd_prefill_kernel.py --layers 0,<layer with the largest -A> --T 256,512 --out x.json
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


def mamba_layers(model_dir=MODEL):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    return [i for i, t in enumerate(cfg["layers_block_type"]) if t == "mamba"]


def make_inputs(p: Params, T: int, n_pad: int, seed: int, state_scale: float = 0.1):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(T, p.hidden, generator=g).to(torch.bfloat16)
    proj = hs @ p.in_proj
    i, c = p.I, p.conv_dim
    xbc, dt_raw = proj[:, i : i + c], proj[:, i + c :]
    real = torch.ones(T, 1)
    if n_pad:
        real[T - n_pad :] = 0
    xbc = xbc * real.to(xbc.dtype)
    prev = (torch.randn(p.K - 1, c, generator=g) * 0.5).to(torch.bfloat16)
    padded = torch.cat([prev, xbc], 0)
    w = p.conv_w
    out = padded[:T] * w[:, 0]
    for tap in range(1, p.K):
        out = out + padded[tap : tap + T] * w[:, tap]
    out = out + p.conv_b
    conv_out = F.silu(out)  # bf16, as forward_prefill
    dt = F.softplus(dt_raw.float() + p.dt_bias) * real
    init = torch.randn(p.nh, p.P, p.N, generator=g) * state_scale
    return conv_out.contiguous(), dt.contiguous(), init


def reference(p: Params, conv_out, dt, init, dtype):
    from nemotron_h.mamba import NemotronHMamba2Mixer

    T = conv_out.shape[0]
    i, bc = p.I, p.G * p.N
    x = conv_out[:, :i].reshape(T, p.nh, p.P).to(dtype)
    hpg = p.nh // p.G

    def to_heads(m):
        return m.reshape(T, p.G, 1, p.N).expand(T, p.G, hpg, p.N).reshape(T, p.nh, p.N).to(dtype)

    B, C = to_heads(conv_out[:, i : i + bc]), to_heads(conv_out[:, i + bc : i + 2 * bc])
    A = -torch.exp(p.A_log.to(dtype))
    fake = SimpleNamespace(chunk_size=128)
    y, st = NemotronHMamba2Mixer.ssd_chunk_scan(fake, x, dt.to(dtype), A, B, C, init.to(dtype))
    # ssd_chunk_scan computes in float32 internally; for fp64 redo it with dtype patched.
    y = y + p.D.to(dtype).view(1, -1, 1) * x
    return y.to(dtype), st.to(dtype)


def reference64(p: Params, conv_out, dt, init):
    """Straight sequential recurrence in float64 (independent of the chunked algorithm)."""
    T = conv_out.shape[0]
    i, bc = p.I, p.G * p.N
    hpg = p.nh // p.G
    x = conv_out[:, :i].double().reshape(T, p.nh, p.P)
    B = conv_out[:, i : i + bc].double().reshape(T, p.G, p.N).repeat_interleave(hpg, dim=1)
    C = conv_out[:, i + bc : i + 2 * bc].double().reshape(T, p.G, p.N).repeat_interleave(hpg, dim=1)
    A = -torch.exp(p.A_log.double())
    d = dt.double()
    h = init.double().clone()
    ys = []
    for t in range(T):
        dA = torch.exp(d[t] * A)  # [H]
        h = h * dA.view(-1, 1, 1) + (d[t].view(-1, 1) * x[t]).unsqueeze(-1) * B[t].unsqueeze(1)
        ys.append((h * C[t].unsqueeze(1)).sum(-1) + p.D.double().view(-1, 1) * x[t])
    return torch.stack(ys), h


def run_sim(p: Params, conv_out, dt, init, lnc: int):
    from libtorch_neuronx_lite.nki.nki_cpu_sim import simulate_nki_kernel
    from nemotron_h.kernels.ssd_prefill import ssd_prefill_kernel, tri_mask

    kw = dict(
        xbc=conv_out, dt=dt, a_neg=-torch.exp(p.A_log), d_skip=p.D, init_state=init.contiguous(),
        tri=tri_mask("cpu"), n_groups=p.G, head_dim=p.P, d_state=p.N,
    )
    y, st = simulate_nki_kernel(ssd_prefill_kernel, lnc, kw)
    return y.reshape(conv_out.shape[0], p.nh, p.P).float(), st.float()


def run_device(p: Params, conv_out, dt, init, lnc: int = 2):
    import libtorch_neuronx_lite  # noqa: F401
    from libtorch_neuronx_lite.compile.native_backend import register
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nemotron_h.kernels.ssd_prefill import ssd_prefill_kernel, tri_mask

    backend = register()
    dev = torch.device("neuron:0")
    G, P, N = p.G, p.P, p.N

    def fn(xbc, dt, a, d, init, tri):
        return wrap_nki(ssd_prefill_kernel)[lnc](
            xbc=xbc, dt=dt, a_neg=a, d_skip=d, init_state=init, tri=tri, n_groups=G, head_dim=P, d_state=N
        )

    torch._dynamo.reset()
    f = torch.compile(fn, backend=backend, fullgraph=True)
    args = [conv_out, dt, -torch.exp(p.A_log), p.D.contiguous(), init.contiguous(), tri_mask("cpu")]
    y, st = f(*[a.to(dev) for a in args])
    return y.cpu().reshape(conv_out.shape[0], p.nh, p.P).float(), st.cpu().float()


def stats(a, b, real_rows):
    d = (a.double() - b.double())[real_rows].abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(), "ref_max": b.double()[real_rows].abs().max().item()}


def fused_case(p: Params, T: int, n_pad: int, seed: int, warm: bool):
    """Inputs of the fused kernel and the torch-path reference (forward_prefill minus out_proj)."""
    from device_ssd_prefill import build_mixer
    from harness import init_dist

    init_dist()
    from nemotron_h import mamba as M

    p.out_proj = torch.zeros(p.I, p.hidden, dtype=torch.bfloat16)
    m = build_mixer(M, p)
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(T, p.hidden, generator=g).to(torch.bfloat16)
    proj = (hs @ p.in_proj).contiguous()
    real = torch.ones(T, 1)
    if n_pad:
        real[T - n_pad :] = 0
    prev = (torch.randn(p.K - 1, p.conv_dim, generator=g) * 0.5) if warm else None
    init = torch.randn(p.nh, p.P, p.N, generator=g) * 0.1 if warm else torch.zeros(p.nh, p.P, p.N)
    z, xbc, dt = m._split_proj(proj)
    xbc = xbc * real.to(xbc.dtype)
    conv_out = F.silu(m._causal_conv(xbc, T, prev))
    x, B, C = m._split_xbc(conv_out)
    x, B, C = x.float(), B.float(), C.float()
    dts = F.softplus(dt.float() + m.dt_bias) * real
    A = -torch.exp(m.A_log.float())
    y, st = m.ssd_chunk_scan(x, dts, A, B, C, init)
    y = y + m.D.float().view(1, -1, 1) * x
    yy = y.reshape(T, p.I).float() * F.silu(z.float())
    gg = yy.reshape(T, p.I // p.group_size, p.group_size)
    gg = gg * torch.rsqrt(gg.pow(2).mean(-1, keepdim=True) + p.eps)
    y_ref = (m.norm_weight.float() * gg.reshape(T, p.I)).to(torch.bfloat16)
    prev_bf = torch.zeros(p.K - 1, p.conv_dim, dtype=torch.bfloat16) if prev is None else prev.to(torch.bfloat16)
    kw = dict(
        proj=proj, conv_prev=prev_bf.contiguous(), real=real.contiguous(), conv_w=p.conv_w, conv_b=p.conv_b,
        dt_bias=p.dt_bias, a_neg=-torch.exp(p.A_log), d_skip=p.D.contiguous(), norm_w=p.norm_w,
        init_state=init.contiguous(), n_groups=p.G, head_dim=p.P, d_state=p.N, eps=float(p.eps),
    )
    return kw, y_ref, st, conv_out, dts, init


def run_fused(kw, mode: str, lnc: int, split: bool = False):
    from nemotron_h.kernels.ssd_prefill import mamba_prefill_fused_kernel, mamba_prefill_fused_split_kernel, tri_mask

    kw = dict(kw, tri=tri_mask("cpu"))
    kern = mamba_prefill_fused_kernel
    if split:
        # The three in_proj outputs of NEMOTRON_H_MAMBA_PREFILL=split; xBC masked to real tokens as the model does.
        proj = kw.pop("proj")
        I = kw["dt_bias"].shape[0] * kw["head_dim"]
        cd = kw["conv_w"].shape[0]
        kw = dict(kw, z=proj[:, :I].contiguous(), xbc=(proj[:, I : I + cd] * kw["real"].to(proj.dtype)).contiguous(),
                  dtr=proj[:, I + cd :].contiguous())
        kern = mamba_prefill_fused_split_kernel
        from nemotron_h.kernels.ssd_prefill import CONV_KMAJOR
        if CONV_KMAJOR:  # R2-03: [K, conv_dim] conv weight
            kw = dict(kw, conv_w=kw["conv_w"].t().contiguous(), conv_kmajor=True)
    if mode == "sim":
        from libtorch_neuronx_lite.nki.nki_cpu_sim import simulate_nki_kernel

        y, st = simulate_nki_kernel(kern, lnc, kw)
        return y, st.float()
    import libtorch_neuronx_lite  # noqa: F401
    from libtorch_neuronx_lite.compile.native_backend import register
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    backend = register()
    dev = torch.device("neuron:0")
    tens = {k: v for k, v in kw.items() if isinstance(v, torch.Tensor)}
    scal = {k: v for k, v in kw.items() if not isinstance(v, torch.Tensor)}
    names = list(tens)

    def fn(*args):
        return wrap_nki(kern)[lnc](**dict(zip(names, args)), **scal)

    torch._dynamo.reset()
    f = torch.compile(fn, backend=backend, fullgraph=True)
    y, st = f(*[tens[n].to(dev) for n in names])
    return y.cpu(), st.cpu().float()


def main_fused(args):
    cases = []
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [mamba_layers()[0]]
    for layer in layers:
        p = Params(layer=layer)
        for T in [int(x) for x in args.T.split(",")]:
            for warm in (False, True):
                kw, y_ref, st_ref, conv_out, dts, init = fused_case(p, T, args.n_pad, seed=T * 31 + layer + warm, warm=warm)
                real_rows = torch.arange(T) < T - args.n_pad
                t0 = time.time()
                y_k, st_k = run_fused(kw, args.mode, args.lnc, split=args.kernel == "fused_split")
                wall = time.time() - t0
                y64, st64 = reference64(p, conv_out, dts, init)
                d = (y_k.float() - y_ref.float())[real_rows].abs()
                ulp = y_ref.float()[real_rows].abs() * 2.0 ** -7 + 1e-30
                r = {
                    "layer": layer, "T": T, "warm": warm, "n_pad": args.n_pad, "lnc": args.lnc, "mode": args.mode,
                    "wall_s": round(wall, 1),
                    "y_vs_torch": {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
                                   "frac_bitexact": (y_k[real_rows] == y_ref[real_rows]).float().mean().item(),
                                   "frac_within_1ulp": (d <= ulp).float().mean().item(),
                                   "ref_max": y_ref.float()[real_rows].abs().max().item()},
                    "state_kernel_vs_fp64": stats(st_k, st64, slice(None)),
                    "state_torch_fp32_vs_fp64": stats(st_ref, st64, slice(None)),
                    "finite": bool(torch.isfinite(y_k.float()).all() and torch.isfinite(st_k).all()),
                }
                print(json.dumps(r), flush=True)
                cases.append(r)
    summary = {
        "kernel": "mamba_prefill_fused_split_kernel" if args.kernel == "fused_split" else "mamba_prefill_fused_kernel", "cases": cases,
        "worst_y_max_abs_vs_torch": max(c["y_vs_torch"]["max_abs"] for c in cases),
        "min_frac_within_1ulp": min(c["y_vs_torch"]["frac_within_1ulp"] for c in cases),
        "worst_state_kernel_vs_fp64": max(c["state_kernel_vs_fp64"]["max_abs"] for c in cases),
        "worst_state_torch_fp32_vs_fp64": max(c["state_torch_fp32_vs_fp64"]["max_abs"] for c in cases),
        "all_finite": all(c["finite"] for c in cases),
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None, help="comma list of Mamba layer ids (default: first + largest -A)")
    ap.add_argument("--T", default="256,512")
    ap.add_argument("--n-pad", type=int, default=37)
    ap.add_argument("--lnc", type=int, default=2)
    ap.add_argument("--mode", default="sim", choices=["sim", "device"])
    ap.add_argument("--kernel", default="scan", choices=["scan", "fused", "fused_split"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    torch.set_grad_enabled(False)
    if args.kernel in ("fused", "fused_split"):
        return main_fused(args)
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = [mamba_layers()[0]]
    cases = []
    for layer in layers:
        p = Params(layer=layer)
        amax = torch.exp(p.A_log).max().item()
        for T in [int(x) for x in args.T.split(",")]:
            conv_out, dt, init = make_inputs(p, T, args.n_pad, seed=T * 31 + layer)
            real_rows = torch.arange(T) < T - args.n_pad
            t0 = time.time()
            if args.mode == "device":
                y_k, st_k = run_device(p, conv_out, dt, init, args.lnc)
            else:
                y_k, st_k = run_sim(p, conv_out, dt, init, args.lnc)
            wall = time.time() - t0
            y32, st32 = reference(p, conv_out, dt, init, torch.float32)
            y64, st64 = reference64(p, conv_out, dt, init)
            r = {
                "layer": layer, "T": T, "n_pad": args.n_pad, "lnc": args.lnc, "mode": args.mode, "neg_A_max": amax,
                "sim_wall_s": round(wall, 1),
                "kernel_vs_fp64": {"y": stats(y_k, y64, real_rows), "state": stats(st_k, st64, slice(None))},
                "torch_fp32_vs_fp64": {"y": stats(y32, y64, real_rows), "state": stats(st32, st64, slice(None))},
                "kernel_vs_torch_fp32": {"y": stats(y_k, y32, real_rows), "state": stats(st_k, st32, slice(None))},
                "finite": bool(torch.isfinite(y_k).all() and torch.isfinite(st_k).all()),
            }
            print(json.dumps(r), flush=True)
            cases.append(r)
    summary = {
        "cases": cases,
        "worst_kernel_vs_fp64_y": max(c["kernel_vs_fp64"]["y"]["max_abs"] for c in cases),
        "worst_torch_fp32_vs_fp64_y": max(c["torch_fp32_vs_fp64"]["y"]["max_abs"] for c in cases),
        "worst_kernel_vs_fp64_state": max(c["kernel_vs_fp64"]["state"]["max_abs"] for c in cases),
        "worst_torch_fp32_vs_fp64_state": max(c["torch_fp32_vs_fp64"]["state"]["max_abs"] for c in cases),
        "all_finite": all(c["finite"] for c in cases),
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
