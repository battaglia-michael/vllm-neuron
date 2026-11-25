# SPDX-License-Identifier: Apache-2.0
"""Fused Mamba-2 decode kernel (``kernels/mamba_decode.py``) vs the port's torch decode step.

Uses the real layer-0 Mamba weights of the checkpoint, sharded as rank 0 of TP=4
(16 heads, 2 groups), random hidden states and random state pages.

Two references:
  * ``torch_bf16``: the exact math of ``NemotronHMamba2Mixer.forward_decode``
    (``DECODE_Y=rows``, ``DT_CLAMP=none``) on bf16 inputs, i.e. what the
    flag-off path computes; the kernel should agree to bf16 rounding.
  * ``torch_fp32``: the same steps in float32 throughout, except x/B/C rounded
    to bf16 after the conv (the kernel and torch both feed bf16 x/B/C to the
    recurrence). This isolates the kernel's arithmetic error.

Run as a script for the simulator (``--mode sim``) or on a Neuron device
(``--mode device``); writes a JSON summary with ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

MODEL = os.environ.get(
    "MODEL_DIR", "/mnt/nvme/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
)
TP = 4
RANK = 0


class Params:
    """Layer weights of one TP rank, in the port's layouts (mamba.py loaders)."""

    def __init__(self, layer: int = 0, model_dir: str = MODEL):
        from safetensors import safe_open

        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        pre = f"backbone.layers.{layer}.mixer."

        def get(name):
            with safe_open(os.path.join(model_dir, idx[pre + name]), "pt") as f:
                return f.get_tensor(pre + name)

        H = cfg["mamba_num_heads"]
        P = cfg["mamba_head_dim"]
        G = cfg["n_groups"]
        N = cfg["ssm_state_size"]
        I_full = H * P
        bc_full = G * N
        self.nh, self.P, self.G, self.N = H // TP, P, G // TP, N
        self.I = self.nh * self.P
        self.conv_dim = self.I + 2 * self.G * N
        self.K = cfg["conv_kernel"]
        self.eps = cfg["norm_eps"]
        self.hidden = cfg["hidden_size"]
        self.group_size = I_full // G

        def seg(t, sizes):
            parts, off = [], 0
            for s in sizes:
                per = s // TP
                parts.append(t[off + RANK * per : off + (RANK + 1) * per])
                off += s
            return torch.cat(parts, 0)

        self.in_proj = seg(get("in_proj.weight"), (I_full, I_full, bc_full, bc_full, H)).T.contiguous()
        self.conv_w = seg(get("conv1d.weight"), (I_full, bc_full, bc_full)).squeeze(1).contiguous()
        self.conv_b = seg(get("conv1d.bias"), (I_full, bc_full, bc_full)).contiguous()
        sl = slice(RANK * self.nh, (RANK + 1) * self.nh)
        self.dt_bias = get("dt_bias")[sl].float().contiguous()
        self.A_log = get("A_log")[sl].float().contiguous()
        self.D = get("D")[sl].float().contiguous()
        self.norm_w = get("norm.weight")[RANK * self.I : (RANK + 1) * self.I].contiguous()
        self.conv_numel = (self.K - 1) * self.conv_dim
        self.ssm_numel = self.nh * self.P * self.N


def torch_decode(p: Params, proj, conv_prev, ssm, fp32: bool):
    """``forward_decode`` between in_proj and out_proj; returns (y_normed, new_conv, new_ssm).

    ``proj [n, proj]`` bf16, ``conv_prev [n, K-1, conv_dim]`` and ``ssm [n, h, P, N]`` float32.
    """
    n = proj.shape[0]
    i, c = p.I, p.conv_dim
    z, xbc, dt = proj[:, :i], proj[:, i : i + c], proj[:, i + c :]
    w, b = p.conv_w, p.conv_b
    if fp32:
        cp, xb, w, b = conv_prev.float(), xbc.float(), w.float(), b.float()
    else:
        cp, xb = conv_prev.to(xbc.dtype), xbc
    last = p.K - 1
    mixed = cp[:, 0] * w[:, 0]
    for tap in range(1, last):
        mixed = mixed + cp[:, tap] * w[:, tap]
    mixed = mixed + xb * w[:, last]
    mixed = mixed + b
    act = F.silu(mixed).to(torch.bfloat16)  # x, B, C are bf16 in the port (and the kernel)
    x = act[:, :i].reshape(n, p.nh, p.P)

    def to_heads(m):
        hpg = p.nh // p.G
        return m.reshape(n, p.G, 1, p.N).expand(n, p.G, hpg, p.N).reshape(n, p.nh, p.N)

    bc = p.G * p.N
    B, C = to_heads(act[:, i : i + bc]), to_heads(act[:, i + bc : i + 2 * bc])
    x, B, C = x.float(), B.float(), C.float()
    dt = F.softplus(dt.float() + p.dt_bias)
    A = -torch.exp(p.A_log.float())
    dA = torch.exp(dt * A.view(1, -1))
    nh, P, N = n * p.nh, p.P, p.N
    rows = nh * P
    ssm2 = ssm.float().reshape(rows, N)
    dA_r = dA.reshape(nh, 1).expand(nh, P).reshape(rows, 1)
    dtx_r = (dt.unsqueeze(-1) * x).reshape(rows, 1)
    B_r = B.reshape(nh, 1, N).expand(nh, P, N).reshape(rows, N)
    C_r = C.reshape(nh, 1, N).expand(nh, P, N).reshape(rows, N)
    ssm2 = ssm2 * dA_r + dtx_r * B_r
    y = (ssm2 * C_r).sum(dim=-1).reshape(n, p.nh, P)
    y = y + p.D.float().view(1, -1, 1) * x
    new_conv = torch.cat([cp[:, 1:].float(), xbc.float().unsqueeze(1)], dim=1)
    # _gated_norm_out without out_proj
    yy = y.reshape(n, i).float() * F.silu(z.float())
    g = yy.reshape(n, i // p.group_size, p.group_size)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + p.eps)
    out = (p.norm_w.float() * g.reshape(n, i)).to(torch.bfloat16)
    return out, new_conv, ssm2.reshape(n, p.nh, P, N)


def make_case(p: Params, n: int, half_rows: int, num_pages: int, n_pad: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    hs = (torch.randn(n, p.hidden, generator=g) * 1.0).to(torch.bfloat16)
    proj = (hs @ p.in_proj).contiguous()
    total = p.conv_numel + p.ssm_numel
    assert 2 * half_rows * 128 >= total
    state_a = torch.randn(num_pages, half_rows, 128, generator=g) * 0.05
    state_b = torch.randn(num_pages, half_rows, 128, generator=g) * 0.05
    # The conv window holds bf16 values (raw xBC of earlier tokens).
    conv_rows = p.conv_numel // 128
    state_a[:, :conv_rows] = (torch.randn(num_pages, conv_rows, 128, generator=g) * 1.5).to(torch.bfloat16).float()
    perm = torch.randperm(num_pages - 1, generator=g)[:n] + 1
    pages = perm.clone()
    live = torch.ones(n, dtype=torch.bool)
    if n_pad:
        pages[n - n_pad :] = 0
        live[n - n_pad :] = False
    wpages = torch.where(live, pages, torch.full_like(pages, num_pages))
    return proj, state_a, state_b, pages, wpages, live


def read_pages(p: Params, state_a, state_b, pages):
    n = pages.shape[0]
    page = torch.cat([state_a[pages].reshape(n, -1), state_b[pages].reshape(n, -1)], 1)
    conv = page[:, : p.conv_numel].reshape(n, p.K - 1, p.conv_dim)
    ssm = page[:, p.conv_numel : p.conv_numel + p.ssm_numel].reshape(n, p.nh, p.P, p.N)
    return conv, ssm


def run_kernel(p: Params, proj, state_a, state_b, pages, wpages, mode: str, lnc: int, extra=None):
    from nemotron_h.kernels.mamba_decode import mamba2_decode_kernel

    kw = dict(
        proj=proj, state_a=state_a, state_b=state_b,
        pages=pages.to(torch.int32).view(-1, 1), wpages=wpages.to(torch.int32).view(-1, 1),
        conv_w=p.conv_w, conv_b=p.conv_b, dt_bias=p.dt_bias, a_neg=-torch.exp(p.A_log),
        d_skip=p.D, norm_w=p.norm_w, n_groups=p.G, head_dim=p.P, d_state=p.N, eps=float(p.eps),
        **(extra or {}),
    )
    if mode == "sim":
        from libtorch_neuronx_lite.nki.nki_cpu_sim import simulate_nki_kernel

        y, sa, sb = simulate_nki_kernel(mamba2_decode_kernel, lnc, kw)
        return y, sa, sb
    raise ValueError(mode)


def compare(p, proj, state_a, state_b, pages, wpages, live, y_k, sa_k, sb_k):
    conv0, ssm0 = read_pages(p, state_a, state_b, pages)
    res = {}
    for name, fp32 in (("torch_fp32", True), ("torch_bf16", False)):
        y_r, conv_r, ssm_r = torch_decode(p, proj, conv0, ssm0, fp32)
        conv_k, ssm_k = read_pages(p, sa_k, sb_k, torch.where(live, pages, torch.zeros_like(pages)))
        L = live
        d_y = (y_k.float() - y_r.float())[L].abs()
        d_s = (ssm_k - ssm_r)[L].abs()
        d_c = (conv_k - conv_r)[L].abs()
        rel_s = (d_s.max() / ssm_r[L].abs().max()).item()
        res[name] = {
            "y_max_abs": d_y.max().item(),
            "y_mean_abs": d_y.mean().item(),
            "y_ref_max": y_r.float()[L].abs().max().item(),
            "y_frac_bitexact": (y_k[L] == y_r[L]).float().mean().item(),
            "y_frac_within_1ulp": (d_y <= y_r.float()[L].abs() * 2**-7 + 1e-30).float().mean().item(),
            "ssm_max_abs": d_s.max().item(),
            "ssm_max_rel_to_max": rel_s,
            "conv_max_abs": d_c.max().item(),
        }
    # Scale of the bf16 path's own rounding: torch bf16 vs torch fp32.
    y32, c32, s32 = torch_decode(p, proj, conv0, ssm0, True)
    y16, c16, s16 = torch_decode(p, proj, conv0, ssm0, False)
    res["torch_bf16_vs_torch_fp32"] = {
        "y_max_abs": (y16.float() - y32.float())[live].abs().max().item(),
        "ssm_max_abs": (s16 - s32)[live].abs().max().item(),
    }
    # Untouched pages must be unchanged (padding writes skipped, other pages intact).
    touched = set(pages[live].tolist())
    others = [q for q in range(state_a.shape[0]) if q not in touched]
    res["untouched_pages_equal"] = bool(
        torch.equal(sa_k[others], state_a[others]) and torch.equal(sb_k[others], state_b[others])
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sim", choices=["sim"])
    ap.add_argument("--batches", default="1,5,32")
    ap.add_argument("--half-rows", default="530,600,1100")
    ap.add_argument("--lnc", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--kw", default="{}", help="extra kernel kwargs as JSON, e.g. '{\"win_vec\": 1}'")
    args = ap.parse_args()
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    p = Params()
    cases = []
    for half_rows in [int(x) for x in args.half_rows.split(",")]:
        for n in [int(x) for x in args.batches.split(",")]:
            n_pad = 1 if n > 2 else 0
            proj, sa, sb, pages, wpages, live = make_case(p, n, half_rows, max(n + 3, 8), n_pad, seed=n * 7 + half_rows)
            t0 = time.time()
            y_k, sa_k, sb_k = run_kernel(p, proj, sa.clone(), sb.clone(), pages, wpages, args.mode, args.lnc, json.loads(args.kw))
            dt = time.time() - t0
            r = compare(p, proj, sa, sb, pages, wpages, live, y_k, sa_k, sb_k)
            r.update(n=n, half_rows=half_rows, n_pad=n_pad, lnc=args.lnc, wall_s=round(dt, 1))
            print(json.dumps(r), flush=True)
            cases.append(r)
    worst = {
        k: max(c[ref][k] for c in cases for ref in ("torch_fp32",))
        for k in ("y_max_abs", "ssm_max_abs", "conv_max_abs")
    }
    summary = {"mode": args.mode, "kernel_kwargs": json.loads(args.kw), "cases": cases, "worst_vs_fp32": worst,
               "worst_vs_bf16": {k: max(c["torch_bf16"][k] for c in cases) for k in ("y_max_abs", "ssm_max_abs", "conv_max_abs")},
               "all_untouched_pages_equal": all(c["untouched_pages_equal"] for c in cases)}
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
