# SPDX-License-Identifier: Apache-2.0
"""A07: routed prefill MoE kernel (kernels/moe_prefill.py) vs the dense all-experts math.

    python test_moe_prefill_kernel.py --mode cpu|sim [--lnc 2] [--t 256] [--e 16] --out x.json

``cpu``: the slot dispatch with a torch stand-in for the kernel (checks the dispatch/combine).
``sim``: the NKI CPU simulator. Device timing is in ``device_moe_prefill.py``.
Shapes: TP=4 rank shapes (H=2688, I=464), fewer experts by default to keep the simulator fast.
"""
import argparse
import json
import time

import torch


def routing(t, e, k, g, skew=0.0):
    logits = torch.randn(t, e, generator=g) + skew * torch.linspace(2, 0, e)
    s = torch.sigmoid(logits)
    _, idx = torch.topk(s, k, dim=-1)
    w = torch.gather(s, 1, idx)
    w = w / w.sum(-1, keepdim=True) * 2.5
    return idx, w


def dense_ref(x, up, down, idx, w, e, rows):
    """fp32 routed sum: sum_k w * down_e(relu(up_e x)^2)."""
    xf = x.float()
    up3 = up.float().view(e, rows, -1)
    dn3 = down.float().view(e, rows, -1)
    out = torch.zeros_like(xf)
    for j in range(idx.shape[1]):
        for ex in idx[:, j].unique().tolist():
            m = idx[:, j] == ex
            hh = torch.relu(xf[m] @ up3[ex].T) ** 2
            out[m] += w[m, j:j + 1].float() * (hh @ dn3[ex])
    return out


def torch_kernel(kw):
    """Plain-torch stand-in for moe_prefill_kernel (same output layout, 2 partials)."""
    x, up, down, ntile, tstart, st, sp, ws = (kw[n] for n in ("x", "up", "down", "ntile", "tstart", "slot_tok",
                                                                 "slot_pair", "ws"))
    rows = kw["geom"].shape[0]
    npad, h = st.shape[0], x.shape[1]
    e = ntile.shape[0]
    out = torch.zeros(2, ws.shape[0], h)
    half = rows // 2
    for ex in range(e):
        s0, n = int(tstart[ex, 0]), int(ntile[ex, 0]) * 128
        xs = x[st[s0:s0 + n, 0].long()].float()
        for c in range(2):
            u = up[ex * rows + c * half: ex * rows + (c + 1) * half].float()
            d = down[ex * rows + c * half: ex * rows + (c + 1) * half].float()
            a = torch.relu(xs @ u.T) ** 2
            out[c, sp[s0:s0 + n, 0].long()] = (a.to(torch.bfloat16).float() @ d) * ws[sp[s0:s0 + n, 0].long()]
    return out.to(torch.bfloat16).view(-1, h)


def torch_kernel_e(kw):
    """Plain-torch stand-in for moe_prefill_kernel_esplit (one row per pair, groups of 2 experts)."""
    x, up, down, ntile, tstart, st, sp, ws = (kw[n] for n in ("x", "up", "down", "ntile", "tstart", "slot_tok",
                                                                 "slot_pair", "ws"))
    rows = kw["geom"].shape[0]
    h = x.shape[1]
    e = tstart.shape[0]
    grp = e // ntile.shape[0]
    out = torch.zeros(ws.shape[0], h)
    for ex in range(e):
        s0, n = int(tstart[ex, 0]), int(ntile[ex // grp, 0]) * 128
        xs = x[st[s0:s0 + n, 0].long()].float()
        u = up[ex * rows:(ex + 1) * rows].float()
        d = down[ex * rows:(ex + 1) * rows].float()
        a = torch.relu(xs @ u.T) ** 2
        p = sp[s0:s0 + n, 0].long()
        out[p] = (a.to(torch.bfloat16).float() @ d) * ws[p]
    return out.to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cpu")
    ap.add_argument("--lnc", type=int, default=2)
    ap.add_argument("--t", type=int, default=256)
    ap.add_argument("--e", type=int, default=16)
    ap.add_argument("--skew", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from nemotron_h.kernels import moe_prefill as MP

    g = torch.Generator().manual_seed(0)
    H, I, K = 2688, 464, 6
    x = (torch.randn(a.t, H, generator=g) * 0.5).to(torch.bfloat16)
    up = (torch.randn(a.e * I, H, generator=g) * 0.02).to(torch.bfloat16)
    down = (torch.randn(a.e * I, H, generator=g) * 0.02).to(torch.bfloat16)
    idx, w = routing(a.t, a.e, K, g, a.skew)
    ref = dense_ref(x, up, down, idx, w, a.e, I)
    if a.mode == "cpu":
        def run(kw, kern=None):
            return torch_kernel_e(kw) if kern is MP.moe_prefill_kernel_esplit else torch_kernel(kw)
    else:
        from libtorch_neuronx_lite.nki.nki_cpu_sim import simulate_nki_kernel

        def run(kw, kern=None):
            return simulate_nki_kernel(kern or MP.moe_prefill_kernel, a.lnc, kw)
    t0 = time.time()
    out = MP.moe_prefill_routed(x, up, down, idx, w, I, a.e, run=run).float()
    dt = time.time() - t0
    bf_ref = ref.to(torch.bfloat16).float()
    counts = torch.bincount(idx.reshape(-1), minlength=a.e)
    r = {"mode": a.mode, "split": MP.SPLIT, "wt_dma_n": MP.WT_DMA_N, "dispatch": MP.DISPATCH_MODE, "lnc": a.lnc, "t": a.t, "experts": a.e, "k": K, "H": H, "I": I, "seconds": round(dt, 1),
         "max_count": int(counts.max()), "tiles": int(((counts + 127) // 128).sum()),
         "rel_l2_vs_fp32_ref": ((out - ref).norm() / ref.norm()).item(),
         "max_abs_vs_fp32_ref": (out - ref).abs().max().item(), "ref_max": ref.abs().max().item(),
         "rel_l2_bf16_ref_vs_fp32_ref": ((bf_ref - ref).norm() / ref.norm()).item(),
         "frac_within_2ulp_bf16": ((out - ref).abs() <= 2 * 2 ** -7 * ref.abs() + 1e-6).float().mean().item()}
    print(json.dumps(r, indent=1))
    json.dump(r, open(a.out, "w"), indent=1)
    ok = r["rel_l2_vs_fp32_ref"] < 1e-2
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
