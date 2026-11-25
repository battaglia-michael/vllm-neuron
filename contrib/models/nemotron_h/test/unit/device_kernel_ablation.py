# SPDX-License-Identifier: Apache-2.0
"""Device timing of ``mamba2_decode_kernel`` alone, with parts switched off (``ablate`` bits).

ablate: 1 = no conv-window DMAs, 2 = no SSM-state DMAs. Numbers are per call,
including a blocking ``.cpu()`` of the [n, I] output; ``null`` is the same for a
trivial graph. For finding the kernel's bottleneck only (the ablated variants
compute wrong values).

    NEURON_RT_VISIBLE_CORES=4 python device_kernel_ablation.py --batches 32,128 --ablate 0,1,2,3 --out x.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_mamba_decode_kernel import Params, make_case  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="32,128")
    ap.add_argument("--ablate", default="0,1,2,3")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--half-rows", type=int, default=530)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default='[{}]', help="JSON list of extra kernel kwargs; each run with every --ablate")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    import libtorch_neuronx_lite  # noqa: F401
    from libtorch_neuronx_lite.compile.native_backend import register
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from nemotron_h.kernels.mamba_decode import mamba2_decode_kernel

    backend = register()
    dev = torch.device("neuron:0")
    p = Params()
    rows = []
    for n in [int(x) for x in args.batches.split(",")]:
        proj, sa, sb, pages, wpages, live = make_case(p, n, args.half_rows, n + 4, 1, seed=n)
        d = lambda t: t.to(dev)
        tens = dict(
            proj=d(proj), state_a=d(sa), state_b=d(sb),
            pages=d(pages.to(torch.int32).view(-1, 1)), wpages=d(wpages.to(torch.int32).view(-1, 1)),
            conv_w=d(p.conv_w), conv_b=d(p.conv_b), dt_bias=d(p.dt_bias), a_neg=d(-torch.exp(p.A_log)),
            d_skip=d(p.D), norm_w=d(p.norm_w),
        )
        for var, a in [(v, a) for v in json.loads(args.variants) for a in [int(x) for x in args.ablate.split(",")]]:
            torch._dynamo.reset()

            def fn(proj, state_a, state_b, pages, wpages, conv_w, conv_b, dt_bias, a_neg, d_skip, norm_w):
                y, _, _ = wrap_nki(mamba2_decode_kernel)[2](
                    proj=proj, state_a=state_a, state_b=state_b, pages=pages, wpages=wpages,
                    conv_w=conv_w, conv_b=conv_b, dt_bias=dt_bias, a_neg=a_neg, d_skip=d_skip,
                    norm_w=norm_w, n_groups=p.G, head_dim=p.P, d_state=p.N, eps=float(p.eps), ablate=a, **var,
                )
                return y

            f = torch.compile(fn, backend=backend, fullgraph=True)
            t0 = time.time()
            f(**tens).cpu()
            c = time.time() - t0
            for _ in range(4):
                f(**tens).cpu()
            t0 = time.time()
            for _ in range(args.iters):
                f(**tens).cpu()
            ms = 1000 * (time.time() - t0) / args.iters
            r = {"n": n, "ablate": a, "variant": var, "call_ms": ms, "compile_s": round(c, 1)}
            print("ABL", json.dumps(r), flush=True)
            rows.append(r)
        torch._dynamo.reset()
        null = torch.compile(lambda h: (h[:, :1] * 2).expand(n, p.I).contiguous(), backend=backend, fullgraph=True)
        for _ in range(4):
            null(tens["proj"]).cpu()
        t0 = time.time()
        for _ in range(args.iters):
            null(tens["proj"]).cpu()
        rows.append({"n": n, "ablate": "null", "call_ms": 1000 * (time.time() - t0) / args.iters})
        print("ABL", json.dumps(rows[-1]), flush=True)
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
