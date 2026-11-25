# SPDX-License-Identifier: Apache-2.0
"""Device timing of ``mamba2_decode_kernel`` state-DMA variants (A05b), with the aligned page layout.

Copy of ``device_decode_dma_v3.py`` for T-mamba-state-align: a variant may set
``"_align": 1``. Its state starts as the same logical state converted to the
aligned layout (``state_align_check.to_aligned``), the kernel gets ``ssm_rows_a``,
and the check compares the logical conv window / SSM state (not raw halves) with
the first variant's.

Per batch size and variant (extra kernel kwargs, e.g. ``{"dma_eng": 3}``), two
graphs are compiled: one kernel call and ``--chain`` calls in sequence (the
state halves are updated in place, so each call depends on the previous one).
Kernel time per call = (t_chain - t_1) / (chain - 1), which removes the
dispatch and ``.cpu()`` overhead. Variants are timed in interleaved rounds and
the median is reported. Every variant's output and state after one call from
the same start state are compared with the first variant's (they only route
DMAs differently, so they should be bit-identical).

    NEURON_RT_VISIBLE_CORES=4 python device_decode_dma_v3.py --batches 1,8,32,128 \
        --variants '[{}, {"dma_eng": 3}]' --out x.json
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
from test_mamba_decode_kernel import Params, make_case  # noqa: E402
from state_align_check import layout, logical, to_aligned  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--variants", default='[{}, {"dma_eng": 3}]')
    ap.add_argument("--chain", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--half-rows", type=int, default=640)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    import libtorch_neuronx_lite  # noqa: F401
    from libtorch_neuronx_lite.compile.native_backend import register
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from nemotron_h.kernels.mamba_decode import mamba2_decode_kernel

    backend = register()
    # One compiled entry per (variant, chain length) of the same code object; keep them all.
    torch._dynamo.config.recompile_limit = 10_000
    torch._dynamo.config.accumulated_recompile_limit = 100_000
    dev = torch.device("neuron:0")
    p = Params()
    variants = json.loads(args.variants)
    out = {"chain": args.chain, "iters": args.iters, "rounds": args.rounds, "half_rows": args.half_rows,
           "cores": os.environ.get("NEURON_RT_VISIBLE_CORES"), "results": []}
    for n in [int(x) for x in args.batches.split(",")]:
        n_pad = 1 if n > 2 else 0
        proj, sa, sb, pages, wpages, live = make_case(p, n, args.half_rows, n + 4, n_pad, seed=n)
        d = lambda t: t.to(dev)
        const = dict(
            proj=d(proj), pages=d(pages.to(torch.int32).view(-1, 1)), wpages=d(wpages.to(torch.int32).view(-1, 1)),
            conv_w=d(p.conv_w), conv_b=d(p.conv_b), dt_bias=d(p.dt_bias), a_neg=d(-torch.exp(p.A_log)),
            d_skip=d(p.D), norm_w=d(p.norm_w),
        )
        torch._dynamo.reset()
        fns = {}
        checks = []
        ref = None
        for vi, var0 in enumerate(variants):
            var = {k: v for k, v in var0.items() if not k.startswith("_")}
            aligned = bool(var0.get("_align"))
            if aligned:
                var["ssm_rows_a"] = layout(p, args.half_rows)[3]
                sa_v, sb_v = to_aligned(p, sa, sb, args.half_rows)
            else:
                sa_v, sb_v = sa, sb
            for L in (1, args.chain):

                def fn(proj, state_a, state_b, pages, wpages, conv_w, conv_b, dt_bias, a_neg, d_skip, norm_w, _L=L, _var=var):
                    for _ in range(_L):
                        y, state_a, state_b = wrap_nki(mamba2_decode_kernel)[2](
                            proj=proj, state_a=state_a, state_b=state_b, pages=pages, wpages=wpages,
                            conv_w=conv_w, conv_b=conv_b, dt_bias=dt_bias, a_neg=a_neg, d_skip=d_skip,
                            norm_w=norm_w, n_groups=p.G, head_dim=p.P, d_state=p.N, eps=float(p.eps), **_var,
                        )
                    return y

                # A distinct code object per graph, so dynamo keeps one cache entry each
                # and never serves one variant's graph for another.
                fn = types.FunctionType(fn.__code__.replace(co_name=f"fn_{vi}_{L}"), fn.__globals__,
                                        None, fn.__defaults__, fn.__closure__)
                f = torch.compile(fn, backend=backend, fullgraph=True)
                st = {"state_a": d(sa_v.clone()), "state_b": d(sb_v.clone())}
                t0 = time.time()
                y = f(**const, **st).cpu()
                comp = round(time.time() - t0, 1)
                if L == 1:
                    a_c, b_c = logical(p, st["state_a"].cpu(), st["state_b"].cpu(), args.half_rows, aligned)
                    if ref is None:
                        ref = (y, a_c, b_c)
                    checks.append({
                        "variant": var0, "compile_s": comp,
                        "y_max_abs_vs_first": (y.float() - ref[0].float()).abs().max().item(),
                        "conv_max_abs_vs_first": (a_c - ref[1]).abs().max().item(),
                        "ssm_max_abs_vs_first": (b_c - ref[2]).abs().max().item(),
                        "y_identical": bool(torch.equal(y, ref[0])),
                        "finite": bool(torch.isfinite(y.float()).all()),
                    })
                    print("CHECK", n, json.dumps(checks[-1]), flush=True)
                for _ in range(3):
                    f(**const, **st).cpu()
                fns[(vi, L)] = (f, st)
        times = {k: [] for k in fns}
        for _ in range(args.rounds):
            for k, (f, st) in fns.items():
                f(**const, **st).cpu()
                t0 = time.time()
                for _ in range(args.iters):
                    f(**const, **st).cpu()
                times[k].append(1000 * (time.time() - t0) / args.iters)
        for vi, var in enumerate(variants):
            t1 = statistics.median(times[(vi, 1)])
            tL = statistics.median(times[(vi, args.chain)])
            per = [(b - a) / (args.chain - 1) for a, b in zip(times[(vi, 1)], times[(vi, args.chain)])]
            r = {"n": n, "variant": var, "call1_ms_median": t1, "chain_ms_median": tL,
                 "kernel_ms_per_call": (tL - t1) / (args.chain - 1),
                 "kernel_ms_per_round": per, "check": checks[vi]}
            print("RES", json.dumps(r), flush=True)
            out["results"].append(r)
        json.dump(out, open(args.out, "w"), indent=1)
        del fns


if __name__ == "__main__":
    main()
