# SPDX-License-Identifier: Apache-2.0
"""Aligned Mamba state page layout (``NEMOTRON_H_STATE_ALIGN=1``) vs the default layout.

The aligned layout stores the same conv window and SSM state, with pad rows between
the SSM groups so none straddles the two state halves. Checks that every path gives
the same values as the default layout on the same logical state:

* ``cpu``: the mixer's torch page helpers (``_read_states`` / ``_write_states`` with
  ``NEMOTRON_H_STATE_KERNEL=0``) round-trip, and the aligned page equals the
  default page with the SSM rows moved.
* ``sim``: the NKI CPU simulator runs ``mamba2_decode_kernel`` (v3 DMA routing) on
  the default layout and on the same state converted to the aligned layout; ``y``
  and the logical state after the step must be bit-identical and untouched pages
  unchanged. Same for ``_state_scatter_split_kernel``; ``_state_gather_rows_kernel``
  (aligned read) must return the default layout's ``[conv | ssm]`` rows.

``--kv`` (T11e): the same checks with the halves bound as the KV-aliased shared pool does
(2-byte ``[pages, block, 1, 128]`` tensors, kernels get ``state_rows``; prefix caching).

    NEURON_LOGICAL_NC_CONFIG=2 python state_align_check.py --what cpu,sim --out x.json
    NEURON_LOGICAL_NC_CONFIG=2 python state_align_check.py --what cpu,sim --kv --half-rows 1024 --out x.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_mamba_decode_kernel import Params, make_case  # noqa: E402

R = 128


def layout(p: Params, half_rows: int):
    """``(conv rows, ssm rows, group rows, ssm rows in half A of the aligned layout)``."""
    cr, rr = p.conv_numel // R, p.ssm_numel // R
    gr = rr // p.G
    in_a = (half_rows - cr) // gr * gr
    return cr, rr, gr, in_a


def logical(p, sa, sb, half_rows, aligned):
    """``(conv [pages, cr, R], ssm [pages, rr, R])`` of every page."""
    cr, rr, gr, in_a = layout(p, half_rows)
    rows = torch.cat([sa, sb], dim=1)
    conv = rows[:, :cr]
    if aligned:
        ssm = torch.cat([sa[:, cr : cr + in_a], sb[:, : rr - in_a]], dim=1)
    else:
        ssm = rows[:, cr : cr + rr]
    return conv, ssm


def to_aligned(p, sa, sb, half_rows):
    """The same logical state in the aligned layout; pad rows keep arbitrary (old) values."""
    cr, rr, gr, in_a = layout(p, half_rows)
    conv, ssm = logical(p, sa, sb, half_rows, False)
    a, b = sb.clone(), sa.clone()  # junk in the pad rows
    a[:, :cr] = conv
    a[:, cr : cr + in_a] = ssm[:, :in_a]
    b[:, : rr - in_a] = ssm[:, in_a:]
    return a, b


def as_kv(t):
    """float32 ``[pages, rows, R]`` as the bf16 ``[pages, 2 * rows, 1, R]`` K / V tensor the shared pool binds."""
    n, rows, r = t.shape
    return t.contiguous().view(torch.bfloat16).view(n, 2 * rows, 1, r)


def from_kv(t, rows):
    """Inverse of :func:`as_kv`."""
    return t.contiguous().view(t.shape[0], rows, 2 * R).view(torch.float32)


def canon_bf16_nan(t):
    """float32 ``t`` with every 2-byte half that is a bf16 NaN set to ``0xffff``.

    The NKI CPU simulator holds 2-byte tensors as bf16 values and canonicalizes NaNs, so
    a float32 value whose low half reads as a bf16 NaN would come back changed after a
    move through the KV-aliased (bf16) view. On the device these are byte-exact DMAs. The
    ``--kv`` sim inputs are canonicalized first, so a copy must still be bit-exact.
    """
    v = t.contiguous().view(torch.int32)
    out = []
    for shift in (0, 16):
        h = (v >> shift) & 0xFFFF
        nan = ((h & 0x7F80) == 0x7F80) & ((h & 0x7F) != 0)
        out.append(torch.where(nan, torch.full_like(h, 0xFFFF), h))
    return ((out[1] << 16) | out[0]).view(torch.float32).view(t.shape)


def check_cpu(p: Params, out: dict, half_rows_list=(640, 600, 700), kv: bool = False):
    os.environ["NEMOTRON_H_STATE_KERNEL"] = "0"
    from nemotron_h import mamba as M

    res = []
    for half_rows in half_rows_list:
        num_pages, n = 9, 4
        g = torch.Generator().manual_seed(half_rows)
        conv = torch.randn(n, p.K - 1, p.conv_dim, generator=g)
        ssm = torch.randn(n, p.nh, p.P, p.N, generator=g)
        idx = torch.tensor([3, 1, 7, 0])
        pages = {}
        for aligned in (False, True):
            m = types.SimpleNamespace(
                conv_numel=p.conv_numel, ssm_numel=p.ssm_numel, _STATE_ROW=R, n_groups=p.G,
                layer_name="t", conv_kernel_size=p.K, conv_dim=p.conv_dim, num_heads=p.nh,
                head_dim=p.P, d_state=p.N, ssm_rows_a=None,
                state_kv=kv, state_shape=(num_pages, half_rows, R),
            )
            for name in ("aligned_ssm_rows_a", "_ssm_from_page", "_page_from_states", "_read_states",
                         "_write_states", "_use_state_kernel", "_f32_state", "_kernel_state_args"):
                setattr(m, name, types.MethodType(getattr(M.NemotronHMamba2Mixer, name), m))
            if aligned:
                m.ssm_rows_a = m.aligned_ssm_rows_a(half_rows * R)
            m.state_a = torch.full((num_pages, half_rows, R), 7.0)
            m.state_b = torch.full((num_pages, half_rows, R), 7.0)
            if kv:
                m.state_a, m.state_b = as_kv(m.state_a), as_kv(m.state_b)
            m._write_states(idx, conv, ssm, torch.tensor([True, True, True, False]))
            c2, s2 = m._read_states(idx)
            sa_f, sb_f = (from_kv(m.state_a, half_rows), from_kv(m.state_b, half_rows)) if kv else (m.state_a, m.state_b)
            pages[aligned] = (sa_f.clone(), sb_f.clone(), c2, s2, m.ssm_rows_a)
        (a0, b0, c0, s0, _), (a1, b1, c1, s1, in_a) = pages[False], pages[True]
        live = torch.tensor([True, True, True, False])
        lc0, ls0 = logical(p, a0, b0, half_rows, False)
        lc1, ls1 = logical(p, a1, b1, half_rows, in_a is not None)
        r = {
            "half_rows": half_rows,
            "kv": kv,
            "ssm_rows_a": in_a,
            "roundtrip_default": bool(torch.equal(c0[live], conv[live]) and torch.equal(s0[live], ssm[live])),
            "roundtrip_aligned": bool(torch.equal(c1[live], conv[live]) and torch.equal(s1[live], ssm[live])),
            "read_equal": bool(torch.equal(c0, c1) and torch.equal(s0, s1)),
            "logical_pages_equal": bool(torch.equal(lc0, lc1) and torch.equal(ls0, ls1)),
        }
        if in_a is not None:
            cr, rr, gr, _ = layout(p, half_rows)
            r["aligned_pad_rows_zero_on_written_pages"] = bool(
                (a1[idx, cr + in_a :] == 0).all() and (b1[idx, rr - in_a :] == 0).all()
            )
        print("CPU", json.dumps(r), flush=True)
        res.append(r)
    out["cpu"] = res
    return all(v for r in res for k, v in r.items() if isinstance(v, bool) and k != "kv")


def check_sim(p: Params, out: dict, batches, half_rows_list, extra: dict, kv: bool = False):
    from libtorch_neuronx_lite.nki.nki_cpu_sim import simulate_nki_kernel

    from nemotron_h.kernels.mamba_decode import mamba2_decode_kernel
    from nemotron_h.kernels.state_cache import _state_gather_rows_kernel, _state_scatter_split_kernel

    res, ok = [], True
    for half_rows in half_rows_list:
        cr, rr, gr, in_a = layout(p, half_rows)
        for n in batches:
            n_pad = 1 if n > 2 else 0
            num_pages = max(n + 3, 8)
            proj, sa, sb, pages, wpages, live = make_case(p, n, half_rows, num_pages, n_pad, seed=n * 7 + half_rows)
            if kv:
                sa, sb = canon_bf16_nan(sa), canon_bf16_nan(sb)
            base = dict(
                proj=proj, pages=pages.to(torch.int32).view(-1, 1), wpages=wpages.to(torch.int32).view(-1, 1),
                conv_w=p.conv_w, conv_b=p.conv_b, dt_bias=p.dt_bias, a_neg=-torch.exp(p.A_log),
                d_skip=p.D, norm_w=p.norm_w, n_groups=p.G, head_dim=p.P, d_state=p.N, eps=float(p.eps),
                **extra,
            )
            sa1, sb1 = to_aligned(p, sa, sb, half_rows)
            if kv:
                # Shared pool: the kernels get the 2-byte K / V tensors plus state_rows.
                st = lambda a, b: dict(state_a=as_kv(a.clone()), state_b=as_kv(b.clone()), state_rows=half_rows)
                back = lambda t: from_kv(t, half_rows)
            else:
                st = lambda a, b: dict(state_a=a.clone(), state_b=b.clone())
                back = lambda t: t
            y0, a0, b0 = simulate_nki_kernel(mamba2_decode_kernel, 2, dict(base, **st(sa, sb)))
            y1, a1, b1 = simulate_nki_kernel(mamba2_decode_kernel, 2, dict(base, **st(sa1, sb1), ssm_rows_a=in_a))
            a0, b0, a1, b1 = back(a0), back(b0), back(a1), back(b1)
            lc0, ls0 = logical(p, a0, b0, half_rows, False)
            lc1, ls1 = logical(p, a1, b1, half_rows, True)
            touched = set(pages[live].tolist())
            others = [q for q in range(num_pages) if q not in touched]
            r = {
                "kernel": "mamba2_decode", "n": n, "half_rows": half_rows, "kv": kv, "ssm_rows_a": in_a,
                "y_identical": bool(torch.equal(y0, y1)),
                "conv_identical": bool(torch.equal(lc0, lc1)),
                "ssm_identical": bool(torch.equal(ls0, ls1)),
                "state_changed": bool(not torch.equal(ls0, logical(p, sa, sb, half_rows, False)[1])),
                "aligned_untouched_pages_equal": bool(torch.equal(a1[others], sa1[others]) and torch.equal(b1[others], sb1[others])),
                "aligned_pad_rows_unchanged": bool(
                    torch.equal(a1[:, cr + in_a :], sa1[:, cr + in_a :]) and torch.equal(b1[:, rr - in_a :], sb1[:, rr - in_a :])
                ),
            }
            print("SIM", json.dumps(r), flush=True)
            res.append(r)
            ok &= all(v for k, v in r.items() if isinstance(v, bool) and k != "kv")

            # Scatter-split kernel (prefill / torch-decode state write).
            g = torch.Generator().manual_seed(n)
            conv = torch.randn(n, cr, R, generator=g)
            rec = torch.randn(n, rr, R, generator=g)
            if kv:
                conv, rec = canon_bf16_nan(conv), canon_bf16_nan(rec)
            kw = dict(conv=conv, rec=rec, pages=wpages.to(torch.int32).view(-1, 1))
            s0a, s0b = simulate_nki_kernel(_state_scatter_split_kernel, 2, dict(kw, **st(sa, sb)))
            s1a, s1b = simulate_nki_kernel(_state_scatter_split_kernel, 2, dict(kw, **st(sa1, sb1), rec_in_a=in_a))
            s0a, s0b, s1a, s1b = back(s0a), back(s0b), back(s1a), back(s1b)
            l0, l1 = logical(p, s0a, s0b, half_rows, False), logical(p, s1a, s1b, half_rows, True)
            wl = wpages[live]
            r = {
                "kernel": "state_scatter_split", "n": n, "half_rows": half_rows, "kv": kv, "ssm_rows_a": in_a,
                "logical_identical": bool(torch.equal(l0[0], l1[0]) and torch.equal(l0[1], l1[1])),
                "written_values": bool(torch.equal(l1[0][wl], conv[live]) and torch.equal(l1[1][wl], rec[live])),
                "aligned_untouched_pages_equal": bool(torch.equal(s1a[others], sa1[others]) and torch.equal(s1b[others], sb1[others])),
            }
            print("SIM", json.dumps(r), flush=True)
            res.append(r)
            ok &= all(v for k, v in r.items() if isinstance(v, bool) and k != "kv")

            # Aligned-layout read kernel (prefill state read): [conv | ssm] rows in order.
            rd = simulate_nki_kernel(
                _state_gather_rows_kernel, 2,
                dict(**st(sa1, sb1), pages=pages.to(torch.int32).view(-1, 1),
                     a_rows=cr + in_a, b_rows=rr - in_a),
            )
            rd = rd[0] if isinstance(rd, (tuple, list)) else rd
            c_ref, s_ref = logical(p, sa, sb, half_rows, False)
            r = {
                "kernel": "state_gather_rows", "n": n, "half_rows": half_rows, "kv": kv, "ssm_rows_a": in_a,
                "equals_default_page": bool(torch.equal(rd, torch.cat([c_ref[pages], s_ref[pages]], dim=1))),
            }
            print("SIM", json.dumps(r), flush=True)
            res.append(r)
            ok &= r["equals_default_page"]
    out["sim"] = res
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="cpu,sim")
    ap.add_argument("--batches", default="1,5,32")
    ap.add_argument("--half-rows", default="640")
    ap.add_argument("--kw", default='{"dma_eng": 1, "dma_eng1": 7}', help="decode kernel DMA kwargs (default v3)")
    ap.add_argument("--kv", action="store_true", help="bind the halves as the KV-aliased shared pool does (T11e)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    torch.set_grad_enabled(False)
    p = Params()
    out = {"kernel_kwargs": json.loads(args.kw)}
    ok = True
    if "cpu" in args.what:
        cpu_rows = [int(x) for x in args.half_rows.split(",")] if args.kv else (640, 600, 700)
        ok &= check_cpu(p, out, cpu_rows, args.kv)
    if "sim" in args.what:
        ok &= check_sim(p, out, [int(x) for x in args.batches.split(",")],
                        [int(x) for x in args.half_rows.split(",")], json.loads(args.kw), args.kv)
    out["kv"] = args.kv
    out["all_ok"] = bool(ok)
    print("ALL_OK", ok)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
