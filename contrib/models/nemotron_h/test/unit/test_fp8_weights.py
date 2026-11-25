# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the FP8 routed-expert variant (NEMOTRON_H_FP8, R2-05): quantization format, scale folding, layouts."""
import pytest
import torch

ml_dtypes = pytest.importorskip("ml_dtypes")

from nemotron_h import moe as MO  # noqa: E402
from nemotron_h.kernels.moe_fp8 import up_from_t_layout, up_t_layout  # noqa: E402


def test_quant_range_and_trn2_bit_compat():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(64, 256, generator=g) * torch.logspace(-4, 1, 64).view(-1, 1)
    q, s = MO.fp8_quant_rows(w.to(torch.bfloat16))
    assert q.dtype == torch.float8_e4m3fn and s.shape == (64,)
    qf = q.float()
    assert qf.abs().max().item() <= 240.0
    # trn2 nl.float8_e4m3 (IEEE-style, max 240) decodes the same bytes to the same values for |x| <= 240
    raw = q.view(torch.uint8).numpy()
    legacy = raw.view(ml_dtypes.float8_e4m3).astype("float32")
    assert (torch.from_numpy(legacy) == qf).all()
    # per-row relative error of e4m3 round-to-nearest: <= 2^-4 of the row amax
    err = ((qf * s.view(-1, 1)) - w.to(torch.bfloat16).float()).abs().amax(1) / w.abs().amax(1)
    assert err.max().item() < 0.07


def test_scale_folding_matches_dequantized_expert():
    g = torch.Generator().manual_seed(1)
    I, H, E = 48, 256, 3
    up = (torch.randn(E * I, H, generator=g) * 0.02).to(torch.bfloat16)
    down = (torch.randn(E * I, H, generator=g) * 0.02).to(torch.bfloat16)
    x = torch.randn(5, H, generator=g)
    qu, su = MO.fp8_quant_rows(up)
    qd, sd = MO.fp8_quant_rows(down)
    c = su * su * sd
    for e in range(E):
        r = slice(e * I, (e + 1) * I)
        deq = torch.relu(x @ (qu[r].float() * su[r].view(-1, 1)).T) ** 2 @ (qd[r].float() * sd[r].view(-1, 1))
        fold = (torch.relu(x @ qu[r].float().T) ** 2 * c[r]) @ qd[r].float()
        torch.testing.assert_close(fold, deq, rtol=1e-4, atol=1e-6)


def test_up_t_layout_roundtrip():
    E, I, H = 3, 48, 256
    q = (torch.randn(E * I, H) * 10).clamp(-240, 240).to(torch.float8_e4m3fn)
    t = up_t_layout(q, E, I)
    assert t.shape == (E * 128, (H // 128) * I)
    nh = H // 128
    e, i, hh, p = 2, 7, 1, 33
    assert t.view(torch.uint8)[e * 128 + p, hh * I + i] == q.view(torch.uint8)[e * I + i, hh * 128 + p]
    back = up_from_t_layout(t.view(torch.uint8), E, I)
    assert torch.equal(back, q.view(torch.uint8))
    assert nh == 2
