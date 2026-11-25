# SPDX-License-Identifier: Apache-2.0
"""T10: the top-k MoE decode path equals the dense all-experts path (CPU, fp32, TP=1).

On CPU the expert gather falls back to ``index_select``; the device kernel is checked by the
on-device gate (results/trn2/opt2/gate in the work tree).
"""

import types

import pytest
import torch


@pytest.mark.parametrize("t", [1, 3, 8])
def test_topk_matches_dense(t, monkeypatch):
    from nemotron_h import moe as M

    class _G:
        world_size = 1

    monkeypatch.setattr(M, "get_tp_group", lambda: _G())
    if not M.UP_OUT_IN or M.DOWN_OUT_IN:
        pytest.skip("top-k path needs NEMOTRON_H_MOE_WT=up")
    cfg = types.SimpleNamespace(
        torch_dtype=torch.float32, hidden_size=64, n_routed_experts=16, num_experts_per_tok=6,
        norm_topk_prob=True, routed_scaling_factor=2.5, moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=48,
    )
    torch.manual_seed(0)
    m = M.NemotronHMoE(cfg, 0)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.2)
    x = torch.randn(t, 64)
    monkeypatch.setattr(M, "MOE_TOPK_MAX_T", 8)
    a = m(x, is_prefill=False)
    monkeypatch.setattr(M, "MOE_TOPK_MAX_T", 0)
    b = m(x, is_prefill=False)
    assert (a - b).abs().max().item() < 1e-4 * max(1.0, b.abs().max().item())
