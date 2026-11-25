# SPDX-License-Identifier: Apache-2.0
"""Module-level CPU checks of the NemotronH port against HF ``transformers``.

No vLLM engine: ``harness.Harness`` builds the model at world size 1, binds its
own caches and passes runner-shaped metadata. Needs a reduced-layer checkpoint
with at least one mamba, moe and attention layer
(``scripts/make_reduced_checkpoint.py``):

    NEMOTRON_H_TEST_CKPT=/mnt/nvme/models/nemotron-h-4l-fp32 \
        VLLM_NEURON_CPU_MODE=1 pytest contrib/models/nemotron_h/test/unit -v

The checkpoint's config ``dtype`` should be float32: these are exactness checks
(tolerance 1e-3 on logits of magnitude ~10).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

CKPT = os.environ.get("NEMOTRON_H_TEST_CKPT", "/mnt/nvme/models/nemotron-h-4l-fp32")
TOL = 1e-3

pytestmark = pytest.mark.skipif(not os.path.isdir(CKPT), reason=f"no checkpoint at {CKPT}")


@pytest.fixture(scope="module")
def models():
    from harness import Harness
    from transformers import AutoModelForCausalLM

    h = Harness(CKPT)
    from nemotron_h.mamba import align_hf_reference

    hf = AutoModelForCausalLM.from_pretrained(CKPT, dtype=h.model.config.torch_dtype).eval()
    align_hf_reference(hf)
    return h, hf


def hf_logits(hf, ids):
    with torch.no_grad():
        return hf(torch.tensor([ids]), use_cache=False).logits[0].float()


PROMPT = [1, 1784, 6129, 1307, 6233, 1395]
LONG = list(range(1000, 1300))


def test_layer_types_present(models):
    h, _ = models
    types = {layer.block_type for layer in h.model.backbone.layers}
    assert {"mamba", "moe", "attention"} <= types


def test_prefill_per_layer(models):
    h, hf = models
    ours = {}
    hooks = [
        blk.register_forward_hook(lambda m, a, o, i=i: ours.__setitem__(i, o.detach().clone()))
        for i, blk in enumerate(h.model.backbone.layers)
    ]
    try:
        logits = h.prefill(PROMPT, 0)
    finally:
        for hk in hooks:
            hk.remove()
    with torch.no_grad():
        out = hf(torch.tensor([PROMPT]), output_hidden_states=True, use_cache=False)
    n = len(h.model.backbone.layers)
    for i in range(n - 1):  # HF's last hidden state is after norm_f
        assert (ours[i] - out.hidden_states[i + 1][0]).abs().max() < TOL, f"layer {i}"
    assert (logits - out.logits[0]).abs().max() < TOL


def test_padded_prefill_then_decode(models):
    """Bucket padding (token 0, repeated last position) must not leak into state or KV."""
    h, hf = models
    logits = h.prefill(PROMPT, 1, pad_to=128)
    assert (logits - hf_logits(hf, PROMPT)).abs().max() < TOL
    seq = list(PROMPT)
    nxt = int(logits[-1].argmax())
    for _ in range(4):
        seq.append(nxt)
        d = h.decode([nxt], [len(seq) - 1], [1])[0]
        ref = hf_logits(hf, seq)[-1]
        assert (d - ref).abs().max() < TOL
        assert int(d.argmax()) == int(ref.argmax())
        nxt = int(d.argmax())


@pytest.mark.parametrize("kv_segment_size", [None, 512])
def test_chunked_prefill(models, kv_segment_size):
    """Two chunks hand Mamba conv/SSM state and the KV prefix over."""
    h, hf = models
    slot = 2 if kv_segment_size is None else 3
    ref = hf_logits(hf, LONG)
    a = h.prefill(LONG[:128], slot, start=0, chunked=True, kv_segment_size=kv_segment_size)
    b = h.prefill(LONG[128:], slot, start=128, chunked=True, pad_to=256, kv_segment_size=kv_segment_size)
    assert (a - ref[:128]).abs().max() < TOL
    assert (b - ref[128:]).abs().max() < TOL
    # ... and decode continues from the chunked state.
    nxt = int(b[-1].argmax())
    d = h.decode([nxt], [len(LONG)], [slot])[0]
    assert (d - hf_logits(hf, LONG + [nxt])[-1]).abs().max() < TOL


def test_batched_decode_matches_hf(models):
    """Decode two sequences in one batch; each row must match HF on its own sequence."""
    h, hf = models
    p1, p2 = PROMPT, [1, 422, 5544, 1278, 4200, 1044, 1278]
    l1, l2 = h.prefill(p1, 4), h.prefill(p2, 5)
    t1, t2 = int(l1[-1].argmax()), int(l2[-1].argmax())
    both = h.decode([t1, t2], [len(p1), len(p2)], [4, 5])
    r1 = hf_logits(hf, p1 + [t1])[-1]
    r2 = hf_logits(hf, p2 + [t2])[-1]
    assert (both[0] - r1).abs().max() < TOL
    assert (both[1] - r2).abs().max() < TOL
