# SPDX-License-Identifier: Apache-2.0
"""VLLM_NEURON_HOST_PREP_CACHE (CPU): the cached per-step decode inputs equal what the
per-step rebuild produces, and are rebuilt when their source changes.

Methods run on a bare ``object.__new__(NeuronModelRunner)`` as in test_mamba_align_state.py.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")


def _nmr():
    from vllm_neuron.vllm.worker import neuron_model_runner

    return neuron_model_runner


def _runner(cache_on=True, **attrs):
    runner = object.__new__(_nmr().NeuronModelRunner)
    runner.device = torch.device("cpu")
    runner._host_prep_cache_on = cache_on
    for k, v in attrs.items():
        setattr(runner, k, v)
    return runner


@pytest.mark.parametrize(
    "top_k,top_p,temp",
    [
        (None, None, None),
        (None, [0.95, 0.5], [1.0, 0.7]),
        ([5, 50000], [0.9, 1.0], None),
    ],
)
@pytest.mark.parametrize("rows", [0, 2, 8])
def test_sampling_params_host_matches_device_builder(top_k, top_p, temp, rows):
    nmr = _nmr()
    md = SimpleNamespace(
        top_k=None if top_k is None else torch.tensor(top_k, dtype=torch.int32),
        top_p=None if top_p is None else torch.tensor(top_p, dtype=torch.float32),
        temperature=None if temp is None else torch.tensor(temp, dtype=torch.float32),
    )
    ref = nmr.build_sampling_params_tensor(md, 2, torch.device("cpu"))
    if ref.shape[0] < rows:
        pad = torch.tensor([[-1.0, 1.0, 0.0]] * (rows - ref.shape[0]), dtype=ref.dtype)
        ref = torch.cat([ref, pad], dim=0)
    got = nmr.build_sampling_params_tensor_host(md, 2, rows)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert torch.equal(got, ref)
    assert got.is_contiguous()


def test_prep_cached_identity_and_key():
    r = _runner()
    a, b = object(), object()
    calls = []

    def f():
        calls.append(1)
        return torch.zeros(1)

    v1 = r._prep_cached(("x", 0), (1,), (a,), f)
    assert r._prep_cached(("x", 0), (1,), (a,), f) is v1
    assert r._prep_cached(("x", 0), (2,), (a,), f) is not v1  # key change
    v3 = r._prep_cached(("x", 0), (2,), (b,), f)  # ref change
    assert len(calls) == 3
    assert r._prep_cached(("x", 1), (2,), (b,), f) is not v3  # slot is per group
    r_off = _runner(cache_on=False)
    assert r_off._prep_cached(("x", 0), (1,), (a,), f) is not r_off._prep_cached(
        ("x", 0), (1,), (a,), f
    )


def _pool_runner(req_ids, live, cache_on=True):
    return _runner(
        cache_on=cache_on,
        _state_pool_slots={},
        _state_pool_free=list(range(8)),
        _state_pool_slot_pages=1,
        requests={r: object() for r in live},
        input_batch=SimpleNamespace(req_ids=list(req_ids)),
    )


def test_state_pool_cached_equals_rebuild_and_tracks_slots():
    ref = _pool_runner(["a", "b"], {"a", "b"})
    got = _pool_runner(["a", "b"], {"a", "b"})
    t_ref = ref._state_pool_table(4, 3)
    t1, c1 = got._state_pool_tables_cached(5, 4, 3)
    assert torch.equal(t1, t_ref) and torch.equal(c1, t_ref) and t1 is not c1
    assert got._state_pool_slots == ref._state_pool_slots
    # unchanged rows: same objects
    t2, c2 = got._state_pool_tables_cached(5, 4, 3)
    assert t2 is t1 and c2 is c1
    # request b finishes, c arrives in its row: slot of b is freed and reused
    for r in (ref, got):
        r.requests = {"a": object(), "c": object()}
        r.input_batch.req_ids = ["a", "c"]
    t_ref = ref._state_pool_table(4, 3)
    t3, _ = got._state_pool_tables_cached(5, 4, 3)
    assert torch.equal(t3, t_ref)
    assert got._state_pool_slots == ref._state_pool_slots
    # a row slot change with the same req ids must rebuild too
    got._state_pool_slots["a"] = 6
    t4, _ = got._state_pool_tables_cached(5, 4, 3)
    assert t4 is not t3 and int(t4[0, 0]) == 7


class _Buf:
    def __init__(self, gpu):
        self.gpu = gpu


class _Table:
    def __init__(self, gpu):
        self.block_table = _Buf(gpu)

    def get_device_tensor(self, n):
        return self.block_table.gpu[:n]


def test_full_block_table_clone_reuse_only_for_tracked_upload():
    nmr = _nmr()
    up = torch.arange(12, dtype=torch.int32).view(4, 3)
    state = SimpleNamespace(uploaded_device=up)
    r = _runner(_block_table_groups={0: state})
    t = _Table(up)
    c1 = r._full_block_table_clone(0, t, 2)
    assert torch.equal(c1, up[:2]) and c1.data_ptr() != up.data_ptr()
    assert r._full_block_table_clone(0, t, 2) is c1
    assert r._full_block_table_clone(0, t, 4) is not c1  # row count change
    # new upload (out of place) -> new clone with the new contents
    up2 = up + 100
    state.uploaded_device = up2
    t.block_table.gpu = up2
    c3 = r._full_block_table_clone(0, t, 4)
    assert torch.equal(c3, up2)
    # a device table not produced by the tracked upload is cloned every call
    t.block_table.gpu = up2.clone()
    assert r._full_block_table_clone(0, t, 4) is not r._full_block_table_clone(0, t, 4)
    assert nmr  # module imported
