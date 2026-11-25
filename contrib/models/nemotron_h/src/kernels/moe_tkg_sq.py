# SPDX-License-Identifier: Apache-2.0
"""NKI Library ``moe_tkg`` (selective-expert mode) for non-gated squared-ReLU experts (T09b).

``nkilib.core.moe.moe_tkg.moe_tkg`` computes ``down(act(gate(x)) * up(x))`` on a fused
``gate_up [E, H, 2, I]`` weight. NemotronH experts have no gate projection and use
``relu(up(x))^2``. The library already has a ``skip_gate_proj`` mode inside its MLP code, but in
nkilib 0.7 (Neuron SDK 2.33):

* the ``moe_tkg()`` entry point does not pass ``skip_gate_proj`` to ``MLPParameters``;
* the skip-gate activation calls ``get_nl_act_fn_from_type``, which rejects ``SquaredReLU``
  ("requires two instructions");
* the expert loops always index ``GateUpDim.UP`` (= 1) of the weight, so a single-projection
  ``[E, H, 1, I]`` weight is out of bounds.

This module patches the three when it is imported (only with ``NEMOTRON_H_MOE_TKG=1``; nothing
else in NemotronH uses ``moe_tkg``; patches stay in place because the NKI trace can run after the
torch-level call returns): ``MLPParameters`` is
built with ``skip_gate_proj=True``, ``GateUpDim.UP`` reads index 0, and ``SquaredReLU`` becomes
``nisa.activation(relu)`` followed by an in-place square on the Vector Engine. The up weight is
``[E, H, 1, I]`` (in, out): one extra copy of the routed up weights (``NemotronHMoE.experts_up_tkg``),
because every other MoE path needs them ``[E * I, H]`` (out, in) (T09; a ``[E, H, I]``-only layout
makes the dense path 2.2x slower at batch 32 and 17% slower in prefill, bench/opt/T09b).

Unit test (bench/opt/T09b/moe_tkg_unit.py, 8 MoE layers at batch 1, TP=4 rank-0 shapes,
results/trn2/opt-tkg/unit/): T14 fused kernel 0.309, this path 0.231 ms per layer (upper bounds incl.
per-call overhead), same error vs the CPU reference.
"""

import importlib
from types import SimpleNamespace

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

_P = "nkilib.core.moe.moe_tkg."
_SQ = object()  # marker op: relu, then square


def _activation(dst, op, data, **kw):
    if op is _SQ:
        nisa.activation(dst=dst, op=nl.relu, data=data, **kw)
        return nisa.tensor_tensor(dst=dst, data1=dst, data2=dst, op=nl.multiply)
    return nisa.activation(dst=dst, op=op, data=data, **kw)


class _NisaProxy:
    def __getattr__(self, name):
        return _activation if name == "activation" else getattr(nisa, name)


def _apply_patches():
    mt = importlib.import_module(_P + "moe_tkg")
    if getattr(mt, "_nemotron_h_sq_patched", False):
        return mt.moe_tkg
    mp = mt.MLPParameters
    mt.MLPParameters = lambda **kw: mp(skip_gate_proj=True, **kw)
    up0 = SimpleNamespace(GATE=SimpleNamespace(value=0), UP=SimpleNamespace(value=0))
    for n in ("selective_expert_impl", "all_expert_impl"):
        m = importlib.import_module(_P + n)
        if hasattr(m, "GateUpDim"):
            m.GateUpDim = up0
    for n in ("mlp_tkg_gate_up_projection", "mlp_tkg_gate_up_projection_lhs_rhs_swap"):
        m = importlib.import_module(_P + n)
        if hasattr(m, "get_nl_act_fn_from_type"):
            orig = m.get_nl_act_fn_from_type
            m.get_nl_act_fn_from_type = lambda t, _o=orig: _SQ if t == ActFnType.SquaredReLU else _o(t)
        m.nisa = _NisaProxy()
    mt._nemotron_h_sq_patched = True
    return mt.moe_tkg


_moe_tkg = _apply_patches()


@nki.jit
def _moe_tkg_sq_kernel(hidden_input, expert_up_weights, expert_down_weights, expert_affinities, expert_index):
    return _moe_tkg(
        hidden_input=hidden_input,
        expert_gate_up_weights=expert_up_weights,
        expert_down_weights=expert_down_weights,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        is_all_expert=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        activation_fn=ActFnType.SquaredReLU,
    )


def moe_tkg_sq_decode(x: Tensor, up_tkg: Tensor, down: Tensor, affinities: Tensor, experts: Tensor) -> Tensor:
    """Routed-expert sum for ``x`` ``[T, H]`` (T <= 128).

    ``up_tkg`` ``[E, H, 1, I]`` (in, out); ``down`` ``[E * I, H]`` (in, out), viewed ``[E, I, H]``;
    ``affinities`` ``[T, E]`` float32 routing weights (zero off the top-k); ``experts`` ``[T, k]``.
    """
    E, H, _, inter = up_tkg.shape
    return wrap_nki(_moe_tkg_sq_kernel)[2](
        hidden_input=x,
        expert_up_weights=up_tkg,
        expert_down_weights=down.view(E, inter, H),
        expert_affinities=affinities.to(torch.float32),
        expert_index=experts.to(torch.int32),
    )
