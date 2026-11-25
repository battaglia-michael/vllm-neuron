# SPDX-License-Identifier: Apache-2.0
"""The class vLLM's ``ModelRegistry`` builds for NemotronH checkpoints.

Same pattern as ``contrib/models/qwen3_5/src/factory.py``: the class declares
the plugin's ``HasRecurrentState`` (the Mamba layers are reported as
``KVSpec.state_layers``) rather than vLLM's ``IsHybrid``, so vLLM's own Mamba
config hook does not resize the attention blocks against the runner's padding.

vLLM inspects the registered class for its text-generation protocol, so it has
an ``__init__`` taking ``vllm_config`` and ``embed_input_ids`` /
``compute_logits``; the Neuron runner builds the model with ``from_configs``
and only calls ``forward``.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class NemotronHForCausalLM(nn.Module):
    # Native compile route: --model-type=generic (as the Qwen3.5 contrib).
    NATIVE_COMPILER_MODEL_TYPE = "generic"

    @classmethod
    def num_recurrent_state_layers(cls, hf_config: PretrainedConfig) -> int:
        """``interfaces.HasRecurrentState``: the number of Mamba layers."""
        from .config import MAMBA, block_types_of

        return block_types_of(hf_config).count(MAMBA)

    def __init__(
        self,
        hf_config: PretrainedConfig | None = None,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config=None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        if hf_config is None:
            raise TypeError(f"build {type(self).__name__} with from_configs()")
        super().__init__()
        self._model = self.from_configs(hf_config, text_neuron_config, encoder_neuron_config)

    def forward(self, input_ids, positions, *args, **kwargs):
        return self._model(input_ids, positions, *args, **kwargs)

    def embed_input_ids(self, input_ids):
        return self._model.backbone.embeddings(input_ids, scatter_tokens=False)

    def compute_logits(self, hidden_states):
        return self._model.lm_head(hidden_states)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config=None,
    ) -> nn.Module:
        from .model import NemotronHForCausalLM as Impl

        return Impl.from_configs(hf_config, text_neuron_config=text_neuron_config)


__all__ = ["NemotronHForCausalLM"]
