# SPDX-License-Identifier: Apache-2.0
"""Register the NemotronH architecture with vLLM.

Installed as a ``vllm.general_plugins`` entry point (``pyproject.toml`` in this
folder), so vLLM runs it in every process it starts before it resolves the
architecture. Registering replaces vLLM's own (GPU) ``NemotronHForCausalLM``.
"""

ARCHITECTURES = ("NemotronHForCausalLM",)


def register() -> None:
    from vllm import ModelRegistry

    for arch in ARCHITECTURES:
        ModelRegistry.register_model(arch, f"nemotron_h.factory:{arch}")
