# SPDX-License-Identifier: Apache-2.0
"""Config for NemotronH (``model_type: nemotron_h``).

NemotronH is a hybrid decoder. Every layer is ONE mixer with a pre-norm and a
residual (there is no separate MLP sub-block):

* ``mamba`` (HF ``linear_attention``): a Mamba-2 mixer with a fixed-size
  per-request state (conv window + SSM state).
* ``moe``: a routed MoE (sigmoid router, selection bias, non-gated squared-ReLU
  experts) plus one always-on shared expert.
* ``attention`` (HF ``full_attention``): GQA with **no positional encoding**
  (no RoPE; HF ``NemotronHAttention`` never rotates q/k).
* ``mlp``: a dense squared-ReLU MLP (not used by the 30B-A3B checkpoint).

Fields are read from the checkpoint config. Anything this port has not been
validated against raises instead of being approximated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

MAMBA = "mamba"
ATTENTION = "attention"
MOE = "moe"
MLP = "mlp"

# transformers 5 renames the legacy block types; accept both spellings.
_BLOCK_ALIASES = {
    "mamba": MAMBA,
    "linear_attention": MAMBA,
    "attention": ATTENTION,
    "full_attention": ATTENTION,
    "moe": MOE,
    "mlp": MLP,
}


def _dtype_of(cfg: PretrainedConfig, default: torch.dtype) -> torch.dtype:
    raw = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
    if raw is None:
        return default
    if isinstance(raw, torch.dtype):
        return raw
    return getattr(torch, str(raw).replace("torch.", ""))


def block_types_of(hf_config: PretrainedConfig) -> tuple[str, ...]:
    raw = getattr(hf_config, "layers_block_type", None) or getattr(
        hf_config, "layer_types", None
    )
    if raw is None:
        raise ValueError("NemotronH config has no layers_block_type / layer_types")
    unknown = sorted(set(raw) - set(_BLOCK_ALIASES))
    if unknown:
        raise NotImplementedError(f"unknown NemotronH block types {unknown}")
    return tuple(_BLOCK_ALIASES[t] for t in raw)


@dataclass
class NemotronHConfig:
    hidden_size: int
    block_types: tuple[str, ...]
    vocab_size: int
    norm_eps: float
    tie_word_embeddings: bool
    torch_dtype: torch.dtype

    # Attention
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    # Mamba-2
    mamba_num_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    n_groups: int
    conv_kernel: int
    chunk_size: int
    use_conv_bias: bool
    time_step_min: float
    ssm_dtype: torch.dtype

    # MoE
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    moe_shared_expert_intermediate_size: int
    n_shared_experts: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    # Dense MLP layers (``mlp`` blocks), if any
    intermediate_size: int

    neuron_config: object | None = None

    @property
    def num_hidden_layers(self) -> int:
        return len(self.block_types)

    @property
    def mamba_intermediate_size(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def mamba_conv_dim(self) -> int:
        return self.mamba_intermediate_size + 2 * self.n_groups * self.ssm_state_size

    def mamba_state_shapes(self, tp_size: int) -> tuple[tuple[int, ...], ...]:
        """Per-rank ``(conv_state_shape, ssm_state_shape)`` of one Mamba layer.

        From vLLM's own calculator, so the shapes agree with the pages vLLM
        sizes. At TP=4: conv ``(3, 1536)``, ssm ``(16, 64, 128)``.
        """
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )

        return tuple(
            MambaStateShapeCalculator.mamba2_state_shape(
                tp_size,
                self.mamba_intermediate_size,
                self.n_groups,
                self.mamba_num_heads,
                self.mamba_head_dim,
                self.ssm_state_size,
                self.conv_kernel,
                0,  # num_spec: the MTP head is not wired up
            )
        )

    @classmethod
    def from_hf(cls, cfg: PretrainedConfig) -> NemotronHConfig:
        block_types = block_types_of(cfg)

        if getattr(cfg, "moe_latent_size", None) is not None:
            raise NotImplementedError("moe_latent_size (latent MoE projection) is not implemented")
        for flag in ("attention_bias", "mlp_bias", "use_bias", "mamba_proj_bias"):
            if getattr(cfg, flag, False):
                raise NotImplementedError(f"{flag}=True is not implemented")
        if getattr(cfg, "mlp_hidden_act", "relu2") != "relu2":
            raise NotImplementedError(
                f"mlp_hidden_act={cfg.mlp_hidden_act!r}; only 'relu2' is implemented"
            )
        if getattr(cfg, "mamba_hidden_act", "silu") != "silu":
            raise NotImplementedError(
                f"mamba_hidden_act={cfg.mamba_hidden_act!r}; only 'silu' is implemented"
            )
        if getattr(cfg, "residual_in_fp32", False):
            raise NotImplementedError("residual_in_fp32=True is not implemented")
        n_group = int(getattr(cfg, "n_group", 1) or 1)
        topk_group = int(getattr(cfg, "topk_group", 1) or 1)
        if MOE in block_types and (n_group != 1 or topk_group != 1):
            raise NotImplementedError(
                f"grouped routing (n_group={n_group}, topk_group={topk_group}) is not "
                "implemented; with n_group=1 the group selection is a no-op"
            )
        n_shared = int(getattr(cfg, "n_shared_experts", 1) or 0)
        if MOE in block_types and n_shared != 1:
            raise NotImplementedError(f"n_shared_experts={n_shared}; only 1 is implemented")

        ssm_dtype = getattr(cfg, "mamba_ssm_cache_dtype", "float32") or "float32"
        ssm_dtype = (
            ssm_dtype
            if isinstance(ssm_dtype, torch.dtype)
            else getattr(torch, str(ssm_dtype).replace("torch.", ""))
        )
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return cls(
            hidden_size=cfg.hidden_size,
            block_types=block_types,
            vocab_size=cfg.vocab_size,
            norm_eps=float(cfg.layer_norm_epsilon),
            tie_word_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
            torch_dtype=_dtype_of(cfg, torch.bfloat16),
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            mamba_num_heads=cfg.mamba_num_heads,
            mamba_head_dim=cfg.mamba_head_dim,
            ssm_state_size=cfg.ssm_state_size,
            n_groups=cfg.n_groups,
            conv_kernel=cfg.conv_kernel,
            chunk_size=cfg.chunk_size,
            use_conv_bias=bool(cfg.use_conv_bias),
            # Used only with NEMOTRON_H_DT_CLAMP=hf (stock transformers clamps
            # dt to time_step_min in its chunked scan); see mamba.DT_CLAMP.
            time_step_min=float(cfg.time_step_min),
            ssm_dtype=ssm_dtype,
            n_routed_experts=int(getattr(cfg, "n_routed_experts", 0) or 0),
            num_experts_per_tok=int(getattr(cfg, "num_experts_per_tok", 0) or 0),
            moe_intermediate_size=int(getattr(cfg, "moe_intermediate_size", 0) or 0),
            moe_shared_expert_intermediate_size=int(
                getattr(cfg, "moe_shared_expert_intermediate_size", 0) or 0
            ),
            n_shared_experts=n_shared,
            norm_topk_prob=bool(getattr(cfg, "norm_topk_prob", True)),
            routed_scaling_factor=float(getattr(cfg, "routed_scaling_factor", 1.0)),
            intermediate_size=int(getattr(cfg, "intermediate_size", 0) or 0),
        )

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, text_neuron_config: object | None = None
    ) -> NemotronHConfig:
        config = cls.from_hf(hf_config)
        config.neuron_config = text_neuron_config
        return config
