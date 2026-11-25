# SPDX-License-Identifier: Apache-2.0
"""Build a reduced-layer NemotronH checkpoint from real weights.

Copies the embeddings, ``norm_f``, ``lm_head`` and the chosen layers (renumbered
0..n-1) into one safetensors file, and writes a ``config.json`` whose
``layers_block_type`` matches. MTP is dropped. The weights stay bf16; ``--dtype``
only sets the config's ``dtype`` (the dtype vLLM and the model compute in).

    python make_reduced_checkpoint.py $MODEL_DIR /mnt/nvme/models/nemotron-h-4l \
        --layers 0,1,5,7 --dtype float32
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

_TYPE_NAMES = {"mamba": "mamba", "linear_attention": "mamba", "attention": "attention",
               "full_attention": "attention", "moe": "moe", "mlp": "mlp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", default="0,1,5,7", help="source layer ids, in order")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    cfg = json.load(open(os.path.join(args.src, "config.json")))
    src_types = cfg["layers_block_type"]
    weight_map = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))["weight_map"]

    wanted: dict[str, str] = {}  # new key -> old key
    for new, old in enumerate(layers):
        prefix = f"backbone.layers.{old}."
        for key in weight_map:
            if key.startswith(prefix):
                wanted[f"backbone.layers.{new}." + key[len(prefix):]] = key
    for key in ("backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"):
        wanted[key] = key

    tensors = {}
    by_file: dict[str, list[tuple[str, str]]] = {}
    for new, old in wanted.items():
        by_file.setdefault(weight_map[old], []).append((new, old))
    for fname, pairs in sorted(by_file.items()):
        with safe_open(os.path.join(args.src, fname), "pt") as f:
            for new, old in pairs:
                tensors[new] = f.get_tensor(old).contiguous()

    os.makedirs(args.dst, exist_ok=True)
    save_file(tensors, os.path.join(args.dst, "model.safetensors"), metadata={"format": "pt"})

    cfg["layers_block_type"] = [_TYPE_NAMES[src_types[i]] for i in layers]
    cfg["num_hidden_layers"] = len(layers)
    cfg["num_nextn_predict_layers"] = 0
    cfg["dtype"] = args.dtype
    cfg["_reduced_from"] = {"source": os.path.abspath(args.src), "layers": layers}
    json.dump(cfg, open(os.path.join(args.dst, "config.json"), "w"), indent=2)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "generation_config.json", "chat_template.jinja"):
        p = os.path.join(args.src, name)
        if os.path.exists(p):
            shutil.copy(p, args.dst)
    print(f"wrote {len(tensors)} tensors, layers {layers} -> {cfg['layers_block_type']}, dtype {args.dtype}")


if __name__ == "__main__":
    main()
