#!/usr/bin/env python3
"""Audit Kimi K3 config/index consistency without downloading model weights."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SHARD_RE = re.compile(r"model-(\d{5})-of-(\d{6})\.safetensors$")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.")


def _text_config(config: dict) -> dict:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("config has no object-valued text_config")
    return text_config


def audit(config: dict, index: dict) -> dict:
    text_config = _text_config(config)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("index has no object-valued weight_map")

    errors: list[str] = []
    names = set(weight_map)
    shards = set(weight_map.values())

    shard_numbers: list[int] = []
    declared_shards: set[int] = set()
    for shard in shards:
        match = SHARD_RE.fullmatch(shard)
        if not match:
            errors.append(f"unexpected shard name: {shard}")
            continue
        number, total = (int(part) for part in match.groups())
        shard_numbers.append(number)
        declared_shards.add(total)

    expected_shards = max(declared_shards) if declared_shards else 0
    if declared_shards and len(declared_shards) != 1:
        errors.append(f"inconsistent shard totals: {sorted(declared_shards)}")
    if expected_shards and sorted(shard_numbers) != list(range(1, expected_shards + 1)):
        errors.append("shard numbering is not contiguous")

    layer_ids = {
        int(match.group(1))
        for name in names
        if (match := LAYER_RE.search(name)) is not None
    }
    num_layers = int(text_config.get("num_hidden_layers", -1))
    if layer_ids != set(range(num_layers)):
        errors.append(
            f"index layer ids do not match config: ids={len(layer_ids)} "
            f"config_layers={num_layers}"
        )

    linear_config = text_config.get("linear_attn_config") or {}
    kda_layers = {int(value) for value in linear_config.get("kda_layers", [])}
    full_layers = {int(value) for value in linear_config.get("full_attn_layers", [])}
    expected_config_layers = set(range(1, num_layers + 1))
    if kda_layers & full_layers:
        errors.append("config KDA/full-attention layer lists overlap")
    if kda_layers | full_layers != expected_config_layers:
        errors.append("config KDA/full-attention layer lists are not a 1-based partition")

    expert_layers: dict[int, set[int]] = {}
    for name in names:
        match = EXPERT_RE.search(name)
        if match is None:
            continue
        layer, expert = (int(part) for part in match.groups())
        expert_layers.setdefault(layer, set()).add(expert)

    expected_experts = int(text_config.get("num_experts", -1))
    expected_moe_layers = set(range(int(text_config.get("first_k_dense_replace", 0)), num_layers))
    if set(expert_layers) != expected_moe_layers:
        errors.append("expert-bearing layer ids do not match first_k_dense_replace")
    for layer, expert_ids in expert_layers.items():
        if expert_ids != set(range(expected_experts)):
            errors.append(f"layer {layer} does not contain all configured experts")
            break

    packed = {name for name in names if name.endswith(".weight_packed")}
    scales = {name for name in names if name.endswith(".weight_scale")}
    expected_scales = {
        name.removesuffix(".weight_packed") + ".weight_scale" for name in packed
    }
    if expected_scales != scales:
        errors.append("packed expert weights and scale tensors are not paired")

    metadata = index.get("metadata") or {}
    total_size = metadata.get("total_size")
    quant = text_config.get("quantization_config") or {}
    config_groups = quant.get("config_groups") or {}
    group_zero = config_groups.get("group_0") or {}
    weight_quant = group_zero.get("weights") or {}

    return {
        "ok": not errors,
        "errors": errors,
        "model_architecture": config.get("architectures"),
        "index_total_size_bytes": total_size,
        "tensor_count": len(names),
        "shard_count": len(shards),
        "shard_number_range": [min(shard_numbers), max(shard_numbers)] if shard_numbers else [],
        "layer_count": len(layer_ids),
        "kda_layer_count": len(kda_layers),
        "full_attention_layer_count": len(full_layers),
        "expert_layer_count": len(expert_layers),
        "experts_per_moe_layer": expected_experts,
        "routed_expert_count": expected_experts,
        "experts_per_token": text_config.get("num_experts_per_token"),
        "shared_expert_count": text_config.get("num_shared_experts"),
        "packed_weight_tensor_count": len(packed),
        "scale_tensor_count": len(scales),
        "quantization_format": quant.get("format"),
        "quantization_bits": weight_quant.get("num_bits"),
        "full_weights_downloaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="fixed-revision config.json")
    parser.add_argument("index", type=Path, help="fixed-revision safetensors index")
    args = parser.parse_args()

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        index = json.loads(args.index.read_text(encoding="utf-8"))
        result = audit(config, index)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"audit input error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
