#!/usr/bin/env python3
"""Local teaching audit for the documented Kimi K2.7 Code request contract.

This does not call Kimi API, load model weights, or reproduce a production cache.
"""

from __future__ import annotations

import json
from typing import Any


DEFAULTS: dict[str, Any] = {
    "thinking": {"type": "enabled"},
    "max_tokens": 32768,
    "temperature": 1.0,
    "top_p": 0.95,
    "n": 1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "tool_choice": "auto",
}
FIXED_VALUES = {
    "temperature": 1.0,
    "top_p": 0.95,
    "n": 1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def audit_request(params: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    effective = DEFAULTS | params
    errors: list[str] = []
    if effective["thinking"] != {"type": "enabled"}:
        errors.append("K2.7 Code requires thinking to remain enabled")
    for name, required in FIXED_VALUES.items():
        if effective[name] != required:
            errors.append(f"{name} must equal the documented fixed value {required}")
    if (
        not isinstance(effective["tool_choice"], str)
        or effective["tool_choice"] not in {"auto", "none"}
    ):
        errors.append("tool_choice must be auto or none")
    if not isinstance(effective["max_tokens"], int) or effective["max_tokens"] <= 0:
        errors.append("max_tokens must be a positive integer")
    return not errors, effective, errors


def audit_tool_round_trip(
    assistant_call: dict[str, Any],
    tool_results: list[dict[str, Any]],
    next_assistant: dict[str, Any],
) -> tuple[bool, list[str]]:
    calls = assistant_call.get("tool_calls", [])
    if not isinstance(calls, list) or not isinstance(tool_results, list):
        return False, ["tool_calls and tool_results must be lists"]
    if any(not isinstance(call, dict) for call in calls) or any(
        not isinstance(result, dict) for result in tool_results
    ):
        return False, ["tool calls and tool results must be objects"]
    call_ids = [call.get("id") for call in calls]
    result_ids = [result.get("tool_call_id") for result in tool_results]
    warnings: list[str] = []
    if any(not isinstance(call_id, str) or not call_id for call_id in call_ids):
        return False, ["tool-call ids must be non-empty strings"]
    if any(not isinstance(result_id, str) or not result_id for result_id in result_ids):
        return False, ["tool-result call ids must be non-empty strings"]
    if len(call_ids) != len(set(call_ids)):
        return False, ["tool-call ids must be present and unique"]
    if sorted(call_ids) != sorted(result_ids):
        return False, ["each tool result must correlate to exactly one proposed call"]
    if assistant_call.get("reasoning_content") and not next_assistant.get("reasoning_content"):
        warnings.append("reasoning_content was not replayed; continuity may degrade")
    return True, warnings


def packed_int4_raw_bytes(parameter_count: int) -> int:
    if parameter_count < 0:
        raise ValueError("parameter_count must be non-negative")
    return (parameter_count * 4 + 7) // 8


def mla_cache_shape_proxy_bytes(
    *,
    layers: int,
    kv_lora_rank: int,
    rope_key_dim: int,
    bytes_per_element: int,
    context_tokens: int,
) -> int:
    """Hypothetical one-latent-plus-one-RoPE-key-per-layer cache layout."""
    return layers * (kv_lora_rank + rope_key_dim) * bytes_per_element * context_tokens


def main() -> None:
    valid, effective, errors = audit_request({"tool_choice": "auto"})
    assert valid and not errors
    assert effective["max_tokens"] == 32768

    invalid, _, invalid_errors = audit_request(
        {"thinking": {"type": "disabled"}, "temperature": 0.2, "tool_choice": "required"}
    )
    assert not invalid and len(invalid_errors) == 3
    malformed, _, malformed_errors = audit_request({"tool_choice": ["auto"]})
    assert not malformed and malformed_errors

    assistant_call = {
        "reasoning_content": "opaque teaching placeholder",
        "tool_calls": [{"id": "call-1", "name": "inspect_file"}],
    }
    tool_result = [{"tool_call_id": "call-1", "content": "synthetic result"}]
    cycle_ok, cycle_warnings = audit_tool_round_trip(
        assistant_call, tool_result, {"reasoning_content": "next opaque state"}
    )
    assert cycle_ok and not cycle_warnings
    missing_state_ok, missing_state_warnings = audit_tool_round_trip(
        assistant_call, tool_result, {"content": "next turn"}
    )
    assert missing_state_ok and missing_state_warnings

    total_parameters = 1_000_000_000_000
    raw_int4 = packed_int4_raw_bytes(total_parameters)
    cache_proxy = mla_cache_shape_proxy_bytes(
        layers=61,
        kv_lora_rank=512,
        rope_key_dim=64,
        bytes_per_element=2,
        context_tokens=262_144,
    )
    report = {
        "ok": True,
        "evidence_level": "local_protocol_toy",
        "network_called": False,
        "documented_defaults": effective,
        "invalid_request_rejected": len(invalid_errors),
        "tool_round_trip_correlated": cycle_ok,
        "missing_reasoning_is_warning_not_schema_failure": missing_state_warnings,
        "hypothetical_1T_all_int4_raw_GiB": round(raw_int4 / 1024**3, 3),
        "hypothetical_MLA_cache_proxy_at_256K_GiB": round(cache_proxy / 1024**3, 3),
        "memory_estimates_are_production_measurements": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
