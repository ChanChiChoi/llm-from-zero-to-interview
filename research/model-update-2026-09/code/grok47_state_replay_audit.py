#!/usr/bin/env python3
"""Audit a teaching replay contract for Grok 4.7 style opaque state.

xAI documents encrypted reasoning/tool state and opaque compaction items that
must be replayed without editing.  This standard-library toy checks the host
side invariants only.  It does not call xAI, decode ciphertext, or reproduce
provider internals.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping


class ReplayError(ValueError):
    """A saved response trace cannot be replayed safely."""


@dataclass(frozen=True)
class ToolReceipt:
    call_id: str
    idempotency_key: str
    result: Mapping[str, Any]
    duplicate: bool = False


class SideEffectLedger:
    """Make duplicate tool replay observable and harmless."""

    def __init__(self) -> None:
        self.side_effect_count = 0
        self.receipts: dict[str, ToolReceipt] = {}

    def execute(self, call: Mapping[str, Any]) -> ToolReceipt:
        key = call.get("idempotency_key")
        if not isinstance(key, str) or not key:
            raise ReplayError("tool call is missing idempotency_key")
        prior = self.receipts.get(key)
        if prior is not None:
            return ToolReceipt(
                call_id=prior.call_id,
                idempotency_key=prior.idempotency_key,
                result=prior.result,
                duplicate=True,
            )
        call_id = call.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ReplayError("tool call is missing call_id")
        self.side_effect_count += 1
        receipt = ToolReceipt(
            call_id=call_id,
            idempotency_key=key,
            result={"artifact": "workspace/report.md", "verified": True},
        )
        self.receipts[key] = receipt
        return receipt


def opaque(value: str) -> str:
    digest = hashlib.sha256(value.encode("ascii")).hexdigest()[:24]
    return f"opaque:{digest}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def validate_trace(trace: Mapping[str, Any]) -> None:
    if trace.get("model") != "grok-4.7":
        raise ReplayError("trace model identity is not pinned")
    if trace.get("transport") != "responses":
        raise ReplayError("opaque reasoning contract is Responses-specific in this toy")
    if trace.get("stored") is False and trace.get("previous_response_id"):
        raise ReplayError("store=false cannot rely on previous_response_id")

    items = trace.get("items")
    if not isinstance(items, list) or not items:
        raise ReplayError("trace items are missing")
    encrypted_positions: list[int] = []
    calls: dict[str, Mapping[str, Any]] = {}
    outputs: dict[str, Mapping[str, Any]] = {}
    compaction_positions: list[int] = []

    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            raise ReplayError(f"item {index} is malformed")
        item_type = item["type"]
        if "encrypted_content" in item:
            encrypted = item["encrypted_content"]
            if not isinstance(encrypted, str) or not encrypted.startswith("opaque:"):
                raise ReplayError(f"item {index} has edited opaque state")
            encrypted_positions.append(index)
        if item_type == "function_call":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or call_id in calls:
                raise ReplayError("function call id is missing or duplicated")
            if not isinstance(item.get("idempotency_key"), str):
                raise ReplayError("function call idempotency key is missing")
            calls[call_id] = item
        elif item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or call_id not in calls:
                raise ReplayError("function output has no matching call")
            if "encrypted_content" not in item:
                raise ReplayError("function output is missing opaque tool state")
            outputs[call_id] = item
        elif item_type == "compaction":
            compaction_positions.append(index)

    if not encrypted_positions:
        raise ReplayError("trace has no opaque reasoning/tool state")
    if len(compaction_positions) != 1:
        raise ReplayError("trace must contain exactly one compaction item")
    for call_id, call in calls.items():
        output = outputs.get(call_id)
        if output is None:
            raise ReplayError(f"tool call has no output: {call_id}")
        if output.get("idempotency_key") != call.get("idempotency_key"):
            raise ReplayError(f"tool lineage changed: {call_id}")


def replay(trace: Mapping[str, Any], ledger: SideEffectLedger) -> list[ToolReceipt]:
    validate_trace(trace)
    receipts: list[ToolReceipt] = []
    for item in trace["items"]:
        if item["type"] == "function_call":
            receipts.append(ledger.execute(item))
    return receipts


def sample_trace() -> dict[str, Any]:
    call_id = "call-write-1"
    key = "write:workspace/report.md:v1"
    return {
        "model": "grok-4.7",
        "transport": "responses",
        "stored": True,
        "response_id": "resp-001",
        "previous_response_id": None,
        "items": [
            {"type": "message", "role": "user", "content": "prepare a report"},
            {
                "type": "reasoning",
                "summary": "plan, call the tool, then verify the artifact",
                "encrypted_content": opaque("reasoning-001"),
            },
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "write_artifact",
                "arguments": {"path": "workspace/report.md"},
                "idempotency_key": key,
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "idempotency_key": key,
                "encrypted_content": opaque("tool-result-001"),
                "output": {"status": "committed"},
            },
            {
                "type": "compaction",
                "encrypted_content": opaque("compaction-001"),
                "cut_after": call_id,
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "The artifact was written and independently verified.",
            },
        ],
    }


def expect_error(action: Any, fragment: str) -> None:
    try:
        action()
    except ReplayError as exc:
        if fragment not in str(exc):
            raise AssertionError(f"unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"expected ReplayError containing {fragment!r}")


def main() -> None:
    trace = sample_trace()
    assert canonical_json(trace) == canonical_json(json.loads(canonical_json(trace)))

    ledger = SideEffectLedger()
    first = replay(trace, ledger)
    second = replay(deepcopy(trace), ledger)
    assert len(first) == len(second) == 1
    assert not first[0].duplicate and second[0].duplicate
    assert ledger.side_effect_count == 1

    edited = deepcopy(trace)
    edited["items"][1]["encrypted_content"] = edited["items"][1]["summary"]
    expect_error(lambda: validate_trace(edited), "edited opaque state")

    missing_tool_state = deepcopy(trace)
    del missing_tool_state["items"][3]["encrypted_content"]
    expect_error(lambda: validate_trace(missing_tool_state), "missing opaque tool state")

    duplicated_compaction = deepcopy(trace)
    duplicated_compaction["items"].append(deepcopy(trace["items"][4]))
    expect_error(lambda: validate_trace(duplicated_compaction), "exactly one compaction")

    invalid_server_state = deepcopy(trace)
    invalid_server_state["stored"] = False
    invalid_server_state["previous_response_id"] = "resp-000"
    expect_error(lambda: validate_trace(invalid_server_state), "store=false")

    print(
        json.dumps(
            {
                "evidence_level": "local_protocol_toy",
                "official_boundary": (
                    "xAI opaque state is not decoded; this does not prove provider internals"
                ),
                "replay": {
                    "first_side_effects": len(first),
                    "duplicate_receipt": second[0].duplicate,
                    "side_effect_count": ledger.side_effect_count,
                },
                "negative_cases": [
                    "edited encrypted content",
                    "missing tool encrypted state",
                    "duplicate compaction item",
                    "store=false with previous_response_id",
                ],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
