#!/usr/bin/env python3
"""Audit a teaching replay contract for Claude Opus 5.5.

Anthropic's public contract makes several migration failures observable:
adaptive thinking cannot be disabled, forced tool choice is rejected, thinking
blocks are bound to a model and conversation prefix, and old computer tools
are platform-dependent. This standard-library toy checks host-side state,
fallback, and idempotency invariants. It does not call Anthropic or reproduce
provider internals.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping


class ProtocolError(ValueError):
    """A request or response cannot be handled under the public contract."""


@dataclass(frozen=True)
class ToolReceipt:
    call_id: str
    idempotency_key: str
    artifact: str
    verified: bool
    duplicate: bool = False


@dataclass(frozen=True)
class FallbackReceipt:
    original_model: str
    actual_model: str | None
    category: str
    outcome: str
    retry_allowed: bool


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def opaque(label: str) -> str:
    return f"opaque:{hashlib.sha256(label.encode('ascii')).hexdigest()[:24]}"


def prefix_hash(system: str, tools: list[Mapping[str, Any]]) -> str:
    return digest({"system": system, "tools": tools})


def validate_request(request: Mapping[str, Any]) -> None:
    if request.get("model") != "claude-opus-5-5":
        raise ProtocolError("model identity is not pinned")
    effort = request.get("output_config", {}).get("effort")
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ProtocolError("effort must be explicit and supported")

    thinking = request.get("thinking", {})
    thinking_type = thinking.get("type")
    if thinking_type == "disabled":
        raise ProtocolError("thinking disabled is not supported")
    if thinking_type == "enabled" or "budget_tokens" in thinking:
        raise ProtocolError("manual thinking budget is not supported")
    if thinking_type not in {None, "adaptive"}:
        raise ProtocolError("unknown thinking type")

    tool_choice = request.get("tool_choice", {"type": "auto"})
    if tool_choice.get("type") in {"any", "tool"}:
        raise ProtocolError("forced tool choice is not supported")

    platform = request.get("platform")
    computer_tools = [
        tool for tool in request.get("tools", [])
        if tool.get("type") == "computer_20251124"
    ]
    if computer_tools and platform in {"claude_api", "google_cloud"}:
        raise ProtocolError("old computer tool is not supported on this platform")


READABLE_PRODUCERS = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-haiku-4-5",
}


def validate_response(response: Mapping[str, Any]) -> None:
    if response.get("model") != "claude-opus-5-5":
        raise ProtocolError("response model identity is not pinned")
    blocks = response.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ProtocolError("response blocks are missing")
    expected_prefix = response.get("prefix_hash")
    if not isinstance(expected_prefix, str):
        raise ProtocolError("conversation prefix hash is missing")

    calls: dict[str, Mapping[str, Any]] = {}
    results: dict[str, Mapping[str, Any]] = {}
    compactions = 0
    saw_tool_use = False
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or not isinstance(block.get("type"), str):
            raise ProtocolError(f"block {index} is malformed")
        block_type = block["type"]
        if block_type == "thinking":
            signature = block.get("signature")
            producer = block.get("producer_model")
            if not isinstance(signature, str) or not signature.startswith("opaque:"):
                raise ProtocolError(f"thinking block {index} has no opaque signature")
            if producer not in READABLE_PRODUCERS:
                raise ProtocolError(f"thinking block producer is not readable: {producer}")
            if block.get("prefix_hash") != expected_prefix:
                raise ProtocolError("thinking block prefix mismatch")
            display = response.get("display", "omitted")
            if block.get("progress") and display == "omitted" and block.get("text", ""):
                raise ProtocolError("omitted progress block contains visible text")
            if block.get("progress") and display == "updates" and not block.get("text"):
                raise ProtocolError("updates display lost progress text")
        elif block_type == "tool_use":
            call_id = block.get("id")
            key = block.get("idempotency_key")
            if not isinstance(call_id, str) or call_id in calls:
                raise ProtocolError("tool use id is missing or duplicated")
            if not isinstance(key, str) or not key:
                raise ProtocolError("tool use idempotency key is missing")
            calls[call_id] = block
            saw_tool_use = True
        elif block_type == "tool_result":
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str) or call_id not in calls:
                raise ProtocolError("tool result has no matching tool use")
            if call_id in results:
                raise ProtocolError("tool result is duplicated")
            if block.get("idempotency_key") != calls[call_id].get("idempotency_key"):
                raise ProtocolError("tool result idempotency lineage changed")
            results[call_id] = block
        elif block_type == "compaction":
            compactions += 1
            if compactions > 1:
                raise ProtocolError("response contains duplicate compaction blocks")
            if not str(block.get("signature", "")).startswith("opaque:"):
                raise ProtocolError("compaction signature is missing")
            replaced = block.get("replaces")
            if not isinstance(replaced, list) or not replaced:
                raise ProtocolError("compaction replacement set is missing")
        elif block_type == "text" and saw_tool_use and not results:
            raise ProtocolError("between-tool progress must be a thinking block")

    for call_id in calls:
        if call_id not in results:
            raise ProtocolError(f"tool use has no result: {call_id}")


class SideEffectLedger:
    """Make a replayed tool call harmless when its key is unchanged."""

    def __init__(self) -> None:
        self.executions = 0
        self.receipts: dict[str, ToolReceipt] = {}

    def execute(self, call: Mapping[str, Any]) -> ToolReceipt:
        key = call["idempotency_key"]
        previous = self.receipts.get(key)
        if previous is not None:
            return ToolReceipt(
                previous.call_id,
                previous.idempotency_key,
                previous.artifact,
                previous.verified,
                duplicate=True,
            )
        self.executions += 1
        receipt = ToolReceipt(
            call_id=call["id"],
            idempotency_key=key,
            artifact="workspace/patch.diff",
            verified=True,
        )
        self.receipts[key] = receipt
        return receipt


def replay(response: Mapping[str, Any], ledger: SideEffectLedger) -> list[ToolReceipt]:
    validate_response(response)
    return [
        ledger.execute(block)
        for block in response["blocks"]
        if block["type"] == "tool_use"
    ]


def route_refusal(
    *, category: str, original_model: str = "claude-opus-5-5"
) -> FallbackReceipt:
    targets = {
        "cyber": "claude-opus-4-8",
        "biology": "claude-opus-5",
        "frontier_llm_kernel": "claude-opus-5",
        "reasoning_extraction": None,
    }
    if category not in targets:
        raise ProtocolError(f"unknown refusal category: {category}")
    target = targets[category]
    if target is None:
        return FallbackReceipt(original_model, None, category, "refused", False)
    return FallbackReceipt(original_model, target, category, "fallback", True)


def sample_request_and_response() -> tuple[dict[str, Any], dict[str, Any]]:
    tools = [{"name": "write_patch", "schema_version": "v1"}]
    prefix = prefix_hash("You are an audited coding agent.", tools)
    request = {
        "model": "claude-opus-5-5",
        "platform": "claude_api",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
        "tool_choice": {"type": "auto"},
        "tools": tools,
    }
    response = {
        "model": "claude-opus-5-5",
        "prefix_hash": prefix,
        "display": "updates",
        "blocks": [
            {
                "type": "thinking",
                "producer_model": "claude-opus-5",
                "prefix_hash": prefix,
                "signature": opaque("thinking-1"),
                "text": "inspect the repository before editing",
            },
            {
                "type": "tool_use",
                "id": "tool-1",
                "name": "write_patch",
                "idempotency_key": "patch:workspace:report:v1",
                "input": {"path": "workspace/report.md"},
            },
            {
                "type": "tool_result",
                "tool_use_id": "tool-1",
                "idempotency_key": "patch:workspace:report:v1",
                "status": "committed",
            },
            {
                "type": "thinking",
                "producer_model": "claude-opus-5-5",
                "prefix_hash": prefix,
                "signature": opaque("progress-1"),
                "progress": True,
                "text": "run the verifier",
            },
            {
                "type": "compaction",
                "signature": opaque("compact-1"),
                "replaces": ["tool-1", "tool-result-1"],
            },
            {"type": "text", "text": "The verified patch is ready."},
        ],
    }
    return request, response


def expect_error(action: Any, fragment: str) -> None:
    try:
        action()
    except ProtocolError as exc:
        if fragment not in str(exc):
            raise AssertionError(f"unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"expected ProtocolError containing {fragment!r}")


def main() -> None:
    request, response = sample_request_and_response()
    validate_request(request)
    validate_response(response)

    ledger = SideEffectLedger()
    first = replay(response, ledger)
    second = replay(deepcopy(response), ledger)
    assert len(first) == len(second) == 1
    assert not first[0].duplicate and second[0].duplicate
    assert ledger.executions == 1

    disabled = deepcopy(request)
    disabled["thinking"] = {"type": "disabled"}
    expect_error(lambda: validate_request(disabled), "thinking disabled")

    manual = deepcopy(request)
    manual["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    expect_error(lambda: validate_request(manual), "manual thinking budget")

    forced = deepcopy(request)
    forced["tool_choice"] = {"type": "any"}
    expect_error(lambda: validate_request(forced), "forced tool choice")

    old_computer = deepcopy(request)
    old_computer["tools"] = [{"type": "computer_20251124"}]
    expect_error(lambda: validate_request(old_computer), "old computer tool")

    changed_prefix = deepcopy(response)
    changed_prefix["blocks"][0]["prefix_hash"] = "prefix:changed"
    expect_error(lambda: validate_response(changed_prefix), "prefix mismatch")

    unsupported_producer = deepcopy(response)
    unsupported_producer["blocks"][0]["producer_model"] = "claude-fable-5-1"
    expect_error(lambda: validate_response(unsupported_producer), "not readable")

    duplicated_compaction = deepcopy(response)
    duplicated_compaction["blocks"].append(deepcopy(response["blocks"][4]))
    expect_error(lambda: validate_response(duplicated_compaction), "duplicate compaction")

    misplaced_progress = deepcopy(response)
    misplaced_progress["blocks"].insert(
        2, {"type": "text", "text": "I am still working."}
    )
    expect_error(
        lambda: validate_response(misplaced_progress),
        "between-tool progress",
    )

    cyber = route_refusal(category="cyber")
    extraction = route_refusal(category="reasoning_extraction")
    assert cyber.actual_model == "claude-opus-4-8" and cyber.retry_allowed
    assert extraction.actual_model is None and not extraction.retry_allowed

    print(
        json.dumps(
            {
                "evidence_level": "local_protocol_toy",
                "official_boundary": (
                    "Anthropic request and block contracts are modeled; no API call, "
                    "provider internals, or model quality is reproduced."
                ),
                "replay": {
                    "first_side_effects": len(first),
                    "duplicate_receipt": second[0].duplicate,
                    "side_effect_executions": ledger.executions,
                },
                "fallback": {
                    "cyber": cyber.__dict__,
                    "reasoning_extraction": extraction.__dict__,
                },
                "negative_cases": [
                    "disabled thinking",
                    "manual thinking budget",
                    "forced tool choice",
                    "platform-specific old computer tool",
                    "thinking prefix mismatch",
                    "unsupported thinking producer",
                    "duplicate compaction",
                    "text progress between tool blocks",
                ],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
