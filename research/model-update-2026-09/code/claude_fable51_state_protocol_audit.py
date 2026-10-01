#!/usr/bin/env python3
"""Audit a teaching state contract for Claude Fable 5.1.

Anthropic's public migration material makes several host-visible rules
explicit: thinking is adaptive and always on, forced tool choice is rejected,
thinking blocks are model/prefix bound, progress updates are not tool results,
and provenance is not a correctness proof.  This standard-library toy checks
those host-side invariants and idempotent replay.  It does not call Anthropic
or reproduce provider internals.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping


class ProtocolError(ValueError):
    """A request or response cannot be handled under the public contract."""


class VerificationError(ProtocolError):
    """A provenance or artifact claim did not pass an independent check."""


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


FABLE_MODEL = "claude-fable-5-1"
EFFORTS = {"low", "medium", "high", "xhigh", "max"}

# The public rule is directional. Fable 5.1 can consume earlier Claude
# thinking blocks and, on the Claude API only, Opus 5.5 blocks. Neither Opus 5
# nor Opus 5.5 can consume Fable 5.1 blocks.
FABLE_READABLE_PRODUCERS = {
    FABLE_MODEL,
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-haiku-4-5",
}
FABLE_CLAUDE_API_ONLY_PRODUCERS = {"claude-opus-5-5"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def opaque(label: str) -> str:
    return f"opaque:{hashlib.sha256(label.encode('ascii')).hexdigest()[:24]}"


def prefix_hash(
    system: str,
    tools: list[Mapping[str, Any]],
    earlier_messages: list[Mapping[str, Any]] | None = None,
) -> str:
    return digest(
        {
            "system": system,
            "tools": tools,
            "earlier_messages": earlier_messages or [],
        }
    )


def validate_request(request: Mapping[str, Any]) -> None:
    if request.get("model") != FABLE_MODEL:
        raise ProtocolError("model identity is not pinned")

    effort = request.get("output_config", {}).get("effort")
    if effort not in EFFORTS:
        raise ProtocolError("effort must be explicit and supported")

    thinking = request.get("thinking")
    if thinking is not None and thinking.get("type") not in {"adaptive"}:
        raise ProtocolError("Fable 5.1 only supports adaptive always-on thinking")
    if thinking and "budget_tokens" in thinking:
        raise ProtocolError("manual thinking budget is not supported")

    tool_choice = request.get("tool_choice", {"type": "auto"})
    if tool_choice.get("type") in {"any", "tool"}:
        raise ProtocolError("forced tool choice is not supported")

    if request.get("prefill"):
        raise ProtocolError("assistant prefill is not supported")


def fable_block_compatibility(producer_model: str, surface: str) -> bool | None:
    if producer_model in FABLE_READABLE_PRODUCERS:
        return True
    if producer_model in FABLE_CLAUDE_API_ONLY_PRODUCERS:
        return True if surface == "claude_api" else None
    return False


def model_binding_compatibility(
    producer_model: str, target_model: str, surface: str
) -> bool | None:
    if target_model == FABLE_MODEL:
        return fable_block_compatibility(producer_model, surface)
    if producer_model == FABLE_MODEL and target_model in {
        "claude-opus-5",
        "claude-opus-5-5",
    }:
        return False
    return None


def validate_model_switch(
    source_model: str, target_model: str, surface: str = "claude_api"
) -> None:
    compatibility = model_binding_compatibility(
        source_model, target_model, surface
    )
    if compatibility is True:
        return
    if compatibility is False:
        raise ProtocolError("target model cannot read Fable 5.1 thinking blocks")
    raise ProtocolError("model pair is outside the documented compatibility matrix")


def validate_response(response: Mapping[str, Any]) -> None:
    if response.get("model") != FABLE_MODEL:
        raise ProtocolError("response model identity is not pinned")

    blocks = response.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ProtocolError("response blocks are missing")
    expected_prefix = response.get("prefix_hash")
    if not isinstance(expected_prefix, str):
        raise ProtocolError("conversation prefix hash is missing")

    calls: dict[str, Mapping[str, Any]] = {}
    results: dict[str, Mapping[str, Any]] = {}
    progress_count = 0
    provenance_count = 0

    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or not isinstance(block.get("type"), str):
            raise ProtocolError(f"block {index} is malformed")
        block_type = block["type"]

        if block_type == "thinking":
            signature = block.get("signature")
            producer = block.get("producer_model")
            if not isinstance(signature, str) or not signature.startswith("opaque:"):
                raise ProtocolError(f"thinking block {index} has no opaque signature")
            surface = response.get("surface", "claude_api")
            if (
                not isinstance(producer, str)
                or fable_block_compatibility(producer, surface) is not True
            ):
                raise ProtocolError(
                    f"thinking block producer is not confirmed readable: {producer}"
                )
            if block.get("prefix_hash") != expected_prefix:
                raise ProtocolError("thinking block prefix mismatch")

        elif block_type == "tool_use":
            call_id = block.get("id")
            key = block.get("idempotency_key")
            if not isinstance(call_id, str) or not call_id or call_id in calls:
                raise ProtocolError("tool use id is missing or duplicated")
            if not isinstance(key, str) or not key:
                raise ProtocolError("tool use idempotency key is missing")
            calls[call_id] = block

        elif block_type == "progress_update":
            if response.get("display") != "updates":
                raise ProtocolError("progress update requires display=updates")
            if not isinstance(block.get("text"), str) or not block["text"]:
                raise ProtocolError("progress update has no visible text")
            progress_count += 1

        elif block_type == "tool_result":
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str) or call_id not in calls:
                raise ProtocolError("tool result has no matching tool use")
            if call_id in results:
                raise ProtocolError("tool result is duplicated")
            if block.get("idempotency_key") != calls[call_id].get("idempotency_key"):
                raise ProtocolError("tool result idempotency lineage changed")
            results[call_id] = block

        elif block_type == "provenance":
            source_ids = block.get("source_ids")
            if not isinstance(source_ids, list) or not source_ids:
                raise ProtocolError("provenance has no source ids")
            if not isinstance(block.get("claim"), str) or not block["claim"]:
                raise ProtocolError("provenance has no claim")
            provenance_count += 1

        elif block_type == "text":
            continue
        else:
            raise ProtocolError(f"unknown block type: {block_type}")

    for call_id in calls:
        if call_id not in results:
            # A visible progress update cannot stand in for execution or
            # verification of the corresponding tool call.
            raise ProtocolError(f"tool use has no result: {call_id}")
    if progress_count and not calls:
        raise ProtocolError("progress update has no associated tool work")
    if provenance_count and not response.get("artifact_verifier"):
        raise VerificationError("provenance has no independent artifact verifier")


def verify_provenance(response: Mapping[str, Any]) -> None:
    for block in response.get("blocks", []):
        if block.get("type") != "provenance":
            continue
        if block.get("supported") is not True:
            raise VerificationError("provenance source does not support the claim")
    if response.get("artifact_verifier") != "verified":
        raise VerificationError("artifact verifier did not pass")


def apply_prefix_edit(
    response: Mapping[str, Any],
    new_prefix: str,
    mismatch_behavior: str = "error",
) -> dict[str, Any]:
    edited = deepcopy(response)
    old_prefix = edited.get("prefix_hash")
    if old_prefix == new_prefix:
        return edited
    if mismatch_behavior == "error":
        raise ProtocolError("editing earlier turns invalidates thinking blocks")
    if mismatch_behavior != "drop_block":
        raise ProtocolError("unknown prefix mismatch behavior")

    edited["prefix_hash"] = new_prefix
    edited["blocks"] = [
        block for block in edited["blocks"] if block.get("type") != "thinking"
    ]
    edited["input_transformations"] = [
        {
            "type": "thinking_block_removed",
            "reason": "prefix_binding_mismatch",
        }
    ]
    return edited


def apply_model_binding(
    response: Mapping[str, Any], target_model: str, surface: str
) -> dict[str, Any]:
    """Model the documented unbilled drop, not an API call or provider schema."""
    switched = deepcopy(response)
    kept: list[Mapping[str, Any]] = []
    transformations: list[dict[str, str]] = []
    for block in switched.get("blocks", []):
        if block.get("type") != "thinking":
            kept.append(block)
            continue
        compatibility = model_binding_compatibility(
            block.get("producer_model", ""), target_model, surface
        )
        if compatibility is True:
            kept.append(block)
        elif compatibility is False:
            transformations.append(
                {
                    "type": "thinking_block_removed",
                    "reason": "model_binding_mismatch",
                }
            )
        else:
            raise ProtocolError("model pair is outside the documented compatibility matrix")
    switched["model"] = target_model
    switched["surface"] = surface
    switched["blocks"] = kept
    switched["input_transformations"] = transformations
    return switched


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
            artifact="workspace/report.md",
            verified=True,
        )
        self.receipts[key] = receipt
        return receipt


def replay(response: Mapping[str, Any], ledger: SideEffectLedger) -> list[ToolReceipt]:
    validate_response(response)
    verify_provenance(response)
    return [
        ledger.execute(block)
        for block in response["blocks"]
        if block["type"] == "tool_use"
    ]


def route_refusal(
    *, category: str, original_model: str = FABLE_MODEL
) -> FallbackReceipt:
    targets = {
        "cyber": "claude-opus-4-8",
        "biology": "claude-opus-5",
        "unsupported": None,
    }
    if category not in targets:
        raise ProtocolError(f"unknown refusal category: {category}")
    target = targets[category]
    if target is None:
        return FallbackReceipt(original_model, None, category, "refused", False)
    return FallbackReceipt(original_model, target, category, "fallback", True)


def sample_request_and_response() -> tuple[dict[str, Any], dict[str, Any]]:
    tools = [{"name": "write_patch", "schema_version": "v1"}]
    prefix = prefix_hash(
        "You are an audited coding agent.",
        tools,
        [{"role": "user", "content": "prepare a report"}],
    )
    request = {
        "model": FABLE_MODEL,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "tool_choice": {"type": "auto"},
        "tools": tools,
    }
    response = {
        "model": FABLE_MODEL,
        "surface": "claude_api",
        "prefix_hash": prefix,
        "display": "updates",
        "artifact_verifier": "verified",
        "blocks": [
            {
                "type": "thinking",
                "producer_model": FABLE_MODEL,
                "prefix_hash": prefix,
                "signature": opaque("thinking-1"),
            },
            {
                "type": "tool_use",
                "id": "tool-1",
                "name": "write_patch",
                "idempotency_key": "patch:workspace:report:v1",
                "input": {"path": "workspace/report.md"},
            },
            {
                "type": "progress_update",
                "text": "The patch is being checked.",
            },
            {
                "type": "tool_result",
                "tool_use_id": "tool-1",
                "idempotency_key": "patch:workspace:report:v1",
                "status": "committed",
            },
            {
                "type": "provenance",
                "source_ids": ["repo-spec-v1"],
                "claim": "artifact follows the requested schema",
                "supported": True,
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
    verify_provenance(response)

    ledger = SideEffectLedger()
    first = replay(response, ledger)
    second = replay(deepcopy(response), ledger)
    assert len(first) == len(second) == 1
    assert not first[0].duplicate and second[0].duplicate
    assert ledger.executions == 1

    disabled = deepcopy(request)
    disabled["thinking"] = {"type": "disabled"}
    expect_error(lambda: validate_request(disabled), "always-on thinking")

    manual = deepcopy(request)
    manual["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    expect_error(lambda: validate_request(manual), "always-on thinking")

    forced = deepcopy(request)
    forced["tool_choice"] = {"type": "any"}
    expect_error(lambda: validate_request(forced), "forced tool choice")

    prefill = deepcopy(request)
    prefill["prefill"] = "The answer is"
    expect_error(lambda: validate_request(prefill), "assistant prefill")

    older_reader = lambda: validate_model_switch(FABLE_MODEL, "claude-opus-5")
    expect_error(older_reader, "cannot read Fable 5.1")
    validate_model_switch("claude-opus-5-5", FABLE_MODEL, "claude_api")
    expect_error(
        lambda: validate_model_switch("claude-opus-5-5", FABLE_MODEL, "google_cloud"),
        "compatibility matrix",
    )
    expect_error(
        lambda: validate_model_switch(FABLE_MODEL, "claude-opus-5-5"),
        "cannot read Fable 5.1",
    )

    opus55_input = deepcopy(response)
    opus55_input["blocks"][0]["producer_model"] = "claude-opus-5-5"
    validate_response(opus55_input)
    opus55_other_surface = deepcopy(opus55_input)
    opus55_other_surface["surface"] = "google_cloud"
    expect_error(
        lambda: validate_response(opus55_other_surface),
        "not confirmed readable",
    )

    changed_prefix = deepcopy(response)
    changed_prefix["prefix_hash"] = "prefix:changed"
    expect_error(lambda: validate_response(changed_prefix), "prefix mismatch")

    dropped = apply_prefix_edit(response, "prefix:changed", "drop_block")
    validate_response(dropped)
    assert dropped["input_transformations"][0]["reason"] == "prefix_binding_mismatch"
    assert not any(block["type"] == "thinking" for block in dropped["blocks"])

    model_dropped = apply_model_binding(response, "claude-opus-5-5", "claude_api")
    assert model_dropped["input_transformations"][0]["reason"] == "model_binding_mismatch"
    assert not any(block["type"] == "thinking" for block in model_dropped["blocks"])

    missing_result = deepcopy(response)
    missing_result["blocks"] = [
        block
        for block in missing_result["blocks"]
        if block["type"] != "tool_result"
    ]
    expect_error(lambda: validate_response(missing_result), "no result")

    unsupported_claim = deepcopy(response)
    unsupported_claim["blocks"][4]["supported"] = False
    expect_error(lambda: replay(unsupported_claim, SideEffectLedger()), "does not support")

    cyber = route_refusal(category="cyber")
    unsupported = route_refusal(category="unsupported")
    assert cyber.actual_model == "claude-opus-4-8" and cyber.retry_allowed
    assert unsupported.actual_model is None and not unsupported.retry_allowed

    print(
        json.dumps(
            {
                "evidence_level": "local_protocol_toy",
                "official_boundary": (
                    "Anthropic public request/state rules are modeled; no API call, "
                    "provider internals, hidden reasoning, or model quality is reproduced."
                ),
                "replay": {
                    "first_side_effects": len(first),
                    "duplicate_receipt": second[0].duplicate,
                    "side_effect_executions": ledger.executions,
                },
                "prefix_edit": {
                    "drop_block": True,
                    "input_transformation": dropped["input_transformations"][0],
                },
                "model_switch": {
                    "dropped_block_count": len(model_dropped["input_transformations"]),
                    "input_transformation": model_dropped["input_transformations"][0],
                    "surface": model_dropped["surface"],
                },
                "thinking_compatibility": {
                    "fable51_reads_opus55_on_claude_api": True,
                    "fable51_reads_opus55_on_other_surfaces": "unverified",
                    "opus55_reads_fable51": False,
                },
                "fallback": {
                    "cyber": cyber.__dict__,
                    "unsupported": unsupported.__dict__,
                },
                "negative_cases": [
                    "disabled or manual thinking",
                    "forced tool choice",
                    "assistant prefill",
                    "Opus 5.5 reading Fable 5.1 thinking",
                    "Opus 5.5 block outside Claude API",
                    "thinking prefix mismatch",
                    "progress without tool result",
                    "unsupported provenance claim",
                ],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
