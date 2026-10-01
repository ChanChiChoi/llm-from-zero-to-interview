"""Dependency-free audit of GPT-6 Sol's public runtime contracts.

This local protocol toy does not call an OpenAI endpoint, expose hidden
reasoning, execute tools, or load weights.  Its evidence level is
``local_protocol_toy``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping


class ProtocolError(ValueError):
    """Raised when a toy transition violates a documented contract."""


class VerificationError(ProtocolError):
    """Raised when an independent artifact verifier rejects a receipt."""


MODEL = "gpt-6-sol"
FAMILY_MODELS = {"gpt-6-astra", "gpt-6-sol", "gpt-6-luna"}
MODES = {"standard", "pro"}
EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
CONTEXT_WINDOW = 1_050_000
MAX_INPUT = 922_000
MAX_OUTPUT = 128_000
REASONING_RESERVE = 25_000
WHOLE_REQUEST_THRESHOLD = 272_000
MIN_CACHEABLE_TOKENS = 1_024
MAX_EXPLICIT_BREAKPOINTS = 4
PRICES_PER_MILLION = {
    "input": 2.0,
    "cached_input": 0.20,
    "cache_write": 2.50,
    "output": 10.0,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:24]


def opaque(label: str) -> str:
    return "encrypted:" + hashlib.sha256(label.encode("utf-8")).hexdigest()[:24]


def expect_error(action: Callable[[], Any], text: str) -> None:
    try:
        action()
    except ProtocolError as exc:
        assert text in str(exc), (text, str(exc))
    else:
        raise AssertionError(f"expected ProtocolError containing {text!r}")


@dataclass(frozen=True)
class ReasoningConfig:
    mode: str = "standard"
    effort: str = "medium"
    endpoint: str = "responses"
    single_agent: bool = True

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ProtocolError("reasoning.mode must be standard or pro")
        if self.effort not in EFFORTS:
            raise ProtocolError("reasoning.effort is unsupported")
        if self.mode == "pro" and self.endpoint != "responses":
            raise ProtocolError("pro reasoning mode requires the Responses API")


def _validate_update_item(item: Mapping[str, Any]) -> None:
    if set(item) - {"type", "reasoning"}:
        raise ProtocolError("configuration_update may change only reasoning.effort")
    reasoning = item.get("reasoning")
    if not isinstance(reasoning, Mapping) or set(reasoning) != {"effort"}:
        raise ProtocolError("configuration_update must contain only reasoning.effort")
    if reasoning["effort"] not in EFFORTS:
        raise ProtocolError("configuration_update effort is unsupported")


def validate_configuration_updates(
    history: list[Mapping[str, Any]],
    config: ReasoningConfig,
    *,
    automatic_compaction: bool = False,
    automatic_truncation: bool = False,
    standalone_compact: bool = False,
) -> None:
    """Validate placement and incompatibilities of session effort updates."""

    config.validate()
    updates = [
        index for index, item in enumerate(history)
        if item.get("type") == "configuration_update"
    ]
    if not updates:
        return
    if config.mode != "standard" or not config.single_agent:
        raise ProtocolError("configuration_update requires standard single-agent mode")
    if automatic_compaction or automatic_truncation:
        raise ProtocolError("configuration_update conflicts with automatic compaction or truncation")
    if standalone_compact:
        raise ProtocolError("standalone compaction rejects configuration_update history")
    if any(right == left + 1 for left, right in zip(updates, updates[1:])):
        raise ProtocolError("adjacent configuration_update items are rejected")

    for index in updates:
        item = history[index]
        _validate_update_item(item)
        if index + 1 >= len(history):
            raise ProtocolError("configuration_update must precede the next user message")
        next_item = history[index + 1]
        if next_item.get("type") != "message" or next_item.get("role") != "user":
            raise ProtocolError("configuration_update must precede the next user message")


@dataclass(frozen=True)
class EffortResolution:
    request_level_effort: str
    effective_effort: str
    response_effort_field: str
    stable_prefix_digest: str


def resolve_effective_effort(
    request_level_effort: str,
    history: list[Mapping[str, Any]],
    stable_prompt_prefix: Any,
) -> EffortResolution:
    """Model the distinction between request config and session-level updates."""

    if request_level_effort not in EFFORTS:
        raise ProtocolError("request-level reasoning effort is unsupported")
    effective_effort = request_level_effort
    for item in history:
        if item.get("type") == "configuration_update":
            _validate_update_item(item)
            effective_effort = item["reasoning"]["effort"]
    return EffortResolution(
        request_level_effort=request_level_effort,
        effective_effort=effective_effort,
        # The official guide says this response field continues to report the
        # request-level value, not the value selected by configuration_update.
        response_effort_field=request_level_effort,
        stable_prefix_digest=digest(stable_prompt_prefix),
    )


def validate_post_compaction_update(history: list[Mapping[str, Any]]) -> None:
    """Require a fresh effort update before the first user message after compaction."""

    compaction_indexes = [
        index for index, item in enumerate(history) if item.get("type") == "compaction"
    ]
    if not compaction_indexes:
        return
    last_compaction = compaction_indexes[-1]
    for index in range(last_compaction + 1, len(history)):
        if history[index].get("type") == "message" and history[index].get("role") == "user":
            if index == last_compaction + 1 or history[index - 1].get("type") != "configuration_update":
                raise ProtocolError("explicit compaction requires a fresh configuration_update")
            return


@dataclass(frozen=True)
class BudgetResult:
    status: str
    reason: str | None
    total_output_tokens: int
    total_context_tokens: int


def evaluate_budget(
    input_tokens: int,
    reasoning_tokens: int,
    visible_output_tokens: int,
    max_output_tokens: int,
) -> BudgetResult:
    values = (input_tokens, reasoning_tokens, visible_output_tokens, max_output_tokens)
    if any(value < 0 for value in values):
        raise ProtocolError("token budgets cannot be negative")
    if input_tokens > MAX_INPUT:
        raise ProtocolError("input exceeds GPT-6 Sol maximum input")
    if max_output_tokens > MAX_OUTPUT:
        raise ProtocolError("max_output_tokens exceeds GPT-6 Sol maximum output")
    total_output = reasoning_tokens + visible_output_tokens
    total_context = input_tokens + total_output
    if total_context > CONTEXT_WINDOW:
        return BudgetResult("incomplete", "context_window", total_output, total_context)
    if total_output > max_output_tokens:
        return BudgetResult("incomplete", "max_output_tokens", total_output, total_context)
    return BudgetResult("completed", None, total_output, total_context)


def estimate_cost(
    input_tokens: int,
    cached_input_tokens: int,
    cache_write_tokens: int,
    output_tokens: int,
    *,
    processing: str = "standard",
    regional: bool = False,
) -> float:
    """Estimate token cost, including the whole-request 272K threshold."""

    if processing not in {"standard", "batch", "flex", "fast"}:
        raise ProtocolError("unsupported processing mode")
    if min(input_tokens, cached_input_tokens, cache_write_tokens, output_tokens) < 0:
        raise ProtocolError("token counts cannot be negative")
    if cached_input_tokens + cache_write_tokens > input_tokens:
        raise ProtocolError("cached and cache-write tokens exceed input tokens")

    uncached_tokens = input_tokens - cached_input_tokens - cache_write_tokens
    threshold_multiplier = 2.0 if input_tokens > WHOLE_REQUEST_THRESHOLD else 1.0
    cost = (
        uncached_tokens * PRICES_PER_MILLION["input"]
        + cached_input_tokens * PRICES_PER_MILLION["cached_input"]
        + cache_write_tokens * PRICES_PER_MILLION["cache_write"]
    ) * threshold_multiplier / 1_000_000
    cost += output_tokens * PRICES_PER_MILLION["output"] * (
        1.5 if input_tokens > WHOLE_REQUEST_THRESHOLD else 1.0
    ) / 1_000_000

    if processing in {"batch", "flex"}:
        cost *= 0.5
    elif processing == "fast":
        cost *= 2.0
    if regional:
        cost *= 1.1
    return cost


def validate_tool_lineage(history: list[Mapping[str, Any]]) -> None:
    calls: set[str] = set()
    outputs: set[str] = set()
    for item in history:
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id or call_id in calls:
                raise ProtocolError("function call id is missing or duplicated")
            calls.add(call_id)
        elif item_type == "function_call_output":
            call_id = item.get("call_id")
            if call_id not in calls or call_id in outputs:
                raise ProtocolError("function output has invalid call lineage")
            outputs.add(call_id)
        elif item_type in {
            "message",
            "reasoning",
            "assistant_phase",
            "configuration_update",
            "compaction",
        }:
            continue
        else:
            raise ProtocolError(f"unknown replay item: {item_type}")
    missing = calls - outputs
    if missing:
        raise ProtocolError(f"function calls have no output: {sorted(missing)}")


@dataclass(frozen=True)
class ToolReceipt:
    call_id: str
    idempotency_key: str
    permission: str
    result: str
    artifact_digest: str
    verified: bool
    duplicate: bool = False


class ToolRuntime:
    """Keep proposal, permission, execution and verification as separate steps."""

    def __init__(self) -> None:
        self.receipts: dict[str, ToolReceipt] = {}
        self.executions = 0

    def execute(
        self, call: Mapping[str, Any], permission: str, result: str
    ) -> ToolReceipt:
        if permission != "allow":
            raise ProtocolError(f"permission decision {permission!r} blocks execution")
        call_id = call.get("call_id")
        key = call.get("idempotency_key")
        if not isinstance(call_id, str) or not call_id or not isinstance(key, str) or not key:
            raise ProtocolError("tool call needs call_id and idempotency_key")
        previous = self.receipts.get(key)
        if previous is not None:
            if previous.call_id != call_id or previous.result != result:
                raise ProtocolError("idempotency key was reused with changed content")
            return ToolReceipt(
                previous.call_id,
                previous.idempotency_key,
                previous.permission,
                previous.result,
                previous.artifact_digest,
                previous.verified,
                duplicate=True,
            )

        self.executions += 1
        receipt = ToolReceipt(
            call_id=call_id,
            idempotency_key=key,
            permission=permission,
            result=result,
            artifact_digest=digest({"call_id": call_id, "result": result}),
            verified=False,
        )
        self.receipts[key] = receipt
        return receipt

    def verify(self, receipt: ToolReceipt, expected_result: str) -> ToolReceipt:
        expected_digest = digest(
            {"call_id": receipt.call_id, "result": expected_result}
        )
        if receipt.result != expected_result:
            raise VerificationError("artifact verifier rejected the result")
        if receipt.artifact_digest != expected_digest:
            raise VerificationError("artifact digest mismatch")
        verified = ToolReceipt(
            receipt.call_id,
            receipt.idempotency_key,
            receipt.permission,
            receipt.result,
            receipt.artifact_digest,
            True,
            receipt.duplicate,
        )
        self.receipts[receipt.idempotency_key] = verified
        return verified


def compact(history: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return an opaque canonical next context for the local protocol toy."""

    validate_tool_lineage(history)
    return [
        {
            "type": "compaction",
            "id": "cmp_1",
            "encrypted_content": opaque(canonical_json(history)),
            "source_digest": digest(history),
        }
    ]


def replay_compaction(
    canonical_window: list[Mapping[str, Any]],
    replayed_window: list[Mapping[str, Any]],
) -> None:
    if not canonical_window or canonical_window[0].get("type") != "compaction":
        raise ProtocolError("compaction output must begin with the canonical item")
    if not canonical_window[0].get("encrypted_content"):
        raise ProtocolError("compaction item must remain opaque")
    if canonical_json(canonical_window) != canonical_json(replayed_window):
        raise ProtocolError("canonical compaction window was edited")


@dataclass
class CacheLedger:
    entries: dict[str, float]
    writes: int = 0
    reads: int = 0

    def write(self, prefix: str, token_count: int, now: float) -> None:
        if token_count < MIN_CACHEABLE_TOKENS:
            raise ProtocolError("prefix is below the GPT-6 Sol cache minimum")
        self.entries[prefix] = now + 30 * 60
        self.writes += 1

    def lookup(self, prefix: str, now: float) -> bool:
        expiry = self.entries.get(prefix)
        if expiry is None or now > expiry:
            return False
        self.entries[prefix] = now + 30 * 60
        self.reads += 1
        return True


def explicit_breakpoints(count: int) -> list[dict[str, str]]:
    if count < 0 or count > MAX_EXPLICIT_BREAKPOINTS:
        raise ProtocolError("a request can create at most four explicit cache writes")
    return [{"mode": "explicit", "index": str(index)} for index in range(count)]


def main() -> None:
    standard_low = ReasoningConfig(mode="standard", effort="low")
    pro_high = ReasoningConfig(mode="pro", effort="high")
    standard_low.validate()
    pro_high.validate()
    assert standard_low.mode != pro_high.mode
    assert standard_low.effort != pro_high.effort

    legal_update_history: list[dict[str, Any]] = [
        {"type": "configuration_update", "reasoning": {"effort": "high"}},
        {"type": "message", "role": "user", "content": "analyze the migration"},
    ]
    validate_configuration_updates(legal_update_history, standard_low)
    stable_prompt_prefix = {
        "instructions": "stable coding policy",
        "tools": ["apply_patch"],
    }
    prefix_digest_before = digest(stable_prompt_prefix)
    persistent_update_history = [
        {"type": "configuration_update", "reasoning": {"effort": "high"}},
        {"type": "message", "role": "user", "content": "analyze the migration"},
        {"type": "message", "role": "assistant", "content": "draft"},
        {"type": "message", "role": "user", "content": "check the rollback risks"},
    ]
    validate_configuration_updates(persistent_update_history, standard_low)
    persistent_effort = resolve_effective_effort(
        standard_low.effort, persistent_update_history, stable_prompt_prefix
    )
    assert persistent_effort.request_level_effort == "low"
    assert persistent_effort.effective_effort == "high"
    assert persistent_effort.response_effort_field == "low"
    assert persistent_effort.stable_prefix_digest == prefix_digest_before

    override_history = persistent_update_history + [
        {"type": "message", "role": "assistant", "content": "follow-up"},
        {"type": "configuration_update", "reasoning": {"effort": "low"}},
        {"type": "message", "role": "user", "content": "summarize routinely"},
    ]
    validate_configuration_updates(override_history, standard_low)
    overridden_effort = resolve_effective_effort(
        standard_low.effort, override_history, stable_prompt_prefix
    )
    assert overridden_effort.effective_effort == "low"
    assert overridden_effort.response_effort_field == "low"
    assert overridden_effort.stable_prefix_digest == prefix_digest_before
    expect_error(
        lambda: validate_configuration_updates(
            legal_update_history, standard_low, automatic_compaction=True
        ),
        "automatic compaction",
    )
    expect_error(
        lambda: validate_configuration_updates(
            legal_update_history, standard_low, standalone_compact=True
        ),
        "standalone compaction",
    )
    expect_error(
        lambda: validate_configuration_updates(legal_update_history, pro_high),
        "standard single-agent",
    )
    expect_error(
        lambda: validate_configuration_updates(
            [
                {"type": "configuration_update", "reasoning": {"effort": "high"}},
                {"type": "configuration_update", "reasoning": {"effort": "max"}},
                {"type": "message", "role": "user", "content": "go"},
            ],
            standard_low,
        ),
        "adjacent",
    )
    expect_error(
        lambda: validate_configuration_updates(
            [
                {
                    "type": "configuration_update",
                    "reasoning": {"effort": "high", "mode": "pro"},
                },
                {"type": "message", "role": "user", "content": "go"},
            ],
            standard_low,
        ),
        "only reasoning.effort",
    )

    post_compaction = [
        {"type": "compaction", "encrypted_content": opaque("cmp")},
        {"type": "configuration_update", "reasoning": {"effort": "max"}},
        {"type": "message", "role": "user", "content": "continue"},
    ]
    validate_configuration_updates(post_compaction, standard_low)
    validate_post_compaction_update(post_compaction)
    expect_error(
        lambda: validate_post_compaction_update(
            [
                {"type": "compaction", "encrypted_content": opaque("cmp")},
                {"type": "message", "role": "user", "content": "continue"},
            ]
        ),
        "fresh configuration_update",
    )

    completed = evaluate_budget(
        input_tokens=100_000,
        reasoning_tokens=18_000,
        visible_output_tokens=2_000,
        max_output_tokens=25_000,
    )
    incomplete = evaluate_budget(
        input_tokens=100_000,
        reasoning_tokens=24_000,
        visible_output_tokens=2_000,
        max_output_tokens=25_000,
    )
    assert completed.status == "completed"
    assert incomplete.status == "incomplete"
    assert incomplete.reason == "max_output_tokens"
    assert REASONING_RESERVE == 25_000

    boundary_cost = estimate_cost(WHOLE_REQUEST_THRESHOLD, 0, 0, 1_000)
    large_request_cost = estimate_cost(300_000, 100_000, 50_000, 1_000)
    expected_large_request_cost = (
        (150_000 * 2.0 + 100_000 * 0.20 + 50_000 * 2.50) * 2.0
        + 1_000 * 10.0 * 1.5
    ) / 1_000_000
    assert abs(large_request_cost - expected_large_request_cost) < 1e-12
    assert large_request_cost > boundary_cost
    assert abs(
        estimate_cost(300_000, 100_000, 50_000, 1_000, processing="batch")
        - large_request_cost * 0.5
    ) < 1e-12

    call = {
        "type": "function_call",
        "call_id": "call_patch_0",
        "name": "apply_patch",
        "idempotency_key": "idem_patch_0",
    }
    runtime = ToolRuntime()
    expect_error(lambda: runtime.execute(call, "ask", "patch applied"), "blocks execution")
    receipt = runtime.execute(call, "allow", "patch applied")
    verified = runtime.verify(receipt, "patch applied")
    duplicate = runtime.execute(call, "allow", "patch applied")
    assert verified.verified and duplicate.duplicate and runtime.executions == 1
    expect_error(
        lambda: runtime.execute(call, "allow", "different patch"), "changed content"
    )
    expect_error(
        lambda: runtime.verify(receipt, "untrusted patch"), "rejected the result"
    )

    history: list[dict[str, Any]] = [
        {"type": "message", "role": "user", "content": "update the repo"},
        {"type": "reasoning", "encrypted_content": opaque("reasoning_0")},
        call,
        {
            "type": "function_call_output",
            "call_id": call["call_id"],
            "output": verified.result,
        },
        {"type": "assistant_phase", "phase": "final_answer"},
    ]
    validate_tool_lineage(history)
    expect_error(lambda: validate_tool_lineage(history[:-2]), "no output")

    canonical_window = compact(history)
    replay_compaction(canonical_window, list(canonical_window))
    edited_window = list(canonical_window)
    edited_window[0] = dict(edited_window[0], source_digest="tampered")
    expect_error(
        lambda: replay_compaction(canonical_window, edited_window), "was edited"
    )

    stable_prefix = digest({"instructions": "stable", "tools": ["apply_patch"]})
    compacted_prefix = digest({"compaction": canonical_window, "tools": ["apply_patch"]})
    cache = CacheLedger(entries={})
    cache.write(stable_prefix, MIN_CACHEABLE_TOKENS, now=0)
    assert cache.lookup(stable_prefix, now=60) is True
    assert cache.lookup(compacted_prefix, now=60) is False
    assert cache.writes == 1 and cache.reads == 1
    assert len(explicit_breakpoints(MAX_EXPLICIT_BREAKPOINTS)) == 4
    expect_error(lambda: explicit_breakpoints(5), "at most four")
    expect_error(lambda: cache.write("short", 1, now=0), "minimum")

    print(
        json.dumps(
            {
                "ok": True,
                "evidence_level": "local_protocol_toy",
                "model": MODEL,
                "network_called": False,
                "reasoning": {
                    "orthogonal_mode_effort": True,
                    "legal_update": True,
                    "configuration_update_changes_effort_only": True,
                    "effective_effort_persists_until_override": persistent_effort.effective_effort == "high",
                    "response_effort_field_is_request_level": persistent_effort.response_effort_field == "low",
                    "stable_prompt_prefix_unchanged": persistent_effort.stable_prefix_digest == prefix_digest_before,
                    "cache_hit_claimed": False,
                    "post_compaction_update_required": True,
                },
                "budget": {
                    "context_window": CONTEXT_WINDOW,
                    "maximum_input": MAX_INPUT,
                    "maximum_output": MAX_OUTPUT,
                    "reserve_recommendation": REASONING_RESERVE,
                    "completed_status": completed.status,
                    "incomplete_reason": incomplete.reason,
                },
                "pricing": {
                    "threshold_tokens": WHOLE_REQUEST_THRESHOLD,
                    "whole_request_multiplier_verified": True,
                    "boundary_cost_usd": boundary_cost,
                    "large_request_cost_usd": large_request_cost,
                },
                "tool_runtime": {
                    "permission_gate": True,
                    "executor_executions": runtime.executions,
                    "idempotent_replay": duplicate.duplicate,
                    "artifact_verified": verified.verified,
                },
                "compaction_and_cache": {
                    "opaque_canonical_window": True,
                    "cache_hit_before_compaction": True,
                    "cache_hit_after_prefix_change": False,
                    "explicit_breakpoints": MAX_EXPLICIT_BREAKPOINTS,
                    "ttl_seconds": 30 * 60,
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
