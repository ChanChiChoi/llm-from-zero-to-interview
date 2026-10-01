"""Dependency-free toy audit for GPT-5.6 Luna runtime contracts.

This script models public protocol boundaries only. It does not call an
OpenAI endpoint, expose hidden reasoning, execute tools, or load weights.
Its evidence level is local_protocol_toy.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping


class ProtocolError(ValueError):
    """Raised when a toy transition violates the documented contract."""


class VerificationError(ProtocolError):
    """Raised when a tool artifact cannot pass an independent verifier."""


MODEL = "gpt-5.6-luna"
MODEL_FAMILY = "gpt-5.6"
FAMILY_MODELS = {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}
OTHER_FAMILY_MODEL = "gpt-5.5"
REASONING_CONTEXTS = {"auto", "current_turn", "all_turns"}
MIN_CACHEABLE_TOKENS = 1_024
MAX_EXPLICIT_BREAKPOINTS = 4


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


def reasoning_items(
    history: list[Mapping[str, Any]],
    context: str,
    active_turn: int,
    target_model: str = MODEL,
) -> list[dict[str, Any]]:
    """Return the opaque items that a GPT-5.6 sample may render."""

    if context not in REASONING_CONTEXTS:
        raise ProtocolError("reasoning.context is unsupported")
    if target_model not in FAMILY_MODELS:
        raise ProtocolError("target model is outside the GPT-5.6 family")

    effective = "all_turns" if context == "auto" else context
    result = []
    for item in history:
        if item.get("type") != "reasoning":
            continue
        if item.get("model_family") != MODEL_FAMILY:
            continue
        if effective == "current_turn" and item.get("turn") != active_turn:
            continue
        result.append(deepcopy(dict(item)))
    return result


def switch_model_family(
    history: list[Mapping[str, Any]], target_model: str
) -> list[dict[str, Any]]:
    """Model-family changes omit incompatible persisted reasoning items."""

    target_family = MODEL_FAMILY if target_model in FAMILY_MODELS else "gpt-5.5"
    return [
        deepcopy(dict(item))
        for item in history
        if item.get("type") != "reasoning"
        or item.get("model_family") == target_family
    ]


def validate_replay(history: list[Mapping[str, Any]]) -> None:
    """Check the item lineage needed for stateless Responses replay."""

    calls: dict[str, Mapping[str, Any]] = {}
    results: set[str] = set()
    for index, item in enumerate(history):
        item_type = item.get("type")
        if item_type == "reasoning":
            if not item.get("encrypted_content"):
                raise ProtocolError(f"reasoning item {index} is not opaque")
        elif item_type == "function_call":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id or call_id in calls:
                raise ProtocolError("function call id is missing or duplicated")
            calls[call_id] = item
        elif item_type == "function_call_output":
            call_id = item.get("call_id")
            if call_id not in calls or call_id in results:
                raise ProtocolError("function output has invalid call lineage")
            results.add(call_id)
        elif item_type in {"message", "assistant_phase", "compaction"}:
            continue
        else:
            raise ProtocolError(f"unknown replay item: {item_type}")

    missing = set(calls) - results
    if missing:
        raise ProtocolError(f"function calls have no output: {sorted(missing)}")


@dataclass(frozen=True)
class ToolReceipt:
    call_id: str
    idempotency_key: str
    result: str
    artifact_digest: str
    verified: bool
    duplicate: bool = False


class ToolLedger:
    """Separate execution, idempotency, and artifact verification."""

    def __init__(self) -> None:
        self.receipts: dict[str, ToolReceipt] = {}
        self.executions = 0

    def execute(self, call: Mapping[str, Any], result: str) -> ToolReceipt:
        key = call.get("idempotency_key")
        call_id = call.get("call_id")
        if not isinstance(key, str) or not key or not isinstance(call_id, str):
            raise ProtocolError("tool call needs call_id and idempotency_key")
        previous = self.receipts.get(key)
        if previous is not None:
            if previous.result != result or previous.call_id != call_id:
                raise ProtocolError("idempotency key was reused with changed content")
            return ToolReceipt(
                previous.call_id,
                previous.idempotency_key,
                previous.result,
                previous.artifact_digest,
                previous.verified,
                duplicate=True,
            )

        self.executions += 1
        receipt = ToolReceipt(
            call_id=call_id,
            idempotency_key=key,
            result=result,
            artifact_digest=digest({"result": result, "call_id": call_id}),
            verified=False,
        )
        self.receipts[key] = receipt
        return receipt

    def verify(self, receipt: ToolReceipt, expected_result: str) -> ToolReceipt:
        if receipt.result != expected_result:
            raise VerificationError("artifact verifier rejected the result")
        expected_digest = digest(
            {"result": expected_result, "call_id": receipt.call_id}
        )
        if receipt.artifact_digest != expected_digest:
            raise VerificationError("artifact digest mismatch")
        verified = ToolReceipt(
            receipt.call_id,
            receipt.idempotency_key,
            receipt.result,
            receipt.artifact_digest,
            True,
            receipt.duplicate,
        )
        self.receipts[receipt.idempotency_key] = verified
        return verified


def compact(history: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the toy's canonical next window with an opaque compaction item."""

    validate_replay(history)
    return [
        {
            "type": "compaction",
            "id": "cmp_1",
            "encrypted_content": opaque(canonical_json(history)),
            "source_digest": digest(history),
        }
    ]


def validate_compaction_window(window: list[Mapping[str, Any]]) -> None:
    if not window or window[0].get("type") != "compaction":
        raise ProtocolError("compaction output must begin with the canonical item")
    if not window[0].get("encrypted_content"):
        raise ProtocolError("compaction item must remain opaque")


def replay_compaction(
    canonical_window: list[Mapping[str, Any]],
    replayed_window: list[Mapping[str, Any]],
) -> None:
    """Require standalone compaction output to be replayed without edits."""

    validate_compaction_window(canonical_window)
    validate_compaction_window(replayed_window)
    if canonical_json(canonical_window) != canonical_json(replayed_window):
        raise ProtocolError("canonical compaction window was edited")


@dataclass
class CacheLedger:
    entries: dict[str, float]
    writes: int = 0
    reads: int = 0

    def write(self, prefix: str, token_count: int, now: float) -> None:
        if token_count < MIN_CACHEABLE_TOKENS:
            raise ProtocolError("prefix is below the GPT-5.6 cache minimum")
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


@dataclass
class ToolSearchSession:
    pending: dict[str, str]
    loaded: list[dict[str, Any]]

    def hosted(self, namespace: str) -> dict[str, Any]:
        return {
            "type": "tool_search_call",
            "execution": "server",
            "call_id": None,
            "namespace": namespace,
        }

    def client_call(self, call_id: str, namespace: str) -> dict[str, Any]:
        if not call_id or call_id in self.pending:
            raise ProtocolError("client tool search call_id must be unique")
        self.pending[call_id] = namespace
        return {
            "type": "tool_search_call",
            "execution": "client",
            "call_id": call_id,
            "namespace": namespace,
        }

    def client_output(
        self, call_id: str, tools: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        if call_id not in self.pending:
            raise ProtocolError("tool search output has no pending client call")
        loaded = [deepcopy(dict(tool)) for tool in tools]
        self.loaded.extend(loaded)
        del self.pending[call_id]
        return {
            "type": "tool_search_output",
            "execution": "client",
            "call_id": call_id,
            "tools": loaded,
        }


def main() -> None:
    history: list[dict[str, Any]] = [
        {"type": "message", "role": "user", "content": "inspect the repo", "turn": 0},
        {
            "type": "reasoning",
            "id": "rs_0",
            "turn": 0,
            "model_family": MODEL_FAMILY,
            "producer_model": MODEL,
            "encrypted_content": opaque("rs_0"),
        },
        {
            "type": "function_call",
            "call_id": "call_0",
            "name": "read_file",
            "idempotency_key": "idem_0",
        },
    ]

    ledger = ToolLedger()
    receipt = ledger.execute(history[-1], "README is present")
    receipt = ledger.verify(receipt, "README is present")
    history.extend(
        [
            {
                "type": "function_call_output",
                "call_id": "call_0",
                "output": receipt.result,
            },
            {"type": "assistant_phase", "phase": "final_answer"},
        ]
    )
    validate_replay(history)
    replay_snapshot = deepcopy(history)
    assert canonical_json(replay_snapshot) == canonical_json(history)

    history.extend(
        [
            {"type": "message", "role": "user", "content": "now summarize", "turn": 1},
            {
                "type": "reasoning",
                "id": "rs_1",
                "turn": 1,
                "model_family": MODEL_FAMILY,
                "producer_model": MODEL,
                "encrypted_content": opaque("rs_1"),
            },
        ]
    )
    assert [item["id"] for item in reasoning_items(history, "current_turn", 1)] == [
        "rs_1"
    ]
    assert [item["id"] for item in reasoning_items(history, "all_turns", 1)] == [
        "rs_0",
        "rs_1",
    ]
    assert [item["id"] for item in reasoning_items(history, "auto", 1)] == [
        "rs_0",
        "rs_1",
    ]
    assert [
        item["id"]
        for item in switch_model_family(history, OTHER_FAMILY_MODEL)
        if item.get("type") == "reasoning"
    ] == []

    expect_error(lambda: validate_replay(history[:3]), "no output")
    duplicate = ledger.execute(history[2], "README is present")
    assert duplicate.duplicate is True and ledger.executions == 1
    ledger.verify(duplicate, "README is present")
    expect_error(
        lambda: ledger.execute(history[2], "README changed"), "changed content"
    )

    compacted = compact(history)
    validate_compaction_window(compacted)
    compacted_replay = deepcopy(compacted)
    replay_compaction(compacted, compacted_replay)
    edited_compaction = deepcopy(compacted)
    edited_compaction[0]["source_digest"] = "tampered"
    expect_error(
        lambda: replay_compaction(compacted, edited_compaction),
        "was edited",
    )
    expect_error(
        lambda: validate_compaction_window([{"type": "message", "content": "summary"}]),
        "canonical item",
    )

    stable_prefix = digest({"developer": "stable instructions", "tools": ["read_file"]})
    compacted_prefix = digest({"compaction": compacted, "tools": ["read_file"]})
    cache = CacheLedger(entries={})
    cache.write(stable_prefix, MIN_CACHEABLE_TOKENS, now=0)
    assert cache.lookup(stable_prefix, now=60) is True
    assert cache.lookup(compacted_prefix, now=60) is False
    assert cache.writes == 1 and cache.reads == 1
    assert len(explicit_breakpoints(MAX_EXPLICIT_BREAKPOINTS)) == 4
    expect_error(lambda: explicit_breakpoints(5), "at most four")
    expect_error(lambda: cache.write("short", 1, now=0), "minimum")

    search = ToolSearchSession(pending={}, loaded=[])
    hosted = search.hosted("repo")
    assert hosted["execution"] == "server" and hosted["call_id"] is None
    client_call = search.client_call("search_0", "repo")
    expect_error(
        lambda: search.client_output("wrong_id", [{"name": "read_file"}]),
        "pending client call",
    )
    client_output = search.client_output("search_0", [{"name": "read_file"}])
    assert client_call["call_id"] == client_output["call_id"] == "search_0"
    assert search.loaded == [{"name": "read_file"}]

    print(
        json.dumps(
            {
                "ok": True,
                "evidence_level": "local_protocol_toy",
                "model": MODEL,
                "reasoning": {
                    "current_turn": ["rs_1"],
                    "all_turns": ["rs_0", "rs_1"],
                    "incompatible_family_items": 0,
                },
                "replay": {
                    "opaque_reasoning_preserved": True,
                    "function_call_output_lineage": True,
                    "idempotent_executions": ledger.executions,
                },
                "compaction": {
                    "canonical_window_preserved": True,
                    "cache_hit_after_prefix_change": False,
                },
                "cache": {
                    "min_tokens": MIN_CACHEABLE_TOKENS,
                    "explicit_breakpoints": MAX_EXPLICIT_BREAKPOINTS,
                    "ttl_seconds": 30 * 60,
                },
                "tool_search": {
                    "hosted_execution": hosted["execution"],
                    "client_call_id_preserved": True,
                    "loaded_tools_are_not_permissions": True,
                },
                "network_called": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
