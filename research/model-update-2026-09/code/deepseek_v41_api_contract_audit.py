#!/usr/bin/env python3
"""Audit a DeepSeek V4.1-Flash API contract without making network calls.

This is a standard-library teaching toy. It models semantic SSE ordering,
strict tool arguments, host authorization, bounded pre-execution retries,
idempotent tool execution, and independent business verification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping


class ProtocolError(ValueError):
    """The event stream or typed item sequence violates its contract."""


class SchemaError(ValueError):
    """A tool argument object does not satisfy the toy strict schema."""


class PermissionDenied(PermissionError):
    """The host refuses to execute a model-proposed tool call."""


class RetryableToolError(RuntimeError):
    """A transient failure before a side effect is committed."""


class NonRetryableToolError(RuntimeError):
    """A failure that must not be retried by this harness."""


TERMINAL_EVENTS = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)


@dataclass(frozen=True)
class SemanticEvent:
    event: str
    sequence_number: int
    data: Mapping[str, Any]


class SemanticSSEParser:
    """Parse semantic SSE frames while retaining sequence and terminal state."""

    def __init__(self) -> None:
        self._buffer = ""
        self._event_name: str | None = None
        self._data_lines: list[str] = []
        self._last_sequence: int | None = None
        self._terminal_event: str | None = None
        self.events: list[SemanticEvent] = []
        self.text_deltas: list[str] = []

    @property
    def terminal_event(self) -> str | None:
        return self._terminal_event

    def feed(self, chunk: str) -> list[SemanticEvent]:
        self._buffer += chunk
        emitted: list[SemanticEvent] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            event = self._consume_line(line.rstrip("\r"))
            if event is not None:
                emitted.append(event)
        return emitted

    def finish(self) -> list[SemanticEvent]:
        if self._buffer:
            self._consume_line(self._buffer.rstrip("\r"))
            self._buffer = ""
        event = self._dispatch()
        return [event] if event is not None else []

    def _consume_line(self, line: str) -> SemanticEvent | None:
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if not separator:
            raise ProtocolError(f"malformed SSE line: {line!r}")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            if self._event_name is not None:
                raise ProtocolError("duplicate event field in one frame")
            self._event_name = value
        elif field == "data":
            self._data_lines.append(value)
        elif field in {"id", "retry"}:
            return None
        else:
            raise ProtocolError(f"unsupported SSE field: {field}")
        return None

    def _dispatch(self) -> SemanticEvent | None:
        if self._event_name is None and not self._data_lines:
            return None
        if self._event_name is None or not self._data_lines:
            raise ProtocolError("SSE frame needs event and data fields")
        event_name = self._event_name
        raw_data = "\n".join(self._data_lines)
        self._event_name = None
        self._data_lines = []
        if self._terminal_event is not None:
            raise ProtocolError("event received after terminal response")
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise ProtocolError("SSE data is not JSON") from exc
        if not isinstance(data, dict):
            raise ProtocolError("SSE data must be an object")
        sequence_number = data.get("sequence_number")
        if isinstance(sequence_number, bool) or not isinstance(sequence_number, int):
            raise ProtocolError("sequence_number must be an integer")
        if self._last_sequence is not None and sequence_number <= self._last_sequence:
            raise ProtocolError("sequence_number must increase strictly")
        self._last_sequence = sequence_number
        event = SemanticEvent(event_name, sequence_number, data)
        self.events.append(event)
        if event_name == "response.output_text.delta":
            delta = data.get("delta")
            if not isinstance(delta, str):
                raise ProtocolError("text delta must be a string")
            self.text_deltas.append(delta)
        if event_name in TERMINAL_EVENTS:
            self._terminal_event = event_name
        return event


@dataclass(frozen=True)
class StrictObjectSchema:
    properties: Mapping[str, str]
    required: frozenset[str]
    additional_properties: bool = False

    def validate(self, value: Any) -> None:
        if not isinstance(value, dict):
            raise SchemaError("tool arguments must be an object")
        missing = sorted(self.required - value.keys())
        if missing:
            raise SchemaError(f"missing required fields: {missing}")
        if not self.additional_properties:
            extra = sorted(set(value) - set(self.properties))
            if extra:
                raise SchemaError(f"additional properties are forbidden: {extra}")
        for name, type_name in self.properties.items():
            if name not in value:
                continue
            item = value[name]
            valid = {
                "string": isinstance(item, str),
                "integer": isinstance(item, int) and not isinstance(item, bool),
                "boolean": isinstance(item, bool),
                "object": isinstance(item, dict),
                "array": isinstance(item, list),
            }.get(type_name)
            if valid is not True:
                raise SchemaError(f"{name} must have type {type_name}")


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    idempotency_key: str


@dataclass(frozen=True)
class ToolReceipt:
    call_id: str
    status: str
    attempts: int
    result: Mapping[str, Any]
    idempotency_key: str
    duplicate: bool = False


class PermissionPolicy:
    def __init__(self, allowed_tools: Iterable[str], allowed_path_prefixes: Iterable[str]):
        self.allowed_tools = frozenset(allowed_tools)
        self.allowed_path_prefixes = tuple(allowed_path_prefixes)

    def authorize(self, call: ToolCall) -> None:
        if call.tool_name not in self.allowed_tools:
            raise PermissionDenied(f"tool is not allowed: {call.tool_name}")
        path = call.arguments.get("path")
        if not isinstance(path, str) or not path.startswith(self.allowed_path_prefixes):
            raise PermissionDenied(f"path is outside the allowed scope: {path!r}")


class ToolHarness:
    """Execute only after host checks; retries happen before side effects."""

    def __init__(self, transient_failures: int = 0, max_attempts: int = 3):
        self.transient_failures = transient_failures
        self.max_attempts = max_attempts
        self.side_effect_count = 0
        self._receipts: dict[str, ToolReceipt] = {}

    def execute(self, call: ToolCall) -> ToolReceipt:
        prior = self._receipts.get(call.idempotency_key)
        if prior is not None:
            return replace(prior, duplicate=True)
        for attempt in range(1, self.max_attempts + 1):
            if self.transient_failures:
                self.transient_failures -= 1
                if attempt == self.max_attempts:
                    raise RetryableToolError("retry budget exhausted before execution")
                continue
            if call.tool_name != "read_line":
                raise NonRetryableToolError(f"unknown executor: {call.tool_name}")
            self.side_effect_count += 1
            result = {
                "ok": True,
                "path": call.arguments["path"],
                "line": call.arguments["line"],
                "observation": "synthetic line content",
            }
            receipt = ToolReceipt(
                call_id=call.call_id,
                status="completed",
                attempts=attempt,
                result=result,
                idempotency_key=call.idempotency_key,
            )
            self._receipts[call.idempotency_key] = receipt
            return receipt
        raise RetryableToolError("unreachable retry state")


def audit_tool_call(
    call: ToolCall,
    schema: StrictObjectSchema,
    policy: PermissionPolicy,
    harness: ToolHarness,
    expected_result: Mapping[str, Any],
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "call_id": call.call_id,
        "schema_valid": False,
        "authorized": False,
        "executed": False,
        "verified": False,
        "attempts": 0,
        "duplicate": False,
        "error": None,
    }
    try:
        schema.validate(call.arguments)
        audit["schema_valid"] = True
        policy.authorize(call)
        audit["authorized"] = True
        receipt = harness.execute(call)
        audit["executed"] = receipt.status == "completed"
        audit["attempts"] = receipt.attempts
        audit["duplicate"] = receipt.duplicate
        audit["verified"] = dict(receipt.result) == dict(expected_result)
    except (SchemaError, PermissionDenied, RetryableToolError, NonRetryableToolError) as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
    return audit


def tool_output_item(call_id: str, output: Any) -> dict[str, Any]:
    """Build a typed observation; output may be text or an image content list."""
    if not isinstance(call_id, str) or not call_id:
        raise ProtocolError("tool output needs a non-empty call_id")
    if not isinstance(output, (str, list, dict)):
        raise ProtocolError("tool output must be text or structured content")
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _frame(event: str, sequence_number: int, **payload: Any) -> str:
    data = {"sequence_number": sequence_number, **payload}
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, separators=(',', ':'), sort_keys=True)}\n\n"
    )


def _parse_chunked_stream(frames: Iterable[str], chunk_size: int = 5) -> SemanticSSEParser:
    parser = SemanticSSEParser()
    stream = "".join(frames)
    for offset in range(0, len(stream), chunk_size):
        parser.feed(stream[offset : offset + chunk_size])
    parser.finish()
    return parser


def _assert_protocol_rejection() -> str:
    parser = SemanticSSEParser()
    parser.feed(_frame("response.output_text.delta", 2, delta="late"))
    try:
        parser.feed(_frame("response.output_text.delta", 1, delta="early"))
    except ProtocolError:
        return "out_of_order_rejected"
    raise AssertionError("out-of-order sequence was accepted")


def main() -> int:
    parser = _parse_chunked_stream(
        [
            _frame("response.output_text.delta", 1, delta="contract "),
            _frame("response.output_text.delta", 2, delta="audit"),
            _frame("response.completed", 3, status="completed"),
        ]
    )
    assert parser.terminal_event == "response.completed"
    assert "".join(parser.text_deltas) == "contract audit"

    schema = StrictObjectSchema(
        properties={"path": "string", "line": "integer"},
        required=frozenset({"path", "line"}),
    )
    policy = PermissionPolicy({"read_line"}, {"src/"})
    harness = ToolHarness(transient_failures=1)
    call = ToolCall(
        call_id="call_1",
        tool_name="read_line",
        arguments={"path": "src/app.py", "line": 7},
        idempotency_key="idem_1",
    )
    expected = {
        "ok": True,
        "path": "src/app.py",
        "line": 7,
        "observation": "synthetic line content",
    }
    first = audit_tool_call(call, schema, policy, harness, expected)
    side_effects_after_first = harness.side_effect_count
    duplicate = audit_tool_call(call, schema, policy, harness, expected)
    assert first["attempts"] == 2
    assert first["verified"] is True
    assert duplicate["duplicate"] is True
    assert harness.side_effect_count == side_effects_after_first == 1

    bad_schema = audit_tool_call(
        replace(call, call_id="call_bad", arguments={**call.arguments, "extra": 1}),
        schema,
        policy,
        harness,
        expected,
    )
    denied = audit_tool_call(
        replace(call, call_id="call_denied", arguments={"path": "secret.txt", "line": 7}),
        schema,
        policy,
        harness,
        expected,
    )
    assert bad_schema["schema_valid"] is False
    assert denied["schema_valid"] is True and denied["authorized"] is False
    assert _assert_protocol_rejection() == "out_of_order_rejected"

    summary = {
        "stream": {
            "event_count": len(parser.events),
            "text": "".join(parser.text_deltas),
            "terminal": parser.terminal_event,
        },
        "tool_success": first,
        "tool_duplicate": duplicate,
        "schema_rejection": bad_schema,
        "permission_rejection": denied,
        "tool_output_text": tool_output_item("call_1", "synthetic observation"),
        "tool_output_image": tool_output_item(
            "call_1", [{"type": "input_image", "file_id": "file_demo"}]
        ),
        "invariants": {
            "duplicate_did_not_repeat_side_effect": harness.side_effect_count == 1,
            "business_verifier_is_independent": first["executed"] and first["verified"],
            "network_called": False,
        },
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
