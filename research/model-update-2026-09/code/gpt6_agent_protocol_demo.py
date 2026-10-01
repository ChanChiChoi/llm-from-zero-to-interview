"""Toy audit for GPT-6 Astra agent protocol concepts.

This is a local state-machine exercise. It does not call an OpenAI API,
execute tools, or model hidden reasoning.
"""

from dataclasses import dataclass, field
import json
from typing import Dict, List, Optional


class ProtocolError(ValueError):
    """Raised when a toy protocol transition violates its contract."""


@dataclass
class AsyncJob:
    task_handle: str
    call_id: str
    response_id: str
    status: str = "pending"
    result: Optional[str] = None
    consumed: bool = False


class AsyncJobRegistry:
    """Separate tool completion from model consumption of the result."""

    def __init__(self) -> None:
        self.jobs: Dict[str, AsyncJob] = {}

    def start(self, response_id: str, task_handle: str, call_id: str) -> None:
        if not response_id or not task_handle or not call_id:
            raise ProtocolError("response_id, task_handle, and call_id are required")
        if task_handle in self.jobs or any(job.call_id == call_id for job in self.jobs.values()):
            raise ProtocolError("task handles and call ids must be unique")
        self.jobs[task_handle] = AsyncJob(task_handle, call_id, response_id)

    def complete(self, task_handle: str, call_id: str, result: str) -> str:
        job = self._get(task_handle)
        if job.call_id != call_id:
            raise ProtocolError("result call_id does not match the registered job")
        if job.status == "completed":
            if job.result != result:
                raise ProtocolError("a completed job cannot change its result")
            return "duplicate"
        if job.status != "pending":
            raise ProtocolError(f"cannot complete a {job.status} job")
        job.status = "completed"
        job.result = result
        return "accepted"

    def result_item(self, task_handle: str) -> Dict[str, str]:
        job = self._get(task_handle)
        if job.status != "completed" or job.result is None:
            raise ProtocolError("the model cannot consume a pending job")
        return {
            "type": "function_call_output",
            "call_id": job.call_id,
            "output": job.result,
        }

    def acknowledge_consumed(self, task_handle: str) -> None:
        job = self._get(task_handle)
        if job.status != "completed":
            raise ProtocolError("only completed jobs can be consumed")
        job.consumed = True

    def _get(self, task_handle: str) -> AsyncJob:
        try:
            return self.jobs[task_handle]
        except KeyError as exc:
            raise ProtocolError("unknown task handle") from exc


@dataclass
class ResponseState:
    response_id: str
    previous_response_id: Optional[str] = None
    status: str = "running"
    incomplete_reason: Optional[str] = None
    side_effects: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    connection_id: str = "ws-1"
    connected: bool = True
    steering_id: Optional[str] = None
    blocked_code: Optional[str] = None


def _assert_connection(response: ResponseState, connection_id: str) -> None:
    if not response.connected or response.connection_id != connection_id:
        raise ProtocolError("steering is scoped to the current WebSocket connection")


def steer(
    response: ResponseState,
    instruction: str,
    connection_id: str = "ws-1",
    steering_id: str = "steer_1",
) -> ResponseState:
    """Model response steering: queue input and preserve existing side effects."""

    _assert_connection(response, connection_id)
    if response.status != "running":
        raise ProtocolError("only a running response can be steered")
    if not instruction.strip():
        raise ProtocolError("steering input must be non-empty")
    if response.steering_id is not None:
        raise ProtocolError("a response can accept steering only once")
    if not steering_id:
        raise ProtocolError("steering_id must be non-empty")
    response.steering_id = steering_id
    response.events.append("response.steer.accepted")
    response.status = "incomplete"
    response.incomplete_reason = "steered"
    response.events.append("response.incomplete:steered")
    response.events.append("response.steer.pending")
    continuation = ResponseState(
        response_id="resp_2",
        previous_response_id=response.response_id,
        connection_id=response.connection_id,
        side_effects=list(response.side_effects),
    )
    continuation.events.append("response.created:continuation")
    return continuation


def disconnect(response: ResponseState, connection_id: str) -> None:
    """Close a WebSocket; queued steering is not a durable cross-connection command."""

    if response.connection_id != connection_id:
        raise ProtocolError("connection does not own this response")
    if not response.connected:
        raise ProtocolError("the WebSocket is already closed")
    response.connected = False
    response.events.append("websocket.closed")


def recover_after_disconnect(
    response: ResponseState, connection_id: str
) -> ResponseState:
    """Create an explicit recovery response without replaying steering implicitly."""

    if response.connected:
        raise ProtocolError("recovery requires a disconnected WebSocket")
    recovered = ResponseState(
        response_id="resp_recovery",
        previous_response_id=response.response_id,
        connection_id=connection_id,
        side_effects=list(response.side_effects),
    )
    recovered.events.extend(
        ["response.created:recovery", "steering.not_replayed"]
    )
    return recovered


def block_for_misalignment(response: ResponseState) -> None:
    """Apply a safety block without pretending that external effects were undone."""

    if response.blocked_code is not None:
        raise ProtocolError("response is already blocked")
    response.blocked_code = "misalignment_policy_violation"
    response.status = "blocked"
    response.events.append("error:misalignment_policy_violation")


def retry_after_block(response: ResponseState) -> None:
    """The host must stop blind retries after a policy block."""

    if response.blocked_code == "misalignment_policy_violation":
        raise ProtocolError(
            "automatic retry is forbidden after misalignment_policy_violation"
        )
    raise ProtocolError("response is not eligible for this retry path")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    details: str


def route_skills(skills: List[Skill], task: str) -> List[Skill]:
    """Select from short descriptions before loading any detailed document."""

    task_words = set(task.lower().replace("/", " ").split())
    selected = []
    for skill in skills:
        description_words = set(skill.description.lower().replace("/", " ").split())
        if task_words & description_words:
            selected.append(skill)
    return selected


def expect_protocol_error(action, expected: str) -> None:
    """Assert a failure gate while keeping the toy dependency-free."""

    try:
        action()
    except ProtocolError as exc:
        assert expected in str(exc), (expected, str(exc))
    else:
        raise AssertionError(f"expected ProtocolError containing {expected!r}")


def main() -> None:
    registry = AsyncJobRegistry()
    registry.start("resp_1", "weather-job", "call_weather")
    expect_protocol_error(
        lambda: registry.start("resp_1", "weather-job", "call_other"),
        "unique",
    )
    expect_protocol_error(
        lambda: registry.start("resp_1", "other-job", "call_weather"),
        "unique",
    )
    expect_protocol_error(
        lambda: registry.result_item("weather-job"),
        "pending",
    )
    expect_protocol_error(
        lambda: registry.complete("weather-job", "wrong_call", "demo weather"),
        "call_id",
    )
    assert registry.complete("weather-job", "call_weather", "demo weather") == "accepted"
    assert registry.complete("weather-job", "call_weather", "demo weather") == "duplicate"
    expect_protocol_error(
        lambda: registry.complete("weather-job", "call_weather", "changed result"),
        "cannot change",
    )
    item = registry.result_item("weather-job")
    assert item["call_id"] == "call_weather"
    assert registry.jobs["weather-job"].consumed is False
    registry.acknowledge_consumed("weather-job")
    assert registry.jobs["weather-job"].consumed is True

    initial = ResponseState("resp_1")
    initial.side_effects.append("started:slow_lookup")
    continuation = steer(initial, "Keep the scope small")
    assert initial.incomplete_reason == "steered"
    assert continuation.previous_response_id == "resp_1"
    assert continuation.side_effects == ["started:slow_lookup"]
    expect_protocol_error(
        lambda: steer(initial, "Steer again"),
        "only a running",
    )

    disconnected = ResponseState("resp_disconnect")
    disconnect(disconnected, "ws-1")
    expect_protocol_error(
        lambda: steer(disconnected, "Reuse the old steering", connection_id="ws-2"),
        "current WebSocket",
    )
    recovered = recover_after_disconnect(disconnected, "ws-2")
    assert recovered.previous_response_id == "resp_disconnect"
    assert recovered.steering_id is None
    assert "steering.not_replayed" in recovered.events

    safety_blocked = ResponseState("resp_safety")
    safety_blocked.side_effects.append("started:external_write")
    block_for_misalignment(safety_blocked)
    expect_protocol_error(
        lambda: retry_after_block(safety_blocked),
        "automatic retry",
    )
    assert safety_blocked.status == "blocked"
    assert safety_blocked.side_effects == ["started:external_write"]

    skills = [
        Skill(
            "migration",
            "Use when adding or changing a database migration.",
            "check schema, apply migration, run rollback test",
        ),
        Skill(
            "database-general",
            "Use for any database query or persistence task.",
            "large unrelated database playbook",
        ),
    ]
    selected = route_skills(skills, "change a database migration")
    manifest = [{"name": skill.name, "description": skill.description} for skill in selected]
    assert all("details" not in entry for entry in manifest)

    print(
        json.dumps(
            {
                "async_tool": {
                    "tool_status": registry.jobs["weather-job"].status,
                    "model_consumed_result": registry.jobs["weather-job"].consumed,
                    "duplicate_result": True,
                },
                "steering": {
                    "original_status": initial.status,
                    "incomplete_reason": initial.incomplete_reason,
                    "continuation_previous_response_id": continuation.previous_response_id,
                    "side_effect_preserved": continuation.side_effects,
                    "events": initial.events,
                },
                "disconnect_recovery": {
                    "previous_response_id": recovered.previous_response_id,
                    "steering_replayed": recovered.steering_id is not None,
                    "events": recovered.events,
                },
                "misalignment": {
                    "status": safety_blocked.status,
                    "code": safety_blocked.blocked_code,
                    "side_effects_preserved": safety_blocked.side_effects,
                    "automatic_retry": "rejected",
                },
                "skill_routing": {
                    "selected": [skill.name for skill in selected],
                    "details_loaded": False,
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
