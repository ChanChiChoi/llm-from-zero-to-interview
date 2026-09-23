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


def steer(response: ResponseState, instruction: str) -> ResponseState:
    """Model response steering: queue input and preserve existing side effects."""

    if response.status != "running":
        raise ProtocolError("only a running response can be steered")
    if not instruction.strip():
        raise ProtocolError("steering input must be non-empty")
    response.events.append("response.steer.accepted")
    response.status = "incomplete"
    response.incomplete_reason = "steered"
    response.events.append("response.incomplete:steered")
    continuation = ResponseState(
        response_id="resp_2",
        previous_response_id=response.response_id,
        side_effects=list(response.side_effects),
    )
    continuation.events.append("response.created:continuation")
    return continuation


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


def main() -> None:
    registry = AsyncJobRegistry()
    registry.start("resp_1", "weather-job", "call_weather")
    assert registry.complete("weather-job", "call_weather", "demo weather") == "accepted"
    assert registry.complete("weather-job", "call_weather", "demo weather") == "duplicate"
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
