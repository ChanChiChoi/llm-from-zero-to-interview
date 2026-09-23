#!/usr/bin/env python3
"""Audit a teaching compaction contract without a model, network, or weights.

Z.ai publicly says GLM-5.3 inherits "SAO with compaction", but does not
publish the compaction schema or implementation.  This toy therefore tests
generic recovery invariants only.  It is not evidence of GLM-5.3 behavior.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ToolEvent:
    call_id: str
    tool_name: str
    arguments_digest: str
    result_digest: str
    permission_scope: str
    idempotency_key: str
    status: str


@dataclass(frozen=True)
class TaskState:
    schema_version: int
    task_id: str
    goal: str
    current_plan: tuple[str, ...]
    completed_steps: tuple[str, ...]
    tool_events: tuple[ToolEvent, ...]
    pending_effects: tuple[str, ...]
    artifact_digest: str
    verifier_status: str
    remaining_budget: int
    cut_marker: str


@dataclass(frozen=True)
class CompactionCandidate:
    state: TaskState
    dropped_history_ids: tuple[str, ...]
    prompt_tokens_before: int
    prompt_tokens_after: int


def state_to_dict(state: TaskState) -> dict[str, Any]:
    value = asdict(state)
    value["current_plan"] = list(state.current_plan)
    value["completed_steps"] = list(state.completed_steps)
    value["tool_events"] = [asdict(event) for event in state.tool_events]
    value["pending_effects"] = list(state.pending_effects)
    return value


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def serialize_state(state: TaskState) -> str:
    return canonical_json(state_to_dict(state))


def deserialize_state(payload: str) -> TaskState:
    raw = json.loads(payload)
    required = {
        "schema_version",
        "task_id",
        "goal",
        "current_plan",
        "completed_steps",
        "tool_events",
        "pending_effects",
        "artifact_digest",
        "verifier_status",
        "remaining_budget",
        "cut_marker",
    }
    if set(raw) != required:
        raise ValueError("state schema keys do not match the contract")
    if raw["schema_version"] != 1:
        raise ValueError("unsupported state schema version")
    events = tuple(ToolEvent(**event) for event in raw["tool_events"])
    return TaskState(
        schema_version=raw["schema_version"],
        task_id=raw["task_id"],
        goal=raw["goal"],
        current_plan=tuple(raw["current_plan"]),
        completed_steps=tuple(raw["completed_steps"]),
        tool_events=events,
        pending_effects=tuple(raw["pending_effects"]),
        artifact_digest=raw["artifact_digest"],
        verifier_status=raw["verifier_status"],
        remaining_budget=raw["remaining_budget"],
        cut_marker=raw["cut_marker"],
    )


def audit_candidate(
    candidate: CompactionCandidate,
    expected: TaskState,
) -> dict[str, bool]:
    """Check recovery invariants that a real implementation should expose."""

    state = candidate.state
    round_trip = deserialize_state(serialize_state(state)) == state
    canonical_event_ids = {event.call_id for event in state.tool_events}
    lineage_ok = all(
        event.call_id
        and event.result_digest
        and event.status in {"completed", "pending", "rejected"}
        for event in state.tool_events
    )
    identity_ok = all(event.idempotency_key for event in state.tool_events)
    return {
        "schema_round_trip": round_trip,
        "goal_preserved": state.goal == expected.goal,
        "plan_preserved": state.current_plan == expected.current_plan,
        "tool_result_lineage": lineage_ok,
        "side_effect_identity_preserved": identity_ok,
        "pending_effects_preserved": state.pending_effects == expected.pending_effects,
        "artifact_digest_preserved": state.artifact_digest == expected.artifact_digest,
        "verifier_status_preserved": state.verifier_status == expected.verifier_status,
        "budget_not_increased": state.remaining_budget <= expected.remaining_budget,
        "cut_marker_present": bool(state.cut_marker),
        "raw_history_is_distinct": not (
            canonical_event_ids & set(candidate.dropped_history_ids)
        ),
        "token_budget_reduced": (
            candidate.prompt_tokens_before > candidate.prompt_tokens_after > 0
        ),
    }


def sample_states() -> tuple[TaskState, TaskState]:
    expected = TaskState(
        schema_version=1,
        task_id="ml-infra-throughput-17",
        goal="improve throughput while preserving output correctness",
        current_plan=("inspect", "patch", "benchmark", "verify"),
        completed_steps=("inspect", "patch"),
        tool_events=(
            ToolEvent(
                call_id="call-read-1",
                tool_name="read_file",
                arguments_digest="sha256:args-read",
                result_digest="sha256:result-read",
                permission_scope="workspace:read",
                idempotency_key="read:src/loader.py",
                status="completed",
            ),
            ToolEvent(
                call_id="call-patch-1",
                tool_name="apply_patch",
                arguments_digest="sha256:args-patch",
                result_digest="sha256:result-patch",
                permission_scope="workspace:write",
                idempotency_key="patch:src/loader.py:v2",
                status="completed",
            ),
        ),
        pending_effects=("run:benchmark:baseline-v2",),
        artifact_digest="sha256:artifact-v2",
        verifier_status="pending",
        remaining_budget=37,
        cut_marker="after:call-patch-1",
    )
    compacted = TaskState(
        schema_version=expected.schema_version,
        task_id=expected.task_id,
        goal=expected.goal,
        current_plan=expected.current_plan,
        completed_steps=expected.completed_steps,
        tool_events=expected.tool_events,
        pending_effects=expected.pending_effects,
        artifact_digest=expected.artifact_digest,
        verifier_status=expected.verifier_status,
        remaining_budget=expected.remaining_budget,
        cut_marker="compacted:turn-42",
    )
    return expected, compacted


def main() -> None:
    expected, good_state = sample_states()
    good = CompactionCandidate(
        state=good_state,
        dropped_history_ids=("turn-1", "turn-2", "turn-3"),
        prompt_tokens_before=1200,
        prompt_tokens_after=420,
    )
    bad_state = TaskState(
        schema_version=1,
        task_id=expected.task_id,
        goal=expected.goal,
        current_plan=expected.current_plan,
        completed_steps=expected.completed_steps,
        tool_events=(
            ToolEvent(
                call_id="call-patch-1",
                tool_name="apply_patch",
                arguments_digest="sha256:args-patch",
                result_digest="",
                permission_scope="workspace:write",
                idempotency_key="",
                status="completed",
            ),
        ),
        pending_effects=(),
        artifact_digest="sha256:stale-artifact",
        verifier_status="unknown",
        remaining_budget=37,
        cut_marker="compacted:turn-42",
    )
    bad = CompactionCandidate(
        state=bad_state,
        dropped_history_ids=("call-patch-1",),
        prompt_tokens_before=1200,
        prompt_tokens_after=420,
    )
    good_checks = audit_candidate(good, expected)
    bad_checks = audit_candidate(bad, expected)
    assert all(good_checks.values())
    assert not all(bad_checks.values())
    print(
        json.dumps(
            {
                "evidence_level": "local_protocol_toy",
                "official_boundary": (
                    "GLM-5.3 compaction schema and implementation remain unverified"
                ),
                "good_candidate": good_checks,
                "unsafe_candidate": bad_checks,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
