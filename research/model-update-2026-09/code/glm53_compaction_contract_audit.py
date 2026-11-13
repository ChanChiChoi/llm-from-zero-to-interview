"""A generic, dependency-free audit toy for compacted task state.

This is not GLM-5.3 code. It checks whether a saved state preserves enough
contract fields to resume safely across three tool-event cut points.
Run with: python3 research/model-update-2026-09/code/glm53_compaction_contract_audit.py
"""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256


def digest(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + sha256(payload).hexdigest()


def audit(state: object) -> tuple[bool, str]:
    if not isinstance(state, dict):
        return False, "state must be an object"

    required = {
        "schema_version",
        "task_id",
        "goal",
        "current_plan",
        "completed_steps",
        "permission_scope",
        "tool_calls",
        "pending_effects",
        "artifact_digest",
        "verifier_status",
        "remaining_budget",
        "cut_marker",
    }
    missing = sorted(required - state.keys())
    if missing:
        return False, "missing required fields: " + ", ".join(missing)
    if state["schema_version"] != 1:
        return False, "unsupported schema_version"
    if not isinstance(state["task_id"], str) or not state["task_id"]:
        return False, "task_id must be a non-empty string"
    if not isinstance(state["goal"], str) or not state["goal"]:
        return False, "goal must be a non-empty string"
    if not isinstance(state["current_plan"], list) or not all(
        isinstance(step, str) for step in state["current_plan"]
    ):
        return False, "current_plan must be a list of strings"
    if not isinstance(state["completed_steps"], list) or not all(
        isinstance(step, str) for step in state["completed_steps"]
    ):
        return False, "completed_steps must be a list of strings"
    if not isinstance(state["permission_scope"], list) or not state["permission_scope"] or not all(
        isinstance(scope, str) and scope for scope in state["permission_scope"]
    ):
        return False, "permission_scope must contain non-empty strings"
    if state["verifier_status"] not in {"pending", "passed", "failed"}:
        return False, "invalid verifier_status"
    if state["verifier_status"] == "passed" and not state["artifact_digest"]:
        return False, "passed verifier requires artifact_digest"
    if state["artifact_digest"] is not None and not (
        isinstance(state["artifact_digest"], str)
        and state["artifact_digest"].startswith("sha256:")
    ):
        return False, "artifact_digest must be null or sha256-prefixed"
    if not isinstance(state["remaining_budget"], dict) or not state["remaining_budget"]:
        return False, "remaining_budget must be a non-empty object"
    if any(type(value) is not int or value < 0 for value in state["remaining_budget"].values()):
        return False, "remaining budget values must be non-negative integers"
    if not isinstance(state["cut_marker"], str) or not state["cut_marker"]:
        return False, "cut_marker must be a non-empty string"

    calls = state["tool_calls"]
    effects = state["pending_effects"]
    if not isinstance(calls, list) or not isinstance(effects, list):
        return False, "tool_calls and pending_effects must be lists"
    call_ids: set[str] = set()
    pending_by_id: dict[str, dict] = {}
    for effect in effects:
        if not isinstance(effect, dict):
            return False, "each pending effect must be an object"
        call_id = effect.get("call_id")
        key = effect.get("idempotency_key")
        if not isinstance(call_id, str) or not call_id or not isinstance(key, str) or not key:
            return False, "pending effect requires call_id and idempotency_key"
        if effect.get("resolution") not in {"pending", "unknown"}:
            return False, "pending effect resolution must be pending or unknown"
        if call_id in pending_by_id:
            return False, "duplicate pending effect: " + call_id
        pending_by_id[call_id] = effect

    for call in calls:
        if not isinstance(call, dict):
            return False, "each tool call must be an object"
        call_id = call.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return False, "tool call requires call_id"
        if call_id in call_ids:
            return False, "duplicate tool call: " + call_id
        call_ids.add(call_id)
        if not isinstance(call.get("tool_name"), str) or not call["tool_name"]:
            return False, "tool call requires tool_name"
        key = call.get("idempotency_key")
        if not isinstance(key, str) or not key:
            return False, "tool call requires idempotency_key"
        status = call.get("status")
        if status not in {"started", "unknown", "completed", "rejected"}:
            return False, "invalid tool-call status: " + str(status)
        effect = pending_by_id.get(call_id)
        if status in {"started", "unknown"}:
            if effect is None:
                return False, "unresolved call lacks pending effect: " + call_id
            if effect["idempotency_key"] != key:
                return False, "pending effect idempotency key mismatch: " + call_id
        elif effect is not None:
            return False, "resolved call still has pending effect: " + call_id
        if status == "completed" and not (
            isinstance(call.get("result_digest"), str)
            and call["result_digest"].startswith("sha256:")
        ):
            return False, "completed call requires result_digest: " + call_id

    orphan_effects = pending_by_id.keys() - call_ids
    if orphan_effects:
        return False, "pending effect has no matching tool call: " + sorted(orphan_effects)[0]
    return True, "accepted"


def make_state(cut: str) -> dict:
    state = {
        "schema_version": 1,
        "task_id": "task-demo-17",
        "goal": "修复请求重试后重复写入的问题",
        "current_plan": ["核对调用日志", "实现幂等写入", "运行回归验证"],
        "completed_steps": ["确认重试发生在提交确认丢失之后"],
        "permission_scope": ["records:write:demo"],
        "tool_calls": [],
        "pending_effects": [],
        "artifact_digest": None,
        "verifier_status": "pending",
        "remaining_budget": {"tool_calls": 3, "tokens": 2400},
        "cut_marker": cut,
    }
    call = {
        "call_id": "call-17",
        "tool_name": "write_record",
        "idempotency_key": "task-demo-17:call-17",
        "status": "started",
        "result_digest": None,
    }
    effect = {
        "call_id": "call-17",
        "idempotency_key": call["idempotency_key"],
        "resolution": "unknown",
    }
    if cut == "before_tool_call":
        return state
    state["tool_calls"] = [call]
    if cut == "between_call_and_result":
        state["pending_effects"] = [effect]
        return state
    if cut == "after_tool_result":
        completed_call = dict(call, status="completed", result_digest=digest({"rows_written": 1}))
        state["tool_calls"] = [completed_call]
        state["artifact_digest"] = digest({"revision": "r2", "tests": "passed"})
        state["verifier_status"] = "passed"
        state["remaining_budget"] = {"tool_calls": 2, "tokens": 1900}
        return state
    raise ValueError("unknown cut: " + cut)


def main() -> None:
    cut_results = {
        cut: audit(make_state(cut))[1]
        for cut in ("before_tool_call", "between_call_and_result", "after_tool_result")
    }

    lossy = deepcopy(make_state("between_call_and_result"))
    lossy["pending_effects"] = []
    lossy_result = audit(lossy)

    orphan = deepcopy(make_state("before_tool_call"))
    orphan["pending_effects"] = [{
        "call_id": "missing-call",
        "idempotency_key": "task-demo-17:missing-call",
        "resolution": "unknown",
    }]
    orphan_result = audit(orphan)

    print(json.dumps({
        "cut_points": cut_results,
        "lossy_candidate": lossy_result[1],
        "orphan_effect_candidate": orphan_result[1],
        "evidence_scope": "generic_protocol_toy_not_glm53_implementation",
    }, ensure_ascii=False, sort_keys=True))

    assert all(value == "accepted" for value in cut_results.values())
    assert not lossy_result[0] and "lacks pending effect" in lossy_result[1]
    assert not orphan_result[0] and "no matching tool call" in orphan_result[1]


if __name__ == "__main__":
    main()
