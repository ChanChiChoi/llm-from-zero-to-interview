#!/usr/bin/env python3
"""Zero-dependency toy for Gemini Interactions state and tool replay.

This is a protocol exercise only. It does not call Gemini, contact the
network, or claim to reproduce provider internals.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional


class ReplayError(ValueError):
    """Raised when a saved interaction cannot be replayed safely."""


@dataclass
class Step:
    step_type: str
    data: Dict[str, Any]
    step_id: Optional[str] = None
    signature: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": self.step_type, **self.data}
        if self.step_id is not None:
            result["id"] = self.step_id
        if self.signature is not None:
            result["signature"] = self.signature
        return result


@dataclass
class Interaction:
    interaction_id: str
    model: str
    stored: bool
    input_text: str
    output_steps: List[Step]
    history: List[Step]
    generation_config: Dict[str, Any]
    tools: List[str]
    status: str = "completed"


def _signature(interaction_id: str, label: str) -> str:
    digest = hashlib.sha256(f"{interaction_id}:{label}".encode("ascii")).hexdigest()
    return f"opaque:{digest[:20]}"


def _step_copy(steps: Iterable[Step]) -> List[Step]:
    return [deepcopy(step) for step in steps]


def validate_history(history: List[Step]) -> None:
    """Validate the fields a stateless client must preserve."""

    calls: Dict[str, Step] = {}
    for step in history:
        if step.step_type == "thought" and not step.signature:
            raise ReplayError("thought signature is missing")
        if step.step_type in {"function_call", "function_response"}:
            if not step.step_id:
                raise ReplayError(f"{step.step_type} id is missing")
            if not step.signature:
                raise ReplayError(f"{step.step_type} signature is missing")
        if step.step_type == "function_call":
            if step.step_id in calls:
                raise ReplayError(f"duplicate function call id: {step.step_id}")
            calls[step.step_id] = step
        elif step.step_type == "function_response":
            if step.step_id not in calls:
                raise ReplayError(f"orphan function response: {step.step_id}")


class InteractionStore:
    """Small in-memory model of stored Interactions and continuation."""

    def __init__(self) -> None:
        self._items: Dict[str, Interaction] = {}
        self._known_history_signatures: List[tuple] = []
        self._next_id = 1

    @property
    def stored_ids(self) -> List[str]:
        return sorted(self._items)

    def create(
        self,
        *,
        input_text: str,
        model: str = "gemini-3.8-flash",
        store: bool = True,
        previous_interaction_id: Optional[str] = None,
        stateless_history: Optional[List[Step]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
    ) -> Interaction:
        if previous_interaction_id and not store:
            raise ReplayError("store=false cannot be continued with previous_interaction_id")
        if previous_interaction_id and stateless_history is not None:
            raise ReplayError("stateful and stateless history cannot be mixed")
        if previous_interaction_id not in {None, *self._items}:
            raise ReplayError(f"unknown previous interaction: {previous_interaction_id}")

        parent_history: List[Step] = []
        if previous_interaction_id:
            parent_history = _step_copy(self._items[previous_interaction_id].history)
        elif stateless_history is not None:
            validate_history(stateless_history)
            signatures = tuple(
                step.signature for step in stateless_history if step.signature is not None
            )
            if signatures not in self._known_history_signatures:
                raise ReplayError("stateless history signature mismatch")
            parent_history = _step_copy(stateless_history)

        interaction_id = f"ix_{self._next_id:03d}"
        self._next_id += 1
        config = dict(generation_config or {})
        tool_names = list(tools or ["get_weather"])

        user_step = Step("user_input", {"text": input_text})
        thought = Step(
            "thought",
            {"summary": "plan tool call and verify its result"},
            signature=_signature(interaction_id, "thought"),
        )
        call_id = f"call_{interaction_id}"
        function_call = Step(
            "function_call",
            {"name": tool_names[0], "arguments": {"city": "Beijing"}},
            step_id=call_id,
            signature=_signature(interaction_id, "function_call"),
        )
        function_response = Step(
            "function_response",
            {"name": tool_names[0], "result": {"temperature_c": 25}},
            step_id=call_id,
            signature=_signature(interaction_id, "function_response"),
        )
        model_output = Step(
            "model_output",
            {"text": "Beijing is 25 C in this synthetic trace."},
        )
        output_steps = [thought, function_call, function_response, model_output]
        history = parent_history + [user_step] + _step_copy(output_steps)
        validate_history(history)

        interaction = Interaction(
            interaction_id=interaction_id,
            model=model,
            stored=store,
            input_text=input_text,
            output_steps=output_steps,
            history=history,
            generation_config=config,
            tools=tool_names,
        )
        if store:
            self._items[interaction_id] = interaction
            self._known_history_signatures.append(
                tuple(step.signature for step in history if step.signature is not None)
            )
        return interaction

    def get(self, interaction_id: str) -> Interaction:
        if interaction_id not in self._items:
            raise ReplayError(f"interaction is not stored: {interaction_id}")
        return self._items[interaction_id]

    def delete(self, interaction_id: str) -> None:
        if interaction_id not in self._items:
            raise ReplayError(f"interaction is not stored: {interaction_id}")
        del self._items[interaction_id]


def stream_events(interaction: Interaction) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = [
        {"event_type": "interaction.created", "interaction_id": interaction.interaction_id}
    ]
    for index, step in enumerate(interaction.output_steps):
        events.append({"event_type": "step.start", "index": index, "type": step.step_type})
        events.append({"event_type": "step.delta", "index": index, "data": step.as_dict()})
        events.append({"event_type": "step.stop", "index": index, "type": step.step_type})
    events.append(
        {
            "event_type": "interaction.completed",
            "interaction_id": interaction.interaction_id,
            "status": interaction.status,
        }
    )
    events.append({"event_type": "done"})
    return events


def retention_days(tier: str) -> int:
    if tier == "paid":
        return 55
    if tier == "free":
        return 1
    raise ReplayError(f"unknown tier: {tier}")


def assert_modality_compatible(previous_output: str, next_input: List[str]) -> None:
    if previous_output == "image" and "image" not in next_input:
        raise ReplayError("next model cannot consume the previous image modality")


def expect_error(action: Any, fragment: str) -> None:
    try:
        action()
    except ReplayError as exc:
        if fragment not in str(exc):
            raise AssertionError(f"wrong error: {exc}") from exc
    else:
        raise AssertionError(f"expected ReplayError containing {fragment!r}")


def main() -> None:
    store = InteractionStore()
    first = store.create(
        input_text="Check the weather.",
        generation_config={"thinking_level": "low"},
    )
    events = stream_events(first)
    assert events[0]["event_type"] == "interaction.created"
    assert events[-2]["event_type"] == "interaction.completed"
    assert events[-1]["event_type"] == "done"
    assert [event["event_type"] for event in events].index("step.start") < (
        [event["event_type"] for event in events].index("interaction.completed")
    )

    second = store.create(
        input_text="Now summarize it.",
        previous_interaction_id=first.interaction_id,
        generation_config={"thinking_level": "high"},
        tools=["summarize_weather"],
    )
    assert len(second.history) > len(first.history)
    assert second.generation_config["thinking_level"] == "high"
    assert first.generation_config["thinking_level"] == "low"
    assert second.history[0].step_type == "user_input"

    stateless_history = _step_copy(first.history)
    stateless = store.create(
        input_text="Replay without server state.",
        store=False,
        stateless_history=stateless_history,
    )
    assert not stateless.stored
    assert stateless.interaction_id not in store.stored_ids

    tampered = _step_copy(first.history)
    tampered[1].signature = "opaque:tampered"
    expect_error(
        lambda: store.create(
            input_text="Reject modified history.",
            store=False,
            stateless_history=tampered,
        ),
        "signature mismatch",
    )
    # The first history element is user input; the second is the thought.
    tampered = _step_copy(first.history)
    tampered[1].signature = None
    expect_error(
        lambda: store.create(
            input_text="Reject missing signature.",
            store=False,
            stateless_history=tampered,
        ),
        "thought signature",
    )

    expect_error(
        lambda: store.create(
            input_text="Cannot chain an unstored turn.",
            store=False,
            previous_interaction_id=first.interaction_id,
        ),
        "store=false",
    )
    store.delete(first.interaction_id)
    expect_error(
        lambda: store.create(
            input_text="Cannot continue deleted state.",
            previous_interaction_id=first.interaction_id,
        ),
        "unknown previous",
    )

    assert retention_days("paid") == 55
    assert retention_days("free") == 1
    assert_modality_compatible("text", ["text"])
    expect_error(
        lambda: assert_modality_compatible("image", ["text"]),
        "image modality",
    )

    result = {
        "ok": True,
        "stateful_parent_history_preserved": True,
        "stateless_history_signature_preserved": True,
        "tool_call_result_id_aligned": True,
        "sse_event_order": [event["event_type"] for event in events],
        "paid_retention_days": retention_days("paid"),
        "free_retention_days": retention_days("free"),
        "stored_ids_after_delete": store.stored_ids,
        "network_called": False,
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
