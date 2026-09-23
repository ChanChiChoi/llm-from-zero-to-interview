#!/usr/bin/env python3
"""Run a zero-dependency SAO teaching toy without network or model weights.

The toy isolates four ideas discussed in arXiv:2607.07508:

* single-rollout completion versus a group barrier;
* token-level ratio clipping/masking for rollout-policy drift;
* repeated value updates as a critic-stability proxy;
* GAE that skips environment-observation tokens.

It is deliberately not a reproduction of the paper, GLM-5.2, or GLM-5.3.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Rollout:
    prompt_id: str
    duration: int


def compare_rollout_schedules(rollouts: Iterable[Rollout]) -> dict[str, Any]:
    """Compare a synchronous group barrier with immediate single-rollout use."""

    items = tuple(rollouts)
    if not items:
        raise ValueError("at least one rollout is required")
    barrier = max(item.duration for item in items)
    barrier_wait = {item.prompt_id: barrier - item.duration for item in items}
    completion_order = [
        item.prompt_id
        for item in sorted(items, key=lambda item: (item.duration, item.prompt_id))
    ]
    return {
        "group_barrier_time": barrier,
        "group_barrier_wait_by_prompt": barrier_wait,
        "group_barrier_wait_total": sum(barrier_wait.values()),
        "single_rollout_completion_order": completion_order,
        "single_rollout_wait_total": 0,
        "single_rollout_updates": len(items),
    }


@dataclass(frozen=True)
class RatioDecision:
    rollout_logprob: float
    current_logprob: float
    ratio: float
    lower_bound: float
    upper_bound: float
    effective_ratio: float | None
    masked: bool


def dis_decision(
    rollout_logprob: float,
    current_logprob: float,
    *,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
) -> RatioDecision:
    """Apply a teaching version of double-sided ratio clipping/masking."""

    if epsilon_low < 0 or epsilon_high < 0:
        raise ValueError("epsilon values must be non-negative")
    ratio = math.exp(current_logprob - rollout_logprob)
    lower_bound = 1.0 - epsilon_low
    upper_bound = 1.0 + epsilon_high
    masked = ratio < lower_bound or ratio > upper_bound
    clipped = min(max(ratio, lower_bound), upper_bound)
    return RatioDecision(
        rollout_logprob=rollout_logprob,
        current_logprob=current_logprob,
        ratio=ratio,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        effective_ratio=None if masked else clipped,
        masked=masked,
    )


@dataclass(frozen=True)
class ActionSegment:
    segment_id: str
    reward: float
    value_at_start: float
    next_action_value: float
    observation_values: tuple[float, ...]


def skip_observation_gae(
    segments: Iterable[ActionSegment],
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> dict[str, float]:
    """Compute segment-level GAE while skipping observation tokens."""

    items = tuple(segments)
    if not items:
        raise ValueError("at least one action segment is required")
    advantages: dict[str, float] = {}
    next_advantage = 0.0
    for segment in reversed(items):
        delta = (
            segment.reward
            + gamma * segment.next_action_value
            - segment.value_at_start
        )
        advantage = delta + gamma * lam * next_advantage
        advantages[segment.segment_id] = advantage
        next_advantage = advantage
    return {segment.segment_id: advantages[segment.segment_id] for segment in items}


@dataclass(frozen=True)
class TokenTransition:
    current_value: float
    next_value: float
    reward: float


def ordinary_token_gae(
    transitions: Iterable[TokenTransition],
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> list[float]:
    """A contrastive token-level GAE that treats observations as transitions."""

    items = tuple(transitions)
    advantages: list[float] = [0.0] * len(items)
    next_advantage = 0.0
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        delta = item.reward + gamma * item.next_value - item.current_value
        next_advantage = delta + gamma * lam * next_advantage
        advantages[index] = next_advantage
    return advantages


def expand_as_token_transitions(
    segments: Iterable[ActionSegment],
) -> list[TokenTransition]:
    """Expand segments so observation length can affect ordinary GAE."""

    transitions: list[TokenTransition] = []
    items = tuple(segments)
    for segment in items:
        first_next = (
            segment.observation_values[0]
            if segment.observation_values
            else segment.next_action_value
        )
        transitions.append(
            TokenTransition(
                current_value=segment.value_at_start,
                next_value=first_next,
                reward=segment.reward,
            )
        )
        observations = segment.observation_values
        for index, value in enumerate(observations):
            next_value = (
                observations[index + 1]
                if index + 1 < len(observations)
                else segment.next_action_value
            )
            transitions.append(
                TokenTransition(current_value=value, next_value=next_value, reward=0.0)
            )
    return transitions


def critic_update_proxy(
    initial_prediction: float,
    target: float,
    updates: int,
) -> dict[str, Any]:
    """Use a deterministic moving-average proxy for critic update count."""

    if updates <= 0:
        raise ValueError("updates must be positive")
    prediction = initial_prediction
    losses: list[float] = []
    for _ in range(updates):
        error = target - prediction
        losses.append(error * error)
        prediction += 0.5 * error
    return {
        "updates": updates,
        "losses": losses,
        "final_prediction": prediction,
        "final_squared_error": (target - prediction) ** 2,
    }


def _segments(observation_values: tuple[float, ...]) -> tuple[ActionSegment, ...]:
    return (
        ActionSegment(
            segment_id="a0",
            reward=0.2,
            value_at_start=0.1,
            next_action_value=0.4,
            observation_values=observation_values,
        ),
        ActionSegment(
            segment_id="a1",
            reward=1.0,
            value_at_start=0.4,
            next_action_value=0.0,
            observation_values=(),
        ),
    )


def run_audit() -> dict[str, Any]:
    schedule = compare_rollout_schedules(
        (
            Rollout("p0", 2),
            Rollout("p1", 5),
            Rollout("p2", 9),
        )
    )
    ratios = [
        dis_decision(0.0, 0.0),
        dis_decision(0.0, math.log(0.1)),
        dis_decision(0.0, math.log(5.0)),
        dis_decision(0.0, math.log(1.1)),
    ]
    short_segments = _segments((0.8,))
    long_segments = _segments((0.8, 0.7, 0.6))
    skip_short = skip_observation_gae(short_segments)
    skip_long = skip_observation_gae(long_segments)
    ordinary_short = ordinary_token_gae(expand_as_token_transitions(short_segments))
    ordinary_long = ordinary_token_gae(expand_as_token_transitions(long_segments))
    critic_k1 = critic_update_proxy(0.0, 1.0, 1)
    critic_k2 = critic_update_proxy(0.0, 1.0, 2)

    if schedule["group_barrier_wait_total"] <= 0:
        raise AssertionError("the schedule must contain a straggler")
    if schedule["single_rollout_wait_total"] != 0:
        raise AssertionError("single-rollout toy should not wait for a group barrier")
    if not ratios[1].masked or not ratios[2].masked:
        raise AssertionError("extreme policy ratios must be masked")
    if ratios[0].masked or ratios[3].masked:
        raise AssertionError("in-bound policy ratios must be kept")
    if not all(
        math.isclose(skip_short[key], skip_long[key], rel_tol=0.0, abs_tol=1e-12)
        for key in skip_short
    ):
        raise AssertionError("skip-observation GAE changed with observation length")
    if math.isclose(ordinary_short[0], ordinary_long[0], rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError("ordinary token GAE should expose observation-length effects")
    if critic_k2["final_squared_error"] >= critic_k1["final_squared_error"]:
        raise AssertionError("the K=2 critic proxy should reduce final error")

    return {
        "evidence_level": "local_protocol_toy",
        "network_called": False,
        "full_weights_loaded": False,
        "paper_reproduced": False,
        "schedule": schedule,
        "dis": [asdict(item) for item in ratios],
        "skip_observation_gae": {
            "short_observation_tokens": 1,
            "long_observation_tokens": 3,
            "short": skip_short,
            "long": skip_long,
            "ordinary_short_first_advantage": ordinary_short[0],
            "ordinary_long_first_advantage": ordinary_long[0],
        },
        "critic_update_proxy": {"k1": critic_k1, "k2": critic_k2},
        "limitations": [
            "Synthetic log-probabilities, values, rewards and durations only.",
            "No GLM-5.2/GLM-5.3 weights, rollout engine, verifier or hardware.",
            "Not a reproduction of SAO paper benchmarks or product compaction.",
        ],
    }


def main() -> int:
    print(json.dumps(run_audit(), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
