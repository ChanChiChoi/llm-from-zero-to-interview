"""Deterministic, dependency-free SAO mechanism toy; not a paper reproduction.

Shows straggler wait, a clipped importance-ratio mask, action-only GAE, a K=2
scalar critic proxy, and trajectory lineage. It uses no model, network, or GPU.
Run with: python3 research/model-update-2026-09/code/sao_async_rl_toy.py
"""

from __future__ import annotations

import json
import math


def group_barrier_wait(latencies: list[float]) -> float:
    return sum(max(latencies) - latency for latency in latencies)


def clipped_ratio_mask(old_logp: list[float], current_logp: list[float], low: float, high: float):
    ratios = [math.exp(current - old) for old, current in zip(old_logp, current_logp)]
    lower, upper = 1.0 - low, 1.0 + high
    clipped = [min(max(ratio, lower), upper) for ratio in ratios]
    inside = [lower <= ratio <= upper for ratio in ratios]
    return ratios, clipped, inside


def action_only_gae(rewards: list[float], values: list[float], gamma: float, lam: float):
    advantages = [0.0] * len(rewards)
    next_advantage = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        next_advantage = delta + gamma * lam * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages


def gae_from_segments(segments: list[dict], gamma: float, lam: float):
    actions = [segment["action"] for segment in segments]
    return action_only_gae(
        [action["reward"] for action in actions],
        [action["value"] for action in actions],
        gamma,
        lam,
    )


def critic_proxy(target: float, learning_rate: float, updates: int):
    prediction = 0.0
    losses = []
    for _ in range(updates):
        error = prediction - target
        losses.append(error * error)
        prediction -= learning_rate * 2.0 * error
    return prediction, losses


def main() -> None:
    # Durations are constructed seconds for one three-sample group.
    latencies = [1.0, 3.0, 2.0]
    barrier_wait = group_barrier_wait(latencies)
    completion_order = [name for _, name in sorted(zip(latencies, ["r1", "r2", "r3"]))]

    ratios, clipped_ratios, ratio_mask = clipped_ratio_mask(
        old_logp=[-0.10, -0.20, -0.30, -0.40],
        current_logp=[-0.08, -0.50, -0.05, -0.40],
        low=0.20,
        high=0.20,
    )

    short_trace = [
        {"action": {"reward": 0.0, "value": 0.2}, "observation_tokens": ["ok"]},
        {"action": {"reward": 1.0, "value": 0.4}, "observation_tokens": []},
    ]
    long_trace = [dict(segment) for segment in short_trace]
    long_trace[0]["observation_tokens"] = ["environment"] * 100
    gae_without_observation = gae_from_segments(short_trace, gamma=0.9, lam=0.8)
    gae_with_long_observation = gae_from_segments(long_trace, gamma=0.9, lam=0.8)

    critic_prediction, critic_losses = critic_proxy(target=1.0, learning_rate=0.1, updates=2)
    report = {
        "group_barrier_wait_total_seconds": round(barrier_wait, 3),
        "async_completion_order": completion_order,
        "importance_ratios": [round(value, 4) for value in ratios],
        "ratios_after_symmetric_clip_0_2": [round(value, 4) for value in clipped_ratios],
        "ratio_inside_clip_mask": ratio_mask,
        "action_only_gae": [round(value, 4) for value in gae_without_observation],
        "same_gae_with_longer_observation": gae_with_long_observation == gae_without_observation,
        "critic_updates_K": 2,
        "critic_prediction_after_K": round(critic_prediction, 4),
        "critic_mse_by_update": [round(value, 4) for value in critic_losses],
        "trajectory_lineage": {"prompt_id": "p1", "rollout_id": "r2", "segments": [0, 1]},
        "glm53_compaction": "unverified",
        "evidence_scope": "generic_mechanism_toy_not_sao_or_glm53_reproduction",
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))

    assert barrier_wait == 3.0
    assert completion_order == ["r1", "r3", "r2"]
    assert ratio_mask == [True, False, False, True]
    assert gae_with_long_observation == gae_without_observation
    assert len(critic_losses) == 2 and math.isclose(critic_prediction, 0.36)
    assert report["glm53_compaction"] == "unverified"


if __name__ == "__main__":
    main()
