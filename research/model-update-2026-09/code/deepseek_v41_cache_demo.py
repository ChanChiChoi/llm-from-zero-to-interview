"""Toy measurements for two DeepSeek V4.1-Flash serving questions.

This file deliberately does not load DeepSeek weights or implement its
production kernels.  It demonstrates two measurements that a real evaluation
must keep separate:

* candidate-pool recall versus final Top-K recall;
* grouped E2M1-like quantization error for a KV-cache toy vector.

The constants in ``main`` mirror the public reference configuration's shape
of the experiment, but the small synthetic scores and vectors are not model
outputs.  Do not use this script as a throughput, accuracy, or acceptance-rate
claim for DeepSeek V4.1-Flash.
"""

from dataclasses import dataclass
from math import isfinite


E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _require_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_finite_sequence(values, name):
    if not values:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must contain numbers")
        if not isfinite(value):
            raise ValueError(f"{name} must contain finite numbers")


def stable_top_indices(scores, limit):
    """Return indices sorted by descending score, then ascending index."""

    _require_finite_sequence(scores, "scores")
    _require_positive_int(limit, "limit")
    return tuple(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:limit])


def candidate_pool(scores, block_size, candidate_blocks):
    """Select positions in the highest-scoring blocks."""

    _require_finite_sequence(scores, "scores")
    _require_positive_int(block_size, "block_size")
    _require_positive_int(candidate_blocks, "candidate_blocks")
    blocks = [
        tuple(range(start, min(start + block_size, len(scores))))
        for start in range(0, len(scores), block_size)
    ]
    block_scores = [max(scores[position] for position in block) for block in blocks]
    selected_blocks = stable_top_indices(block_scores, min(candidate_blocks, len(blocks)))
    return tuple(position for block in selected_blocks for position in blocks[block])


def recall(retrieved, relevant):
    """Compute set recall and reject an undefined empty-evidence denominator."""

    relevant = set(relevant)
    if not relevant:
        raise ValueError("relevant evidence must not be empty")
    return len(set(retrieved) & relevant) / len(relevant)


@dataclass(frozen=True)
class RecallReport:
    candidate_positions: tuple
    final_positions: tuple
    candidate_recall: float
    conditional_topk_recall: float
    end_to_end_recall: float


def two_stage_recall(
    scores, relevant, block_size, candidate_blocks, final_topk
):
    """Measure candidate recall and Top-K recall as separate quantities."""

    _require_positive_int(final_topk, "final_topk")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("relevant evidence must not be empty")
    candidate_positions = candidate_pool(scores, block_size, candidate_blocks)
    final_candidates = tuple(
        stable_top_indices([scores[position] for position in candidate_positions],
                           min(final_topk, len(candidate_positions)))
    )
    final_positions = tuple(candidate_positions[index] for index in final_candidates)
    candidate_hits = len(set(candidate_positions) & relevant)
    final_hits = len(set(final_positions) & relevant)
    candidate_recall_value = candidate_hits / len(relevant)
    conditional = final_hits / candidate_hits if candidate_hits else 0.0
    return RecallReport(
        candidate_positions,
        final_positions,
        candidate_recall_value,
        conditional,
        final_hits / len(relevant),
    )


@dataclass(frozen=True)
class QuantizationReport:
    scales: tuple
    dequantized: tuple
    mse: float
    max_abs_error: float


def e2m1_like(values, group_size=16):
    """Quantize/dequantize a vector with a teaching-only E2M1-like codebook."""

    _require_finite_sequence(values, "values")
    _require_positive_int(group_size, "group_size")
    scales = []
    dequantized = []
    for start in range(0, len(values), group_size):
        group = values[start : start + group_size]
        max_abs = max(abs(value) for value in group)
        scale = max_abs / max(E2M1_LEVELS) if max_abs else 1.0
        scales.append(scale)
        for value in group:
            normalized = abs(value) / scale
            level = min(E2M1_LEVELS, key=lambda candidate: abs(candidate - normalized))
            dequantized.append((1.0 if value >= 0 else -1.0) * level * scale)
    errors = [actual - estimate for actual, estimate in zip(values, dequantized)]
    return QuantizationReport(
        tuple(scales),
        tuple(dequantized),
        sum(error * error for error in errors) / len(errors),
        max(abs(error) for error in errors),
    )


def main():
    # Synthetic indexer scores.  The values are intentionally small so that
    # the script is easy to inspect without NumPy or a GPU.
    scores = (
        0.90, 0.85, 0.30, 0.20,
        0.95, 0.40, 0.35, 0.25,
        0.88, 0.87, 0.20, 0.10,
        0.70, 0.20, 0.65, 0.10,
        0.86, 0.30, 0.25, 0.20,
        0.80, 0.79, 0.78, 0.10,
    )
    evidence = {1, 6, 9, 14, 22}
    report = two_stage_recall(
        scores,
        evidence,
        block_size=4,
        candidate_blocks=3,
        final_topk=5,
    )
    assert report.candidate_recall == 0.6
    assert report.end_to_end_recall == 0.4
    assert report.end_to_end_recall <= report.candidate_recall

    quantized = e2m1_like(
        (0.12, 1.2, -2.8, 5.6, 0.0, -0.75, 3.7, -6.0),
        group_size=4,
    )
    assert len(quantized.scales) == 2
    assert quantized.mse >= 0.0
    assert quantized.max_abs_error >= 0.0

    print("candidate recall:", f"{report.candidate_recall:.3f}")
    print("conditional Top-K recall:", f"{report.conditional_topk_recall:.3f}")
    print("end-to-end recall:", f"{report.end_to_end_recall:.3f}")
    print("toy FP4-like MSE:", f"{quantized.mse:.6f}")
    print("toy FP4-like max abs error:", f"{quantized.max_abs_error:.6f}")


if __name__ == "__main__":
    main()
