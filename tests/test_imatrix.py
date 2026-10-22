"""p6-03 AC2 gate: imatrix accumulator against the transcribed contract.

Contract source: docs/rea/imatrix_contract.md §1 (imatrix.cpp:655-664 verbatim).
Synthetic activation rows, no model files (C1 trivially satisfied).
"""
from __future__ import annotations

import numpy as np

from stratashift.meta.imatrix import (
    add_activation_rows,
    expert_usage,
    new_stats,
    normalized_imatrix,
)

NE0, NMAT = 4, 3


def test_accumulation_matches_transcribed_loop():
    """acc[j] += x[j]*x[j], act[j] += x[j], counts++ per batch (imatrix.cpp:658-663)."""
    stats = new_stats(NE0, NMAT)
    batch = np.array([[1.0, 2.0, 0.0, -1.0], [2.0, 0.0, 1.0, 1.0]])  # 2 rows one batch
    add_activation_rows(stats, mat_idx=1, x=batch)
    # column-wise: col0 1+4=5, col1 4+0=4, col2 0+1=1, col3 1+1=2
    assert np.allclose(stats["in_sum2"][1], [5.0, 4.0, 1.0, 2.0])
    assert np.allclose(stats["in_sum"][1], [3.0, 2.0, 1.0, 0.0])
    assert stats["counts"][1] == 1.0
    assert stats["counts"][0] == 0.0


def test_additive_across_chunks():
    """Streaming chunks add into the same planes — the C1 guarantee (unit-local
    accumulation, no full model)."""
    stats = new_stats(NE0, NMAT)
    for _ in range(5):
        add_activation_rows(stats, 2, np.ones((8, NE0)))
    # 5 batches × 8 rows × 1² = 40.0 per column; counts = 5 batches
    assert np.allclose(stats["in_sum2"][2], np.full(NE0, 40.0))
    assert stats["counts"][2] == 5.0


def test_width_guard():
    stats = new_stats(NE0, NMAT)
    try:
        add_activation_rows(stats, 0, np.ones((2, NE0 + 1)))
        raise AssertionError("width mismatch must raise")
    except ValueError:
        pass


def test_normalized_imatrix_mean_of_squares():
    stats = new_stats(NE0, NMAT)
    add_activation_rows(stats, 0, np.array([[3.0, 4.0, 0.0, 0.0]]))
    assert np.allclose(normalized_imatrix(stats, 0), [9.0, 16.0, 0.0, 0.0])
    # zero-count unit: ones, verbatim quantize.cpp:210-214 (uncalibrated expert = uniform)
    assert np.allclose(normalized_imatrix(stats, 1), np.ones(NE0))


def test_expert_usage_normalization():
    """arxiv 2406.08155 §3.3: normalize(routing counts)."""
    counts = np.array([10.0, 30.0, 60.0])
    assert np.allclose(expert_usage(counts), [0.1, 0.3, 0.6])
    assert np.allclose(expert_usage(np.zeros(3)), np.zeros(3))
