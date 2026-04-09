"""Shared task score normalization helpers."""

from __future__ import annotations

from typing import Any


OPEN_INTERVAL_MIN = 0.01
OPEN_INTERVAL_MAX = 0.99


def open_interval_score(score: Any) -> float:
    """
    Normalize task scores to the strict-open interval required by validation.

    Output is always in [0.01, 0.99] and rounded for stable comparisons/logging.
    """
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        numeric = OPEN_INTERVAL_MIN

    clamped = min(OPEN_INTERVAL_MAX, max(OPEN_INTERVAL_MIN, numeric))
    return round(clamped, 4)
