"""Calendar exponential sample weights for trainers that equal-weight history.

``TrainWindow.time_decay`` is the shared knob. LightGBM reads it; Dixon–Coles
keeps its own ``xi`` and must not also apply this field (that would double-decay).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np


def calendar_sample_weights(
    event_times: np.ndarray,
    as_of: datetime,
    time_decay: float | None,
) -> np.ndarray:
    """Return ``exp(-λ × days_before_as_of)`` weights, or ones when λ is unset.

    Days are measured from each event timestamp to ``as_of`` (the PIT cutoff),
    clamped at zero so an event at or after the cutoff still gets weight 1.

    Args:
        event_times: Event timestamps, one per training row. Typically fixture
            ``kickoff_time`` as a numpy datetime64 array.
        as_of: Point-in-time cutoff (usually ``PITView.cutoff`` / deadline).
        time_decay: Decay rate λ in inverse days. ``None`` disables decay
            (all weights 1). Must be non-negative when set.

    Returns:
        Float64 weights of length ``len(event_times)``.

    Raises:
        ValueError: ``time_decay`` is negative.
    """
    events = np.asarray(event_times)
    n = int(events.shape[0])
    if time_decay is None:
        return np.ones(n, dtype=np.float64)
    if time_decay < 0:
        raise ValueError(f"time_decay must be non-negative, got {time_decay}")
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    cutoff = np.datetime64(_naive_utc(as_of))
    # Match Dixon–Coles: timedelta64 division → float days before as_of.
    delta = (cutoff - events.astype("datetime64[ns]")).astype("timedelta64[ns]")
    days = np.asarray(delta / np.timedelta64(1, "D"), dtype=np.float64)
    days = np.maximum(days, 0.0)
    return np.exp(-float(time_decay) * days)


def _naive_utc(value: datetime) -> datetime:
    """Drop tzinfo so numpy datetime64 math matches polars UTC columns."""
    if value.tzinfo is None:
        return value
    return value.replace(tzinfo=None)
