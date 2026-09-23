from __future__ import annotations

from typing import Iterable


def _unique_sorted(values: Iterable[float], *, epsilon: float = 1e-4) -> list[float]:
    rows: list[float] = []
    for raw in sorted(float(value) for value in values):
        if not rows or abs(raw - rows[-1]) > epsilon:
            rows.append(raw)
    return rows


def _evenly_limit(values: list[float], limit: int) -> list[float]:
    limit = max(0, int(limit))
    if limit == 0 or not values:
        return []
    if len(values) <= limit:
        return list(values)
    if limit == 1:
        return [values[len(values) // 2]]
    indices = {
        round(index * (len(values) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [values[index] for index in sorted(indices)]


def merge_mandatory_timestamps(
    base_timestamps: Iterable[float],
    mandatory_timestamps: Iterable[float] | None,
    *,
    start_time: float,
    end_time: float,
    budget: int,
) -> list[float]:
    """Reserve fixed-budget sampling slots for upstream temporal anchors.

    The mandatory timestamps replace uniformly sampled frames; they never
    increase the frame budget. Remaining slots retain broad temporal coverage.
    """
    budget = max(1, int(budget))
    start_time = float(start_time)
    end_time = float(end_time)
    mandatory = _unique_sorted(
        value
        for value in (mandatory_timestamps or [])
        if start_time <= float(value) <= end_time
    )
    mandatory = _evenly_limit(mandatory, budget)

    base = _unique_sorted(
        value
        for value in base_timestamps
        if start_time <= float(value) <= end_time
        and all(abs(float(value) - anchor) > 1e-4 for anchor in mandatory)
    )
    remaining = budget - len(mandatory)
    selected = mandatory + _evenly_limit(base, remaining)
    return _unique_sorted(selected)[:budget]


def supplement_frame_indices(
    selected: Iterable[int],
    fallback: Iterable[int],
    *,
    budget: int,
) -> list[int]:
    """Fill a de-duplicated timestamp conversion without dropping anchors."""
    budget = max(1, int(budget))
    rows = list(dict.fromkeys(int(value) for value in selected))
    for value in fallback:
        index = int(value)
        if index not in rows:
            rows.append(index)
        if len(rows) >= budget:
            break
    return sorted(rows[:budget])
