from __future__ import annotations

from typing import Any


_SOURCE_PRIORITY = {
    "frame_verify": 5.0,
    "focus": 4.0,
    "localize_qwen": 3.0,
    "focus_qwen": 2.5,
    "skim_qwen": 2.0,
    "overview": 1.0,
}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_window(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start = _safe_float(value[0])
    end = _safe_float(value[1])
    if start is None or end is None or end <= start:
        return None
    return [start, end]


def _anchor_score(row: dict[str, Any]) -> float:
    source_tool = str(row.get("source_tool") or "")
    score = _SOURCE_PRIORITY.get(source_tool, 0.5)
    confidence = str(row.get("confidence") or "").strip().lower()
    score += {"high": 2.0, "medium": 1.0, "routing_only": 0.5}.get(
        confidence, 0.0
    )
    if str(row.get("target_match") or "").strip().lower() == "matched":
        score += 2.0
    if str(row.get("event_match") or "").strip().lower() == "direct":
        score += 2.0
    if row.get("needs_focus"):
        score += 0.75
    if row.get("supports_options"):
        score += 0.75
    return score


def _deduplicate_anchor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (-float(item["score"]), item["timestamp_s"])):
        duplicate = next(
            (
                item
                for item in selected
                if abs(float(item["timestamp_s"]) - float(row["timestamp_s"])) <= 0.15
            ),
            None,
        )
        if duplicate is None:
            selected.append(row)
    return sorted(selected, key=lambda item: float(item["timestamp_s"]))


def _limit_anchor_rows(
    rows: list[dict[str, Any]],
    *,
    window: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    """Keep strong anchors while retaining temporal coverage within the window."""
    limit = max(1, int(limit))
    if len(rows) <= limit:
        return rows

    start, end = window
    center = (start + end) / 2.0
    # Anchors just outside a requested boundary are especially easy to lose in
    # the planner-to-tool handoff, so reserve them before filling by strength.
    ordered = sorted(
        rows,
        key=lambda item: (
            0 if not start <= float(item["timestamp_s"]) <= end else 1,
            -float(item["score"]),
            abs(float(item["timestamp_s"]) - center),
        ),
    )
    kept = ordered[:limit]
    return sorted(kept, key=lambda item: float(item["timestamp_s"]))


def collect_memory_anchor_rows(
    memory: dict[str, Any] | None,
    windows: list[list[float]],
    *,
    margin_s: float = 0.0,
    per_window: int = 4,
) -> list[list[dict[str, Any]]]:
    """Return exact prior observation timestamps for each requested window.

    This is a lossless tool handoff, not a routing policy: it only reuses
    timestamps already present in the trajectory and never creates evidence.
    """
    if not memory:
        return [[] for _ in windows]

    source_rows: list[dict[str, Any]] = []
    for item in memory.get("timestamped_observations") or []:
        if not isinstance(item, dict):
            continue
        timestamp = _safe_float(item.get("timestamp_s"))
        if timestamp is None or "qwen_omitted_frame" in (item.get("event_tags") or []):
            continue
        row = dict(item)
        row["timestamp_s"] = round(timestamp, 3)
        row["score"] = _anchor_score(row)
        source_rows.append(row)

    for item in memory.get("candidate_binding_memory") or []:
        if not isinstance(item, dict):
            continue
        timestamp = _safe_float(item.get("best_timestamp_s"))
        if timestamp is None:
            continue
        row = {
            "timestamp_s": round(timestamp, 3),
            "source_tool": item.get("source_tool") or "frame_verify",
            "confidence": "high",
            "target_match": item.get("target_match") or "unknown",
            "event_match": item.get("event_match") or "unknown",
            "description": item.get("observed_fact") or "",
            "supports_options": item.get("supports_options") or [],
            "anchor_ref": item.get("id") or "",
        }
        row["score"] = _anchor_score(row) + 2.0
        source_rows.append(row)

    margin = max(0.0, float(margin_s))
    normalized_windows = [_safe_window(value) for value in windows]
    groups: list[list[dict[str, Any]]] = [[] for _ in windows]
    for item in _deduplicate_anchor_rows(source_rows):
        timestamp = float(item["timestamp_s"])
        inside = [
            index
            for index, window in enumerate(normalized_windows)
            if window is not None and window[0] <= timestamp <= window[1]
        ]
        eligible = inside or [
            index
            for index, window in enumerate(normalized_windows)
            if window is not None
            and window[0] - margin <= timestamp <= window[1] + margin
        ]
        if not eligible:
            continue
        # One existing observation should not silently join two independent
        # candidates. Prefer an actual containing window, then the narrowest
        # and temporally closest request.
        best_index = min(
            eligible,
            key=lambda index: (
                normalized_windows[index][1] - normalized_windows[index][0],
                abs(
                    timestamp
                    - (
                        normalized_windows[index][0]
                        + normalized_windows[index][1]
                    )
                    / 2.0
                ),
                index,
            ),
        )
        groups[best_index].append(dict(item))

    for index, window in enumerate(normalized_windows):
        if window is None:
            groups[index] = []
            continue
        groups[index] = _limit_anchor_rows(
            _deduplicate_anchor_rows(groups[index]),
            window=window,
            limit=max(1, int(per_window)),
        )
    return groups


def collect_memory_anchor_timestamps(
    memory: dict[str, Any] | None,
    windows: list[list[float]],
    *,
    margin_s: float = 0.0,
    per_window: int = 4,
) -> list[list[float]]:
    return [
        [round(float(item["timestamp_s"]), 3) for item in rows]
        for rows in collect_memory_anchor_rows(
            memory,
            windows,
            margin_s=margin_s,
            per_window=per_window,
        )
    ]


def expand_windows_to_anchors(
    windows: list[list[float]],
    anchors_by_window: list[list[float]],
    *,
    duration: float,
    pad_s: float = 0.25,
) -> list[list[float]]:
    """Include nearby observed anchors without merging independent windows."""
    expanded: list[list[float]] = []
    pad = max(0.0, float(pad_s))
    for index, raw_window in enumerate(windows):
        window = _safe_window(raw_window)
        if window is None:
            continue
        start, end = window
        anchors = anchors_by_window[index] if index < len(anchors_by_window) else []
        for value in anchors:
            anchor = float(value)
            if anchor < start:
                start = anchor - pad
            elif anchor > end:
                end = anchor + pad
        expanded.append(
            [
                round(max(0.0, start), 3),
                round(min(float(duration), end), 3),
            ]
        )
    return expanded
