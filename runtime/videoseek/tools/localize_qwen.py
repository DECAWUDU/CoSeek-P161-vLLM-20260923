import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from videoseek.codec import codec_aware_timestamps
from videoseek.core.local_observation_cache import (
    annotate_candidate_novelty,
    lookup_caption_rows,
    store_caption_rows,
)
from videoseek.core.memory import extract_v10_payload
from videoseek.core.minimal_global_fsm import should_use_minimal_global_fsm
from videoseek.core.tool_evidence_handoff import (
    collect_memory_anchor_rows,
    expand_windows_to_anchors,
)
from videoseek.tools.focus_qwen import execute_focus_qwen
from videoseek.tools.skim_qwen import execute_skim_qwen
from videoseek.tools.v10_format import format_v10_observation


localize_qwen_tool = {
    "type": "function",
    "function": {
        "name": "localize_qwen",
        "description": (
            "Use local Qwen to perform one hierarchical localization action: "
            "coarse scan the requested windows, rank candidate moments, and densely "
            "inspect the best candidates. The result is candidate evidence only; use "
            "frame_verify for the final visual decision."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "search_windows": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "description": "Absolute/global time windows to search.",
                },
                "localization_goal": {
                    "type": "string",
                    "description": "The visual event or detail to localize, without selecting an option.",
                },
                "evidence_profile": {
                    "type": "string",
                    "enum": [
                        "generic",
                        "entity_presence",
                        "event_boundary",
                        "state_change",
                        "relation",
                        "temporal_order",
                    ],
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 6,
                },
            },
            "required": [
                "search_windows",
                "localization_goal",
                "evidence_profile",
                "top_k",
            ],
            "additionalProperties": False,
        },
    },
}


def _as_window(value: Any, *, duration: float) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start = max(0.0, min(float(value[0]), duration))
        end = max(0.0, min(float(value[1]), duration))
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return [round(start, 3), round(end, 3)]


def _normalize_windows(parameters: dict, *, duration: float, max_windows: int) -> list[list[float]]:
    values = parameters.get("search_windows") or parameters.get("windows") or []
    if not values and parameters.get("start_time") is not None:
        values = [[parameters.get("start_time"), parameters.get("end_time")]]
    valid = []
    for value in values:
        window = _as_window(value, duration=duration)
        if window is not None and window not in valid:
            valid.append(window)
    valid.sort(key=lambda item: (item[0], item[1]))

    merged: list[list[float]] = []
    for start, end in valid:
        if not merged or start > merged[-1][1] + 0.1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged[: max(1, max_windows)]


def _expand_search_windows(
    windows: list[list[float]],
    *,
    duration: float,
    margin_s: float,
) -> list[list[float]]:
    """Add bounded context without merging independent planner candidates."""
    margin_s = max(0.0, float(margin_s))
    if margin_s <= 0.0:
        return [list(window) for window in windows]

    expanded: list[list[float]] = []
    for index, (start, end) in enumerate(windows):
        left = max(0.0, start - margin_s)
        right = min(duration, end + margin_s)

        # When two requested windows are close, split their shared context at
        # the midpoint. This preserves independent candidate coverage and
        # avoids scanning the same frames twice.
        if index > 0:
            previous_end = windows[index - 1][1]
            left = max(left, (previous_end + start) / 2.0)
        if index + 1 < len(windows):
            next_start = windows[index + 1][0]
            right = min(right, (end + next_start) / 2.0)

        expanded.append([round(left, 3), round(right, 3)])
    return expanded


def _effective_search_context_margin(config: dict, *, duration: float) -> float:
    max_margin_s = max(
        0.0,
        float(config.get("localize_qwen_search_context_margin_s") or 0.0),
    )
    if not config.get("localize_qwen_adaptive_search_context", False):
        return max_margin_s
    overview_frames = max(
        2,
        int(config.get("frame_sampling_factor") or 1)
        * int(config.get("overview_base") or 1),
    )
    overview_step_s = max(0.0, float(duration)) / (overview_frames - 1)
    return round(min(max_margin_s, overview_step_s), 3)


def _overlap_ratio(left: list[float], right: list[float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    shortest = max(1e-6, min(left[1] - left[0], right[1] - right[0]))
    return overlap / shortest


def _window_center(window: list[float]) -> float:
    return (window[0] + window[1]) / 2.0


def _allocate_budget(windows: list[list[float]], total_budget: int, minimum: int) -> list[int]:
    if not windows:
        return []
    total_budget = max(len(windows), int(total_budget))
    minimum = max(1, min(int(minimum), total_budget // len(windows)))
    budgets = [minimum] * len(windows)
    remaining = total_budget - sum(budgets)
    if remaining <= 0:
        return budgets
    weights = [max(1.0, end - start) for start, end in windows]
    weight_sum = sum(weights)
    exact = [remaining * weight / weight_sum for weight in weights]
    for index, value in enumerate(exact):
        budgets[index] += int(math.floor(value))
    for index in sorted(
        range(len(windows)),
        key=lambda idx: exact[idx] - math.floor(exact[idx]),
        reverse=True,
    )[: total_budget - sum(budgets)]:
        budgets[index] += 1
    return budgets


def _allocate_duration_budgets(
    windows: list[list[float]],
    *,
    total_budget: int,
    minimum: int,
    target_fps: float,
    per_window_max: int,
) -> list[int]:
    """Use the frame cap only when window duration actually needs it."""
    if not windows:
        return []
    minimum = max(1, int(minimum))
    total_budget = max(len(windows) * minimum, int(total_budget))
    per_window_max = max(minimum, int(per_window_max))
    desired = [
        min(
            per_window_max,
            max(minimum, int(math.ceil(max(0.1, end - start) * target_fps)) + 1),
        )
        for start, end in windows
    ]
    if sum(desired) <= total_budget:
        return desired

    budgets = [minimum] * len(windows)
    remaining = total_budget - sum(budgets)
    while remaining > 0:
        candidates = [
            index for index, value in enumerate(budgets) if value < desired[index]
        ]
        if not candidates:
            break
        index = max(candidates, key=lambda item: desired[item] - budgets[item])
        budgets[index] += 1
        remaining -= 1
    return budgets


def _sampling_anchors(
    config: dict,
    parameters: dict,
    *,
    window: list[float],
    frame_budget: int,
    preferred: list[float] | None = None,
) -> list[float]:
    anchors = [
        float(value)
        for value in (preferred or [])
        if window[0] <= float(value) <= window[1]
    ]
    if config.get("localize_qwen_codec_anchor_sampling_enabled", False):
        codec_budget = min(6, max(2, int(frame_budget) // 4))
        anchors.extend(
            codec_aware_timestamps(
                video_path=str(parameters.get("video_path") or ""),
                duration_s=float(parameters.get("duration") or window[1]),
                start_time=window[0],
                end_time=window[1],
                num_frames=codec_budget,
            )
        )
    return sorted({round(float(value), 3) for value in anchors})


def _cached_observer_payload(
    rows: list[dict],
    *,
    window: list[float],
    stage: str,
    cache_audit: dict,
) -> dict:
    relevant = [
        row
        for row in rows
        if str(row.get("confidence") or "").strip().lower() in {"medium", "high"}
    ]
    summary_rows = relevant or rows
    summary = "; ".join(
        str(row.get("description") or "").strip()
        for row in summary_rows[:6]
        if str(row.get("description") or "").strip()
    )[:480]
    return {
        "t_range": list(window),
        "timestamp_observations": [dict(row, focus_request_window=list(window)) for row in rows] if stage == "fine" else rows,
        "suggest_focus_windows": [],
        "possible_evidence": bool(relevant),
        "relevance": min(1.0, len(relevant) / max(1, len(rows))),
        "overall_summary": summary,
        "observer_backend": "local_qwen_cache",
        "observer_status": "complete",
        "observer_wall_s": 0.0,
        "num_frames": 0,
        "observer_batch_count": 0,
        "parse_ok": True,
        "caption_cache_stage": stage,
        "caption_cache_audit": cache_audit,
    }


def _observation_index(payloads: list[dict], goal: str) -> dict:
    """Index returned pixels and literal descriptions; never infer new events."""
    frames, records, unbound = set(), set(), 0
    for payload in payloads:
        for row in _timestamp_rows(payload):
            if row.get("qwen_omitted") or "qwen_omitted_frame" in row.get("event_tags", []):
                continue
            index = row.get("frame_index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                unbound += 1
                continue
            frames.add(index)
            identity = [index, " ".join(goal.split()), " ".join(str(row.get("description") or "").split()),
                        row.get("target_match"), row.get("event_match"), row.get("target_id"), row.get("target_event_ids")]
            records.add(hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
    return {"frame_indices": sorted(frames), "observation_hashes": sorted(records), "unbound_rows": unbound}


def _merge_observer_payloads(payloads: list[dict], *, window: list[float]) -> dict:
    confidence_rank = {"high": 2, "medium": 1, "low": 0}
    rows_by_timestamp: dict[tuple, dict] = {}
    for payload in payloads:
        for row in _timestamp_rows(payload):
            timestamp = round(float(row.get("timestamp_s")), 3)
            owner = tuple(row.get("focus_request_window") or [])
            key = (owner, row.get("frame_index", timestamp)) if owner else ((), timestamp)
            previous = rows_by_timestamp.get(key)
            old_rank = confidence_rank.get(
                str((previous or {}).get("confidence") or "low").lower(), 0
            )
            new_rank = confidence_rank.get(
                str(row.get("confidence") or "low").lower(), 0
            )
            if previous is None or new_rank >= old_rank:
                rows_by_timestamp[key] = dict(row)
    rows = sorted(rows_by_timestamp.values(), key=lambda row: (float(row["timestamp_s"]), tuple(row.get("focus_request_window") or [])))
    suggestions: list[list[float]] = []
    for payload in payloads:
        for value in payload.get("suggest_focus_windows") or []:
            candidate = _as_window(value, duration=float(window[1]))
            if candidate is not None and candidate not in suggestions:
                suggestions.append(candidate)
    summaries = [
        str(payload.get("overall_summary") or payload.get("window_summary") or "").strip()
        for payload in payloads
        if str(payload.get("overall_summary") or payload.get("window_summary") or "").strip()
    ]
    return {
        "t_range": list(window),
        "timestamp_observations": rows,
        "suggest_focus_windows": suggestions,
        "possible_evidence": any(payload.get("possible_evidence") is True for payload in payloads),
        "relevance": max(
            (float(payload.get("relevance") or 0.0) for payload in payloads),
            default=0.0,
        ),
        "overall_summary": " | ".join(summaries)[:640],
        "observer_backend": "local_qwen_cache" if all(
            payload.get("observer_backend") == "local_qwen_cache" for payload in payloads
        ) else "local_qwen",
        "observer_status": "partial" if any(
            _observer_payload_failed(payload) or payload.get("observer_status") == "partial" for payload in payloads
        ) else "complete",
        "observer_wall_s": sum(float(payload.get("observer_wall_s") or 0.0) for payload in payloads),
        "num_frames": sum(int(payload.get("num_frames") or 0) for payload in payloads),
        "observed_frame_count": sum(int(payload.get("observed_frame_count", payload.get("num_frames")) or 0) for payload in payloads),
        "observer_batch_count": sum(
            int(payload.get("observer_batch_count") or 0) for payload in payloads
        ),
        "parse_ok": all(payload.get("parse_ok") is not False for payload in payloads),
    }


def _boundary_directions(
    payload: dict,
    window: list[float],
    *,
    trigger_s: float,
) -> tuple[bool, bool]:
    left = False
    right = False
    for row in _timestamp_rows(payload):
        confidence = str(row.get("confidence") or "").strip().lower()
        if confidence not in {"medium", "high"}:
            continue
        timestamp = float(row.get("timestamp_s"))
        left = left or timestamp - window[0] <= trigger_s
        right = right or window[1] - timestamp <= trigger_s
    return left, right


def _boundary_expansion_strips(
    windows: list[list[float]],
    payloads: list[dict],
    *,
    duration: float,
    margin_s: float,
    trigger_s: float,
) -> list[tuple[int, list[float]]]:
    strips: list[tuple[int, list[float]]] = []
    for index, (window, payload) in enumerate(zip(windows, payloads)):
        expand_left, expand_right = _boundary_directions(
            payload,
            window,
            trigger_s=trigger_s,
        )
        if expand_left and window[0] > 0.0:
            left = max(0.0, window[0] - margin_s)
            if index > 0:
                left = max(left, (windows[index - 1][1] + window[0]) / 2.0)
            if window[0] - left >= 0.25:
                strips.append((index, [round(left, 3), window[0]]))
        if expand_right and window[1] < duration:
            right = min(duration, window[1] + margin_s)
            if index + 1 < len(windows):
                right = min(right, (window[1] + windows[index + 1][0]) / 2.0)
            if right - window[1] >= 0.25:
                strips.append((index, [window[1], round(right, 3)]))
    return strips


def _timestamp_rows(payload: dict) -> list[dict]:
    return [
        item
        for item in (payload.get("timestamp_observations") or [])
        if isinstance(item, dict) and item.get("timestamp_s") is not None
    ]


def _row_query_relevance(row: dict) -> str:
    value = str(
        row.get("confidence") or row.get("query_relevance") or "low"
    ).strip().lower()
    return value if value in {"low", "medium", "high", "unknown"} else "unknown"


def _row_evidence_class(row: dict) -> str:
    """Classify a local observation without treating visual certainty as truth.

    P124 asks the local observer for target/event labels. Older cached rows do
    not have them, so a legacy high query-relevance row remains recall-safe and
    is sent to the strong verifier. A label conflict is ambiguous rather than
    silently negative.
    """
    relevance = _row_query_relevance(row)
    target_match = str(row.get("target_match") or "unknown").strip().lower()
    event_match = str(row.get("event_match") or "unknown").strip().lower()
    labels_present = target_match != "unknown" or event_match != "unknown"

    if target_match == "matched" and event_match == "direct":
        return "positive"
    if (
        target_match == "matched"
        and event_match == "unknown"
        and relevance == "high"
    ):
        return "positive"
    if not labels_present:
        return "ambiguous"
    if target_match == "not_matched" or event_match == "not_matched":
        return "negative" if relevance == "low" else "ambiguous"
    if (
        target_match in {"matched", "possible"}
        or event_match in {"direct", "context_only"}
        or relevance in {"medium", "high"}
    ):
        return "ambiguous"
    return "negative"


def _candidate_evidence(payload: dict, window: list[float]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in _timestamp_rows(payload):
        try:
            timestamp = float(row.get("timestamp_s"))
        except (TypeError, ValueError):
            continue
        if window[0] <= timestamp <= window[1]:
            rows.append(row)
    observed_rows = [row for row in rows
                     if not row.get("qwen_omitted")
                     and "qwen_omitted_frame" not in (row.get("event_tags") or [])]
    positive_rows = [row for row in observed_rows if _row_evidence_class(row) == "positive"]
    ambiguous_rows = [row for row in observed_rows if _row_evidence_class(row) == "ambiguous"]
    evidence_class = (
        "positive" if positive_rows else "ambiguous" if ambiguous_rows else "unobserved" if rows and not observed_rows else "negative"
    )

    def compact(row: dict) -> dict[str, Any]:
        return {
            "timestamp_s": round(float(row.get("timestamp_s")), 3),
            **{k: row[k] for k in ("frame_index", "source_timestamp_s", "window_id", "frame_ids") if k in row},
            "description": str(row.get("description") or "").strip()[:240],
            "query_relevance": _row_query_relevance(row),
            "visual_confidence": str(
                row.get("visual_confidence") or "unknown"
            ).strip().lower(),
            "target_match": str(row.get("target_match") or "unknown").strip().lower(),
            "event_match": str(row.get("event_match") or "unknown").strip().lower(),
        }

    return {
        "evidence_class": evidence_class,
        "positive_anchors": [compact(row) for row in positive_rows],
        "ambiguous_anchors": [compact(row) for row in ambiguous_rows],
        "row_count": len(rows),
    }


def _fine_candidate_rows(payload: dict, window: list[float]) -> list[dict]:
    """Follow the Fine request owner, not the old coarse temporal crop.

    Legacy observations have no owner; retain their conservative time-only mapping.
    Cache hits are associated with the current lookup window without changing pixels.
    """
    return [row for row in _timestamp_rows(payload)
            if (list(row["focus_request_window"]) == list(window)
                if row.get("focus_request_window") else
                window[0] <= float(row["timestamp_s"]) <= window[1])]


def _observer_payload_failed(payload: dict) -> bool:
    """Return whether a child observation failed before making a visual claim."""
    backend = str(payload.get("observer_backend") or "").lower()
    status = str(payload.get("observer_status") or "").lower()
    if "error" in backend or status == "unavailable":
        return True
    return bool(
        payload.get("parse_ok") is False
        and not _timestamp_rows(payload)
        and int(payload.get("num_frames") or 0) <= 0
    )


def _observer_health(payloads: list[dict]) -> tuple[str, int]:
    failures = sum(_observer_payload_failed(payload) for payload in payloads)
    if payloads and failures == len(payloads):
        return "unavailable", failures
    if failures or any(p.get("observer_status") == "partial" for p in payloads):
        return "partial", failures
    return "complete", 0


def _candidate_score_components(
    payload: dict,
    window: list[float],
    rank: int,
) -> dict[str, float | int]:
    """Score a candidate from normalized local evidence, not raw row count.

    The old score summed every sampled row and clipped the result. Dense or
    long windows therefore saturated at the same value even when only a small
    fraction of their frames was query-relevant. These components are bounded
    independently, so changing the sampling density does not change candidate
    priority by itself.
    """
    rows: list[tuple[float, float, float, float]] = []
    confidence_weights = {"high": 1.0, "medium": 0.55, "low": 0.0}
    target_weights = {"matched": 1.0, "possible": 0.55, "not_matched": 0.0}
    event_weights = {"direct": 1.0, "context_only": 0.45, "not_matched": 0.0}
    for row in sorted(
        _timestamp_rows(payload),
        key=lambda item: float(item.get("timestamp_s") or -1.0),
    ):
        try:
            timestamp = float(row.get("timestamp_s"))
        except (TypeError, ValueError):
            continue
        if not window[0] <= timestamp <= window[1]:
            continue
        relevance = confidence_weights.get(_row_query_relevance(row), 0.0)
        target_match = str(row.get("target_match") or "unknown").strip().lower()
        event_match = str(row.get("event_match") or "unknown").strip().lower()
        target_relevance = target_weights.get(target_match, relevance)
        event_relevance = event_weights.get(event_match, relevance)
        rows.append((timestamp, relevance, target_relevance, event_relevance))

    relevant_count = sum(
        max(target_relevance, event_relevance) > 0.0
        for _, _, target_relevance, event_relevance in rows
    )
    mean_target_relevance = (
        sum(target_relevance for _, _, target_relevance, _ in rows) / len(rows)
        if rows
        else 0.0
    )
    mean_event_relevance = (
        sum(event_relevance for _, _, _, event_relevance in rows) / len(rows)
        if rows
        else 0.0
    )
    support_fraction = relevant_count / len(rows) if rows else 0.0
    peak_relevance = max(
        (max(target_relevance, event_relevance) for _, _, target_relevance, event_relevance in rows),
        default=0.0,
    )

    longest_run = 0
    current_run = 0
    for _, _, target_relevance, event_relevance in rows:
        if max(target_relevance, event_relevance) > 0.0:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    continuity = longest_run / len(rows) if rows else 0.0
    score = (
        0.45 * mean_target_relevance
        + 0.30 * mean_event_relevance
        + 0.15 * continuity
        + 0.05 * support_fraction
        + 0.05 * peak_relevance
    )
    return {
        "score": round(score, 4),
        "sampled_row_count": len(rows),
        "relevant_row_count": relevant_count,
        "mean_target_relevance": round(mean_target_relevance, 4),
        "mean_event_relevance": round(mean_event_relevance, 4),
        "support_fraction": round(support_fraction, 4),
        "continuity": round(continuity, 4),
        "peak_relevance": round(peak_relevance, 4),
        "local_rank_used": 0,
    }


def _candidate_score(
    payload: dict,
    window: list[float],
    rank: int,
    *,
    normalized: bool = False,
) -> float:
    if normalized:
        return float(_candidate_score_components(payload, window, rank)["score"])
    try:
        relevance = float(payload.get("relevance") or 0.0)
    except (TypeError, ValueError):
        relevance = 0.0
    row_score = 0.0
    for row in _timestamp_rows(payload):
        try:
            timestamp = float(row.get("timestamp_s"))
        except (TypeError, ValueError):
            continue
        if not window[0] <= timestamp <= window[1]:
            continue
        confidence = str(
            row.get("confidence") or row.get("query_relevance") or ""
        ).lower()
        row_score += {"high": 0.12, "medium": 0.06}.get(confidence, 0.02)
    return round(1.0 / (1.0 + rank) + min(0.5, relevance * 0.35 + row_score), 4)


def _relevance_windows_from_rows(
    payload: dict,
    source_window: list[float],
    *,
    candidate_window_s: float,
    max_windows: int,
) -> list[list[float]]:
    """Recover candidate windows from query-aware per-frame relevance.

    Density-batched skim already merges every timestamped caption, but an
    unreliable child ``suggest_focus_windows`` can otherwise discard a much
    stronger high-relevance run. Keep this derivation deterministic and local:
    it only groups medium/high rows that Qwen has already emitted.
    """
    relevant: list[tuple[float, int]] = []
    for row in _timestamp_rows(payload):
        confidence = str(row.get("confidence") or "").strip().lower()
        weight = {"high": 2, "medium": 1}.get(confidence, 0)
        if not weight or "qwen_omitted_frame" in (row.get("event_tags") or []):
            continue
        try:
            timestamp = float(row.get("timestamp_s"))
        except (TypeError, ValueError):
            continue
        if source_window[0] <= timestamp <= source_window[1]:
            relevant.append((timestamp, weight))
    if not relevant:
        return []

    relevant.sort()
    cluster_gap_s = max(2.0, min(6.0, float(candidate_window_s) / 3.0))
    clusters: list[list[tuple[float, int]]] = []
    for row in relevant:
        if not clusters or row[0] - clusters[-1][-1][0] > cluster_gap_s:
            clusters.append([row])
        else:
            clusters[-1].append(row)

    ranked_clusters = sorted(
        clusters,
        key=lambda cluster: (
            sum(weight for _, weight in cluster),
            sum(1 for _, weight in cluster if weight == 2),
            len(cluster),
        ),
        reverse=True,
    )
    windows: list[list[float]] = []
    max_span = max(2.0, float(candidate_window_s))
    for cluster in ranked_clusters:
        start = max(source_window[0], cluster[0][0] - 2.0)
        end = min(source_window[1], cluster[-1][0] + 2.0)
        if end - start > max_span:
            center = sum(timestamp * weight for timestamp, weight in cluster) / sum(
                weight for _, weight in cluster
            )
            start = max(source_window[0], center - max_span / 2.0)
            end = min(source_window[1], start + max_span)
            start = max(source_window[0], end - max_span)
        if end <= start:
            continue
        window = [round(start, 3), round(end, 3)]
        if any(_overlap_ratio(window, old) >= 0.75 for old in windows):
            continue
        windows.append(window)
        if len(windows) >= max(1, int(max_windows)):
            break
    return windows


def _fallback_candidate_windows(payload: dict, source_window: list[float], window_s: float) -> list[list[float]]:
    anchors = []
    for row in _timestamp_rows(payload):
        confidence = str(row.get("confidence") or "").lower()
        description = str(row.get("description") or "").strip()
        if confidence not in {"medium", "high"} and not description:
            continue
        try:
            anchors.append(float(row.get("timestamp_s")))
        except (TypeError, ValueError):
            continue
    if not anchors and payload.get("possible_evidence") is not True:
        return []
    anchor = anchors[0] if anchors else _window_center(source_window)
    half = max(1.0, window_s / 2.0)
    start = max(source_window[0], anchor - half)
    end = min(source_window[1], anchor + half)
    if end - start < min(2.0, source_window[1] - source_window[0]):
        start = max(source_window[0], end - window_s)
    return [[round(start, 3), round(end, 3)]]


def _rank_coarse_candidates(
    coarse_payloads: list[tuple[list[float], dict]],
    *,
    top_k: int,
    candidate_window_s: float,
    row_candidates_enabled: bool = False,
    normalized_score_enabled: bool = False,
    preserve_source_coverage: bool = False,
    evidence_preserving: bool = False,
    ambiguous_reserve: int = 0,
    retain_weak_source_candidates: bool = False,
) -> list[dict]:
    raw: list[dict] = []
    for source_index, (source_window, payload) in enumerate(coarse_payloads):
        windows = payload.get("suggest_focus_windows") or []
        normalized = (
            _relevance_windows_from_rows(
                payload,
                source_window,
                candidate_window_s=candidate_window_s,
                max_windows=(
                    max(top_k + max(0, int(ambiguous_reserve)), len(_timestamp_rows(payload)))
                    if evidence_preserving
                    else top_k
                ),
            )
            if row_candidates_enabled or evidence_preserving
            else []
        )
        for value in windows:
            window = _as_window(value, duration=source_window[1])
            if window is None:
                continue
            window = [max(source_window[0], window[0]), min(source_window[1], window[1])]
            if window[1] > window[0] and not any(
                _overlap_ratio(window, old) >= 0.75 for old in normalized
            ):
                normalized.append(window)
        if not normalized:
            normalized = _fallback_candidate_windows(
                payload,
                source_window,
                candidate_window_s,
            )
        for rank, window in enumerate(normalized):
            evidence = _candidate_evidence(payload, window)
            score_components = (
                _candidate_score_components(payload, window, rank)
                if normalized_score_enabled
                else {}
            )
            raw.append(
                {
                    "window": [round(window[0], 3), round(window[1], 3)],
                    "score": (
                        score_components["score"]
                        if score_components
                        else _candidate_score(payload, window, rank)
                    ),
                    "score_components": score_components,
                    "source_search_window": source_window,
                    "source_index": source_index,
                    "coarse_summary": str(
                        payload.get("overall_summary")
                        or payload.get("window_summary")
                        or ""
                    ).strip()[:320],
                    "coarse_evidence_class": evidence["evidence_class"],
                    "coarse_positive_anchors": evidence["positive_anchors"],
                    "coarse_ambiguous_anchors": evidence["ambiguous_anchors"],
                    "mandatory_positive": evidence["evidence_class"] == "positive",
                }
            )

    if evidence_preserving:
        class_priority = {"positive": 2, "ambiguous": 1, "negative": 0}
        ordered = sorted(
            raw,
            key=lambda item: (
                class_priority.get(item.get("coarse_evidence_class"), 0),
                float(item.get("score") or 0.0),
                -float(item["window"][0]),
            ),
            reverse=True,
        )
        deduplicated: list[dict] = []
        for candidate in ordered:
            candidate_positive_timestamps = {
                float(row.get("timestamp_s"))
                for row in candidate.get("coarse_positive_anchors") or []
                if row.get("timestamp_s") is not None
            }
            duplicate = next(
                (
                    old
                    for old in deduplicated
                    if (
                        _overlap_ratio(candidate["window"], old["window"]) >= 0.75
                        or bool(
                            candidate_positive_timestamps
                            & {
                                float(row.get("timestamp_s"))
                                for row in old.get("coarse_positive_anchors") or []
                                if row.get("timestamp_s") is not None
                            }
                        )
                    )
                ),
                None,
            )
            if duplicate is None:
                deduplicated.append(candidate)
                continue
            for field in ("coarse_positive_anchors", "coarse_ambiguous_anchors"):
                seen = {
                    float(row.get("timestamp_s"))
                    for row in duplicate.get(field) or []
                    if row.get("timestamp_s") is not None
                }
                for row in candidate.get(field) or []:
                    timestamp = float(row.get("timestamp_s"))
                    if timestamp not in seen:
                        duplicate.setdefault(field, []).append(row)
                        seen.add(timestamp)
            if candidate.get("mandatory_positive"):
                duplicate["mandatory_positive"] = True
                duplicate["coarse_evidence_class"] = "positive"

        positives = [item for item in deduplicated if item.get("mandatory_positive")]
        ambiguous = [
            item
            for item in deduplicated
            if item.get("coarse_evidence_class") == "ambiguous"
        ]
        selected = positives + ambiguous[
            : max(1, top_k) + max(0, int(ambiguous_reserve))
        ]
        if retain_weak_source_candidates:
            # P131 treats every requested source window as an independent routing
            # domain.  Keep its best weak row so a fine negative cannot silently
            # erase the domain before the strong verifier sees it.  The agent owns
            # the final eight-window cap and adds a routing-only candidate only
            # when a source has no local candidate at all.
            covered_sources = {
                int(item.get("source_index") or 0) for item in selected
            }
            for candidate in ordered:
                source_index = int(candidate.get("source_index") or 0)
                if source_index in covered_sources:
                    continue
                selected.append(candidate)
                covered_sources.add(source_index)
        return selected

    ordered = sorted(raw, key=lambda item: item["score"], reverse=True)
    prioritized: list[dict] = []
    if preserve_source_coverage:
        best_by_source: dict[int, dict] = {}
        for candidate in ordered:
            best_by_source.setdefault(candidate["source_index"], candidate)
        prioritized.extend(
            sorted(
                best_by_source.values(),
                key=lambda item: item["score"],
                reverse=True,
            )
        )
    prioritized.extend(
        candidate for candidate in ordered if candidate not in prioritized
    )

    selected = []
    for candidate in prioritized:
        if any(_overlap_ratio(candidate["window"], old["window"]) >= 0.75 for old in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max(1, top_k):
            break
    return selected


def _tight_verify_window(window: list[float], rows: list[dict], *, max_window_s: float) -> list[float]:
    timestamps = []
    for row in rows:
        try:
            timestamp = float(row.get("timestamp_s"))
        except (TypeError, ValueError):
            continue
        if window[0] <= timestamp <= window[1]:
            timestamps.append(timestamp)
    center = (
        timestamps[len(timestamps) // 2]
        if timestamps
        else _window_center(window)
    )
    span = min(max_window_s, window[1] - window[0])
    start = max(window[0], center - span / 2.0)
    end = min(window[1], start + span)
    start = max(window[0], end - span)
    return [round(start, 3), round(end, 3)]


def _expand_verify_window(
    window: list[float],
    *,
    source_window: list[float],
    margin_s: float,
) -> list[float]:
    """Retain a small amount of context around a localized event boundary."""
    margin = max(0.0, float(margin_s))
    if margin <= 0:
        return list(window)
    return [
        round(max(float(source_window[0]), float(window[0]) - margin), 3),
        round(min(float(source_window[1]), float(window[1]) + margin), 3),
    ]


def _expand_verify_window_to_nearby_anchors(
    window: list[float],
    *,
    source_window: list[float],
    anchors: list[float],
    max_gap_s: float,
    pad_s: float,
) -> list[float]:
    """Keep a nearby recalled boundary anchor inside the actual verify window."""
    start, end = float(window[0]), float(window[1])
    source_start, source_end = float(source_window[0]), float(source_window[1])
    max_gap = max(0.0, float(max_gap_s))
    pad = max(0.0, float(pad_s))
    for raw_timestamp in anchors:
        try:
            timestamp = float(raw_timestamp)
        except (TypeError, ValueError):
            continue
        if timestamp < start and start - timestamp <= max_gap:
            start = max(source_start, timestamp - pad)
        elif timestamp > end and timestamp - end <= max_gap:
            end = min(source_end, timestamp + pad)
    return [round(start, 3), round(end, 3)]


def _verification_anchors(
    rows: list[dict],
    *,
    localized_window: list[float],
    verify_window: list[float],
    limit: int = 12,
) -> list[float]:
    """Keep local evidence timestamps plus explicit pre/post context anchors."""
    values = {
        round(float(verify_window[0]), 3),
        round(float(localized_window[0]), 3),
        round(float(localized_window[1]), 3),
        round(float(verify_window[1]), 3),
    }
    for row in rows:
        try:
            timestamp = float(row.get("timestamp_s"))
        except (TypeError, ValueError):
            continue
        if verify_window[0] <= timestamp <= verify_window[1]:
            values.add(round(timestamp, 3))
    anchors = sorted(values)
    if len(anchors) > max(2, int(limit)):
        positions = np.linspace(0, len(anchors) - 1, int(limit)).round().astype(int)
        anchors = [anchors[int(position)] for position in sorted(set(positions.tolist()))]
    return anchors


def _write_trace(output_dir: str, payload: dict) -> str:
    if not output_dir:
        return ""
    trace_dir = Path(output_dir) / "localize_qwen_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"localize_{time.time_ns()}.json"
    trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(trace_path)


def execute_localize_qwen(config: dict, parameters: dict) -> str:
    """Run coarse and fine local-Qwen localization as one Agent action."""
    vr = parameters.get("vr")
    if vr is None:
        raise ValueError("localize_qwen requires the active video reader")
    duration = float(parameters.get("duration") or round(len(vr) / vr.get_avg_fps(), 2))
    goal = str(
        parameters.get("localization_goal")
        or parameters.get("query")
        or parameters.get("question")
        or ""
    ).strip()
    if not goal:
        raise ValueError("localize_qwen requires localization_goal")
    # Inherited by coarse batches and fine windows, without changing overview calls.
    parameters = dict(parameters, local_search_question=str(parameters.get("question") or ""))
    p131_enabled = bool(
        config.get("p131_minimal_global_repairs_enabled", False)
        and should_use_minimal_global_fsm(str(parameters.get("question") or ""))
    )
    max_search_windows = int(config.get("localize_qwen_max_search_windows") or 8)
    requested_search_windows = _normalize_windows(
        parameters,
        duration=duration,
        max_windows=max_search_windows,
    )
    if not requested_search_windows:
        raise ValueError("localize_qwen requires at least one valid search window")
    boundary_expansion_enabled = bool(
        config.get("localize_qwen_boundary_expansion_enabled", False)
    )
    search_context_margin_s = (
        0.0
        if boundary_expansion_enabled
        else _effective_search_context_margin(config, duration=duration)
    )
    search_windows = _expand_search_windows(
        requested_search_windows,
        duration=duration,
        margin_s=search_context_margin_s,
    )
    tool_integrity_enabled = bool(
        config.get("coseek1_tool_integrity_repair_enabled", False)
    )
    memory_anchor_rows_by_window: list[list[dict[str, Any]]] = [
        [] for _ in requested_search_windows
    ]
    if tool_integrity_enabled:
        memory_anchor_rows_by_window = collect_memory_anchor_rows(
            parameters.get("memory"),
            requested_search_windows,
            margin_s=float(
                config.get("coseek1_tool_memory_anchor_margin_s") or 2.0
            ),
            per_window=int(
                config.get("coseek1_tool_memory_anchors_per_window") or 4
            ),
        )
        memory_anchor_timestamps_by_window = [
            [round(float(item["timestamp_s"]), 3) for item in rows]
            for rows in memory_anchor_rows_by_window
        ]
        search_windows = expand_windows_to_anchors(
            search_windows,
            memory_anchor_timestamps_by_window,
            duration=duration,
            pad_s=float(config.get("coseek1_tool_memory_anchor_pad_s") or 0.25),
        )
    else:
        memory_anchor_timestamps_by_window = [
            [] for _ in requested_search_windows
        ]
    requested_anchor_timestamps = [
        round(_window_center(window), 3) for window in requested_search_windows
    ]
    preferred_anchor_timestamps_by_window = [
        sorted(
            {
                requested_anchor_timestamps[index],
                *memory_anchor_timestamps_by_window[index],
                *(float(t) for t in parameters.get("mandatory_timestamps") or []
                  if isinstance(t, (int, float)) and requested_search_windows[index][0] <= t <= requested_search_windows[index][1]),
            }
        )
        for index in range(len(requested_search_windows))
    ]

    requested_top_k = max(
        1,
        min(
            int(parameters.get("top_k") or config.get("localize_qwen_top_k") or 3),
            int(config.get("localize_qwen_max_top_k") or 4),
        ),
    )
    top_k = requested_top_k
    if config.get("localize_qwen_cover_search_windows", False):
        top_k = max(
            requested_top_k,
            min(
                int(config.get("localize_qwen_max_top_k") or 4),
                len(search_windows),
            ),
        )
    profile = str(parameters.get("evidence_profile") or "generic")
    coarse_budget = int(config.get("localize_qwen_coarse_max_frames") or 96)
    if config.get("localize_qwen_duration_budget_enabled", False):
        coarse_budgets = _allocate_duration_budgets(
            search_windows,
            total_budget=coarse_budget,
            minimum=int(config.get("localize_qwen_duration_budget_min_frames") or 8),
            target_fps=float(config.get("localize_qwen_duration_budget_fps") or 2.0),
            per_window_max=int(
                config.get("localize_qwen_duration_budget_max_frames") or 32
            ),
        )
    else:
        coarse_budgets = _allocate_budget(search_windows, coarse_budget, minimum=4)
    coarse_payloads: list[tuple[list[float], dict]] = []
    coarse_raw = []
    cache_enabled = bool(config.get("localize_qwen_caption_cache_enabled", False))
    cache_goal_similarity = float(
        config.get("localize_qwen_caption_cache_goal_similarity") or 0.72
    )
    cache_coverage_ratio = float(
        config.get("localize_qwen_caption_cache_coverage_ratio") or 0.70
    )
    cache_hits = 0
    reused_caption_frames = 0
    fresh_caption_rows = 0
    fresh_observed_rows = 0

    def observe_coarse(
        window: list[float],
        frame_budget: int,
        *,
        preferred_timestamps: list[float],
    ) -> tuple[dict, str, dict]:
        nonlocal cache_hits, reused_caption_frames, fresh_caption_rows, fresh_observed_rows
        cache_audit: dict = {"sufficient": False, "cached_frame_count": 0}
        if cache_enabled:
            cached_rows, cache_audit = lookup_caption_rows(
                parameters=parameters,
                stage="coarse",
                goal=goal,
                window=window,
                expected_frames=frame_budget,
                goal_similarity_threshold=cache_goal_similarity,
                coverage_ratio_threshold=cache_coverage_ratio,
            )
            if cache_audit.get("sufficient"):
                cache_hits += 1
                reused_caption_frames += len(cached_rows)
                return (
                    _cached_observer_payload(
                        cached_rows,
                        window=window,
                        stage="coarse",
                        cache_audit=cache_audit,
                    ),
                    "",
                    cache_audit,
                )

        child_config = dict(config)
        child_config["skim_qwen_temporal_density_enabled"] = True
        child_config["skim_qwen_adaptive_density_enabled"] = False
        child_config["skim_qwen_density_max_frames"] = frame_budget
        child_parameters = dict(parameters)
        child_parameters.update(
            {
                "query": goal,
                "start_time": window[0],
                "end_time": window[1],
                "mode": "normal",
                "mandatory_timestamps": _sampling_anchors(
                    config,
                    parameters,
                    window=window,
                    frame_budget=frame_budget,
                    preferred=preferred_timestamps,
                ),
            }
        )
        output = execute_skim_qwen(child_config, child_parameters)
        payload = extract_v10_payload(output) or {}
        fresh_observed_rows += sum("qwen_omitted_frame" not in r.get("event_tags", []) for r in _timestamp_rows(payload))
        if cache_enabled and not _observer_payload_failed(payload):
            fresh_caption_rows += store_caption_rows(
                parameters=parameters,
                stage="coarse",
                goal=goal,
                rows=_timestamp_rows(payload),
            )
        return payload, output, cache_audit

    for index, (window, frame_budget) in enumerate(zip(search_windows, coarse_budgets), start=1):
        payload, output, cache_audit = observe_coarse(
            window,
            frame_budget,
            preferred_timestamps=preferred_anchor_timestamps_by_window[index - 1],
        )
        coarse_payloads.append((window, payload))
        coarse_raw.append(
            {
                "search_window": window,
                "frame_budget": frame_budget,
                "cache_audit": cache_audit,
                "payload": payload,
                "raw_output": output,
            }
        )

    boundary_expansion_strips: list[tuple[int, list[float]]] = []
    if boundary_expansion_enabled:
        boundary_expansion_strips = _boundary_expansion_strips(
            search_windows,
            [payload for _, payload in coarse_payloads],
            duration=duration,
            margin_s=float(
                config.get("localize_qwen_boundary_context_margin_s") or 8.0
            ),
            trigger_s=float(config.get("localize_qwen_boundary_trigger_s") or 2.0),
        )
        boundary_expansion_strips = boundary_expansion_strips[: max(2, top_k * 2)]

    if boundary_expansion_strips:
        expansion_windows = [window for _, window in boundary_expansion_strips]
        expansion_budgets = _allocate_duration_budgets(
            expansion_windows,
            total_budget=int(
                config.get("localize_qwen_boundary_expansion_max_frames") or 24
            ),
            minimum=2,
            target_fps=1.0,
            per_window_max=8,
        )
        additions_by_source: dict[int, list[dict]] = {}
        for (source_index, window), frame_budget in zip(
            boundary_expansion_strips,
            expansion_budgets,
        ):
            payload, output, cache_audit = observe_coarse(
                window,
                frame_budget,
                preferred_timestamps=[_window_center(window)],
            )
            additions_by_source.setdefault(source_index, []).append(payload)
            coarse_raw.append(
                {
                    "search_window": window,
                    "source_index": source_index,
                    "boundary_expansion": True,
                    "frame_budget": frame_budget,
                    "cache_audit": cache_audit,
                    "payload": payload,
                    "raw_output": output,
                }
            )

        merged_payloads: list[tuple[list[float], dict]] = []
        merged_search_windows: list[list[float]] = []
        for source_index, (window, payload) in enumerate(coarse_payloads):
            additions = additions_by_source.get(source_index) or []
            if additions:
                related_strips = [
                    strip
                    for index, strip in boundary_expansion_strips
                    if index == source_index
                ]
                merged_window = [
                    min([window[0], *[strip[0] for strip in related_strips]]),
                    max([window[1], *[strip[1] for strip in related_strips]]),
                ]
                payload = _merge_observer_payloads(
                    [payload, *additions],
                    window=merged_window,
                )
                window = [round(merged_window[0], 3), round(merged_window[1], 3)]
            merged_payloads.append((window, payload))
            merged_search_windows.append(window)
        coarse_payloads = merged_payloads
        search_windows = merged_search_windows

    expanded_anchor_recollection = False
    if tool_integrity_enabled and config.get(
        "coseek1_tool_recollect_expanded_anchors_enabled", False
    ):
        # Boundary expansion changes the actual search domain. Recollect prior
        # observations inside that final domain so a timestamp that was just
        # outside the planner's narrow request is not lost before fine scan.
        memory_anchor_rows_by_window = collect_memory_anchor_rows(
            parameters.get("memory"),
            search_windows,
            margin_s=0.0,
            per_window=int(
                config.get("coseek1_tool_memory_anchors_per_window") or 4
            ),
        )
        memory_anchor_timestamps_by_window = [
            [round(float(item["timestamp_s"]), 3) for item in rows]
            for rows in memory_anchor_rows_by_window
        ]
        preferred_anchor_timestamps_by_window = [
            sorted(
                {
                    requested_anchor_timestamps[index],
                    *memory_anchor_timestamps_by_window[index],
                }
            )
            for index in range(len(search_windows))
        ]
        expanded_anchor_recollection = True

    observer_status, observer_error_window_count = _observer_health(
        [payload for _, payload in coarse_payloads]
    )

    ranked = _rank_coarse_candidates(
        coarse_payloads,
        top_k=top_k,
        candidate_window_s=float(config.get("skim_qwen_candidate_window_s") or 18.0),
        row_candidates_enabled=bool(
            config.get("localize_qwen_row_candidate_windows", False)
        ),
        normalized_score_enabled=bool(
            config.get("localize_qwen_normalized_candidate_score_enabled", False)
        ),
        preserve_source_coverage=bool(
            config.get("localize_qwen_cover_search_windows", False)
        ),
        evidence_preserving=bool(
            config.get("localize_qwen_evidence_preserving_selection_enabled", False)
        ),
        ambiguous_reserve=int(
            config.get("localize_qwen_ambiguous_reserve") or 0
        ),
        retain_weak_source_candidates=p131_enabled,
    )
    fine_payload: dict = {}
    fine_raw = ""
    fine_cache_audits: list[dict] = []
    fine_cache_hits = 0
    fine_windows = []
    if ranked and observer_status != "unavailable":
        fine_total_budget = max(1, int(config.get("localize_qwen_fine_max_frames") or 24))
        fine_windows = [candidate["window"] for candidate in ranked[:max(1, fine_total_budget // 2)]]
        fine_budgets = _allocate_budget(fine_windows, fine_total_budget, minimum=2)
        cached_fine_payloads: list[dict] = []
        missing_fine_windows: list[list[float]] = []
        for window, per_window_budget in zip(fine_windows, fine_budgets):
            cache_audit: dict = {"sufficient": False, "cached_frame_count": 0}
            cached_rows: list[dict] = []
            if cache_enabled:
                cached_rows, cache_audit = lookup_caption_rows(
                    parameters=parameters,
                    stage="fine",
                    goal=goal,
                    window=window,
                    expected_frames=per_window_budget,
                    goal_similarity_threshold=cache_goal_similarity,
                    coverage_ratio_threshold=cache_coverage_ratio,
                )
            fine_cache_audits.append({"window": window, **cache_audit})
            if cache_audit.get("sufficient"):
                cache_hits += 1
                fine_cache_hits += 1
                reused_caption_frames += len(cached_rows)
                cached_fine_payloads.append(
                    _cached_observer_payload(
                        cached_rows,
                        window=window,
                        stage="fine",
                        cache_audit=cache_audit,
                    )
                )
            else:
                missing_fine_windows.append(window)

        fresh_fine_payload: dict = {}
        if missing_fine_windows:
            fine_config = dict(config)
            fine_config["focus_qwen_multi_window_enabled"] = True
            fine_config["focus_qwen_multi_window_max_windows"] = len(
                missing_fine_windows
            )
            fine_config["focus_qwen_temporal_density_enabled"] = True
            fine_config["focus_qwen_max_frames"] = max(fine_budgets)
            fine_parameters = dict(parameters)
            preferred_fine_timestamps = list(requested_anchor_timestamps)
            for anchors in memory_anchor_timestamps_by_window:
                preferred_fine_timestamps.extend(anchors)
            for window in missing_fine_windows:
                per_window_budget = fine_budgets[fine_windows.index(window)]
                preferred_fine_timestamps.extend(
                    _sampling_anchors(
                        config,
                        parameters,
                        window=window,
                        frame_budget=per_window_budget,
                        preferred=[_window_center(window)],
                    )
                )
            for candidate in ranked:
                for anchor in candidate.get("coarse_positive_anchors") or []:
                    if anchor.get("timestamp_s") is not None:
                        preferred_fine_timestamps.append(
                            float(anchor["timestamp_s"])
                        )
            fine_parameters.update(
                {
                    "query": goal,
                    "windows": missing_fine_windows,
                    "_frame_budgets": [fine_budgets[fine_windows.index(w)] for w in missing_fine_windows],
                    "start_time": min(window[0] for window in missing_fine_windows),
                    "end_time": max(window[1] for window in missing_fine_windows),
                    "mode": profile,
                    "mandatory_timestamps": preferred_fine_timestamps,
                }
            )
            fine_raw = execute_focus_qwen(fine_config, fine_parameters)
            fresh_fine_payload = extract_v10_payload(fine_raw) or {}
            fresh_observed_rows += sum("qwen_omitted_frame" not in r.get("event_tags", []) for r in _timestamp_rows(fresh_fine_payload))
            if cache_enabled and not _observer_payload_failed(fresh_fine_payload):
                fresh_caption_rows += store_caption_rows(
                    parameters=parameters,
                    stage="fine",
                    goal=goal,
                    rows=_timestamp_rows(fresh_fine_payload),
                )

        fine_parts = [*cached_fine_payloads]
        if fresh_fine_payload:
            fine_parts.append(fresh_fine_payload)
        if fine_parts:
            fine_payload = _merge_observer_payloads(
                fine_parts,
                window=[
                    min(window[0] for window in fine_windows),
                    max(window[1] for window in fine_windows),
                ],
            )

    strict_evidence_status = bool(
        config.get("localize_qwen_strict_evidence_status_enabled", False)
    )
    max_verify_window_s = float(config.get("localize_qwen_verify_window_s") or 12.0)
    verify_context_margin_s = max(
        0.0,
        float(config.get("localize_qwen_verify_context_margin_s") or 0.0),
    )
    ranked_candidates = []
    for index, candidate in enumerate(ranked, start=1):
        window = candidate["window"]
        source_index = int(candidate.get("source_index") or 0)
        rows = _fine_candidate_rows(fine_payload, window)
        # Include exact source times as well as rounded display labels. Neither
        # should fall outside the candidate after a padding/clamping operation.
        observed_times = [float(row[key]) for row in rows
                          for key in ("timestamp_s", "source_timestamp_s") if row.get(key) is not None]
        window = [min([window[0], *observed_times]), max([window[1], *observed_times])]
        candidate_fine_payload = dict(fine_payload, timestamp_observations=rows)
        relevant_rows = [
            row
            for row in rows
            if str(row.get("confidence") or row.get("query_relevance") or "low")
            .strip()
            .lower()
            in {"medium", "high"}
        ]
        fine_evidence = _candidate_evidence(candidate_fine_payload, window)
        if not rows:
            fine_evidence["evidence_class"] = "unobserved"
        coarse_evidence_class = str(
            candidate.get("coarse_evidence_class") or "ambiguous"
        )
        p131_fine_negative = bool(
            p131_enabled and fine_evidence["evidence_class"] == "negative"
        )
        mandatory_positive = bool(candidate.get("mandatory_positive")) or fine_evidence["evidence_class"] == "positive"
        summary_rows = relevant_rows if strict_evidence_status else rows
        summary = " ".join(
            str(row.get("description") or "").strip()
            for row in summary_rows
            if str(row.get("description") or "").strip()
        )[:480]
        tight_window = _tight_verify_window(
            window,
            rows,
            max_window_s=max_verify_window_s,
        )
        source_search_window = list(candidate.get("source_search_window") or window)
        verify_window = _expand_verify_window(
            tight_window,
            source_window=source_search_window,
            margin_s=verify_context_margin_s,
        )
        localized_anchors = [
            round(float(row.get("timestamp_s")), 3) for row in rows
        ][:12]
        confidence_rank = {"high": 2, "medium": 1, "low": 0}
        evidence_anchor_by_timestamp: dict[float, str] = {}
        for row in rows:
            timestamp = round(float(row.get("timestamp_s")), 3)
            confidence = str(
                row.get("confidence") or row.get("query_relevance") or "low"
            ).strip().lower()
            if confidence not in confidence_rank:
                confidence = "low"
            previous = evidence_anchor_by_timestamp.get(timestamp)
            if previous is None or confidence_rank[confidence] > confidence_rank[previous]:
                evidence_anchor_by_timestamp[timestamp] = confidence
        localized_evidence_anchors = [
            {"timestamp_s": timestamp, "confidence": confidence}
            for timestamp, confidence in sorted(evidence_anchor_by_timestamp.items())
            if confidence in {"medium", "high"}
        ][:12]
        high_confidence_count = sum(
            1
            for confidence in evidence_anchor_by_timestamp.values()
            if confidence == "high"
        )
        medium_confidence_count = sum(
            1
            for confidence in evidence_anchor_by_timestamp.values()
            if confidence == "medium"
        )
        evidence_preserving = bool(
            config.get("localize_qwen_evidence_preserving_selection_enabled", False)
        )
        if evidence_preserving:
            localization_status = (
                "found"
                if mandatory_positive
                else "ambiguous"
                if fine_evidence["evidence_class"] == "ambiguous"
                or coarse_evidence_class == "ambiguous"
                else "not_found"
            )
        elif strict_evidence_status:
            localization_status = (
                "found"
                if high_confidence_count > 0
                else "ambiguous"
                if medium_confidence_count > 0
                else "not_found"
            )
        else:
            localization_status = "found" if summary else "ambiguous"
        memory_anchors = (
            memory_anchor_timestamps_by_window[source_index]
            if source_index < len(memory_anchor_timestamps_by_window)
            else []
        )
        memory_anchors = sorted(
            {
                round(float(timestamp), 3)
                for timestamp in memory_anchors
                if source_search_window[0]
                <= float(timestamp)
                <= source_search_window[1]
            }
        )
        if config.get("coseek1_tool_recollect_expanded_anchors_enabled", False):
            verify_window = _expand_verify_window_to_nearby_anchors(
                verify_window,
                source_window=source_search_window,
                anchors=memory_anchors,
                max_gap_s=float(
                    config.get("coseek1_tool_memory_anchor_margin_s") or 2.0
                ),
                pad_s=float(config.get("coseek1_tool_memory_anchor_pad_s") or 0.25),
            )
        # Context expansion may still clip to the original search domain.
        # Keep the pixels actually inspected by this Fine request legal to cite.
        if observed_times:
            verify_window = [min(verify_window[0], min(observed_times)),
                             max(verify_window[1], max(observed_times))]
        verify_anchors = _verification_anchors(
            rows,
            localized_window=tight_window,
            verify_window=verify_window,
        )
        verify_anchors = sorted({*verify_anchors, *memory_anchors})
        coarse_positive_timestamps = [
            round(float(anchor["timestamp_s"]), 3)
            for anchor in candidate.get("coarse_positive_anchors") or []
            if anchor.get("timestamp_s") is not None
        ]
        localized_evidence_timestamps = {
            float(anchor["timestamp_s"])
            for anchor in localized_evidence_anchors
            if anchor.get("timestamp_s") is not None
        }
        for anchor in (
            candidate.get("coarse_positive_anchors") or []
        ):
            timestamp = round(float(anchor["timestamp_s"]), 3)
            if timestamp in localized_evidence_timestamps:
                continue
            localized_evidence_anchors.append(
                {
                    "timestamp_s": timestamp,
                    "confidence": str(
                        anchor.get("query_relevance") or "high"
                    ).strip().lower(),
                    "anchor_source": "coarse_explicit_positive",
                }
            )
            localized_evidence_timestamps.add(timestamp)
        localized_evidence_anchors.sort(key=lambda item: float(item["timestamp_s"]))
        coarse_ambiguous_timestamps = [
            round(float(anchor["timestamp_s"]), 3)
            for anchor in candidate.get("coarse_ambiguous_anchors") or []
            if anchor.get("timestamp_s") is not None
        ]
        # Ambiguous pixels remain retrievable without becoming positive evidence.
        verify_anchors = sorted({*verify_anchors, *coarse_positive_timestamps,
                                 *coarse_ambiguous_timestamps})
        fine_score_components = _candidate_score_components(candidate_fine_payload, window, 0)
        fine_score = (
            float(fine_score_components["score"])
            if rows
            else float(candidate["score"])
        )
        clue_rows = [
            dict(row, stage=stage)
            for stage, source_rows in (
                ("fine", fine_evidence["positive_anchors"] + fine_evidence["ambiguous_anchors"]),
                ("coarse", (candidate.get("coarse_positive_anchors") or []) + (candidate.get("coarse_ambiguous_anchors") or [])),
            ) for row in source_rows
        ]
        # A negative refinement remains inspectable, including its literal caption.
        if not clue_rows:
            clue_rows = [dict(row, stage="fine") for row in rows]
        fine_scan_status = "unavailable"
        if candidate["window"] not in fine_windows and observer_status != "unavailable":
            fine_scan_status = "deferred_budget"
        elif rows:
            fine_scan_status = ("partial" if any(
                "qwen_omitted_frame" in row.get("event_tags", []) for row in rows
            ) else "observed")
        ranked_candidates.append(
            {
                "candidate_id": "",
                "local_clues": clue_rows,
                "fine_observations": rows,
                "fine_scan_status": fine_scan_status,
                "rank": 0,
                "source_index": source_index,
                "score": fine_score,
                "coarse_score": candidate["score"],
                "fine_score": fine_score,
                "coarse_score_components": candidate.get("score_components") or {},
                "fine_score_components": fine_score_components,
                "t_range": window,
                "source_search_window": source_search_window,
                "localized_verify_window": tight_window,
                "recommended_verify_window": verify_window,
                "verify_context_margin_s": verify_context_margin_s,
                "fine_summary": (
                    summary
                    if summary
                    else candidate["coarse_summary"]
                ),
                "localized_timestamp_anchors": localized_anchors,
                "localized_evidence_anchors": localized_evidence_anchors,
                "coarse_positive_anchors": (
                    candidate.get("coarse_positive_anchors") or []
                ),
                "fine_positive_anchors": fine_evidence["positive_anchors"],
                "coarse_evidence_class": coarse_evidence_class,
                "fine_evidence_class": fine_evidence["evidence_class"],
                "selection_class": (
                    "mandatory_positive"
                    if mandatory_positive
                    else "weak_relevant"
                    if p131_fine_negative
                    else "ambiguous"
                ),
                "mandatory_positive": mandatory_positive,
                "memory_timestamp_anchors": memory_anchors,
                "timestamp_anchors": verify_anchors,
                "localization_status": localization_status,
                "verification_eligible": (
                    localization_status == "found"
                    or (evidence_preserving and localization_status == "ambiguous")
                ),
                "confidence_counts": {
                    "high": high_confidence_count,
                    "medium": medium_confidence_count,
                    "total": len(evidence_anchor_by_timestamp),
                },
                "evidence_level": "candidate",
            }
        )

    ranked_candidates.sort(
        key=lambda item: (
            1 if item.get("mandatory_positive") else 0,
            float(item.get("score") or 0.0),
            -float((item.get("t_range") or [0.0])[0]),
        ),
        reverse=True,
    )
    for index, candidate in enumerate(ranked_candidates, start=1):
        candidate["candidate_id"] = f"LQ{index:03d}"
        candidate["rank"] = index

    novelty_audit = {"status": "not_assessed"}
    if config.get("localize_qwen_candidate_novelty_enabled", False):
        novelty_audit = annotate_candidate_novelty(
            parameters=parameters,
            goal=goal,
            candidates=ranked_candidates,
            overlap_threshold=float(
                config.get("localize_qwen_candidate_novelty_overlap") or 0.80
            ),
            goal_similarity_threshold=float(
                config.get("localize_qwen_caption_cache_goal_similarity") or 0.30
            ),
        )
    found_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate.get("localization_status") == "found"
    ]
    verification_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate.get("verification_eligible") is True
    ]
    new_found_count = sum(
        1
        for candidate in found_candidates
        if candidate.get("novelty_status") == "new"
    )
    ambiguous_count = sum(
        1
        for candidate in ranked_candidates
        if candidate.get("localization_status") == "ambiguous"
    )
    localization_progress = (
        "not_found"
        if not found_candidates and ambiguous_count == 0
        else "ambiguous_candidates"
        if not found_candidates
        else "new_candidates"
        if new_found_count > 0
        else "refreshed_existing_candidates"
        if fresh_observed_rows > 0
        else "no_new_information"
    )

    observation_index = _observation_index([payload for _, payload in coarse_payloads] + [fine_payload], goal)
    if not config.get("localize_qwen_candidate_novelty_enabled", False):
        localization_progress = "observations_received" if observation_index["frame_indices"] or observation_index["unbound_rows"] else "no_observations"

    observer_wall_s = sum(
        float(payload.get("observer_wall_s") or 0.0) for _, payload in coarse_payloads
    ) + float(fine_payload.get("observer_wall_s") or 0.0)
    coarse_frames = sum(int(payload.get("num_frames") or 0) for _, payload in coarse_payloads)
    fine_frames = int(fine_payload.get("num_frames") or 0)
    if fine_payload and (_observer_payload_failed(fine_payload) or fine_payload.get("observer_status") == "partial"):
        observer_status = "partial" if observer_status == "complete" else observer_status
        observer_error_window_count += int(_observer_payload_failed(fine_payload))
    recommended = [
        candidate["recommended_verify_window"] for candidate in verification_candidates
    ]
    covered_source_indices = sorted(
        {
            int(candidate.get("source_index") or 0)
            for candidate in ranked_candidates
        }
    )
    uncovered_requested_windows = [
        requested_search_windows[index]
        for index in range(len(requested_search_windows))
        if index not in covered_source_indices
    ]
    compact_rows = [row for candidate in ranked_candidates
                    for row in candidate["fine_observations"]]

    trace = {
        "tool": "localize_qwen",
        "localization_goal": goal,
        "evidence_profile": profile,
        "requested_search_windows": requested_search_windows,
        "requested_anchor_timestamps": requested_anchor_timestamps,
        "memory_anchor_rows_by_window": memory_anchor_rows_by_window,
        "preferred_anchor_timestamps_by_window": preferred_anchor_timestamps_by_window,
        "expanded_anchor_recollection": expanded_anchor_recollection,
        "search_windows": search_windows,
        "search_context_margin_s": search_context_margin_s,
        "boundary_expansion_strips": boundary_expansion_strips,
        "coarse": coarse_raw,
        "ranked_coarse_candidates": ranked,
        "fine": {
            "payload": fine_payload,
            "raw_output": fine_raw,
            "cache_audits": fine_cache_audits,
        },
        "ranked_candidates": ranked_candidates,
        "localization_progress": localization_progress,
        "candidate_novelty": novelty_audit,
        "observation_index": observation_index,
    }
    trace_path = ""
    if config.get("localize_qwen_trace_enabled", True):
        trace_path = _write_trace(str(parameters.get("output_dir") or ""), trace)

    payload = {
        "tool": "localize_qwen",
        "window_id": "localize_qwen_result",
        "scene_id": "multi_scene",
        "t_range": [search_windows[0][0], search_windows[-1][1]],
        "requested_search_windows": requested_search_windows,
        "searched_windows": search_windows,
        "coverage": {
            "requested_window_count": len(requested_search_windows),
            "covered_requested_window_count": len(covered_source_indices),
            "covered_source_indices": covered_source_indices,
            "uncovered_requested_windows": uncovered_requested_windows,
            "memory_anchor_count": sum(
                len(rows) for rows in memory_anchor_rows_by_window
            ),
            "expanded_anchor_recollection": expanded_anchor_recollection,
            "search_context_margin_s": search_context_margin_s,
            "requested_top_k": requested_top_k,
            "effective_top_k": top_k,
            "coarse_frame_budget": coarse_budget,
            "coarse_frames": coarse_frames,
            "fine_frame_budget": int(config.get("localize_qwen_fine_max_frames") or 24),
            "fine_frames": fine_frames,
            "fine_observed_frames": int(fine_payload.get("observed_frame_count", fine_frames) or 0),
            "fine_deferred_candidate_ids": [c["candidate_id"] for c in ranked_candidates if c["fine_scan_status"] == "deferred_budget"],
            "observer_error_window_count": observer_error_window_count,
            "boundary_expansion_count": len(boundary_expansion_strips),
            "caption_cache_hits": cache_hits,
            "fine_caption_cache_hits": fine_cache_hits,
            "reused_caption_frames": reused_caption_frames,
            "fresh_caption_rows": fresh_caption_rows,
            "fresh_observed_rows": fresh_observed_rows,
            "found_candidate_count": len(found_candidates),
            "verification_candidate_count": len(verification_candidates),
            "mandatory_positive_count": sum(
                1 for candidate in ranked_candidates if candidate.get("mandatory_positive")
            ),
            "new_found_candidate_count": new_found_count if config.get("localize_qwen_candidate_novelty_enabled", False) else None,
            "ambiguous_candidate_count": ambiguous_count,
        },
        "ranked_candidates": ranked_candidates,
        "memory_anchor_timestamps_by_window": memory_anchor_timestamps_by_window,
        "uncovered_requested_windows": uncovered_requested_windows,
        "localization_progress": localization_progress,
        "candidate_novelty": novelty_audit,
        "observation_index": observation_index,
        "alternative_search_recommended": localization_progress
        in {"not_found", "ambiguous_candidates", "no_new_information"},
        "suggest_focus_windows": recommended,
        "suggest_focus_window": recommended[0] if recommended else None,
        "suggest_frame_verify_query": goal,
        "timestamp_observations": compact_rows,
        "scene_summaries": [
            {
                "scene_id": "localize_qwen",
                "window_id": candidate["candidate_id"],
                "t_range": candidate["t_range"],
                "summary": candidate["fine_summary"],
                "suggest_focus_windows": [candidate["recommended_verify_window"]],
            }
            for candidate in ranked_candidates
        ],
        "overall_summary": " | ".join(
            f"{candidate['candidate_id']} {candidate['t_range']}: {candidate['fine_summary']}"
            for candidate in ranked_candidates
        )[:1200],
        "possible_evidence": bool(verification_candidates),
        "detail_sufficient": False,
        "supports_options": [],
        "contradicts_options": [],
        "missing_detail": (
            "Local Qwen was unavailable; no negative visual inference may be drawn from this scan."
            if observer_status == "unavailable"
            else "Some local-Qwen captions are missing or failed; absence is unknown for those frames. Use only returned visual observations."
            if observer_status == "partial"
            else "This localization added no new visual information; inspect a different scene or time region."
            if localization_progress == "no_new_information"
            else "Only partial target context was found; inspect another candidate region before verification."
            if localization_progress == "ambiguous_candidates"
            else "Use frame_verify on a recommended window for the final visual decision."
            if verification_candidates
            else "No query-relevant local candidate was found in the searched windows."
        ),
        "evidence_level": "candidate",
        "observer_backend": (
            "local_qwen_error"
            if observer_status == "unavailable"
            else "local_qwen_partial"
            if observer_status == "partial"
            else "local_qwen"
        ),
        "observer_status": observer_status,
        "observer_wall_s": round(observer_wall_s, 3),
        "num_frames": coarse_frames + fine_frames,
        "coarse_frames": coarse_frames,
        "fine_frames": fine_frames,
        "internal_qwen_calls": sum(
            int(payload.get("observer_batch_count") or 1) for _, payload in coarse_payloads
        ) + int(fine_payload.get("observer_batch_count") or (1 if fine_payload else 0)),
        "trace_path": trace_path,
        "parse_ok": observer_status != "unavailable"
        and all(payload.get("parse_ok") is not False for _, payload in coarse_payloads)
        and (not fine_payload or fine_payload.get("parse_ok") is not False),
        "compact_render": True,
    }
    return format_v10_observation(payload, fallback_text=payload["overall_summary"])
