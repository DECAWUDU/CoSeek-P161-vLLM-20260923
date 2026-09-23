from __future__ import annotations

import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any


_MAX_VIDEOS = 32
_MAX_ROWS_PER_VIDEO = 2048
_MAX_CANDIDATES_PER_VIDEO = 96

_FRAME_CACHE: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
_CANDIDATE_HISTORY: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

_TOKEN_ALIASES = {
    "individual": "person",
    "individuals": "person",
    "participant": "person",
    "participants": "person",
    "people": "person",
    "persons": "person",
    "interacted": "interaction",
    "interacting": "interaction",
    "interacts": "interaction",
    "grappling": "interaction",
    "shoving": "interaction",
    "striking": "interaction",
    "altercation": "interaction",
    "number": "count",
    "counting": "count",
    "view": "visible",
    "shown": "visible",
    "shots": "frame",
    "moments": "frame",
    "timestamps": "frame",
}


def _video_key(parameters: dict[str, Any]) -> str:
    video_path = str(parameters.get("video_path") or "").strip()
    if video_path:
        try:
            return str(Path(video_path).resolve())
        except Exception:
            return video_path
    return f"video_reader:{id(parameters.get('vr'))}"


def _tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", str(text or "").lower()):
        if len(token) <= 1:
            continue
        tokens.add(_TOKEN_ALIASES.get(token, token))
    return tokens


def text_similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 1.0 if str(left or "").strip() == str(right or "").strip() else 0.0
    intersection = len(left_tokens & right_tokens)
    jaccard = intersection / len(left_tokens | right_tokens)
    containment = intersection / min(len(left_tokens), len(right_tokens))
    # Planner follow-up goals are often narrower paraphrases of the same task.
    # Containment keeps those compatible without treating unrelated goals as equal.
    return max(jaccard, 0.75 * containment)


def _task_context(parameters: dict[str, Any], goal: str) -> str:
    return str(parameters.get("question") or goal or "").strip()


def _bounded_bucket(
    store: OrderedDict[str, list[dict[str, Any]]],
    video_key: str,
) -> list[dict[str, Any]]:
    bucket = store.setdefault(video_key, [])
    store.move_to_end(video_key)
    while len(store) > _MAX_VIDEOS:
        store.popitem(last=False)
    return bucket


def reset_local_observation_cache() -> None:
    _FRAME_CACHE.clear()
    _CANDIDATE_HISTORY.clear()


def store_caption_rows(
    *,
    parameters: dict[str, Any],
    stage: str,
    goal: str,
    rows: list[dict[str, Any]],
) -> int:
    video_key = _video_key(parameters)
    task_context = _task_context(parameters, goal)
    bucket = _bounded_bucket(_FRAME_CACHE, video_key)
    added = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("timestamp_s") is None:
            continue
        try:
            timestamp = round(float(row.get("timestamp_s")), 3)
        except (TypeError, ValueError):
            continue
        description = str(row.get("description") or "").strip()
        if not description:
            continue
        replacement = {
            "stage": str(stage),
            "goal": str(goal),
            "task_context": task_context,
            "timestamp_s": timestamp,
            "row": deepcopy(row),
        }
        matching_index = next(
            (
                index
                for index, old in enumerate(bucket)
                if old.get("stage") == stage
                and str(old.get("task_context") or old.get("goal") or "")
                == task_context
                and abs(float(old.get("timestamp_s") or -1.0) - timestamp) <= 0.05
            ),
            None,
        )
        if matching_index is None:
            bucket.append(replacement)
            added += 1
        elif bucket[matching_index].get("row") != replacement["row"]:
            bucket[matching_index] = replacement
            added += 1
    if len(bucket) > _MAX_ROWS_PER_VIDEO:
        del bucket[: len(bucket) - _MAX_ROWS_PER_VIDEO]
    return added


def lookup_caption_rows(
    *,
    parameters: dict[str, Any],
    stage: str,
    goal: str,
    window: list[float],
    expected_frames: int,
    goal_similarity_threshold: float = 0.72,
    coverage_ratio_threshold: float = 0.70,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    video_key = _video_key(parameters)
    task_context = _task_context(parameters, goal)
    bucket = _FRAME_CACHE.get(video_key) or []
    selected: dict[float, tuple[float, dict[str, Any]]] = {}
    for entry in bucket:
        if entry.get("stage") != stage:
            continue
        entry_task = str(entry.get("task_context") or entry.get("goal") or "")
        if text_similarity(task_context, entry_task) < 0.95:
            continue
        similarity = text_similarity(goal, str(entry.get("goal") or ""))
        if similarity < goal_similarity_threshold:
            continue
        timestamp = float(entry.get("timestamp_s") or -1.0)
        if not float(window[0]) <= timestamp <= float(window[1]):
            continue
        previous = selected.get(timestamp)
        if previous is None or similarity > previous[0]:
            selected[timestamp] = (similarity, deepcopy(entry.get("row") or {}))

    rows = [item[1] for _, item in sorted(selected.items())]
    expected = max(1, int(expected_frames))
    ratio = min(1.0, len(rows) / expected)
    span = max(0.1, float(window[1]) - float(window[0]))
    expected_gap = span / max(1, expected - 1)
    timestamps = [float(row.get("timestamp_s")) for row in rows]
    points = [float(window[0]), *timestamps, float(window[1])]
    max_gap = max(
        (right - left for left, right in zip(points, points[1:])),
        default=span,
    )
    sufficient = bool(
        rows
        and ratio >= float(coverage_ratio_threshold)
        and max_gap <= max(1.5, expected_gap * 2.75)
    )
    audit = {
        "video_key": video_key,
        "stage": stage,
        "cached_frame_count": len(rows),
        "expected_frame_count": expected,
        "coverage_ratio": round(ratio, 4),
        "max_gap_s": round(max_gap, 3),
        "task_context_matched": bool(rows),
        "sufficient": sufficient,
    }
    return rows, audit


def _window_overlap(left: list[float], right: list[float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    shortest = max(1e-6, min(left[1] - left[0], right[1] - right[0]))
    return overlap / shortest


def annotate_candidate_novelty(
    *,
    parameters: dict[str, Any],
    goal: str,
    candidates: list[dict[str, Any]],
    overlap_threshold: float = 0.80,
    goal_similarity_threshold: float = 0.30,
) -> dict[str, int]:
    video_key = _video_key(parameters)
    task_context = _task_context(parameters, goal)
    history = _bounded_bucket(_CANDIDATE_HISTORY, video_key)
    new_count = 0
    repeated_count = 0

    for candidate in candidates:
        window = candidate.get("recommended_verify_window") or candidate.get("t_range")
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            candidate["novelty_status"] = "new"
            new_count += 1
            continue
        current_window = [float(window[0]), float(window[1])]
        summary = str(candidate.get("fine_summary") or "")
        best: tuple[float, dict[str, Any]] | None = None
        for old in history:
            old_task = str(old.get("task_context") or old.get("goal") or "")
            if text_similarity(task_context, old_task) < 0.95:
                continue
            goal_similarity = text_similarity(goal, str(old.get("goal") or ""))
            if goal_similarity < goal_similarity_threshold:
                continue
            overlap = _window_overlap(current_window, list(old.get("window") or current_window))
            if overlap < overlap_threshold:
                continue
            summary_similarity = text_similarity(summary, str(old.get("summary") or ""))
            score = 0.65 * overlap + 0.20 * goal_similarity + 0.15 * summary_similarity
            if best is None or score > best[0]:
                best = (score, old)

        if best is None:
            candidate["novelty_status"] = "new"
            new_count += 1
        else:
            candidate["novelty_status"] = "repeated"
            candidate["previous_candidate_window"] = list(best[1].get("window") or [])
            candidate["candidate_reuse_score"] = round(best[0], 4)
            repeated_count += 1

        history.append(
            {
                "goal": str(goal),
                "task_context": task_context,
                "window": current_window,
                "summary": summary,
            }
        )

    if len(history) > _MAX_CANDIDATES_PER_VIDEO:
        del history[: len(history) - _MAX_CANDIDATES_PER_VIDEO]
    return {"new": new_count, "repeated": repeated_count}
