import re
from copy import deepcopy
from typing import Any


_QUERY_STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "between", "by",
    "check", "clearly", "confirm", "describe", "determine", "do", "does",
    "during", "for", "frame", "frames", "from", "if", "in", "inspect",
    "is", "it", "look", "of", "on", "or", "report", "see", "show", "shown",
    "specifically", "state", "that", "the", "their", "there", "these", "this",
    "time", "timestamp", "timestamps", "to", "verify", "visible", "what",
    "whether", "window", "within", "with", "brief", "clearest", "exact",
    "identify", "justification", "locate", "note", "report", "short",
}


_QUERY_ALIASES = {
    "actions": "action",
    "actively": "active",
    "bystanders": "bystander",
    "engaged": "participate",
    "engaging": "participate",
    "grappled": "grapple",
    "grappling": "grapple",
    "interacted": "interact",
    "interacting": "interact",
    "interaction": "interact",
    "interactions": "interact",
    "involved": "participate",
    "joined": "join",
    "joins": "join",
    "making": "make",
    "made": "make",
    "occurrences": "occurrence",
    "passersby": "bystander",
    "people": "person",
    "persons": "person",
    "physically": "physical",
    "pliers": "plier",
    "railings": "railing",
    "shoved": "shove",
    "shoving": "shove",
    "staircase": "stair",
    "stairs": "stair",
    "steps": "stair",
    "threaded": "thread",
    "threading": "thread",
    "tumbling": "tumble",
    "used": "use",
    "uses": "use",
    "using": "use",
    "beads": "bead",
}


def _safe_span(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return [round(start, 3), round(end, 3)]


def verification_windows(
    parameters: dict[str, Any] | None,
    *,
    context_pad_s: float = 0.0,
    duration: float | None = None,
) -> list[list[float]]:
    parameters = parameters or {}
    windows: list[list[float]] = []
    for item in parameters.get("candidate_windows") or []:
        value = item if isinstance(item, (list, tuple)) else (
            item.get("recommended_verify_window") or item.get("t_range")
            if isinstance(item, dict)
            else None
        )
        span = _safe_span(value)
        if span is not None and span not in windows:
            windows.append(span)
    if not windows:
        span = _safe_span(
            [parameters.get("start_time"), parameters.get("end_time")]
        )
        if span is not None:
            pad = max(0.0, float(context_pad_s or 0.0))
            if pad:
                span = [
                    max(0.0, span[0] - pad),
                    span[1] + pad,
                ]
                if duration is not None:
                    span[1] = min(max(0.0, float(duration)), span[1])
                span = [round(span[0], 3), round(span[1], 3)]
            windows.append(span)
    return sorted(windows)


def observed_windows(
    payload: dict[str, Any] | None,
    *,
    parameters: dict[str, Any] | None = None,
    context_pad_s: float = 0.0,
    duration: float | None = None,
) -> list[list[float]]:
    """Return the pixels actually covered by a completed verification.

    Direct frame verification expands a narrow planner request with temporal
    context. Cache entries therefore use the observer's returned ``t_range``
    instead of the unexpanded request. Multi-window verification keeps the
    explicit candidate windows because its aggregate ``t_range`` may span
    large uninspected gaps between disjoint candidates.
    """
    payload = payload or {}
    parameters = parameters or {}
    if parameters.get("candidate_windows"):
        verified: list[list[float]] = []
        for value in payload.get("verified_windows") or []:
            span = _safe_span(value)
            if span is not None and span not in verified:
                verified.append(span)
        return sorted(verified) or verification_windows(parameters)
    span = _safe_span(payload.get("t_range"))
    if span is not None:
        return [span]
    return verification_windows(
        parameters,
        context_pad_s=context_pad_s,
        duration=duration,
    )


def query_terms(value: Any) -> set[str]:
    normalized = str(value or "").replace("-", " ")
    words = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", normalized)
    }
    return {
        _QUERY_ALIASES.get(word, word)
        for word in words
        if word not in _QUERY_STOPWORDS
    }


def _span_overlap_ratio(left: list[float], right: list[float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    shorter = min(left[1] - left[0], right[1] - right[0])
    return overlap / max(0.001, shorter)


def _window_set_similarity(left: list[list[float]], right: list[list[float]]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    remaining = list(right)
    scores: list[float] = []
    for left_span in left:
        best_index, best_span = max(
            enumerate(remaining),
            key=lambda row: _span_overlap_ratio(left_span, row[1]),
        )
        best_score = _span_overlap_ratio(left_span, best_span)
        scores.append(best_score)
        remaining.pop(best_index)
    return min(scores) if scores else 0.0


def _query_similarity(left: set[str], right: set[str]) -> tuple[float, float]:
    if not left or not right:
        return 0.0, 0.0
    shared = left & right
    jaccard = len(shared) / max(1, len(left | right))
    current_coverage = len(shared) / max(1, len(right))
    return jaccard, current_coverage


def find_reusable_observation(
    cache: list[dict[str, Any]],
    *,
    parameters: dict[str, Any],
    overlap_threshold: float = 0.92,
    query_similarity_threshold: float = 0.55,
    context_pad_s: float = 0.0,
    duration: float | None = None,
) -> tuple[dict[str, Any] | None, dict[str, float]]:
    current_windows = verification_windows(
        parameters,
        context_pad_s=context_pad_s,
        duration=duration,
    )
    current_terms = query_terms(parameters.get("query"))
    best: tuple[tuple[float, float, float], dict[str, Any], dict[str, float]] | None = None
    for entry in cache:
        window_similarity = _window_set_similarity(
            entry.get("windows") or [], current_windows
        )
        if window_similarity < float(overlap_threshold):
            continue
        jaccard, current_coverage = _query_similarity(
            set(entry.get("query_terms") or []), current_terms
        )
        if not (
            jaccard >= float(query_similarity_threshold)
            or (
                current_coverage >= 0.8
                and jaccard >= max(0.35, float(query_similarity_threshold) - 0.15)
            )
        ):
            continue
        scores = {
            "window_overlap": round(window_similarity, 4),
            "query_jaccard": round(jaccard, 4),
            "current_query_coverage": round(current_coverage, 4),
        }
        key = (window_similarity, current_coverage, jaccard)
        if best is None or key > best[0]:
            best = (key, entry, scores)
    if best is None:
        return None, {}
    return best[1], best[2]


def make_cache_entry(
    *,
    cache_id: str,
    parameters: dict[str, Any],
    payload: dict[str, Any],
    context_pad_s: float = 0.0,
    duration: float | None = None,
) -> dict[str, Any]:
    return {
        "cache_id": cache_id,
        "query": str(parameters.get("query") or ""),
        "query_terms": sorted(query_terms(parameters.get("query"))),
        "windows": observed_windows(
            payload,
            parameters=parameters,
            context_pad_s=context_pad_s,
            duration=duration,
        ),
        "payload": deepcopy(payload),
        "reuse_count": 0,
    }


def reused_payload(
    entry: dict[str, Any],
    *,
    scores: dict[str, float],
) -> dict[str, Any]:
    entry["reuse_count"] = int(entry.get("reuse_count") or 0) + 1
    payload = deepcopy(entry.get("payload") or {})
    payload.update(
        {
            "observation_reused": True,
            "no_new_visual_information": True,
            "reuse_cache_id": entry.get("cache_id") or "",
            "reuse_count": entry["reuse_count"],
            "reuse_match": scores,
            "reuse_reason": (
                "The requested frame_verify covers the same pixels and a semantically "
                "equivalent visual question as an earlier API verification."
            ),
        }
    )
    return payload
