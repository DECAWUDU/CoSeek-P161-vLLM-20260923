from __future__ import annotations

from copy import deepcopy
from typing import Any


_VALID_STATUSES = {
    "uninspected",
    "skimmed",
    "focused",
    "verify_ready",
    "verified",
    "rejected",
}


def init_caption_planner_state() -> dict[str, Any]:
    return {
        "version": "caption_planner_v2",
        "localization_status": "unknown",
        "missing_evidence": "",
        "last_state": "",
        "candidates": [],
        "ranked_verify_windows": [],
    }


def ensure_caption_planner_state(memory: dict[str, Any]) -> dict[str, Any]:
    state = memory.get("caption_planner_state")
    if not isinstance(state, dict):
        state = init_caption_planner_state()
        memory["caption_planner_state"] = state
    state.setdefault("candidates", [])
    state.setdefault("ranked_verify_windows", [])
    return state


def _valid_window(value: Any, *, duration: float | None = None) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except Exception:
        return None
    if duration is not None:
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
    if end <= start:
        return None
    return [round(start, 3), round(end, 3)]


def _candidate_id(index: int) -> str:
    return f"C{index:03d}"


def _window_overlap(left: list[float], right: list[float]) -> float:
    overlap = min(left[1], right[1]) - max(left[0], right[0])
    if overlap <= 0:
        return 0.0
    return overlap / max(1e-6, min(left[1] - left[0], right[1] - right[0]))


def merge_caption_planner_payload(
    memory: dict[str, Any],
    payload: dict[str, Any],
    *,
    duration: float | None = None,
    max_candidates: int = 12,
) -> dict[str, Any]:
    """Merge planner-authored search state without judging visual semantics."""
    state = ensure_caption_planner_state(memory)
    if payload.get("state") is not None:
        state["last_state"] = str(payload.get("state") or "")[:500]
    localization = str(payload.get("localization_status") or "").strip().lower()
    if localization in {"found", "not_found", "ambiguous", "unknown"}:
        state["localization_status"] = localization
    if payload.get("missing_evidence") is not None:
        state["missing_evidence"] = str(payload.get("missing_evidence") or "")[:500]

    candidates = [item for item in state.get("candidates") or [] if isinstance(item, dict)]
    by_id = {str(item.get("candidate_id")): item for item in candidates if item.get("candidate_id")}
    updates = payload.get("candidate_updates") or []
    if not isinstance(updates, list):
        updates = []

    remapped_ids: dict[str, str] = {}
    for raw in updates:
        if not isinstance(raw, dict):
            continue
        window = _valid_window(
            raw.get("t_range") or raw.get("window") or raw.get("time_range"),
            duration=duration,
        )
        requested_id = str(raw.get("candidate_id") or "").strip().upper()
        candidate_id = requested_id
        if not candidate_id:
            candidate_id = _candidate_id(len(candidates) + 1)
        existing = by_id.get(candidate_id)
        if (
            existing is not None
            and window is not None
            and _valid_window(existing.get("t_range")) is not None
            and _window_overlap(window, _valid_window(existing.get("t_range"))) < 0.5
        ):
            next_index = len(candidates) + 1
            candidate_id = _candidate_id(next_index)
            while candidate_id in by_id:
                next_index += 1
                candidate_id = _candidate_id(next_index)
            existing = None
        if requested_id:
            remapped_ids[requested_id] = candidate_id
        if existing is None:
            if window is None:
                continue
            existing = {"candidate_id": candidate_id, "t_range": window}
            candidates.append(existing)
            by_id[candidate_id] = existing
        elif window is not None:
            existing["t_range"] = window

        for key in ("query", "reason", "source_observation", "status"):
            if raw.get(key) is not None:
                existing[key] = str(raw.get(key) or "")[:500]
        status = str(existing.get("status") or "uninspected").lower()
        existing["status"] = status if status in _VALID_STATUSES else "uninspected"
        try:
            existing["priority"] = int(raw.get("priority", existing.get("priority", 999)))
        except Exception:
            existing["priority"] = 999

    ranked: list[dict[str, Any]] = []
    raw_ranked = payload.get("ranked_verify_windows") or []
    if isinstance(raw_ranked, list):
        for rank, raw in enumerate(raw_ranked, start=1):
            if not isinstance(raw, dict):
                continue
            candidate_id = str(raw.get("candidate_id") or "").strip().upper()
            candidate_id = remapped_ids.get(candidate_id, candidate_id)
            candidate = by_id.get(candidate_id)
            window = _valid_window(
                raw.get("t_range") or raw.get("window") or (candidate or {}).get("t_range"),
                duration=duration,
            )
            if window is None:
                continue
            ranked.append(
                {
                    "rank": rank,
                    "candidate_id": candidate_id or None,
                    "t_range": window,
                    "reason": str(raw.get("reason") or "")[:300],
                }
            )
    state["ranked_verify_windows"] = ranked
    state["candidates"] = sorted(
        candidates,
        key=lambda item: (int(item.get("priority", 999)), float((item.get("t_range") or [1e12])[0])),
    )[: max(1, int(max_candidates))]
    return state


def resolve_candidate_windows(
    memory: dict[str, Any],
    candidate_ids: list[Any],
    *,
    max_windows: int = 3,
) -> list[dict[str, Any]]:
    state = ensure_caption_planner_state(memory)
    by_id = {
        str(item.get("candidate_id")): item
        for item in state.get("candidates") or []
        if isinstance(item, dict) and item.get("candidate_id")
    }
    resolved: list[dict[str, Any]] = []
    for value in candidate_ids[: max(1, int(max_windows))]:
        candidate_id = str(value).strip().upper()
        item = by_id.get(candidate_id)
        if not item:
            continue
        window = _valid_window(item.get("t_range"))
        if window is None:
            continue
        resolved.append(
            {
                "candidate_id": candidate_id,
                "t_range": window,
                "query": str(item.get("query") or ""),
                "reason": str(item.get("reason") or ""),
            }
        )
    return deepcopy(resolved)


def format_caption_planner_state(memory: dict[str, Any], *, max_candidates: int = 12) -> str:
    state = ensure_caption_planner_state(memory)
    candidates = state.get("candidates") or []
    if not candidates and not state.get("last_state"):
        return ""
    lines = [
        "Caption Planner State:",
        f"- localization_status={state.get('localization_status', 'unknown')}",
        f"- missing_evidence={state.get('missing_evidence') or 'none'}",
    ]
    if state.get("last_state"):
        lines.append(f"- planner_state={state.get('last_state')}")
    if candidates:
        lines.append("Candidate Pool (planner-authored from timestamped captions):")
        for item in candidates[: max(1, int(max_candidates))]:
            lines.append(
                f"- {item.get('candidate_id')} {item.get('t_range')} "
                f"status={item.get('status', 'uninspected')} priority={item.get('priority', 999)} "
                f"query={item.get('query') or ''} reason={item.get('reason') or ''}"
            )
    ranked = state.get("ranked_verify_windows") or []
    if ranked:
        lines.append("Planner-ranked verify windows:")
        for item in ranked:
            lines.append(
                f"- rank={item.get('rank')} candidate={item.get('candidate_id')} "
                f"window={item.get('t_range')} reason={item.get('reason') or ''}"
            )
    return "\n".join(lines)
