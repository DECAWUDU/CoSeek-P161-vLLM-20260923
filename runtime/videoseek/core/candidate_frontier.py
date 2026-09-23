from __future__ import annotations

from copy import deepcopy
from typing import Any


_SOURCE_STATUS = {
    "overview": "uninspected",
    "skim_qwen": "skimmed",
    "focus_qwen": "focused",
    "frame_verify": "verified",
    "focus": "verified",
}


def init_candidate_frontier() -> dict[str, Any]:
    return {
        "version": "coseek_v30_investigation_frontier_v3",
        "search_goal": "",
        "leading_hypothesis": "",
        "strongest_competitor": "",
        "decision_critical_evidence": "",
        "active_investigation": {},
        "investigation_history": [],
        "candidates": [],
        "temporal_boundary_checks": [],
        "routing_audit": [],
        "next_candidate_index": 1,
    }


def ensure_candidate_frontier(memory: dict[str, Any]) -> dict[str, Any]:
    state = memory.get("candidate_frontier")
    if not isinstance(state, dict):
        state = init_candidate_frontier()
        memory["candidate_frontier"] = state
    state.setdefault("candidates", [])
    state.setdefault("temporal_boundary_checks", [])
    state.setdefault("routing_audit", [])
    state.setdefault("active_investigation", {})
    state.setdefault("investigation_history", [])
    state.setdefault("next_candidate_index", 1)
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


def _candidate_windows(item: dict[str, Any], *, duration: float | None) -> list[list[float]]:
    raw = item.get("suggest_focus_windows") or []
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and all(
        isinstance(value, (int, float)) for value in raw
    ):
        raw = [raw]
    windows: list[list[float]] = []
    if isinstance(raw, (list, tuple)):
        for value in raw:
            window = _valid_window(value, duration=duration)
            if window is not None:
                windows.append(window)
    if not windows:
        window = _valid_window(item.get("t_range"), duration=duration)
        if window is not None:
            windows.append(window)
    return windows


def _same_window(left: list[float], right: list[float], tolerance_s: float = 0.25) -> bool:
    return abs(left[0] - right[0]) <= tolerance_s and abs(left[1] - right[1]) <= tolerance_s


def _source_status(item: dict[str, Any]) -> str:
    source = str(item.get("source_tool") or "")
    status = _SOURCE_STATUS.get(source, "uninspected")
    if source in {"frame_verify", "focus"} and item.get("detail_sufficient") is False:
        return "api_inspected_uncertain"
    if source in {"skim_qwen", "focus_qwen"} and item.get("possible_evidence") is False:
        return "no_query_evidence"
    return status


def _coverage_text(item: dict[str, Any]) -> str:
    # The legacy overview annotator attaches global requirement coverage to its
    # final scene summary. That is not scene-local evidence, so do not expose it
    # as candidate-level coverage in the frontier.
    if item.get("source_tool") == "overview":
        return ""
    coverage = item.get("requirement_coverage")
    if not isinstance(coverage, dict):
        return ""
    matched = coverage.get("matched_requirements") or []
    missing = coverage.get("missing_requirements") or []
    relation = coverage.get("relation_verified")
    parts = []
    if matched:
        parts.append(f"matched={matched}")
    if missing:
        parts.append(f"missing={missing}")
    if relation is not None:
        parts.append(f"relation_verified={relation}")
    return "; ".join(parts)


def _coverage_ratio(inner: list[float], outer: list[float]) -> float:
    overlap = min(inner[1], outer[1]) - max(inner[0], outer[0])
    if overlap <= 0:
        return 0.0
    return overlap / max(1e-6, inner[1] - inner[0])


def _timestamp_cue_window(
    item: dict[str, Any],
    *,
    duration: float | None,
    radius_s: float,
) -> list[float] | None:
    try:
        timestamp_s = float(item.get("timestamp_s"))
    except Exception:
        return None
    radius_s = max(0.5, float(radius_s))
    return _valid_window(
        [max(0.0, timestamp_s - radius_s), timestamp_s + radius_s],
        duration=duration,
    )


def _is_timestamp_cue(item: dict[str, Any]) -> bool:
    # Qwen/API frame observations are already represented by their parent
    # window, and payload-level missing_detail may be copied onto every frame.
    # Promote only overview navigation cues so local frames cannot flood the
    # frontier or evict earlier global candidates.
    if str(item.get("source_tool") or "") != "overview":
        return False
    if not str(item.get("needs_focus") or "").strip():
        return False
    return True


def _refresh_temporal_boundary_checks(
    state: dict[str, Any],
    *,
    max_gap_s: float,
) -> None:
    cues = [
        item
        for item in state.get("candidates") or []
        if item.get("origin_kind") == "timestamp_cue"
        and isinstance(item.get("anchor_timestamp_s"), (int, float))
    ]
    cues.sort(key=lambda item: float(item["anchor_timestamp_s"]))
    checks: list[dict[str, Any]] = []
    for left, right in zip(cues, cues[1:]):
        gap_s = float(right["anchor_timestamp_s"]) - float(left["anchor_timestamp_s"])
        if gap_s < 0.0 or gap_s > max(0.0, float(max_gap_s)):
            continue
        checks.append(
            {
                "left_candidate_id": left.get("candidate_id"),
                "right_candidate_id": right.get("candidate_id"),
                "left_timestamp_s": left.get("anchor_timestamp_s"),
                "right_timestamp_s": right.get("anchor_timestamp_s"),
                "gap_s": round(gap_s, 3),
                "left_scene_id": left.get("scene_id"),
                "right_scene_id": right.get("scene_id"),
                "continuity_status": "unresolved",
            }
        )
    state["temporal_boundary_checks"] = checks


_STATUS_RANK = {
    "uninspected": 0,
    "no_query_evidence": 1,
    "skimmed": 2,
    "focused": 3,
    "api_inspected_uncertain": 4,
    "verified": 5,
}


def refresh_candidate_frontier(
    memory: dict[str, Any],
    *,
    question: str | None = None,
    duration: float | None = None,
    max_candidates: int = 24,
    include_timestamp_cues: bool = True,
    timestamp_cue_radius_s: float = 6.0,
    include_evidence_scope: bool = True,
    include_temporal_boundaries: bool = True,
    temporal_boundary_max_gap_s: float = 20.0,
) -> dict[str, Any]:
    """Merge observed windows into a persistent, planner-visible frontier.

    This function deliberately does not score semantic relevance or select the
    next window. It preserves observation provenance and inspection state so
    the planner can choose evidence that discriminates between hypotheses.
    """
    state = ensure_candidate_frontier(memory)
    if question:
        state["search_goal"] = str(question).split("\n", 1)[0][:800]

    candidates = [item for item in state.get("candidates") or [] if isinstance(item, dict)]
    for scene_index, item in enumerate(memory.get("scene_memory") or []):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_tool") or "")
        if source not in _SOURCE_STATUS:
            continue
        for window in _candidate_windows(item, duration=duration):
            existing = next(
                (
                    candidate
                    for candidate in candidates
                    if _same_window(candidate.get("t_range") or [], window)
                ),
                None,
            )
            if existing is None:
                next_index = int(state.get("next_candidate_index") or 1)
                existing = {
                    "candidate_id": f"CF{next_index:03d}",
                    "t_range": window,
                    "source_tool": source,
                    "source_tools": [source],
                    "scene_id": item.get("scene_id"),
                    "window_id": item.get("window_id"),
                    "created_from_scene_index": scene_index,
                    "origin_kind": "scene_window",
                    "origin_kinds": ["scene_window"],
                }
                state["next_candidate_index"] = next_index + 1
                candidates.append(existing)

            source_tools = list(existing.get("source_tools") or [])
            if source not in source_tools:
                source_tools.append(source)
            coverage_text = _coverage_text(item) or str(
                existing.get("requirement_coverage") or ""
            )

            existing.update(
                {
                    "source_tool": source,
                    "source_tools": source_tools,
                    "scene_id": item.get("scene_id") or existing.get("scene_id"),
                    "window_id": item.get("window_id") or existing.get("window_id"),
                    "summary": str(item.get("summary") or "")[:800],
                    "missing_evidence": str(item.get("missing_detail") or "")[:500],
                    "possible_evidence": item.get("possible_evidence"),
                    "detail_sufficient": item.get("detail_sufficient"),
                    "supports_options": deepcopy(item.get("supports_options") or []),
                    "contradicts_options": deepcopy(item.get("contradicts_options") or []),
                    "requirement_coverage": coverage_text,
                    "status": _source_status(item),
                }
            )
            if include_evidence_scope:
                existing["evidence_scope"] = "local_window"

    if include_timestamp_cues:
        for observation_index, item in enumerate(memory.get("timestamped_observations") or []):
            if not isinstance(item, dict) or not _is_timestamp_cue(item):
                continue
            window = _timestamp_cue_window(
                item,
                duration=duration,
                radius_s=timestamp_cue_radius_s,
            )
            if window is None:
                continue
            existing = next(
                (
                    candidate
                    for candidate in candidates
                    if _same_window(candidate.get("t_range") or [], window)
                ),
                None,
            )
            source = str(item.get("source_tool") or "overview")
            if existing is None:
                next_index = int(state.get("next_candidate_index") or 1)
                existing = {
                    "candidate_id": f"CF{next_index:03d}",
                    "t_range": window,
                    "source_tool": source,
                    "source_tools": [source],
                    "scene_id": item.get("scene_id"),
                    "window_id": item.get("window_id"),
                    "created_from_observation_index": observation_index,
                    "origin_kind": "timestamp_cue",
                    "origin_kinds": ["timestamp_cue"],
                    "anchor_timestamp_s": round(float(item.get("timestamp_s")), 3),
                    "event_tags": deepcopy(item.get("event_tags") or []),
                    "summary": str(item.get("description") or "")[:800],
                    "missing_evidence": str(item.get("needs_focus") or "")[:500],
                    "cue_question": str(item.get("needs_focus") or "")[:500],
                    "possible_evidence": True,
                    "detail_sufficient": None,
                    "supports_options": [],
                    "contradicts_options": [],
                    "requirement_coverage": "",
                    "status": _SOURCE_STATUS.get(source, "uninspected"),
                }
                state["next_candidate_index"] = next_index + 1
                candidates.append(existing)
            else:
                source_tools = list(existing.get("source_tools") or [])
                if source not in source_tools:
                    source_tools.append(source)
                existing["source_tools"] = source_tools
                origin_kinds = list(existing.get("origin_kinds") or [])
                previous_origin = existing.get("origin_kind")
                if previous_origin and previous_origin not in origin_kinds:
                    origin_kinds.append(previous_origin)
                if "timestamp_cue" not in origin_kinds:
                    origin_kinds.append("timestamp_cue")
                existing["origin_kind"] = "timestamp_cue"
                existing["origin_kinds"] = origin_kinds
                existing["anchor_timestamp_s"] = round(float(item.get("timestamp_s")), 3)
                existing["event_tags"] = deepcopy(item.get("event_tags") or [])
                existing["cue_question"] = str(item.get("needs_focus") or "")[:500]
                if not existing.get("missing_evidence") and existing.get("status") != "verified":
                    existing["missing_evidence"] = existing["cue_question"]
            if include_evidence_scope:
                existing["evidence_scope"] = "local_timestamp"

    inspectors = [
        item
        for item in candidates
        if item.get("status") in {
            "skimmed",
            "focused",
            "no_query_evidence",
            "api_inspected_uncertain",
            "verified",
        }
    ]
    for candidate in candidates:
        candidate_window = candidate.get("t_range")
        if not isinstance(candidate_window, list) or len(candidate_window) != 2:
            continue
        for inspector in inspectors:
            if inspector is candidate:
                continue
            inspector_window = inspector.get("t_range")
            if not isinstance(inspector_window, list) or len(inspector_window) != 2:
                continue
            if _coverage_ratio(candidate_window, inspector_window) < 0.8:
                continue
            new_status = str(inspector.get("status") or "uninspected")
            old_status = str(candidate.get("status") or "uninspected")
            if _STATUS_RANK.get(new_status, 0) <= _STATUS_RANK.get(old_status, 0):
                continue
            candidate["status"] = new_status
            candidate["inspected_by"] = inspector.get("candidate_id")

    state["candidates"] = candidates[-max(1, int(max_candidates)) :]
    if include_temporal_boundaries:
        _refresh_temporal_boundary_checks(
            state,
            max_gap_s=temporal_boundary_max_gap_s,
        )
    else:
        state["temporal_boundary_checks"] = []
    return state


def record_planner_proposal(
    memory: dict[str, Any],
    *,
    planner_payload: dict[str, Any] | None,
    include_investigation_state: bool = False,
    max_history_items: int = 24,
) -> None:
    if not isinstance(planner_payload, dict):
        return
    state = ensure_candidate_frontier(memory)
    mappings = {
        "leading_hypothesis": "leading_hypothesis",
        "strongest_competitor": "strongest_competitor",
        "decision_critical_evidence": "decision_critical_evidence",
    }
    for source_key, target_key in mappings.items():
        if planner_payload.get(source_key) is not None:
            state[target_key] = str(planner_payload.get(source_key) or "")[:600]

    if not include_investigation_state:
        return
    target = planner_payload.get("investigation_target")
    if not isinstance(target, dict):
        return
    candidate_id = str(target.get("candidate_id") or "").strip()
    discriminator = str(target.get("discriminator") or "").strip()[:600]
    expected_information = str(target.get("expected_information") or "").strip()[:600]
    if not candidate_id and not discriminator and not expected_information:
        return

    binding = {
        "candidate_id": candidate_id or None,
        "discriminator": discriminator,
        "expected_information": expected_information,
        "leading_hypothesis": state.get("leading_hypothesis") or "",
        "strongest_competitor": state.get("strongest_competitor") or "",
    }
    state["active_investigation"] = binding
    history = state.setdefault("investigation_history", [])
    if not history or history[-1] != binding:
        history.append(deepcopy(binding))
        del history[: max(0, len(history) - max(1, int(max_history_items)))]

    if candidate_id:
        candidate = next(
            (
                item
                for item in state.get("candidates") or []
                if str(item.get("candidate_id") or "") == candidate_id
            ),
            None,
        )
        if isinstance(candidate, dict):
            candidate["planner_binding"] = deepcopy(binding)


def _candidate_discriminator(item: dict[str, Any]) -> str:
    return str(
        item.get("cue_question")
        or item.get("missing_evidence")
        or ""
    ).strip()


def _open_discrimination_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_items: int,
) -> list[dict[str, Any]]:
    open_statuses = {"uninspected", "skimmed", "focused", "api_inspected_uncertain"}
    explicit_cues = []
    other_candidates = []
    for item in candidates:
        if str(item.get("status") or "uninspected") not in open_statuses:
            continue
        if not _candidate_discriminator(item):
            continue
        if item.get("possible_evidence") is False and item.get("status") != "api_inspected_uncertain":
            continue
        if item.get("cue_question"):
            explicit_cues.append(item)
        else:
            other_candidates.append(item)

    # Explicit overview cue questions are retained first because they can be
    # lost when the normal candidate list is truncated by recency. Within each
    # group, preserve temporal/creation order rather than assigning a semantic score.
    selected = explicit_cues + other_candidates
    return selected[: max(1, int(max_items))]


def _action_dict(action: Any) -> dict[str, Any] | None:
    if action is None:
        return None
    if isinstance(action, dict):
        return deepcopy(action)
    to_dict = getattr(action, "to_dict", None)
    if callable(to_dict):
        return deepcopy(to_dict())
    return None


def record_routing_audit(
    memory: dict[str, Any],
    *,
    step: int,
    planner_action: Any,
    executed_action: Any,
    reason: str,
    max_items: int = 40,
) -> dict[str, Any]:
    state = ensure_candidate_frontier(memory)
    proposed = _action_dict(planner_action)
    executed = _action_dict(executed_action)
    changed = proposed != executed
    entry = {
        "step": int(step),
        "planner_proposed_action": proposed,
        "router_executed_action": executed,
        "router_changed_action": changed,
        "router_reason": str(reason or ("planner_action_executed" if not changed else "runtime_policy"))[:800],
    }
    audit = state.setdefault("routing_audit", [])
    audit.append(entry)
    del audit[: max(0, len(audit) - max(1, int(max_items)))]
    return deepcopy(entry)


def format_candidate_frontier_for_prompt(
    memory: dict[str, Any],
    *,
    max_candidates: int = 16,
    max_audit_items: int = 5,
    include_investigation_state: bool = False,
    max_investigation_candidates: int = 12,
) -> str:
    state = ensure_candidate_frontier(memory)
    candidates = state.get("candidates") or []
    lines = [
        "Candidate Frontier (planner chooses; runtime does not rank by a fixed semantic score):",
        f"- search_goal={state.get('search_goal') or 'not set'}",
        f"- leading_hypothesis={state.get('leading_hypothesis') or 'not set'}",
        f"- strongest_competitor={state.get('strongest_competitor') or 'not set'}",
        f"- decision_critical_evidence={state.get('decision_critical_evidence') or 'not set'}",
    ]
    if include_investigation_state:
        lines.append(
            "Investigation Frontier (persistent state, not a gate or forced route):"
        )
        active = state.get("active_investigation") or {}
        if active:
            lines.append(
                f"- active_candidate={active.get('candidate_id') or 'none'} "
                f"tests={active.get('discriminator') or 'not set'} "
                f"expected_information={active.get('expected_information') or 'not set'} "
                f"hypotheses={active.get('leading_hypothesis') or '?'} vs "
                f"{active.get('strongest_competitor') or '?'}"
            )
        open_candidates = _open_discrimination_candidates(
            candidates,
            max_items=max_investigation_candidates,
        )
        if open_candidates:
            lines.append(
                "Open candidate-to-discriminator bindings. `tests` states what new "
                "information the candidate can supply; it is not verified evidence:"
            )
            for item in open_candidates:
                lines.append(
                    f"- {item.get('candidate_id')} {item.get('t_range')} "
                    f"status={item.get('status')} scope={item.get('evidence_scope') or 'unknown'} "
                    f"tests={_candidate_discriminator(item)} "
                    f"observed={item.get('summary') or 'none'}"
                )
        else:
            lines.append("Open candidate-to-discriminator bindings: none.")
    if candidates:
        status_counts: dict[str, int] = {}
        for item in candidates:
            status = str(item.get("status") or "uninspected")
            status_counts[status] = status_counts.get(status, 0) + 1
        uninspected_possible = [
            str(item.get("candidate_id"))
            for item in candidates
            if item.get("status") == "uninspected"
            and item.get("possible_evidence") is True
        ]
        lines.append(f"Candidate coverage: status_counts={status_counts}")
        lines.append(
            "Uninspected evidence-bearing candidates: "
            + (", ".join(uninspected_possible) if uninspected_possible else "none")
            + ". This is coverage context, not an instruction to inspect every window."
        )
        lines.append("Candidate windows:")
        for item in candidates[-max(1, int(max_candidates)) :]:
            anchor = (
                f" anchor={item.get('anchor_timestamp_s')}s"
                if item.get("anchor_timestamp_s") is not None
                else ""
            )
            cue = (
                f" cue={item.get('cue_question')}"
                if item.get("cue_question")
                else ""
            )
            lines.append(
                f"- {item.get('candidate_id')} {item.get('t_range')} "
                f"sources={item.get('source_tools') or [item.get('source_tool')]} scene={item.get('scene_id') or '?'} "
                f"origin={item.get('origin_kind') or 'window'} scope={item.get('evidence_scope') or 'unknown'}{anchor} "
                f"status={item.get('status')} possible={item.get('possible_evidence')} "
                f"coverage={item.get('requirement_coverage') or 'unknown'} "
                f"summary={item.get('summary') or 'none'} "
                f"missing={item.get('missing_evidence') or 'none'}{cue}"
            )
    else:
        lines.append("Candidate windows: none; obtain an overview first.")

    boundary_checks = state.get("temporal_boundary_checks") or []
    if boundary_checks:
        lines.append(
            "Temporal boundary checks (candidate cues remain distinct while continuity is unresolved; "
            "overlap caused by padded tool windows is not continuity evidence):"
        )
        for item in boundary_checks[-8:]:
            lines.append(
                f"- {item.get('left_candidate_id')}@{item.get('left_timestamp_s')}s -> "
                f"{item.get('right_candidate_id')}@{item.get('right_timestamp_s')}s "
                f"gap={item.get('gap_s')}s scenes={item.get('left_scene_id') or '?'}"
                f"->{item.get('right_scene_id') or '?'} continuity={item.get('continuity_status')}"
            )

    audit = state.get("routing_audit") or []
    if audit:
        lines.append("Recent planner/router audit:")
        for item in audit[-max(1, int(max_audit_items)) :]:
            proposed = (item.get("planner_proposed_action") or {}).get("function")
            executed = (item.get("router_executed_action") or {}).get("function")
            lines.append(
                f"- step={item.get('step')} proposed={proposed} executed={executed} "
                f"changed={item.get('router_changed_action')} reason={item.get('router_reason')}"
            )
    return "\n".join(lines)
