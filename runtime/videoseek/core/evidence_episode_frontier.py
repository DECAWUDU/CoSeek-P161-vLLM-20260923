from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


_SOURCE_STATUS = {
    "overview": "uninspected",
    "skim_qwen": "skimmed",
    "focus_qwen": "focused",
    "frame_verify": "verified",
    "focus": "verified",
}

_STATUS_RANK = {
    "uninspected": 0,
    "no_query_evidence": 1,
    "skimmed": 2,
    "focused": 3,
    "api_inspected_uncertain": 4,
    "partially_verified": 5,
    "verified": 6,
}

_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "onto", "that", "this",
    "what", "when", "where", "which", "who", "why", "how", "does", "did",
    "was", "were", "are", "after", "before", "then", "there", "video", "scene",
    "person", "people", "someone", "something", "shown", "visible", "view", "near",
    "option", "answer", "kind", "type", "color", "colour", "first", "second",
}

_GENERIC_ALIASES = {
    "animals": "animal",
    "animal": "animal",
    "livestock": "animal",
    "pets": "animal",
    "pet": "animal",
    "cages": "enclosure",
    "cage": "enclosure",
    "pens": "enclosure",
    "pen": "enclosure",
    "fencing": "enclosure",
    "fence": "enclosure",
    "enclosures": "enclosure",
    "enclosure": "enclosure",
    "kept": "enclosure",
    "keeping": "enclosure",
    "entering": "enter",
    "enters": "enter",
    "entered": "enter",
    "arrives": "enter",
    "arrived": "enter",
    "leaves": "leave",
    "left": "leave",
    "exits": "leave",
    "running": "run",
    "runs": "run",
    "walking": "walk",
    "walks": "walk",
    "dancing": "dance",
    "dances": "dance",
    "cooking": "cook",
    "cooks": "cook",
    "holding": "hold",
    "holds": "hold",
    "placing": "place",
    "places": "place",
    "putting": "place",
}

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def init_evidence_episode_frontier() -> dict[str, Any]:
    return {
        "version": "coseek_v31_evidence_episode_frontier_v1",
        "search_goal": "",
        "leading_hypothesis": "",
        "strongest_competitor": "",
        "decision_critical_evidence": "",
        "active_episode_id": None,
        "episodes": [],
        "event_boundaries": [],
        "answer_audits": [],
        "next_episode_index": 1,
    }


def ensure_evidence_episode_frontier(memory: dict[str, Any]) -> dict[str, Any]:
    state = memory.get("evidence_episode_frontier")
    if not isinstance(state, dict):
        state = init_evidence_episode_frontier()
        memory["evidence_episode_frontier"] = state
    for key, default in init_evidence_episode_frontier().items():
        if key not in state:
            state[key] = deepcopy(default)
    return state


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _valid_window(value: Any, *, duration: float | None = None) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start = _safe_float(value[0])
    end = _safe_float(value[1])
    if start is None or end is None:
        return None
    if duration is not None:
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
    if end <= start:
        return None
    return [round(start, 3), round(end, 3)]


def _tokens(text: str | None) -> set[str]:
    output: set[str] = set()
    for raw in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", (text or "").lower()):
        if raw in _STOPWORDS or len(raw) < 3:
            continue
        output.add(_GENERIC_ALIASES.get(raw, raw))
    return output


def _question_text(question: str | None) -> str:
    return re.sub(
        r"Please directly answer.*$",
        "",
        question or "",
        flags=re.IGNORECASE | re.DOTALL,
    )


def _query_score(text: str, question: str | None) -> int:
    return len(_tokens(text) & _tokens(_question_text(question)))


def _source_status(item: dict[str, Any]) -> str:
    source = str(item.get("source_tool") or "")
    status = _SOURCE_STATUS.get(source, "uninspected")
    if source in {"frame_verify", "focus"} and item.get("detail_sufficient") is False:
        return "api_inspected_uncertain"
    if source in {"skim_qwen", "focus_qwen"} and item.get("possible_evidence") is False:
        return "no_query_evidence"
    return status


def _window_overlap(left: list[float], right: list[float]) -> float:
    overlap = min(left[1], right[1]) - max(left[0], right[0])
    if overlap <= 0:
        return 0.0
    return overlap / max(1e-6, min(left[1] - left[0], right[1] - right[0]))


def _contains(outer: list[float], inner: list[float], tolerance_s: float = 0.5) -> bool:
    return outer[0] - tolerance_s <= inner[0] and outer[1] + tolerance_s >= inner[1]


def _number_mentions(text: str | None) -> set[int]:
    out = {int(value) for value in re.findall(r"\b\d+\b", text or "")}
    lowered = (text or "").lower()
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            out.add(value)
    return out


def _semantic_boundary(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, str]:
    left_text = str(left.get("description") or "")
    right_text = str(right.get("description") or "")
    left_numbers = _number_mentions(left_text)
    right_numbers = _number_mentions(right_text)
    if left_numbers and right_numbers and left_numbers != right_numbers:
        return True, "visible participant/object count changed"

    left_tags = set(str(tag).lower() for tag in left.get("event_tags") or [])
    right_tags = set(str(tag).lower() for tag in right.get("event_tags") or [])
    if left_tags and right_tags and left_tags.isdisjoint(right_tags):
        return True, "event tags changed"

    left_terms = _tokens(left_text)
    right_terms = _tokens(right_text)
    if len(left_terms) >= 2 and len(right_terms) >= 2:
        union = left_terms | right_terms
        similarity = len(left_terms & right_terms) / max(1, len(union))
        if similarity < 0.12:
            return True, "caption content changed"
    return False, ""


def _episode_id(
    state: dict[str, Any],
    old_by_key: dict[str, dict[str, Any]],
    key: str,
) -> str:
    existing = old_by_key.get(key)
    if existing and existing.get("episode_id"):
        return str(existing["episode_id"])
    index = int(state.get("next_episode_index") or 1)
    state["next_episode_index"] = index + 1
    return f"EP{index:03d}"


def _make_episode(
    state: dict[str, Any],
    old_by_key: dict[str, dict[str, Any]],
    *,
    key: str,
    t_range: list[float],
    origin: str,
    source_tool: str,
    scene_id: Any,
    window_id: Any,
    summary: str,
    possible_evidence: Any,
    missing_evidence: str,
    status: str,
    question: str | None,
    parent_episode_key: str | None = None,
    anchors: list[dict[str, Any]] | None = None,
    supports_options: list[Any] | None = None,
    contradicts_options: list[Any] | None = None,
    requirement_coverage: dict[str, Any] | None = None,
    detail_sufficient: Any = None,
) -> dict[str, Any]:
    text = " ".join([summary, missing_evidence])
    episode = {
        "episode_id": _episode_id(state, old_by_key, key),
        "episode_key": key,
        "parent_episode_key": parent_episode_key,
        "t_range": t_range,
        "duration_s": round(t_range[1] - t_range[0], 3),
        "origin": origin,
        "source_tool": source_tool,
        "source_tools": [source_tool],
        "scene_id": scene_id,
        "window_id": window_id,
        "summary": summary[:900],
        "possible_evidence": possible_evidence,
        "missing_evidence": missing_evidence[:600],
        "status": status,
        "evidence_scope": "local_episode",
        "query_match_count": _query_score(text, question),
        "anchors": anchors or [],
        "supports_options": deepcopy(supports_options or []),
        "contradicts_options": deepcopy(contradicts_options or []),
        "requirement_coverage": deepcopy(requirement_coverage or {}),
        "detail_sufficient": detail_sufficient,
        "inspected_windows": [],
    }
    old = old_by_key.get(key) or {}
    if old.get("planner_binding"):
        episode["planner_binding"] = deepcopy(old["planner_binding"])
    return episode


def _episode_relation_status(episode: dict[str, Any]) -> str:
    coverage = episode.get("requirement_coverage") or {}
    if isinstance(coverage, dict):
        relation = coverage.get("relation_verified")
        if relation is True and episode.get("status") in {"verified", "partially_verified"}:
            return "verified"
        if relation is False:
            return "unresolved"
    if episode.get("status") in {"skimmed", "focused"}:
        return "routing_only"
    return "unknown"


def _episode_recommended_tool(episode: dict[str, Any], *, focus_window_s: float) -> str:
    status = str(episode.get("status") or "uninspected")
    duration_s = float(episode.get("duration_s") or 0.0)
    precise = duration_s <= focus_window_s or bool(episode.get("anchors"))
    if status == "verified":
        return "none"
    if status == "api_inspected_uncertain":
        return "frame_verify_or_alternative_episode"
    if status == "partially_verified":
        return "inspect_unverified_part_or_alternative_episode"
    if status == "focused":
        return "frame_verify"
    if status == "skimmed":
        return "focus_qwen" if precise else "skim_qwen_or_alternative_episode"
    if status == "no_query_evidence":
        return "alternative_episode"
    return "focus_qwen" if precise else "skim_qwen"


def refresh_evidence_episode_frontier(
    memory: dict[str, Any],
    *,
    question: str | None = None,
    duration: float | None = None,
    max_episodes: int = 80,
    anchor_radius_s: float = 6.0,
    focus_window_s: float = 20.0,
) -> dict[str, Any]:
    """Compile window observations into persistent, local-scope event episodes.

    The compiler is representational. It does not block answers and does not
    infer that a short verified window proves a claim about the full video.
    """
    state = ensure_evidence_episode_frontier(memory)
    if question:
        state["search_goal"] = _question_text(question).split("\n", 1)[0][:800]
    old_episodes = [item for item in state.get("episodes") or [] if isinstance(item, dict)]
    old_by_key = {str(item.get("episode_key")): item for item in old_episodes}
    episodes: list[dict[str, Any]] = []
    scene_parent_keys: dict[tuple[str, str], str] = {}

    scene_items = [item for item in memory.get("scene_memory") or [] if isinstance(item, dict)]
    timestamp_items = [
        item for item in memory.get("timestamped_observations") or [] if isinstance(item, dict)
    ]

    for index, item in enumerate(scene_items):
        source = str(item.get("source_tool") or "")
        if source not in _SOURCE_STATUS:
            continue
        t_range = _valid_window(item.get("t_range"), duration=duration)
        if t_range is None:
            continue
        scene_id = str(item.get("scene_id") or "unknown")
        window_id = str(item.get("window_id") or "")
        key = f"window:{source}:{window_id or scene_id}:{t_range[0]:.3f}:{t_range[1]:.3f}"
        matching_anchors = [
            deepcopy(obs)
            for obs in timestamp_items
            if str(obs.get("source_tool") or "") == source
            and str(obs.get("scene_id") or "unknown") == scene_id
            and (window_id == "" or str(obs.get("window_id") or "") in {"", window_id})
            and (ts := _safe_float(obs.get("timestamp_s"))) is not None
            and t_range[0] - 0.5 <= ts <= t_range[1] + 0.5
        ]
        episode = _make_episode(
            state,
            old_by_key,
            key=key,
            t_range=t_range,
            origin="scene_window",
            source_tool=source,
            scene_id=item.get("scene_id"),
            window_id=item.get("window_id"),
            summary=str(item.get("summary") or ""),
            possible_evidence=item.get("possible_evidence"),
            missing_evidence=str(item.get("missing_detail") or ""),
            status=_source_status(item),
            question=question,
            anchors=matching_anchors,
            supports_options=item.get("supports_options") or [],
            contradicts_options=item.get("contradicts_options") or [],
            requirement_coverage=item.get("requirement_coverage") or {},
            detail_sufficient=item.get("detail_sufficient"),
        )
        episode["created_from_scene_index"] = index
        episodes.append(episode)
        scene_parent_keys[(source, scene_id)] = key

        raw_windows = item.get("suggest_focus_windows") or []
        if isinstance(raw_windows, (list, tuple)) and len(raw_windows) == 2 and all(
            isinstance(value, (int, float)) for value in raw_windows
        ):
            raw_windows = [raw_windows]
        for child_index, raw_window in enumerate(raw_windows if isinstance(raw_windows, list) else []):
            child_range = _valid_window(raw_window, duration=duration)
            if child_range is None:
                continue
            child_key = f"suggest:{key}:{child_range[0]:.3f}:{child_range[1]:.3f}"
            child_anchors = [
                deepcopy(obs)
                for obs in matching_anchors
                if (ts := _safe_float(obs.get("timestamp_s"))) is not None
                and child_range[0] - 0.5 <= ts <= child_range[1] + 0.5
            ]
            child = _make_episode(
                state,
                old_by_key,
                key=child_key,
                t_range=child_range,
                origin="suggested_window",
                source_tool=source,
                scene_id=item.get("scene_id"),
                window_id=item.get("window_id"),
                summary=str(item.get("summary") or ""),
                possible_evidence=item.get("possible_evidence"),
                missing_evidence=str(item.get("missing_detail") or ""),
                status=_SOURCE_STATUS.get(source, "uninspected"),
                question=question,
                parent_episode_key=key,
                anchors=child_anchors,
            )
            child["created_from_suggested_index"] = child_index
            episodes.append(child)

    # Promote query-relevant overview timestamps even when needs_focus is empty.
    for index, obs in enumerate(timestamp_items):
        source = str(obs.get("source_tool") or "")
        ts = _safe_float(obs.get("timestamp_s"))
        if source != "overview" or ts is None:
            continue
        text = " ".join(
            [
                str(obs.get("description") or ""),
                " ".join(str(tag) for tag in obs.get("event_tags") or []),
                str(obs.get("needs_focus") or ""),
            ]
        )
        query_score = _query_score(text, question)
        if not str(obs.get("needs_focus") or "").strip() and query_score <= 0:
            continue
        micro_range = _valid_window(
            [ts - max(0.5, anchor_radius_s), ts + max(0.5, anchor_radius_s)],
            duration=duration,
        )
        if micro_range is None:
            continue
        scene_id = str(obs.get("scene_id") or "unknown")
        parent_key = scene_parent_keys.get(("overview", scene_id))
        key = f"anchor:overview:{scene_id}:{ts:.3f}"
        episode = _make_episode(
            state,
            old_by_key,
            key=key,
            t_range=micro_range,
            origin="query_timestamp_anchor",
            source_tool="overview",
            scene_id=obs.get("scene_id"),
            window_id=obs.get("window_id"),
            summary=str(obs.get("description") or ""),
            possible_evidence=True,
            missing_evidence=str(obs.get("needs_focus") or "query-relevant timestamp needs inspection"),
            status="uninspected",
            question=question,
            parent_episode_key=parent_key,
            anchors=[deepcopy(obs)],
        )
        episode["query_match_count"] = query_score
        episode["created_from_observation_index"] = index
        episodes.append(episode)

    # Compile non-overview frame captions into local event segments. A segment
    # boundary is explicit state, not proof that two events are independent.
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for obs in timestamp_items:
        source = str(obs.get("source_tool") or "")
        if source not in {"skim_qwen", "focus_qwen", "frame_verify", "focus"}:
            continue
        key = (
            source,
            str(obs.get("window_id") or ""),
            str(obs.get("scene_id") or "unknown"),
        )
        grouped.setdefault(key, []).append(obs)

    boundaries: list[dict[str, Any]] = []
    for (source, window_id, scene_id), observations in grouped.items():
        observations.sort(key=lambda item: _safe_float(item.get("timestamp_s")) or 0.0)
        clusters: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for obs in observations:
            if current:
                is_boundary, reason = _semantic_boundary(current[-1], obs)
                if is_boundary:
                    left_ts = _safe_float(current[-1].get("timestamp_s"))
                    right_ts = _safe_float(obs.get("timestamp_s"))
                    boundaries.append(
                        {
                            "source_tool": source,
                            "window_id": window_id or None,
                            "scene_id": scene_id,
                            "left_timestamp_s": left_ts,
                            "right_timestamp_s": right_ts,
                            "reason": reason,
                            "continuity_status": "unresolved",
                        }
                    )
                    clusters.append(current)
                    current = []
            current.append(obs)
        if current:
            clusters.append(current)

        for cluster_index, cluster in enumerate(clusters):
            first_ts = _safe_float(cluster[0].get("timestamp_s"))
            last_ts = _safe_float(cluster[-1].get("timestamp_s"))
            if first_ts is None or last_ts is None:
                continue
            captions = " | ".join(str(item.get("description") or "") for item in cluster)
            relevant = any(
                str(item.get("confidence") or "").lower() in {"medium", "high"}
                or str(item.get("needs_focus") or "").strip()
                or _query_score(str(item.get("description") or ""), question) > 0
                for item in cluster
            )
            if not relevant:
                continue
            pad = 1.0 if first_ts != last_ts else 2.0
            cluster_range = _valid_window(
                [first_ts - pad, last_ts + pad],
                duration=duration,
            )
            if cluster_range is None:
                continue
            key = (
                f"segment:{source}:{window_id or scene_id}:"
                f"{first_ts:.3f}:{last_ts:.3f}:{cluster_index}"
            )
            parent_key = scene_parent_keys.get((source, scene_id))
            episode = _make_episode(
                state,
                old_by_key,
                key=key,
                t_range=cluster_range,
                origin="caption_event_segment",
                source_tool=source,
                scene_id=scene_id,
                window_id=window_id or None,
                summary=captions,
                possible_evidence=True,
                missing_evidence=(
                    "requires strong visual verification"
                    if source in {"skim_qwen", "focus_qwen"}
                    else ""
                ),
                status=_SOURCE_STATUS.get(source, "uninspected"),
                question=question,
                parent_episode_key=parent_key,
                anchors=[deepcopy(item) for item in cluster],
            )
            episodes.append(episode)

    # Add local inspection state to broader episodes without turning a short
    # verified clip into global verification of its parent scene.
    inspectors = [
        item
        for item in episodes
        if item.get("source_tool") in {"skim_qwen", "focus_qwen", "frame_verify", "focus"}
    ]
    for episode in episodes:
        episode_range = episode.get("t_range")
        if not isinstance(episode_range, list):
            continue
        for inspector in inspectors:
            if inspector is episode:
                continue
            inspector_range = inspector.get("t_range")
            if not isinstance(inspector_range, list) or _window_overlap(episode_range, inspector_range) <= 0:
                continue
            episode["inspected_windows"].append(
                {
                    "episode_id": inspector.get("episode_id"),
                    "t_range": deepcopy(inspector_range),
                    "source_tool": inspector.get("source_tool"),
                    "status": inspector.get("status"),
                }
            )
            if not _contains(inspector_range, episode_range):
                if inspector.get("status") == "verified" and episode.get("status") != "verified":
                    episode["status"] = "partially_verified"
                continue
            new_status = str(inspector.get("status") or "uninspected")
            old_status = str(episode.get("status") or "uninspected")
            if _STATUS_RANK.get(new_status, 0) > _STATUS_RANK.get(old_status, 0):
                episode["status"] = new_status

    for episode in episodes:
        episode["relation_status"] = _episode_relation_status(episode)
        episode["recommended_tool"] = _episode_recommended_tool(
            episode,
            focus_window_s=focus_window_s,
        )
        episode["is_broad"] = float(episode.get("duration_s") or 0.0) > focus_window_s

    episodes.sort(key=lambda item: (item["t_range"][0], item["t_range"][1], item["episode_id"]))
    state["episodes"] = episodes[: max(1, int(max_episodes))]
    state["event_boundaries"] = boundaries[-40:]
    return state


def record_episode_planner_proposal(
    memory: dict[str, Any],
    *,
    planner_payload: dict[str, Any] | None,
) -> None:
    if not isinstance(planner_payload, dict):
        return
    state = ensure_evidence_episode_frontier(memory)
    for key in ("leading_hypothesis", "strongest_competitor", "decision_critical_evidence"):
        if planner_payload.get(key) is not None:
            state[key] = str(planner_payload.get(key) or "")[:600]
    target = planner_payload.get("investigation_target")
    if not isinstance(target, dict):
        return
    episode_id = str(target.get("episode_id") or target.get("candidate_id") or "").strip()
    if not episode_id:
        return
    state["active_episode_id"] = episode_id
    binding = {
        "episode_id": episode_id,
        "discriminator": str(target.get("discriminator") or "")[:600],
        "expected_information": str(target.get("expected_information") or "")[:600],
    }
    for episode in state.get("episodes") or []:
        if episode.get("episode_id") == episode_id:
            episode["planner_binding"] = binding
            break


def _episode_priority(episode: dict[str, Any]) -> tuple[int, int, int, float]:
    status = str(episode.get("status") or "uninspected")
    open_status = status in {
        "uninspected", "skimmed", "focused", "api_inspected_uncertain", "partially_verified"
    }
    possible = episode.get("possible_evidence") is True
    query_match = int(episode.get("query_match_count") or 0)
    precise = not bool(episode.get("is_broad"))
    return (
        int(open_status and possible),
        int(open_status and query_match > 0),
        int(precise),
        -float(episode.get("t_range", [0.0])[0]),
    )


def _select_prompt_episodes(episodes: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    if len(episodes) <= max_items:
        return episodes
    selected: list[dict[str, Any]] = []
    for episode in sorted(episodes, key=_episode_priority, reverse=True):
        if episode not in selected:
            selected.append(episode)
        if len(selected) >= max_items:
            break
    return sorted(selected, key=lambda item: item["t_range"][0])


def format_evidence_episode_frontier_for_prompt(
    memory: dict[str, Any],
    *,
    max_episodes: int = 18,
) -> str:
    state = ensure_evidence_episode_frontier(memory)
    episodes = [item for item in state.get("episodes") or [] if isinstance(item, dict)]
    lines = [
        "Evidence Episode Frontier (investigation state, not an answer gate):",
        f"- search_goal={state.get('search_goal') or 'not set'}",
        f"- hypotheses={state.get('leading_hypothesis') or '?'} vs {state.get('strongest_competitor') or '?'}",
        f"- discriminator={state.get('decision_critical_evidence') or 'not set'}",
        f"- active_episode={state.get('active_episode_id') or 'none'}",
        "- Tool semantics: broad unresolved episode -> one logical skim_qwen; precise anchor -> focus_qwen; object/relation/option decision -> frame_verify. Recommendations are advisory.",
        "- A verified short episode has local scope and does not globally contradict uninspected episodes.",
    ]
    if not episodes:
        lines.append("Episodes: none; obtain overview first.")
        return "\n".join(lines)

    counts: dict[str, int] = {}
    for item in episodes:
        status = str(item.get("status") or "uninspected")
        counts[status] = counts.get(status, 0) + 1
    lines.append(f"- episode_status_counts={counts}")
    lines.append("Episodes selected for the current planner context:")
    for item in _select_prompt_episodes(episodes, max(1, int(max_episodes))):
        anchor_text = ", ".join(
            f"{float(anchor.get('timestamp_s')):.1f}s:{str(anchor.get('description') or '')[:70]}"
            for anchor in (item.get("anchors") or [])[:3]
            if _safe_float(anchor.get("timestamp_s")) is not None
        )
        lines.append(
            f"- {item.get('episode_id')} {item.get('t_range')} origin={item.get('origin')} "
            f"scene={item.get('scene_id') or '?'} status={item.get('status')} broad={item.get('is_broad')} "
            f"possible={item.get('possible_evidence')} query_matches={item.get('query_match_count')} "
            f"relation={item.get('relation_status')} next={item.get('recommended_tool')} "
            f"summary={str(item.get('summary') or '')[:180]} missing={str(item.get('missing_evidence') or '')[:120]} "
            f"anchors={anchor_text or 'none'}"
        )

    boundaries = state.get("event_boundaries") or []
    if boundaries:
        lines.append("Unresolved event boundaries from timestamped captions:")
        for item in boundaries[-8:]:
            lines.append(
                f"- {item.get('left_timestamp_s')}s -> {item.get('right_timestamp_s')}s "
                f"reason={item.get('reason')} continuity={item.get('continuity_status')}"
            )
    return "\n".join(lines)


def format_evidence_episode_frontier_for_answer(
    memory: dict[str, Any],
    *,
    max_episodes: int = 12,
) -> str:
    state = ensure_evidence_episode_frontier(memory)
    episodes = [item for item in state.get("episodes") or [] if isinstance(item, dict)]
    if not episodes:
        return ""
    selected = _select_prompt_episodes(episodes, max(1, int(max_episodes)))
    lines = [
        "Episode evidence digest:",
        "Short-window verification is local; compare it with alternative episodes before generalizing.",
    ]
    for item in selected:
        lines.append(
            f"- {item.get('episode_id')} {item.get('t_range')} status={item.get('status')} "
            f"scope={item.get('evidence_scope')} relation={item.get('relation_status')} "
            f"support={item.get('supports_options') or []} contradict={item.get('contradicts_options') or []} "
            f"summary={str(item.get('summary') or '')[:200]}"
        )
    return "\n".join(lines)


def matching_episode_for_range(
    memory: dict[str, Any],
    t_range: list[float],
) -> dict[str, Any] | None:
    state = ensure_evidence_episode_frontier(memory)
    matches = []
    for item in state.get("episodes") or []:
        item_range = item.get("t_range")
        if not isinstance(item_range, list):
            continue
        overlap = _window_overlap(item_range, t_range)
        if overlap <= 0:
            continue
        matches.append((overlap, -float(item.get("duration_s") or 0.0), item))
    if not matches:
        return None
    matches.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return matches[0][2]


def record_answer_evidence_audit(
    memory: dict[str, Any],
    *,
    answer: str | None,
    step: int,
    planner_payload: dict[str, Any] | None = None,
    max_items: int = 20,
) -> dict[str, Any]:
    """Record answer provenance after generation; never block or rewrite it."""
    state = ensure_evidence_episode_frontier(memory)
    match = re.search(r"\b([A-Z])\b", str(answer or "").upper())
    letter = match.group(1) if match else ""
    structured = memory.get("structured_evidence") or {}
    support_row = next(
        (
            row
            for row in structured.get("option_support") or []
            if str(row.get("option") or "").upper() == letter
        ),
        {},
    )
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in structured.get("evidence_items") or []
        if isinstance(item, dict) and item.get("evidence_id")
    }
    support_refs = list(support_row.get("supports") or [])
    weak_refs = list(support_row.get("weak_supports") or [])
    contradict_refs = list(support_row.get("contradicts") or [])
    support_items = [evidence_by_id.get(str(ref), {}) for ref in support_refs + weak_refs]
    verified_refs = [
        ref
        for ref in support_refs
        if str(evidence_by_id.get(str(ref), {}).get("evidence_level") or "") == "verified"
    ]
    if verified_refs:
        support_level = "verified"
    elif support_refs or weak_refs:
        levels = {str(item.get("evidence_level") or "") for item in support_items}
        support_level = "candidate" if levels & {"candidate", "uncertain"} else "routing"
    else:
        support_level = "none"

    qwen_only = bool(support_items) and all(
        str(item.get("source_tool") or "") in {"skim_qwen", "focus_qwen"}
        or str(item.get("backend") or "") == "local_qwen"
        for item in support_items
    )
    open_episodes = [
        item
        for item in state.get("episodes") or []
        if item.get("possible_evidence") is True
        and item.get("status") in {
            "uninspected", "skimmed", "focused", "api_inspected_uncertain", "partially_verified"
        }
        and int(item.get("query_match_count") or 0) > 0
    ]
    choices = [str(item.get("letter") or "") for item in structured.get("choices") or []]
    covered = {
        str(row.get("option") or "")
        for row in structured.get("option_support") or []
        if (row.get("supports") or row.get("weak_supports") or row.get("contradicts"))
    }
    relation_verified = any(
        item.get("relation_status") == "verified"
        and item.get("status") in {"verified", "partially_verified"}
        for item in state.get("episodes") or []
    )
    audit = {
        "audit_only": True,
        "step": int(step),
        "answer": letter or str(answer or "")[:40],
        "answer_support_level": support_level,
        "answer_support_refs": support_refs + weak_refs,
        "answer_verified_support_refs": verified_refs,
        "answer_contradict_refs": contradict_refs,
        "qwen_only_answer": qwen_only,
        "has_unverified_qwen_candidate": any(
            item.get("status") in {"skimmed", "focused"}
            and item.get("possible_evidence") is True
            for item in state.get("episodes") or []
        ),
        "uncovered_options": [letter for letter in choices if letter and letter not in covered],
        "uninspected_high_value_episode_ids": [
            str(item.get("episode_id")) for item in sorted(open_episodes, key=_episode_priority, reverse=True)[:8]
        ],
        "answer_evidence_scopes": sorted(
            {
                str(item.get("evidence_scope") or "unknown")
                for item in support_items
            }
        ),
        "relation_verified": relation_verified,
        "recommended_actions_ignored": [
            f"{item.get('episode_id')}:{item.get('recommended_tool')}"
            for item in sorted(open_episodes, key=_episode_priority, reverse=True)[:5]
            if item.get("recommended_tool") not in {None, "none"}
        ],
        "planner_leading_hypothesis": str((planner_payload or {}).get("leading_hypothesis") or "")[:300],
    }
    audits = state.setdefault("answer_audits", [])
    audits.append(audit)
    del audits[: max(0, len(audits) - max(1, int(max_items)))]
    memory["answer_evidence_audit"] = deepcopy(audit)
    return audit
