from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


def init_structured_evidence_state() -> dict[str, Any]:
    return {
        "version": "coseek_v14_structured_evidence_state",
        "evidence_items": [],
        "occurrence_views": [],
        "timeline_views": [],
        "object_state_views": [],
        "choices": [],
        "option_evidence": [],
        "unmapped_option_labels": [],
        "option_support": [],
        "uncertainties": [],
    }


def ensure_structured_evidence_state(memory: dict[str, Any]) -> dict[str, Any]:
    state = memory.get("structured_evidence")
    if not isinstance(state, dict):
        state = init_structured_evidence_state()
        memory["structured_evidence"] = state
    for key, default in init_structured_evidence_state().items():
        if key not in state:
            state[key] = deepcopy(default)
    return state


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _safe_range(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start = _safe_float(value[0])
    end = _safe_float(value[1])
    if start is None or end is None or end < start:
        return None
    return [round(start, 1), round(end, 1)]


def _extract_choices(question_context: str | None) -> list[dict[str, str]]:
    choices: list[dict[str, str]] = []
    for match in re.finditer(
        r"(?m)^\s*\(([A-Z])\)\s*(.+?)\s*$",
        question_context or "",
    ):
        choices.append(
            {
                "letter": match.group(1).upper(),
                "text": match.group(2).strip(),
            }
        )
    return choices


def _normalize_for_match(text: str | None) -> str:
    normalized = (text or "").lower()
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _choice_terms(text: str | None) -> set[str]:
    stop = {
        "the", "a", "an", "to", "of", "for", "with", "and", "or", "in",
        "on", "at", "by", "into", "from", "up",
    }
    return {
        token
        for token in _normalize_for_match(text).split()
        if len(token) >= 3 and token not in stop
    }


def _explicit_option_letters(text: str | None) -> list[str]:
    raw = str(text or "").strip()
    out: list[str] = []
    patterns = [
        r"(?i)\boption\s*([A-Z])\b",
        r"^\s*\(([A-Z])\)\s*",
        r"^\s*([A-Z])\s*[:.)-]\s*",
        r"^\s*([A-Z])\s*$",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, raw):
            letter = match.group(1).upper()
            if letter not in out:
                out.append(letter)
    return out


def _matched_choice_letters(text: str | None, choices: list[dict[str, str]]) -> list[str]:
    explicit = _explicit_option_letters(text)
    if explicit:
        return [letter for letter in explicit if any(choice["letter"] == letter for choice in choices)]

    normalized = _normalize_for_match(text)
    out: list[str] = []
    for choice in choices:
        choice_text = choice.get("text") or ""
        choice_norm = _normalize_for_match(choice_text)
        if choice_norm and re.search(rf"(?<![a-z0-9]){re.escape(choice_norm)}(?![a-z0-9])", normalized):
            out.append(choice["letter"])
            continue
        terms = _choice_terms(choice_text)
        if terms and terms.issubset(set(normalized.split())):
            out.append(choice["letter"])
    return out


def _text_negates_choice(text: str | None, choice_text: str | None) -> bool:
    normalized = _normalize_for_match(text)
    if not normalized:
        return False
    terms = _choice_terms(choice_text)
    if not terms:
        return False
    tokens = normalized.split()
    negators = {"no", "not", "non", "without", "absent", "never"}
    for idx, token in enumerate(tokens):
        if token not in terms:
            continue
        before = tokens[max(0, idx - 5):idx]
        if any(word in negators for word in before):
            return True
    return False


def _choice_text(choices: list[dict[str, str]], letter: str) -> str:
    for choice in choices:
        if choice.get("letter") == letter:
            return choice.get("text") or ""
    return ""


def _normalize_option_status(value: Any, *, detail_sufficient: Any, ambiguous: bool = False) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"supports", "supported", "strong_support", "yes", "true"}:
        raw = "support"
    if raw in {"contradicts", "contradicted", "refute", "refutes", "against", "no", "false"}:
        raw = "contradict"
    if raw in {"unknown", "unclear", "not_enough", "insufficient", "missing"}:
        raw = "unresolved"
    if raw not in {"support", "weak_support", "contradict", "unresolved"}:
        raw = "unresolved"
    if raw == "support" and (detail_sufficient is False or ambiguous):
        return "weak_support"
    return raw


def _payload_has_unverified_query_premise(payload: dict[str, Any]) -> bool:
    if payload.get("decision_sufficiency_validated") is True:
        return False
    missing = re.sub(
        r"\s+",
        " ",
        str(payload.get("missing_detail") or ""),
    ).strip(" .")
    if not missing:
        return False
    lowered = missing.lower()
    if lowered in {"none", "null", "n/a", "no missing detail", "nothing missing"}:
        return False
    if re.match(r"^(none|null|n/a)\s*[,;:]", lowered):
        return False
    patterns = [
        "cannot verify",
        "can't verify",
        "cannot confirm",
        "can't confirm",
        "premise",
        "context",
        "not shown",
        "not visible",
        "no kitchen",
        "no cooking",
        "missing required",
        "missing requirement",
    ]
    return any(pattern in lowered for pattern in patterns)


def _payload_option_evidence_records(
    payload: dict[str, Any],
    *,
    choices: list[dict[str, str]],
    evidence_id: str,
) -> list[dict[str, Any]]:
    if not choices:
        return []

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_record(letter: str, status: str, reason: str, source: str, ambiguous: bool = False) -> None:
        if not any(choice.get("letter") == letter for choice in choices):
            return
        final_status = _normalize_option_status(
            status,
            detail_sufficient=(
                True
                if payload.get("decision_sufficiency_validated") is True
                else payload.get("detail_sufficient")
            ),
            ambiguous=ambiguous or _payload_has_unverified_query_premise(payload),
        )
        reason_text = str(reason or "").strip()
        key = (letter, final_status, reason_text[:180])
        if key in seen:
            return
        seen.add(key)
        records.append(
            {
                "evidence_id": evidence_id,
                "option": letter,
                "choice": _choice_text(choices, letter),
                "status": final_status,
                "source": source,
                "reason": reason_text,
                "detail_sufficient": payload.get("detail_sufficient"),
                "backend": _backend(payload),
            }
        )

    explicit = payload.get("option_evidence")
    if isinstance(explicit, dict):
        for key, value in explicit.items():
            letters = _matched_choice_letters(str(key), choices)
            if not letters and str(key).strip().upper() in {c["letter"] for c in choices}:
                letters = [str(key).strip().upper()]
            if isinstance(value, dict):
                status = value.get("status") or value.get("verdict") or value.get("relation")
                reason = value.get("reason") or value.get("evidence") or value.get("description") or ""
            else:
                status = value
                reason = str(value)
            for letter in letters:
                add_record(letter, str(status or "unresolved"), reason, "option_evidence")
    elif isinstance(explicit, list):
        for entry in explicit:
            if not isinstance(entry, dict):
                continue
            option = entry.get("option") or entry.get("letter") or entry.get("choice")
            letters = _matched_choice_letters(str(option or ""), choices)
            status = entry.get("status") or entry.get("verdict") or entry.get("relation")
            reason = entry.get("reason") or entry.get("evidence") or entry.get("description") or ""
            for letter in letters:
                add_record(letter, str(status or "unresolved"), reason, "option_evidence")

    for source_key, base_status in (
        ("supports_options", "support"),
        ("contradicts_options", "contradict"),
    ):
        for label in payload.get(source_key) or []:
            label_text = str(label)
            letters = _matched_choice_letters(label_text, choices)
            if not letters:
                continue
            ambiguous = base_status == "support" and len(letters) > 1
            for letter in letters:
                status = base_status
                if base_status == "support" and _text_negates_choice(label_text, _choice_text(choices, letter)):
                    status = "contradict"
                add_record(letter, status, label_text, source_key, ambiguous=ambiguous)

    return records


def _evidence_id(index: int) -> str:
    return f"E{index:05d}"


def _backend(payload: dict[str, Any]) -> str:
    return str(payload.get("observer_backend") or "")


def _evidence_scope(
    *,
    tool_name: str,
    source_kind: str,
    timestamp_s: float | None,
    t_range: list[float] | None,
) -> str:
    if source_kind == "aggregate_verifier_decision":
        return "query_sufficient"
    if timestamp_s is not None:
        return "local_timestamp"
    if t_range is not None:
        return "local_window"
    if tool_name == "overview" and source_kind == "tool_summary":
        return "global_routing"
    return "unknown"


def _has_support_or_contradiction(payload: dict[str, Any]) -> bool:
    return bool(payload.get("supports_options") or payload.get("contradicts_options"))


def _initial_level(tool_name: str, payload: dict[str, Any]) -> str:
    backend = _backend(payload)
    if payload.get("parse_ok") is False or "error" in backend:
        return "uncertain"
    if tool_name == "overview":
        return "routing"
    if tool_name == "skim_qwen":
        return "routing"
    if tool_name in {"focus_qwen", "localize_qwen"}:
        return "candidate"
    if tool_name == "skim":
        return "candidate"
    if tool_name == "frame_verify":
        if payload.get("decision_sufficiency_validated") is True:
            return "verified"
        if _payload_has_unverified_query_premise(payload):
            return "candidate"
        if payload.get("detail_sufficient") is True:
            return "verified"
        if _has_support_or_contradiction(payload):
            return "candidate"
        return "candidate"
    if tool_name == "focus":
        if payload.get("detail_sufficient") is True or _has_support_or_contradiction(payload):
            return "verified"
        return "candidate"
    return "candidate"


def _per_candidate_direct_level(
    *,
    tool_name: str,
    payload: dict[str, Any],
    assessment: dict[str, Any],
    timestamp_s: float | None,
    default_level: str,
    strict_anchor: bool = False,
) -> str:
    """Admit one grounded direct anchor independently of batch sufficiency.

    A multi-window child batch may be unable to decide a whole-video question,
    even though one candidate contains a directly observed event.  P125 keeps
    that local visual fact verified when the candidate binding is complete and
    the timestamp is the assessment's explicit best anchor.  Other timestamps
    in the same candidate retain the aggregate level so a negative context frame
    cannot inherit the candidate's positive verdict.
    """
    if tool_name != "frame_verify":
        return default_level
    if payload.get("candidate_binding_complete") is not True:
        return default_level
    if not isinstance(assessment, dict) or not assessment:
        return default_level
    # A sufficient multi-candidate call does not make every timestamp a
    # verified positive.  P127 starts each bound anchor conservatively, then
    # admits only the explicit best matched/direct anchor below.
    fallback_level = "candidate" if strict_anchor else default_level
    if default_level == "verified" and not strict_anchor:
        return default_level
    if assessment.get("assessment_present") is False:
        return fallback_level
    if str(assessment.get("target_match") or "").strip().lower() != "matched":
        return fallback_level
    if str(assessment.get("event_match") or "").strip().lower() != "direct":
        return fallback_level
    if assessment.get("option_set_conflict") is True:
        return fallback_level

    best_timestamp = _safe_float(assessment.get("best_timestamp_s"))
    if timestamp_s is None or best_timestamp is None:
        return fallback_level
    if abs(float(timestamp_s) - best_timestamp) > 0.25:
        return fallback_level
    event_span = _safe_range(assessment.get("event_span"))
    if event_span is None or not event_span[0] <= float(timestamp_s) <= event_span[1]:
        return fallback_level
    return "verified"


def _uncertainty_text(payload: dict[str, Any], item: dict[str, Any] | None = None) -> str:
    parts = []
    if item:
        need = item.get("needs_focus") or item.get("missing_detail")
        if need:
            parts.append(str(need))
    missing = (
        None
        if payload.get("decision_sufficiency_validated") is True
        else payload.get("missing_detail")
    )
    if missing:
        parts.append(str(missing))
    if payload.get("parse_ok") is False:
        parts.append("tool output was not parsed as structured JSON")
    backend = _backend(payload)
    if "error" in backend:
        parts.append(f"observer backend error: {backend}")
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        text = part.strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return "; ".join(out)


def _tool_summary(payload: dict[str, Any]) -> str:
    return str(
        payload.get("global_summary")
        or payload.get("overall_summary")
        or payload.get("observed_event")
        or payload.get("local_motion")
        or payload.get("interaction")
        or ""
    ).strip()


def _append_evidence_item(
    state: dict[str, Any],
    *,
    tool_name: str,
    parameters: dict[str, Any] | None,
    payload: dict[str, Any],
    description: str,
    timestamp_s: float | None = None,
    t_range: list[float] | None = None,
    scene_id: str | None = None,
    window_id: str | None = None,
    candidate_id: str | None = None,
    event_tags: list[Any] | None = None,
    evidence_level: str | None = None,
    uncertainty: str = "",
    source_kind: str,
    source_index: int | None,
    choices: list[dict[str, str]] | None = None,
    include_evidence_scope: bool = True,
) -> None:
    description = str(description or "").strip()
    if not description and not uncertainty:
        return
    items = state.setdefault("evidence_items", [])
    item = {
        "evidence_id": _evidence_id(len(items) + 1),
        "source_tool": tool_name,
        "backend": _backend(payload),
        "scene_id": scene_id or payload.get("scene_id") or "",
        "window_id": window_id or payload.get("window_id") or "",
        "candidate_id": candidate_id or payload.get("candidate_id") or "",
        "timestamp_s": round(float(timestamp_s), 1) if timestamp_s is not None else None,
        "t_range": t_range or _safe_range(payload.get("t_range")),
        "evidence_level": evidence_level or _initial_level(tool_name, payload),
        "description": description,
        "event_tags": [str(tag) for tag in (event_tags or [])],
        "supports_options": [str(item) for item in (payload.get("supports_options") or [])],
        "contradicts_options": [str(item) for item in (payload.get("contradicts_options") or [])],
        "possible_evidence": (
            payload.get("possible_evidence")
            if "possible_evidence" in payload
            else payload.get("contains_evidence")
        ),
        "detail_sufficient": payload.get("detail_sufficient"),
        "decision_sufficient": payload.get("decision_sufficiency_validated") is True,
        "decision_evidence_need": str(payload.get("decision_evidence_need") or ""),
        "residual_visual_detail": str(payload.get("residual_visual_detail") or ""),
        "target_entity_or_event": str(payload.get("target_entity_or_event") or ""),
        "target_match": str(payload.get("target_match") or "unknown"),
        "event_match": str(payload.get("event_match") or payload.get("target_event_match") or "unknown"),
        "target_binding_reason": str(payload.get("target_binding_reason") or ""),
        "question_scope": str(payload.get("question_scope") or "unknown"),
        "scope_coverage": str(payload.get("scope_coverage") or "unknown"),
        "scope_coverage_reason": str(payload.get("scope_coverage_reason") or ""),
        "observed_fact": str(payload.get("observed_fact") or ""),
        "option_set_conflict": payload.get("option_set_conflict") is True,
        "evidence_need": str(payload.get("evidence_need") or ""),
        "evidence_needs": [
            str(value) for value in (payload.get("evidence_needs") or [])
        ],
        "observer_conflict_options": [
            str(value) for value in (payload.get("observer_conflict_options") or [])
        ],
        "uncertainty": uncertainty,
        "refs": {
            "source_kind": source_kind,
            "source_index": source_index,
            "parameters": deepcopy(parameters or {}),
        },
    }
    if payload.get("receipt_schema") == "local_visual_receipt_v1":
        for key in ("receipt_schema", "verification_id", "anchor_observations", "target_event_id", "missing_detail"):
            item[key] = deepcopy(payload.get(key))
    if include_evidence_scope:
        item["evidence_scope"] = _evidence_scope(
            tool_name=tool_name,
            source_kind=source_kind,
            timestamp_s=timestamp_s,
            t_range=t_range or _safe_range(payload.get("t_range")),
        )
    option_records: list[dict[str, Any]] = []
    if source_kind in {
        "scene_summary",
        "tool_summary",
        "aggregate_verifier_decision",
    }:
        option_records = _payload_option_evidence_records(
            payload,
            choices=choices or [],
            evidence_id=item["evidence_id"],
        )
        if option_records:
            if include_evidence_scope:
                for record in option_records:
                    record["evidence_scope"] = item.get("evidence_scope")
                    record["t_range"] = deepcopy(item.get("t_range"))
            item["option_evidence"] = option_records
    items.append(item)
    if option_records:
        state.setdefault("option_evidence", []).extend(option_records)


def _has_complete_aggregate_verifier_decision(
    tool_name: str,
    payload: dict[str, Any],
    *,
    choices: list[dict[str, str]],
    parameters: dict[str, Any] | None = None,
    require_validated_decision: bool = False,
) -> bool:
    """Return whether a verifier's aggregate option comparison is complete.

    Timestamp rows are visual facts, while support/contradiction fields describe
    the verifier's decision over the complete requested window set. Preserve that
    second level only when the verifier explicitly reports a bound, sufficient,
    conflict-free comparison with one supported option.
    """
    if tool_name != "frame_verify":
        return False
    decision_complete = payload.get("decision_sufficiency_validated") is True
    if require_validated_decision and not decision_complete:
        return False
    if not decision_complete and payload.get("detail_sufficient") is not True:
        return False
    candidate_windows = [
        item
        for item in (parameters or {}).get("candidate_windows") or []
        if isinstance(item, dict)
    ]
    candidate_assessments = [
        item
        for item in payload.get("candidate_assessments") or []
        if isinstance(item, dict)
    ]
    if len(candidate_windows) < 2 or len(candidate_assessments) < 2:
        return False
    if payload.get("candidate_binding_complete") is not True:
        return False
    if str(payload.get("target_match") or "unknown").strip().lower() != "matched":
        return False
    if str(payload.get("scope_coverage") or "unknown").strip().lower() != "sufficient":
        return False
    if not decision_complete:
        evidence_need = str(payload.get("evidence_need") or "sufficient").strip().lower()
        if evidence_need not in {"", "sufficient"}:
            return False
        evidence_needs = {
            str(value).strip().lower()
            for value in (payload.get("evidence_needs") or [])
            if str(value).strip()
        }
        if evidence_needs - {"sufficient"}:
            return False
    if payload.get("option_set_conflict") is True:
        return False
    if payload.get("observer_conflict_options"):
        return False
    if not decision_complete and _meaningful_uncertainty(payload.get("missing_detail")):
        return False

    supported_letters: set[str] = set()
    for label in payload.get("supports_options") or []:
        supported_letters.update(_matched_choice_letters(str(label), choices))
    return len(supported_letters) == 1


def update_structured_evidence_state(
    memory: dict[str, Any],
    *,
    tool_name: str,
    parameters: dict[str, Any] | None,
    payload: dict[str, Any],
    question_context: str | None = None,
    include_evidence_scope: bool = True,
    scope_aware_evidence: bool = False,
    preserve_aggregate_verifier_decision: bool = False,
) -> dict[str, Any]:
    """Compile the latest parsed tool observation into structured evidence.

    This function only updates memory representation. It never creates tool
    calls, blocks answers, or changes planner control flow.
    """
    state = ensure_structured_evidence_state(memory)
    choices = _extract_choices(question_context)
    if choices:
        state["choices"] = choices
    level = _initial_level(tool_name, payload)
    candidate_ranges: dict[str, list[float] | None] = {}
    for candidate in (parameters or {}).get("candidate_windows") or []:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if candidate_id:
            candidate_ranges[candidate_id] = _safe_range(
                candidate.get("t_range") or candidate.get("recommended_verify_window")
            )
    assessment_by_id = {
        str(item.get("candidate_id")): item
        for item in (payload.get("candidate_assessments") or [])
        if isinstance(item, dict) and str(item.get("candidate_id") or "").strip()
    }
    per_candidate_direct_enabled = bool(
        (memory.get("runtime_config") or {}).get(
            "p125_per_candidate_direct_evidence_enabled", False
        )
    )
    strict_per_anchor_enabled = bool(
        (memory.get("runtime_config") or {}).get(
            "p127_strict_per_anchor_evidence_enabled", False
        )
    )
    timestamp_items = payload.get("timestamp_observations") or []
    for idx, obs in enumerate(timestamp_items, start=1):
        if not isinstance(obs, dict):
            continue
        ts_value = obs.get("timestamp_s")
        if ts_value is None:
            ts_value = obs.get("timestamp")
        ts = _safe_float(ts_value)
        desc = obs.get("description") or obs.get("desc") or ""
        candidate_id = str(obs.get("candidate_id") or "").strip()
        assessment = assessment_by_id.get(candidate_id) or {}
        item_level = level
        if per_candidate_direct_enabled or strict_per_anchor_enabled:
            item_level = _per_candidate_direct_level(
                tool_name=tool_name,
                payload=payload,
                assessment=assessment,
                timestamp_s=ts,
                default_level=level,
                strict_anchor=strict_per_anchor_enabled,
            )
        item_payload = dict(payload)
        if candidate_id:
            item_payload["candidate_id"] = candidate_id
            for key in (
                "target_match",
                "event_match",
                "target_binding_reason",
                "observed_fact",
                "supports_options",
                "contradicts_options",
                "option_set_conflict",
            ):
                if key in assessment:
                    item_payload[key] = assessment[key]
                if key in obs:
                    item_payload[key] = obs[key]
        if payload.get("receipt_schema") == "local_visual_receipt_v1":
            for key in ("missing_detail", "anchor_observations", "target_event_id"):
                item_payload[key] = deepcopy(assessment.get(key))
        _append_evidence_item(
            state,
            tool_name=tool_name,
            parameters=parameters,
            payload=item_payload,
            description=str(desc),
            timestamp_s=ts,
            t_range=_safe_range(obs.get("time_range_s"))
            or candidate_ranges.get(candidate_id)
            or _safe_range(payload.get("t_range")),
            scene_id=obs.get("scene_id") or payload.get("scene_id"),
            window_id=obs.get("window_id") or payload.get("window_id"),
            candidate_id=candidate_id,
            event_tags=obs.get("event_tags") or [],
            evidence_level=item_level,
            uncertainty=_uncertainty_text(item_payload, obs),
            source_kind="timestamp_observation",
            source_index=idx,
            choices=choices,
            include_evidence_scope=include_evidence_scope,
        )

    scene_items = payload.get("scene_summaries") or []
    for idx, scene in enumerate(scene_items, start=1):
        if not isinstance(scene, dict):
            continue
        summary = scene.get("summary") or scene.get("observed_event") or ""
        scene_level = level
        if tool_name == "overview":
            scene_level = "routing"
        elif scene.get("possible_evidence") is False and scene.get("missing_detail"):
            scene_level = "uncertain"
        _append_evidence_item(
            state,
            tool_name=tool_name,
            parameters=parameters,
            payload=payload,
            description=str(summary),
            timestamp_s=None,
            t_range=_safe_range(scene.get("t_range") or scene.get("time_range_s") or payload.get("t_range")),
            scene_id=scene.get("scene_id") or payload.get("scene_id"),
            window_id=scene.get("window_id") or payload.get("window_id"),
            event_tags=[],
            evidence_level=scene_level,
            uncertainty=_uncertainty_text(payload, scene),
            source_kind="scene_summary",
            source_index=idx,
            choices=choices,
            include_evidence_scope=include_evidence_scope,
        )

    aggregate_decision_appended = False
    if preserve_aggregate_verifier_decision and _has_complete_aggregate_verifier_decision(
        tool_name,
        payload,
        choices=choices,
        parameters=parameters,
        require_validated_decision=(
            payload.get("decision_sufficiency_requested") is True
            or "decision_sufficiency_validated" in payload
        ),
    ):
        aggregate_uncertainty = _uncertainty_text(payload)
        if not _meaningful_uncertainty(aggregate_uncertainty):
            aggregate_uncertainty = ""
        _append_evidence_item(
            state,
            tool_name=tool_name,
            parameters=parameters,
            payload=payload,
            description=str(
                payload.get("observed_fact")
                or _tool_summary(payload)
                or "Complete verifier option comparison."
            ),
            timestamp_s=None,
            t_range=_safe_range(payload.get("t_range")),
            scene_id=payload.get("scene_id"),
            window_id=payload.get("window_id"),
            event_tags=[],
            evidence_level=level,
            uncertainty=aggregate_uncertainty,
            source_kind="aggregate_verifier_decision",
            source_index=None,
            choices=choices,
            include_evidence_scope=include_evidence_scope,
        )
        aggregate_decision_appended = True

    if not timestamp_items and not scene_items and not aggregate_decision_appended:
        _append_evidence_item(
            state,
            tool_name=tool_name,
            parameters=parameters,
            payload=payload,
            description=_tool_summary(payload),
            timestamp_s=None,
            t_range=_safe_range(payload.get("t_range")),
            scene_id=payload.get("scene_id"),
            window_id=payload.get("window_id"),
            event_tags=[],
            evidence_level=level,
            uncertainty=_uncertainty_text(payload),
            source_kind="tool_summary",
            source_index=None,
            choices=choices,
            include_evidence_scope=include_evidence_scope,
        )

    _refresh_option_support(state, scope_aware_evidence=scope_aware_evidence)
    _refresh_uncertainties(state, scope_aware_evidence=scope_aware_evidence)
    return state


def _refresh_option_support(
    state: dict[str, Any],
    *,
    scope_aware_evidence: bool = False,
) -> None:
    by_option: dict[str, dict[str, Any]] = {}
    choices = state.get("choices") or []
    option_records = [
        rec
        for rec in state.get("option_evidence") or []
        if isinstance(rec, dict)
    ]
    if option_records:
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            letter = str(choice.get("letter") or "").upper()
            if not letter:
                continue
            by_option.setdefault(
                letter,
                {
                    "option": letter,
                    "choice": str(choice.get("text") or ""),
                    "supports": [],
                    "weak_supports": [],
                    "contradicts": [],
                    "local_contradicts": [],
                    "unresolved": [],
                },
            )

    for rec in option_records:
        letter = str(rec.get("option") or "").upper()
        if not letter:
            continue
        row = by_option.setdefault(
            letter,
            {
                "option": letter,
                "choice": str(rec.get("choice") or ""),
                "supports": [],
                "weak_supports": [],
                "contradicts": [],
                "local_contradicts": [],
                "unresolved": [],
            },
        )
        eid = rec.get("evidence_id")
        status = str(rec.get("status") or "").lower()
        if status == "support" and eid not in row["supports"]:
            row["supports"].append(eid)
        elif status == "weak_support" and eid not in row["weak_supports"]:
            row["weak_supports"].append(eid)
        elif status == "contradict":
            scope = str(rec.get("evidence_scope") or "")
            target = (
                "local_contradicts"
                if scope_aware_evidence and scope in {"local_timestamp", "local_window"}
                else "contradicts"
            )
            if eid not in row[target]:
                row[target].append(eid)
        elif status == "unresolved" and eid not in row["unresolved"]:
            row["unresolved"].append(eid)

    if by_option:
        state["option_support"] = sorted(by_option.values(), key=lambda row: row["option"])
        return

    for item in state.get("evidence_items") or []:
        if not isinstance(item, dict):
            continue
        for label in item.get("supports_options") or []:
            rec = by_option.setdefault(
                str(label),
                {"option": str(label), "supports": [], "weak_supports": [], "contradicts": [], "local_contradicts": [], "unresolved": []},
            )
            rec["supports"].append(item.get("evidence_id"))
        for label in item.get("contradicts_options") or []:
            rec = by_option.setdefault(
                str(label),
                {"option": str(label), "supports": [], "weak_supports": [], "contradicts": [], "local_contradicts": [], "unresolved": []},
            )
            scope = str(item.get("evidence_scope") or "")
            target = (
                "local_contradicts"
                if scope_aware_evidence and scope in {"local_timestamp", "local_window"}
                else "contradicts"
            )
            rec[target].append(item.get("evidence_id"))
    state["option_support"] = sorted(by_option.values(), key=lambda row: row["option"])


def _meaningful_uncertainty(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" .")
    if not text:
        return False
    lowered = text.lower()
    if lowered in {"none", "null", "n/a", "no missing detail", "nothing missing"}:
        return False
    if re.match(r"^(none|null|n/a)\s*[,;:]", lowered):
        return False
    return True


def _refresh_uncertainties(
    state: dict[str, Any],
    *,
    scope_aware_evidence: bool = False,
) -> None:
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in state.get("evidence_items") or []:
        if not isinstance(item, dict):
            continue
        uncertainty = str(item.get("uncertainty") or "").strip()
        if not uncertainty and item.get("evidence_level") != "uncertain":
            continue
        if scope_aware_evidence:
            if (
                item.get("source_tool") == "overview"
                or item.get("detail_sufficient") is True
                or item.get("decision_sufficient") is True
            ):
                continue
            if uncertainty and not _meaningful_uncertainty(uncertainty):
                continue
        key = (
            item.get("source_tool"),
            item.get("scene_id"),
            str(item.get("t_range")),
            uncertainty or "evidence is not verified",
        )
        if scope_aware_evidence and key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "evidence_id": item.get("evidence_id"),
                "scene_id": item.get("scene_id"),
                "t_range": item.get("t_range"),
                "timestamp_s": item.get("timestamp_s"),
                "uncertainty": uncertainty or "evidence is not verified",
            }
        )
    state["uncertainties"] = out


def _format_evidence_item(item: dict[str, Any]) -> str:
    eid = item.get("evidence_id") or "E?"
    level = item.get("evidence_level") or "candidate"
    tool = item.get("source_tool") or "tool"
    backend = item.get("backend")
    backend_text = f"/{backend}" if backend else ""
    scene = item.get("scene_id") or "?"
    when = item.get("timestamp_s")
    if when is None:
        when = item.get("t_range")
    desc = str(item.get("description") or "").strip()
    support = item.get("supports_options") or []
    contradict = item.get("contradicts_options") or []
    suffix_parts = []
    candidate_id = str(item.get("candidate_id") or "").strip()
    if candidate_id:
        suffix_parts.append(f"candidate={candidate_id}")
        suffix_parts.append(f"target_match={item.get('target_match') or 'unknown'}")
        suffix_parts.append(f"event_match={item.get('event_match') or 'unknown'}")
    if item.get("option_set_conflict") is True:
        suffix_parts.append("option_set_conflict=true")
    evidence_need = str(item.get("evidence_need") or "").strip()
    if evidence_need:
        suffix_parts.append(f"need={evidence_need}")
    if item.get("decision_sufficient") is True:
        suffix_parts.append("decision_sufficient=true")
    scope = str(item.get("evidence_scope") or "").strip()
    if scope:
        suffix_parts.append(f"scope={scope}")
    if support:
        suffix_parts.append(f"support={support}")
    if contradict:
        suffix_parts.append(f"contradict={contradict}")
    option_evidence = item.get("option_evidence") or []
    if option_evidence:
        compact = []
        for rec in option_evidence[:4]:
            compact.append(
                f"{rec.get('option')}:{rec.get('status')}"
            )
        suffix_parts.append(f"option_evidence={compact}")
    uncertainty = str(item.get("uncertainty") or "").strip()
    if uncertainty and level in {"uncertain", "candidate", "routing"}:
        suffix_parts.append(f"uncertain={uncertainty[:100]}")
    suffix = " " + " ".join(suffix_parts) if suffix_parts else ""
    return f"- [{eid}] {level} {tool}{backend_text} {scene} {when}: {desc[:160]}{suffix}"


def _latest_by_level(state: dict[str, Any], levels: set[str], max_items: int) -> list[dict[str, Any]]:
    items = [
        item
        for item in state.get("evidence_items") or []
        if isinstance(item, dict) and item.get("evidence_level") in levels
    ]
    return items[-max_items:]


def format_structured_evidence_for_prompt(
    memory: dict[str, Any] | None,
    *,
    max_items: int = 10,
) -> str:
    if not memory:
        return ""
    state = ensure_structured_evidence_state(memory)
    items = [item for item in state.get("evidence_items") or [] if isinstance(item, dict)]
    if not items:
        return ""

    lines = [
        "Structured Evidence State:",
        "Evidence levels: routing/candidate are not final proof; verified is strongest; uncertain needs caution.",
        "Evidence scope: local_timestamp/local_window claims apply only to that time range; local absence is not a global contradiction.",
    ]
    verified = _latest_by_level(state, {"verified"}, max(2, max_items // 3))
    candidate = _latest_by_level(state, {"candidate", "uncertain"}, max(3, max_items // 2))
    routing = _latest_by_level(state, {"routing"}, max(2, max_items - len(verified) - len(candidate)))
    if verified:
        lines.append("Verified evidence:")
        lines.extend(_format_evidence_item(item) for item in verified)
    if candidate:
        lines.append("Candidate/uncertain evidence:")
        lines.extend(_format_evidence_item(item) for item in candidate)
    if routing:
        lines.append("Routing evidence:")
        lines.extend(_format_evidence_item(item) for item in routing)
    option_support = state.get("option_support") or []
    if option_support:
        lines.append("Option-wise evidence refs:")
        for rec in option_support[:4]:
            lines.append(
                f"- {rec.get('option')} {rec.get('choice') or ''}: "
                f"support={rec.get('supports') or []} "
                f"weak_support={rec.get('weak_supports') or []} "
                f"contradict={rec.get('contradicts') or []} "
                f"local_contradict={rec.get('local_contradicts') or []} "
                f"unresolved={rec.get('unresolved') or []}"
            )
    uncertainties = state.get("uncertainties") or []
    if uncertainties:
        lines.append("Open evidence uncertainties:")
        for item in uncertainties[-3:]:
            lines.append(
                f"- {item.get('evidence_id')} {item.get('scene_id')} "
                f"{item.get('timestamp_s') or item.get('t_range')}: {str(item.get('uncertainty') or '')[:140]}"
            )
    return "\n".join(lines)


def format_structured_evidence_for_answer(
    memory: dict[str, Any] | None,
    *,
    max_items: int = 16,
) -> str:
    if not memory:
        return ""
    state = ensure_structured_evidence_state(memory)
    items = [item for item in state.get("evidence_items") or [] if isinstance(item, dict)]
    if not items:
        return ""

    lines = [
        "Structured evidence digest:",
        "Use verified evidence first. Do not treat routing evidence as confirmed details.",
        "Respect evidence scope: local_timestamp/local_window absence cannot exclude an event elsewhere in the video.",
        "For multiple-choice questions, prefer the Option-wise evidence table over free-text impressions.",
        "Do not select an option that has only contradictions when another option has verified support.",
        "Treat weak_support as suggestive rather than decisive, especially when detail_sufficient=false.",
    ]
    verified = _latest_by_level(state, {"verified"}, max(4, max_items // 3))
    candidate = _latest_by_level(state, {"candidate"}, max(4, max_items // 3))
    uncertain = _latest_by_level(state, {"uncertain"}, max(3, max_items // 4))
    routing = _latest_by_level(state, {"routing"}, max(2, max_items - len(verified) - len(candidate) - len(uncertain)))
    if verified:
        lines.append("Strong verified evidence:")
        lines.extend(_format_evidence_item(item) for item in verified)
    if candidate:
        lines.append("Candidate evidence:")
        lines.extend(_format_evidence_item(item) for item in candidate)
    if uncertain:
        lines.append("Uncertain evidence:")
        lines.extend(_format_evidence_item(item) for item in uncertain)
    if routing:
        lines.append("Routing evidence:")
        lines.extend(_format_evidence_item(item) for item in routing)
    option_support = state.get("option_support") or []
    if option_support:
        lines.append("Option-wise evidence table:")
        for rec in option_support:
            lines.append(
                f"- {rec.get('option')} {rec.get('choice') or ''}: "
                f"support={rec.get('supports') or []} "
                f"weak_support={rec.get('weak_supports') or []} "
                f"contradict={rec.get('contradicts') or []} "
                f"local_contradict={rec.get('local_contradicts') or []} "
                f"unresolved={rec.get('unresolved') or []}"
            )
        option_records = [
            rec
            for rec in state.get("option_evidence") or []
            if isinstance(rec, dict)
        ]
        if option_records:
            lines.append("Recent option evidence reasons:")
            for rec in option_records[-max_items:]:
                lines.append(
                    f"- {rec.get('evidence_id')} {rec.get('option')} {rec.get('choice')}: "
                    f"{rec.get('status')} because {str(rec.get('reason') or '')[:180]}"
                )
    return "\n".join(lines)
