import base64
from copy import deepcopy
from io import BytesIO
import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from videoseek.codec import timestamps_to_frame_indices
from videoseek.core.memory import extract_v10_payload
from videoseek.core.minimal_global_fsm import (
    detect_global_mode,
    parse_global_question,
    should_use_minimal_global_fsm,
)
from videoseek.core.p130_runtime import RUNTIME_SCHEMA_VERSION, merge_p131_count_occurrences, parse_local_observations
import json
import re
from videoseek.core.tool_evidence_handoff import collect_memory_anchor_rows
from videoseek.observer import observe_content
from videoseek.skeleton import scene_aware_timestamps
from videoseek.utils import extract_json_object, append_window_subtitles

from .focus import _question_scope_hint, execute_focus
from .v10_format import LOCAL_RECEIPT_SCHEMA, local_receipt_contract, normalize_local_receipt, merge_local_receipts

from .spatial_grounding import prepare_grounded_frames
from .v10_format import format_v10_observation, snap_timestamp_observations


frame_verify_tool = {
    "type": "function",
    "function": {
        "name": "frame_verify",
        "description": (
            "Strong API visual verification over a short real-frame window. "
            "Use this as the final visual evidence tool after overview or local "
            "Qwen localization has proposed candidate frames/windows."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The concrete visual claim or option detail to verify.",
                },
                "start_time": {
                    "type": "number",
                    "description": "The start time of the short clip to verify.",
                },
                "end_time": {
                    "type": "number",
                    "description": "The end time of the short clip to verify.",
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "detail_verify, temporal_strip, option_verify, wider_context, "
                        "timeline, or count_occurrence."
                    ),
                },
            },
            "required": ["query", "start_time", "end_time", "mode"],
            "additionalProperties": False,
        },
    },
}


_P130_OPTION_LIST_FIELDS = {
    "supports_options",
    "contradicts_options",
    "context_supports_options",
    "context_contradicts_options",
    "candidate_set_aggregate_supports_options",
    "candidate_set_aggregate_contradicts_options",
    "observer_conflict_options",
    "option_conflict_candidate_ids",
}
_P130_OPTION_DICT_FIELDS = {
    "option_evidence",
    "candidate_set_aggregate_option_evidence",
}
_P130_OPTION_BOOL_FIELDS = {
    "option_set_conflict",
    "candidate_set_aggregate_decision_preserved",
    "candidate_set_aggregate_decision_reported",
}
_P130_OPTION_TEXT_FIELDS = {
    "candidate_set_aggregate_decision_reason",
}


def _p130_global_context(config: dict, parameters: dict) -> dict[str, Any]:
    """Return the P130 verifier contract without changing P128 behaviour."""
    question = str(parameters.get("question") or "").strip()
    mode = detect_global_mode(question)
    p131_active = bool(
        config.get("p131_minimal_global_repairs_enabled", False)
        and should_use_minimal_global_fsm(question)
    )
    active = bool(
        (
            config.get("p130_minimal_global_fsm_enabled", False)
            or p131_active
        )
        and should_use_minimal_global_fsm(question)
    )
    parsed = parse_global_question(question) if active else {}
    assigned_target_by_candidate: dict[str, str] = {}
    if p131_active and mode == "order":
        for item in parameters.get("candidate_windows") or []:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id") or "").strip()
            raw_target = item.get("target_event_id")
            if raw_target in (None, ""):
                values = item.get("target_event_ids") or []
                if isinstance(values, (list, tuple)) and len(values) == 1:
                    raw_target = values[0]
            target_id = _p130_normalize_event_id(raw_target)
            if candidate_id and target_id:
                assigned_target_by_candidate[candidate_id] = target_id
    return {
        "active": active,
        "p131_active": p131_active,
        "p132_active": bool(active and config.get("p132_weak_lead_visible_core_enabled", False)),
        "mode": mode,
        "question": question,
        "parsed": parsed,
        "required_event_ids": [
            str(value) for value in parsed.get("required_event_ids") or []
        ],
        "assigned_target_by_candidate": assigned_target_by_candidate,
    }


def _p130_event_catalog_instruction(context: dict[str, Any]) -> str:
    """Describe typed Order IDs while withholding answer-option semantics."""
    if not context.get("active") or context.get("mode") != "order":
        return ""
    events = context.get("parsed", {}).get("required_events") or []
    if not events:
        return ""
    catalog = "\n".join(
        f"- event_id={row.get('event_id')}: {row.get('description')}"
        for row in events
    )
    if context.get("p131_active"):
        assignments = context.get("assigned_target_by_candidate") or {}
        assignment_text = (
            "Candidate assignments:\n"
            + "\n".join(
                f"- {candidate_id} -> event_id={event_id}"
                for candidate_id, event_id in assignments.items()
            )
            + "\n"
            if assignments
            else ""
        )
        return (
            "P131 local-fact contract for this chronological-order question:\n"
            "Event catalog (the only legal target event IDs):\n"
            f"{catalog}\n"
            f"{assignment_text}"
            "Assess each assigned candidate only against its assigned event. Mark it "
            "direct when the visible pixels establish that event's distinctive "
            "observable core and it uniquely matches that catalog entry. Narrative "
            "purpose or intent clauses need not be visually observable and must not, by "
            "themselves, downgrade a unique visible core to context_only. Nearby frames "
            "may disambiguate the event, but context alone is not direct proof. If the "
            "candidate instead shows another catalog event, keep the assigned binding "
            "empty and report the visible event only as a possible candidate ID. Never "
            "infer an ID from answer choices, catalog order, timestamps, routing captions, "
            "or expected chronology. The verifier reports local facts only.\n"
        )
    return (
        "P130 local-fact contract for this chronological-order question:\n"
        "Event catalog (the only legal target event IDs):\n"
        f"{catalog}\n"
        "For each candidate and audited anchor, report candidate_event_ids as the "
        "catalog IDs that the visible pixels could describe. Report target_event_id "
        "as one catalog ID only when event_match is direct and the pixels uniquely "
        "establish that exact described event. Otherwise use an empty string. Do not "
        "infer an ID from answer choices, expected chronology, timestamps, routing "
        "captions, or nearby context. The verifier reports local facts only and must "
        "not select, support, or contradict an answer option.\n"
    )


def _p130_local_verification_objective(context: dict[str, Any]) -> str:
    stem = str(context.get("parsed", {}).get("stem") or "").strip()
    if context.get("mode") == "order":
        return (
            "Identify which catalog event, if any, is directly visible in each "
            "shown window, and report its local event span."
        )
    if context.get("p132_active"):
        return (
            "Determine whether the target action is directly visible in each shown "
            "window. Match its observable meaning, including visually equivalent "
            "ways of performing it, and report each local occurrence with its span. "
            f"Stem: {stem}"
        )
    if context.get("p131_active"):
        return (
            "Determine whether the action named by this Count stem is directly visible "
            "in each shown window, and report each local occurrence. Match the action by "
            "its observable meaning rather than exact wording or cookware: pan cooking, "
            "grilling, and open-flame cooking all count when the target food/action is "
            f"visibly the same. Stem: {stem}"
        )
    return (
        "Determine whether the action named by this question stem is directly "
        f"visible in each shown window, and report each local occurrence: {stem}"
    )


def _p130_normalize_event_id(value: Any) -> str:
    """Accept the small aliases used by verifiers and reduce them to numeric IDs."""
    text = str(value or "").strip().lower()
    if not text:
        return ""
    for prefix in ("event_id=", "event id", "event", "evt"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip(" :=#()[]")
            break
    if text.isdigit():
        return str(int(text))
    return ""


def _p130_normalize_typed_event_binding(
    row: dict[str, Any],
    *,
    required_event_ids: set[str],
    assigned_target: str = "",
) -> None:
    """Keep only one conflict-free direct Order binding.

    Candidate IDs remain available as an ambiguity frontier.  Exact aliases are
    deliberately cleared when the verifier returns an invalid ID, multiple IDs,
    a non-direct match, or mutually inconsistent fields.
    """
    raw_exact: list[Any] = []
    for key in ("target_event_id", "event_index", "ordered_event_id"):
        if row.get(key) not in (None, ""):
            raw_exact.append(row.get(key))
    for key in ("target_event_ids", "matched_event_ids"):
        value = row.get(key)
        if isinstance(value, (list, tuple, set)):
            raw_exact.extend(value)
        elif value not in (None, ""):
            raw_exact.append(value)
    raw_candidates: list[Any] = []
    for key in ("candidate_event_ids", "possible_event_ids"):
        value = row.get(key)
        if isinstance(value, (list, tuple, set)):
            raw_candidates.extend(value)
        elif value not in (None, ""):
            raw_candidates.append(value)

    raw_id_values = raw_candidates + raw_exact
    invalid_raw_ids = {
        str(value).strip()
        for value in raw_id_values
        if str(value or "").strip()
        and (
            not _p130_normalize_event_id(value)
            or _p130_normalize_event_id(value) not in required_event_ids
        )
    }
    exact_ids = [
        event_id
        for event_id in (_p130_normalize_event_id(value) for value in raw_exact)
        if event_id
    ]
    candidate_ids = [
        event_id
        for event_id in (
            _p130_normalize_event_id(value)
            for value in raw_candidates + raw_exact
        )
        if event_id
    ]
    invalid_ids = sorted(
        {
            event_id
            for event_id in exact_ids + candidate_ids
            if event_id not in required_event_ids
        }
        | invalid_raw_ids
    )
    exact_ids = list(
        dict.fromkeys(event_id for event_id in exact_ids if event_id in required_event_ids)
    )
    candidate_ids = list(
        dict.fromkeys(
            event_id for event_id in candidate_ids if event_id in required_event_ids
        )
    )
    exact_claims = set(exact_ids)
    list_claims = set(
        event_id
        for event_id in (
            _p130_normalize_event_id(value) for value in raw_candidates
        )
        if event_id in required_event_ids
    )
    field_conflict = bool(
        len(exact_claims) > 1
        or (exact_claims and list_claims and not exact_claims.issubset(list_claims))
    )
    unique_ids = set(candidate_ids)
    direct = str(row.get("event_match") or "").strip().lower() == "direct"
    exact = (
        next(iter(exact_claims))
        if direct
        and len(exact_claims) == 1
        and len(unique_ids) == 1
        and not invalid_ids
        and not field_conflict
        else ""
    )
    if assigned_target:
        # The candidate's task is fixed before verification. Other possible
        # catalog IDs are context, while contradictory exact claims still fail.
        field_conflict = bool(
            field_conflict
            or row.get("event_id_conflict") is True
            or row.get("invalid_event_ids")
            or assigned_target not in required_event_ids
            or (exact_claims and exact_claims != {assigned_target})
        )
        span = row.get("event_span")
        anchor = row.get("best_timestamp_s", row.get("timestamp_s"))
        try:
            legal_span = (
                isinstance(span, (list, tuple)) and len(span) == 2
                and all(math.isfinite(float(value)) for value in span)
                and 0 <= float(span[0]) < float(span[1])
            )
            legal_anchor = (
                anchor is not None and math.isfinite(float(anchor))
                and float(anchor) >= 0
                and (not legal_span or float(span[0]) <= float(anchor) <= float(span[1]))
            )
        except (TypeError, ValueError):
            legal_span = legal_anchor = False
        exact = assigned_target if (
            direct and row.get("target_match") == "matched"
            and str(row.get("observed_fact") or "").strip()
            and (legal_span or legal_anchor)
            and assigned_target in unique_ids
            and not invalid_ids and not field_conflict
        ) else ""
        row["assigned_target_event_id"] = assigned_target

    row["candidate_event_ids"] = candidate_ids
    row["target_event_id"] = exact
    row["event_index"] = exact
    row["target_event_ids"] = [exact] if exact else []
    row["matched_event_ids"] = [exact] if exact else []
    row["binding_confidence"] = "direct" if exact else "ambiguous"
    row["event_id_conflict"] = bool(
        field_conflict or invalid_ids
        or (not assigned_target and direct and len(unique_ids) > 1)
    )
    if invalid_ids:
        row["invalid_event_ids"] = invalid_ids
    else:
        row.pop("invalid_event_ids", None)


def _p130_clear_exact_event_binding(row: dict[str, Any]) -> None:
    row["target_event_id"] = ""
    row["event_index"] = ""
    row["target_event_ids"] = []
    row["matched_event_ids"] = []
    row["binding_confidence"] = "ambiguous"
    row["event_id_conflict"] = True


def _p131_enforce_assigned_candidate_target(
    row: dict[str, Any],
    *,
    assigned_target_by_candidate: dict[str, str],
) -> None:
    """Prevent one Order candidate from being rebound to a different catalog ID."""
    candidate_id = str(row.get("candidate_id") or "").strip()
    assigned = assigned_target_by_candidate.get(candidate_id)
    if not assigned:
        return
    row["assigned_target_event_id"] = assigned
    exact = _p130_normalize_event_id(row.get("target_event_id"))
    if exact and exact != assigned:
        row["p131_assignment_conflict"] = True
        _p130_clear_exact_event_binding(row)
        return
    if exact == assigned:
        row["target_event_id"] = assigned
        row["event_index"] = assigned
        row["target_event_ids"] = [assigned]
        row["matched_event_ids"] = [assigned]


def _p130_localize_option_fields(value: Any) -> None:
    """Recursively retain local audits while removing answer-level claims."""
    if isinstance(value, list):
        for item in value:
            _p130_localize_option_fields(item)
        return
    if not isinstance(value, dict):
        return
    for key in list(value):
        if key.startswith("local_"):
            continue
        if key in _P130_OPTION_LIST_FIELDS:
            local_key = f"local_{key}"
            if value.get(key):
                value[local_key] = deepcopy(value.get(key))
            value[key] = []
        elif key in _P130_OPTION_DICT_FIELDS:
            local_key = f"local_{key}"
            if value.get(key):
                value[local_key] = deepcopy(value.get(key))
            value[key] = {}
        elif key in _P130_OPTION_BOOL_FIELDS:
            local_key = f"local_{key}"
            if value.get(key) is not None:
                value[local_key] = value.get(key)
            value[key] = False
        elif key in _P130_OPTION_TEXT_FIELDS:
            local_key = f"local_{key}"
            if value.get(key):
                value[local_key] = str(value.get(key))
            value[key] = "p130_local_receipt_only"
    for key, item in list(value.items()):
        if not key.startswith("local_"):
            _p130_localize_option_fields(item)


def _apply_p130_local_receipt_contract(
    payload: dict[str, Any],
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Make a verifier observation local-only before memory can consume it."""
    if not context.get("active"):
        return payload
    if context.get("mode") == "order":
        required_event_ids = set(context.get("required_event_ids") or [])
        assigned_target_by_candidate = dict(
            context.get("assigned_target_by_candidate") or {}
        )
        if not payload.get("event_match") and payload.get("target_event_match"):
            payload["event_match"] = payload.get("target_event_match")
        _p130_normalize_typed_event_binding(
            payload,
            required_event_ids=required_event_ids,
        )
        for row in payload.get("candidate_assessments") or []:
            if isinstance(row, dict):
                _p130_normalize_typed_event_binding(
                    row,
                    required_event_ids=required_event_ids,
                    assigned_target=(
                        assigned_target_by_candidate.get(str(row.get("candidate_id") or ""), "")
                        if context.get("p132_active") else ""
                    ),
                )
                if context.get("p131_active"):
                    _p131_enforce_assigned_candidate_target(
                        row,
                        assigned_target_by_candidate=assigned_target_by_candidate,
                    )
        for row in payload.get("anchor_assessments") or []:
            if isinstance(row, dict):
                _p130_normalize_typed_event_binding(
                    row,
                    required_event_ids=required_event_ids,
                    assigned_target=(
                        assigned_target_by_candidate.get(str(row.get("candidate_id") or ""), "")
                        if context.get("p132_active") else ""
                    ),
                )
                if context.get("p131_active"):
                    _p131_enforce_assigned_candidate_target(
                        row,
                        assigned_target_by_candidate=assigned_target_by_candidate,
                    )
    _p130_localize_option_fields(payload)
    if "decision_sufficient" in payload:
        payload.setdefault(
            "local_decision_sufficient",
            payload.get("decision_sufficient") is True,
        )
    if "scope_coverage" in payload:
        payload.setdefault(
            "local_scope_coverage",
            str(payload.get("scope_coverage") or ""),
        )
    payload["decision_sufficient"] = False
    payload["scope_coverage"] = "partial"
    payload["scope_coverage_reason"] = (
        "P130 treats frame verification as a local evidence receipt; only the "
        "deterministic global reducer may establish Count/Order sufficiency."
    )
    payload["p130_local_receipt_only"] = True
    payload["p130_global_mode"] = str(context.get("mode") or "")
    return payload


def _apply_p130_contract_to_output(
    output: str,
    *,
    context: dict[str, Any],
) -> str:
    if not context.get("active"):
        return output
    payload = extract_v10_payload(output) or {}
    if not payload:
        return output
    _apply_p130_local_receipt_contract(payload, context=context)
    return format_v10_observation(payload, fallback_text=output)


def _multiwindow_verifier_instruction(
    *,
    question: str,
    query: str,
    candidate_text: str,
    retrieval_hint: str = "",
    semantic_decoupling_enabled: bool = False,
    p130_context: dict[str, Any] | None = None,
) -> str:
    prefix = (
        "You are the final visual verifier in a video QA agent. Several disjoint "
        "candidate windows were proposed by a local visual model. Compare all shown "
        "candidates neutrally; do not assume rank 1 is correct. Qwen captions are "
        "routing hints and may be wrong. Base support or contradiction only on the "
        "actual images.\n\n"
    )
    p130_context = p130_context or {"active": False}
    if p130_context.get("active"):
        question_stem = str(
            p130_context.get("parsed", {}).get("stem") or question
        ).strip()
        return (
            prefix
            + f"Question stem (answer choices intentionally withheld):\n{question_stem}\n\n"
            + "Local verification objective:\n"
            + _p130_local_verification_objective(p130_context)
            + "\n\n"
            + _p130_event_catalog_instruction(p130_context)
            + f"Candidate routing captions:\n{candidate_text}\n\n"
        )
    if not semantic_decoupling_enabled:
        return (
            prefix
            + f"Original multiple-choice question:\n{question}\n\n"
            + f"Verification objective:\n{query}\n\n"
            + f"Candidate routing captions:\n{candidate_text}\n\n"
        )
    return (
        prefix
        + f"Original multiple-choice question (authoritative):\n{question}\n\n"
        + "Use the original question and its answer choices as the only definition of "
        "what counts as evidence. The local retrieval objective is intentionally not "
        "part of the verification criterion.\n\n"
        + f"Candidate routing captions:\n{candidate_text}\n\n"
    )


def execute_frame_verify(config: dict, parameters: dict) -> str:
    """Run the strong visual observer through the existing focus implementation.

    CoSeek1 treats this as the only final frame-level verification tool. The
    wrapper forces the observer backend to API so local Qwen remains a routing
    aid rather than the final evidence source.
    """
    if config.get("p130_minimal_global_fsm_enabled") and should_use_minimal_global_fsm(str(parameters.get("question") or "")):
        return _execute_global_observation(config, parameters)
    verify_parameters = dict(parameters)
    if config.get("local_verifier_receipt_enabled", False):
        verify_parameters["_local_receipt_request"] = True
    p130_context = _p130_global_context(config, verify_parameters)
    if (
        config.get("coseek1_tool_integrity_repair_enabled", False)
        and not p130_context.get("p131_active")
    ):
        per_window = int(config.get("coseek1_tool_memory_anchors_per_window") or 4)
        if verify_parameters.get("candidate_windows"):
            candidates = [
                dict(item) if isinstance(item, dict) else {"t_range": item}
                for item in verify_parameters.get("candidate_windows") or []
            ]
            spans = [
                item.get("recommended_verify_window") or item.get("t_range")
                for item in candidates
            ]
            anchor_groups = collect_memory_anchor_rows(
                verify_parameters.get("memory"),
                spans,
                margin_s=0.0,
                per_window=per_window,
            )
            for candidate, rows in zip(candidates, anchor_groups):
                anchors = list(candidate.get("timestamp_anchors") or [])
                for row in rows:
                    timestamp = round(float(row["timestamp_s"]), 3)
                    if timestamp not in anchors:
                        anchors.append(timestamp)
                candidate["timestamp_anchors"] = sorted(anchors)
            verify_parameters["candidate_windows"] = candidates
        else:
            span = [
                verify_parameters.get("start_time"),
                verify_parameters.get("end_time"),
            ]
            anchor_groups = collect_memory_anchor_rows(
                verify_parameters.get("memory"),
                [span],
                margin_s=0.0,
                per_window=per_window,
            )
            verify_parameters["mandatory_timestamps"] = [
                round(float(item["timestamp_s"]), 3)
                for item in (anchor_groups[0] if anchor_groups else [])
            ]

    if verify_parameters.get("candidate_windows"):
        return execute_multiwindow_frame_verify(config, verify_parameters)

    verify_config = deepcopy(config)
    verify_config["observer_backend"] = "api"
    verify_config["local_qwen_tools"] = ""
    verify_parameters["force_option_evidence"] = not p130_context["active"]
    if p130_context["active"]:
        verify_parameters["query"] = (
            _p130_local_verification_objective(p130_context)
            + "\n\n"
            + _p130_event_catalog_instruction(p130_context)
            + (
                "In the returned JSON, also include target_event_id and "
                "candidate_event_ids at the top level."
                if p130_context["mode"] == "order"
                else ""
            )
        ).strip()
    raw = execute_focus(verify_config, verify_parameters)
    payload = extract_v10_payload(raw) or {}
    if payload.get("receipt_schema") == LOCAL_RECEIPT_SCHEMA:
        return raw
    if payload and config.get("packet_adaptive_evidence_need_enabled", False):
        _demote_partial_global_option_claims(payload)
        payload["evidence_need"] = _normalize_evidence_need(payload)
        payload["evidence_needs"] = _normalize_evidence_needs(payload)
        if config.get("packet_decision_consistency_normalization_enabled", False):
            _normalize_decision_completeness(payload, config.get("_question_option_letters"))
        raw = format_v10_observation(payload, fallback_text=raw)
    if (
        payload
        and config.get("packet_detail_grounding_direct_escalation_enabled", False)
        and config.get("packet_detail_grounding_escalation_enabled", False)
        and config.get("grounded_direct_verify_packet_enabled", False)
    ):
        raw = _run_direct_packet_detail_grounding_escalation(
            config=config,
            parameters=verify_parameters,
            initial_output=raw,
            initial_payload=payload,
        )
    raw = _apply_p130_contract_to_output(raw, context=p130_context)
    return raw.replace("Focus observation", "Frame-verify observation")


def _configured_max_windows(config: dict, parameters: dict) -> int:
    max_windows = int(config.get("localize_inline_verify_max_windows") or 3)
    if (
        config.get("p131_minimal_global_repairs_enabled", False)
        and should_use_minimal_global_fsm(
            str(parameters.get("question") or "")
        )
    ):
        max_windows = max(
            max_windows,
            min(8, len(parameters.get("candidate_windows") or [])),
        )
    if parameters.get("inline_event_coverage") is True:
        max_windows = max(
            max_windows,
            int(parameters.get("event_coverage_max_windows") or max_windows),
        )
    return max(1, max_windows)


def _candidate_id_for_timestamp(
    timestamp: Any,
    candidates: list[dict[str, Any]],
) -> str:
    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        return ""
    for candidate in candidates:
        start, end = candidate["t_range"]
        if start - 0.11 <= value <= end + 0.11:
            return candidate["candidate_id"]
    return ""


def _covered_candidate_ids(
    payload: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> list[str]:
    if not isinstance(payload, dict):
        return []
    requested = {item["candidate_id"] for item in candidates}
    covered: list[str] = []
    for row in payload.get("timestamp_observations") or []:
        if not isinstance(row, dict):
            continue
        try:
            timestamp = float(row.get("timestamp_s"))
        except (TypeError, ValueError):
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        candidate = next(
            (item for item in candidates if item["candidate_id"] == candidate_id),
            None,
        )
        if candidate is not None:
            start, end = candidate["t_range"]
            if not (start - 0.11 <= timestamp <= end + 0.11):
                candidate_id = ""
        if candidate_id not in requested:
            candidate_id = _candidate_id_for_timestamp(timestamp, candidates)
        if candidate_id and candidate_id not in covered:
            row["candidate_id"] = candidate_id
            covered.append(candidate_id)
    return covered


def _covered_candidate_binding_ids(
    payload: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> list[str]:
    if not isinstance(payload, dict):
        return []
    requested = {item["candidate_id"] for item in candidates}
    covered: list[str] = []
    for row in payload.get("candidate_assessments") or []:
        if not isinstance(row, dict):
            continue
        if row.get("assessment_present") is False:
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id in requested and candidate_id not in covered:
            covered.append(candidate_id)
    return covered


def _merge_multiwindow_recovery(
    *,
    candidates: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    retry_errors: list[str],
    retry_calls: int,
    initial_output: str,
    event_binding_enabled: bool = False,
    decision_consistency_enabled: bool = False,
    option_letters=None,
    p130_context: dict[str, Any] | None = None,
) -> str:
    if any(p.get("receipt_schema") == LOCAL_RECEIPT_SCHEMA for p in payloads):
        merged = merge_local_receipts(payloads, candidates)
        merged.update(recovery_calls=retry_calls, recovery_errors=retry_errors,
                      visual_packet_audits=[p.get("visual_packet_audit") for p in payloads if p.get("visual_packet_audit")],
                      num_frames=sum(int(p.get("num_frames") or 0) for p in payloads))
        return format_v10_observation(merged)
    p130_context = p130_context or {"active": False}
    required_event_ids = (
        set(p130_context.get("required_event_ids") or [])
        if p130_context.get("active") and p130_context.get("mode") == "order"
        else None
    )
    requested_ids = [item["candidate_id"] for item in candidates]
    observations: list[dict[str, Any]] = []
    covered_ids: list[str] = []
    seen_observations: set[tuple[Any, ...]] = set()
    for payload in payloads:
        for row in payload.get("timestamp_observations") or []:
            if not isinstance(row, dict):
                continue
            candidate_id = str(row.get("candidate_id") or "").strip()
            if candidate_id not in requested_ids:
                candidate_id = _candidate_id_for_timestamp(row.get("timestamp_s"), candidates)
            if not candidate_id:
                continue
            normalized = dict(row)
            normalized["candidate_id"] = candidate_id
            key = (
                candidate_id,
                normalized.get("timestamp_s"),
                str(normalized.get("description") or "").strip(),
            )
            if key in seen_observations:
                continue
            seen_observations.add(key)
            observations.append(normalized)
            if candidate_id not in covered_ids:
                covered_ids.append(candidate_id)

    missing_ids = [item for item in requested_ids if item not in covered_ids]
    supports: list[str] = []
    contradicts: list[str] = []
    option_evidence: dict[str, Any] = {}
    summaries: list[str] = []
    facts: list[str] = []
    missing_details: list[str] = []
    for payload in payloads:
        for letter in _option_letters(payload.get("supports_options")):
            if letter not in supports:
                supports.append(letter)
        for letter in _option_letters(payload.get("contradicts_options")):
            if letter not in contradicts:
                contradicts.append(letter)
        if isinstance(payload.get("option_evidence"), dict):
            option_evidence.update(payload["option_evidence"])
        summary = str(payload.get("overall_summary") or "").strip()
        if summary and summary not in summaries:
            summaries.append(summary)
        fact = str(payload.get("observed_fact") or "").strip()
        if fact and fact not in facts:
            facts.append(fact)
        detail = str(payload.get("missing_detail") or "").strip()
        # A child merge's coverage diagnostic is obsolete after recovery. Only
        # remove our exact generated sentence; retain semantic missing details.
        detail = re.sub(r"No timestamped observation returned for candidates \[[^\]]*\]\.\s*", "", detail).strip()
        if detail and detail not in missing_details:
            missing_details.append(detail)

    complete = not missing_ids and len(covered_ids) == len(requested_ids)
    child_detail_sufficient = bool(payloads) and all(
        payload.get("detail_sufficient") is True for payload in payloads
    )
    binding_payload_present = any(
        isinstance(payload.get("candidate_assessments"), list) for payload in payloads
    )
    candidate_assessments: list[dict[str, Any]] = []
    candidate_binding_complete = False
    binding_summary: dict[str, Any] = {}
    if binding_payload_present:
        raw_assessments: list[dict[str, Any]] = []
        for payload in payloads:
            raw_assessments.extend(
                item
                for item in (payload.get("candidate_assessments") or [])
                if isinstance(item, dict)
            )
        candidate_assessments, candidate_binding_complete = (
            _normalize_candidate_assessments(
                raw_assessments,
                candidates=candidates,
                event_binding_enabled=event_binding_enabled,
                required_event_ids=required_event_ids,
                count_occurrences_enabled=bool(
                    p130_context.get("p131_active")
                    and p130_context.get("mode") == "count"
                ),
                assigned_target_by_candidate=(
                    p130_context.get("assigned_target_by_candidate")
                    if p130_context.get("p132_active") else None
                ),
            )
        )
        _bind_candidate_observations(
            {"timestamp_observations": observations},
            candidates=candidates,
            assessments=candidate_assessments,
            allowed_by_candidate={},
            synthesize_assessment_anchor=not bool(p130_context.get("p131_active")),
        )
        binding_summary = _candidate_binding_summary(
            candidate_assessments,
            event_binding_enabled=event_binding_enabled,
        )
        target_match = binding_summary["target_match"]
    else:
        target_matches = {
            str(payload.get("target_match") or "").strip().lower() for payload in payloads
        }
        if "matched" in target_matches:
            target_match = "matched"
        elif "partial" in target_matches:
            target_match = "partial"
        elif "ambiguous" in target_matches:
            target_match = "ambiguous"
        elif target_matches == {"not_visible"}:
            target_match = "not_visible"
        else:
            target_match = "unknown"
    missing_detail = "; ".join(missing_details)[:600]
    if missing_ids:
        missing_detail = (
            f"No timestamped observation returned for candidates {missing_ids}. "
            + missing_detail
        ).strip()
    merged = {
        "tool": "frame_verify",
        "window_id": "inline_multiwindow_verify_recovered",
        "scene_id": "multi_scene",
        "t_range": [
            min(item["t_range"][0] for item in candidates),
            max(item["t_range"][1] for item in candidates),
        ],
        "timestamp_observations": observations,
        "overall_summary": " | ".join(summaries)[:1400],
        "observed_fact": " | ".join(facts)[:1400],
        "requested_candidate_ids": requested_ids,
        "verified_candidate_ids": covered_ids,
        "missing_candidate_ids": missing_ids,
        "verified_windows": [
            item["t_range"] for item in candidates if item["candidate_id"] in covered_ids
        ],
        "candidate_coverage_complete": complete,
        "target_match": target_match,
        "target_binding_reason": (
            "Every requested candidate has a timestamped visual observation."
            if complete
            else "Some requested candidates still lack timestamped visual observations."
        ),
        "supports_options": supports,
        "contradicts_options": contradicts,
        "option_evidence": option_evidence,
        "detail_sufficient": complete and child_detail_sufficient,
        "missing_detail": missing_detail,
        "observer_backend": "api",
        "parse_ok": any(payload.get("parse_ok") is True for payload in payloads),
        "num_frames": sum(int(payload.get("num_frames") or 0) for payload in payloads),
        "inline_localize_verify": True,
        "compact_render": True,
        "multiwindow_recovery_used": True,
        "multiwindow_recovery_calls": retry_calls,
        "multiwindow_recovery_errors": retry_errors,
        "retrieval_verify_semantic_decoupling": any(
            payload.get("retrieval_verify_semantic_decoupling") is True
            for payload in payloads
        ),
        "verification_objective_source": (
            "original_question"
            if any(
                payload.get("verification_objective_source") == "original_question"
                for payload in payloads
            )
            else "query"
        ),
        "retrieval_hint_present": any(
            payload.get("retrieval_hint_present") is True for payload in payloads
        ),
        "retrieval_hint_exposed_to_verifier": False,
    }
    if binding_payload_present:
        merged.update(binding_summary)
        merged["candidate_assessments"] = candidate_assessments
        merged["candidate_binding_complete"] = candidate_binding_complete
        if event_binding_enabled:
            _apply_event_binding_option_scope(merged)
    anchor_assessments = [
        deepcopy(row)
        for payload in payloads
        for row in payload.get("anchor_assessments") or []
        if isinstance(row, dict)
    ]
    if anchor_assessments:
        merged["anchor_assessments"] = anchor_assessments
    if p130_context.get("active"):
        merged["local_child_option_audits"] = [
            {
                key: deepcopy(value)
                for key, value in payload.items()
                if key.startswith("local_")
                and (
                    "option" in key
                    or key in {"local_decision_sufficient", "local_scope_coverage"}
                )
            }
            for payload in payloads
            if any(key.startswith("local_") for key in payload)
        ]
    if decision_consistency_enabled:
        _normalize_decision_completeness(merged, option_letters)
    _apply_p130_local_receipt_contract(merged, context=p130_context)
    return format_v10_observation(merged, fallback_text=initial_output)


def execute_multiwindow_frame_verify(config: dict, parameters: dict) -> str:
    """Verify candidates once, then recover only missing windows in-tool."""
    vr = parameters.get("vr")
    if vr is None:
        raise ValueError("multi-window frame_verify requires the active video reader")
    duration = float(parameters.get("duration") or round(len(vr) / vr.get_avg_fps(), 2))
    candidates = _normalize_candidate_windows(
        parameters.get("candidate_windows"),
        duration=duration,
        max_windows=_configured_max_windows(config, parameters),
        p131_enabled=bool(
            config.get("p131_minimal_global_repairs_enabled", False)
            and should_use_minimal_global_fsm(
                str(parameters.get("question") or "")
            )
        ),
        p131_mode=detect_global_mode(str(parameters.get("question") or "")),
        p132_enabled=bool(config.get("p132_weak_lead_visible_core_enabled", False)),
    )
    if not candidates:
        raise ValueError("multi-window frame_verify requires valid candidate_windows")

    p130_context = _p130_global_context(config, parameters)

    event_identity_repair_enabled = bool(
        config.get("coseek1_tool_event_identity_repair_enabled", False)
    )
    event_groups = _assign_candidate_context_groups(
        candidates,
        max_gap_s=float(config.get("grounded_candidate_event_group_max_gap_s") or 24.0),
        decouple_event_identity=event_identity_repair_enabled,
    )
    event_binding_enabled = _should_apply_candidate_event_binding(
        requested=bool(
            config.get("grounded_candidate_binding_enabled", False)
            and config.get("grounded_candidate_event_binding_enabled", False)
        ),
        event_groups=event_groups,
    )
    if config.get("grounded_candidate_event_binding_force_enabled", False):
        event_binding_enabled = bool(
            config.get("grounded_candidate_binding_enabled", False)
            and config.get("grounded_candidate_event_binding_enabled", False)
        )
    if p130_context["active"]:
        event_binding_enabled = True

    batch_enabled = bool(
        config.get("localize_inline_verify_batch_enabled", False)
        and not parameters.get("p124_verify_batch_child")
    )
    batch_size = max(
        1,
        int(config.get("localize_inline_verify_batch_size") or 4),
    )
    if batch_enabled and len(candidates) > batch_size:
        payloads: list[dict[str, Any]] = []
        batch_errors: list[str] = []
        batch_outputs: list[str] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            batch_parameters = dict(parameters)
            batch_parameters["candidate_windows"] = batch
            batch_parameters["windows"] = [item["t_range"] for item in batch]
            batch_parameters["start_time"] = min(
                item["t_range"][0] for item in batch
            )
            batch_parameters["end_time"] = max(
                item["t_range"][1] for item in batch
            )
            batch_parameters["p124_verify_batch_child"] = True
            try:
                batch_output = execute_multiwindow_frame_verify(
                    config,
                    batch_parameters,
                )
            except Exception as exc:
                batch_errors.append(
                    f"batch_{start // batch_size + 1}: "
                    f"{type(exc).__name__}: {str(exc)[:240]}"
                )
                continue
            batch_outputs.append(batch_output)
            batch_payload = extract_v10_payload(batch_output) or {}
            if batch_payload:
                payloads.append(batch_payload)
            else:
                batch_errors.append(
                    f"batch_{start // batch_size + 1}: no parseable observation"
                )
        return _merge_multiwindow_recovery(
            option_letters=config.get("_question_option_letters"),
            candidates=candidates,
            payloads=payloads,
            retry_errors=batch_errors,
            retry_calls=max(0, len(batch_outputs) - 1),
            initial_output=batch_outputs[0] if batch_outputs else "",
            event_binding_enabled=event_binding_enabled,
            decision_consistency_enabled=bool(
                config.get("packet_decision_consistency_normalization_enabled", False)
            ),
            p130_context=p130_context,
        )

    initial_error = ""
    try:
        initial_output = _execute_multiwindow_frame_verify_once(config, parameters)
    except Exception as exc:
        initial_error = f"{type(exc).__name__}: {str(exc)[:240]}"
        initial_output = ""
    initial_payload = extract_v10_payload(initial_output) or {}
    if (
        config.get("packet_detail_grounding_escalation_enabled", False)
        and config.get("grounded_verify_packet_enabled", False)
        and initial_payload
        and initial_payload.get("receipt_schema") != LOCAL_RECEIPT_SCHEMA
    ):
        initial_output = _run_packet_detail_grounding_escalation(
            config=config,
            parameters=parameters,
            candidates=candidates,
            initial_output=initial_output,
            initial_payload=initial_payload,
        )
        initial_payload = extract_v10_payload(initial_output) or initial_payload
    covered = _covered_candidate_ids(initial_payload, candidates)
    if config.get("grounded_candidate_binding_enabled", False):
        binding_covered = set(
            _covered_candidate_binding_ids(initial_payload, candidates)
        )
        covered = [item for item in covered if item in binding_covered]
    missing_ids = [
        item["candidate_id"] for item in candidates if item["candidate_id"] not in covered
    ]
    recovery_enabled = bool(config.get("multiwindow_verify_recovery_enabled", True))
    if not recovery_enabled or not missing_ids:
        return _apply_p130_contract_to_output(
            initial_output,
            context=p130_context,
        )

    max_calls = max(0, int(config.get("multiwindow_verify_recovery_max_calls") or 2))
    batch_size = max(1, int(config.get("multiwindow_verify_recovery_batch_size") or 2))
    missing_candidates = [
        item for item in candidates if item["candidate_id"] in missing_ids
    ]
    payloads = [initial_payload] if initial_payload else []
    retry_errors = [initial_error] if initial_error else []
    retry_calls = 0
    for start in range(0, len(missing_candidates), batch_size):
        if retry_calls >= max_calls:
            break
        batch = missing_candidates[start : start + batch_size]
        retry_parameters = dict(parameters)
        retry_parameters["candidate_windows"] = batch
        retry_parameters["windows"] = [item["t_range"] for item in batch]
        retry_parameters["start_time"] = min(item["t_range"][0] for item in batch)
        retry_parameters["end_time"] = max(item["t_range"][1] for item in batch)
        retry_parameters["multiwindow_recovery_subset"] = True
        retry_calls += 1
        try:
            retry_output = _execute_multiwindow_frame_verify_once(config, retry_parameters)
        except Exception as exc:
            retry_errors.append(f"{type(exc).__name__}: {str(exc)[:240]}")
            continue
        retry_payload = extract_v10_payload(retry_output) or {}
        if retry_payload:
            payloads.append(retry_payload)
        else:
            retry_errors.append("retry returned no parseable observation")

    return _merge_multiwindow_recovery(
        option_letters=config.get("_question_option_letters"),
        candidates=candidates,
        payloads=payloads,
        retry_errors=retry_errors,
        retry_calls=retry_calls,
        initial_output=initial_output,
        event_binding_enabled=event_binding_enabled,
        decision_consistency_enabled=bool(
            config.get("packet_decision_consistency_normalization_enabled", False)
        ),
        p130_context=p130_context,
    )


def _normalize_candidate_windows(
    values: Any,
    *,
    duration: float,
    max_windows: int,
    p131_enabled: bool = False,
    p131_mode: str = "",
    p132_enabled: bool = False,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(values or []):
        item = dict(value) if isinstance(value, dict) else {"t_range": value}
        span = item.get("recommended_verify_window") or item.get("t_range")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        try:
            start = max(0.0, min(float(span[0]), duration))
            end = max(0.0, min(float(span[1]), duration))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        candidate = {
            "candidate_id": str(item.get("candidate_id") or f"LQ{index + 1:03d}"),
            "t_range": [round(start, 3), round(end, 3)],
            "summary": str(item.get("fine_summary") or item.get("summary") or "").strip()[:480],
            "rank": int(item.get("rank") or index + 1),
            "source_index": int(item.get("source_index") or 0),
            "timestamp_anchors": [],
            "localized_timestamp_anchors": [],
            "memory_timestamp_anchors": [],
            "localized_evidence_anchors": [],
        }
        localized_span = item.get("localized_verify_window")
        if isinstance(localized_span, (list, tuple)) and len(localized_span) == 2:
            try:
                localized_start = max(start, float(localized_span[0]))
                localized_end = min(end, float(localized_span[1]))
            except (TypeError, ValueError):
                localized_start = localized_end = 0.0
            if localized_end > localized_start:
                candidate["localized_verify_window"] = [
                    round(localized_start, 3),
                    round(localized_end, 3),
                ]
        core_span = candidate.get("localized_verify_window") or candidate["t_range"]
        source_span = item.get("source_search_window")
        if isinstance(source_span, (list, tuple)) and len(source_span) == 2:
            try:
                source_start = max(0.0, min(float(source_span[0]), duration))
                source_end = max(0.0, min(float(source_span[1]), duration))
            except (TypeError, ValueError):
                source_start = source_end = 0.0
            if source_end > source_start:
                candidate["source_search_window"] = [
                    round(source_start, 3),
                    round(source_end, 3),
                ]
        for field in (
            "timestamp_anchors",
            "localized_timestamp_anchors",
            "memory_timestamp_anchors",
        ):
            if p131_enabled and field == "memory_timestamp_anchors":
                continue
            for raw_anchor in item.get(field) or []:
                try:
                    anchor = float(raw_anchor)
                except (TypeError, ValueError):
                    continue
                normalized_anchor = round(anchor, 3)
                if (
                    (core_span[0] if p131_enabled else start)
                    <= anchor
                    <= (core_span[1] if p131_enabled else end)
                    and normalized_anchor not in candidate[field]
                ):
                    candidate[field].append(normalized_anchor)
        if p131_enabled:
            candidate["summary"] = ""
            candidate["mandatory_positive"] = False
            candidate["selection_class"] = str(
                item.get("selection_class") or "ambiguous"
            )
            candidate["p131_routing_only_coverage"] = bool(
                item.get("p131_routing_only_coverage", False)
            )
            raw_target = item.get("target_event_id")
            raw_targets = item.get("target_event_ids") or []
            if raw_target in (None, "") and isinstance(raw_targets, (list, tuple)):
                raw_target = raw_targets[0] if len(raw_targets) == 1 else ""
            target_event_id = _p130_normalize_event_id(raw_target)
            candidate["target_event_id"] = target_event_id
            candidate["target_event_ids"] = (
                [target_event_id] if target_event_id else []
            )
            if p131_mode == "count" and not p132_enabled:
                start_core, end_core = core_span
                dense = [
                    round(start_core + (end_core - start_core) * index / 4.0, 3)
                    for index in range(5)
                ]
                candidate["timestamp_anchors"] = sorted(
                    {
                        *candidate["timestamp_anchors"],
                        *dense,
                    }
                )
        confidence_rank = {"high": 2, "medium": 1, "low": 0}
        evidence_by_timestamp: dict[float, str] = {}
        for raw_item in item.get("localized_evidence_anchors") or []:
            if not isinstance(raw_item, dict):
                continue
            try:
                anchor = float(raw_item.get("timestamp_s"))
            except (TypeError, ValueError):
                continue
            if not start <= anchor <= end:
                continue
            confidence = str(raw_item.get("confidence") or "low").strip().lower()
            if confidence not in confidence_rank:
                confidence = "low"
            timestamp = round(anchor, 3)
            previous = evidence_by_timestamp.get(timestamp)
            if previous is None or confidence_rank[confidence] > confidence_rank[previous]:
                evidence_by_timestamp[timestamp] = confidence
        candidate["localized_evidence_anchors"] = [
            {"timestamp_s": timestamp, "confidence": confidence}
            for timestamp, confidence in sorted(evidence_by_timestamp.items())
        ]
        if any(
            old["t_range"] == candidate["t_range"]
            and (
                not p131_enabled
                or old.get("target_event_id") == candidate.get("target_event_id")
            )
            for old in normalized
        ):
            continue
        normalized.append(candidate)
        if len(normalized) >= max(1, int(max_windows)):
            break
    return normalized


def _allocate_candidate_frames(window_count: int, total_frames: int) -> list[int]:
    if window_count <= 0:
        return []
    total_frames = max(window_count * 2, int(total_frames))
    base, remainder = divmod(total_frames, window_count)
    return [base + (1 if index < remainder else 0) for index in range(window_count)]


def _evenly_limit_positions(values: list[int], limit: int) -> list[int]:
    """Keep temporal coverage when a compact packet has few detail tiles."""
    unique = sorted(set(int(value) for value in values))
    limit = max(1, int(limit))
    if len(unique) <= limit:
        return unique
    indices = np.linspace(0, len(unique) - 1, limit).round().astype(int)
    return [unique[int(index)] for index in sorted(set(indices.tolist()))]


def _packet_detail_anchor_positions(
    *,
    candidate: dict[str, Any],
    positioned_rows: list[tuple[int, float, Any]],
    limit: int,
    query_context_roles_enabled: bool = False,
    boundary_context_enabled: bool = False,
) -> tuple[list[int], str]:
    """Choose temporally distributed high-resolution packet anchors.

    Local Qwen normally emits many relevant timestamps. Taking the first N
    systematically magnifies the beginning of every candidate and leaves the
    event peak/end only in the low-resolution strip. Prefer the localized
    evidence span, distribute the fixed tile budget over it, and only use
    uniform candidate frames when Qwen supplied too few distinct anchors.
    """
    if not positioned_rows:
        return [], "none"
    limit = max(1, int(limit))
    confidence_rank = {"high": 2, "medium": 1, "low": 0}
    scored_positions: dict[int, int] = {}
    for item in candidate.get("localized_evidence_anchors") or []:
        if not isinstance(item, dict):
            continue
        try:
            anchor_value = float(item.get("timestamp_s"))
        except (TypeError, ValueError):
            continue
        nearest_position = min(
            positioned_rows,
            key=lambda row: abs(float(row[1]) - anchor_value),
        )[0]
        confidence = str(item.get("confidence") or "low").strip().lower()
        scored_positions[nearest_position] = max(
            scored_positions.get(nearest_position, -1),
            confidence_rank.get(confidence, 0),
        )
    if scored_positions:
        best_score = max(scored_positions.values())
        strongest = sorted(
            position
            for position, score in scored_positions.items()
            if score == best_score
        )
        if query_context_roles_enabled and limit >= 2:
            row_by_position = {row[0]: row for row in positioned_rows}
            localized_window = candidate.get("localized_verify_window") or []
            if isinstance(localized_window, (list, tuple)) and len(localized_window) == 2:
                try:
                    localized_start = float(localized_window[0])
                    localized_end = float(localized_window[1])
                    query_center = (localized_start + localized_end) / 2.0
                except (TypeError, ValueError):
                    localized_start = localized_end = None
                    query_center = sum(row_by_position[pos][1] for pos in strongest) / len(
                        strongest
                    )
            else:
                localized_start = localized_end = None
                query_center = sum(row_by_position[pos][1] for pos in strongest) / len(
                    strongest
                )
            query_peak = min(
                strongest,
                key=lambda position: (
                    abs(float(row_by_position[position][1]) - query_center),
                    position,
                ),
            )
            if boundary_context_enabled:
                query_timestamp = float(row_by_position[query_peak][1])
                memory_positions: list[int] = []
                for raw_timestamp in candidate.get("memory_timestamp_anchors") or []:
                    try:
                        timestamp = float(raw_timestamp)
                    except (TypeError, ValueError):
                        continue
                    nearest = min(
                        positioned_rows,
                        key=lambda row: abs(float(row[1]) - timestamp),
                    )[0]
                    if nearest != query_peak and nearest not in memory_positions:
                        memory_positions.append(nearest)
                post_memory = [
                    position
                    for position in memory_positions
                    if float(row_by_position[position][1]) > query_timestamp + 0.05
                ]
                if post_memory:
                    event_context = max(
                        post_memory,
                        key=lambda position: float(row_by_position[position][1]),
                    )
                else:
                    localized_rows = [
                        row
                        for row in positioned_rows
                        if row[0] != query_peak
                        and localized_start is not None
                        and localized_end is not None
                        and localized_start <= float(row[1]) <= localized_end
                    ]
                    post_rows = [
                        row
                        for row in localized_rows
                        if float(row[1]) > query_timestamp + 0.05
                    ]
                    context_pool = post_rows or localized_rows
                    event_context = (
                        max(
                            context_pool,
                            key=lambda row: (
                                abs(float(row[1]) - query_timestamp),
                                float(row[1]),
                            ),
                        )[0]
                        if context_pool
                        else None
                    )
                positions = [query_peak]
                if event_context is not None:
                    positions.append(event_context)
                for position in strongest:
                    if position not in positions:
                        positions.append(position)
                    if len(positions) >= limit:
                        break
                if len(positions) < limit:
                    supplements = _evenly_limit_positions(
                        [row[0] for row in positioned_rows],
                        limit,
                    )
                    for position in supplements:
                        if position not in positions:
                            positions.append(position)
                        if len(positions) >= limit:
                            break
                return positions[:limit], "query_peak_plus_boundary_context"
            context_rows = [
                row
                for row in positioned_rows
                if row[0] != query_peak
                and localized_start is not None
                and localized_end is not None
                and (
                    float(row[1]) < localized_start - 0.05
                    or float(row[1]) > localized_end + 0.05
                )
            ]
            if not context_rows:
                context_rows = [
                    row
                    for row in (positioned_rows[0], positioned_rows[-1])
                    if row[0] != query_peak
                ]
            query_frame = row_by_position[query_peak][2]

            def context_score(row: tuple[int, float, Any]) -> tuple[float, float]:
                try:
                    query_image = Image.fromarray(query_frame).convert("RGB").resize((24, 24))
                    context_image = Image.fromarray(row[2]).convert("RGB").resize((24, 24))
                    visual_change = float(
                        np.abs(
                            np.asarray(query_image, dtype=np.float32)
                            - np.asarray(context_image, dtype=np.float32)
                        ).mean()
                    )
                except (TypeError, ValueError, OSError):
                    visual_change = 0.0
                temporal_distance = abs(
                    float(row[1]) - float(row_by_position[query_peak][1])
                )
                return visual_change, temporal_distance

            if context_rows:
                event_context = max(context_rows, key=context_score)[0]
                positions = [query_peak, event_context]
            else:
                positions = [query_peak]
            for position in strongest:
                if position not in positions:
                    positions.append(position)
                if len(positions) >= limit:
                    break
            if len(positions) < limit:
                supplements = _evenly_limit_positions(
                    [row[0] for row in positioned_rows],
                    limit,
                )
                for position in supplements:
                    if position not in positions:
                        positions.append(position)
                    if len(positions) >= limit:
                        break
            return positions[:limit], "query_peak_plus_event_context"
        positions = _evenly_limit_positions(strongest, limit)
        source = "query_confidence_then_context"
        if len(positions) < limit:
            supplements = _evenly_limit_positions(
                [row[0] for row in positioned_rows],
                limit,
            )
            for position in supplements:
                if position not in positions:
                    positions.append(position)
                if len(positions) >= limit:
                    break
        return sorted(positions[:limit]), source

    preferred = (
        candidate.get("localized_timestamp_anchors")
        or candidate.get("timestamp_anchors")
        or []
    )
    positions: list[int] = []
    for anchor in preferred:
        try:
            anchor_value = float(anchor)
        except (TypeError, ValueError):
            continue
        nearest_position = min(
            positioned_rows,
            key=lambda row: abs(float(row[1]) - anchor_value),
        )[0]
        if nearest_position not in positions:
            positions.append(nearest_position)
    source = (
        "localized_span_even"
        if candidate.get("localized_timestamp_anchors")
        else "timestamp_span_even"
        if candidate.get("timestamp_anchors")
        else "uniform_span_even"
    )
    positions = _evenly_limit_positions(positions, limit) if positions else []
    if len(positions) < limit:
        supplements = _evenly_limit_positions(
            [row[0] for row in positioned_rows],
            limit,
        )
        for position in supplements:
            if position not in positions:
                positions.append(position)
            if len(positions) >= limit:
                break
        positions = sorted(positions)
    return positions[:limit], source


def _merge_candidate_sampling_timestamps(
    base_timestamps: list[float],
    timestamp_anchors: list[float],
    *,
    start: float,
    end: float,
    budget: int,
    priority_anchors: list[float] | None = None,
) -> list[float]:
    """Keep exact local anchors, then use remaining slots for temporal context."""
    budget = max(1, int(budget))
    priorities = sorted(
        {
            round(float(value), 3)
            for value in (priority_anchors or [])
            if start <= float(value) <= end
        }
    )
    if len(priorities) > budget:
        positions = np.linspace(0, len(priorities) - 1, budget).round().astype(int)
        priorities = [
            priorities[int(position)] for position in sorted(set(positions.tolist()))
        ]
    anchors = sorted(
        {
            round(float(value), 3)
            for value in timestamp_anchors
            if start <= float(value) <= end
        }
    )
    anchor_budget = max(0, budget - len(priorities))
    anchors = [
        anchor
        for anchor in anchors
        if all(abs(anchor - priority) > 0.05 for priority in priorities)
    ]
    if len(anchors) > anchor_budget and anchor_budget > 0:
        positions = np.linspace(0, len(anchors) - 1, anchor_budget).round().astype(int)
        anchors = [anchors[int(position)] for position in sorted(set(positions.tolist()))]
    elif anchor_budget <= 0:
        anchors = []
    selected = list(priorities) + list(anchors[:anchor_budget])
    remaining = [
        round(float(value), 3)
        for value in base_timestamps
        if start <= float(value) <= end
        and all(abs(float(value) - anchor) > 0.05 for anchor in selected)
    ]
    slots = budget - len(selected)
    if slots > 0 and remaining:
        if len(remaining) > slots:
            positions = np.linspace(0, len(remaining) - 1, slots).round().astype(int)
            remaining = [
                remaining[int(position)] for position in sorted(set(positions.tolist()))
            ]
        selected.extend(remaining[:slots])
    return sorted(set(selected))


def _option_letters(values: Any) -> list[str]:
    letters: list[str] = []
    for value in values or []:
        if isinstance(value, dict):
            raw = value.get("option") or value.get("letter") or value.get("label") or ""
        else:
            raw = value
        text = str(raw).strip().upper()
        match = re.fullmatch(r"(?:OPTION\s*)?\(?([A-Z])\)?[.]?", text)
        letter = match.group(1) if match else ""
        if letter and letter not in letters:
            letters.append(letter)
    return letters


_CANDIDATE_TARGET_MATCHES = {
    "matched",
    "partial",
    "ambiguous",
    "mismatch",
    "not_visible",
    "unknown",
}

_CANDIDATE_EVENT_MATCHES = {
    "direct",
    "context_only",
    "different_event",
    "ambiguous",
    "not_visible",
    "unknown",
}


def _candidate_event_groups(
    candidates: list[dict[str, Any]],
    *,
    max_gap_s: float,
) -> dict[str, str]:
    """Group only temporally connected candidates for verifier presentation."""
    groups: dict[str, str] = {}
    group_index = 0
    group_end: float | None = None
    for candidate in sorted(candidates, key=lambda item: item["t_range"]):
        start, end = candidate["t_range"]
        if group_end is None or start - group_end > max(0.0, float(max_gap_s)):
            group_index += 1
            group_end = end
        else:
            group_end = max(group_end, end)
        groups[candidate["candidate_id"]] = f"EG{group_index:03d}"
    return groups


def _assign_candidate_context_groups(
    candidates: list[dict[str, Any]],
    *,
    max_gap_s: float,
    decouple_event_identity: bool,
) -> dict[str, str]:
    """Assign visual context groups without inventing semantic event identity."""
    groups = _candidate_event_groups(
        candidates,
        max_gap_s=0.0 if decouple_event_identity else max_gap_s,
    )
    for candidate in candidates:
        group_id = groups[candidate["candidate_id"]]
        if decouple_event_identity:
            candidate.pop("event_group_id", None)
            candidate["packet_group_id"] = group_id.replace("EG", "PG", 1)
        else:
            candidate["event_group_id"] = group_id
            candidate.pop("packet_group_id", None)
    return groups


def _should_apply_candidate_event_binding(
    *,
    requested: bool,
    event_groups: dict[str, str],
) -> bool:
    """Use event-instance semantics only when candidates span separate episodes.

    A single candidate or one temporally connected chain already has an
    unambiguous local episode. Asking the verifier for another event label in
    that case adds schema noise without resolving cross-window causality.
    """
    return bool(requested and len(set(event_groups.values())) > 1)


def _candidate_allowed_timestamps(
    sampled: list[tuple[str, float, Any]],
) -> dict[str, list[float]]:
    allowed: dict[str, list[float]] = {}
    for candidate_id, timestamp, _ in sampled:
        rows = allowed.setdefault(str(candidate_id), [])
        value = round(float(timestamp), 1)
        if value not in rows:
            rows.append(value)
    return allowed


def _normalize_candidate_assessments(
    values: Any,
    *,
    candidates: list[dict[str, Any]],
    allowed_by_candidate: dict[str, list[float]] | None = None,
    event_binding_enabled: bool = False,
    required_event_ids: set[str] | None = None,
    count_occurrences_enabled: bool = False,
    assigned_target_by_candidate: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Validate per-candidate binding without turning it into routing policy."""
    allowed_by_candidate = allowed_by_candidate or {}
    requested_ids = [item["candidate_id"] for item in candidates]
    candidate_by_id = {item["candidate_id"]: item for item in candidates}
    requested = set(requested_ids)
    parsed_by_id: dict[str, dict[str, Any]] = {}
    count_occurrences_by_id: dict[str, list[dict[str, Any]]] = {}
    returned_ids: set[str] = set()
    if isinstance(values, list):
        for raw in values:
            if not isinstance(raw, dict):
                continue
            candidate_id = str(raw.get("candidate_id") or "").strip()
            if candidate_id not in requested:
                continue
            status = str(raw.get("target_match") or "unknown").strip().lower()
            if status not in _CANDIDATE_TARGET_MATCHES:
                status = "unknown"
            event_match = str(raw.get("event_match") or "unknown").strip().lower()
            if event_match not in _CANDIDATE_EVENT_MATCHES:
                event_match = "unknown"
            if (
                count_occurrences_enabled
                and event_match == "direct"
                and status in {"mismatch", "not_visible"}
            ):
                # The two legacy status fields describe the same local Count
                # proposition.  An explicit contradiction must fail closed.
                event_match = "ambiguous"
            timestamp = raw.get("best_timestamp_s")
            try:
                timestamp = float(timestamp) if timestamp is not None else None
            except (TypeError, ValueError):
                timestamp = None
            allowed = allowed_by_candidate.get(candidate_id) or []
            if timestamp is not None and allowed:
                nearest = min(allowed, key=lambda item: abs(item - timestamp))
                timestamp = (
                    None if assigned_target_by_candidate
                    and (not math.isfinite(timestamp) or abs(nearest - timestamp) > 0.11)
                    else nearest
                )
            event_span = raw.get("event_span")
            normalized_event_span = None
            candidate_span = (candidate_by_id.get(candidate_id) or {}).get("t_range") or []
            if (
                isinstance(event_span, (list, tuple))
                and len(event_span) == 2
                and isinstance(candidate_span, (list, tuple))
                and len(candidate_span) == 2
            ):
                try:
                    event_start = max(float(candidate_span[0]), float(event_span[0]))
                    event_end = min(float(candidate_span[1]), float(event_span[1]))
                except (TypeError, ValueError):
                    event_start = event_end = 0.0
                if event_end > event_start:
                    normalized_event_span = [
                        round(event_start, 3),
                        round(event_end, 3),
                    ]
            option_conflict = raw.get("option_set_conflict") is True or str(
                raw.get("option_set_conflict") or ""
            ).strip().lower() in {"true", "yes", "1"}
            normalized = {
                "candidate_id": candidate_id,
                "event_group_id": str(
                    (candidate_by_id.get(candidate_id) or {}).get("event_group_id") or ""
                ),
                "packet_group_id": str(
                    (candidate_by_id.get(candidate_id) or {}).get("packet_group_id") or ""
                ),
                "target_match": status,
                "event_match": event_match if event_binding_enabled else "unknown",
                "observed_fact": str(
                    raw.get("observed_fact") or raw.get("description") or ""
                ).strip()[:600],
                "target_binding_reason": str(
                    raw.get("target_binding_reason") or raw.get("binding_reason") or ""
                ).strip()[:600],
                "best_timestamp_s": round(timestamp, 1) if timestamp is not None else None,
                "event_span": normalized_event_span,
                "supports_options": _option_letters(raw.get("supports_options")),
                "contradicts_options": _option_letters(raw.get("contradicts_options")),
                "option_set_conflict": option_conflict,
                "assessment_present": True,
            }
            if count_occurrences_enabled:
                nested = raw.get("count_occurrences")
                if not isinstance(nested, list):
                    nested = raw.get("occurrences")
                occurrence_sources = (
                    [item for item in nested if isinstance(item, dict)]
                    if isinstance(nested, list) and nested
                    else [raw]
                )
                occurrence_rows = count_occurrences_by_id.setdefault(candidate_id, [])
                for occurrence_raw in occurrence_sources:
                    occurrence_match = str(
                        occurrence_raw.get("event_match") or event_match or "unknown"
                    ).strip().lower()
                    if occurrence_match not in _CANDIDATE_EVENT_MATCHES:
                        occurrence_match = "unknown"
                    occurrence_target = str(
                        occurrence_raw.get("target_match") or status or "unknown"
                    ).strip().lower()
                    if occurrence_target not in _CANDIDATE_TARGET_MATCHES:
                        occurrence_target = "unknown"
                    if occurrence_match == "direct" and (
                        occurrence_target in {"mismatch", "not_visible"}
                        or status in {"mismatch", "not_visible"}
                    ):
                        occurrence_match = "ambiguous"
                    occurrence_span = occurrence_raw.get("event_span")
                    normalized_occurrence_span = None
                    if (
                        isinstance(occurrence_span, (list, tuple))
                        and len(occurrence_span) == 2
                        and isinstance(candidate_span, (list, tuple))
                        and len(candidate_span) == 2
                    ):
                        try:
                            occurrence_start = max(
                                float(candidate_span[0]), float(occurrence_span[0])
                            )
                            occurrence_end = min(
                                float(candidate_span[1]), float(occurrence_span[1])
                            )
                        except (TypeError, ValueError):
                            occurrence_start = occurrence_end = 0.0
                        if occurrence_end > occurrence_start:
                            normalized_occurrence_span = [
                                round(occurrence_start, 3),
                                round(occurrence_end, 3),
                            ]
                    raw_occurrence_timestamp = occurrence_raw.get("best_timestamp_s")
                    if raw_occurrence_timestamp is None:
                        raw_occurrence_timestamp = occurrence_raw.get("timestamp_s")
                    try:
                        occurrence_timestamp = (
                            float(raw_occurrence_timestamp)
                            if raw_occurrence_timestamp is not None
                            else None
                        )
                    except (TypeError, ValueError):
                        occurrence_timestamp = None
                    if occurrence_timestamp is not None and allowed:
                        nearest = min(
                            allowed,
                            key=lambda item: abs(item - occurrence_timestamp),
                        )
                        occurrence_timestamp = (
                            nearest
                            if abs(nearest - occurrence_timestamp) <= 0.11
                            else None
                        )
                    occurrence_fact = str(
                        occurrence_raw.get("observed_fact")
                        or occurrence_raw.get("description")
                        or ""
                    ).strip()[:600]
                    occurrence = {
                        "event_match": occurrence_match,
                        "target_match": occurrence_target,
                        "event_span": normalized_occurrence_span,
                        "best_timestamp_s": (
                            round(occurrence_timestamp, 1)
                            if occurrence_timestamp is not None
                            else None
                        ),
                        "observed_fact": occurrence_fact,
                    }
                    occurrence_rows = merge_p131_count_occurrences(
                        occurrence_rows,
                        [occurrence],
                    )
                    count_occurrences_by_id[candidate_id] = occurrence_rows
            for key, value in raw.items():
                if str(key).startswith("local_"):
                    normalized[str(key)] = deepcopy(value)
            if required_event_ids is not None:
                for key in (
                    "target_event_id",
                    "event_index",
                    "ordered_event_id",
                    "candidate_event_ids",
                    "possible_event_ids",
                    "target_event_ids",
                    "matched_event_ids",
                ):
                    if key in raw:
                        normalized[key] = deepcopy(raw.get(key))
                if assigned_target_by_candidate:
                    for key in ("event_id_conflict", "invalid_event_ids"):
                        if key in raw:
                            normalized[key] = deepcopy(raw[key])
                _p130_normalize_typed_event_binding(
                    normalized,
                    required_event_ids=required_event_ids,
                    assigned_target=(assigned_target_by_candidate or {}).get(candidate_id, ""),
                )
            old = parsed_by_id.get(candidate_id)
            if old is None or (
                old.get("target_match") == "unknown" and status != "unknown"
            ):
                parsed_by_id[candidate_id] = normalized
            returned_ids.add(candidate_id)

    assessments: list[dict[str, Any]] = []
    for candidate_id in requested_ids:
        row = parsed_by_id.get(candidate_id)
        if row is None:
            row = {
                "candidate_id": candidate_id,
                "event_group_id": str(
                    (candidate_by_id.get(candidate_id) or {}).get("event_group_id") or ""
                ),
                "packet_group_id": str(
                    (candidate_by_id.get(candidate_id) or {}).get("packet_group_id") or ""
                ),
                "target_match": "unknown",
                "event_match": "unknown",
                "observed_fact": "",
                "target_binding_reason": "No candidate-specific assessment was returned.",
                "best_timestamp_s": None,
                "event_span": None,
                "supports_options": [],
                "contradicts_options": [],
                "option_set_conflict": False,
                "assessment_present": False,
            }
            if required_event_ids is not None:
                _p130_normalize_typed_event_binding(
                    row,
                    required_event_ids=required_event_ids,
                )
        assessments.append(row)
        if count_occurrences_enabled and candidate_id in count_occurrences_by_id:
            row["count_occurrences"] = count_occurrences_by_id[candidate_id]
    return assessments, returned_ids == requested


def _normalize_anchor_assessments(
    values: Any,
    *,
    allowed_by_candidate: dict[str, list[float]],
    event_binding_enabled: bool,
    required_event_ids: set[str] | None = None,
    assigned_target_by_candidate: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Validate one observation for each high-resolution packet anchor."""
    expected = {
        (candidate_id, round(float(timestamp), 1))
        for candidate_id, timestamps in allowed_by_candidate.items()
        for timestamp in timestamps
    }
    normalized_by_key: dict[tuple[str, float], dict[str, Any]] = {}
    for raw in values if isinstance(values, list) else []:
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("candidate_id") or "").strip()
        allowed = allowed_by_candidate.get(candidate_id) or []
        if not allowed:
            continue
        try:
            timestamp = float(raw.get("timestamp_s"))
        except (TypeError, ValueError):
            continue
        nearest = min(allowed, key=lambda item: abs(item - timestamp))
        if assigned_target_by_candidate and (
            not math.isfinite(timestamp) or abs(nearest - timestamp) > 0.11
        ):
            continue
        timestamp = round(nearest, 1)
        target_match = str(raw.get("target_match") or "unknown").strip().lower()
        if target_match not in _CANDIDATE_TARGET_MATCHES:
            target_match = "unknown"
        event_match = str(raw.get("event_match") or "unknown").strip().lower()
        if event_match not in _CANDIDATE_EVENT_MATCHES:
            event_match = "unknown"
        row = {
            "candidate_id": candidate_id,
            "timestamp_s": timestamp,
            "target_match": target_match,
            "event_match": event_match if event_binding_enabled else "unknown",
            "observed_fact": str(
                raw.get("observed_fact") or raw.get("description") or ""
            ).strip()[:600],
            "target_binding_reason": str(
                raw.get("target_binding_reason") or raw.get("binding_reason") or ""
            ).strip()[:600],
            "supports_options": _option_letters(raw.get("supports_options")),
            "contradicts_options": _option_letters(raw.get("contradicts_options")),
            "option_set_conflict": raw.get("option_set_conflict") is True,
            "anchor_audit_observation": True,
        }
        for key, value in raw.items():
            if str(key).startswith("local_"):
                row[str(key)] = deepcopy(value)
        if required_event_ids is not None:
            for key in (
                "target_event_id",
                "event_index",
                "ordered_event_id",
                "candidate_event_ids",
                "possible_event_ids",
                "target_event_ids",
                "matched_event_ids",
            ):
                if key in raw:
                    row[key] = deepcopy(raw.get(key))
            if assigned_target_by_candidate:
                for key in ("event_id_conflict", "invalid_event_ids"):
                    if key in raw:
                        row[key] = deepcopy(raw[key])
            _p130_normalize_typed_event_binding(
                row,
                required_event_ids=required_event_ids,
                assigned_target=(assigned_target_by_candidate or {}).get(candidate_id, ""),
            )
        normalized_by_key[(candidate_id, timestamp)] = row
    return list(normalized_by_key.values()), expected.issubset(normalized_by_key)


def _merge_anchor_audit_into_candidate_assessments(
    assessments: list[dict[str, Any]],
    anchor_assessments: list[dict[str, Any]],
    *,
    event_binding_enabled: bool,
    required_event_ids: set[str] | None = None,
    assigned_target_by_candidate: dict[str, str] | None = None,
) -> None:
    """Preserve direct evidence found at any inspected anchor.

    A candidate-level summary must not erase a direct event visible at another
    timestamp in the same packet. Only an explicit anchor-level `direct` label
    can upgrade a candidate; no event is inferred from captions or keywords.
    """
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in anchor_assessments:
        rows_by_candidate.setdefault(str(row.get("candidate_id") or ""), []).append(row)
    for assessment in assessments:
        rows = rows_by_candidate.get(str(assessment.get("candidate_id") or "")) or []
        assessment["anchor_audit_observation_count"] = len(rows)
        if not event_binding_enabled:
            continue
        direct_rows = [row for row in rows if row.get("event_match") == "direct"]
        if not direct_rows:
            continue
        best = next(
            (row for row in direct_rows if row.get("target_match") == "matched"),
            direct_rows[0],
        )
        assessment["event_match"] = "direct"
        if best.get("target_match") in {"matched", "partial", "ambiguous"}:
            assessment["target_match"] = best["target_match"]
        assessment["observed_fact"] = best.get("observed_fact") or assessment.get(
            "observed_fact", ""
        )
        assessment["target_binding_reason"] = best.get(
            "target_binding_reason"
        ) or assessment.get("target_binding_reason", "")
        assessment["best_timestamp_s"] = best.get("timestamp_s")
        assessment["supports_options"] = list(
            dict.fromkeys(
                letter
                for row in direct_rows
                for letter in row.get("supports_options") or []
            )
        )
        assessment["contradicts_options"] = list(
            dict.fromkeys(
                letter
                for row in direct_rows
                for letter in row.get("contradicts_options") or []
            )
        )
        assessment["option_set_conflict"] = any(
            row.get("option_set_conflict") is True for row in direct_rows
        )
        assessment["anchor_audit_direct_preserved"] = True
        if required_event_ids is not None:
            candidate_ids = list(assessment.get("candidate_event_ids") or [])
            exact_ids = list(assessment.get("matched_event_ids") or [])
            binding_conflict = assessment.get("event_id_conflict") is True
            for row in direct_rows:
                for event_id in row.get("candidate_event_ids") or []:
                    if event_id not in candidate_ids:
                        candidate_ids.append(event_id)
                for event_id in row.get("matched_event_ids") or []:
                    if event_id not in exact_ids:
                        exact_ids.append(event_id)
                binding_conflict = binding_conflict or row.get("event_id_conflict") is True
            possible_ids = set(candidate_ids)
            assigned = (assigned_target_by_candidate or {}).get(
                str(assessment.get("candidate_id") or ""), ""
            )
            if assigned:
                assessment["candidate_event_ids"] = candidate_ids
                assessment["matched_event_ids"] = exact_ids
                assessment["event_id_conflict"] = binding_conflict
                _p130_normalize_typed_event_binding(
                    assessment, required_event_ids=required_event_ids,
                    assigned_target=assigned,
                )
                continue
            if (
                not binding_conflict
                and len(possible_ids) <= 1
                and len(set(exact_ids)) <= 1
                and (not possible_ids or set(exact_ids).issubset(possible_ids))
            ):
                exact = next(iter(set(exact_ids)), "")
                assessment["candidate_event_ids"] = candidate_ids
                assessment["target_event_id"] = exact
                assessment["event_index"] = exact
                assessment["target_event_ids"] = [exact] if exact else []
                assessment["matched_event_ids"] = [exact] if exact else []
                assessment["binding_confidence"] = "direct" if exact else "ambiguous"
                assessment["event_id_conflict"] = False
            else:
                assessment["candidate_event_ids"] = candidate_ids
                _p130_clear_exact_event_binding(assessment)


def _materialize_anchor_audit_observations(
    payload: dict[str, Any],
    anchor_assessments: list[dict[str, Any]],
) -> None:
    observations = payload.get("timestamp_observations")
    if not isinstance(observations, list):
        observations = []
        payload["timestamp_observations"] = observations
    rows_by_key = {
        (
            str(row.get("candidate_id") or ""),
            round(float(row.get("timestamp_s")), 1),
        ): row
        for row in observations
        if isinstance(row, dict) and row.get("timestamp_s") is not None
    }
    for assessment in anchor_assessments:
        key = (
            str(assessment.get("candidate_id") or ""),
            round(float(assessment.get("timestamp_s")), 1),
        )
        if key in rows_by_key:
            row = rows_by_key[key]
            row.update(assessment)
            row["description"] = assessment.get("observed_fact") or row.get(
                "description", ""
            )
            continue
        row = {
            **assessment,
            "description": assessment.get("observed_fact") or "",
        }
        observations.append(row)
        rows_by_key[key] = row


def _candidate_binding_summary(
    assessments: list[dict[str, Any]],
    *,
    event_binding_enabled: bool = False,
) -> dict[str, Any]:
    statuses = [str(item.get("target_match") or "unknown") for item in assessments]
    if "matched" in statuses:
        target_match = "matched"
    elif "partial" in statuses:
        target_match = "partial"
    elif "ambiguous" in statuses:
        target_match = "ambiguous"
    elif statuses and all(item == "not_visible" for item in statuses):
        target_match = "not_visible"
    elif statuses and all(item in {"mismatch", "not_visible"} for item in statuses):
        target_match = "mismatch"
    else:
        target_match = "unknown"

    matched_ids = [
        item["candidate_id"]
        for item in assessments
        if item.get("target_match") == "matched"
    ]
    mismatched_ids = [
        item["candidate_id"]
        for item in assessments
        if item.get("target_match") in {"mismatch", "not_visible"}
    ]
    conflict_ids = [
        item["candidate_id"]
        for item in assessments
        if item.get("option_set_conflict") is True
    ]
    direct_event_ids = [
        item["candidate_id"]
        for item in assessments
        if item.get("event_match") == "direct"
    ]
    context_event_ids = [
        item["candidate_id"]
        for item in assessments
        if item.get("event_match") in {"context_only", "different_event"}
    ]
    if matched_ids:
        search_status = "matched_candidate_found"
    elif assessments and len(mismatched_ids) == len(assessments):
        search_status = "all_candidates_mismatch_or_absent"
    else:
        search_status = "unresolved"
    summary = {
        "target_match": target_match,
        "matched_candidate_ids": matched_ids,
        "mismatched_candidate_ids": mismatched_ids,
        "option_conflict_candidate_ids": conflict_ids,
        "candidate_search_status": search_status,
    }
    if event_binding_enabled:
        summary.update(
            {
                "direct_event_candidate_ids": direct_event_ids,
                "context_event_candidate_ids": context_event_ids,
                "target_event_match": "direct" if direct_event_ids else "unresolved",
            }
        )
    return summary


def _apply_event_binding_option_scope(payload: dict[str, Any]) -> None:
    """Keep context from a different episode from becoming answer support."""
    assessments = [
        item
        for item in payload.get("candidate_assessments") or []
        if isinstance(item, dict)
    ]
    if not assessments:
        return

    direct = [item for item in assessments if item.get("event_match") == "direct"]
    context = [
        item
        for item in assessments
        if item.get("event_match") in {"context_only", "different_event"}
    ]
    context_supports: list[str] = []
    context_contradicts: list[str] = []
    for item in context:
        for letter in _option_letters(item.get("supports_options")):
            if letter not in context_supports:
                context_supports.append(letter)
        for letter in _option_letters(item.get("contradicts_options")):
            if letter not in context_contradicts:
                context_contradicts.append(letter)
    if context_supports:
        payload["context_supports_options"] = context_supports
    if context_contradicts:
        payload["context_contradicts_options"] = context_contradicts

    original_supports = _option_letters(payload.get("supports_options"))
    original_contradicts = _option_letters(payload.get("contradicts_options"))
    direct_supports: list[str] = []
    direct_contradicts: list[str] = []
    for item in direct:
        for letter in _option_letters(item.get("supports_options")):
            if letter not in direct_supports:
                direct_supports.append(letter)
        for letter in _option_letters(item.get("contradicts_options")):
            if letter not in direct_contradicts:
                direct_contradicts.append(letter)

    payload["supports_options"] = direct_supports
    payload["contradicts_options"] = direct_contradicts
    if direct:
        option_evidence = payload.get("option_evidence")
        if isinstance(option_evidence, dict):
            for letter, row in option_evidence.items():
                if not isinstance(row, dict):
                    continue
                status = str(row.get("status") or "").strip().lower()
                if status in {"support", "supported"} and letter not in direct_supports:
                    row["status"] = "unresolved"
                    row["reason"] = (
                        f"Option {letter} was not supported by a candidate that directly "
                        "shows the queried event instance."
                    )
                elif status in {"contradict", "contradicted"} and letter not in direct_contradicts:
                    row["status"] = "unresolved"
                    row["reason"] = (
                        f"Option {letter} was not contradicted by a candidate that directly "
                        "shows the queried event instance."
                    )
        return

    if original_supports:
        payload["context_supports_options"] = list(
            dict.fromkeys(context_supports + original_supports)
        )
    if original_contradicts:
        payload["context_contradicts_options"] = list(
            dict.fromkeys(context_contradicts + original_contradicts)
        )
    payload["target_event_match"] = "unresolved"
    payload["detail_sufficient"] = False
    payload["scope_coverage"] = "partial"
    reason = (
        "No candidate directly shows the exact event instance asked in the question; "
        "same-entity context and different episodes remain routing evidence only."
    )
    existing = str(payload.get("missing_detail") or "").strip()
    payload["missing_detail"] = f"{reason} {existing}".strip()
    payload["evidence_need"] = "temporal_coverage"
    option_evidence = payload.get("option_evidence")
    if isinstance(option_evidence, dict):
        for letter, row in option_evidence.items():
            if not isinstance(row, dict):
                continue
            if str(row.get("status") or "").strip().lower() in {
                "support",
                "supported",
                "contradict",
                "contradicted",
            }:
                row["status"] = "unresolved"
                row["reason"] = (
                    f"Only a context or different-event candidate addressed option {letter}; "
                    "the queried event instance was not directly observed."
                )


def _restore_resolved_candidate_set_decision(
    payload: dict[str, Any],
    *,
    aggregate_snapshot: dict[str, Any],
    candidate_coverage_complete: bool,
) -> bool:
    """Preserve a complete set-level decision beside candidate-level bindings."""
    payload["candidate_set_aggregate_decision_reported"] = bool(aggregate_snapshot)
    if not aggregate_snapshot:
        payload["candidate_set_aggregate_decision_reason"] = "not_reported"
        return False
    if aggregate_snapshot.get("decision_sufficient") is not True:
        payload["candidate_set_aggregate_decision_reason"] = "not_decision_sufficient"
        return False
    if not candidate_coverage_complete or payload.get("candidate_binding_complete") is not True:
        payload["candidate_set_aggregate_decision_reason"] = "incomplete_candidate_coverage"
        return False
    assessments = [
        item
        for item in payload.get("candidate_assessments") or []
        if isinstance(item, dict)
    ]
    if not assessments:
        payload["candidate_set_aggregate_decision_reason"] = "missing_candidate_assessments"
        return False
    resolved_event_matches = {"direct", "different_event", "not_visible"}
    event_matches = {
        str(item.get("event_match") or "unknown").strip().lower()
        for item in assessments
    }
    if "direct" not in event_matches or not event_matches.issubset(resolved_event_matches):
        payload["candidate_set_aggregate_decision_reason"] = "ambiguous_candidate_binding"
        return False
    if payload.get("option_set_conflict") is True or any(
        item.get("option_set_conflict") is True for item in assessments
    ):
        payload["candidate_set_aggregate_decision_reason"] = "candidate_conflict"
        return False
    supports = _option_letters(aggregate_snapshot.get("supports_options"))
    contradicts = _option_letters(aggregate_snapshot.get("contradicts_options"))
    if len(supports) != 1 or supports[0] in contradicts:
        payload["candidate_set_aggregate_decision_reason"] = "non_unique_aggregate_support"
        return False

    payload["candidate_set_aggregate_supports_options"] = supports
    payload["candidate_set_aggregate_contradicts_options"] = contradicts
    payload["candidate_set_aggregate_option_evidence"] = deepcopy(
        aggregate_snapshot.get("option_evidence") or {}
    )
    payload["supports_options"] = supports
    payload["contradicts_options"] = contradicts
    payload["option_evidence"] = deepcopy(
        aggregate_snapshot.get("option_evidence") or {}
    )
    payload["candidate_set_aggregate_decision_preserved"] = True
    payload["candidate_set_aggregate_decision_reason"] = "resolved_candidate_set"
    return True


def _missing_detail_is_empty(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    return bool(
        text in {"none", "n/a", "no", "nothing"}
        or text.startswith("none;")
        or "no remaining" in text
        or "evidence is clear" in text
    )


def _normalize_decision_completeness(payload: dict[str, Any], option_letters=None) -> bool:
    """Repair an internally inconsistent verifier sufficiency flag."""
    if payload.get("detail_sufficient") is True:
        return False
    if str(payload.get("target_match") or "").strip().lower() != "matched":
        return False
    if str(payload.get("scope_coverage") or "").strip().lower() != "sufficient":
        return False
    if payload.get("target_event_match") not in {None, "", "direct"}:
        return False
    supports = _option_letters(payload.get("supports_options"))
    contradicts = set(_option_letters(payload.get("contradicts_options")))
    if len(supports) != 1:
        return False
    if set(option_letters or "ABCD") - {supports[0]} - contradicts:
        return False
    if not _missing_detail_is_empty(payload.get("missing_detail")):
        return False
    if str(payload.get("evidence_need") or "").strip().lower() in {
        "conflict",
        "semantic_binding",
        "spatial_detail",
    }:
        return False
    payload["detail_sufficient"] = True
    payload["missing_detail"] = ""
    payload["evidence_need"] = "sufficient"
    payload["evidence_needs"] = ["sufficient"]
    payload["decision_consistency_repaired"] = True
    return True


def _apply_reported_decision_completeness(
    payload: dict[str, Any],
    *,
    candidate_count: int,
    option_letters=None,
) -> bool:
    """Separate answer discrimination from optional visual-detail refinement.

    This applies only to a real multi-window comparison. It never infers a
    decision from a single local crop or from an unbound/conflicted response.
    """
    if candidate_count < 2 or payload.get("decision_sufficient") is not True:
        return False
    if payload.get("candidate_coverage_complete") is not True:
        return False
    if payload.get("candidate_binding_complete") is not True:
        return False
    if str(payload.get("target_match") or "").strip().lower() != "matched":
        return False
    if str(payload.get("scope_coverage") or "").strip().lower() != "sufficient":
        return False
    if payload.get("target_event_match") not in {None, "", "direct"}:
        return False
    if payload.get("option_set_conflict") is True or payload.get("observer_conflict_options"):
        return False
    supports = _option_letters(payload.get("supports_options"))
    contradicts = set(_option_letters(payload.get("contradicts_options")))
    if len(supports) != 1 or set(option_letters or "ABCD") - {supports[0]} - contradicts:
        return False

    payload["visual_detail_sufficient"] = payload.get("detail_sufficient") is True
    payload["residual_visual_detail"] = str(payload.get("missing_detail") or "").strip()
    payload["decision_evidence_need"] = "sufficient"
    payload["decision_sufficiency_validated"] = True
    return True


_EVIDENCE_NEEDS = {
    "sufficient",
    "spatial_detail",
    "temporal_coverage",
    "semantic_binding",
    "conflict",
}


def _demote_partial_global_option_claims(payload: dict[str, Any]) -> None:
    """Retain local option audit without presenting it as video-wide exclusion."""
    scope = str(payload.get("question_scope") or "").strip().lower()
    coverage = str(payload.get("scope_coverage") or "").strip().lower()
    if scope != "global_video" or coverage not in {"partial", "insufficient"}:
        return

    local_contradicts = _option_letters(payload.get("contradicts_options"))
    if local_contradicts:
        payload["local_contradicts_options"] = local_contradicts
        payload["contradicts_options"] = []

    if payload.get("detail_sufficient") is not True:
        local_supports = _option_letters(payload.get("supports_options"))
        if local_supports:
            payload["local_supports_options"] = local_supports
            payload["supports_options"] = []

    option_evidence = payload.get("option_evidence")
    if isinstance(option_evidence, dict):
        for item in option_evidence.values():
            if not isinstance(item, dict):
                continue
            if str(item.get("status") or "").strip().lower() != "contradicted":
                continue
            reason = str(item.get("reason") or "").strip()
            item["status"] = "unresolved"
            item["reason"] = (
                "Not observed in the inspected local window; this does not exclude "
                "the option elsewhere in the video."
                + (f" Local note: {reason}" if reason else "")
            )


def _normalize_evidence_need(payload: dict[str, Any]) -> str:
    """Normalize the remaining evidence need without selecting an answer."""
    supports = set(_option_letters(payload.get("supports_options")))
    contradicts = set(_option_letters(payload.get("contradicts_options")))
    if payload.get("option_set_conflict") is True or supports.intersection(contradicts):
        return "conflict"

    raw = str(payload.get("evidence_need") or "").strip().lower()
    detail_sufficient = payload.get("detail_sufficient") is True
    target_match = str(payload.get("target_match") or "unknown").strip().lower()
    option_evidence = payload.get("option_evidence")
    option_statuses = [
        str(item.get("status") or "").strip().lower()
        for item in (option_evidence or {}).values()
        if isinstance(item, dict)
    ] if isinstance(option_evidence, dict) else []
    no_supported_option = (
        not supports
        and bool(option_statuses)
        and all(
            status in {
                "unresolved",
                "unknown",
                "insufficient",
                "not_visible",
                "not observed",
            }
            for status in option_statuses
        )
    )
    if (
        not detail_sufficient
        and no_supported_option
        and target_match in {"matched", "partial", "ambiguous"}
    ):
        return "semantic_binding"

    scope = str(payload.get("question_scope") or "").strip().lower()
    coverage = str(payload.get("scope_coverage") or "").strip().lower()
    if scope == "global_video" and coverage in {"partial", "insufficient"}:
        return "temporal_coverage"

    if not detail_sufficient:
        if raw in _EVIDENCE_NEEDS - {"sufficient"}:
            return raw
        if target_match in {"unknown", "ambiguous", "not_visible", "mismatch"}:
            return "semantic_binding"
        if isinstance(payload.get("detail_target"), dict) and str(
            payload.get("detail_query") or ""
        ).strip():
            return "spatial_detail"
        # Once target binding is present, an otherwise unclassified missing
        # observation requires more temporal evidence, not a blind crop.
        return "temporal_coverage"
    if target_match in {"unknown", "ambiguous", "not_visible", "mismatch"}:
        return "semantic_binding"
    if raw == "conflict":
        return "conflict"
    return "sufficient"


def _normalize_evidence_needs(payload: dict[str, Any]) -> list[str]:
    """Return composable needs while keeping one primary planner label."""
    needs = [_normalize_evidence_need(payload)]
    target_match = str(payload.get("target_match") or "unknown").strip().lower()
    has_detail_target = isinstance(payload.get("detail_target"), dict) and bool(
        str(payload.get("detail_query") or "").strip()
    )
    if (
        payload.get("detail_sufficient") is not True
        and has_detail_target
        and target_match in {"matched", "partial", "ambiguous"}
        and "spatial_detail" not in needs
    ):
        needs.append("spatial_detail")
    return needs


def _has_spatial_evidence_need(payload: dict[str, Any]) -> bool:
    reported = payload.get("evidence_needs")
    if isinstance(reported, list) and "spatial_detail" in {
        str(item).strip().lower() for item in reported
    }:
        return True
    return _normalize_evidence_need(payload) == "spatial_detail"


def _normalize_packet_detail_request(
    payload: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    allow_evidence_need: bool = False,
    allow_ambiguous_mandatory: bool = False,
) -> dict[str, Any] | None:
    """Accept only explicit, candidate-bound local-detail escalation requests."""
    assessments_by_id = {
        str(item.get("candidate_id") or "").strip(): item
        for item in payload.get("candidate_assessments") or []
        if isinstance(item, dict) and str(item.get("candidate_id") or "").strip()
    }
    forced_candidate = None
    if allow_ambiguous_mandatory:
        for candidate in candidates:
            if candidate.get("mandatory_positive") is not True:
                continue
            assessment = assessments_by_id.get(candidate["candidate_id"]) or {}
            target_match = str(
                assessment.get("target_match") or "unknown"
            ).strip().lower()
            event_match = str(
                assessment.get("event_match")
                or assessment.get("target_event_match")
                or "unknown"
            ).strip().lower()
            if target_match in {"partial", "ambiguous"} and event_match in {
                "unknown",
                "partial",
                "ambiguous",
                "context_only",
            }:
                forced_candidate = candidate
                break

    if payload.get("detail_sufficient") is not False and forced_candidate is None:
        return None
    raw = payload.get("detail_escalation")
    if forced_candidate is not None:
        assessment = assessments_by_id[forced_candidate["candidate_id"]]
        raw = {
            "needed": True,
            "candidate_id": forced_candidate["candidate_id"],
            "timestamp_s": assessment.get("best_timestamp_s"),
            "query": (
                "Inspect this distant candidate at the highest available resolution. "
                "Report whether the airborne object directly shows paragliding "
                "(canopy, suspended pilot, harness, or lines) or a non-paragliding "
                "lookalike/artifact."
            ),
            "reason": "mandatory local candidate remained ambiguous after packet verification",
        }
    explicit_request = isinstance(raw, dict) and raw.get("needed") is True
    if (
        forced_candidate is None
        and allow_evidence_need
        and payload.get("evidence_need")
        and not _has_spatial_evidence_need(payload)
    ):
        return None
    if not explicit_request:
        if not allow_evidence_need or not _has_spatial_evidence_need(payload):
            return None
        raw = payload.get("detail_target")
        if not isinstance(raw, dict):
            raw = {}

    candidate_id = str(raw.get("candidate_id") or "").strip()
    if not candidate_id and allow_evidence_need:
        eligible = [
            item
            for item in payload.get("candidate_assessments") or []
            if isinstance(item, dict)
            and str(item.get("target_match") or "").strip().lower()
            in {"matched", "partial", "ambiguous"}
        ]
        rank = {"matched": 0, "partial": 1, "ambiguous": 2}
        eligible.sort(
            key=lambda item: rank.get(
                str(item.get("target_match") or "").strip().lower(), 3
            )
        )
        if eligible:
            candidate_id = str(eligible[0].get("candidate_id") or "").strip()
    candidate = next(
        (item for item in candidates if item["candidate_id"] == candidate_id),
        None,
    )
    if candidate is None:
        return None
    assessment = next(
        (
            item
            for item in payload.get("candidate_assessments") or []
            if isinstance(item, dict)
            and str(item.get("candidate_id") or "").strip() == candidate_id
        ),
        None,
    )
    if not isinstance(assessment, dict) or str(
        assessment.get("target_match") or "unknown"
    ).strip().lower() not in {"matched", "partial", "ambiguous"}:
        return None

    timestamp = raw.get("timestamp_s")
    if timestamp is None:
        timestamp = assessment.get("best_timestamp_s")
    if timestamp is None and candidate.get("timestamp_anchors"):
        timestamp = candidate["timestamp_anchors"][0]
    start, end = candidate["t_range"]
    try:
        timestamp = float(timestamp)
    except (TypeError, ValueError):
        timestamp = (float(start) + float(end)) / 2.0
    timestamp = max(float(start), min(float(end), timestamp))

    query = str(
        raw.get("query")
        or payload.get("detail_query")
        or ""
    ).strip()
    if not query and allow_evidence_need:
        missing = str(payload.get("missing_detail") or "the unresolved local detail").strip()
        query = (
            "Inspect only the localized question target and neutrally report the "
            f"small visual detail still unresolved: {missing}"
        )
    if not query:
        return None
    return {
        "candidate_id": candidate_id,
        "timestamp_s": round(timestamp, 3),
        "query": query[:600],
        "reason": str(raw.get("reason") or payload.get("missing_detail") or "").strip()[:600],
    }


def _merge_packet_detail_escalation(
    *,
    candidates: list[dict[str, Any]],
    initial_payload: dict[str, Any],
    detail_payload: dict[str, Any],
    request: dict[str, Any],
) -> str:
    """Replace one weak candidate assessment with its crop-based verification."""
    merged = deepcopy(initial_payload)
    candidate_id = request["candidate_id"]

    assessments_by_id = {
        str(item.get("candidate_id") or ""): dict(item)
        for item in initial_payload.get("candidate_assessments") or []
        if isinstance(item, dict)
    }
    detail_assessment = next(
        (
            dict(item)
            for item in detail_payload.get("candidate_assessments") or []
            if isinstance(item, dict)
            and str(item.get("candidate_id") or "") == candidate_id
        ),
        None,
    )
    if detail_assessment is not None:
        assessments_by_id[candidate_id] = detail_assessment
    assessments = [
        assessments_by_id[item["candidate_id"]]
        for item in candidates
        if item["candidate_id"] in assessments_by_id
    ]
    if assessments:
        merged["candidate_assessments"] = assessments
        merged.update(_candidate_binding_summary(assessments))
        merged["candidate_binding_complete"] = len(assessments) == len(candidates)

    observations: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for payload in (initial_payload, detail_payload):
        for row in payload.get("timestamp_observations") or []:
            if not isinstance(row, dict):
                continue
            key = (
                str(row.get("candidate_id") or ""),
                row.get("timestamp_s"),
                str(row.get("description") or "").strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            observations.append(dict(row))
    merged["timestamp_observations"] = observations

    option_evidence = deepcopy(initial_payload.get("option_evidence") or {})
    conflict_options: list[str] = []
    for letter, detail_row in (detail_payload.get("option_evidence") or {}).items():
        initial_row = option_evidence.get(letter)
        if isinstance(initial_row, dict) and isinstance(detail_row, dict):
            old_status = str(initial_row.get("status") or "").strip().lower()
            new_status = str(detail_row.get("status") or "").strip().lower()
            if {old_status, new_status} == {"support", "contradict"}:
                merged_row = dict(detail_row)
                merged_row["observer_conflict"] = True
                merged_row["reason"] = " | ".join(
                    item
                    for item in (
                        str(initial_row.get("reason") or "").strip(),
                        str(detail_row.get("reason") or "").strip(),
                    )
                    if item
                )[:700]
                option_evidence[letter] = merged_row
                conflict_options.append(str(letter))
                continue
        option_evidence[letter] = detail_row
    merged["option_evidence"] = option_evidence
    supports: list[str] = []
    contradicts: list[str] = []
    for payload in (initial_payload, detail_payload):
        for letter in _option_letters(payload.get("supports_options")):
            if letter not in supports:
                supports.append(letter)
        for letter in _option_letters(payload.get("contradicts_options")):
            if letter not in contradicts:
                contradicts.append(letter)
    for assessment in assessments:
        for letter in _option_letters(assessment.get("supports_options")):
            if letter not in supports:
                supports.append(letter)
        for letter in _option_letters(assessment.get("contradicts_options")):
            if letter not in contradicts:
                contradicts.append(letter)
    for letter, row in option_evidence.items():
        if not isinstance(row, dict) or not re.fullmatch(r"[A-Z]", str(letter)):
            continue
        status = str(row.get("status") or "").strip().lower()
        if status == "support" and letter not in supports:
            supports.append(letter)
        elif status == "contradict" and letter not in contradicts:
            contradicts.append(letter)
    merged["supports_options"] = supports
    merged["contradicts_options"] = contradicts
    for letter in sorted(set(supports).intersection(contradicts)):
        if letter not in conflict_options:
            conflict_options.append(letter)
    if conflict_options:
        merged["observer_conflict_options"] = conflict_options
        merged["option_set_conflict"] = True
        merged["evidence_need"] = "conflict"
        merged["evidence_needs"] = ["conflict"]

    initial_summary = str(initial_payload.get("overall_summary") or "").strip()
    detail_summary = str(detail_payload.get("overall_summary") or "").strip()
    merged["overall_summary"] = " | ".join(
        item for item in (initial_summary, detail_summary) if item
    )[:1400]
    merged["observed_fact"] = (
        detail_payload.get("observed_fact")
        or detail_summary
        or initial_payload.get("observed_fact")
        or initial_summary
    )
    merged["detail_sufficient"] = (
        detail_payload.get("detail_sufficient") is True and not conflict_options
    )
    merged["missing_detail"] = str(detail_payload.get("missing_detail") or "").strip()
    merged["num_frames"] = int(initial_payload.get("num_frames") or 0) + int(
        detail_payload.get("num_frames") or 0
    )
    merged["packet_detail_escalation_used"] = True
    merged["packet_detail_escalation_request"] = request
    merged["packet_detail_grounding_audit"] = detail_payload.get("grounding_audit") or {}
    merged["packet_detail_visual_packet_audit"] = (
        detail_payload.get("visual_packet_audit") or {}
    )
    merged["packet_detail_initial_summary"] = initial_summary
    merged["packet_detail_updated_summary"] = detail_summary
    merged["detail_escalation"] = {"needed": False}
    if not conflict_options:
        merged["evidence_need"] = _normalize_evidence_need(merged)
        merged["evidence_needs"] = _normalize_evidence_needs(merged)
    return format_v10_observation(merged)


def _run_packet_detail_grounding_escalation(
    *,
    config: dict,
    parameters: dict,
    candidates: list[dict[str, Any]],
    initial_output: str,
    initial_payload: dict[str, Any],
) -> str:
    request = _normalize_packet_detail_request(
        initial_payload,
        candidates=candidates,
        allow_evidence_need=bool(
            config.get("packet_adaptive_evidence_need_enabled", False)
        ),
        allow_ambiguous_mandatory=bool(
            config.get(
                "p127_ambiguous_mandatory_detail_recovery_enabled", False
            )
        ),
    )
    if request is None:
        return initial_output
    candidate = next(
        item for item in candidates if item["candidate_id"] == request["candidate_id"]
    )
    detail_candidate = dict(candidate)
    detail_candidate["timestamp_anchors"] = [request["timestamp_s"]]

    detail_config = deepcopy(config)
    detail_config.update(
        {
            "packet_detail_grounding_escalation_enabled": False,
            "grounded_frame_verify_enabled": True,
            "grounded_high_recall_proposal_enabled": True,
            "grounded_verify_packet_enabled": True,
            "grounded_frame_verify_max_anchors": 1,
            "grounded_verify_packet_anchors_per_candidate": 1,
            "localize_inline_verify_max_windows": 1,
            "localize_inline_verify_max_frames": max(
                2,
                int(config.get("packet_detail_grounding_escalation_frames") or 6),
            ),
            "multiwindow_verify_recovery_enabled": False,
        }
    )
    detail_parameters = dict(parameters)
    detail_parameters.update(
        {
            "candidate_windows": [detail_candidate],
            "windows": [detail_candidate["t_range"]],
            "start_time": detail_candidate["t_range"][0],
            "end_time": detail_candidate["t_range"][1],
            "query": request["query"],
            "packet_detail_grounding_escalation_run": True,
            "grounding_escalation_require_crop": True,
        }
    )
    try:
        detail_output = _execute_multiwindow_frame_verify_once(
            detail_config,
            detail_parameters,
        )
    except Exception as exc:
        detail_output = format_v10_observation(
            {
                "tool": "frame_verify",
                "detail_escalation_skipped": True,
                "detail_escalation_skip_reason": f"{type(exc).__name__}: {exc}"[:500],
            }
        )
    detail_payload = extract_v10_payload(detail_output) or {}
    if detail_payload.get("detail_escalation_skipped") is True:
        preserved = deepcopy(initial_payload)
        preserved["packet_detail_escalation_used"] = False
        preserved["packet_detail_escalation_attempted"] = True
        preserved["packet_detail_escalation_request"] = request
        preserved["packet_detail_escalation_skip_reason"] = str(
            detail_payload.get("detail_escalation_skip_reason") or "no_valid_local_crop"
        )[:500]
        preserved["packet_detail_grounding_audit"] = (
            detail_payload.get("grounding_audit") or {}
        )
        return format_v10_observation(preserved, fallback_text=initial_output)
    if not detail_payload or detail_payload.get("parse_ok") is not True:
        preserved = deepcopy(initial_payload)
        preserved["packet_detail_escalation_used"] = False
        preserved["packet_detail_escalation_attempted"] = True
        preserved["packet_detail_escalation_request"] = request
        preserved["packet_detail_escalation_skip_reason"] = "detail_verifier_failed"
        return format_v10_observation(preserved, fallback_text=initial_output)
    return _merge_packet_detail_escalation(
        candidates=candidates,
        initial_payload=initial_payload,
        detail_payload=detail_payload,
        request=request,
    )


def _run_direct_packet_detail_grounding_escalation(
    *,
    config: dict,
    parameters: dict,
    initial_output: str,
    initial_payload: dict[str, Any],
) -> str:
    """Give a direct Packet observation the same one-anchor detail path."""
    span = initial_payload.get("t_range") or [
        parameters.get("start_time"),
        parameters.get("end_time"),
    ]
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return initial_output
    try:
        start, end = float(span[0]), float(span[1])
    except (TypeError, ValueError):
        return initial_output
    if end <= start:
        return initial_output

    timestamp = None
    focus_window = initial_payload.get("suggest_focus_window")
    if isinstance(focus_window, (list, tuple)) and len(focus_window) == 2:
        try:
            timestamp = (float(focus_window[0]) + float(focus_window[1])) / 2.0
        except (TypeError, ValueError):
            timestamp = None
    if timestamp is None:
        rows = [
            item
            for item in initial_payload.get("timestamp_observations") or []
            if isinstance(item, dict) and item.get("timestamp_s") is not None
        ]
        if rows:
            middle = rows[len(rows) // 2]
            try:
                timestamp = float(middle.get("timestamp_s"))
            except (TypeError, ValueError):
                timestamp = None
    if timestamp is None:
        timestamp = (start + end) / 2.0
    timestamp = max(start, min(end, float(timestamp)))

    candidate_id = "DIRECT001"
    candidate = {
        "candidate_id": candidate_id,
        "rank": 1,
        "t_range": [round(start, 3), round(end, 3)],
        "summary": str(
            initial_payload.get("observed_fact")
            or initial_payload.get("overall_summary")
            or ""
        )[:480],
        "timestamp_anchors": [round(timestamp, 3)],
    }
    normalized_payload = deepcopy(initial_payload)
    normalized_payload["evidence_need"] = _normalize_evidence_need(normalized_payload)
    normalized_payload["evidence_needs"] = _normalize_evidence_needs(
        normalized_payload
    )
    normalized_payload["candidate_assessments"] = [
        {
            "candidate_id": candidate_id,
            "target_match": str(
                normalized_payload.get("target_match") or "unknown"
            ).strip().lower(),
            "observed_fact": str(normalized_payload.get("observed_fact") or ""),
            "target_binding_reason": str(
                normalized_payload.get("target_binding_reason") or ""
            ),
            "best_timestamp_s": round(timestamp, 3),
            "supports_options": _option_letters(
                normalized_payload.get("supports_options")
            ),
            "contradicts_options": _option_letters(
                normalized_payload.get("contradicts_options")
            ),
            "option_set_conflict": normalized_payload.get("option_set_conflict") is True,
            "assessment_present": True,
        }
    ]
    detail_target = normalized_payload.get("detail_target")
    if isinstance(detail_target, dict):
        detail_target = dict(detail_target)
        detail_target.setdefault("candidate_id", candidate_id)
        detail_target.setdefault("timestamp_s", round(timestamp, 3))
        normalized_payload["detail_target"] = detail_target

    detail_parameters = dict(parameters)
    detail_parameters.update(
        {
            "candidate_windows": [candidate],
            "windows": [candidate["t_range"]],
            "start_time": start,
            "end_time": end,
        }
    )
    output = _run_packet_detail_grounding_escalation(
        config=config,
        parameters=detail_parameters,
        candidates=[candidate],
        initial_output=initial_output,
        initial_payload=normalized_payload,
    )
    payload = extract_v10_payload(output) or {}
    if not payload:
        return output
    if not (
        payload.get("packet_detail_escalation_used") is True
        or payload.get("packet_detail_escalation_attempted") is True
    ):
        return output
    payload["direct_packet_detail_escalation"] = True
    payload["scene_id"] = initial_payload.get("scene_id") or payload.get("scene_id")
    payload["window_id"] = initial_payload.get("window_id") or payload.get("window_id")
    payload["t_range"] = [round(start, 3), round(end, 3)]
    return format_v10_observation(payload, fallback_text=output)


def _bind_candidate_observations(
    payload: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    assessments: list[dict[str, Any]],
    allowed_by_candidate: dict[str, list[float]],
    synthesize_assessment_anchor: bool = True,
) -> None:
    assessment_by_id = {item["candidate_id"]: item for item in assessments}
    requested = set(assessment_by_id)
    observations = payload.get("timestamp_observations")
    if not isinstance(observations, list):
        observations = []
        payload["timestamp_observations"] = observations
    bound_ids: set[str] = set()
    for row in observations:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id not in requested:
            candidate_id = _candidate_id_for_timestamp(row.get("timestamp_s"), candidates)
        if candidate_id not in requested:
            continue
        row["candidate_id"] = candidate_id
        bound_ids.add(candidate_id)
        allowed = allowed_by_candidate.get(candidate_id) or []
        try:
            timestamp = float(row.get("timestamp_s", row.get("timestamp")))
        except (TypeError, ValueError):
            timestamp = allowed[0] if allowed else None
        if timestamp is not None and allowed:
            timestamp = min(allowed, key=lambda item: abs(item - timestamp))
        if timestamp is not None:
            row["timestamp_s"] = round(float(timestamp), 1)
        assessment = assessment_by_id[candidate_id]
        row["target_match"] = assessment["target_match"]
        row["event_match"] = assessment.get("event_match") or "unknown"
        row["target_binding_reason"] = assessment["target_binding_reason"]
        row["option_set_conflict"] = assessment["option_set_conflict"]
        row["supports_options"] = list(assessment["supports_options"])
        row["contradicts_options"] = list(assessment["contradicts_options"])
        for key in (
            "target_event_id",
            "event_index",
            "candidate_event_ids",
            "target_event_ids",
            "matched_event_ids",
            "binding_confidence",
            "event_id_conflict",
        ):
            if key in assessment:
                row[key] = deepcopy(assessment.get(key))
        if (
            synthesize_assessment_anchor
            and assessment.get("best_timestamp_s") is None
            and timestamp is not None
        ):
            assessment["best_timestamp_s"] = round(float(timestamp), 1)

    # Compact verifier responses may omit a duplicate timestamp_observations
    # array. Materialize one observation per assessment deterministically so
    # downstream coverage and memory semantics remain unchanged.
    for candidate_id, assessment in assessment_by_id.items():
        if candidate_id in bound_ids:
            continue
        if assessment.get("assessment_present") is False:
            continue
        allowed = allowed_by_candidate.get(candidate_id) or []
        timestamp = assessment.get("best_timestamp_s")
        if timestamp is None and allowed and synthesize_assessment_anchor:
            timestamp = allowed[len(allowed) // 2]
        if timestamp is not None and allowed:
            timestamp = min(allowed, key=lambda item: abs(item - float(timestamp)))
        if timestamp is not None and synthesize_assessment_anchor:
            assessment["best_timestamp_s"] = round(float(timestamp), 1)
        observation = {
            "candidate_id": candidate_id,
            "timestamp_s": (
                round(float(timestamp), 1) if timestamp is not None else None
            ),
            "description": assessment.get("observed_fact") or "",
            "target_match": assessment.get("target_match") or "unknown",
            "event_match": assessment.get("event_match") or "unknown",
            "target_binding_reason": assessment.get("target_binding_reason") or "",
            "option_set_conflict": assessment.get("option_set_conflict") is True,
            "supports_options": list(assessment.get("supports_options") or []),
            "contradicts_options": list(
                assessment.get("contradicts_options") or []
            ),
        }
        for key in (
            "target_event_id",
            "event_index",
            "candidate_event_ids",
            "target_event_ids",
            "matched_event_ids",
            "binding_confidence",
            "event_id_conflict",
        ):
            if key in assessment:
                observation[key] = deepcopy(assessment.get(key))
        observations.append(observation)


def _frame_data_url(frame, *, max_side: int = 0, quality: int = 88, label: str = "") -> str:
    image = Image.fromarray(frame).convert("RGB")
    header_height = 32 if label else 0
    if max_side > 0 and max(image.width, image.height+header_height) > max_side:
        image.thumbnail((max_side, max_side-header_height), Image.Resampling.LANCZOS)
    if label:
        # Bind the source ID to its pixels, without obscuring the video frame.
        header = Image.new("RGB", (max(image.width, 160), image.height+header_height), "black")
        header.paste(image, (0, header_height))
        ImageDraw.Draw(header).text((8, 3), label, font=ImageFont.load_default(size=22), fill="white")
        image = header
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("utf-8")


def _image_from_data_url(value: str) -> Image.Image:
    _, encoded = str(value).split(",", 1)
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        return image.convert("RGB").copy()


def _resize_pil_long_side(image: Image.Image, max_side: int) -> Image.Image:
    resized = image.convert("RGB").copy()
    if max_side > 0 and max(resized.size) > max_side:
        resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


def _candidate_contact_sheet_data_url(
    rows: list[tuple[float, Any]],
    *,
    candidate_id: str,
) -> str:
    """Pack one candidate's temporal strip into one clearly labelled image."""
    if not rows:
        raise ValueError("candidate contact sheet requires at least one frame")
    columns = min(3, len(rows))
    sheet_rows = (len(rows) + columns - 1) // columns
    cell_width = 384
    cell_height = 216
    sheet = Image.new(
        "RGB",
        (columns * cell_width, sheet_rows * cell_height),
        color=(0, 0, 0),
    )
    for index, (timestamp, frame) in enumerate(rows):
        image = Image.fromarray(frame).convert("RGB")
        image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_width + (cell_width - image.width) // 2
        y = (index // columns) * cell_height + (cell_height - image.height) // 2
        sheet.paste(image, (x, y))
        draw = ImageDraw.Draw(sheet)
        label = f"{candidate_id} | {timestamp:.1f}s"
        label_x = (index % columns) * cell_width + 8
        label_y = (index // columns) * cell_height + 8
        box = draw.textbbox((label_x, label_y), label)
        draw.rectangle(
            (box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3),
            fill=(0, 0, 0),
        )
        draw.text((label_x, label_y), label, fill=(255, 255, 255))
    output = BytesIO()
    sheet.save(output, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("utf-8")


def _candidate_detail_sheet_data_url(
    rows: list[tuple[float, Image.Image, str]],
    *,
    candidate_id: str,
    tile_max_side: int,
) -> str:
    """Keep multiple grounded anchors in one high-detail API image."""
    if not rows:
        raise ValueError("candidate detail sheet requires at least one image")
    tile_side = max(128, int(tile_max_side))
    label_height = 34
    prepared = [
        (timestamp, _resize_pil_long_side(image, tile_side), label)
        for timestamp, image, label in rows
    ]
    sheet = Image.new(
        "RGB",
        (tile_side * len(prepared), tile_side + label_height),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (timestamp, image, label) in enumerate(prepared):
        left = index * tile_side
        x = left + (tile_side - image.width) // 2
        y = label_height + (tile_side - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text(
            (left + 8, 9),
            f"{candidate_id} | {timestamp:.1f}s | {label}",
            fill=(255, 255, 255),
        )
    output = BytesIO()
    sheet.save(output, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("utf-8")


def _generic_candidate_contract(allowed_text: str, packet_anchor_timestamps_by_candidate: dict) -> str:
    """One candidate verdict; anchor captions are source observations only."""
    return (
        f"Allowed timestamps: [{allowed_text}]. Use each candidate's own timestamps.\n"
        f"Describe these detail anchors: {packet_anchor_timestamps_by_candidate}.\n"
        "Describe visible facts at anchors without selecting options or assigning event labels. "
        "Then assess each candidate once using all its frames. The original question defines "
        "the person, event and relation; answer-choice phrases are alternative explanations, "
        "not target definitions or extra prerequisites. An action resembling a choice does "
        "not establish that the depicted people/event belong to the question. Retain useful "
        "facts even when that binding is partial or uncertain. Do not require an unasked action. "
        "Distinguish visible absence from an unreadable or occluded detail. Absence applies only "
        "to inspected pixels and cannot establish whole-video absence. State this limitation "
        "when comparing an option that requires a global conclusion. If anchor and context "
        "observations disagree, explain the unresolved relation in the candidate judgment. "
        "event_span covers only the visible local event; best_timestamp_s must reference a "
        "shown frame bearing on that judgment. Keep unclear relations uncertain. "
        "Return compact JSON only:\n"
        '{"candidate_assessments":[{"candidate_id":"LQ001",'
        '"target_match":"matched|partial|ambiguous|mismatch|not_visible",'
        '"event_match":"direct|context_only|different_event|ambiguous|not_visible",'
        '"observed_fact":"what the frames show",'
        '"target_binding_reason":"relation to original target and unresolved details",'
        '"best_timestamp_s":1.0,"event_span":[0.5,1.5],'
        '"supports_options":[],"contradicts_options":[],"option_set_conflict":false}],'
        '"anchor_assessments":[{"candidate_id":"LQ001","timestamp_s":1.0,'
        '"observed_fact":"visible fact or unreadable detail only"}],'
        '"detail_sufficient":false,"missing_detail":"unresolved discriminator and scope",'
        '"scope_coverage":"sufficient|partial|insufficient",'
        '"scope_coverage_reason":"what the shown windows establish and leave unresolved",'
        '"evidence_need":"sufficient|spatial_detail|temporal_coverage|semantic_binding|conflict",'
        '"detail_target":{"candidate_id":"","timestamp_s":null},'
        '"detail_query":"neutral question about the unresolved visual detail"}'
    )


def _execute_multiwindow_frame_verify_once(config: dict, parameters: dict) -> str:
    """Verify several disjoint Qwen candidates in one remote visual request."""
    vr = parameters.get("vr")
    if vr is None:
        raise ValueError("multi-window frame_verify requires the active video reader")
    duration = float(parameters.get("duration") or round(len(vr) / vr.get_avg_fps(), 2))
    candidates = _normalize_candidate_windows(
        parameters.get("candidate_windows"),
        duration=duration,
        max_windows=_configured_max_windows(config, parameters),
        p131_enabled=bool(
            config.get("p131_minimal_global_repairs_enabled", False)
            and should_use_minimal_global_fsm(
                str(parameters.get("question") or "")
            )
        ),
        p131_mode=detect_global_mode(str(parameters.get("question") or "")),
        p132_enabled=bool(config.get("p132_weak_lead_visible_core_enabled", False)),
    )
    if not candidates:
        raise ValueError("multi-window frame_verify requires valid candidate_windows")

    frame_budgets = _allocate_candidate_frames(
        len(candidates),
        int(config.get("localize_inline_verify_max_frames") or 18),
    )
    fps = float(vr.get_avg_fps())
    total_video_frames = len(vr)
    sampled: list[tuple[str, float, Any]] = []
    sampled_frame_indices: list[int] = []
    candidate_anchor_indices: dict[int, list[str]] = {}
    for candidate, frame_budget in zip(candidates, frame_budgets):
        start, end = candidate["t_range"]
        timestamps = scene_aware_timestamps(
            video_path=str(parameters.get("video_path") or ""),
            duration_s=duration,
            start_time=start,
            end_time=end,
            num_frames=frame_budget,
            scene=None,
        )
        timestamp_anchors = candidate.get("timestamp_anchors") or []
        confidence_rank = {"high": 2, "medium": 1, "low": 0}
        evidence_anchors = candidate.get("localized_evidence_anchors") or []
        strongest_confidence = max(
            (
                confidence_rank.get(
                    str(item.get("confidence") or "low").strip().lower(),
                    0,
                )
                for item in evidence_anchors
                if isinstance(item, dict)
            ),
            default=-1,
        )
        priority_anchors = [
            float(item.get("timestamp_s"))
            for item in evidence_anchors
            if isinstance(item, dict)
            and confidence_rank.get(
                str(item.get("confidence") or "low").strip().lower(),
                0,
            )
            == strongest_confidence
            and item.get("timestamp_s") is not None
        ]
        if timestamp_anchors:
            timestamps = _merge_candidate_sampling_timestamps(
                list(timestamps),
                list(timestamp_anchors),
                start=start,
                end=end,
                budget=frame_budget,
                priority_anchors=priority_anchors,
            )
        indices = timestamps_to_frame_indices(
            timestamps=timestamps,
            fps=fps,
            total_frames=total_video_frames,
        )
        if len(indices) == 0:
            continue
        frames = vr.get_batch(indices).asnumpy()
        if timestamp_anchors:
            sampled_times = [float(frame_index) / fps for frame_index in indices.tolist()]
            for anchor in timestamp_anchors:
                nearest_position = min(
                    range(len(sampled_times)),
                    key=lambda position: abs(sampled_times[position] - float(anchor)),
                )
                anchor_index = int(indices[nearest_position])
                candidate_anchor_indices.setdefault(anchor_index, []).append(
                    candidate["candidate_id"]
                )
        elif len(indices) > 0:
            center_index = int(indices[len(indices) // 2])
            candidate_anchor_indices.setdefault(center_index, []).append(
                candidate["candidate_id"]
            )
        for frame_index, frame in zip(indices.tolist(), frames):
            sampled_frame_indices.append(int(frame_index))
            sampled.append(
                (
                    candidate["candidate_id"],
                    round(float(frame_index) / fps, 1),
                    frame,
                )
            )
    if not sampled:
        raise RuntimeError("multi-window frame_verify could not sample candidate frames")

    question = str(parameters.get("question") or "").strip()
    p130_context = _p130_global_context(config, parameters)
    p130_required_event_ids = (
        set(p130_context.get("required_event_ids") or [])
        if p130_context["active"] and p130_context["mode"] == "order"
        else None
    )
    query = str(parameters.get("query") or question).strip()
    retrieval_hint = str(parameters.get("retrieval_hint") or "").strip()
    semantic_decoupling_enabled = bool(
        config.get("coseek1_retrieval_verify_semantic_decoupling_enabled", False)
        and retrieval_hint
    )
    scope = _question_scope_hint(question)
    allowed_timestamps = [timestamp for _, timestamp, _ in sampled]
    allowed_by_candidate = _candidate_allowed_timestamps(sampled)
    allowed_text = ", ".join(f"{timestamp:.1f}s" for timestamp in allowed_timestamps)
    candidate_binding_enabled = bool(
        config.get("grounded_candidate_binding_enabled", False)
    )
    if p130_context["active"]:
        candidate_binding_enabled = True
    event_identity_repair_enabled = bool(
        config.get("coseek1_tool_event_identity_repair_enabled", False)
    )
    event_groups = _assign_candidate_context_groups(
        candidates,
        max_gap_s=float(config.get("grounded_candidate_event_group_max_gap_s") or 24.0),
        decouple_event_identity=event_identity_repair_enabled,
    )
    candidate_event_binding_requested = bool(
        candidate_binding_enabled
        and config.get("grounded_candidate_event_binding_enabled", False)
    )
    candidate_event_binding_enabled = _should_apply_candidate_event_binding(
        requested=candidate_event_binding_requested,
        event_groups=event_groups,
    )
    if config.get("grounded_candidate_event_binding_force_enabled", False):
        candidate_event_binding_enabled = candidate_event_binding_requested
    if p130_context["active"]:
        candidate_event_binding_requested = True
        candidate_event_binding_enabled = True
    if p130_context.get("p131_active") and p130_context.get("mode") == "order":
        candidate_text = "\n".join(
            f"- {item['candidate_id']} assigned_event_id="
            f"{item.get('target_event_id') or 'unassigned'} {item['t_range']}"
            for item in candidates
        )
    else:
        candidate_text = "\n".join(
            (
                f"- {item['candidate_id']} packet_group={item['packet_group_id']} "
                f"{item['t_range']}: {item['summary']}"
                if candidate_event_binding_enabled and event_identity_repair_enabled
                else f"- {item['candidate_id']} event_group={item['event_group_id']} "
                f"{item['t_range']}: {item['summary']}"
                if candidate_event_binding_enabled
                else f"- {item['candidate_id']} {item['t_range']}: {item['summary']}"
            )
            for item in candidates
        )
    packet_enabled = bool(config.get("grounded_verify_packet_enabled", False))
    adaptive_detail_request_enabled = bool(
        config.get("packet_detail_grounding_escalation_enabled", False)
        and packet_enabled
        and not parameters.get("packet_detail_grounding_escalation_run")
    )
    evidence_need_enabled = bool(
        config.get("packet_adaptive_evidence_need_enabled", False)
        and packet_enabled
        and not parameters.get("packet_detail_grounding_escalation_run")
    )
    decision_sufficiency_enabled = bool(
        config.get("packet_multiwindow_decision_sufficiency_enabled", False)
        and not parameters.get("packet_detail_grounding_escalation_run")
    )
    anchor_audit_enabled = bool(
        config.get("grounded_verify_packet_anchor_audit_enabled", False)
        and packet_enabled
        and not parameters.get("packet_detail_grounding_escalation_run")
    )
    packet_anchor_timestamps_by_candidate: dict[str, list[float]] = {}
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _multiwindow_verifier_instruction(
                question=question,
                query=query,
                candidate_text=candidate_text,
                retrieval_hint=retrieval_hint,
                semantic_decoupling_enabled=semantic_decoupling_enabled,
                p130_context=p130_context,
            ),
        }
    ]
    contact_sheets = bool(config.get("localize_inline_verify_contact_sheets", False))
    grounded_replacements: dict[int, list[dict]] = {}
    grounding_audit: dict | None = None
    if config.get("grounded_frame_verify_enabled", False) and (
        not contact_sheets or packet_enabled
    ):
        grounding_config = deepcopy(config)
        if candidate_binding_enabled:
            anchors_per_candidate = (
                max(
                    1,
                    int(
                        config.get("grounded_verify_packet_anchors_per_candidate")
                        or 2
                    ),
                )
                if packet_enabled
                else 1
            )
            grounding_config["grounded_frame_verify_max_anchors"] = max(
                int(config.get("grounded_frame_verify_max_anchors") or 1),
                len(candidates) * anchors_per_candidate,
            )
        grounded_replacements, grounding_audit = prepare_grounded_frames(
            grounding_config,
            frames=np.asarray([frame for _, _, frame in sampled]),
            frame_indices=sampled_frame_indices,
            timestamps=allowed_timestamps,
            mandatory_anchor_indices=set(),
            candidate_anchor_indices=candidate_anchor_indices,
            query=query,
            question=question,
            output_dir=parameters.get("output_dir"),
        )
    if parameters.get("grounding_escalation_require_crop") is True and (
        not grounding_audit or int(grounding_audit.get("crop_count") or 0) < 1
    ):
        return format_v10_observation(
            {
                "tool": "frame_verify",
                "window_id": "packet_detail_grounding_skipped",
                "scene_id": "multi_scene",
                "t_range": candidates[0]["t_range"],
                "detail_escalation_skipped": True,
                "detail_escalation_skip_reason": "no_valid_local_crop",
                "grounding_audit": grounding_audit or {},
                "observer_backend": "local_qwen",
                "parse_ok": True,
                "num_frames": len(sampled),
            }
        )
    visual_packet_audit: dict[str, Any] | None = None
    if packet_enabled:
        sampled_by_candidate: dict[str, list[tuple[int, float, Any]]] = {
            item["candidate_id"]: [] for item in candidates
        }
        for position, (candidate_id, timestamp, frame) in enumerate(sampled):
            sampled_by_candidate.setdefault(candidate_id, []).append(
                (position, timestamp, frame)
            )
        anchors_by_candidate: dict[str, list[int]] = {}
        for record in (grounding_audit or {}).get("records") or []:
            try:
                position = int(record.get("position"))
            except (TypeError, ValueError):
                continue
            for candidate_id in record.get("candidate_ids") or []:
                rows = anchors_by_candidate.setdefault(str(candidate_id), [])
                if position not in rows:
                    rows.append(position)

        peak_max_side = int(config.get("grounded_verify_packet_peak_max_side") or 960)
        anchors_per_candidate = max(
            1,
            int(config.get("grounded_verify_packet_anchors_per_candidate") or 2),
        )
        packet_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            positioned_rows = sampled_by_candidate.get(candidate_id) or []
            if not positioned_rows:
                continue
            temporal_rows = [
                (timestamp, frame) for _, timestamp, frame in positioned_rows
            ]
            valid_positions = {position for position, _, _ in positioned_rows}
            anchor_positions = [
                position
                for position in anchors_by_candidate.get(candidate_id, [])
                if position in valid_positions
            ][:anchors_per_candidate]
            anchor_selection = "grounding_proposal"
            if not anchor_positions:
                if config.get(
                    "grounded_verify_packet_temporal_anchor_spread_enabled", False
                ):
                    anchor_positions, anchor_selection = (
                        _packet_detail_anchor_positions(
                            candidate=candidate,
                            positioned_rows=positioned_rows,
                            limit=anchors_per_candidate,
                            query_context_roles_enabled=bool(
                                config.get(
                                    "grounded_verify_packet_query_context_roles_enabled",
                                    False,
                                )
                            ),
                            boundary_context_enabled=bool(
                                config.get(
                                    "grounded_verify_packet_boundary_context_enabled",
                                    False,
                                )
                            ),
                        )
                    )
                else:
                    for anchor in candidate.get("timestamp_anchors") or []:
                        nearest_position = min(
                            positioned_rows,
                            key=lambda row: abs(float(row[1]) - float(anchor)),
                        )[0]
                        if nearest_position not in anchor_positions:
                            anchor_positions.append(nearest_position)
                        if len(anchor_positions) >= anchors_per_candidate:
                            break
                    anchor_selection = "timestamp_first"
            row_by_position = {row[0]: row for row in positioned_rows}
            anchor_roles = (
                ["query_peak", "event_context"]
                + ["query_context_supplement"] * max(0, len(anchor_positions) - 2)
                if anchor_selection == "query_peak_plus_event_context"
                else ["query_peak", "boundary_context"]
                + ["query_context_supplement"] * max(0, len(anchor_positions) - 2)
                if anchor_selection == "query_peak_plus_boundary_context"
                else ["detail_anchor"] * len(anchor_positions)
            )
            detail_rows: list[tuple[float, Image.Image, str]] = []
            crop_tiles = 0
            full_tiles = 0
            for anchor_index, anchor_position in enumerate(anchor_positions):
                _, anchor_timestamp, anchor_frame = row_by_position[anchor_position]
                anchor_role = anchor_roles[anchor_index]
                replacement = grounded_replacements.get(anchor_position) or []
                image_item = next(
                    (item for item in replacement if item.get("type") == "image_url"),
                    None,
                )
                if image_item:
                    try:
                        detail_image = _image_from_data_url(
                            image_item["image_url"]["url"]
                        )
                        detail_label = f"{anchor_role.replace('_', ' ')}; crop proposal"
                        crop_tiles += 1
                    except (KeyError, TypeError, ValueError, OSError):
                        detail_image = Image.fromarray(anchor_frame).convert("RGB")
                        detail_label = f"{anchor_role.replace('_', ' ')}; full frame"
                        full_tiles += 1
                else:
                    detail_image = Image.fromarray(anchor_frame).convert("RGB")
                    detail_label = f"{anchor_role.replace('_', ' ')}; full frame"
                    full_tiles += 1
                detail_rows.append((anchor_timestamp, detail_image, detail_label))
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"{candidate_id} mandatory high-resolution direct-anchor audit "
                        f"({len(detail_rows)} timestamp(s))"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _candidate_detail_sheet_data_url(
                            detail_rows,
                            candidate_id=candidate_id,
                            tile_max_side=peak_max_side,
                        ),
                        "detail": "high",
                    },
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"{candidate_id} secondary low-resolution temporal context "
                        f"{candidate['t_range']} ({len(temporal_rows)} ordered frames)"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _candidate_contact_sheet_data_url(
                            temporal_rows,
                            candidate_id=candidate_id,
                        ),
                        "detail": "low",
                    },
                }
            )
            if crop_tiles and full_tiles:
                detail_mode = "mixed"
            elif crop_tiles:
                detail_mode = "crop"
            else:
                detail_mode = "full_anchor"
            packet_rows.append(
                {
                    "candidate_id": candidate_id,
                    "source_frame_count": len(positioned_rows),
                    "anchor_timestamps_s": [
                        round(float(row[0]), 1) for row in detail_rows
                    ],
                    "detail_tile_count": len(detail_rows),
                    "anchor_selection": anchor_selection,
                    "anchor_roles": anchor_roles,
                    "crop_tile_count": crop_tiles,
                    "full_anchor_tile_count": full_tiles,
                    "detail_mode": detail_mode,
                }
            )
            packet_anchor_timestamps_by_candidate[candidate_id] = [
                round(float(row[0]), 1) for row in detail_rows
            ]
        visual_packet_audit = {
            "enabled": True,
            "source_frame_count": len(sampled),
            "candidate_packet_count": len(packet_rows),
            "temporal_sheet_count": len(packet_rows),
            "detail_crop_count": sum(
                row["crop_tile_count"] for row in packet_rows
            ),
            "detail_full_anchor_count": sum(
                row["full_anchor_tile_count"] for row in packet_rows
            ),
            "api_image_count": len(packet_rows) * 2,
            "packets": packet_rows,
        }
    elif contact_sheets:
        sampled_by_candidate: dict[str, list[tuple[float, Any]]] = {
            item["candidate_id"]: [] for item in candidates
        }
        for candidate_id, timestamp, frame in sampled:
            sampled_by_candidate.setdefault(candidate_id, []).append((timestamp, frame))
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            rows = sampled_by_candidate.get(candidate_id) or []
            if not rows:
                continue
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"{candidate_id} temporal strip {candidate['t_range']} "
                        f"({len(rows)} ordered frames)"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _candidate_contact_sheet_data_url(
                            rows,
                            candidate_id=candidate_id,
                        )
                    },
                }
            )
    else:
        for position, (candidate_id, timestamp, frame) in enumerate(sampled):
            content.append({"type": "text", "text": f"{candidate_id} | {timestamp:.1f}s"})
            replacement = grounded_replacements.get(position)
            if replacement:
                content.extend(replacement)
            else:
                content.append({"type": "image_url", "image_url": {"url": _frame_data_url(frame)}})
    content.append(
        {
            "type": "text",
            "text": (
                f"Allowed timestamps: [{allowed_text}]. Do not invent timestamps.\n"
                + (
                    "Each candidate is shown anchor-first: one high-resolution detail sheet containing its strongest timestamped evidence, followed by a low-resolution temporal context strip. Audit the high-resolution anchors independently before using context to interpret event continuity. Crops are untrusted local proposals; validate identity and relation against the context strip before using them as evidence.\n"
                    if packet_enabled
                    else (
                        "Some timestamps contain a low-resolution global frame and an untrusted high-resolution crop proposal. Validate crop identity against its global frame before using crop details; the proposal itself is not evidence.\n"
                        if grounded_replacements
                        else ""
                    )
                )
                + (
                    "Before writing candidate_assessments, audit every timestamped image in each high-resolution detail sheet independently. Return exactly one anchor_assessment for each listed high-resolution anchor. A direct target event visible at any anchor must remain direct in that candidate's final assessment; another context-only anchor in the same candidate must not erase it. Do not infer a direct event from the routing caption or low-resolution strip alone.\n"
                    if anchor_audit_enabled
                    else ""
                )
                + (
                    "Required anchor audit targets: "
                    + "; ".join(
                        f"{candidate_id}={timestamps}"
                        for candidate_id, timestamps in packet_anchor_timestamps_by_candidate.items()
                    )
                    + ".\n"
                    if anchor_audit_enabled
                    else ""
                )
                +
                "Return at least one timestamp_observation for every candidate_id. If the "
                "target is absent in a candidate, report that visible negative fact instead "
                "of omitting the candidate. Every observation must use the candidate_id shown "
                "beside its image and one allowed timestamp from that candidate.\n"
                + (
                    "Return ONLY valid JSON. Report candidate-local visible facts and "
                    "event identity only. Do not compare, support, contradict, or select "
                    "answer options. An entity being present is not enough to establish "
                    "a relation or event.\n"
                    if p130_context["active"]
                    else
                    "Return ONLY valid JSON. Explicitly bind visible facts to the target in the "
                    "question and compare every option listed in the original question. An entity being present is not enough "
                    "to establish a relation or event. Use support only for direct visual proof; "
                    "otherwise use weak_support or unresolved.\n"
                )
                +
                "Unless the question explicitly requires simultaneity or one shot, assess "
                "participation over the temporally continuous event shown across adjacent shots. "
                "Do not require every participant to be co-visible in one frame, and do not join "
                "unrelated shots into one event.\n"
                + (
                    "For this Count question, one candidate tile may contain more than one "
                    "temporally disjoint occurrence. Return each occurrence separately in the "
                    "candidate assessment's occurrences array, with its own event_span, "
                    "best_timestamp_s, observed_fact, target_match, and event_match. Merge "
                    "adjacent shots of one continuous action into one occurrence; do not collapse "
                    "two separated actions merely because they share candidate_id.\n"
                    if p130_context.get("p131_active")
                    and p130_context.get("mode") == "count"
                    else ""
                )
                + (
                    "Assess each candidate independently. Use mismatch when the visible object, "
                    "person, or relation is clearly not the question target; use not_visible "
                    "when that candidate is absent. Candidate-specific facts must not be "
                    "collapsed into one global conclusion. event_span must describe only the "
                    "visible event inside that candidate; never use the min/max range of the "
                    "whole multi-candidate request. Use null when the event boundary is not visible.\n"
                    if candidate_binding_enabled
                    else ""
                )
                + (
                    (
                        "Also bind the exact event instance independently from entity identity. "
                        "packet_group only identifies overlapping input windows for visual "
                        "presentation; it does not assert that two candidates are the same or "
                        "different semantic occurrence. Determine continuity only from the shown "
                        "frames and timestamped transitions. For each candidate return "
                        "event_match=direct only when its frames show the action/state asked in "
                        "the original question or its visually continuous lead-in/outcome. Use "
                        "context_only for useful context and different_event for another visible "
                        "occurrence involving the same entity.\n"
                        if event_identity_repair_enabled
                        else
                        "Also bind the exact event instance independently from entity identity. "
                        "For each candidate return event_match=direct only when its frames show the "
                        "action/state asked in the original question, or a temporally continuous "
                        "lead-in/outcome in the same event_group. Use context_only for useful context "
                        "that does not show that event, and different_event for another occurrence "
                        "involving the same entity. Same person, object, location, or theme alone is "
                        "never direct event evidence. Do not infer causality across event_groups.\n"
                    )
                    if candidate_event_binding_enabled
                    else ""
                )
                + (
                    "A spatial-detail escalation is optional. Request it only when exactly one "
                    "best candidate is already temporally and semantically bound to the question, "
                    "but a small local visual attribute (for example color, text, identity, or a "
                    "small object) is the only remaining blocker. Do not request it for missing "
                    "time coverage, event counting/order, an unseen action or relation, or an "
                    "absent target. Select one shown timestamp and state a neutral crop query.\n"
                    if adaptive_detail_request_enabled
                    else ""
                )
                + (
                    "Classify the principal remaining evidence need independently of the answer: "
                    "sufficient means the shown evidence resolves this verification objective; "
                    "spatial_detail means the target and moment are already bound but a small "
                    "attribute is unreadable; temporal_coverage means more moments must be seen; "
                    "semantic_binding means the visible entity/event is not yet bound to the "
                    "question target; conflict means shown observations disagree. For "
                    "spatial_detail, identify one candidate/timestamp and write a neutral "
                    "detail_query. Do not label an unseen target as spatial_detail.\n"
                    if evidence_need_enabled
                    else ""
                )
                + (
                    "Judge decision_sufficient independently from visual-detail perfection. "
                    "Set it true only when the complete shown candidate set binds the queried "
                    "event and distinguishes one option from every competitor without conflict. "
                    "Keep option_evidence consistent with event binding: when an inspected cue "
                    "for an option belongs to a confirmed different_event, it contradicts that "
                    "option as the explanation of the queried event; use unresolved only when "
                    "the candidate set still lacks evidence to compare that option. "
                    "Optional footage that would merely make an already discriminated answer "
                    "more explicit belongs in missing_detail and does not make "
                    "decision_sufficient false.\n"
                    if decision_sufficiency_enabled and not p130_context["active"]
                    else ""
                )
                +
                "{\n"
                '  "window_id": "inline_multiwindow_verify",\n'
                '  "scene_id": "multi_scene",\n'
                '  "timestamp_observations": [{"candidate_id": "LQ001", "timestamp_s": 1.0, "description": "visible fact or visible absence"}],\n'
                + (
                    '  "candidate_assessments": [\n'
                    '    {"candidate_id": "LQ001", "target_match": "matched|partial|ambiguous|mismatch|not_visible", '
                    + (
                        '"event_match": "direct|context_only|different_event|ambiguous|not_visible", '
                        if candidate_event_binding_enabled
                        else ""
                    )
                    + (
                        '"target_event_id": "1 or empty", "candidate_event_ids": ["1"], '
                        if p130_context["active"] and p130_context["mode"] == "order"
                        else ""
                    )
                    + (
                        '"occurrences": [{"target_match": "matched", "event_match": "direct", '
                        '"observed_fact": "one visible occurrence", "best_timestamp_s": 1.0, '
                        '"event_span": [0.5, 1.5]}], '
                        if p130_context.get("p131_active")
                        and p130_context.get("mode") == "count"
                        else ""
                    )
                    + '"observed_fact": "candidate-specific visible fact", "target_binding_reason": "identity or relation binding", "best_timestamp_s": 1.0, "event_span": [0.5, 1.5], "supports_options": [], "contradicts_options": [], "option_set_conflict": false}\n'
                    '  ],\n'
                    if candidate_binding_enabled
                    else ""
                )
                + (
                    '  "anchor_assessments": [\n'
                    '    {"candidate_id": "LQ001", "timestamp_s": 1.0, "target_match": "matched|partial|ambiguous|mismatch|not_visible", "event_match": "direct|context_only|different_event|ambiguous|not_visible", '
                    + (
                        '"target_event_id": "1 or empty", "candidate_event_ids": ["1"], '
                        if p130_context["active"] and p130_context["mode"] == "order"
                        else ""
                    )
                    + '"observed_fact": "fact visible at this anchor only", "target_binding_reason": "why this anchor binds or does not bind", "supports_options": [], "contradicts_options": [], "option_set_conflict": false}\n'
                    '  ],\n'
                    if anchor_audit_enabled
                    else ""
                )
                +
                '  "overall_summary": "comparison across candidate windows",\n'
                '  "target_entity_or_event": "question target",\n'
                '  "target_match": "matched|partial|ambiguous|not_visible",\n'
                '  "target_binding_reason": "why the evidence binds to the target",\n'
                f'  "question_scope": "{scope}",\n'
                '  "scope_coverage": "sufficient|partial|insufficient",\n'
                '  "scope_coverage_reason": "what these windows establish",\n'
                '  "observed_fact": "neutral verified fact",\n'
                '  "supports_options": [],\n'
                '  "contradicts_options": [],\n'
                + (
                    '  "option_evidence": {},\n'
                    if p130_context["active"]
                    else
                    '  "option_evidence": ' + json.dumps({letter: {"status": "unresolved", "reason": "visual reason"} for letter in config.get("_question_option_letters", "ABCD")}) + ',\n'
                )
                + (
                    '  "detail_escalation": {"needed": false, "candidate_id": "", "timestamp_s": null, "query": "", "reason": ""},\n'
                    if adaptive_detail_request_enabled
                    else ""
                )
                + (
                    '  "evidence_need": "sufficient|spatial_detail|temporal_coverage|semantic_binding|conflict",\n'
                    '  "detail_target": {"candidate_id": "", "timestamp_s": null},\n'
                    '  "detail_query": "neutral local-detail question or empty string",\n'
                    if evidence_need_enabled
                    else ""
                )
                + (
                    '  "decision_sufficient": false,\n'
                    if decision_sufficiency_enabled
                    else ""
                )
                +
                '  "detail_sufficient": false,\n'
                '  "missing_detail": "remaining uncertainty"\n'
                "}\n"
            ),
        }
    )

    # Generic tasks have one semantic verdict per candidate. Anchor rows retain
    # pixels' descriptions only; they cannot overwrite candidate identity or votes.
    candidate_verdict_only = candidate_binding_enabled and not p130_context["active"]
    if candidate_verdict_only:
        content[-1]["text"] = _generic_candidate_contract(allowed_text, packet_anchor_timestamps_by_candidate)

    local_receipt = bool(config.get("local_verifier_receipt_enabled", False)) and not p130_context["active"]
    if local_receipt:
        task_context = content[0]["text"].split("\n\n", 1)[-1]
        prefix, contract = local_receipt_contract(task_context, allowed_by_candidate, packet_anchor_timestamps_by_candidate)
        content[0]["text"], content[-1]["text"] = prefix, contract
    append_window_subtitles(content, parameters, [c["t_range"] for c in candidates])
    verify_config = deepcopy(config)
    verify_config["observer_backend"] = "api"
    verify_config["local_qwen_tools"] = ""
    raw, observer_backend = observe_content(
        verify_config,
        content=content,
        tool_name="focus",
        tool_mode=str(parameters.get("mode") or "option_verify"),
        output_dir=parameters.get("output_dir"),
        return_json=True,
    )
    if raw is None and not local_receipt:
        return "Frame-verify observation failed: model response is empty."
    parsed = extract_json_object(raw or "") or {}
    if local_receipt:
        payload = normalize_local_receipt(parsed, candidates=candidates, allowed=allowed_by_candidate,
                                          anchors=packet_anchor_timestamps_by_candidate, backend=observer_backend)
        payload.update(window_id="inline_multiwindow_verify", scene_id="multi_scene", num_frames=len(sampled),
                       visual_packet_audit=visual_packet_audit, grounding_audit=grounding_audit)
        if not payload["parse_ok"]:
            payload["parse_error_excerpt"] = str(raw)[:1200]
        return format_v10_observation(payload)
    if not parsed:
        return format_v10_observation(
            {
                "tool": "frame_verify",
                "window_id": "inline_multiwindow_verify",
                "scene_id": "multi_scene",
                "t_range": [candidates[0]["t_range"][0], candidates[-1]["t_range"][1]],
                "requested_candidate_ids": [item["candidate_id"] for item in candidates],
                "verified_windows": [],
                "verified_candidate_ids": [],
                "missing_candidate_ids": [item["candidate_id"] for item in candidates],
                "candidate_coverage_complete": False,
                "detail_sufficient": False,
                "missing_detail": "The multi-window verifier did not return valid JSON.",
                "parse_error_excerpt": str(raw)[:1200],
                "observer_backend": observer_backend,
                "parse_ok": False,
                "num_frames": len(sampled),
            },
            fallback_text=raw,
        )

    if candidate_verdict_only:
        judgments = [row for row in parsed.get("candidate_assessments", [])
                     if isinstance(row, dict) and row.get("candidate_id") in allowed_by_candidate]
        parsed["timestamp_observations"] = []  # materialized once from each candidate below
        parsed["overall_summary"] = "; ".join(
            f"{row.get('candidate_id')}: {row.get('observed_fact') or ''}" for row in judgments
        )
        parsed["observed_fact"] = parsed["overall_summary"]
        parsed["target_entity_or_event"] = str(parameters.get("question") or "").split("\n(A)")[0]
        for field in ("supports_options", "contradicts_options"):
            parsed[field] = list(dict.fromkeys(
                letter for row in judgments for letter in _option_letters(row.get(field))
            ))
        parsed["option_evidence"] = {}  # no independent second set of option verdicts
        parsed.setdefault("scope_coverage", "unknown")
        parsed.setdefault("scope_coverage_reason", "Only the shown candidate frames were inspected.")

    target_match = str(parsed.get("target_match") or "unknown").strip().lower()
    if target_match not in _CANDIDATE_TARGET_MATCHES:
        target_match = "unknown"
    if candidate_binding_enabled:
        assessments, binding_complete = _normalize_candidate_assessments(
            parsed.get("candidate_assessments"),
            candidates=candidates,
            allowed_by_candidate=allowed_by_candidate,
            event_binding_enabled=candidate_event_binding_enabled,
            required_event_ids=p130_required_event_ids,
            count_occurrences_enabled=bool(
                p130_context.get("p131_active")
                and p130_context.get("mode") == "count"
            ),
            assigned_target_by_candidate=(
                p130_context.get("assigned_target_by_candidate")
                if p130_context.get("p132_active") else None
            ),
        )
        anchor_assessments: list[dict[str, Any]] = []
        anchor_audit_complete = False
        if anchor_audit_enabled:
            anchor_assessments, anchor_audit_complete = _normalize_anchor_assessments(
                parsed.get("anchor_assessments"),
                allowed_by_candidate=packet_anchor_timestamps_by_candidate,
                event_binding_enabled=candidate_event_binding_enabled,
                required_event_ids=p130_required_event_ids,
                assigned_target_by_candidate=(
                    p130_context.get("assigned_target_by_candidate")
                    if p130_context.get("p132_active") else None
                ),
            )
            if not candidate_verdict_only:
                _merge_anchor_audit_into_candidate_assessments(
                    assessments,
                    anchor_assessments,
                    event_binding_enabled=candidate_event_binding_enabled,
                    required_event_ids=p130_required_event_ids,
                    assigned_target_by_candidate=(
                        p130_context.get("assigned_target_by_candidate")
                        if p130_context.get("p132_active") else None
                    ),
                )
            anchor_override_ids = [
                row["candidate_id"]
                for row in assessments
                if row.get("anchor_audit_direct_preserved") is True
            ]
        _bind_candidate_observations(
            parsed,
            candidates=candidates,
            assessments=assessments,
            allowed_by_candidate=allowed_by_candidate,
            synthesize_assessment_anchor=not bool(p130_context.get("p131_active")),
        )
        if anchor_audit_enabled and not candidate_verdict_only:
            _materialize_anchor_audit_observations(parsed, anchor_assessments)
        binding_summary = _candidate_binding_summary(
            assessments,
            event_binding_enabled=candidate_event_binding_enabled,
        )
        target_match = binding_summary["target_match"]
        parsed.update(binding_summary)
        parsed["candidate_assessments"] = assessments
        parsed["candidate_binding_complete"] = binding_complete
        if anchor_audit_enabled:
            parsed["anchor_assessments"] = anchor_assessments
            parsed["anchor_audit_complete"] = anchor_audit_complete
            parsed["anchor_audit_expected_count"] = sum(
                len(rows) for rows in packet_anchor_timestamps_by_candidate.values()
            )
            parsed["anchor_audit_candidate_overrides"] = anchor_override_ids
            if anchor_override_ids:
                direct_rows = [
                    row
                    for row in assessments
                    if row.get("event_match") == "direct"
                ]
                direct_groups = {
                    str(row.get("event_group_id") or row.get("candidate_id") or "")
                    for row in direct_rows
                }
                facts = "; ".join(
                    f"{row.get('candidate_id')}@{row.get('best_timestamp_s')}s: "
                    f"{row.get('observed_fact') or 'direct target evidence'}"
                    for row in direct_rows
                )
                parsed["pre_anchor_audit_overall_summary"] = str(
                    parsed.get("overall_summary") or ""
                ).strip()
                parsed["overall_summary"] = (
                    f"Anchor audit preserves direct target evidence in "
                    f"{len(direct_groups)} distinct candidate/event group(s). {facts}"
                )[:1800]
                parsed["observed_fact"] = parsed["overall_summary"]
    parsed.update(
        {
            "tool": "frame_verify",
            "window_id": "inline_multiwindow_verify",
            "scene_id": "multi_scene",
            "t_range": [
                min(item["t_range"][0] for item in candidates),
                max(item["t_range"][1] for item in candidates),
            ],
            "target_match": target_match,
            "question_scope": scope,
            "observer_backend": observer_backend,
            "parse_ok": True,
            "num_frames": len(sampled),
            "visual_layout": (
                "candidate_verify_packets"
                if packet_enabled
                else (
                    "candidate_contact_sheets" if contact_sheets else "individual_frames"
                )
            ),
            "inline_localize_verify": True,
            "compact_render": True,
            "candidate_event_binding_requested": candidate_event_binding_requested,
            "candidate_event_binding_applied": candidate_event_binding_enabled,
            "packet_anchor_audit_requested": anchor_audit_enabled,
            "retrieval_verify_semantic_decoupling": semantic_decoupling_enabled,
            "verification_objective_source": (
                "original_question" if semantic_decoupling_enabled else "query"
            ),
            "retrieval_hint_present": bool(retrieval_hint),
            "retrieval_hint_exposed_to_verifier": False,
        }
    )
    if event_identity_repair_enabled:
        parsed["candidate_packet_group_count"] = len(set(event_groups.values()))
        parsed["candidate_event_identity_preassigned"] = False
    else:
        parsed["candidate_event_group_count"] = len(set(event_groups.values()))
    if grounding_audit:
        parsed["grounding_audit"] = grounding_audit
    if visual_packet_audit:
        parsed["visual_packet_audit"] = visual_packet_audit
    supports = _option_letters(parsed.get("supports_options"))
    contradicts = _option_letters(parsed.get("contradicts_options"))
    if candidate_binding_enabled:
        for assessment in parsed.get("candidate_assessments") or []:
            for letter in assessment.get("supports_options") or []:
                if letter not in supports:
                    supports.append(letter)
            for letter in assessment.get("contradicts_options") or []:
                if letter not in contradicts:
                    contradicts.append(letter)
    option_evidence = parsed.get("option_evidence") or {}
    if isinstance(option_evidence, dict):
        for letter in config.get("_question_option_letters", "ABCD"):
            row = option_evidence.get(letter)
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "").strip().lower()
            if status == "support" and letter not in supports:
                supports.append(letter)
            elif status == "contradict" and letter not in contradicts:
                contradicts.append(letter)
    parsed["supports_options"] = supports
    parsed["contradicts_options"] = contradicts
    aggregate_snapshot: dict[str, Any] = {}
    if (
        candidate_event_binding_enabled
        and config.get(
            "packet_preserve_resolved_candidate_set_decision_enabled", False
        )
    ):
        aggregate_snapshot = {
            "supports_options": list(supports),
            "contradicts_options": list(contradicts),
            "option_evidence": deepcopy(parsed.get("option_evidence") or {}),
            "decision_sufficient": parsed.get("decision_sufficient") is True,
        }
    if candidate_event_binding_enabled:
        _apply_event_binding_option_scope(parsed)
    inspected_duration = sum(item["t_range"][1] - item["t_range"][0] for item in candidates)
    if scope == "global_video" and duration > 0 and inspected_duration / duration < 0.8:
        parsed["scope_coverage"] = "partial"
        parsed["scope_coverage_reason"] = (
            f"The verified candidate windows cover {inspected_duration:.1f}s of a "
            f"{duration:.1f}s video, so they do not establish a global claim."
        )
    parsed.setdefault("observed_fact", parsed.get("overall_summary") or "")
    parsed.setdefault("detail_sufficient", False)
    parsed.setdefault("missing_detail", "")
    if evidence_need_enabled:
        _demote_partial_global_option_claims(parsed)
        parsed["evidence_need"] = _normalize_evidence_need(parsed)
        parsed["evidence_needs"] = _normalize_evidence_needs(parsed)
    if not candidate_binding_enabled:
        snap_timestamp_observations(parsed, allowed_timestamps=allowed_timestamps)
    requested_ids = [item["candidate_id"] for item in candidates]
    covered_ids = _covered_candidate_ids(parsed, candidates)
    missing_ids = [item for item in requested_ids if item not in covered_ids]
    parsed.update(
        {
            "requested_candidate_ids": requested_ids,
            "verified_candidate_ids": covered_ids,
            "missing_candidate_ids": missing_ids,
            "verified_windows": [
                item["t_range"]
                for item in candidates
                if item["candidate_id"] in covered_ids
            ],
            "candidate_coverage_complete": not missing_ids,
        }
    )
    if aggregate_snapshot:
        aggregate_restored = _restore_resolved_candidate_set_decision(
            parsed,
            aggregate_snapshot=aggregate_snapshot,
            candidate_coverage_complete=not missing_ids,
        )
        if aggregate_restored and evidence_need_enabled:
            parsed["evidence_need"] = _normalize_evidence_need(parsed)
            parsed["evidence_needs"] = _normalize_evidence_needs(parsed)
    if missing_ids:
        parsed["detail_sufficient"] = False
        coverage_gap = f"No timestamped observation returned for candidates {missing_ids}."
        existing_gap = str(parsed.get("missing_detail") or "").strip()
        parsed["missing_detail"] = (
            f"{coverage_gap} {existing_gap}".strip() if existing_gap else coverage_gap
        )
    if decision_sufficiency_enabled:
        parsed["decision_sufficiency_requested"] = True
        _apply_reported_decision_completeness(
            parsed,
            candidate_count=len(candidates),
            option_letters=config.get("_question_option_letters"),
        )
    if (
        not missing_ids
        and not decision_sufficiency_enabled
        and config.get("packet_decision_consistency_normalization_enabled", False)
    ):
        _normalize_decision_completeness(parsed, config.get("_question_option_letters"))
    _apply_p130_local_receipt_contract(parsed, context=p130_context)
    return format_v10_observation(parsed, fallback_text=raw)


def _execute_global_observation(config: dict, parameters: dict) -> str:
    """One frame inventory, one local response, no hidden semantic recovery."""
    vr = parameters["vr"]
    fps, count = float(vr.get_avg_fps()), len(vr)
    duration = count/fps
    candidates = parameters["candidate_windows"]
    total = min(int(config.get("localize_inline_verify_max_frames") or 24),
                int(parameters.get("frame_count") or 24))
    budgets = _allocate_candidate_frames(len(candidates), total)
    seen = set(parameters.get("seen_frame_ids") or [])
    sampled = {}
    clue_refs_by_candidate = {}
    for candidate, budget in zip(candidates, budgets):
        a, b = candidate["t_range"]
        anchors = list(candidate.get("timestamp_anchors") or [])
        key_indices = []
        for key in candidate.get("key_frames") or []:
            index = key.get("frame_index")
            if not isinstance(index, int) or isinstance(index, bool):
                timestamp = key.get("source_timestamp_s", key.get("timestamp_s"))
                index = int(round(float(timestamp) * fps))
            if not 0 <= index < count or not a <= index / fps <= b:
                raise ValueError("Key frame is outside its source candidate")
            if index not in key_indices:
                key_indices.append(index)
        clue_refs_by_candidate[candidate["candidate_id"]] = [f"F{index}" for index in key_indices]
        if len(key_indices) > budget:
            return format_v10_observation({"schema_version": RUNTIME_SCHEMA_VERSION, "parse_ok": False,
                "local_observations": [], "sampled_frames": [], "contract_errors": [
                    f"Candidate {candidate['candidate_id']} requires {len(key_indices)} key frames but has {budget} slots; select fewer candidates in this verification action."]})
        anchor_indices = timestamps_to_frame_indices(timestamps=anchors, fps=fps, total_frames=count).tolist() if anchors else []
        context = [i for i in anchor_indices if i not in key_indices]
        slots = budget - len(key_indices)
        context = _evenly_limit_positions(context, slots) if slots else []
        indices = key_indices + context
        grid = timestamps_to_frame_indices(timestamps=np.linspace(a, b, max(4*budget, 2)).tolist(), fps=fps, total_frames=count).tolist()
        while len(indices) < budget:
            pool = [i for i in grid if i not in indices]
            if not pool:
                break
            best = max(pool, key=lambda i: (f"F{i}" not in seen, min((abs(i-j) for j in indices), default=0)))
            indices.append(best)
        for index in sorted(indices):
            ref = f"F{index}"
            if ref not in sampled:
                sampled[ref] = {"frame_id": ref, "frame_index": index, "timestamp_s": round(index/fps, 3),
                                "frame_duration_s": 1/fps, "candidate_ids": [], "detail": "high"}
            sampled[ref]["candidate_ids"].append(candidate["candidate_id"])
    frames = sorted(sampled.values(), key=lambda r: r["frame_index"])
    # Candidate membership follows source time, not which sampling pass selected a frame.
    bounds = {c["candidate_id"]: timestamps_to_frame_indices(
        timestamps=c["t_range"], fps=fps, total_frames=count).tolist() for c in candidates}
    for row in frames:
        row["candidate_ids"] = [cid for cid, indices in bounds.items()
                                if indices[0] <= row["frame_index"] <= indices[-1]]
    if parameters.get("resample") and all(r["frame_id"] in seen for r in frames):
        return format_v10_observation({"schema_version": RUNTIME_SCHEMA_VERSION, "parse_ok": False,
                                      "local_observations": [], "sampled_frames": [], "contract_errors": ["no_new_frames"]})
    parsed = parse_global_question(str(parameters["question"]))
    targets = {r["event_id"]: r["description"] for r in parsed.get("required_events") or []}
    if not targets:
        targets = {"target": parsed["stem"]}
    request = {
        "original_question": parsed["stem"],
        "verification_question": str(parameters.get("query") or "").strip(),
        "targets": targets,
        "candidates": [{"candidate_id": c["candidate_id"], "range": c["t_range"],
                        "target_id": c.get("target_id"),
                        **({"unverified_local_clue": c["unverified_clue"]} if c.get("unverified_clue") else {})} for c in candidates],
        "frames": frames,
    }
    frame_inventory = {row["frame_id"]: row for row in frames}
    for candidate, projected in zip(candidates, request["candidates"]):
        clue_refs = clue_refs_by_candidate[candidate["candidate_id"]]
        if any(ref not in frame_inventory or candidate["candidate_id"] not in frame_inventory[ref]["candidate_ids"] for ref in clue_refs):
            raise ValueError("Local clue anchor missing from its candidate's uploaded frames")
        if clue_refs:
            projected["unverified_clue_frame_ids"] = clue_refs
    example = {"match": "direct|negative|ambiguous", "fact": "visible fact", "frame_ids": ["F123"]}
    if len(targets) > 1:
        example = {"target_id": next(iter(targets)), **example}
    instruction = (
        "Inspect the actual video frames in time order. Each image has equal evidence status. "
        "Assess targets at the level requested by the original question: scene, action, object, or relation. "
        "Preserve its qualifiers and counting unit without adding prerequisites. "
        "Report one observation per visible local occurrence, allowing several occurrences per candidate. "
        "Use direct only when visible evidence establishes the requested target; mere association is insufficient. "
        "Do not require an unasked action or infer an unseen one. Use ambiguous when evidence does not "
        "resolve the requested target; state any unresolved local detail in fact. Negative applies only to the "
        "inspected frames, never to an entire unobserved interval. Keep separated occurrences separate. "
        "For one continuous occurrence, cite its supporting frames together; each observation must "
        "use frames sharing an input candidate. The source frame inventory supplies their location and time. "
        "frame_ids must name only images that support this observation's fact. A candidate "
        "may contain unrelated context or a scene change: do not cite every image in its "
        "window as support, and do not extend an occurrence across unrelated frames. "
        "Do not infer events from candidate assignments or expected order. "
        "The verification_question is an unresolved local question, not a fact or a new target definition. "
        "Address it in the visible fact or explain what remains unresolved; the original question takes precedence. "
        "Local clues are unverified search descriptions, not evidence; check them against pixels. "
        "For a candidate with target_id, assess only that target. Otherwise inspect all catalog targets "
        "and also report ambiguous observations for targets not established in this candidate. "
        "For a single catalog target, the request supplies the target identity. For multiple targets, "
        "copy one exact target_id key from the targets dictionary, not its description. "
        "Return only JSON with this shape: " + json.dumps({"observations": [example]}) +
        ". No option claims or separate audit judgments.\n"
        + json.dumps(request, ensure_ascii=False)
    )
    content = [{"type": "text", "text": instruction}]
    pixels = vr.get_batch([r["frame_index"] for r in frames]).asnumpy()
    for row, frame in zip(frames, pixels):
        content.append({"type": "text", "text": f"{row['frame_id']} | {row['timestamp_s']}s | {','.join(row['candidate_ids'])}"})
        content.append({"type": "image_url", "image_url": {"url": _frame_data_url(frame, max_side=1024, label=row['frame_id']), "detail": "high"}})
    append_window_subtitles(content, parameters, [c["t_range"] for c in candidates])
    observer_config = dict(config, observer_backend="api", local_qwen_tools="")
    raw, backend = observe_content(observer_config, content=content, tool_name="frame_verify",
                                   output_dir=parameters.get("output_dir"), return_json=True)
    try:
        decoded = extract_json_object(raw) or {}
    except (ValueError, TypeError):
        decoded = {}
    observations, errors = parse_local_observations(decoded, frames=frames, candidates=candidates, parsed=parsed)
    payload = {"schema_version": RUNTIME_SCHEMA_VERSION, "parse_ok": bool(observations),
               "local_observations": observations, "sampled_frames": frames,
               "raw_observer_response": raw, "contract_errors": errors, "observer_backend": backend}
    return format_v10_observation(payload)
