"""Deterministic, read-only Planner evidence projections for P99 and P100.

The complete observation memory remains the canonical local state.  This module
only produces a smaller API-facing view and an audit record.  It does not choose
tools, answer questions, merge events by overlap/text similarity, or mutate its
input.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import re
from typing import Any, Iterable


CAPSULE_VERSION = "p99_planner_evidence_capsule_v1"
DECISION_NEUTRAL_CAPSULE_VERSION = "p100_decision_neutral_capsule_v1"
CAPSULE_HEADER = (
    "Planner Evidence Capsule (read-only projection; full local memory is retained). "
    "Cloud reviews in candidate_frontier are local pixel observations with uncertainty, not global answer sufficiency. Other candidates are unverified routing clues.\n"
)


@dataclass(frozen=True)
class PlannerCapsuleResult:
    """Rendered API projection and deterministic diagnostics."""

    text: str
    audit: dict[str, Any]


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _natural_id_key(value: Any) -> tuple[str, int, str]:
    text = str(value or "")
    match = re.match(r"^(.*?)(\d+)$", text)
    if not match:
        return text, -1, text
    return match.group(1), int(match.group(2)), text


def _sorted_ids(values: Iterable[Any]) -> list[str]:
    return sorted(_unique(values), key=_natural_id_key)


def _clean_text(value: Any, limit: int | None = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit is None or len(text) <= limit:
        return text
    cut = text[: max(1, limit - 1)].rstrip()
    boundary = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(", "), cut.rfind(" "))
    if boundary >= max(40, limit // 2):
        cut = cut[:boundary].rstrip(" ,;")
    return cut + "…"


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _short_hash(value: Any, prefix: str) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"{prefix}{hashlib.sha256(raw).hexdigest()[:10].upper()}"


def _memory_hash(memory: dict[str, Any]) -> str:
    raw = json.dumps(
        memory,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=1)
def _token_encoder() -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except (ImportError, KeyError, ValueError):
        return None


def estimate_tokens(text: str) -> tuple[int, str]:
    """Return deterministic token accounting used by the runtime and audits."""

    encoder = _token_encoder()
    if encoder is not None:
        return len(encoder.encode(text)), "o200k_base"
    # Conservative for mixed Chinese/English compared with the old chars/4
    # approximation.  It is only used when the frozen tokenizer is unavailable.
    return max(1, (len(text) + 2) // 3), "char_fallback_div3"


def _time_range(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = round(float(value[0]), 1), round(float(value[1]), 1)
    except (TypeError, ValueError):
        return None
    if end < start:
        start, end = end, start
    return [start, end]


def _ranges_overlap(left: Any, right: Any) -> bool:
    a = _time_range(left)
    b = _time_range(right)
    if a is None or b is None:
        return False
    return min(a[1], b[1]) >= max(a[0], b[0])


def _merge_intervals(values: Iterable[Any]) -> list[list[float]]:
    intervals = sorted(
        (span for span in (_time_range(value) for value in values) if span is not None),
        key=lambda span: (span[0], span[1]),
    )
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 0.1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [[round(start, 1), round(end, 1)] for start, end in merged]


def _source_call_identity(item: dict[str, Any]) -> dict[str, Any]:
    refs = item.get("refs") if isinstance(item.get("refs"), dict) else {}
    params = refs.get("parameters") if isinstance(refs.get("parameters"), dict) else {}
    # These fields identify a real source call without copying its long query,
    # candidate summaries, anchors, or images into the Planner projection.
    compact_params = {
        key: params.get(key)
        for key in ("start_time", "end_time", "mode")
        if params.get(key) is not None
    }
    candidate_windows = _dict_rows(params.get("candidate_windows"))
    if candidate_windows:
        compact_params["candidate_windows"] = [
            {
                "candidate_id": row.get("candidate_id"),
                "t_range": _time_range(row.get("t_range")),
            }
            for row in candidate_windows
        ]
    return {
        "tool": item.get("source_tool") or "unknown",
        "backend": item.get("backend") or "unknown",
        "scene": item.get("scene_id") or "",
        "window": item.get("window_id") or "",
        "parameters": compact_params,
    }


def _answer_state(
    memory: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    compact = memory.get("compact_investigation_state") or {}
    answer = compact.get("answer_status") if isinstance(compact, dict) else {}
    if not isinstance(answer, dict):
        answer = {}
    hypotheses = _dict_rows(compact.get("option_hypotheses") if isinstance(compact, dict) else [])

    leading = str(answer.get("option") or "").strip()
    if not leading:
        ranked = sorted(
            hypotheses,
            key=lambda row: (
                -len(_unique(row.get("verified_support") or [])),
                -len(_unique(row.get("support") or [])),
                str(row.get("option") or ""),
            ),
        )
        if ranked and (ranked[0].get("verified_support") or ranked[0].get("support")):
            leading = str(ranked[0].get("option") or "")

    competitors = [row for row in hypotheses if str(row.get("option") or "") != leading]
    competitors.sort(
        key=lambda row: (
            -len(_unique(row.get("verified_support") or [])),
            -int(bool(row.get("evidence_conflict"))),
            -len(_unique(row.get("support") or [])),
            str(row.get("option") or ""),
        )
    )
    competitor = str(competitors[0].get("option") or "") if competitors else ""
    decisive = set(_unique(answer.get("support_refs") or []))

    return (
        {
            "leading_option": leading or None,
            "strongest_competitor": competitor or None,
            "readiness": str(answer.get("status") or "no_verified_answer_yet"),
            "decisive_support_ids": _sorted_ids(decisive),
            "uncovered_options": _unique(answer.get("uncovered_options") or []),
        },
        hypotheses,
        decisive,
    )


def _build_conflicts(
    memory: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    leading_option: str,
    *,
    max_conflicts: int,
) -> tuple[list[dict[str, Any]], set[str], int]:
    rows: list[dict[str, Any]] = []

    for conflict in _dict_rows(memory.get("observation_conflicts")):
        refs = _sorted_ids(
            list(conflict.get("fact_refs") or [])
            + list(conflict.get("verification_ids") or [])
        )
        facts = [_clean_text(value, 180) for value in conflict.get("facts") or [] if value]
        identity = {
            "kind": "recorded_observation_conflict",
            "type": conflict.get("conflict_type") or "unknown",
            "refs": refs,
            "range": _time_range(conflict.get("t_range")),
        }
        rows.append(
            {
                "conflict_id": str(conflict.get("conflict_id") or _short_hash(identity, "CF-")),
                "kind": identity["kind"],
                "evidence_refs": refs,
                "unresolved_difference": facts[:2]
                or [_clean_text(conflict.get("conflict_type") or "recorded conflict", 180)],
                "time_range": identity["range"],
                "priority": 3,
            }
        )

    for row in hypotheses:
        if row.get("evidence_conflict") is not True:
            continue
        option = str(row.get("option") or "")
        support = _sorted_ids(row.get("verified_support") or [])
        contradict = _sorted_ids(row.get("verified_contradict") or [])
        refs = _sorted_ids(support + contradict)
        identity = {"kind": "option_evidence_conflict", "option": option, "refs": refs}
        rows.append(
            {
                "conflict_id": _short_hash(identity, "CF-"),
                "kind": identity["kind"],
                "option": option,
                "evidence_refs": refs,
                "support_refs": support,
                "contradict_refs": contradict,
                "unresolved_difference": "verified support and contradiction disagree",
                "priority": 4 if option == leading_option else 2,
            }
        )

    # Deduplicate the same source conflict exposed through two local views.
    deduped: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get("evidence_refs") or []) or (str(row.get("conflict_id")),)
        current = deduped.get(key)
        if current is None or int(row.get("priority") or 0) > int(current.get("priority") or 0):
            deduped[key] = row
    ordered = sorted(
        deduped.values(),
        key=lambda row: (-int(row.get("priority") or 0), str(row.get("conflict_id") or "")),
    )

    # The item limit is soft for recorded strong conflicts: omission would make
    # the projection look falsely consistent.  The token budget remains hard.
    selected = ordered[:max_conflicts]
    if len(ordered) > max_conflicts:
        selected.extend(row for row in ordered[max_conflicts:] if row.get("evidence_refs"))
    for row in selected:
        row.pop("priority", None)
    strong_evidence = {
        ref
        for row in selected
        for ref in row.get("evidence_refs") or []
        if re.fullmatch(r"E\d+", str(ref))
    }
    return selected, strong_evidence, max(0, len(ordered) - len(selected))


def _matching_binding(
    item: dict[str, Any], bindings: list[dict[str, Any]]
) -> dict[str, Any] | None:
    candidate_id = str(item.get("candidate_id") or "")
    if not candidate_id:
        return None
    candidates = [row for row in bindings if str(row.get("candidate_id") or "") == candidate_id]
    if not candidates:
        return None
    overlapping = [row for row in candidates if _ranges_overlap(row.get("t_range"), item.get("t_range"))]
    pool = overlapping or candidates
    return sorted(
        pool,
        key=lambda row: (
            str(row.get("verification_id") or ""),
            str(row.get("id") or ""),
        ),
    )[0]


def _matching_ledger_event(
    item: dict[str, Any], ledger_events: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return the canonical verified event containing this direct anchor.

    Candidate windows may overlap and may describe the same occurrence.  The
    temporal ledger has already performed that conservative episode merge, so
    P125 uses its identity instead of exposing every candidate as a separate
    verified event.  Exact ledger anchors are required whenever they exist;
    this prevents an unrelated context timestamp inside a positive candidate
    from inheriting the candidate-level direct label.
    """
    if str(item.get("source_tool") or "") != "frame_verify":
        return None
    if str(item.get("target_match") or "").strip().lower() != "matched":
        return None
    if str(item.get("event_match") or "").strip().lower() != "direct":
        return None
    candidate_id = str(item.get("candidate_id") or "").strip()
    try:
        timestamp = float(item.get("timestamp_s"))
    except (TypeError, ValueError):
        return None
    if not candidate_id:
        return None

    matches: list[dict[str, Any]] = []
    for event in ledger_events:
        if candidate_id not in _unique(event.get("candidate_ids") or []):
            continue
        anchors: list[float] = []
        for value in event.get("anchor_timestamps") or []:
            try:
                anchors.append(float(value))
            except (TypeError, ValueError):
                continue
        if anchors:
            if min(abs(timestamp - value) for value in anchors) > 0.25:
                continue
        else:
            span = _time_range(event.get("t_range"))
            if span is None or not span[0] <= timestamp <= span[1]:
                continue
        matches.append(event)
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda event: (
            _natural_id_key(event.get("event_id")),
            str(event.get("verification_id") or ""),
        ),
    )[0]


def _verified_fact_groups(
    memory: dict[str, Any],
    *,
    required_evidence_ids: set[str],
    max_verified: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    structured = memory.get("structured_evidence") or {}
    all_evidence_rows = _dict_rows(
        structured.get("evidence_items") if isinstance(structured, dict) else []
    )
    ledger_bridge_enabled = bool(
        (memory.get("runtime_config") or {}).get(
            "p125_ledger_capsule_bridge_enabled", False
        )
    )
    ledger = memory.get("temporal_evidence_ledger") or {}
    ledger_events = (
        _dict_rows(ledger.get("events"))
        if ledger_bridge_enabled and isinstance(ledger, dict)
        else []
    )
    ledger_event_by_evidence_id = {
        str(row.get("evidence_id") or ""): event
        for row in all_evidence_rows
        if (event := _matching_ledger_event(row, ledger_events)) is not None
    }
    evidence = [
        row
        for row in all_evidence_rows
        if str(row.get("evidence_level") or "") == "verified"
        or str(row.get("evidence_id") or "") in ledger_event_by_evidence_id
    ]
    bindings = _dict_rows(memory.get("candidate_binding_memory"))
    grouped: dict[str, dict[str, Any]] = {}

    for item in evidence:
        evidence_id = str(item.get("evidence_id") or "").strip()
        ledger_event = ledger_event_by_evidence_id.get(evidence_id)
        ledger_fact = str((ledger_event or {}).get("fact") or "")
        source_fact = str(item.get("observed_fact") or item.get("description") or "")
        # Older ledgers contain a shortened copy. Recover its exact continuation
        # from the already source-bound observation, without rewriting semantics.
        prefix = ledger_fact[:-3] if ledger_fact.endswith("...") else ledger_fact[:-1] if ledger_fact.endswith("…") else None
        if prefix and source_fact.startswith(prefix):
            ledger_fact = source_fact
        fact = _clean_text(ledger_fact or source_fact, None)
        call_identity = _source_call_identity(item)
        binding = _matching_binding(item, bindings)
        if ledger_event is not None:
            event_identity_basis = "temporal_evidence_ledger"
            event_source = {
                "ledger_event_id": ledger_event.get("event_id"),
                "verification_ids": _sorted_ids(
                    ledger_event.get("verification_ids") or []
                ),
                "anchor_timestamps": sorted(
                    float(value)
                    for value in ledger_event.get("anchor_timestamps") or []
                ),
            }
        elif binding and binding.get("verification_id"):
            event_identity_basis = "explicit_verification_binding"
            event_source = {
                "verification_id": binding.get("verification_id"),
                "candidate_id": binding.get("candidate_id"),
                "event_group_id": binding.get("event_group_id")
                or binding.get("local_event_group_id")
                or "",
            }
        else:
            # Conservative identity: a source call is one provenance unit.  It
            # intentionally does not merge distinct calls merely because their
            # windows overlap or their text looks similar.
            event_identity_basis = "source_call_identity"
            event_source = {
                "source_call": _short_hash(call_identity, "SC-"),
                "candidate_id": item.get("candidate_id") or "",
                "scene_id": item.get("scene_id") or "",
            }
        event_id = _short_hash(event_source, "EV-")
        fact_id = _short_hash(_normalized_text(fact), "F-")
        supports = _unique(
            (ledger_event or {}).get("supports_options")
            or item.get("supports_options")
            or []
        )
        contradicts = _unique(
            (ledger_event or {}).get("contradicts_options")
            or item.get("contradicts_options")
            or []
        )
        group_identity = {"event_source": event_source}
        if ledger_event is None:
            group_identity.update(
                {
                    "fact": _normalized_text(fact),
                    "supports": sorted(supports),
                    "contradicts": sorted(contradicts),
                }
            )
        group_id = _short_hash(group_identity, "VG-")
        row = grouped.setdefault(
            group_id,
            {
                "verified_group_id": group_id,
                "event_id": event_id,
                "fact_id": fact_id,
                "fact": fact,
                "evidence_ids": [],
                "source_tool": str(item.get("source_tool") or "unknown"),
                "backend": str(item.get("backend") or "unknown"),
                "scene_id": str(item.get("scene_id") or "") or None,
                "time_range": _time_range(
                    (ledger_event or {}).get("t_range") or item.get("t_range")
                ),
                "supports": [],
                "contradicts": [],
                "target_match": (
                    "matched"
                    if ledger_event is not None
                    else str(item.get("target_match") or "unknown")
                ),
                "event_match": (
                    "direct"
                    if ledger_event is not None
                    else str(item.get("event_match") or "unknown")
                ),
                "scope": (
                    "local_event_span"
                    if ledger_event is not None
                    else str(item.get("evidence_scope") or "unknown")
                ),
                "decision_sufficient": bool(item.get("decision_sufficient")),
                "target": _clean_text(item.get("target_entity_or_event"), 180),
                "required": False,
                "source_call_id": _short_hash(call_identity, "SC-"),
                "event_identity_basis": event_identity_basis,
            },
        )
        row["evidence_ids"] = _sorted_ids(list(row["evidence_ids"]) + [evidence_id])
        row["supports"] = sorted(set(row["supports"]) | set(supports))
        row["contradicts"] = sorted(
            set(row["contradicts"]) | set(contradicts)
        )
        row["decision_sufficient"] = bool(row["decision_sufficient"] or item.get("decision_sufficient"))
        row["required"] = bool(row["required"] or evidence_id in required_evidence_ids)

    def score(row: dict[str, Any]) -> tuple[Any, ...]:
        direct = row.get("target_match") == "matched" and row.get("event_match") in {"direct", "matched"}
        return (
            -int(bool(row.get("required"))),
            -int(bool(row.get("decision_sufficient"))),
            -int(bool(row.get("supports"))),
            -int(bool(row.get("contradicts"))),
            -int(direct),
            _natural_id_key((row.get("evidence_ids") or [""])[-1]),
            str(row.get("verified_group_id") or ""),
        )

    ordered = sorted(grouped.values(), key=score)
    required = [row for row in ordered if row.get("required")]
    optional = [row for row in ordered if not row.get("required")]
    selected = required + optional[: max(0, max_verified - len(required))]
    selected_ids = {str(row.get("verified_group_id")) for row in selected}
    omitted = [row for row in ordered if str(row.get("verified_group_id")) not in selected_ids]

    fact_catalog: dict[str, str] = {}
    rendered: list[dict[str, Any]] = []
    for row in selected:
        fact_catalog.setdefault(str(row["fact_id"]), str(row["fact"]))
        rendered.append(
            {
                "verified_group_id": row["verified_group_id"],
                "evidence_ids": row["evidence_ids"],
                "event_id": row["event_id"],
                "fact_id": row["fact_id"],
                "time_range": row["time_range"],
                "supports": row["supports"],
                "contradicts": row["contradicts"],
                "target_match": row["target_match"],
                "event_match": row["event_match"],
                "scope": row["scope"],
                "source": f"{row['source_tool']}/{row['backend']}",
                "source_call_id": row["source_call_id"],
                "event_identity_basis": row["event_identity_basis"],
                "scene_id": row["scene_id"],
            }
        )

    catalog = [
        {"fact_id": fact_id, "text": text}
        for fact_id, text in sorted(fact_catalog.items())
    ]
    selected_evidence = {
        evidence_id for row in rendered for evidence_id in row.get("evidence_ids") or []
    }
    all_evidence = {
        evidence_id
        for row in ordered
        for evidence_id in row.get("evidence_ids") or []
    }
    metadata = {
        "all_verified_group_count": len(ordered),
        "selected_verified_group_count": len(rendered),
        "omitted_verified_group_count": len(omitted),
        "selected_verified_evidence_ids": _sorted_ids(selected_evidence),
        "omitted_verified_evidence_ids": _sorted_ids(all_evidence - selected_evidence),
        "omitted_required_evidence_ids": _sorted_ids(required_evidence_ids - selected_evidence),
        "required_group_limit_overflow": max(0, len(required) - max_verified),
        "ledger_bridge_event_count": sum(
            row.get("event_identity_basis") == "temporal_evidence_ledger"
            for row in ordered
        ),
        "internal_rows": selected,
    }
    return rendered, catalog, metadata


def _candidate_rows(
    memory: dict[str, Any],
    *,
    max_candidates: int,
    receipt_first: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, str], int]:
    source = _dict_rows(memory.get("candidate_pool") or memory.get("overview_candidates"))
    # Project receipts through the existing candidate view; no second mutable state.
    known = {str(c.get("candidate_id")) for c in source}
    seen_receipts = set()
    for e in reversed(_dict_rows((memory.get("structured_evidence") or {}).get("evidence_items"))):
        if e.get("receipt_schema") != "local_visual_receipt_v1":
            continue
        signature = _short_hash({**{k: e.get(k) for k in (
            "candidate_id", "t_range", "observed_fact", "target_binding_reason",
            "uncertainty", "anchor_observations", "target_event_id")},
            "query": ((e.get("refs") or {}).get("parameters") or {}).get("query")}, "R-")
        if signature in seen_receipts:
            continue
        seen_receipts.add(signature)
        source_cid, verification_id = e.get("candidate_id"), e.get("verification_id")
        cid = f"{verification_id}:{source_cid}"
        if e.get("receipt_schema") == "local_visual_receipt_v1" and source_cid and verification_id and cid not in known:
            source.append({"candidate_id": cid, "source_candidate_id": source_cid,
                           "verification_id": verification_id, "receipt_schema": e["receipt_schema"], "t_range": e.get("t_range"),
                           "status": "inspected_partial", "source": "frame_verify/api"})
            known.add(cid)
    coverage_by_scene = {
        str(row.get("scene_id") or ""): row for row in _dict_rows(memory.get("scene_coverage"))
    }
    deduped: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(source):
        candidate_id = str(item.get("candidate_id") or "").strip()
        if not candidate_id:
            candidate_id = _short_hash(
                {
                    "scene": item.get("scene_id"),
                    "range": _time_range(item.get("t_range")),
                    "source_index": index,
                },
                "C-",
            )
        scene_id = str(item.get("scene_id") or "")
        coverage = coverage_by_scene.get(scene_id) or {}
        ratio = float(coverage.get("coverage_ratio") or 0.0)
        status = str(item.get("status") or "unvisited")
        if status in {"unvisited", "routing_only", "unknown"} and ratio > 0:
            status = "inspected_partial" if ratio < 0.95 else "inspected"
        event_id = _short_hash({"candidate_id": candidate_id}, "EV-C-")
        row = {
            "event_id": event_id,
            "candidate_id": candidate_id,
            "scene_id": scene_id or None,
            "time_range": _time_range(item.get("t_range")),
            "status": status,
            "expected_new_information": _clean_text(item.get("missing_detail"), 180)
            or "not recorded",
            "possible_evidence": item.get("possible_evidence"),
            "query_relevance": int(item.get("query_relevance_score") or 0),
            "coverage_ratio": round(ratio, 3),
        }
        existing = deduped.get(candidate_id)
        if existing is None or (
            row["query_relevance"], row["possible_evidence"] is True
        ) > (
            existing["query_relevance"], existing["possible_evidence"] is True
        ):
            deduped[candidate_id] = row

    unresolved_status = {"unvisited", "routing_only", "unknown", "inspected_partial"}
    ordered = sorted(
        deduped.values(),
        key=lambda row: (
            -int(row.get("status") in unresolved_status),
            -int(row.get("possible_evidence") is True),
            -int(row.get("query_relevance") or 0),
            float(row.get("coverage_ratio") or 0.0),
            str(row.get("scene_id") or ""),
            str(row.get("candidate_id") or ""),
        ),
    )
    # Keep a compact view of the existing pool; the token fitter, rather than
    # an unrelated eight-row cutoff, determines how much reaches the Planner.
    # This does not change candidate generation, ranking or verification state.
    by_id = {str(item.get("candidate_id")): item for item in source}
    evidence = _dict_rows((memory.get("structured_evidence") or {}).get("evidence_items"))
    rendered = []
    for row in ordered:
        item = by_id.get(row["candidate_id"], {})
        cues = _dict_rows(item.get("timestamp_cues"))
        # Cues arrive in relevance order, not necessarily temporal order.
        cues = sorted(cues, key=lambda cue: float(cue.get("timestamp_s") or 0))
        cues = [cues[0], cues[-1]] if len(cues) > 1 else cues
        observations = [
            {"timestamp_s": cue.get("timestamp_s"),
             "text": _clean_text(cue.get("description"), 180)}
            for cue in cues if cue.get("description")
        ]
        for observation in observations:
            source_row = next((e for e in evidence
                               if e.get("timestamp_s") == observation["timestamp_s"]
                               and _clean_text(e.get("description"), 180) == observation["text"]), None)
            if source_row:
                observation["ref"] = source_row.get("evidence_id")
                observation["source"] = f"{source_row.get('source_tool')}/{source_row.get('backend')}"
        projected = {
            "event_id": row["event_id"], "candidate_id": row["candidate_id"],
            "scene_id": row["scene_id"], "time_range": row["time_range"],
            "status": row["status"], "source": item.get("source") or "candidate_pool",
            "observations": observations or [{"text": _clean_text(item.get("summary"), 180)}],
        }
        # Window overlap retrieves observations; candidate identity stays explicit.
        # Keep the latest receipt for each source candidate, not only the latest
        # observation anywhere inside a broad scene. References avoid repetition.
        latest_by_candidate = {}
        for e in reversed(evidence):
            if (e.get("source_tool") == "frame_verify" and e.get("backend") == "api"
                    and ((e.get("candidate_id") == item.get("source_candidate_id")
                          and e.get("verification_id") == item.get("verification_id"))
                         if e.get("receipt_schema") == "local_visual_receipt_v1"
                         else (item.get("receipt_schema") != "local_visual_receipt_v1" and _ranges_overlap(e.get("t_range"), row["time_range"])))
                    and (e.get("observed_fact") or e.get("description"))):
                identity = e.get("candidate_id") or _short_hash(_source_call_identity(e), "SC-")
                latest_by_candidate.setdefault(identity, e)
        reviews = []
        for recent in latest_by_candidate.values():
            evidence_id = recent.get("evidence_id")
            reviews.append({
                "evidence_id": evidence_id, "candidate_id": recent.get("candidate_id"),
                "source": "frame_verify/api", "timestamp_s": recent.get("timestamp_s"),
                "association": "window_overlap_only",
                "text": _clean_text(recent.get("observed_fact") or recent.get("description"), None),
                "target_match": recent.get("target_match") or "unknown",
                "event_match": recent.get("event_match") or "unknown",
                "binding_reason": _clean_text(recent.get("target_binding_reason"), None),
                "uncertainty": _clean_text(recent.get("uncertainty"), None),
            })
            if recent.get("receipt_schema") == "local_visual_receipt_v1":
                reviews[-1].update(receipt_schema=recent["receipt_schema"], association="verification_id_and_candidate_id",
                                   verification_id=recent.get("verification_id"), time_range=recent.get("t_range"),
                                   target_event_id=recent.get("target_event_id") or "",
                                   anchor_observations=recent.get("anchor_observations") or [])
                reviews[-1].pop("target_match", None)
                reviews[-1].pop("event_match", None)
        if item.get("receipt_schema") == "local_visual_receipt_v1":
            projected["source_candidate_id"] = item["source_candidate_id"]
            projected["verification_id"] = item["verification_id"]
        if reviews:
            projected["review"] = reviews[0]
            if len(reviews) > 1:
                projected["additional_reviews"] = reviews[1:]
        # Do not substitute workflow boilerplate for the observed content.
        if item.get("missing_detail") and not str(item["missing_detail"]).startswith(("Local ", "Use frame_verify")):
            projected["expected_new_information"] = _clean_text(item["missing_detail"], 140)
        if receipt_first and item.get("receipt_schema") == "local_visual_receipt_v1":
            # Identity and scope already live on the candidate. Keep factual text
            # and uncertainty verbatim; remove only repeated routing metadata.
            projected.pop("scene_id", None)
            projected.pop("observations", None)
            for review in [projected.get("review") or {}] + projected.get("additional_reviews", []):
                for key in ("source", "candidate_id", "verification_id", "time_range", "receipt_schema", "association"):
                    review.pop(key, None)
        rendered.append(projected)
    if receipt_first:
        # Chronology comes from recorded tool calls, never relevance or lexical IDs.
        verification_order = {str(row.get("verification_id")): index
                              for index, row in enumerate(_dict_rows(memory.get("tool_observations")))
                              if row.get("verification_id")}
        fallback = {str(row.get("verification_id")): index for index, row in enumerate(evidence)
                    if row.get("verification_id")}
        rendered.sort(key=lambda row: (
            0 if row.get("verification_id") else 1,
            -verification_order.get(str(row.get("verification_id")),
                                    fallback.get(str(row.get("verification_id")), -1)),
            _natural_id_key(row.get("source_candidate_id")) if row.get("verification_id") else ("", 0, ""),
        ))
    return rendered, {row["candidate_id"]: row["event_id"] for row in rendered}, 0


def _matching_candidate_events(
    *,
    scene_id: Any,
    time_range: Any,
    candidates: list[dict[str, Any]],
) -> list[str]:
    scene = str(scene_id or "")
    exact_scene = [
        str(row.get("event_id"))
        for row in candidates
        if scene and str(row.get("scene_id") or "") == scene
    ]
    if exact_scene:
        return _unique(exact_scene)
    return _unique(
        row.get("event_id")
        for row in candidates
        if _ranges_overlap(time_range, row.get("time_range"))
    )


def _obligations(
    memory: dict[str, Any],
    *,
    answer_state: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_obligations: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    uncovered = _unique(answer_state.get("uncovered_options") or [])
    if uncovered:
        identity = {
            "kind": "uncovered_options",
            "leading": answer_state.get("leading_option"),
            "uncovered": uncovered,
        }
        rows.append(
            {
                "obligation_id": _short_hash(identity, "O-"),
                "discriminator": (
                    f"Distinguish leading option {answer_state.get('leading_option') or '?'} "
                    f"from uncovered options {','.join(uncovered)} with verified evidence."
                ),
                "required_scope": "question scope",
                "candidate_event_ids": [],
                "priority": 4,
            }
        )

    gap_groups: dict[str, dict[str, Any]] = {}
    for gap in _dict_rows(memory.get("open_gaps")):
        if str(gap.get("status") or "open").lower() in {"resolved", "closed", "done"}:
            continue
        text = _clean_text(gap.get("gap") or gap.get("missing_detail"), 220)
        if not text:
            continue
        normalized = _normalized_text(text)
        event_ids = _matching_candidate_events(
            scene_id=gap.get("scene_id"),
            time_range=gap.get("suggested_window"),
            candidates=candidates,
        )
        row = gap_groups.setdefault(
            normalized,
            {
                "obligation_id": _short_hash({"gap": normalized}, "O-"),
                "source_gap_ids": [],
                "discriminator": text,
                "required_scope": _time_range(gap.get("suggested_window")) or "unspecified",
                "candidate_event_ids": [],
                "priority": 3 if event_ids else 2,
            },
        )
        row["source_gap_ids"] = _sorted_ids(
            list(row["source_gap_ids"]) + [gap.get("gap_id")]
        )
        row["candidate_event_ids"] = _unique(
            list(row["candidate_event_ids"]) + event_ids
        )

    rows.extend(gap_groups.values())
    existing_candidate_events = {
        event_id for row in rows for event_id in row.get("candidate_event_ids") or []
    }
    for candidate in candidates:
        event_id = str(candidate.get("event_id") or "")
        if event_id in existing_candidate_events:
            continue
        if candidate.get("status") not in {"unvisited", "routing_only", "unknown", "inspected_partial"}:
            continue
        detail = str(candidate.get("expected_new_information") or "")
        if not detail or detail == "not recorded":
            continue
        rows.append(
            {
                "obligation_id": _short_hash(
                    {"candidate": candidate.get("candidate_id"), "detail": detail}, "O-"
                ),
                "discriminator": detail,
                "required_scope": candidate.get("time_range") or "candidate event",
                "candidate_event_ids": [event_id],
                "priority": 1,
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("obligation_id") or "")
        if key and key not in deduped:
            deduped[key] = row
    ordered = sorted(
        deduped.values(),
        key=lambda row: (-int(row.get("priority") or 0), str(row.get("obligation_id") or "")),
    )
    selected = ordered[:max_obligations]
    for row in selected:
        row.pop("priority", None)
        if not row.get("source_gap_ids"):
            row.pop("source_gap_ids", None)
    return selected, max(0, len(ordered) - len(selected))


def _coverage_rows(
    memory: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    verified: list[dict[str, Any]],
    max_rows: int,
) -> list[dict[str, Any]]:
    event_by_scene: dict[str, list[str]] = {}
    for row in candidates + verified:
        scene = str(row.get("scene_id") or "")
        event_id = str(row.get("event_id") or "")
        if scene and event_id:
            event_by_scene.setdefault(scene, []).append(event_id)
    output: list[dict[str, Any]] = []
    for row in _dict_rows(memory.get("scene_coverage")):
        scene = str(row.get("scene_id") or "")
        event_ids = _unique(event_by_scene.get(scene) or [])
        if not event_ids:
            continue
        ratio = float(row.get("coverage_ratio") or 0.0)
        status = "uninspected" if ratio <= 0 else ("inspected" if ratio >= 0.95 else "partial")
        output.append(
            {
                "event_ids": event_ids,
                "scene_id": scene,
                "inspected_intervals": _merge_intervals(row.get("covered_intervals") or []),
                "coverage_ratio": round(ratio, 3),
                "status": status,
                "evidence_found": bool(row.get("evidence_found")),
            }
        )
    return sorted(output, key=lambda row: (str(row.get("scene_id")), row.get("event_ids")))[:max_rows]


def _question_scope(
    memory: dict[str, Any],
    *,
    question: str,
    verified_internal: list[dict[str, Any]],
) -> dict[str, Any]:
    request = re.split(r"\n\s*\([A-Z]\)\s+", question or "", maxsplit=1)[0]
    scopes = [
        str(row.get("scope") or "")
        for row in verified_internal
        if str(row.get("scope") or "") not in {"", "unknown"}
    ]
    temporal_scope = Counter(scopes).most_common(1)[0][0] if scopes else "unknown"
    targets = _unique(row.get("target") for row in verified_internal if row.get("target"))
    compact = memory.get("compact_investigation_state") or {}
    answer = compact.get("answer_status") if isinstance(compact, dict) else {}
    search_coverage = answer.get("search_coverage") if isinstance(answer, dict) else {}
    return {
        "request": _clean_text(request, 280),
        "requested_target": targets[0] if targets else None,
        "temporal_scope_from_evidence": temporal_scope,
        "coverage_requirement": {
            "overview_scene_count": (search_coverage or {}).get("overview_scene_count"),
            "materially_inspected_scene_count": (search_coverage or {}).get(
                "materially_inspected_scene_count"
            ),
        },
    }


def _constraints(
    memory: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    verified: list[dict[str, Any]],
    omitted_candidates: int,
) -> dict[str, Any]:
    inspected_candidates = _sorted_ids(
        row.get("candidate_id")
        for row in _dict_rows(memory.get("candidate_pool") or memory.get("overview_candidates"))
        if str(row.get("status") or "unvisited") not in {"unvisited", "routing_only", "unknown"}
    )
    inspected_events = _unique(
        [row.get("event_id") for row in verified]
        + [
            row.get("event_id")
            for row in candidates
            if row.get("status") not in {"unvisited", "routing_only", "unknown"}
        ]
    )
    all_intervals = [
        interval
        for row in _dict_rows(memory.get("scene_coverage"))
        for interval in row.get("covered_intervals") or []
    ]
    merged = _merge_intervals(all_intervals)
    return {
        "already_inspected_event_ids": inspected_events,
        "already_inspected_candidate_ids": inspected_candidates,
        "no_new_pixel_windows": merged[:12],
        "omitted_inspected_interval_count": max(0, len(merged) - 12),
        "omitted_candidate_count": omitted_candidates,
        "evidence_rule": (
            "Routing clues are not verified evidence; revisit only for a distinct "
            "unresolved discriminator or genuinely new pixels."
        ),
    }


def _render(payload: dict[str, Any]) -> str:
    # Share complete fact bodies already present in this exact fitted view.
    # Recompute after each fit step so dropping a fact never leaves a dangling ref.
    view = deepcopy(payload)
    facts = {row["text"]: row["fact_id"] for row in view.get("fact_catalog") or []}
    rendered_reviews: set[str] = set()
    uncertainty_sources: dict[str, str] = {}
    for candidate in view.get("candidate_frontier") or []:
        reviews = []
        first = candidate.get("review") or (
            {"review_ref": candidate["review_ref"]} if candidate.get("review_ref") else {}
        )
        for review in [first] + (candidate.get("additional_reviews") or []):
            if not review:
                continue
            evidence_id = review.get("evidence_id")
            if evidence_id and evidence_id in rendered_reviews:
                reviews.append({"review_ref": evidence_id})
                continue
            if evidence_id:
                rendered_reviews.add(evidence_id)
            if review.get("text") in facts:
                review["fact_id"] = facts[review.pop("text")]
            # Exact shared uncertainty stays verbatim at its first surviving
            # review. The reference denotes shared text, not a shared event.
            uncertainty = review.get("uncertainty")
            if evidence_id and uncertainty:
                if uncertainty in uncertainty_sources:
                    review["uncertainty_ref"] = uncertainty_sources[uncertainty]
                    review.pop("uncertainty")
                else:
                    uncertainty_sources[uncertainty] = evidence_id
            reviews.append(review)
        if reviews:
            candidate.pop("review", None)
            candidate.pop("review_ref", None)
            candidate.pop("additional_reviews", None)
            if "review_ref" in reviews[0]:
                candidate["review_ref"] = reviews[0]["review_ref"]
            else:
                candidate["review"] = reviews[0]
            if len(reviews) > 1:
                candidate["additional_reviews"] = reviews[1:]
    return CAPSULE_HEADER + json.dumps(
        view,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _prune_unused_facts(payload: dict[str, Any]) -> None:
    used = {str(row.get("fact_id") or "") for row in payload.get("verified_evidence") or []}
    payload["fact_catalog"] = [
        row for row in payload.get("fact_catalog") or [] if str(row.get("fact_id") or "") in used
    ]


def _fit_payload(
    payload: dict[str, Any],
    *,
    token_budget: int,
    required_evidence_ids: set[str],
    defer_linked_candidates: bool = False,
) -> tuple[dict[str, Any], str, int, str, list[str]]:
    """Drop only optional low-priority rows until the hard budget fits."""

    fitted = deepcopy(payload)
    reductions: list[str] = []

    def current() -> tuple[str, int, str]:
        text = _render(fitted)
        count, tokenizer = estimate_tokens(text)
        return text, count, tokenizer

    text, count, tokenizer = current()
    while count > token_budget:
        intervals = (fitted.get("constraints") or {}).get("no_new_pixel_windows") or []
        if len(intervals) > 4:
            intervals.pop()
            reductions.append("drop_low_priority_covered_interval")
        elif fitted.get("coverage"):
            fitted["coverage"].pop()
            reductions.append("drop_low_priority_coverage_row")
        else:
            obligation_events = {
                event_id
                for row in fitted.get("unresolved_obligations") or []
                for event_id in row.get("candidate_event_ids") or []
            }
            removable_candidate = next(
                (
                    index
                    for index in range(len(fitted.get("candidate_frontier") or []) - 1, -1, -1)
                    if defer_linked_candidates or fitted["candidate_frontier"][index].get("event_id")
                    not in obligation_events
                ),
                None,
            )
            if removable_candidate is not None:
                omitted = fitted["candidate_frontier"].pop(removable_candidate)
                fitted.setdefault("constraints", {}).setdefault("deferred_candidate_ids", []).append(omitted["candidate_id"])
                if defer_linked_candidates:
                    # The question and its time scope remain visible. Candidate
                    # identity moves to deferred refs; never leave dangling links.
                    for obligation in fitted.get("unresolved_obligations") or []:
                        obligation["candidate_event_ids"] = [
                            event for event in obligation.get("candidate_event_ids", [])
                            if event != omitted.get("event_id")
                        ]
                reductions.append("drop_optional_candidate")
            else:
                removable_verified = next(
                    (
                        index
                        for index in range(len(fitted.get("verified_evidence") or []) - 1, -1, -1)
                        if not required_evidence_ids.intersection(
                            fitted["verified_evidence"][index].get("evidence_ids") or []
                        )
                    ),
                    None,
                )
                if removable_verified is not None:
                    fitted["verified_evidence"].pop(removable_verified)
                    _prune_unused_facts(fitted)
                    reductions.append("drop_optional_verified_group")
                elif len(fitted.get("unresolved_obligations") or []) > 1:
                    fitted["unresolved_obligations"].pop()
                    reductions.append("drop_low_priority_obligation")
                else:
                    break
        text, count, tokenizer = current()
    return fitted, text, count, tokenizer, reductions


def _all_valid_source_ids(memory: dict[str, Any]) -> tuple[set[str], set[str]]:
    structured = memory.get("structured_evidence") or {}
    evidence_ids = {
        str(row.get("evidence_id") or "")
        for row in _dict_rows(structured.get("evidence_items") if isinstance(structured, dict) else [])
        + _dict_rows(structured.get("option_evidence") if isinstance(structured, dict) else [])
        if row.get("evidence_id")
    }
    candidate_ids = {
        str(row.get("candidate_id") or "")
        for row in _dict_rows(memory.get("candidate_pool") or memory.get("overview_candidates"))
        + _dict_rows(memory.get("candidate_binding_memory"))
        if row.get("candidate_id")
    }
    candidate_ids.update(
        f"{row['verification_id']}:{row['candidate_id']}"
        for row in _dict_rows(structured.get("evidence_items") if isinstance(structured, dict) else [])
        if row.get("receipt_schema") == "local_visual_receipt_v1"
        and row.get("verification_id") and row.get("candidate_id")
    )
    return evidence_ids, candidate_ids


def _invalid_references(payload: dict[str, Any], memory: dict[str, Any]) -> list[str]:
    valid_evidence, valid_candidates = _all_valid_source_ids(memory)
    event_ids = {
        str(row.get("event_id") or "")
        for row in (payload.get("verified_evidence") or []) + (payload.get("candidate_frontier") or [])
        if row.get("event_id")
    }
    invalid: list[str] = []
    for ref in payload.get("answer_state", {}).get("decisive_support_ids") or []:
        if ref not in valid_evidence:
            invalid.append(f"answer_evidence:{ref}")
    for row in payload.get("verified_evidence") or []:
        for ref in row.get("evidence_ids") or []:
            if ref not in valid_evidence:
                invalid.append(f"verified_evidence:{ref}")
    for row in payload.get("candidate_frontier") or []:
        candidate = str(row.get("candidate_id") or "")
        if candidate not in valid_candidates:
            invalid.append(f"candidate:{candidate}")
    for row in payload.get("unresolved_obligations") or []:
        for event_id in row.get("candidate_event_ids") or []:
            if event_id not in event_ids:
                invalid.append(f"obligation_event:{event_id}")
    for row in payload.get("coverage") or []:
        for event_id in row.get("event_ids") or []:
            if event_id not in event_ids:
                invalid.append(f"coverage_event:{event_id}")
    return sorted(set(invalid))


def build_planner_evidence_capsule(
    memory: dict[str, Any] | None,
    *,
    question: str,
    full_memory_text: str,
    token_budget: int = 4000,
    max_verified: int = 8,
    max_candidates: int = 8,
    max_obligations: int = 4,
    max_conflicts: int = 2,
    _decision_neutral: bool = False,
) -> PlannerCapsuleResult:
    """Project complete local memory into a bounded Planner evidence capsule."""

    source = memory if isinstance(memory, dict) else {}
    before_hash = _memory_hash(source)
    full_tokens, full_tokenizer = estimate_tokens(full_memory_text)
    has_observation = any(
        source.get(key)
        for key in (
            "tool_observations",
            "candidate_pool",
            "timestamped_observations",
            "scene_memory",
            "verification_facts",
        )
    )
    if not has_observation or full_memory_text == "(empty)":
        after_hash = _memory_hash(source)
        return PlannerCapsuleResult(
            text=full_memory_text,
            audit={
                "version": CAPSULE_VERSION,
                "activated": False,
                "decision_neutral_activated": False,
                "reason": "empty_memory",
                "full_memory_chars": len(full_memory_text),
                "full_memory_estimated_tokens": full_tokens,
                "capsule_chars": len(full_memory_text),
                "capsule_estimated_tokens": full_tokens,
                "tokenizer": full_tokenizer,
                "token_budget": token_budget,
                "memory_hash_before": before_hash,
                "memory_hash_after": after_hash,
                "memory_mutated": before_hash != after_hash,
                "capsule_hash": hashlib.sha256(full_memory_text.encode("utf-8")).hexdigest(),
                "fail_open": False,
                "over_budget": False,
            },
        )

    answer_state, hypotheses, decisive_ids = _answer_state(source)
    conflicts, conflict_evidence_ids, omitted_conflicts = _build_conflicts(
        source,
        hypotheses,
        str(answer_state.get("leading_option") or ""),
        max_conflicts=max_conflicts,
    )
    required_ids = set(decisive_ids) | set(conflict_evidence_ids)
    verified, fact_catalog, verified_meta = _verified_fact_groups(
        source,
        required_evidence_ids=required_ids,
        max_verified=max_verified,
    )
    candidates, _event_by_candidate, omitted_candidates = _candidate_rows(
        source,
        max_candidates=max_candidates,
        receipt_first=_decision_neutral,
    )
    removed_fields = []
    if _decision_neutral:
        for field in ("leading_option", "strongest_competitor", "uncovered_options"):
            if field in answer_state:
                answer_state.pop(field)
                removed_fields.append(f"answer_state.{field}")
    obligations, omitted_obligations = _obligations(
        source,
        answer_state=answer_state,
        candidates=candidates,
        max_obligations=max_obligations,
    )
    coverage = _coverage_rows(
        source,
        candidates=candidates,
        verified=verified,
        max_rows=max_candidates,
    )
    payload = {
        "schema_version": CAPSULE_VERSION,
        "question_scope": _question_scope(
            source,
            question=question,
            verified_internal=verified_meta["internal_rows"],
        ),
        "answer_state": answer_state,
        "fact_catalog": fact_catalog,
        "verified_evidence": verified,
        "candidate_frontier": candidates,
        "coverage": coverage,
        "unresolved_obligations": obligations,
        "conflicts": conflicts,
        "constraints": _constraints(
            source,
            candidates=candidates,
            verified=verified,
            omitted_candidates=omitted_candidates,
        ),
    }
    fitted, capsule_text, capsule_tokens, tokenizer, reductions = _fit_payload(
        payload,
        token_budget=max(1, int(token_budget)),
        required_evidence_ids=required_ids,
        defer_linked_candidates=_decision_neutral,
    )
    fail_open = capsule_tokens > token_budget
    if fail_open:
        output_text = full_memory_text
        output_tokens, tokenizer = estimate_tokens(output_text)
    else:
        output_text = capsule_text
        output_tokens = capsule_tokens

    included_evidence_ids = {
        evidence_id
        for row in fitted.get("verified_evidence") or []
        for evidence_id in row.get("evidence_ids") or []
    }
    included_event_ids = _unique(
        [row.get("event_id") for row in fitted.get("verified_evidence") or []]
        + [row.get("event_id") for row in fitted.get("candidate_frontier") or []]
    )
    omitted_decisive = _sorted_ids(decisive_ids - included_evidence_ids)
    omitted_conflict_evidence = _sorted_ids(conflict_evidence_ids - included_evidence_ids)
    rendered_payload = json.loads(capsule_text[len(CAPSULE_HEADER):])
    invalid_refs = _invalid_references(rendered_payload, source)
    facts_in_view = {row["fact_id"] for row in rendered_payload.get("fact_catalog") or []}
    reviews_in_view = {
        review.get("evidence_id")
        for candidate in rendered_payload.get("candidate_frontier") or []
        for review in [candidate.get("review") or {}] + (candidate.get("additional_reviews") or [])
        if review.get("evidence_id")
    }
    uncertainty_sources_in_view = {
        review.get("evidence_id")
        for candidate in rendered_payload.get("candidate_frontier") or []
        for review in [candidate.get("review") or {}] + (candidate.get("additional_reviews") or [])
        if review.get("uncertainty")
    }
    for group in rendered_payload.get("verified_evidence") or []:
        if group.get("fact_id") not in facts_in_view:
            invalid_refs.append(f"capsule_fact:{group.get('fact_id')}")
    for candidate in rendered_payload.get("candidate_frontier") or []:
        for review in [candidate.get("review") or candidate] + (candidate.get("additional_reviews") or []):
            if review.get("review_ref") and review["review_ref"] not in reviews_in_view:
                invalid_refs.append(f"capsule_review:{review['review_ref']}")
            if review.get("uncertainty_ref") and review["uncertainty_ref"] not in uncertainty_sources_in_view:
                invalid_refs.append(f"capsule_uncertainty:{review['uncertainty_ref']}")
            if review.get("fact_id") and review["fact_id"] not in facts_in_view:
                invalid_refs.append(f"capsule_fact:{review['fact_id']}")
    fact_texts = [
        _normalized_text(row.get("text")) for row in fitted.get("fact_catalog") or []
    ]
    duplicate_fact_bodies = len(fact_texts) - len(set(fact_texts))
    after_hash = _memory_hash(source)
    audit = {
        "version": CAPSULE_VERSION,
        "activated": True,
        "full_memory_chars": len(full_memory_text),
        "full_memory_estimated_tokens": full_tokens,
        "capsule_chars": len(output_text),
        "capsule_estimated_tokens": output_tokens,
        "tokenizer": tokenizer,
        "token_budget": int(token_budget),
        "estimated_memory_token_reduction": round(
            1.0 - output_tokens / max(1, full_tokens), 6
        ),
        "included_evidence_ids": _sorted_ids(included_evidence_ids),
        "included_event_ids": included_event_ids,
        "included_candidate_ids": _sorted_ids(
            row.get("candidate_id") for row in fitted.get("candidate_frontier") or []
        ),
        "candidate_event_map": {
            str(row.get("candidate_id")): str(row.get("event_id"))
            for row in fitted.get("candidate_frontier") or []
            if row.get("candidate_id") and row.get("event_id")
        },
        "unresolved_obligation_ids": _sorted_ids(
            row.get("obligation_id") for row in fitted.get("unresolved_obligations") or []
        ),
        "all_verified_group_count": verified_meta["all_verified_group_count"],
        "included_verified_group_count": len(fitted.get("verified_evidence") or []),
        "omitted_verified_group_count": (
            verified_meta["all_verified_group_count"]
            - len(fitted.get("verified_evidence") or [])
        ),
        "omitted_verified_evidence_ids": _sorted_ids(
            set(verified_meta["selected_verified_evidence_ids"])
            .union(verified_meta["omitted_verified_evidence_ids"])
            - included_evidence_ids
        ),
        "omitted_decisive_support_ids": omitted_decisive,
        "omitted_strong_conflict_evidence_ids": omitted_conflict_evidence,
        "required_group_limit_overflow": verified_meta["required_group_limit_overflow"],
        "ledger_bridge_event_count": verified_meta["ledger_bridge_event_count"],
        "omitted_candidate_count": (
            omitted_candidates
            + max(0, len(candidates) - len(fitted.get("candidate_frontier") or []))
        ),
        "omitted_obligation_count": (
            omitted_obligations
            + max(0, len(obligations) - len(fitted.get("unresolved_obligations") or []))
        ),
        "omitted_conflict_count": omitted_conflicts,
        "invalid_references": invalid_refs,
        "duplicate_fact_bodies": duplicate_fact_bodies,
        "event_identity_policy": (
            "temporal_ledger_for_grounded_direct_candidates_else_explicit_source_identity"
            if verified_meta["ledger_bridge_event_count"]
            else "explicit_source_identity_only_no_overlap_or_text_merge"
        ),
        "budget_reductions": reductions,
        "fail_open": fail_open,
        "over_budget": fail_open,
        "memory_hash_before": before_hash,
        "memory_hash_after": after_hash,
        "memory_mutated": before_hash != after_hash,
        "capsule_hash": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
    }
    if _decision_neutral:
        audit.update(version="p159_planner_local_receipt_capsule_v1",
                     decision_neutral_activated=not fail_open,
                     decision_neutral_reason="policy_removed_before_fit",
                     removed_policy_fields=removed_fields,
                     no_backfill=False,
                     omitted_local_receipt_ids=[
                         row["candidate_id"] for row in candidates if row.get("verification_id")
                         and row["candidate_id"] not in audit["included_candidate_ids"]])
    return PlannerCapsuleResult(text=output_text, audit=audit)


def build_decision_neutral_planner_capsule(
    memory: dict[str, Any] | None,
    *,
    question: str,
    full_memory_text: str,
    token_budget: int = 4000,
    max_verified: int = 8,
    max_candidates: int = 8,
    max_obligations: int = 4,
    max_conflicts: int = 2,
) -> PlannerCapsuleResult:
    """Build one policy-neutral projection, then fit it to the requested budget."""
    return build_planner_evidence_capsule(
        memory, question=question, full_memory_text=full_memory_text,
        token_budget=token_budget, max_verified=max_verified,
        max_candidates=max_candidates, max_obligations=max_obligations,
        max_conflicts=max_conflicts, _decision_neutral=True,
    )
