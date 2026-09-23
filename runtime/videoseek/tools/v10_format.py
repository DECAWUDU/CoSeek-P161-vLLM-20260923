import json
from typing import Any


def local_search_instruction(question: str, inspection_hint: str) -> str:
    """Derive visual targets from the question; retain hints only in action logs."""
    from videoseek.core.minimal_global_fsm import parse_global_question

    parsed = parse_global_question(question)
    stem = parsed["stem"]
    targets = {row["event_id"]: row["description"] for row in parsed["required_events"]}
    return (
        "Local task: locate and describe visible entities, actions and relations relevant "
        "to the original question in the supplied frames; retain timestamps and uncertainty. "
        "The original question and its event catalog define the targets and qualifiers. "
        "Do not determine whole-video totals, cross-window occurrence independence, "
        "global event order or answer options from this local sample. Quantities that "
        "describe visible entities remain part of the target. Absence applies only to "
        "the inspected pixels.\n"
        f"Original question (context, not a request for a global answer): {stem}\n"
        + (f"Event catalog (IDs do not imply temporal order): {json.dumps(targets, ensure_ascii=False)}\n"
           if targets else "")
    )


def snap_timestamp_observations(
    payload: dict[str, Any],
    *,
    allowed_timestamps: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    if not allowed_timestamps:
        return payload
    allowed = [round(float(item), 1) for item in allowed_timestamps]
    observations = payload.get("timestamp_observations") or []
    if not isinstance(observations, list):
        return payload
    for item in observations:
        if not isinstance(item, dict):
            continue
        try:
            ts = float(item.get("timestamp_s", item.get("timestamp")))
        except Exception:
            item["timestamp_s"] = allowed[0]
            continue
        nearest = min(allowed, key=lambda candidate: abs(candidate - ts))
        item["timestamp_s"] = nearest
    return payload


def format_v10_observation(payload: dict[str, Any], *, fallback_text: str | None = None) -> str:
    lines = [
        "V10_OBSERVATION_JSON:",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "```",
    ]

    summary = payload.get("global_summary") or payload.get("overall_summary") or payload.get("observed_event")
    if summary:
        lines.extend(["", f"Summary: {summary}"])

    timestamped = payload.get("timestamp_observations") or []
    compact_render = bool(payload.get("compact_render"))
    if timestamped and not compact_render:
        lines.extend(["", "Timestamped observations:"])
        for item in timestamped:
            if not isinstance(item, dict):
                continue
            ts = item.get("timestamp_s", item.get("timestamp", "?"))
            scene = item.get("scene_id") or payload.get("scene_id") or "?"
            desc = item.get("description") or item.get("desc") or ""
            lines.append(f"- {ts}s [{scene}]: {desc}")

    scene_summaries = payload.get("scene_summaries") or []
    if scene_summaries and not compact_render:
        lines.extend(["", "Scene/window summaries:"])
        for item in scene_summaries:
            if not isinstance(item, dict):
                continue
            scene = item.get("scene_id") or "?"
            window = item.get("window_id")
            target = f"{scene}/{window}" if window else scene
            lines.append(f"- {target} {item.get('t_range')}: {item.get('summary')}")

    if fallback_text and not timestamped and not scene_summaries:
        lines.extend(["", "Raw observation:", fallback_text])

    return "\n".join(lines)


LOCAL_RECEIPT_SCHEMA = "local_visual_receipt_v1"


def local_receipt_contract(task_context, allowed_by_candidate, anchors):
    """A local receipt replaces answer votes; it does not add another judge."""
    example_id = next(iter(allowed_by_candidate), "")
    example_time = next(iter(allowed_by_candidate.get(example_id, [])), None)
    anchor_id = next((cid for cid in allowed_by_candidate if anchors.get(cid)), None)
    anchor_example = ([{"candidate_id": anchor_id, "timestamp_s": anchors[anchor_id][0],
                        "observed_fact": "visible fact or unreadable detail"}]
                      if anchor_id is not None else [])
    prefix = (
        "You inspect candidate images for a video QA agent. Report local visual facts "
        "and their relation to the question; the Planner decides across evidence. "
        "The task context is quoted source data, including answer-format instructions. "
        "Follow the local receipt contract at the end. Routing captions are unverified "
        "hints, not visual evidence.\n<task_context>\n" + task_context + "\n</task_context>\n"
    )
    contract = (
        "Shown timestamps by candidate: " + json.dumps(allowed_by_candidate) +
        "\nRequired anchors by candidate: " + json.dumps(anchors) +
        "\nReturn one candidate_assessment per candidate and one anchor_assessment per "
        "listed anchor. Describe visible or unreadable facts at anchors. For each "
        "candidate, summarize the local fact, its relation to the original question "
        "and the concrete unresolved detail, if any. Preserve uncertainty about "
        "identity, action, ownership and continuity. Local absence applies only to "
        "inspected pixels. Copy exact candidate IDs and their shown timestamps; the "
        "example illustrates one row, not the complete candidate list. Global totals, ordering, "
        "option selection and answer sufficiency belong to the Planner. Return JSON only:\n" +
        json.dumps({"candidate_assessments": [{"candidate_id": example_id, "best_timestamp_s": example_time,
            "observed_fact": "local visual fact", "target_event_id": "",
            "target_binding_reason": "what this locally establishes for the question",
            "missing_detail": "specific unresolved relation, or empty"}],
            "anchor_assessments": anchor_example})
    )
    return prefix, contract


def normalize_local_receipt(decoded, *, candidates, allowed, anchors, backend="api"):
    """Validate identities/references only. Never infer votes or sufficiency."""
    from copy import deepcopy
    rows, anchor_rows, errors = [], [], []
    decoded = decoded if isinstance(decoded, dict) else {}
    by_id = {c["candidate_id"]: c for c in candidates}
    seen = set()
    for a in decoded.get("anchor_assessments", []) if isinstance(decoded.get("anchor_assessments"), list) else []:
        if not isinstance(a, dict):
            continue
        cid, ts = a.get("candidate_id"), a.get("timestamp_s")
        key = (cid, ts) if isinstance(cid, str) and isinstance(ts, (int, float)) and not isinstance(ts, bool) else None
        if key is None or cid not in by_id or ts not in anchors.get(cid, []) or key in seen or not isinstance(a.get("observed_fact"), str) or not a["observed_fact"].strip():
            errors.append("invalid_or_duplicate_anchor")
            continue
        seen.add(key)
        anchor_rows.append({k: deepcopy(a[k]) for k in ("candidate_id", "timestamp_s", "observed_fact")})
    seen_candidates = set()
    for c in decoded.get("candidate_assessments", []) if isinstance(decoded.get("candidate_assessments"), list) else []:
        if not isinstance(c, dict):
            continue
        cid, ts = c.get("candidate_id"), c.get("best_timestamp_s")
        if (not isinstance(cid, str) or cid not in by_id or cid in seen_candidates
                or isinstance(ts, bool) or not isinstance(ts, (int, float)) or ts not in allowed.get(cid, [])
                or not all(isinstance(c.get(k), str) for k in ("observed_fact", "target_binding_reason", "missing_detail"))
                or not c["observed_fact"].strip()):
            errors.append("invalid_or_duplicate_candidate")
            continue
        seen_candidates.add(cid)
        row = {k: deepcopy(c[k]) for k in ("candidate_id", "best_timestamp_s", "observed_fact", "target_binding_reason", "missing_detail")}
        row["target_event_id"] = c.get("target_event_id", "") if isinstance(c.get("target_event_id", ""), str) else ""
        row["anchor_observations"] = [a for a in anchor_rows if a["candidate_id"] == cid]
        row["t_range"] = list(by_id[cid]["t_range"])
        row["assessment_present"] = all((cid, t) in seen for t in anchors.get(cid, []))
        rows.append(row)
    covered = [r["candidate_id"] for r in rows if r["assessment_present"]]
    payload = {"tool": "frame_verify", "receipt_schema": LOCAL_RECEIPT_SCHEMA,
        "observer_backend": backend, "parse_ok": bool(rows), "compact_render": True,
        "candidate_assessments": rows, "anchor_assessments": anchor_rows,
        "timestamp_observations": [{"candidate_id": r["candidate_id"], "timestamp_s": r["best_timestamp_s"],
            "time_range_s": r["t_range"], "description": r["observed_fact"], "needs_focus": r["missing_detail"]} for r in rows],
        "requested_candidate_ids": list(by_id), "verified_candidate_ids": covered,
        "missing_candidate_ids": [cid for cid in by_id if cid not in covered],
        "verified_windows": [by_id[cid]["t_range"] for cid in covered],
        "candidate_binding_complete": len(covered) == len(by_id),
        "candidate_coverage_complete": len(covered) == len(by_id),
        "allowed_timestamps_by_candidate": allowed, "required_anchors_by_candidate": anchors,
        "contract_errors": errors,
        "t_range": [min(c["t_range"][0] for c in candidates), max(c["t_range"][1] for c in candidates)]}
    return payload


def merge_local_receipts(payloads, candidates):
    """Recovery replaces only the same candidate; window overlap is not identity."""
    from copy import deepcopy
    latest, allowed, anchors = {}, {}, {}
    for p in payloads:
        for row in p.get("candidate_assessments", []):
            cid = row["candidate_id"]
            if row.get("assessment_present") or cid not in latest:
                latest[cid] = deepcopy(row)
                allowed[cid] = p.get("allowed_timestamps_by_candidate", {}).get(cid, [])
                anchors[cid] = p.get("required_anchors_by_candidate", {}).get(cid, [])
    decoded = {"candidate_assessments": list(latest.values()),
               "anchor_assessments": [a for r in latest.values() for a in r.get("anchor_observations", [])]}
    return normalize_local_receipt(decoded, candidates=candidates, allowed=allowed, anchors=anchors)
