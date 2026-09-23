#!/usr/bin/env python3
"""Score the preregistered P99 fixed-state causal Planner A/B."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


P99_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = P99_ROOT / "offline" / "p99_fixed_state_manifest.json"
DEFAULT_CALLS = P99_ROOT / "phase2" / "p99_fixed_state_calls.jsonl"
DEFAULT_JSON = P99_ROOT / "phase2" / "p99_fixed_state_analysis.json"
DEFAULT_MD = P99_ROOT / "phase2" / "p99_fixed_state_analysis.md"
ALLOWED_TOOLS = {"overview", "localize_qwen", "frame_verify", "answer"}
EVIDENCE_RE = re.compile(r"\bE\d{5}\b")
ID_RE = re.compile(r"\b(?:EV-[A-Z0-9-]+|C[A-F0-9]{7})\b")


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(stripped)):
        char = stripped[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(stripped[start : idx + 1])
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
    return None


def _load_calls(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            call_id = str(record.get("call_id") or "")
            if not call_id or call_id in seen:
                raise RuntimeError(f"invalid or duplicate call at line {line_no}: {call_id!r}")
            seen.add(call_id)
            records.append(record)
    return records


def _action(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    action = parsed.get("action")
    return action if isinstance(action, dict) else None


def _parameters(action: dict[str, Any]) -> dict[str, Any]:
    parameters = action.get("parameters")
    return parameters if isinstance(parameters, dict) else {}


def _valid_action(parsed: dict[str, Any] | None) -> tuple[bool, str]:
    action = _action(parsed)
    if action is None:
        return False, "missing_action"
    tool = action.get("tool")
    if tool not in ALLOWED_TOOLS:
        return False, "invalid_tool"
    params = _parameters(action)
    if tool == "overview":
        return (isinstance(action.get("parameters"), dict), "overview_parameters")
    if tool == "answer":
        answer = str(action.get("answer") or "").strip().upper()
        refs = action.get("support_refs")
        return (answer in {"A", "B", "C", "D"} and isinstance(refs, list), "answer_schema")
    if tool == "frame_verify":
        required = {"query", "start_time", "end_time", "mode"}
        valid = required.issubset(params) and _finite_number(params.get("start_time")) and _finite_number(params.get("end_time"))
        return valid, "frame_verify_schema"
    required = {"search_windows", "localization_goal", "evidence_profile", "top_k"}
    valid = required.issubset(params) and isinstance(params.get("search_windows"), list)
    return valid, "localize_qwen_schema"


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _referenced_ids(action: dict[str, Any] | None) -> tuple[set[str], set[str]]:
    if not action:
        return set(), set()
    text = "\n".join(_walk_strings(action))
    evidence = set(EVIDENCE_RE.findall(text))
    identities = set(ID_RE.findall(text))
    return evidence, identities


def _interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if not _finite_number(value[0]) or not _finite_number(value[1]):
        return None
    start, end = float(value[0]), float(value[1])
    return (min(start, end), max(start, end))


def _selected_ranges(action: dict[str, Any] | None) -> list[tuple[float, float]]:
    if not action:
        return []
    tool = action.get("tool")
    params = _parameters(action)
    ranges: list[tuple[float, float]] = []
    if tool == "frame_verify":
        item = _interval([params.get("start_time"), params.get("end_time")])
        if item:
            ranges.append(item)
    elif tool == "localize_qwen":
        for raw in params.get("search_windows") or []:
            item = _interval(raw)
            if item:
                ranges.append(item)
    return ranges


def _contained(inner: tuple[float, float], outer: tuple[float, float], tolerance: float = 0.25) -> bool:
    return inner[0] >= outer[0] - tolerance and inner[1] <= outer[1] + tolerance


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _solved_repeat(state: dict[str, Any], action: dict[str, Any] | None) -> bool:
    if not action or action.get("tool") in {None, "answer", "overview"}:
        return False
    _, identities = _referenced_ids(action)
    constraints = state.get("capsule_constraints") or {}
    inspected = set(constraints.get("already_inspected_candidate_ids") or []) | set(
        constraints.get("already_inspected_event_ids") or []
    )
    ranges = _selected_ranges(action)
    if identities & inspected and not ranges:
        return True
    no_new = [item for raw in constraints.get("no_new_pixel_windows") or [] if (item := _interval(raw))]
    return bool(ranges) and bool(no_new) and all(any(_contained(item, old) for old in no_new) for item in ranges)


def _semantic_signature(action: dict[str, Any] | None) -> str:
    if not action:
        return "INVALID"
    tool = str(action.get("tool") or "INVALID")
    if tool == "answer":
        return f"answer:{str(action.get('answer') or '').strip().upper()}"
    if tool == "overview":
        return "overview"
    _, identities = _referenced_ids(action)
    if identities:
        return f"{tool}:ids=" + ",".join(sorted(identities))
    ranges = _selected_ranges(action)
    bins = [(int(math.floor(start / 10.0)), int(math.floor(end / 10.0))) for start, end in ranges]
    return f"{tool}:bins={bins}"


def _exact_signature(action: dict[str, Any] | None) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if action else "INVALID"


def _obligation_equivalent(state: dict[str, Any], action: dict[str, Any] | None) -> bool:
    if not action:
        return False
    if action.get("tool") == "answer":
        return True
    _, identities = _referenced_ids(action)
    ranges = _selected_ranges(action)
    for obligation in state.get("capsule_unresolved_obligations") or []:
        candidate_events = set(obligation.get("candidate_event_ids") or [])
        if identities & candidate_events:
            return True
        scope = _interval(obligation.get("required_scope"))
        if scope and any(_overlap(scope, item) > 0 for item in ranges):
            return True
    return False


def _score_call(record: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    status_ok = record.get("status") == "ok"
    parsed = extract_json_object(record.get("content")) if status_ok else None
    action = _action(parsed)
    valid, schema_reason = _valid_action(parsed)
    evidence_refs, _ = _referenced_ids(action)
    visible = set(state.get("valid_evidence_ids") or [])
    if record.get("arm") == "treatment":
        visible = set(state.get("included_evidence_ids") or [])
    verified = set(state.get("verified_evidence_ids") or [])
    unseen_refs = sorted(evidence_refs - visible)
    answer_refs = set(action.get("support_refs") or []) if action and action.get("tool") == "answer" else set()
    level_invalid_refs = sorted(answer_refs - verified)
    answer = str(action.get("answer") or "").strip().upper() if action and action.get("tool") == "answer" else None
    answer_correct = bool(valid and answer == str(state.get("ground_truth") or "").upper())
    usage = record.get("usage") or {}
    return {
        **record,
        "parsed": parsed,
        "valid_action": bool(status_ok and valid),
        "schema_reason": schema_reason,
        "tool": action.get("tool") if action else None,
        "answer": answer,
        "answer_correct": answer_correct,
        "evidence_refs": sorted(evidence_refs),
        "unseen_evidence_refs": unseen_refs,
        "level_invalid_answer_refs": level_invalid_refs,
        "has_invalid_reference": bool(unseen_refs or level_invalid_refs),
        "solved_event_repeat": _solved_repeat(state, action),
        "obligation_equivalent": _obligation_equivalent(state, action),
        "semantic_signature": _semantic_signature(action),
        "exact_signature_sha256": hashlib.sha256(_exact_signature(action).encode("utf-8")).hexdigest(),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
    }


def _arm_summary(scored: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    rows = [row for row in scored if row["arm"] == arm]
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[row["state_id"]].append(row)
    invalid_states = sorted({row["state_id"] for row in rows if row["has_invalid_reference"]})
    repeat_states = sorted({row["state_id"] for row in rows if row["solved_event_repeat"]})
    disagreement_states = sorted(
        state_id
        for state_id, items in by_state.items()
        if len({item["semantic_signature"] for item in items}) > 1
    )
    exact_disagreement_states = sorted(
        state_id
        for state_id, items in by_state.items()
        if len({item["exact_signature_sha256"] for item in items}) > 1
    )
    answer_rows = [row for row in rows if row["category"] == "answer_ready"]
    return {
        "calls": len(rows),
        "ok_calls": sum(row["status"] == "ok" for row in rows),
        "valid_actions": sum(row["valid_action"] for row in rows),
        "valid_action_rate": round(sum(row["valid_action"] for row in rows) / len(rows), 6) if rows else 0.0,
        "answer_ready_calls": len(answer_rows),
        "answer_ready_correct": sum(row["answer_correct"] for row in answer_rows),
        "answer_ready_accuracy": round(sum(row["answer_correct"] for row in answer_rows) / len(answer_rows), 6) if answer_rows else 0.0,
        "invalid_reference_calls": sum(row["has_invalid_reference"] for row in rows),
        "invalid_reference_states": invalid_states,
        "solved_repeat_calls": sum(row["solved_event_repeat"] for row in rows),
        "solved_repeat_states": repeat_states,
        "obligation_equivalent_calls": sum(row["obligation_equivalent"] for row in rows),
        "disagreement_states": disagreement_states,
        "disagreement_state_count": len(disagreement_states),
        "exact_disagreement_states": exact_disagreement_states,
        "exact_disagreement_state_count": len(exact_disagreement_states),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--calls", type=Path, default=DEFAULT_CALLS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    states = manifest.get("states") or []
    state_map = {state["state_id"]: state for state in states}
    records = _load_calls(args.calls)
    expected_ids = {
        f"{state['state_id']}_R{repeat}_{arm}"
        for state in states
        for repeat in range(1, 4)
        for arm in ("control", "treatment")
    }
    actual_ids = {record["call_id"] for record in records}
    scored = [_score_call(record, state_map[record["state_id"]]) for record in records]
    control = _arm_summary(scored, "control")
    treatment = _arm_summary(scored, "treatment")
    prompt_reduction = (
        1.0 - treatment["prompt_tokens"] / control["prompt_tokens"]
        if control["prompt_tokens"]
        else 0.0
    )
    control_invalid = set(control["invalid_reference_states"])
    treatment_invalid = set(treatment["invalid_reference_states"])
    control_repeats = set(control["solved_repeat_states"])
    treatment_repeats = set(treatment["solved_repeat_states"])
    gates = {
        "all_144_unique_calls_present": actual_ids == expected_ids and len(records) == 144,
        "zero_infrastructure_errors": all(row["status"] == "ok" for row in scored),
        "answer_ready_correct_not_lower": treatment["answer_ready_correct"] >= control["answer_ready_correct"],
        "no_new_invalid_reference_state": treatment_invalid.issubset(control_invalid),
        "no_new_solved_repeat_state": treatment_repeats.issubset(control_repeats),
        "treatment_disagreement_at_most_control_plus_one": treatment["disagreement_state_count"] <= control["disagreement_state_count"] + 1,
        "valid_action_rate_not_lower": treatment["valid_action_rate"] >= control["valid_action_rate"],
        "actual_prompt_reduction_at_least_25pct": prompt_reduction >= 0.25,
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "experiment": "P99 Phase 2 fixed-state causal Planner A/B",
        "manifest": str(args.manifest),
        "calls": str(args.calls),
        "expected_call_count": 144,
        "recorded_call_count": len(records),
        "missing_call_ids": sorted(expected_ids - actual_ids),
        "unexpected_call_ids": sorted(actual_ids - expected_ids),
        "control": control,
        "treatment": treatment,
        "actual_prompt_token_reduction": round(prompt_reduction, 6),
        "new_invalid_reference_states": sorted(treatment_invalid - control_invalid),
        "new_solved_repeat_states": sorted(treatment_repeats - control_repeats),
        "gates": gates,
        "calls_scored": scored,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P99 Phase-2 fixed-state causal Planner A/B",
        "",
        f"- Promotion gate: **{'PASS' if gates['all_pass'] else 'FAIL'}**.",
        f"- Calls: `{len(records)}/144`; infrastructure errors: `{sum(row['status'] != 'ok' for row in scored)}`.",
        f"- Actual Planner prompt tokens: Control `{control['prompt_tokens']:,}`, Treatment `{treatment['prompt_tokens']:,}` (`{prompt_reduction:.2%}` reduction).",
        f"- Valid actions: Control `{control['valid_actions']}/{control['calls']}`, Treatment `{treatment['valid_actions']}/{treatment['calls']}`.",
        f"- Answer-ready correctness: Control `{control['answer_ready_correct']}/{control['answer_ready_calls']}`, Treatment `{treatment['answer_ready_correct']}/{treatment['answer_ready_calls']}`.",
        f"- Semantic disagreement states: Control `{control['disagreement_state_count']}`, Treatment `{treatment['disagreement_state_count']}`.",
        f"- Invalid-reference states: Control `{control['invalid_reference_states']}`, Treatment `{treatment['invalid_reference_states']}`.",
        f"- Solved-repeat states: Control `{control['solved_repeat_states']}`, Treatment `{treatment['solved_repeat_states']}`.",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- `{name}`: **{value}**" for name, value in gates.items())
    lines.extend(["", "## Per-state arm signatures", "", "| state | category | C signatures | T signatures | C correct | T correct |", "|---|---|---:|---:|---:|---:|"])
    for state in states:
        state_rows = [row for row in scored if row["state_id"] == state["state_id"]]
        c_rows = [row for row in state_rows if row["arm"] == "control"]
        t_rows = [row for row in state_rows if row["arm"] == "treatment"]
        lines.append(
            f"| {state['state_id']} | {state['category']} | {len({row['semantic_signature'] for row in c_rows})} | "
            f"{len({row['semantic_signature'] for row in t_rows})} | {sum(row['answer_correct'] for row in c_rows)} | {sum(row['answer_correct'] for row in t_rows)} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:18]))
    return 0 if gates["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
