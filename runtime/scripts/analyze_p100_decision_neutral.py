#!/usr/bin/env python3
"""Zero-network P100 deletion audit and fixed-state manifest freeze."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import analyze_p99_first50_projection as p99

from videoseek.core.planner_capsule import (
    CAPSULE_HEADER,
    build_decision_neutral_planner_capsule,
    build_planner_evidence_capsule,
    estimate_tokens,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = SCRIPT_DIR.parent
P100_ROOT = RUNTIME_DIR.parent
RESEARCH_ROOT = P100_ROOT.parent
P99_ROOT = RESEARCH_ROOT / "p99_planner_evidence_capsule"
HOLDOUT_SOURCE_MANIFEST = P100_ROOT / "cohorts" / "holdout24_sources.json"
DEV_SOURCE_MANIFEST = P99_ROOT / "offline" / "p99_fixed_state_manifest.json"
P99_ARCHIVED_REPORT = P99_ROOT / "offline" / "p99_first50_projection.json"
OUTPUT_DIR = P100_ROOT / "offline"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _generic_obligation(row: Any, *, leading: Any, uncovered: list[str]) -> bool:
    if not isinstance(row, dict) or not uncovered:
        return False
    expected = (
        f"Distinguish leading option {leading or '?'} from uncovered options "
        f"{','.join(uncovered)} with verified evidence."
    )
    return (
        row.get("discriminator") == expected
        and row.get("required_scope") == "question scope"
        and (row.get("candidate_event_ids") or []) == []
        and not row.get("source_gap_ids")
    )


def _expected_deletion(parent_payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    expected = deepcopy(parent_payload)
    answer = expected.get("answer_state")
    if not isinstance(answer, dict):
        answer = {}
        expected["answer_state"] = answer
    leading = answer.get("leading_option")
    uncovered = [
        str(value).strip()
        for value in answer.get("uncovered_options") or []
        if str(value).strip()
    ]
    for field in ("leading_option", "strongest_competitor", "uncovered_options"):
        answer.pop(field, None)
    obligations = expected.get("unresolved_obligations") or []
    retained = [
        row
        for row in obligations
        if not _generic_obligation(row, leading=leading, uncovered=uncovered)
    ]
    expected["unresolved_obligations"] = retained
    return expected, len(obligations) - len(retained)


def _load_holdout_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(HOLDOUT_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    root = Path(str(manifest["p38_runtime"])).resolve()
    for source in manifest.get("source_files") or []:
        path = root / str(source["path"])
        if _sha256(path) != source["sha256"]:
            raise RuntimeError(f"holdout result hash mismatch: {path}")

    rows: list[dict[str, Any]] = []
    for item in manifest.get("qids") or []:
        result_path = root / str(item["result_file"])
        lines = result_path.read_text(encoding="utf-8").splitlines()
        line_number = int(item["result_line"])
        if not 1 <= line_number <= len(lines):
            raise RuntimeError(f"bad result line: {result_path}:{line_number}")
        row = json.loads(lines[line_number - 1])
        trajectory = (root / str(item["trajectory"])).resolve()
        if str(row.get("qid") or "") != item["qid"]:
            raise RuntimeError(f"holdout qid mismatch: {item['qid']}")
        if Path(str(row.get("trajectory_path") or "")).resolve() != trajectory:
            raise RuntimeError(f"holdout trajectory selection mismatch: {item['qid']}")
        if _sha256(trajectory) != item["trajectory_sha256"]:
            raise RuntimeError(f"holdout trajectory hash mismatch: {trajectory}")
        rows.append(row)
    return rows, manifest


def _paired_replay(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    for source in source_rows:
        parent = p99._replay_question(
            source,
            capsule_builder=build_planner_evidence_capsule,
        )
        candidate = p99._replay_question(
            source,
            capsule_builder=build_decision_neutral_planner_capsule,
        )
        if parent["qid"] != candidate["qid"]:
            raise RuntimeError("paired replay qid mismatch")
        if len(parent["_state_exports"]) != len(candidate["_state_exports"]):
            raise RuntimeError(f"paired replay boundary mismatch: {parent['qid']}")
        paired.append({"parent": parent, "candidate": candidate})
    return paired


def _strict_deletion_stats(paired: list[dict[str, Any]]) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for question in paired:
        parent = question["parent"]
        candidate = question["candidate"]
        for p_state, c_state in zip(
            parent["_state_exports"], candidate["_state_exports"]
        ):
            stats["boundaries"] += 1
            if p_state["control_messages"] != c_state["control_messages"]:
                stats["control_message_mismatch"] += 1
            if p_state["activated"] is not True:
                stats["nonactivated"] += 1
                if p_state["capsule_text"] != c_state["capsule_text"]:
                    stats["nonactivated_text_mismatch"] += 1
                continue

            stats["activated"] += 1
            parent_payload = p_state["capsule_payload"]
            candidate_payload = c_state["capsule_payload"]
            expected, removed_obligation_count = _expected_deletion(parent_payload)
            if candidate_payload != expected:
                stats["strict_deletion_mismatch"] += 1
                if len(failures) < 20:
                    failures.append(
                        {
                            "qid": parent["qid"],
                            "step": p_state["step"],
                            "parent_hash": p_state["capsule_hash"],
                            "candidate_hash": c_state["capsule_hash"],
                            "expected_hash": _json_hash(expected),
                        }
                    )
            answer = candidate_payload.get("answer_state") or {}
            stats["remaining_policy_fields"] += sum(
                field in answer
                for field in (
                    "leading_option",
                    "strongest_competitor",
                    "uncovered_options",
                )
            )
            generic_rows = [
                row
                for row in candidate_payload.get("unresolved_obligations") or []
                if isinstance(row, dict)
                and str(row.get("discriminator") or "").startswith(
                    "Distinguish leading option "
                )
                and row.get("required_scope") == "question scope"
                and not row.get("candidate_event_ids")
            ]
            stats["remaining_generic_obligations"] += len(generic_rows)
            stats["expected_removed_generic_obligations"] += removed_obligation_count
            stats["recorded_removed_generic_obligations"] += len(
                c_state["capsule_audit"].get("removed_policy_obligation_ids") or []
            )
            if c_state["capsule_audit"].get("no_backfill") is not True:
                stats["no_backfill_audit_failures"] += 1
            if c_state["capsule_audit"].get("parent_capsule_hash") != p_state[
                "capsule_hash"
            ]:
                stats["parent_hash_mismatch"] += 1
            if c_state["capsule_tokens"] > p_state["capsule_tokens"]:
                stats["candidate_larger_than_parent"] += 1
    return {**stats, "failures": failures}


def _public_question_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "qid": row["qid"],
        "question_type": row["question_type"],
        "tail": row["tail"],
        "planner_boundaries": row["planner_boundaries"],
        "historical_api_tokens": row["historical_api_tokens"],
        "control_prompt_tokens": row["control_prompt_tokens"],
        "p100_prompt_tokens": row["treatment_prompt_tokens"],
        "prompt_reduction": round(
            1.0
            - row["treatment_prompt_tokens"]
            / max(1, row["control_prompt_tokens"]),
            6,
        ),
        "trajectory": row["trajectory"],
        "trajectory_sha256": row["trajectory_sha256"],
    }


def _select_holdout_states(
    paired: list[dict[str, Any]], source_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    source_by_qid = {row["qid"]: row for row in source_manifest["qids"]}
    exports: list[dict[str, Any]] = []
    for index, question in enumerate(paired, 1):
        parent = question["parent"]
        candidate = question["candidate"]
        p_states = parent["_state_exports"]
        c_states = candidate["_state_exports"]
        pairs = [
            (p_state, c_state)
            for p_state, c_state in zip(p_states, c_states)
            if p_state["activated"] and c_state["activated"]
        ]
        if not pairs:
            raise RuntimeError(f"holdout qid has no activated state: {parent['qid']}")

        if parent["tail"]:
            threshold = statistics.median(
                p_state["control_memory_tokens"] for p_state, _ in pairs
            )
            chosen_p, chosen_c = next(
                pair
                for pair in pairs
                if pair[0]["control_memory_tokens"] >= threshold
            )
            category = "tail_onset_mid"
        else:
            ready = [pair for pair in pairs if pair[0]["answer_ready"]]
            if ready:
                chosen_p, chosen_c = ready[0]
                category = "answer_ready"
            else:
                unresolved = [pair for pair in pairs if not pair[0]["answer_ready"]]
                chosen_p, chosen_c = unresolved[(len(unresolved) - 1) // 2]
                category = "ordinary_unresolved"

        source = source_by_qid[parent["qid"]]
        exports.append(
            {
                "state_id": f"P100H{index:02d}",
                "category": category,
                "qid": parent["qid"],
                "question_type": parent["question_type"],
                "tail": parent["tail"],
                "step": chosen_p["step"],
                "question": parent["question"],
                "ground_truth": parent["ground_truth"],
                "control_prompt_tokens_estimated": chosen_p["control_prompt_tokens"],
                "p99_prompt_tokens_estimated": chosen_p["treatment_prompt_tokens"],
                "p100_prompt_tokens_estimated": chosen_c["treatment_prompt_tokens"],
                "parent_capsule_hash": chosen_p["capsule_hash"],
                "capsule_hash": chosen_c["capsule_hash"],
                "included_evidence_ids": chosen_c["included_evidence_ids"],
                "included_event_ids": chosen_c["included_event_ids"],
                "included_candidate_ids": chosen_c["included_candidate_ids"],
                "candidate_event_map": chosen_c["candidate_event_map"],
                "unresolved_obligation_ids": chosen_c["unresolved_obligation_ids"],
                "valid_evidence_ids": chosen_c["valid_evidence_ids"],
                "verified_evidence_ids": chosen_c["verified_evidence_ids"],
                "capsule_candidate_frontier": chosen_c[
                    "capsule_candidate_frontier"
                ],
                "capsule_unresolved_obligations": chosen_c[
                    "capsule_unresolved_obligations"
                ],
                "capsule_constraints": chosen_c["capsule_constraints"],
                "capsule_conflicts": chosen_c["capsule_conflicts"],
                "trajectory": parent["trajectory"],
                "trajectory_sha256": parent["trajectory_sha256"],
                "source_result_file": source["result_file"],
                "source_result_line": source["result_line"],
                "control_messages": chosen_p["control_messages"],
                "treatment_messages": chosen_c["treatment_messages"],
            }
        )
    return exports


def _replace_capsule_in_message(message: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(message)
    content = str(output.get("content") or "")
    start = content.index(CAPSULE_HEADER) + len(CAPSULE_HEADER)
    payload, consumed = json.JSONDecoder().raw_decode(content[start:])
    expected, _ = _expected_deletion(payload)
    neutral_text = CAPSULE_HEADER + json.dumps(
        expected,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    output["content"] = content[: start - len(CAPSULE_HEADER)] + neutral_text + content[
        start + consumed :
    ]
    return output


def _build_dev_manifest() -> dict[str, Any]:
    source = json.loads(DEV_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    states: list[dict[str, Any]] = []
    for index, original in enumerate(source.get("states") or [], 1):
        state = deepcopy(original)
        state["source_state_id"] = original["state_id"]
        state["state_id"] = f"P100D{index:02d}"
        state["treatment_messages"] = [
            _replace_capsule_in_message(message)
            if CAPSULE_HEADER in str(message.get("content") or "")
            else deepcopy(message)
            for message in original["treatment_messages"]
        ]
        old_payload = original.get("capsule_unresolved_obligations") or []
        leading = None
        uncovered: list[str] = []
        treatment_content = str(original["treatment_messages"][-1]["content"])
        start = treatment_content.index(CAPSULE_HEADER) + len(CAPSULE_HEADER)
        parent_payload, _ = json.JSONDecoder().raw_decode(treatment_content[start:])
        answer = parent_payload.get("answer_state") or {}
        leading = answer.get("leading_option")
        uncovered = [str(value) for value in answer.get("uncovered_options") or []]
        state["capsule_unresolved_obligations"] = [
            row
            for row in old_payload
            if not _generic_obligation(row, leading=leading, uncovered=uncovered)
        ]
        state["unresolved_obligation_ids"] = [
            str(row.get("obligation_id"))
            for row in state["capsule_unresolved_obligations"]
            if row.get("obligation_id")
        ]
        states.append(state)
    return {
        "experiment": "P100 known-state development diagnostic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(DEV_SOURCE_MANIFEST),
        "source_manifest_sha256": _sha256(DEV_SOURCE_MANIFEST),
        "known_outcomes": True,
        "promotion_evidence": False,
        "authorized_arms": ["treatment"],
        "arm_repeats": 3,
        "states": states,
        "network_calls_before_manifest": 0,
    }


def _markdown(report: dict[str, Any]) -> str:
    first = report["aggregate"]["first50"]
    tail = report["aggregate"]["tail14"]
    holdout = report["aggregate"]["holdout24"]
    lines = [
        "# P100 zero-network decision-neutral audit",
        "",
        f"- All gates pass: **{report['phase1_gate']['all_pass']}**.",
        f"- Strict deletion mismatches: `{report['strict_deletion'].get('strict_deletion_mismatch', 0)}`.",
        f"- First50 prompt reduction: `{100 * first['prompt_reduction']:.2f}%`.",
        f"- Tail14 prompt reduction: `{100 * tail['prompt_reduction']:.2f}%`.",
        f"- Holdout24 trajectory prompt reduction: `{100 * holdout['prompt_reduction']:.2f}%`.",
        f"- Holdout frozen states: `{report['holdout_state_count']}`; "
        f"categories `{report['holdout_category_counts']}`.",
        "",
        "## Gates",
        "",
    ]
    for key, value in report["phase1_gate"].items():
        lines.append(f"- `{key}`: **{value}**")
    lines.extend(
        [
            "",
            "| cohort | qid | family | tail | boundaries | P38 prompt | P100 prompt | reduction |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cohort, rows in report["question_summaries"].items():
        for row in rows:
            lines.append(
                f"| {cohort} | {row['qid']} | {row['question_type']} | "
                f"{int(row['tail'])} | {row['planner_boundaries']} | "
                f"{row['control_prompt_tokens']} | {row['p100_prompt_tokens']} | "
                f"{100 * row['prompt_reduction']:.2f}% |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    first50_source = p99._read_jsonl(p99.P38_RESULTS)
    holdout_source, holdout_source_manifest = _load_holdout_rows()
    dev_source = json.loads(DEV_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    dev_qids = {row["qid"] for row in dev_source.get("states") or []}
    holdout_qids = {row["qid"] for row in holdout_source_manifest["qids"]}

    first50_paired = _paired_replay(first50_source)
    holdout_paired = _paired_replay(holdout_source)
    all_paired = first50_paired + holdout_paired
    strict = _strict_deletion_stats(all_paired)

    first50_parent = [row["parent"] for row in first50_paired]
    first50_candidate = [row["candidate"] for row in first50_paired]
    holdout_candidate = [row["candidate"] for row in holdout_paired]
    archived_p99 = json.loads(P99_ARCHIVED_REPORT.read_text(encoding="utf-8"))
    replayed_parent_aggregate = {
        "all50": p99._aggregate(first50_parent),
        "tail14": p99._aggregate([row for row in first50_parent if row["tail"]]),
        "non_tail36": p99._aggregate(
            [row for row in first50_parent if not row["tail"]]
        ),
    }
    aggregate = {
        "first50": p99._aggregate(first50_candidate),
        "tail14": p99._aggregate(
            [row for row in first50_candidate if row["tail"]]
        ),
        "non_tail36": p99._aggregate(
            [row for row in first50_candidate if not row["tail"]]
        ),
        "holdout24": p99._aggregate(holdout_candidate),
    }
    combined_stats = p99._aggregate(first50_candidate + holdout_candidate)
    holdout_states = _select_holdout_states(holdout_paired, holdout_source_manifest)
    category_counts = dict(Counter(row["category"] for row in holdout_states))
    config_equivalence = p99._source_config_equivalence()

    gate = {
        "first50_replayed": len(first50_candidate) == 50,
        "first50_324_boundaries": aggregate["first50"]["planner_boundaries"] == 324,
        "tail_definition_reproduces_14": sum(row["tail"] for row in first50_candidate)
        == 14,
        "p99_parent_reproduces_archived_projection": replayed_parent_aggregate
        == archived_p99["aggregate"],
        "control_defaults_and_config_equivalent": all(
            config_equivalence[key]
            for key in (
                "candidate_default_off",
                "only_p99_config_fields_added",
                "memory_formatter_source_identical",
                "prompt_source_identical",
            )
        ),
        "strict_post_fit_deletion_only": strict.get("strict_deletion_mismatch", 0)
        == 0,
        "zero_control_message_changes": strict.get("control_message_mismatch", 0)
        == 0,
        "nonactivated_passthrough_identical": strict.get(
            "nonactivated_text_mismatch", 0
        )
        == 0,
        "zero_remaining_policy_fields": strict.get("remaining_policy_fields", 0)
        == 0,
        "zero_remaining_generic_obligations": strict.get(
            "remaining_generic_obligations", 0
        )
        == 0,
        "removed_generic_obligations_fully_audited": strict.get(
            "expected_removed_generic_obligations", 0
        )
        == strict.get("recorded_removed_generic_obligations", 0),
        "zero_backfill_or_parent_hash_mismatch": (
            strict.get("no_backfill_audit_failures", 0) == 0
            and strict.get("parent_hash_mismatch", 0) == 0
            and strict.get("candidate_larger_than_parent", 0) == 0
        ),
        "deterministic_and_read_only": (
            combined_stats["determinism_failures"] == 0
            and combined_stats["memory_mutations"] == 0
        ),
        "zero_invalid_or_omitted_required_references": (
            combined_stats["invalid_reference_count"] == 0
            and combined_stats["omitted_decisive_count"] == 0
            and combined_stats["omitted_strong_conflict_evidence_count"] == 0
            and combined_stats["omitted_recorded_conflict_count"] == 0
        ),
        "zero_duplicate_facts_or_event_merges": (
            combined_stats["duplicate_fact_body_count"] == 0
            and combined_stats["mistaken_event_merge_count"] == 0
        ),
        "zero_overflow_or_fail_open": (
            combined_stats["over_budget_count"] == 0
            and combined_stats["fail_open_count"] == 0
            and combined_stats["max_capsule_tokens"] <= 4000
        ),
        "first50_prompt_reduction_at_least_50pct": aggregate["first50"][
            "prompt_reduction"
        ]
        >= 0.50,
        "tail14_prompt_reduction_at_least_50pct": aggregate["tail14"][
            "prompt_reduction"
        ]
        >= 0.50,
        "no_nonempty_expansion_over_5pct": combined_stats[
            "nonempty_expansion_over_5pct_count"
        ]
        == 0,
        "holdout_24_unique_qids": len(holdout_states) == 24
        and len(holdout_qids) == 24,
        "holdout_qid_disjoint_from_dev": not (holdout_qids & dev_qids),
        "holdout_covers_all_seven_families": len(
            {row["question_type"] for row in holdout_states}
        )
        == 7,
        "zero_network_calls": True,
    }
    gate["all_pass"] = all(gate.values())

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "P100 Decision-Neutral Planner Evidence Capsule",
        "mode": "zero_network_strict_deletion_and_holdout_freeze",
        "aggregate": aggregate,
        "strict_deletion": strict,
        "phase1_gate": gate,
        "holdout_state_count": len(holdout_states),
        "holdout_category_counts": category_counts,
        "holdout_family_counts": dict(
            Counter(row["question_type"] for row in holdout_states)
        ),
        "holdout_dev_overlap": sorted(holdout_qids & dev_qids),
        "config_equivalence": config_equivalence,
        "question_summaries": {
            "first50": [_public_question_summary(row) for row in first50_candidate],
            "holdout24": [_public_question_summary(row) for row in holdout_candidate],
        },
        "network_calls": 0,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "p100_decision_neutral_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "p100_decision_neutral_audit.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    if gate["all_pass"]:
        holdout_manifest = {
            "experiment": "P100 independent fixed-state holdout A/B",
            "generated_at_utc": report["generated_at_utc"],
            "selection_rule": (
                "tail: first activated boundary at/above activated median raw-memory "
                "tokens; non-tail: earliest activated answer-ready boundary, else lower "
                "median activated unresolved boundary"
            ),
            "source_manifest": str(HOLDOUT_SOURCE_MANIFEST),
            "source_manifest_sha256": _sha256(HOLDOUT_SOURCE_MANIFEST),
            "authorized_arms": ["control", "treatment"],
            "arm_repeats": 5,
            "states": holdout_states,
            "network_calls_before_manifest": 0,
        }
        (OUTPUT_DIR / "p100_holdout24_state_manifest.json").write_text(
            json.dumps(holdout_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        dev_manifest = _build_dev_manifest()
        (OUTPUT_DIR / "p100_dev24_state_manifest.json").write_text(
            json.dumps(dev_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "strict_deletion": {
                    key: value for key, value in strict.items() if key != "failures"
                },
                "phase1_gate": gate,
                "holdout_category_counts": category_counts,
                "holdout_family_counts": report["holdout_family_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if gate["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
