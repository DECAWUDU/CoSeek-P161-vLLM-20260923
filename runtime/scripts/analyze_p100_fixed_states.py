#!/usr/bin/env python3
"""Score P100 development diagnostics or independent fixed-state A/B."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import analyze_p99_fixed_state_ab as p99_score


P100_ROOT = Path(__file__).resolve().parents[2]
P99_ROOT = P100_ROOT.parent / "p99_planner_evidence_capsule"


def _load_calls(path: Path) -> list[dict[str, Any]]:
    return p99_score._load_calls(path)


def _call_id(state_id: str, repeat: int, arm: str) -> str:
    return f"{state_id}_R{repeat}_{arm}"


def _majority_correct_states(
    scored: list[dict[str, Any]], arm: str, eligible: set[str]
) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for row in scored:
        if row["arm"] == arm and row["state_id"] in eligible and row["answer_correct"]:
            counts[row["state_id"]] += 1
    full = {state_id: counts.get(state_id, 0) for state_id in sorted(eligible)}
    return sum(value >= 3 for value in full.values()), full


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['experiment']}",
        "",
        f"- All gates pass: **{report['gates']['all_pass']}**.",
        f"- Calls: `{report['recorded_call_count']}/{report['expected_call_count']}`.",
    ]
    if report["mode"] == "dev":
        treatment = report["p100_treatment"]
        archived = report["archived_block4"]
        lines.extend(
            [
                f"- Answer-ready correct: P100 `{treatment['answer_ready_correct']}/"
                f"{treatment['answer_ready_calls']}`, archived Control "
                f"`{archived['control']['answer_ready_correct']}/"
                f"{archived['control']['answer_ready_calls']}`, archived P99 "
                f"`{archived['p99']['answer_ready_correct']}/"
                f"{archived['p99']['answer_ready_calls']}`.",
                f"- Actual prompt reduction vs archived Control: "
                f"`{100 * report['prompt_reduction_vs_archived_control']:.2f}%`.",
                f"- Combined reduction vs archived Control: "
                f"`{100 * report['combined_reduction_vs_archived_control']:.2f}%`.",
            ]
        )
    else:
        control = report["control"]
        treatment = report["treatment"]
        lines.extend(
            [
                f"- Answer-ready correct calls: Control "
                f"`{control['answer_ready_correct']}/{control['answer_ready_calls']}`, "
                f"P100 `{treatment['answer_ready_correct']}/"
                f"{treatment['answer_ready_calls']}`.",
                f"- Correct-majority states: Control "
                f"`{report['control_majority_correct_states']}`, P100 "
                f"`{report['treatment_majority_correct_states']}`.",
                f"- Actual prompt reduction: "
                f"`{100 * report['actual_prompt_reduction']:.2f}%`; combined "
                f"`{100 * report['actual_combined_reduction']:.2f}%`.",
            ]
        )
    lines.extend(["", "## Gates", ""])
    for key, value in report["gates"].items():
        lines.append(f"- `{key}`: **{value}**")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dev", "holdout"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    states = manifest.get("states") or []
    repeats = int(manifest.get("arm_repeats") or 0)
    arms = tuple(manifest.get("authorized_arms") or [])
    state_by_id = {str(row["state_id"]): row for row in states}
    expected = {
        _call_id(str(state["state_id"]), repeat, arm)
        for state in states
        for repeat in range(1, repeats + 1)
        for arm in arms
    }
    calls = _load_calls(args.calls)
    recorded = {str(row.get("call_id") or "") for row in calls}
    scored = [
        p99_score._score_call(row, state_by_id[str(row["state_id"])])
        for row in calls
        if str(row.get("state_id") or "") in state_by_id
    ]

    base: dict[str, Any] = {
        "mode": args.mode,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": p99_score.hashlib.sha256(
            args.manifest.read_bytes()
        ).hexdigest(),
        "calls": str(args.calls.resolve()),
        "expected_call_count": len(expected),
        "recorded_call_count": len(recorded),
        "missing_call_ids": sorted(expected - recorded),
        "unexpected_call_ids": sorted(recorded - expected),
    }

    if args.mode == "dev":
        archived_path = (
            P99_ROOT
            / "phase2"
            / "backup_block4"
            / "p99_fixed_state_analysis.json"
        )
        archived = json.loads(archived_path.read_text(encoding="utf-8"))
        treatment = p99_score._arm_summary(scored, "treatment")
        control_ref = archived["control"]
        p99_ref = archived["treatment"]
        prompt_reduction = 1.0 - treatment["prompt_tokens"] / max(
            1, control_ref["prompt_tokens"]
        )
        combined = treatment["prompt_tokens"] + treatment["completion_tokens"]
        control_combined = control_ref["prompt_tokens"] + control_ref[
            "completion_tokens"
        ]
        combined_reduction = 1.0 - combined / max(1, control_combined)
        gates = {
            "all_72_unique_calls_present": recorded == expected and len(expected) == 72,
            "zero_infrastructure_errors": treatment["ok_calls"] == 72,
            "all_actions_valid": treatment["valid_actions"] == 72,
            "zero_invalid_references": treatment["invalid_reference_calls"] == 0,
            "solved_repeat_states_at_most_archived_p99": len(
                treatment["solved_repeat_states"]
            )
            <= len(p99_ref["solved_repeat_states"]),
            "prompt_reduction_at_least_50pct": prompt_reduction >= 0.50,
            "combined_reduction_at_least_40pct": combined_reduction >= 0.40,
        }
        gates["all_pass"] = all(gates.values())
        report = {
            **base,
            "experiment": "P100 known-state development diagnostic",
            "p100_treatment": treatment,
            "archived_block4": {"control": control_ref, "p99": p99_ref},
            "prompt_reduction_vs_archived_control": round(prompt_reduction, 6),
            "combined_reduction_vs_archived_control": round(
                combined_reduction, 6
            ),
            "gates": gates,
            "calls_scored": scored,
        }
    else:
        control = p99_score._arm_summary(scored, "control")
        treatment = p99_score._arm_summary(scored, "treatment")
        eligible = {
            str(state["state_id"])
            for state in states
            if state.get("category") == "answer_ready"
        }
        control_majority, control_counts = _majority_correct_states(
            scored, "control", eligible
        )
        treatment_majority, treatment_counts = _majority_correct_states(
            scored, "treatment", eligible
        )
        severe = [
            state_id
            for state_id in sorted(eligible)
            if control_counts[state_id] >= 4 and treatment_counts[state_id] <= 1
        ]
        prompt_reduction = 1.0 - treatment["prompt_tokens"] / max(
            1, control["prompt_tokens"]
        )
        treatment_combined = treatment["prompt_tokens"] + treatment[
            "completion_tokens"
        ]
        control_combined = control["prompt_tokens"] + control["completion_tokens"]
        combined_reduction = 1.0 - treatment_combined / max(1, control_combined)
        gates = {
            "all_240_unique_calls_present": recorded == expected
            and len(expected) == 240,
            "zero_infrastructure_errors": control["ok_calls"] == 120
            and treatment["ok_calls"] == 120,
            "valid_action_rate_not_lower": treatment["valid_action_rate"]
            >= control["valid_action_rate"],
            "answer_ready_correct_calls_not_lower": treatment[
                "answer_ready_correct"
            ]
            >= control["answer_ready_correct"],
            "majority_correct_states_not_lower": treatment_majority
            >= control_majority,
            "zero_severe_accuracy_regressions": not severe,
            "invalid_reference_calls_and_states_not_higher": (
                treatment["invalid_reference_calls"]
                <= control["invalid_reference_calls"]
                and len(treatment["invalid_reference_states"])
                <= len(control["invalid_reference_states"])
            ),
            "solved_repeat_calls_and_states_not_higher": (
                treatment["solved_repeat_calls"] <= control["solved_repeat_calls"]
                and len(treatment["solved_repeat_states"])
                <= len(control["solved_repeat_states"])
            ),
            "semantic_disagreement_at_most_control_plus_one": treatment[
                "disagreement_state_count"
            ]
            <= control["disagreement_state_count"] + 1,
            "prompt_reduction_at_least_50pct": prompt_reduction >= 0.50,
            "combined_reduction_at_least_40pct": combined_reduction >= 0.40,
        }
        gates["all_pass"] = all(gates.values())
        report = {
            **base,
            "experiment": "P100 independent fixed-state repeated A/B",
            "control": control,
            "treatment": treatment,
            "answer_ready_state_ids": sorted(eligible),
            "control_correct_repeats_by_state": control_counts,
            "treatment_correct_repeats_by_state": treatment_counts,
            "control_majority_correct_states": control_majority,
            "treatment_majority_correct_states": treatment_majority,
            "severe_accuracy_regression_states": severe,
            "actual_prompt_reduction": round(prompt_reduction, 6),
            "actual_combined_reduction": round(combined_reduction, 6),
            "gates": gates,
            "calls_scored": scored,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "calls_scored"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["gates"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

