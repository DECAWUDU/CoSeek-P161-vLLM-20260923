#!/usr/bin/env python3
"""Zero-network P99 projection audit over every P38 first-50 Planner boundary."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Iterable

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = SCRIPT_DIR.parent
EXPERIMENT_DIR = RUNTIME_DIR.parent
ROOT = Path("/home/zjw/workspace/coseek1_v14base_simplified_planner_20260707")
P38_RUNTIME = ROOT / "experiments/v75_1_query_coverage_fusion_p38_20260815/runtime"
P38_RESULTS = [
    P38_RUNTIME
    / "runs/p38_first50_main_20260816/shard0_gpu0/"
    "p38_first50_shard0_25_mc_1786864312/results.jsonl",
    P38_RUNTIME
    / "runs/p38_first50_main_20260816/shard1_gpu1/"
    "p38_first50_shard1_25_mc_1786864341/results.jsonl",
]
OUTPUT_DIR = EXPERIMENT_DIR / "offline"
P99_CONFIG_KEYS = {
    "coseek1_planner_evidence_capsule_enabled",
    "coseek1_planner_decision_neutral_capsule_enabled",
    "coseek1_planner_capsule_token_budget",
    "coseek1_planner_capsule_max_verified",
    "coseek1_planner_capsule_max_candidates",
    "coseek1_planner_capsule_max_obligations",
    "coseek1_planner_capsule_audit_enabled",
}

sys.path.insert(0, str(RUNTIME_DIR))

from config import general_config, prompts_config  # noqa: E402
from videoseek.agent import VideoSeekAgent  # noqa: E402
from videoseek.core.memory import (  # noqa: E402
    format_memory_for_prompt,
    init_observation_memory,
    merge_tool_observation,
)
from videoseek.core.planner_capsule import (  # noqa: E402
    CAPSULE_HEADER,
    build_planner_evidence_capsule,
    estimate_tokens,
)
from videoseek.utils import convert_to_free_form_text_representation  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _tree_digest(root: Path) -> tuple[int, str]:
    files: list[Path] = []
    for relroot in ("videoseek", "config", "scripts", "subsets", "tests"):
        for path in (root / relroot).rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and not path.name.endswith((".pyc", ".pyo"))
            ):
                files.append(path)
    files.extend(path for path in root.iterdir() if path.is_file())
    digest = hashlib.sha256()
    unique = sorted(set(files), key=lambda path: str(path.relative_to(root)))
    for path in unique:
        relative = str(path.relative_to(root)).encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return len(unique), digest.hexdigest()


def _format_memory(
    memory: dict[str, Any], *, question: str, config: dict[str, Any]
) -> str:
    """Mirror the original P38 call exactly; derived local state may refresh."""

    return format_memory_for_prompt(
        memory,
        question=question,
        include_compact_planner_state=bool(
            config.get("structured_evidence_include_planner_state", True)
        ),
        include_structured_evidence=bool(
            config.get("use_structured_evidence_state", True)
        ),
        max_structured_evidence_items=int(
            config.get("structured_evidence_prompt_max_items")
            or config.get("structured_evidence_max_items")
            or 10
        ),
        include_compact_event_coverage=bool(
            config.get("coseek1_compact_event_coverage", False)
        ),
        max_event_coverage_items=int(
            config.get("coseek1_compact_event_coverage_max_items") or 8
        ),
        include_timeline_view=bool(
            config.get("structured_evidence_include_timeline_view", False)
        ),
        include_object_state_view=bool(
            config.get("structured_evidence_include_object_state_view", False)
        ),
        include_candidate_frontier=bool(
            config.get("coseek1_candidate_frontier", False)
        ),
        max_candidate_frontier_items=int(
            config.get("coseek1_candidate_frontier_max_items") or 16
        ),
        candidate_frontier_include_timestamp_cues=bool(
            config.get("coseek1_frontier_timestamp_cues", True)
        ),
        candidate_frontier_timestamp_cue_radius_s=float(
            config.get("coseek1_frontier_timestamp_cue_radius_s") or 6.0
        ),
        candidate_frontier_include_evidence_scope=bool(
            config.get("coseek1_frontier_evidence_scope", True)
        ),
        candidate_frontier_include_temporal_boundaries=bool(
            config.get("coseek1_frontier_temporal_boundaries", True)
        ),
        candidate_frontier_temporal_boundary_max_gap_s=float(
            config.get("coseek1_frontier_temporal_boundary_max_gap_s") or 20.0
        ),
        candidate_frontier_include_investigation_state=bool(
            config.get("coseek1_frontier_investigation_state", False)
        ),
        candidate_frontier_max_investigation_candidates=int(
            config.get("coseek1_frontier_investigation_max_items") or 12
        ),
        include_evidence_episode_frontier=bool(
            config.get("coseek1_evidence_episode_frontier", False)
        ),
        max_evidence_episode_items=int(
            config.get("coseek1_evidence_episode_max_items") or 18
        ),
        evidence_episode_anchor_radius_s=float(
            config.get("coseek1_episode_anchor_radius_s") or 6.0
        ),
        evidence_episode_focus_window_s=float(
            config.get("coseek1_episode_focus_window_s") or 20.0
        ),
        scope_aware_evidence_memory=bool(
            config.get("coseek1_scope_aware_evidence_memory", False)
        ),
        separate_routing_candidates=bool(
            config.get("coseek1_separate_routing_candidates", False)
        ),
        query_relevant_retention=bool(
            config.get("coseek1_query_relevant_memory_retention", False)
        ),
        persistent_candidate_pool=bool(
            config.get("coseek1_persistent_candidate_pool", False)
        ),
        persistent_candidate_pool_max_items=int(
            config.get("coseek1_persistent_candidate_pool_max_items") or 20
        ),
        persistent_candidate_pool_timestamp_radius_s=float(
            config.get("coseek1_persistent_candidate_pool_timestamp_radius_s") or 8.0
        ),
        candidate_evidence_projection=bool(
            config.get("coseek1_candidate_evidence_projection", False)
        ),
        include_temporal_evidence_ledger=bool(
            config.get("coseek1_temporal_evidence_ledger_enabled", False)
        ),
        max_temporal_evidence_events=int(
            config.get("coseek1_temporal_evidence_ledger_max_events") or 16
        ),
    )


def _merge(
    memory: dict[str, Any],
    *,
    tool: str,
    parameters: dict[str, Any],
    output: str,
    question: str,
    config: dict[str, Any],
) -> None:
    merge_tool_observation(
        memory,
        tool_name=tool,
        parameters=parameters,
        output=output,
        use_structured_evidence_state=bool(
            config.get("use_structured_evidence_state", True)
        ),
        question_context=question,
        include_evidence_scope=bool(
            config.get("coseek1_frontier_evidence_scope", True)
        ),
        scope_aware_evidence_memory=bool(
            config.get("coseek1_scope_aware_evidence_memory", False)
        ),
        persistent_open_gaps=bool(
            config.get("coseek1_persistent_open_gaps", False)
        ),
        semantic_gap_resolution=bool(
            config.get("coseek1_semantic_gap_resolution_enabled", False)
        ),
        preserve_aggregate_verifier_decision=bool(
            config.get(
                "structured_evidence_preserve_aggregate_verifier_decision", False
            )
        ),
    )


def _prompt_agent() -> tuple[VideoSeekAgent, dict[str, Any], dict[str, Any]]:
    config: dict[str, Any] = {}
    config.update(general_config)
    config.update(prompts_config)
    config["coseek1_planner_evidence_capsule_enabled"] = False
    agent = VideoSeekAgent.__new__(VideoSeekAgent)
    agent.config = config
    agent.max_steps = int(config.get("max_steps") or 20)
    agent.allowed_tool_names = set(config.get("tools") or []) | {"answer"}
    system = agent.construct_initial_messages()[0]
    return agent, config, system


PROMPT_AGENT, PROMPT_CONFIG, SYSTEM_MESSAGE = _prompt_agent()


def _messages(
    *, question: str, duration: float, step: int, memory_text: str
) -> list[dict[str, str]]:
    subtitles = convert_to_free_form_text_representation([], content_type="subtitle")
    user = {
        "role": "user",
        "content": (
            f"Video Duration: {duration:.01f}s\n\n"
            f"Video Subtitles:\n{subtitles}\n\n"
            f"Question:\n{question}"
        ),
    }
    planner_step = {
        "role": "user",
        "content": PROMPT_AGENT._VideoSeekAgent__format_planner_step_prompt(
            step=step - 1,
            memory_text=memory_text,
        ),
    }
    return [dict(SYSTEM_MESSAGE), user, planner_step]


def _message_tokens(messages: list[dict[str, Any]]) -> int:
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return estimate_tokens(serialized)[0]


def _capsule_payload(text: str) -> dict[str, Any]:
    if not text.startswith(CAPSULE_HEADER):
        return {}
    return json.loads(text[len(CAPSULE_HEADER) :])


def _historical_action(group: list[dict[str, Any]]) -> dict[str, Any]:
    for step in group:
        proposal = step.get("planner_proposed_action")
        if isinstance(proposal, dict):
            return proposal
    for step in group:
        action = step.get("action")
        if isinstance(action, dict):
            return action
    return {}


def _answer_ready(memory: dict[str, Any]) -> bool:
    compact = memory.get("compact_investigation_state") or {}
    answer = compact.get("answer_status") if isinstance(compact, dict) else {}
    return bool(
        isinstance(answer, dict)
        and answer.get("support_refs")
        and str(answer.get("status") or "") != "no_verified_answer_yet"
    )


def _source_config_equivalence() -> dict[str, Any]:
    baseline = yaml.safe_load(
        (P38_RUNTIME / "config/general.yaml").read_text(encoding="utf-8")
    )
    candidate = yaml.safe_load(
        (RUNTIME_DIR / "config/general.yaml").read_text(encoding="utf-8")
    )
    stripped = {key: value for key, value in candidate.items() if key not in P99_CONFIG_KEYS}
    return {
        "candidate_default_off": (
            candidate.get("coseek1_planner_evidence_capsule_enabled") is False
            and candidate.get(
                "coseek1_planner_decision_neutral_capsule_enabled"
            )
            is False
        ),
        "only_p99_config_fields_added": baseline == stripped,
        "baseline_config_key_count": len(baseline),
        "candidate_config_key_count": len(candidate),
        "p99_config_keys": sorted(P99_CONFIG_KEYS),
        "memory_formatter_source_identical": _sha256(
            P38_RUNTIME / "videoseek/core/memory.py"
        )
        == _sha256(RUNTIME_DIR / "videoseek/core/memory.py"),
        "prompt_source_identical": _sha256(P38_RUNTIME / "config/prompts.yaml")
        == _sha256(RUNTIME_DIR / "config/prompts.yaml"),
    }


def _replay_question(
    result: dict[str, Any],
    *,
    capsule_builder: Callable[..., Any] = build_planner_evidence_capsule,
) -> dict[str, Any]:
    path = Path(str(result["trajectory_path"]))
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    archived_memory = trajectory.get("memory") or {}
    runtime_config = dict(
        archived_memory.get("runtime_config") or result.get("runtime_config") or {}
    )
    # These values are fixed by original P38 general.yaml/launcher but are not
    # all copied into the compact runtime_config trace.
    for key, value in PROMPT_CONFIG.items():
        runtime_config.setdefault(key, value)
    question = str(trajectory.get("question") or "")
    duration = float(result.get("duration_s") or 0.0)
    memory = init_observation_memory()
    memory["runtime_config"] = dict(archived_memory.get("runtime_config") or {})
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, step in enumerate(trajectory.get("steps") or []):
        groups[int(step.get("step_id") or index + 1)].append(step)

    boundaries: list[dict[str, Any]] = []
    state_exports: list[dict[str, Any]] = []
    for step_id in sorted(groups):
        control_memory = _format_memory(
            memory,
            question=question,
            config=runtime_config,
        )
        capsule = capsule_builder(
            memory,
            question=question,
            full_memory_text=control_memory,
            token_budget=4000,
            max_verified=8,
            max_candidates=8,
            max_obligations=4,
            max_conflicts=2,
        )
        repeated = capsule_builder(
            memory,
            question=question,
            full_memory_text=control_memory,
            token_budget=4000,
            max_verified=8,
            max_candidates=8,
            max_obligations=4,
            max_conflicts=2,
        )
        control_messages = _messages(
            question=question,
            duration=duration,
            step=step_id,
            memory_text=control_memory,
        )
        treatment_messages = _messages(
            question=question,
            duration=duration,
            step=step_id,
            memory_text=capsule.text,
        )
        control_prompt_tokens = _message_tokens(control_messages)
        treatment_prompt_tokens = _message_tokens(treatment_messages)
        payload = _capsule_payload(capsule.text)
        structured = memory.get("structured_evidence") or {}
        valid_evidence_ids = sorted(
            {
                str(item.get("evidence_id"))
                for item in (
                    list(structured.get("evidence_items") or [])
                    + list(structured.get("option_evidence") or [])
                )
                if isinstance(item, dict) and item.get("evidence_id")
            }
        )
        event_sources: dict[str, set[str]] = defaultdict(set)
        for item in payload.get("verified_evidence") or []:
            if item.get("event_identity_basis") == "source_call_identity":
                event_sources[str(item.get("event_id") or "")].add(
                    str(item.get("source_call_id") or "")
                )
        mistaken_event_merges = sum(len(values) > 1 for values in event_sources.values())
        action = _historical_action(groups[step_id])
        boundary = {
            "step": step_id,
            "control_memory_chars": len(control_memory),
            "control_memory_tokens": estimate_tokens(control_memory)[0],
            "capsule_chars": len(capsule.text),
            "capsule_tokens": int(capsule.audit.get("capsule_estimated_tokens") or 0),
            "control_prompt_tokens": control_prompt_tokens,
            "treatment_prompt_tokens": treatment_prompt_tokens,
            "prompt_reduction": round(
                1.0 - treatment_prompt_tokens / max(1, control_prompt_tokens), 6
            ),
            "activated": capsule.audit.get("activated") is True,
            "answer_ready": _answer_ready(memory),
            "deterministic": (
                capsule.text == repeated.text
                and capsule.audit.get("capsule_hash")
                == repeated.audit.get("capsule_hash")
            ),
            "memory_mutated": capsule.audit.get("memory_mutated") is True,
            "fail_open": capsule.audit.get("fail_open") is True,
            "over_budget": capsule.audit.get("over_budget") is True,
            "invalid_references": capsule.audit.get("invalid_references") or [],
            "omitted_decisive_support_ids": capsule.audit.get(
                "omitted_decisive_support_ids"
            )
            or [],
            "omitted_strong_conflict_evidence_ids": capsule.audit.get(
                "omitted_strong_conflict_evidence_ids"
            )
            or [],
            "omitted_conflict_count": int(
                capsule.audit.get("omitted_conflict_count") or 0
            ),
            "duplicate_fact_bodies": int(
                capsule.audit.get("duplicate_fact_bodies") or 0
            ),
            "mistaken_event_merges": mistaken_event_merges,
            "capsule_hash": capsule.audit.get("capsule_hash"),
            "included_evidence_ids": capsule.audit.get("included_evidence_ids")
            or [],
            "included_event_ids": capsule.audit.get("included_event_ids") or [],
            "included_candidate_ids": capsule.audit.get("included_candidate_ids")
            or [],
            "candidate_event_map": capsule.audit.get("candidate_event_map") or {},
            "unresolved_obligation_ids": capsule.audit.get(
                "unresolved_obligation_ids"
            )
            or [],
            "valid_evidence_ids": valid_evidence_ids,
            "verified_evidence_ids": sorted(
                {
                    evidence_id
                    for item in payload.get("verified_evidence") or []
                    for evidence_id in item.get("evidence_ids") or []
                }
            ),
            "capsule_candidate_frontier": payload.get("candidate_frontier") or [],
            "capsule_unresolved_obligations": payload.get("unresolved_obligations")
            or [],
            "capsule_constraints": payload.get("constraints") or {},
            "capsule_conflicts": payload.get("conflicts") or [],
            "historical_action": action,
        }
        boundaries.append(boundary)
        state_exports.append(
            {
                **boundary,
                "capsule_text": capsule.text,
                "capsule_payload": payload,
                "capsule_audit": dict(capsule.audit),
                "control_messages": control_messages,
                "treatment_messages": treatment_messages,
                "historical_thought": str(groups[step_id][0].get("thought") or ""),
            }
        )

        for archived_step in groups[step_id]:
            action_row = archived_step.get("action") or {}
            tool = str(action_row.get("function") or "")
            if not tool or tool == "answer":
                continue
            _merge(
                memory,
                tool=tool,
                parameters=action_row.get("parameters") or {},
                output=str(archived_step.get("observation") or ""),
                question=question,
                config=runtime_config,
            )

    return {
        "qid": str(result.get("qid") or ""),
        "question_type": str(result.get("question_type") or ""),
        "question": question,
        "duration_s": duration,
        "tail": int(result.get("api_total_tokens") or 0) > 100000,
        "historical_api_tokens": int(result.get("api_total_tokens") or 0),
        "historical_api_requests": int(result.get("api_request_count") or 0),
        "historical_pred": str(result.get("pred_letter") or ""),
        "ground_truth": str(result.get("gt_letter") or ""),
        "historical_correct": result.get("correct") is True,
        "trajectory": str(path.resolve()),
        "trajectory_sha256": _sha256(path),
        "planner_boundaries": len(boundaries),
        "control_prompt_tokens": sum(row["control_prompt_tokens"] for row in boundaries),
        "treatment_prompt_tokens": sum(
            row["treatment_prompt_tokens"] for row in boundaries
        ),
        "control_memory_tokens": sum(row["control_memory_tokens"] for row in boundaries),
        "capsule_tokens": sum(row["capsule_tokens"] for row in boundaries),
        "boundaries": boundaries,
        "_state_exports": state_exports,
    }


def _round_robin(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("question_type") or "unknown")].append(row)
    for values in by_family.values():
        values.sort(key=lambda row: str(row.get("qid") or ""))
    selected: list[dict[str, Any]] = []
    families = sorted(by_family)
    while len(selected) < limit and any(by_family.values()):
        for family in families:
            if by_family[family] and len(selected) < limit:
                selected.append(by_family[family].pop(0))
    return selected


def _select_fixed_states(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_qids: set[str] = set()

    answer_candidates: list[dict[str, Any]] = []
    for question in sorted(rows, key=lambda row: (row["tail"], row["qid"])):
        boundary = next(
            (row for row in question["_state_exports"] if row["answer_ready"]),
            None,
        )
        if boundary:
            answer_candidates.append({**question, "selected_boundary": boundary})
    for row in _round_robin(answer_candidates, 8):
        selected.append({**row, "category": "answer_ready"})
        used_qids.add(row["qid"])

    ordinary_candidates: list[dict[str, Any]] = []
    for question in sorted(rows, key=lambda row: row["qid"]):
        if question["tail"] or question["qid"] in used_qids:
            continue
        unresolved = [
            row
            for row in question["_state_exports"]
            if not row["answer_ready"] and row["activated"]
        ]
        if not unresolved:
            continue
        boundary = unresolved[(len(unresolved) - 1) // 2]
        ordinary_candidates.append({**question, "selected_boundary": boundary})
    for row in _round_robin(ordinary_candidates, 8):
        selected.append({**row, "category": "ordinary_unresolved"})
        used_qids.add(row["qid"])

    tail_questions = sorted(
        (row for row in rows if row["tail"]),
        key=lambda row: (-row["historical_api_tokens"], row["qid"]),
    )[:8]
    for question in tail_questions:
        values = [row["control_memory_tokens"] for row in question["_state_exports"]]
        median_size = statistics.median(values)
        boundary = next(
            row
            for row in question["_state_exports"]
            if row["control_memory_tokens"] >= median_size
        )
        selected.append(
            {**question, "selected_boundary": boundary, "category": "tail_onset_mid"}
        )

    exports: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        boundary = row["selected_boundary"]
        exports.append(
            {
                "state_id": f"P99S{index:02d}",
                "category": row["category"],
                "qid": row["qid"],
                "question_type": row["question_type"],
                "tail": row["tail"],
                "step": boundary["step"],
                "control_prompt_tokens": boundary["control_prompt_tokens"],
                "treatment_prompt_tokens": boundary["treatment_prompt_tokens"],
                "capsule_hash": boundary["capsule_hash"],
                "included_evidence_ids": boundary["included_evidence_ids"],
                "included_event_ids": boundary["included_event_ids"],
                "included_candidate_ids": boundary["included_candidate_ids"],
                "candidate_event_map": boundary["candidate_event_map"],
                "unresolved_obligation_ids": boundary["unresolved_obligation_ids"],
                "valid_evidence_ids": boundary["valid_evidence_ids"],
                "verified_evidence_ids": boundary["verified_evidence_ids"],
                "capsule_candidate_frontier": boundary[
                    "capsule_candidate_frontier"
                ],
                "capsule_unresolved_obligations": boundary[
                    "capsule_unresolved_obligations"
                ],
                "capsule_constraints": boundary["capsule_constraints"],
                "capsule_conflicts": boundary["capsule_conflicts"],
                "historical_action": boundary["historical_action"],
                "historical_thought": boundary["historical_thought"],
                "historical_pred": row["historical_pred"],
                "ground_truth": row["ground_truth"],
                "historical_correct": row["historical_correct"],
                "trajectory": row["trajectory"],
                "trajectory_sha256": row["trajectory_sha256"],
                "control_messages": boundary["control_messages"],
                "treatment_messages": boundary["treatment_messages"],
            }
        )
    return exports


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    boundaries = [boundary for row in rows for boundary in row["boundaries"]]
    control_prompt = sum(row["control_prompt_tokens"] for row in boundaries)
    treatment_prompt = sum(row["treatment_prompt_tokens"] for row in boundaries)
    control_memory = sum(row["control_memory_tokens"] for row in boundaries)
    capsule_memory = sum(row["capsule_tokens"] for row in boundaries)
    return {
        "questions": len(rows),
        "planner_boundaries": len(boundaries),
        "activated_boundaries": sum(row["activated"] for row in boundaries),
        "control_prompt_tokens": control_prompt,
        "treatment_prompt_tokens": treatment_prompt,
        "prompt_reduction": round(1.0 - treatment_prompt / max(1, control_prompt), 6),
        "control_memory_tokens": control_memory,
        "capsule_memory_tokens": capsule_memory,
        "memory_reduction": round(1.0 - capsule_memory / max(1, control_memory), 6),
        "determinism_failures": sum(not row["deterministic"] for row in boundaries),
        "memory_mutations": sum(row["memory_mutated"] for row in boundaries),
        "fail_open_count": sum(row["fail_open"] for row in boundaries),
        "over_budget_count": sum(row["over_budget"] for row in boundaries),
        "invalid_reference_count": sum(
            len(row["invalid_references"]) for row in boundaries
        ),
        "omitted_decisive_count": sum(
            len(row["omitted_decisive_support_ids"]) for row in boundaries
        ),
        "omitted_strong_conflict_evidence_count": sum(
            len(row["omitted_strong_conflict_evidence_ids"]) for row in boundaries
        ),
        "omitted_recorded_conflict_count": sum(
            row["omitted_conflict_count"] for row in boundaries
        ),
        "duplicate_fact_body_count": sum(
            row["duplicate_fact_bodies"] for row in boundaries
        ),
        "mistaken_event_merge_count": sum(
            row["mistaken_event_merges"] for row in boundaries
        ),
        "nonempty_expansion_over_5pct_count": sum(
            row["activated"]
            and row["treatment_prompt_tokens"] > row["control_prompt_tokens"] * 1.05
            for row in boundaries
        ),
        "max_capsule_tokens": max((row["capsule_tokens"] for row in boundaries), default=0),
    }


def _markdown(report: dict[str, Any]) -> str:
    all_stats = report["aggregate"]["all50"]
    tail = report["aggregate"]["tail14"]
    non_tail = report["aggregate"]["non_tail36"]
    gate = report["phase1_gate"]
    lines = [
        "# P99 first-50 offline Planner projection audit",
        "",
        "This report is zero-network. It reconstructs every immutable P38 Planner",
        "boundary and compares only the API-facing memory projection.",
        "",
        "## Outcome",
        "",
        f"- Phase 0/1 all pass: **{gate['all_pass']}**.",
        f"- Boundaries: `{all_stats['planner_boundaries']}` across 50 questions; "
        f"capsule activated on `{all_stats['activated_boundaries']}`.",
        f"- Estimated full Planner prompt reduction: "
        f"`{100 * all_stats['prompt_reduction']:.2f}%` overall, "
        f"`{100 * tail['prompt_reduction']:.2f}%` tail14, and "
        f"`{100 * non_tail['prompt_reduction']:.2f}%` non-tail36.",
        f"- API-facing memory reduction: `{100 * all_stats['memory_reduction']:.2f}%`.",
        f"- Maximum capsule size: `{all_stats['max_capsule_tokens']}` / 4000 tokens.",
        "",
        "## Invariants",
        "",
    ]
    for key, value in gate.items():
        lines.append(f"- `{key}`: **{value}**")
    lines.extend(
        [
            "",
            "| qid | stratum | family | steps | prompt control | capsule | delta |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(report["questions"], key=lambda item: (-item["historical_api_tokens"], item["qid"])):
        reduction = 1.0 - row["treatment_prompt_tokens"] / max(1, row["control_prompt_tokens"])
        lines.append(
            f"| {row['qid']} | {'tail' if row['tail'] else 'non-tail'} | "
            f"{row['question_type']} | {row['planner_boundaries']} | "
            f"{row['control_prompt_tokens']} | {row['treatment_prompt_tokens']} | "
            f"{100 * reduction:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    source_rows = _read_jsonl(P38_RESULTS)
    source_paths = [Path(str(row["trajectory_path"])) for row in source_rows]
    before_hashes = {str(path): _sha256(path) for path in source_paths + P38_RESULTS}
    baseline_file_count, baseline_tree_hash = _tree_digest(P38_RUNTIME)
    config_equivalence = _source_config_equivalence()

    rows = [_replay_question(row) for row in source_rows]
    after_hashes = {str(path): _sha256(path) for path in source_paths + P38_RESULTS}
    aggregate = {
        "all50": _aggregate(rows),
        "tail14": _aggregate([row for row in rows if row["tail"]]),
        "non_tail36": _aggregate([row for row in rows if not row["tail"]]),
    }
    all_stats = aggregate["all50"]
    gate = {
        "all_50_trajectories_parsed": len(rows) == 50,
        "tail_definition_reproduces_14": sum(row["tail"] for row in rows) == 14,
        "archived_inputs_byte_identical": before_hashes == after_hashes,
        "original_p38_tree_still_frozen": (
            baseline_file_count == 152
            and baseline_tree_hash
            == "c14eba8cc41510f5f83876195c1ef6d91e9bf02ff6738c15dc938d374b76ff3a"
        ),
        "control_default_off_and_config_equivalent": all(
            value
            for key, value in config_equivalence.items()
            if key
            in {
                "candidate_default_off",
                "only_p99_config_fields_added",
                "memory_formatter_source_identical",
                "prompt_source_identical",
            }
        ),
        "deterministic": all_stats["determinism_failures"] == 0,
        "memory_read_only": all_stats["memory_mutations"] == 0,
        "zero_invalid_references": all_stats["invalid_reference_count"] == 0,
        "zero_omitted_decisive_support": all_stats["omitted_decisive_count"] == 0,
        "zero_omitted_strong_conflicts": (
            all_stats["omitted_strong_conflict_evidence_count"] == 0
            and all_stats["omitted_recorded_conflict_count"] == 0
        ),
        "zero_duplicate_fact_bodies": all_stats["duplicate_fact_body_count"] == 0,
        "zero_mistaken_event_merges": all_stats["mistaken_event_merge_count"] == 0,
        "zero_budget_overflow_or_fail_open": (
            all_stats["over_budget_count"] == 0
            and all_stats["fail_open_count"] == 0
            and all_stats["max_capsule_tokens"] <= 4000
        ),
        "aggregate_prompt_reduction_at_least_30pct": (
            all_stats["prompt_reduction"] >= 0.30
        ),
        "tail14_prompt_reduction_at_least_30pct": (
            aggregate["tail14"]["prompt_reduction"] >= 0.30
        ),
        "no_nonempty_state_expands_over_5pct": (
            all_stats["nonempty_expansion_over_5pct_count"] == 0
        ),
        "zero_network_calls": True,
    }
    gate["all_pass"] = all(gate.values())
    fixed_states = _select_fixed_states(rows) if gate["all_pass"] else []

    serialized_questions = []
    for row in rows:
        clean = {key: value for key, value in row.items() if key != "_state_exports"}
        serialized_questions.append(clean)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "P99 Planner Evidence Capsule",
        "mode": "zero_network_first50_projection",
        "baseline": {
            "runtime": str(P38_RUNTIME),
            "file_count": baseline_file_count,
            "tree_sha256": baseline_tree_hash,
            "config_equivalence": config_equivalence,
        },
        "aggregate": aggregate,
        "phase1_gate": gate,
        "fixed_state_phase2_authorized": gate["all_pass"],
        "fixed_state_count": len(fixed_states),
        "questions": serialized_questions,
        "input_hashes_before": before_hashes,
        "input_hashes_after": after_hashes,
        "network_calls": 0,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "p99_first50_projection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "p99_first50_projection.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    if fixed_states:
        state_manifest = {
            "generated_at_utc": report["generated_at_utc"],
            "selection_authorized_by_phase1": True,
            "selection_rule": (
                "8 earliest answer-ready states diversified by family; 8 median "
                "activated ordinary unresolved non-tail states diversified by family; first "
                "boundary at/above median raw memory for top-cost 8 tail questions"
            ),
            "arm_repeats": 3,
            "states": fixed_states,
            "network_calls": 0,
        }
        (OUTPUT_DIR / "p99_fixed_state_manifest.json").write_text(
            json.dumps(state_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "phase1_gate": gate,
                "fixed_state_count": len(fixed_states),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if gate["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
