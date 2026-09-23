#!/usr/bin/env python3
"""
Run VideoSeek over an MLVU subset and write a per-question summary.

Usage:
  source .env
  python run_mlvu.py --subset dev_short2_mc.json                 # default
  python run_mlvu.py --subset dev_first8_mc.json --limit 4       # cap
  python run_mlvu.py --subset dev_first8_mc.json --types anomaly_reco,plotQA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Make sure we can import videoseek when run from inside this directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import general_config, init_config, prompts_config  # type: ignore
from videoseek.agent import VideoSeekAgent  # type: ignore
from videoseek.core.memory import extract_v10_payload  # type: ignore


MLVU_ROOT = Path("/home/zjw/workspace/ReKV/data/mlvu")


def letter(i: int) -> str:
    return chr(ord("A") + i)


def build_query(question: str, choices: list[str]) -> str:
    """Use exact choices from the dataset — never paraphrase."""
    lines = [question.strip()]
    for i, c in enumerate(choices):
        lines.append(f"({letter(i)}) {c}")
    lines.append(
        "Please directly answer with the best option's letter from the given "
        f"choices directly ({', '.join(letter(i) for i in range(len(choices)))})."
    )
    return "\n".join(lines)


def gt_letter(answer: str, choices: list[str]) -> str:
    """Index of the GT answer in choices → letter. First match wins (handles
    MLVU's occasional duplicate options)."""
    for i, c in enumerate(choices):
        if c == answer:
            return letter(i)
    return "?"


def normalize_pred(pred: str, choices=None) -> str:
    """Read explicit option tokens, never a letter embedded in an English word."""
    import re
    allowed = "".join(letter(i) for i in range(len(choices))) if choices is not None else "ABCD"
    text = str(pred or "").strip().strip("`").strip()
    patterns = [rf"\(?([{allowed}])\)?[.]?", rf"(?:final\s+answer|answer)(?:\s+is)?\s*:\s*\(?([{allowed}])\)?[.]?"]
    for pattern in patterns:
        match = re.fullmatch(pattern, text, re.I)
        if match: return match.group(1).upper()
    return ""


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:160]


def summarize_observer_usage(traj_dict: dict) -> dict:
    usage = {
        "api_observer_count": 0,
        "local_qwen_observer_count": 0,
        "api_after_qwen_error_count": 0,
        "qwen_tool_call_count": 0,
        "qwen_internal_call_count": 0,
        "api_visual_tool_count": 0,
        "router_changed_action_count": 0,
        "direct_planner_answer_count": 0,
        "dual_path_overview_count": 0,
        "dual_path_local_frames": 0,
        "dual_path_local_candidates": 0,
        "dual_path_local_novel_candidates": 0,
        "dual_path_exposed_wait_s": 0.0,
        "admissibility_rejection_count": 0,
        "post_success_overview_blocked_count": 0,
    }
    for step in traj_dict.get("steps", []) or []:
        routing_audit = step.get("routing_audit") or {}
        if routing_audit.get("router_changed_action") is True:
            usage["router_changed_action_count"] += 1
        action = step.get("action") or {}
        tool = action.get("function") or ""
        executor_blocked = routing_audit.get("executor_blocked") is True
        if executor_blocked:
            usage["admissibility_rejection_count"] += 1
            if tool == "overview":
                usage["post_success_overview_blocked_count"] += 1
        if tool in {"skim_qwen", "focus_qwen", "localize_qwen"}:
            usage["qwen_tool_call_count"] += 1
        if tool == "answer" and str(
            (action.get("parameters") or {}).get("answer") or ""
        ).upper() in {"A", "B", "C", "D"}:
            usage["direct_planner_answer_count"] += 1
        if (
            tool in {"overview", "skim", "focus", "frame_verify"}
            and not executor_blocked
        ):
            usage["api_visual_tool_count"] += 1
        payload = extract_v10_payload(step.get("observation") or "")
        if not isinstance(payload, dict):
            continue
        dual_path = payload.get("dual_path_overview") or {}
        if dual_path:
            coverage = dual_path.get("local_coverage") or {}
            timing = dual_path.get("timing") or {}
            usage["dual_path_overview_count"] += 1
            usage["dual_path_local_frames"] += int(
                coverage.get("frames_processed") or 0
            )
            usage["dual_path_local_candidates"] += int(
                dual_path.get("local_candidate_count") or 0
            )
            usage["dual_path_local_novel_candidates"] += int(
                dual_path.get("local_novel_candidate_count") or 0
            )
            usage["dual_path_exposed_wait_s"] += float(
                timing.get("exposed_local_wait_s") or 0.0
            )
        backend = str(payload.get("observer_backend") or "")
        usage["qwen_internal_call_count"] += int(
            payload.get("internal_qwen_calls")
            or (1 if tool in {"skim_qwen", "focus_qwen"} else 0)
        )
        if backend == "local_qwen":
            usage["local_qwen_observer_count"] += 1
        elif backend == "api":
            usage["api_observer_count"] += 1
        elif backend:
            usage["api_after_qwen_error_count"] += 1
    return usage


def read_api_usage_log(path: Path | None) -> list[dict]:
    """Read complete usage records while tolerating a missing or partial log."""
    if path is None or not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def summarize_api_usage(rows: list[dict]) -> dict:
    visual_rows = [row for row in rows if bool(row.get("has_images"))]
    text_rows = [row for row in rows if not bool(row.get("has_images"))]

    def total(items: list[dict], key: str) -> int:
        return sum(int(item.get(key) or 0) for item in items)

    return {
        "api_request_count": len(rows),
        "api_visual_request_count": len(visual_rows),
        "api_text_request_count": len(text_rows),
        "api_image_count": total(rows, "image_count"),
        "api_prompt_tokens": total(rows, "prompt_tokens"),
        "api_completion_tokens": total(rows, "completion_tokens"),
        "api_total_tokens": total(rows, "total_tokens"),
        "api_visual_prompt_tokens": total(visual_rows, "prompt_tokens"),
        "api_visual_completion_tokens": total(visual_rows, "completion_tokens"),
        "api_visual_total_tokens": total(visual_rows, "total_tokens"),
        "api_text_prompt_tokens": total(text_rows, "prompt_tokens"),
        "api_text_completion_tokens": total(text_rows, "completion_tokens"),
        "api_text_total_tokens": total(text_rows, "total_tokens"),
    }


def make_config(args) -> dict:
    cfg: dict = {}
    cfg.update(general_config)
    cfg.update(prompts_config)
    cfg = init_config(cfg, args)
    if isinstance(cfg.get("tools"), str):
        cfg["tools"] = [item.strip() for item in cfg["tools"].split(",") if item.strip()]
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="dev_short2_mc.json",
                    help="Filename under ReKV/data/mlvu/")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on total questions to run.")
    ap.add_argument("--types", default=None,
                    help="Comma-separated question_type whitelist.")
    ap.add_argument("--output_dir", default="./runs/mlvu_eval")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--tools", default=None,
                    help="Comma-separated tool list, e.g. overview,skim,focus,focus_qwen")

    # Forward defaults from videoseek's general.yaml so init_config works.
    ap.add_argument("--model_name", default=general_config["model_name"])
    ap.add_argument("--api_base", default=general_config["api_base"])
    ap.add_argument("--api_key", default=general_config["api_key"])
    ap.add_argument("--api_version", default=general_config["api_version"])
    ap.add_argument("--observer_model_name", default=general_config.get("observer_model_name"))
    ap.add_argument("--observer_api_base", default=general_config.get("observer_api_base"))
    ap.add_argument("--observer_api_key", default=general_config.get("observer_api_key"))
    ap.add_argument("--observer_api_version", default=general_config.get("observer_api_version"))
    ap.add_argument("--observer_reasoning_effort", default=general_config.get("observer_reasoning_effort"))
    ap.add_argument("--observer_backend", default=general_config.get("observer_backend"),
                    choices=["api", "local_qwen", "hybrid"])
    ap.add_argument("--local_qwen_tools", default=general_config.get("local_qwen_tools"))
    ap.add_argument("--local_qwen_focus_modes", default=general_config.get("local_qwen_focus_modes"))
    ap.add_argument("--local_qwen_skim_modes", default=general_config.get("local_qwen_skim_modes"))
    ap.add_argument("--local_qwen_model_path", default=general_config.get("local_qwen_model_path"))
    ap.add_argument("--local_qwen_python", default=general_config.get("local_qwen_python"))
    ap.add_argument("--local_qwen_persistent_worker", action=argparse.BooleanOptionalAction,
                    default=general_config.get("local_qwen_persistent_worker"))
    ap.add_argument("--local_qwen_cuda_visible_devices", default=general_config.get("local_qwen_cuda_visible_devices"))
    ap.add_argument("--local_qwen_device_map", default=general_config.get("local_qwen_device_map"))
    ap.add_argument("--local_qwen_max_memory", default=general_config.get("local_qwen_max_memory"))
    ap.add_argument("--local_qwen_torch_dtype", default=general_config.get("local_qwen_torch_dtype"),
                    choices=["bfloat16", "float16"])
    ap.add_argument("--local_qwen_max_new_tokens", type=int,
                    default=general_config.get("local_qwen_max_new_tokens"))
    ap.add_argument("--local_qwen_max_images", type=int,
                    default=general_config.get("local_qwen_max_images"))
    ap.add_argument("--local_qwen_max_image_side", type=int,
                    default=general_config.get("local_qwen_max_image_side"))
    ap.add_argument("--local_qwen_timeout_s", type=int,
                    default=general_config.get("local_qwen_timeout_s"))
    ap.add_argument("--local_qwen_no_cpu_offload", action=argparse.BooleanOptionalAction,
                    default=general_config.get("local_qwen_no_cpu_offload"))
    ap.add_argument("--local_qwen_fallback_to_api", action=argparse.BooleanOptionalAction,
                    default=general_config.get("local_qwen_fallback_to_api"))
    ap.add_argument("--qwen_caption_only_output", action=argparse.BooleanOptionalAction,
                    default=general_config.get("qwen_caption_only_output"))
    ap.add_argument("--reasoning_effort", default=general_config["reasoning_effort"])
    ap.add_argument("--seed", type=int, default=general_config["seed"])
    ap.add_argument("--temperature", type=float, default=general_config["temperature"])
    ap.add_argument("--max_tokens", type=int, default=general_config["max_tokens"])
    ap.add_argument("--max_steps", type=int, default=general_config["max_steps"])
    ap.add_argument("--use_evidence_reducer", action=argparse.BooleanOptionalAction,
                    default=general_config.get("use_evidence_reducer"))
    ap.add_argument("--use_structured_evidence_state", action=argparse.BooleanOptionalAction,
                    default=general_config.get("use_structured_evidence_state"))
    ap.add_argument("--structured_evidence_max_items", type=int,
                    default=general_config.get("structured_evidence_max_items"))
    ap.add_argument("--structured_evidence_prompt_max_items", type=int,
                    default=general_config.get("structured_evidence_prompt_max_items"))
    ap.add_argument("--structured_evidence_answer_max_items", type=int,
                    default=general_config.get("structured_evidence_answer_max_items"))
    ap.add_argument("--structured_evidence_include_planner_state", action=argparse.BooleanOptionalAction,
                    default=general_config.get("structured_evidence_include_planner_state"))
    ap.add_argument("--coseek1_compact_event_coverage", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_compact_event_coverage"))
    ap.add_argument("--coseek1_compact_event_coverage_max_items", type=int,
                    default=general_config.get("coseek1_compact_event_coverage_max_items"))
    ap.add_argument("--coseek1_event_coverage_inline_verify", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_event_coverage_inline_verify"))
    ap.add_argument("--coseek1_event_coverage_inline_verify_max_windows", type=int,
                    default=general_config.get("coseek1_event_coverage_inline_verify_max_windows"))
    ap.add_argument("--coseek1_event_coverage_inline_verify_radius_s", type=float,
                    default=general_config.get("coseek1_event_coverage_inline_verify_radius_s"))
    ap.add_argument("--structured_evidence_include_occurrence_view", action=argparse.BooleanOptionalAction,
                    default=general_config.get("structured_evidence_include_occurrence_view"))
    ap.add_argument("--structured_evidence_include_timeline_view", action=argparse.BooleanOptionalAction,
                    default=general_config.get("structured_evidence_include_timeline_view"))
    ap.add_argument("--structured_evidence_include_object_state_view", action=argparse.BooleanOptionalAction,
                    default=general_config.get("structured_evidence_include_object_state_view"))
    ap.add_argument("--structured_evidence_preserve_aggregate_verifier_decision", action=argparse.BooleanOptionalAction,
                    default=general_config.get("structured_evidence_preserve_aggregate_verifier_decision"))
    ap.add_argument("--coseek1_compact_planner_context", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_compact_planner_context"))
    ap.add_argument("--coseek1_compact_answer_context", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_compact_answer_context"))
    ap.add_argument("--coseek1_planner_evidence_capsule_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_planner_evidence_capsule_enabled"))
    ap.add_argument("--coseek1_planner_capsule_token_budget", type=int,
                    default=general_config.get("coseek1_planner_capsule_token_budget"))
    ap.add_argument("--coseek1_planner_capsule_max_verified", type=int,
                    default=general_config.get("coseek1_planner_capsule_max_verified"))
    ap.add_argument("--coseek1_planner_capsule_max_candidates", type=int,
                    default=general_config.get("coseek1_planner_capsule_max_candidates"))
    ap.add_argument("--coseek1_planner_capsule_max_obligations", type=int,
                    default=general_config.get("coseek1_planner_capsule_max_obligations"))
    ap.add_argument("--coseek1_planner_capsule_audit_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_planner_capsule_audit_enabled"))
    ap.add_argument("--coseek1_direct_planner_answer", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_direct_planner_answer"))
    ap.add_argument("--coseek1_localize_qwen_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_localize_qwen_enabled"))
    ap.add_argument("--localize_qwen_top_k", type=int,
                    default=general_config.get("localize_qwen_top_k"))
    ap.add_argument("--localize_qwen_max_top_k", type=int,
                    default=general_config.get("localize_qwen_max_top_k"))
    ap.add_argument("--localize_qwen_row_candidate_windows", action=argparse.BooleanOptionalAction,
                    default=general_config.get("localize_qwen_row_candidate_windows"))
    ap.add_argument("--localize_qwen_cover_search_windows", action=argparse.BooleanOptionalAction,
                    default=general_config.get("localize_qwen_cover_search_windows"))
    ap.add_argument("--localize_qwen_max_search_windows", type=int,
                    default=general_config.get("localize_qwen_max_search_windows"))
    ap.add_argument("--coseek1_tool_integrity_repair_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_tool_integrity_repair_enabled"))
    ap.add_argument("--coseek1_tool_memory_anchor_margin_s", type=float,
                    default=general_config.get("coseek1_tool_memory_anchor_margin_s"))
    ap.add_argument("--coseek1_tool_memory_anchors_per_window", type=int,
                    default=general_config.get("coseek1_tool_memory_anchors_per_window"))
    ap.add_argument("--coseek1_tool_memory_anchor_pad_s", type=float,
                    default=general_config.get("coseek1_tool_memory_anchor_pad_s"))
    ap.add_argument("--coseek1_tool_event_identity_repair_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_tool_event_identity_repair_enabled"))
    ap.add_argument("--coseek1_tool_recollect_expanded_anchors_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_tool_recollect_expanded_anchors_enabled"))
    ap.add_argument("--grounded_verify_packet_boundary_context_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("grounded_verify_packet_boundary_context_enabled"))
    ap.add_argument("--localize_qwen_coarse_max_frames", type=int,
                    default=general_config.get("localize_qwen_coarse_max_frames"))
    ap.add_argument("--localize_qwen_fine_max_frames", type=int,
                    default=general_config.get("localize_qwen_fine_max_frames"))
    ap.add_argument("--localize_qwen_verify_window_s", type=float,
                    default=general_config.get("localize_qwen_verify_window_s"))
    ap.add_argument("--localize_qwen_verify_context_margin_s", type=float,
                    default=general_config.get("localize_qwen_verify_context_margin_s"))
    ap.add_argument("--localize_qwen_search_context_margin_s", type=float,
                    default=general_config.get("localize_qwen_search_context_margin_s"))
    ap.add_argument("--localize_qwen_adaptive_search_context", action=argparse.BooleanOptionalAction,
                    default=general_config.get("localize_qwen_adaptive_search_context"))
    ap.add_argument("--localize_qwen_normalized_candidate_score_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("localize_qwen_normalized_candidate_score_enabled"))
    ap.add_argument("--localize_qwen_duration_budget_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("localize_qwen_duration_budget_enabled"))
    ap.add_argument("--localize_qwen_duration_budget_fps", type=float,
                    default=general_config.get("localize_qwen_duration_budget_fps"))
    ap.add_argument("--localize_qwen_duration_budget_min_frames", type=int,
                    default=general_config.get("localize_qwen_duration_budget_min_frames"))
    ap.add_argument("--localize_qwen_duration_budget_max_frames", type=int,
                    default=general_config.get("localize_qwen_duration_budget_max_frames"))
    ap.add_argument("--localize_qwen_codec_anchor_sampling_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("localize_qwen_codec_anchor_sampling_enabled"))
    ap.add_argument("--localize_qwen_boundary_expansion_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("localize_qwen_boundary_expansion_enabled"))
    ap.add_argument("--localize_qwen_boundary_context_margin_s", type=float,
                    default=general_config.get("localize_qwen_boundary_context_margin_s"))
    ap.add_argument("--localize_qwen_boundary_trigger_s", type=float,
                    default=general_config.get("localize_qwen_boundary_trigger_s"))
    ap.add_argument("--localize_qwen_boundary_expansion_max_frames", type=int,
                    default=general_config.get("localize_qwen_boundary_expansion_max_frames"))
    ap.add_argument("--localize_qwen_caption_cache_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("localize_qwen_caption_cache_enabled"))
    ap.add_argument("--localize_qwen_caption_cache_goal_similarity", type=float,
                    default=general_config.get("localize_qwen_caption_cache_goal_similarity"))
    ap.add_argument("--localize_qwen_caption_cache_coverage_ratio", type=float,
                    default=general_config.get("localize_qwen_caption_cache_coverage_ratio"))
    ap.add_argument("--localize_qwen_candidate_novelty_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("localize_qwen_candidate_novelty_enabled"))
    ap.add_argument("--localize_qwen_candidate_novelty_overlap", type=float,
                    default=general_config.get("localize_qwen_candidate_novelty_overlap"))
    ap.add_argument("--localize_qwen_strict_evidence_status_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("localize_qwen_strict_evidence_status_enabled"))
    ap.add_argument("--localize_qwen_trace_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("localize_qwen_trace_enabled"))
    ap.add_argument("--coseek1_localize_inline_verify", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_localize_inline_verify"))
    ap.add_argument("--coseek1_retrieval_verify_semantic_decoupling_enabled",
                    action=argparse.BooleanOptionalAction,
                    default=general_config.get(
                        "coseek1_retrieval_verify_semantic_decoupling_enabled"
                    ))
    ap.add_argument("--localize_inline_verify_max_windows", type=int,
                    default=general_config.get("localize_inline_verify_max_windows"))
    ap.add_argument("--localize_inline_verify_max_frames", type=int,
                    default=general_config.get("localize_inline_verify_max_frames"))
    ap.add_argument("--localize_inline_verify_contact_sheets", action=argparse.BooleanOptionalAction,
                    default=general_config.get("localize_inline_verify_contact_sheets"))
    ap.add_argument("--multiwindow_verify_recovery_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("multiwindow_verify_recovery_enabled"))
    ap.add_argument("--multiwindow_verify_recovery_max_calls", type=int,
                    default=general_config.get("multiwindow_verify_recovery_max_calls"))
    ap.add_argument("--multiwindow_verify_recovery_batch_size", type=int,
                    default=general_config.get("multiwindow_verify_recovery_batch_size"))
    ap.add_argument("--grounded_frame_verify_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("grounded_frame_verify_enabled"))
    ap.add_argument("--grounded_frame_verify_max_anchors", type=int,
                    default=general_config.get("grounded_frame_verify_max_anchors"))
    ap.add_argument("--grounded_frame_verify_crop_area_threshold", type=float,
                    default=general_config.get("grounded_frame_verify_crop_area_threshold"))
    ap.add_argument("--grounded_frame_verify_padding_ratio", type=float,
                    default=general_config.get("grounded_frame_verify_padding_ratio"))
    ap.add_argument("--grounded_frame_verify_min_crop_side_ratio", type=float,
                    default=general_config.get("grounded_frame_verify_min_crop_side_ratio"))
    ap.add_argument("--grounded_frame_verify_context_short_side", type=int,
                    default=general_config.get("grounded_frame_verify_context_short_side"))
    ap.add_argument("--grounded_frame_verify_max_new_tokens", type=int,
                    default=general_config.get("grounded_frame_verify_max_new_tokens"))
    ap.add_argument("--grounded_frame_verify_cache_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("grounded_frame_verify_cache_enabled"))
    ap.add_argument("--grounded_frame_verify_cache_dir",
                    default=general_config.get("grounded_frame_verify_cache_dir"))
    ap.add_argument("--grounded_high_recall_proposal_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("grounded_high_recall_proposal_enabled"))
    ap.add_argument("--grounded_verify_packet_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("grounded_verify_packet_enabled"))
    ap.add_argument("--grounded_verify_packet_anchors_per_candidate", type=int,
                    default=general_config.get("grounded_verify_packet_anchors_per_candidate"))
    ap.add_argument("--grounded_verify_packet_temporal_anchor_spread_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("grounded_verify_packet_temporal_anchor_spread_enabled"))
    ap.add_argument("--grounded_verify_packet_query_context_roles_enabled",
                    action=argparse.BooleanOptionalAction,
                    default=general_config.get(
                        "grounded_verify_packet_query_context_roles_enabled"
                    ))
    ap.add_argument("--grounded_verify_packet_anchor_audit_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("grounded_verify_packet_anchor_audit_enabled"))
    ap.add_argument("--grounded_verify_packet_crop_max_side", type=int,
                    default=general_config.get("grounded_verify_packet_crop_max_side"))
    ap.add_argument("--grounded_verify_packet_peak_max_side", type=int,
                    default=general_config.get("grounded_verify_packet_peak_max_side"))
    ap.add_argument("--packet_detail_grounding_escalation_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("packet_detail_grounding_escalation_enabled"))
    ap.add_argument("--packet_detail_grounding_escalation_frames", type=int,
                    default=general_config.get("packet_detail_grounding_escalation_frames"))
    ap.add_argument("--packet_adaptive_evidence_need_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("packet_adaptive_evidence_need_enabled"))
    ap.add_argument("--packet_detail_grounding_direct_escalation_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("packet_detail_grounding_direct_escalation_enabled"))
    ap.add_argument("--grounded_direct_verify_packet_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("grounded_direct_verify_packet_enabled"))
    ap.add_argument("--grounded_direct_verify_packet_group_size", type=int,
                    default=general_config.get("grounded_direct_verify_packet_group_size"))
    ap.add_argument("--grounded_direct_verify_packet_anchor_count", type=int,
                    default=general_config.get("grounded_direct_verify_packet_anchor_count"))
    ap.add_argument("--grounded_direct_verify_packet_detail_max_side", type=int,
                    default=general_config.get("grounded_direct_verify_packet_detail_max_side"))
    ap.add_argument("--grounded_candidate_binding_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("grounded_candidate_binding_enabled"))
    ap.add_argument("--grounded_candidate_event_binding_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("grounded_candidate_event_binding_enabled"))
    ap.add_argument("--grounded_candidate_event_group_max_gap_s", type=float,
                    default=general_config.get("grounded_candidate_event_group_max_gap_s"))
    ap.add_argument("--packet_decision_consistency_normalization_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("packet_decision_consistency_normalization_enabled"))
    ap.add_argument("--packet_multiwindow_decision_sufficiency_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("packet_multiwindow_decision_sufficiency_enabled"))
    ap.add_argument("--packet_preserve_resolved_candidate_set_decision_enabled",
                    action=argparse.BooleanOptionalAction,
                    default=general_config.get(
                        "packet_preserve_resolved_candidate_set_decision_enabled"
                    ))
    ap.add_argument("--coseek1_scope_aware_evidence_memory", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_scope_aware_evidence_memory"))
    ap.add_argument("--coseek1_separate_routing_candidates", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_separate_routing_candidates"))
    ap.add_argument("--coseek1_persistent_open_gaps", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_persistent_open_gaps"))
    ap.add_argument("--coseek1_semantic_gap_resolution_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_semantic_gap_resolution_enabled"))
    ap.add_argument("--coseek1_observer_conflict_memory_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_observer_conflict_memory_enabled"))
    ap.add_argument("--coseek1_query_relevant_memory_retention", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_query_relevant_memory_retention"))
    ap.add_argument("--coseek1_persistent_candidate_pool", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_persistent_candidate_pool"))
    ap.add_argument("--coseek1_persistent_candidate_pool_max_items", type=int,
                    default=general_config.get("coseek1_persistent_candidate_pool_max_items"))
    ap.add_argument("--coseek1_persistent_candidate_pool_timestamp_radius_s", type=float,
                    default=general_config.get("coseek1_persistent_candidate_pool_timestamp_radius_s"))
    ap.add_argument("--coseek1_candidate_evidence_projection", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_candidate_evidence_projection"))
    ap.add_argument("--coseek1_temporal_evidence_ledger_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_temporal_evidence_ledger_enabled"))
    ap.add_argument("--coseek1_temporal_evidence_ledger_max_events", type=int,
                    default=general_config.get("coseek1_temporal_evidence_ledger_max_events"))
    ap.add_argument("--coseek1_verified_observation_reuse_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_verified_observation_reuse_enabled"))
    ap.add_argument("--coseek1_verified_observation_reuse_overlap", type=float,
                    default=general_config.get("coseek1_verified_observation_reuse_overlap"))
    ap.add_argument("--coseek1_verified_observation_reuse_query_similarity", type=float,
                    default=general_config.get("coseek1_verified_observation_reuse_query_similarity"))
    ap.add_argument("--coseek1_candidate_context_localize_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_candidate_context_localize_enabled"))
    ap.add_argument("--focus_qwen_multi_window_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("focus_qwen_multi_window_enabled"))
    ap.add_argument("--focus_qwen_multi_window_max_windows", type=int,
                    default=general_config.get("focus_qwen_multi_window_max_windows"))
    ap.add_argument("--overview_require_all_timestamps", action=argparse.BooleanOptionalAction,
                    default=general_config.get("overview_require_all_timestamps"))
    ap.add_argument("--overview_fill_omitted_timestamps", action=argparse.BooleanOptionalAction,
                    default=general_config.get("overview_fill_omitted_timestamps"))
    ap.add_argument("--overview_frame_caption_words", type=int,
                    default=general_config.get("overview_frame_caption_words"))
    ap.add_argument("--overview_frame_short_side", type=int,
                    default=general_config.get("overview_frame_short_side"))
    ap.add_argument("--overview_embed_timestamp_labels", action=argparse.BooleanOptionalAction,
                    default=general_config.get("overview_embed_timestamp_labels"))
    ap.add_argument("--overview_contact_sheet_group_size", type=int,
                    choices=[4, 8],
                    default=general_config.get("overview_contact_sheet_group_size"))
    ap.add_argument("--dual_path_overview_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("dual_path_overview_enabled"))
    ap.add_argument("--dual_path_overview_shadow_only", action=argparse.BooleanOptionalAction,
                    default=general_config.get("dual_path_overview_shadow_only"))
    ap.add_argument("--dual_path_overview_trace_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("dual_path_overview_trace_enabled"))
    ap.add_argument("--dual_path_integration_mode",
                    choices=["append", "reserve", "fused", "selective"],
                    default=general_config.get("dual_path_integration_mode"))
    ap.add_argument("--dual_path_local_frames", type=int,
                    default=general_config.get("dual_path_local_frames"))
    ap.add_argument("--dual_path_local_min_frames", type=int,
                    default=general_config.get("dual_path_local_min_frames"))
    ap.add_argument("--dual_path_local_max_frames", type=int,
                    default=general_config.get("dual_path_local_max_frames"))
    ap.add_argument("--dual_path_local_target_interval_s", type=float,
                    default=general_config.get("dual_path_local_target_interval_s"))
    ap.add_argument("--dual_path_local_batch_size", type=int,
                    default=general_config.get("dual_path_local_batch_size"))
    ap.add_argument("--dual_path_local_min_processed_frames", type=int,
                    default=general_config.get("dual_path_local_min_processed_frames"))
    ap.add_argument("--dual_path_local_top_k", type=int,
                    default=general_config.get("dual_path_local_top_k"))
    ap.add_argument("--dual_path_local_candidate_pool_k", type=int,
                    default=general_config.get("dual_path_local_candidate_pool_k"))
    ap.add_argument("--dual_path_fused_top_k", type=int,
                    default=general_config.get("dual_path_fused_top_k"))
    ap.add_argument("--dual_path_semantic_prototype_ranking_enabled",
                    action=argparse.BooleanOptionalAction,
                    default=general_config.get("dual_path_semantic_prototype_ranking_enabled"))
    ap.add_argument("--dual_path_choice_aware_retrieval_enabled",
                    action=argparse.BooleanOptionalAction,
                    default=general_config.get("dual_path_choice_aware_retrieval_enabled"))
    ap.add_argument("--dual_path_query_target_reserve_enabled",
                    action=argparse.BooleanOptionalAction,
                    default=general_config.get("dual_path_query_target_reserve_enabled"))
    ap.add_argument("--dual_path_planner_caption_limit", type=int,
                    default=general_config.get("dual_path_planner_caption_limit"))
    ap.add_argument("--dual_path_local_deadline_s", type=float,
                    default=general_config.get("dual_path_local_deadline_s"))
    ap.add_argument("--dual_path_local_batch_timeout_s", type=int,
                    default=general_config.get("dual_path_local_batch_timeout_s"))
    ap.add_argument("--dual_path_local_caption_words", type=int,
                    default=general_config.get("dual_path_local_caption_words"))
    ap.add_argument("--dual_path_local_max_new_tokens", type=int,
                    default=general_config.get("dual_path_local_max_new_tokens"))
    ap.add_argument("--dual_path_local_frame_short_side", type=int,
                    default=general_config.get("dual_path_local_frame_short_side"))
    ap.add_argument("--dual_path_candidate_window_s", type=float,
                    default=general_config.get("dual_path_candidate_window_s"))
    ap.add_argument("--dual_path_remote_exclusion_s", type=float,
                    default=general_config.get("dual_path_remote_exclusion_s"))
    ap.add_argument("--dual_path_local_min_gap_s", type=float,
                    default=general_config.get("dual_path_local_min_gap_s"))
    ap.add_argument("--dual_path_remote_dedup_overlap", type=float,
                    default=general_config.get("dual_path_remote_dedup_overlap"))
    ap.add_argument("--coseek1_candidate_frontier", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_candidate_frontier"))
    ap.add_argument("--coseek1_candidate_frontier_max_items", type=int,
                    default=general_config.get("coseek1_candidate_frontier_max_items"))
    ap.add_argument("--coseek1_frontier_timestamp_cues", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_frontier_timestamp_cues"))
    ap.add_argument("--coseek1_frontier_timestamp_cue_radius_s", type=float,
                    default=general_config.get("coseek1_frontier_timestamp_cue_radius_s"))
    ap.add_argument("--coseek1_frontier_evidence_scope", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_frontier_evidence_scope"))
    ap.add_argument("--coseek1_frontier_temporal_boundaries", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_frontier_temporal_boundaries"))
    ap.add_argument("--coseek1_frontier_temporal_boundary_max_gap_s", type=float,
                    default=general_config.get("coseek1_frontier_temporal_boundary_max_gap_s"))
    ap.add_argument("--coseek1_frontier_investigation_state", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_frontier_investigation_state"))
    ap.add_argument("--coseek1_frontier_investigation_max_items", type=int,
                    default=general_config.get("coseek1_frontier_investigation_max_items"))
    ap.add_argument("--coseek1_evidence_episode_frontier", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_evidence_episode_frontier"))
    ap.add_argument("--coseek1_evidence_episode_max_items", type=int,
                    default=general_config.get("coseek1_evidence_episode_max_items"))
    ap.add_argument("--coseek1_episode_anchor_radius_s", type=float,
                    default=general_config.get("coseek1_episode_anchor_radius_s"))
    ap.add_argument("--coseek1_episode_focus_window_s", type=float,
                    default=general_config.get("coseek1_episode_focus_window_s"))
    ap.add_argument("--coseek1_episode_broad_window_s", type=float,
                    default=general_config.get("coseek1_episode_broad_window_s"))
    ap.add_argument("--coseek1_episode_backend_router", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_episode_backend_router"))
    ap.add_argument("--coseek1_answer_evidence_audit", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_answer_evidence_audit"))
    ap.add_argument("--direct_answer_from_letter_thought", action=argparse.BooleanOptionalAction,
                    default=general_config.get("direct_answer_from_letter_thought"))
    ap.add_argument("--coseek1_defer_insufficient_answer", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_defer_insufficient_answer"))
    ap.add_argument("--extract_explicit_thought_action", action=argparse.BooleanOptionalAction,
                    default=general_config.get("extract_explicit_thought_action"))
    ap.add_argument("--prefer_explicit_thought_action", action=argparse.BooleanOptionalAction,
                    default=general_config.get("prefer_explicit_thought_action"))
    ap.add_argument("--require_scene_skim_before_focus", action=argparse.BooleanOptionalAction,
                    default=general_config.get("require_scene_skim_before_focus"))
    ap.add_argument("--pre_focus_skim_context_s", type=float,
                    default=general_config.get("pre_focus_skim_context_s"))
    ap.add_argument("--pre_focus_skim_max_window_s", type=float,
                    default=general_config.get("pre_focus_skim_max_window_s"))
    ap.add_argument("--qwen_first_observer", action=argparse.BooleanOptionalAction,
                    default=general_config.get("qwen_first_observer"))
    ap.add_argument("--qwen_first_skim", action=argparse.BooleanOptionalAction,
                    default=general_config.get("qwen_first_skim"))
    ap.add_argument("--qwen_first_focus", action=argparse.BooleanOptionalAction,
                    default=general_config.get("qwen_first_focus"))
    ap.add_argument("--qwen_first_skim_modes",
                    default=general_config.get("qwen_first_skim_modes"))
    ap.add_argument("--qwen_first_focus_modes",
                    default=general_config.get("qwen_first_focus_modes"))
    ap.add_argument("--qwen_first_skim_max_window_s", type=float,
                    default=general_config.get("qwen_first_skim_max_window_s"))
    ap.add_argument("--qwen_first_focus_max_window_s", type=float,
                    default=general_config.get("qwen_first_focus_max_window_s"))
    ap.add_argument("--qwen_first_verify_uncertain_skim", action=argparse.BooleanOptionalAction,
                    default=general_config.get("qwen_first_verify_uncertain_skim"))
    ap.add_argument("--qwen_first_verify_uncertain_focus", action=argparse.BooleanOptionalAction,
                    default=general_config.get("qwen_first_verify_uncertain_focus"))
    ap.add_argument("--qwen_first_verify_parse_error", action=argparse.BooleanOptionalAction,
                    default=general_config.get("qwen_first_verify_parse_error"))
    ap.add_argument("--qwen_first_repeat_window_escalates_to_api", action=argparse.BooleanOptionalAction,
                    default=general_config.get("qwen_first_repeat_window_escalates_to_api"))
    ap.add_argument("--qwen_first_skim_api_relevance_threshold", type=float,
                    default=general_config.get("qwen_first_skim_api_relevance_threshold"))
    ap.add_argument("--query_aware_scene_probe", action=argparse.BooleanOptionalAction,
                    default=general_config.get("query_aware_scene_probe"))
    ap.add_argument("--query_aware_scene_probe_max_per_question", type=int,
                    default=general_config.get("query_aware_scene_probe_max_per_question"))
    ap.add_argument("--query_aware_scene_probe_max_window_s", type=float,
                    default=general_config.get("query_aware_scene_probe_max_window_s"))
    ap.add_argument("--query_aware_scene_probe_prefer_scene_range", action=argparse.BooleanOptionalAction,
                    default=general_config.get("query_aware_scene_probe_prefer_scene_range"))
    ap.add_argument("--query_aware_scene_probe_api_verify", action=argparse.BooleanOptionalAction,
                    default=general_config.get("query_aware_scene_probe_api_verify"))
    ap.add_argument("--query_aware_scene_probe_api_relevance_threshold", type=float,
                    default=general_config.get("query_aware_scene_probe_api_relevance_threshold"))
    ap.add_argument("--query_aware_scene_probe_single_images", action=argparse.BooleanOptionalAction,
                    default=general_config.get("query_aware_scene_probe_single_images"))
    ap.add_argument("--query_aware_scene_probe_short_side", type=int,
                    default=general_config.get("query_aware_scene_probe_short_side"))
    ap.add_argument("--query_aware_scene_probe_max_frames", type=int,
                    default=general_config.get("query_aware_scene_probe_max_frames"))
    ap.add_argument("--query_aware_scene_probe_max_new_tokens", type=int,
                    default=general_config.get("query_aware_scene_probe_max_new_tokens"))
    ap.add_argument("--focus_context_pad_s", type=float,
                    default=general_config.get("focus_context_pad_s"))
    ap.add_argument("--focus_qwen_max_frames", type=int,
                    default=general_config.get("focus_qwen_max_frames"))
    ap.add_argument("--focus_qwen_temporal_density_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("focus_qwen_temporal_density_enabled"))
    ap.add_argument("--focus_qwen_target_fps", type=float,
                    default=general_config.get("focus_qwen_target_fps"))
    ap.add_argument("--focus_qwen_context_pad_s", type=float,
                    default=general_config.get("focus_qwen_context_pad_s"))
    ap.add_argument("--focus_qwen_max_window_s", type=float,
                    default=general_config.get("focus_qwen_max_window_s"))
    ap.add_argument("--focus_qwen_short_side", type=int,
                    default=general_config.get("focus_qwen_short_side"))
    ap.add_argument("--focus_qwen_max_new_tokens", type=int,
                    default=general_config.get("focus_qwen_max_new_tokens"))
    ap.add_argument("--focus_qwen_fallback_to_api", action=argparse.BooleanOptionalAction,
                    default=general_config.get("focus_qwen_fallback_to_api"))
    ap.add_argument("--focus_qwen_frame_caption_words", type=int,
                    default=general_config.get("focus_qwen_frame_caption_words"))
    ap.add_argument("--focus_qwen_summary_words", type=int,
                    default=general_config.get("focus_qwen_summary_words"))
    ap.add_argument("--skim_long_window_s", type=float,
                    default=general_config.get("skim_long_window_s"))
    ap.add_argument("--skim_long_num_frames", type=int,
                    default=general_config.get("skim_long_num_frames"))
    ap.add_argument("--skim_long_target_step_s", type=float,
                    default=general_config.get("skim_long_target_step_s"))
    ap.add_argument("--skim_qwen_max_frames", type=int,
                    default=general_config.get("skim_qwen_max_frames"))
    ap.add_argument("--skim_qwen_temporal_density_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("skim_qwen_temporal_density_enabled"))
    ap.add_argument("--skim_qwen_target_fps", type=float,
                    default=general_config.get("skim_qwen_target_fps"))
    ap.add_argument("--skim_qwen_adaptive_density_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("skim_qwen_adaptive_density_enabled"))
    ap.add_argument("--skim_qwen_density_short_limit_s", type=float,
                    default=general_config.get("skim_qwen_density_short_limit_s"))
    ap.add_argument("--skim_qwen_density_mid_limit_s", type=float,
                    default=general_config.get("skim_qwen_density_mid_limit_s"))
    ap.add_argument("--skim_qwen_density_short_fps", type=float,
                    default=general_config.get("skim_qwen_density_short_fps"))
    ap.add_argument("--skim_qwen_density_mid_fps", type=float,
                    default=general_config.get("skim_qwen_density_mid_fps"))
    ap.add_argument("--skim_qwen_density_long_fps", type=float,
                    default=general_config.get("skim_qwen_density_long_fps"))
    ap.add_argument("--skim_qwen_density_batch_frames", type=int,
                    default=general_config.get("skim_qwen_density_batch_frames"))
    ap.add_argument("--skim_qwen_density_max_frames", type=int,
                    default=general_config.get("skim_qwen_density_max_frames"))
    ap.add_argument("--skim_qwen_short_side", type=int,
                    default=general_config.get("skim_qwen_short_side"))
    ap.add_argument("--skim_qwen_max_new_tokens", type=int,
                    default=general_config.get("skim_qwen_max_new_tokens"))
    ap.add_argument("--skim_qwen_recommended_max_window_s", type=float,
                    default=general_config.get("skim_qwen_recommended_max_window_s"))
    ap.add_argument("--skim_qwen_fallback_to_api", action=argparse.BooleanOptionalAction,
                    default=general_config.get("skim_qwen_fallback_to_api"))
    ap.add_argument("--skim_qwen_frame_caption_words", type=int,
                    default=general_config.get("skim_qwen_frame_caption_words"))
    ap.add_argument("--skim_qwen_relevant_detail_words", type=int,
                    default=general_config.get("skim_qwen_relevant_detail_words"))
    ap.add_argument("--skim_qwen_window_summary_words", type=int,
                    default=general_config.get("skim_qwen_window_summary_words"))
    ap.add_argument("--overview_sampling_mode", choices=["hybrid", "iframes"],
                    default=general_config.get("overview_sampling_mode"))
    ap.add_argument("--coseek1_visited_window_information_gain", action=argparse.BooleanOptionalAction,
                    default=general_config.get("coseek1_visited_window_information_gain"))
    ap.add_argument("--coseek1_visited_window_overlap", type=float,
                    default=general_config.get("coseek1_visited_window_overlap"))
    ap.add_argument("--use_skeleton_overview", action=argparse.BooleanOptionalAction,
                    default=general_config.get("use_skeleton_overview"))
    ap.add_argument("--scene_aware_skim", action=argparse.BooleanOptionalAction,
                    default=general_config.get("scene_aware_skim"))
    ap.add_argument("--skeleton_cache_dir", default=general_config.get("skeleton_cache_dir"))
    ap.add_argument("--skeleton_scene_min_s", type=float,
                    default=general_config.get("skeleton_scene_min_s"))
    ap.add_argument("--skeleton_scene_max_s", type=float,
                    default=general_config.get("skeleton_scene_max_s"))
    ap.add_argument("--skeleton_max_scenes", type=int,
                    default=general_config.get("skeleton_max_scenes"))
    ap.add_argument("--skeleton_anchors_per_scene", type=int,
                    default=general_config.get("skeleton_anchors_per_scene"))
    ap.add_argument("--skeleton_caption_enabled", action=argparse.BooleanOptionalAction,
                    default=general_config.get("skeleton_caption_enabled"))
    ap.add_argument("--skeleton_caption_strategy",
                    default=general_config.get("skeleton_caption_strategy"))
    ap.add_argument("--skeleton_caption_frames", type=int,
                    default=general_config.get("skeleton_caption_frames"))
    ap.add_argument("--skeleton_caption_frames_per_scene", type=int,
                    default=general_config.get("skeleton_caption_frames_per_scene"))
    ap.add_argument("--skeleton_caption_scenes_per_chunk", type=int,
                    default=general_config.get("skeleton_caption_scenes_per_chunk"))
    ap.add_argument("--skeleton_caption_retries", type=int,
                    default=general_config.get("skeleton_caption_retries"))
    ap.add_argument("--skeleton_caption_retry_failed_cache", action=argparse.BooleanOptionalAction,
                    default=general_config.get("skeleton_caption_retry_failed_cache"))
    ap.add_argument("--skeleton_caption_short_side", type=int,
                    default=general_config.get("skeleton_caption_short_side"))
    ap.add_argument("--skeleton_single_frame_short_side", type=int,
                    default=general_config.get("skeleton_single_frame_short_side"))
    ap.add_argument("--skeleton_single_frame_aggregate", action=argparse.BooleanOptionalAction,
                    default=general_config.get("skeleton_single_frame_aggregate"))
    ap.add_argument("--skeleton_mini_sheet_group_size", type=int,
                    default=general_config.get("skeleton_mini_sheet_group_size"))
    ap.add_argument("--skeleton_mini_sheet_short_side", type=int,
                    default=general_config.get("skeleton_mini_sheet_short_side"))
    ap.add_argument("--skeleton_overview_max_scenes", type=int,
                    default=general_config.get("skeleton_overview_max_scenes"))
    args = ap.parse_args()

    subset_path = MLVU_ROOT / args.subset
    if not subset_path.is_file():
        print(f"ERROR: subset not found: {subset_path}", file=sys.stderr)
        return 2
    data = json.load(open(subset_path))
    type_filter = set(t.strip() for t in args.types.split(",")) if args.types else None

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{subset_path.stem}_{int(time.time())}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = make_config(args)
    print(
        f"[init] version={config.get('coseek_version', 'unknown')}  "
        f"model={config['model_name']}  api_base={config['api_base']}"
    )
    print(f"[init] run_dir={run_dir}")
    print(f"[init] subset={subset_path.name}  videos={len(data)}")

    results: list[dict] = []
    n_done = n_correct = n_err = 0
    usage_log_value = os.environ.get("COSEEK_USAGE_LOG")
    usage_log_path = Path(usage_log_value).expanduser() if usage_log_value else None

    for item in data:
        video_rel = item["video_path"]
        video_abs = (MLVU_ROOT.parent.parent / video_rel)
        if not video_abs.is_file():
            video_abs = MLVU_ROOT / video_rel.split("data/mlvu/", 1)[-1]
        if not video_abs.is_file():
            print(f"[skip] missing video: {video_rel}")
            continue

        for q in item["conversations"]:
            qtype = q.get("question_type", "")
            if type_filter and qtype not in type_filter:
                continue
            if args.limit is not None and n_done >= args.limit:
                break

            query = build_query(q["question"], q["choices"])
            gt = gt_letter(q["answer"], q["choices"])
            qid = f"{item['video_id']}::{qtype}::{n_done}"
            t0 = time.time()
            usage_start = len(read_api_usage_log(usage_log_path))
            print(f"\n[{n_done+1}] {qid}  dur={item.get('duration')}s")
            print(f"     Q: {q['question'][:80]}")
            print(f"     GT: {gt} ({q['answer']})")

            try:
                agent = VideoSeekAgent(
                    config=config,
                    video_path=str(video_abs),
                    subtitle_path=None,
                    output_dir=str(run_dir),
                    tools=config["tools"],
                    verbose=args.verbose,
                )
                traj = agent.run(query)
                traj_dict = traj.to_dict()
                traj_path = run_dir / f"trajectory_{safe_name(qid)}.json"
                traj_path.write_text(json.dumps(traj_dict, ensure_ascii=False, indent=2))
                pred_raw = traj_dict.get("final_answer", "")
                global_state = (traj_dict.get("memory") or {}).get("p130_global")
                if isinstance(global_state, dict) and "answer_decision" in global_state:
                    pred = global_state["answer_decision"].get("selected_option") or ""
                    if pred_raw != pred:
                        raise ValueError("global terminal serialization differs from decision")
                else:
                    pred = normalize_pred(pred_raw, q["choices"])
                ok = (pred == gt)
                memory = traj_dict.get("memory") or {}
                overview_execution = memory.get("overview_execution_state") or {}
                overview_receipt = overview_execution.get("valid_receipt") or {}
                receipt_attempt_id = str(
                    overview_receipt.get("attempt_id") or ""
                )
                try:
                    receipt_attempt_number = int(receipt_attempt_id.removeprefix("OA"))
                except (TypeError, ValueError):
                    receipt_attempt_number = 0
                post_success_overview_visual_request_count = 0
                for overview_attempt in overview_execution.get("attempts") or []:
                    if not isinstance(overview_attempt, dict):
                        continue
                    attempt_id = str(overview_attempt.get("attempt_id") or "")
                    try:
                        attempt_number = int(attempt_id.removeprefix("OA"))
                    except (TypeError, ValueError):
                        continue
                    if receipt_attempt_number and attempt_number > receipt_attempt_number:
                        post_success_overview_visual_request_count += 1
                actions = [
                    ((step.get("action") or {}).get("function"))
                    for step in traj_dict.get("steps", [])
                ]
                observer_usage = summarize_observer_usage(traj_dict)
                episode_state = memory.get("evidence_episode_frontier") or {}
                answer_audit = memory.get("answer_evidence_audit") or {}
                capsule_audits = [
                    row
                    for row in memory.get("planner_capsule_audit") or []
                    if isinstance(row, dict)
                ]
                capsule_full_tokens = sum(
                    int(row.get("full_memory_estimated_tokens") or 0)
                    for row in capsule_audits
                )
                capsule_projected_tokens = sum(
                    int(row.get("capsule_estimated_tokens") or 0)
                    for row in capsule_audits
                )
                n_correct += int(ok)
                rec = {
                    "qid": qid,
                    "video_id": item["video_id"],
                    "question_type": qtype,
                    "question": q["question"],
                    "choices": q["choices"],
                    "gt_letter": gt,
                    "gt_answer": q["answer"],
                    "pred_letter": pred,
                    "pred_raw": pred_raw,
                    "correct": ok,
                    "steps": traj_dict.get("total_steps"),
                    "finish_reason": traj_dict.get("finish_reason"),
                    "actions": actions,
                    **observer_usage,
                    "timestamped_count": len(memory.get("timestamped_observations") or []),
                    "scene_memory_count": len(memory.get("scene_memory") or []),
                    "open_gaps_count": len(memory.get("open_gaps") or []),
                    "temporal_evidence_event_count": len(
                        (memory.get("temporal_evidence_ledger") or {}).get("events") or []
                    ),
                    "verified_observation_reuse_count": int(
                        (memory.get("verified_observation_reuse_stats") or {}).get(
                            "reuse_count"
                        )
                        or 0
                    ),
                    "runtime_config": memory.get("runtime_config") or {},
                    "valid_overview_receipt_count": int(bool(overview_receipt)),
                    "overview_attempt_count": int(
                        overview_execution.get("attempt_count") or 0
                    ),
                    "failed_overview_attempt_count": int(
                        overview_execution.get("failed_attempt_count") or 0
                    ),
                    "post_success_overview_proposal_count": int(
                        overview_execution.get("blocked_overview_count") or 0
                    ),
                    "post_success_overview_blocked_count": int(
                        overview_execution.get("blocked_overview_count") or 0
                    ),
                    "post_success_overview_visual_request_count": (
                        post_success_overview_visual_request_count
                    ),
                    "evidence_episode_count": len(episode_state.get("episodes") or []),
                    "event_boundary_count": len(episode_state.get("event_boundaries") or []),
                    "answer_support_level": answer_audit.get("answer_support_level"),
                    "answer_verified_support_count": len(
                        answer_audit.get("answer_verified_support_refs") or []
                    ),
                    "uninspected_high_value_episode_count": len(
                        answer_audit.get("uninspected_high_value_episode_ids") or []
                    ),
                    "answer_relation_verified": answer_audit.get("relation_verified"),
                    "planner_capsule_enabled": bool(
                        config.get("coseek1_planner_evidence_capsule_enabled")
                    ),
                    "planner_capsule_audit_count": len(capsule_audits),
                    "planner_capsule_activation_count": sum(
                        row.get("activated") is True for row in capsule_audits
                    ),
                    "planner_capsule_full_memory_estimated_tokens": capsule_full_tokens,
                    "planner_capsule_projected_memory_tokens": capsule_projected_tokens,
                    "planner_capsule_memory_reduction": round(
                        1.0
                        - capsule_projected_tokens / max(1, capsule_full_tokens),
                        6,
                    ) if capsule_audits else 0.0,
                    "planner_capsule_actual_prompt_tokens": sum(
                        int(row.get("actual_planner_prompt_tokens") or 0)
                        for row in capsule_audits
                    ),
                    "planner_capsule_fail_open_count": sum(
                        row.get("fail_open") is True for row in capsule_audits
                    ),
                    "planner_capsule_invalid_reference_count": sum(
                        len(row.get("invalid_references") or [])
                        for row in capsule_audits
                    ),
                    "planner_capsule_omitted_decisive_count": sum(
                        len(row.get("omitted_decisive_support_ids") or [])
                        for row in capsule_audits
                    ),
                    "trajectory_path": str(traj_path),
                    "duration_s": item.get("duration"),
                    "wall_s": round(time.time() - t0, 1),
                }
            except Exception as e:
                n_err += 1
                rec = {
                    "qid": qid, "video_id": item["video_id"],
                    "question_type": qtype, "question": q["question"],
                    "choices": q["choices"], "gt_letter": gt, "gt_answer": q["answer"],
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                    "wall_s": round(time.time() - t0, 1),
                }
                print(f"     ERROR: {rec['error']}")

            usage_rows = read_api_usage_log(usage_log_path)[usage_start:]
            rec.update(summarize_api_usage(usage_rows))
            rec["api_usage_rows"] = usage_rows
            n_done += 1
            results.append(rec)
            (run_dir / "results.jsonl").open("a").write(
                json.dumps(rec, ensure_ascii=False) + "\n"
            )
            print(f"     pred={rec.get('pred_letter','?')}  "
                  f"correct={rec.get('correct')}  "
                  f"steps={rec.get('steps')}  wall={rec['wall_s']}s  "
                  f"api_obs={rec.get('api_observer_count', 0)}  "
                  f"qwen_obs={rec.get('local_qwen_observer_count', 0)}  "
                  f"api_calls={rec.get('api_request_count', 0)}  "
                  f"tokens={rec.get('api_total_tokens', 0)}")

        if args.limit is not None and n_done >= args.limit:
            break

    n_eval = n_done - n_err
    acc = (n_correct / n_eval) if n_eval else 0.0
    summary = {
        "coseek_version": config.get("coseek_version"),
        "subset": subset_path.name, "model": config["model_name"],
        "qwen_caption_only_output": bool(config.get("qwen_caption_only_output")),
        "overview_sampling_mode": config.get("overview_sampling_mode"),
        "dual_path_overview": {
            "enabled": bool(config.get("dual_path_overview_enabled")),
            "shadow_only": bool(config.get("dual_path_overview_shadow_only")),
            "integration_mode": config.get("dual_path_integration_mode"),
            "local_frames": config.get("dual_path_local_frames"),
            "local_min_frames": config.get("dual_path_local_min_frames"),
            "local_max_frames": config.get("dual_path_local_max_frames"),
            "local_min_processed_frames": config.get(
                "dual_path_local_min_processed_frames"
            ),
            "local_target_interval_s": config.get(
                "dual_path_local_target_interval_s"
            ),
            "local_batch_size": config.get("dual_path_local_batch_size"),
            "local_top_k": config.get("dual_path_local_top_k"),
            "local_candidate_pool_k": config.get(
                "dual_path_local_candidate_pool_k"
            ),
            "fused_top_k": config.get("dual_path_fused_top_k"),
            "semantic_prototype_ranking": bool(
                config.get("dual_path_semantic_prototype_ranking_enabled")
            ),
            "planner_caption_limit": config.get(
                "dual_path_planner_caption_limit"
            ),
            "local_deadline_s": config.get("dual_path_local_deadline_s"),
        },
        "focus_qwen_temporal_density_enabled": bool(
            config.get("focus_qwen_temporal_density_enabled")
        ),
        "focus_qwen_target_fps": config.get("focus_qwen_target_fps"),
        "focus_qwen_max_frames": config.get("focus_qwen_max_frames"),
        "skim_qwen_temporal_density_enabled": bool(
            config.get("skim_qwen_temporal_density_enabled")
        ),
        "skim_qwen_target_fps": config.get("skim_qwen_target_fps"),
        "skim_qwen_adaptive_density_enabled": bool(
            config.get("skim_qwen_adaptive_density_enabled")
        ),
        "skim_qwen_density_fps": {
            "short": config.get("skim_qwen_density_short_fps"),
            "mid": config.get("skim_qwen_density_mid_fps"),
            "long": config.get("skim_qwen_density_long_fps"),
        },
        "skim_qwen_density_batch_frames": config.get("skim_qwen_density_batch_frames"),
        "skim_qwen_density_max_frames": config.get("skim_qwen_density_max_frames"),
        "coseek1_direct_planner_answer": bool(
            config.get("coseek1_direct_planner_answer")
        ),
        "planner_evidence_capsule": {
            "enabled": bool(
                config.get("coseek1_planner_evidence_capsule_enabled")
            ),
            "token_budget": config.get("coseek1_planner_capsule_token_budget"),
            "max_verified": config.get("coseek1_planner_capsule_max_verified"),
            "max_candidates": config.get("coseek1_planner_capsule_max_candidates"),
            "max_obligations": config.get("coseek1_planner_capsule_max_obligations"),
            "audit_enabled": bool(
                config.get("coseek1_planner_capsule_audit_enabled")
            ),
        },
        "valid_overview_receipt_guard": {
            "enabled": bool(
                config.get("coseek1_valid_overview_receipt_guard_enabled")
            ),
            "schema_version": "p105_overview_admissibility_v1",
        },
        "coseek1_localize_qwen_enabled": bool(
            config.get("coseek1_localize_qwen_enabled")
        ),
        "coseek1_localize_inline_verify": bool(
            config.get("coseek1_localize_inline_verify")
        ),
        "coseek1_retrieval_verify_semantic_decoupling_enabled": bool(
            config.get("coseek1_retrieval_verify_semantic_decoupling_enabled")
        ),
        "coseek1_compact_event_coverage": bool(
            config.get("coseek1_compact_event_coverage")
        ),
        "coseek1_compact_event_coverage_max_items": config.get(
            "coseek1_compact_event_coverage_max_items"
        ),
        "coseek1_event_coverage_inline_verify": bool(
            config.get("coseek1_event_coverage_inline_verify")
        ),
        "coseek1_event_coverage_inline_verify_max_windows": config.get(
            "coseek1_event_coverage_inline_verify_max_windows"
        ),
        "localize_inline_verify_budget": {
            "max_windows": config.get("localize_inline_verify_max_windows"),
            "max_frames": config.get("localize_inline_verify_max_frames"),
            "contact_sheets": bool(
                config.get("localize_inline_verify_contact_sheets")
            ),
        },
        "multiwindow_verify_recovery": {
            "enabled": bool(config.get("multiwindow_verify_recovery_enabled")),
            "max_calls": config.get("multiwindow_verify_recovery_max_calls"),
            "batch_size": config.get("multiwindow_verify_recovery_batch_size"),
        },
        "grounded_candidate_binding_enabled": bool(
            config.get("grounded_candidate_binding_enabled")
        ),
        "grounded_candidate_event_binding_enabled": bool(
            config.get("grounded_candidate_event_binding_enabled")
        ),
        "grounded_candidate_event_group_max_gap_s": config.get(
            "grounded_candidate_event_group_max_gap_s"
        ),
        "packet_decision_consistency_normalization_enabled": bool(
            config.get("packet_decision_consistency_normalization_enabled")
        ),
        "packet_multiwindow_decision_sufficiency_enabled": bool(
            config.get("packet_multiwindow_decision_sufficiency_enabled")
        ),
        "packet_preserve_resolved_candidate_set_decision_enabled": bool(
            config.get("packet_preserve_resolved_candidate_set_decision_enabled")
        ),
        "structured_evidence_preserve_aggregate_verifier_decision": bool(
            config.get("structured_evidence_preserve_aggregate_verifier_decision")
        ),
        "grounded_verify_packet": {
            "enabled": bool(config.get("grounded_verify_packet_enabled")),
            "high_recall_proposal": bool(
                config.get("grounded_high_recall_proposal_enabled")
            ),
            "anchors_per_candidate": config.get(
                "grounded_verify_packet_anchors_per_candidate"
            ),
            "temporal_anchor_spread": bool(
                config.get("grounded_verify_packet_temporal_anchor_spread_enabled")
            ),
            "query_context_anchor_roles": bool(
                config.get("grounded_verify_packet_query_context_roles_enabled")
            ),
            "anchor_audit": bool(
                config.get("grounded_verify_packet_anchor_audit_enabled")
            ),
            "crop_max_side": config.get("grounded_verify_packet_crop_max_side"),
            "peak_max_side": config.get("grounded_verify_packet_peak_max_side"),
            "adaptive_detail_grounding": bool(
                config.get("packet_detail_grounding_escalation_enabled")
            ),
            "adaptive_detail_frames": config.get(
                "packet_detail_grounding_escalation_frames"
            ),
        },
        "grounded_direct_verify_packet": {
            "enabled": bool(config.get("grounded_direct_verify_packet_enabled")),
            "group_size": config.get("grounded_direct_verify_packet_group_size"),
            "anchor_count": config.get("grounded_direct_verify_packet_anchor_count"),
            "detail_max_side": config.get(
                "grounded_direct_verify_packet_detail_max_side"
            ),
        },
        "localize_qwen_budget": {
            "top_k": config.get("localize_qwen_top_k"),
            "max_top_k": config.get("localize_qwen_max_top_k"),
            "row_candidate_windows": bool(
                config.get("localize_qwen_row_candidate_windows")
            ),
            "cover_search_windows": bool(
                config.get("localize_qwen_cover_search_windows")
            ),
            "normalized_candidate_score": bool(
                config.get("localize_qwen_normalized_candidate_score_enabled")
            ),
            "search_context_margin_s": config.get(
                "localize_qwen_search_context_margin_s"
            ),
            "adaptive_search_context": bool(
                config.get("localize_qwen_adaptive_search_context")
            ),
            "duration_budget": bool(
                config.get("localize_qwen_duration_budget_enabled")
            ),
            "duration_budget_fps": config.get("localize_qwen_duration_budget_fps"),
            "codec_anchors": bool(
                config.get("localize_qwen_codec_anchor_sampling_enabled")
            ),
            "boundary_expansion": bool(
                config.get("localize_qwen_boundary_expansion_enabled")
            ),
            "caption_cache": bool(
                config.get("localize_qwen_caption_cache_enabled")
            ),
            "candidate_novelty": bool(
                config.get("localize_qwen_candidate_novelty_enabled")
            ),
            "strict_evidence_status": bool(
                config.get("localize_qwen_strict_evidence_status_enabled")
            ),
            "coarse_frames": config.get("localize_qwen_coarse_max_frames"),
            "fine_frames": config.get("localize_qwen_fine_max_frames"),
        },
        "tool_integrity_repair": {
            "enabled": bool(config.get("coseek1_tool_integrity_repair_enabled")),
            "memory_anchor_margin_s": config.get(
                "coseek1_tool_memory_anchor_margin_s"
            ),
            "memory_anchors_per_window": config.get(
                "coseek1_tool_memory_anchors_per_window"
            ),
            "memory_anchor_pad_s": config.get("coseek1_tool_memory_anchor_pad_s"),
            "event_identity_repair": bool(
                config.get("coseek1_tool_event_identity_repair_enabled")
            ),
            "recollect_expanded_anchors": bool(
                config.get("coseek1_tool_recollect_expanded_anchors_enabled")
            ),
            "packet_boundary_context": bool(
                config.get("grounded_verify_packet_boundary_context_enabled")
            ),
        },
        "coseek1_visited_window_information_gain": bool(
            config.get("coseek1_visited_window_information_gain")
        ),
        "coseek1_candidate_frontier": bool(
            config.get("coseek1_candidate_frontier")
        ),
        "coseek1_evidence_episode_frontier": bool(
            config.get("coseek1_evidence_episode_frontier")
        ),
        "coseek1_episode_backend_router": bool(
            config.get("coseek1_episode_backend_router")
        ),
        "coseek1_answer_evidence_audit": bool(
            config.get("coseek1_answer_evidence_audit")
        ),
        "coseek1_frontier_timestamp_cues": bool(
            config.get("coseek1_frontier_timestamp_cues")
        ),
        "coseek1_frontier_evidence_scope": bool(
            config.get("coseek1_frontier_evidence_scope")
        ),
        "coseek1_frontier_temporal_boundaries": bool(
            config.get("coseek1_frontier_temporal_boundaries")
        ),
        "coseek1_frontier_investigation_state": bool(
            config.get("coseek1_frontier_investigation_state")
        ),
        "coseek1_intent_preserving_router": bool(
            config.get("coseek1_intent_preserving_router")
        ),
        "coseek1_candidate_context_localize_enabled": bool(
            config.get("coseek1_candidate_context_localize_enabled")
        ),
        "coseek1_candidate_evidence_projection": bool(
            config.get("coseek1_candidate_evidence_projection")
        ),
        "coseek1_temporal_evidence_ledger_enabled": bool(
            config.get("coseek1_temporal_evidence_ledger_enabled")
        ),
        "coseek1_verified_observation_reuse_enabled": bool(
            config.get("coseek1_verified_observation_reuse_enabled")
        ),
        "local_qwen_model_path": config.get("local_qwen_model_path"),
        "skim_qwen_frame_caption_words": config.get("skim_qwen_frame_caption_words"),
        "focus_qwen_frame_caption_words": config.get("focus_qwen_frame_caption_words"),
        "v36_runtime_config": {
            "coseek1_scope_aware_evidence_memory": bool(
                config.get("coseek1_scope_aware_evidence_memory")
            ),
            "coseek1_separate_routing_candidates": bool(
                config.get("coseek1_separate_routing_candidates")
            ),
            "coseek1_persistent_open_gaps": bool(
                config.get("coseek1_persistent_open_gaps")
            ),
            "coseek1_query_relevant_memory_retention": bool(
                config.get("coseek1_query_relevant_memory_retention")
            ),
            "coseek1_persistent_candidate_pool": bool(
                config.get("coseek1_persistent_candidate_pool")
            ),
            "coseek1_persistent_candidate_pool_max_items": config.get(
                "coseek1_persistent_candidate_pool_max_items"
            ),
            "coseek1_persistent_candidate_pool_timestamp_radius_s": config.get(
                "coseek1_persistent_candidate_pool_timestamp_radius_s"
            ),
            "coseek1_candidate_evidence_projection": bool(
                config.get("coseek1_candidate_evidence_projection")
            ),
            "focus_qwen_multi_window_enabled": bool(
                config.get("focus_qwen_multi_window_enabled")
            ),
            "focus_qwen_multi_window_max_windows": config.get(
                "focus_qwen_multi_window_max_windows"
            ),
            "overview_require_all_timestamps": bool(
                config.get("overview_require_all_timestamps")
            ),
            "overview_fill_omitted_timestamps": bool(
                config.get("overview_fill_omitted_timestamps")
            ),
            "overview_frame_caption_words": config.get(
                "overview_frame_caption_words"
            ),
            "overview_frame_short_side": config.get("overview_frame_short_side"),
            "overview_embed_timestamp_labels": bool(
                config.get("overview_embed_timestamp_labels")
            ),
            "overview_contact_sheet_group_size": config.get(
                "overview_contact_sheet_group_size"
            ),
            "overview_sampling_mode": config.get("overview_sampling_mode"),
            "focus_qwen_temporal_density_enabled": bool(
                config.get("focus_qwen_temporal_density_enabled")
            ),
            "focus_qwen_target_fps": config.get("focus_qwen_target_fps"),
            "focus_qwen_max_frames": config.get("focus_qwen_max_frames"),
        },
        "n_done": n_done, "n_correct": n_correct, "n_err": n_err,
        "accuracy": acc,
        "api_usage": {
            "request_count": sum(int(row.get("api_request_count") or 0) for row in results),
            "visual_request_count": sum(
                int(row.get("api_visual_request_count") or 0) for row in results
            ),
            "text_request_count": sum(
                int(row.get("api_text_request_count") or 0) for row in results
            ),
            "image_count": sum(int(row.get("api_image_count") or 0) for row in results),
            "prompt_tokens": sum(int(row.get("api_prompt_tokens") or 0) for row in results),
            "completion_tokens": sum(
                int(row.get("api_completion_tokens") or 0) for row in results
            ),
            "total_tokens": sum(int(row.get("api_total_tokens") or 0) for row in results),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
