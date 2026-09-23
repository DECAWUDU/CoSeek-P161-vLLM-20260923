from __future__ import annotations

import unittest
from unittest.mock import patch

from videoseek.agent import VideoSeekAgent
from videoseek.core import Action
from videoseek.core.memory import (
    build_compact_investigation_state,
    extract_v10_payload,
    format_compact_investigation_state_for_prompt,
    init_observation_memory,
    merge_tool_observation,
)
from videoseek.tools.frame_verify import (
    _apply_reported_decision_completeness,
    _apply_event_binding_option_scope,
    _candidate_event_groups,
    _should_apply_candidate_event_binding,
    _demote_partial_global_option_claims,
    _normalize_candidate_assessments,
    _normalize_decision_completeness,
    _normalize_evidence_need,
    _normalize_evidence_needs,
    _packet_detail_anchor_positions,
    _execute_multiwindow_frame_verify_once,
    execute_frame_verify,
    execute_multiwindow_frame_verify,
)
from videoseek.tools.focus import _question_scope_hint
from videoseek.tools.localize_qwen import (
    _candidate_score,
    _candidate_score_components,
    _effective_search_context_margin,
    _expand_search_windows,
    _expand_verify_window,
    _observer_health,
    _rank_coarse_candidates,
    _verification_anchors,
)
from videoseek.tools.skim_qwen import _execute_density_batched_skim_qwen
from videoseek.tools.v10_format import format_v10_observation

from test_packet_adaptive_grounding import (
    _VideoReaderStub,
    _config,
    _detail_response,
)


def _direct_parameters() -> dict:
    return {
        "vr": _VideoReaderStub(),
        "video_path": "/tmp/video.mp4",
        "duration": 100.0,
        "output_dir": "/tmp",
        "subtitles": [],
        "question": "What is held?\n(A) cup\n(B) flower\n(C) book\n(D) phone",
        "query": "Neutrally inspect what the person holds.",
        "start_time": 70.0,
        "end_time": 76.0,
        "mode": "option_verify",
    }


def _direct_initial(*, evidence_need: str, with_detail_target: bool = True) -> str:
    payload = {
        "tool": "frame_verify",
        "scene_id": "S007",
        "window_id": "S007_focus_window",
        "t_range": [70.0, 76.0],
        "timestamp_observations": [
            {
                "timestamp_s": 73.0,
                "scene_id": "S007",
                "description": "A hand holds a small stem-like object.",
            }
        ],
        "overall_summary": "The target moment is found, but the object is small.",
        "observed_fact": "The hand holds a small object.",
        "target_match": "partial",
        "target_binding_reason": "The hand and object are co-visible.",
        "question_scope": "local_window",
        "scope_coverage": "sufficient",
        "supports_options": [],
        "contradicts_options": [],
        "option_evidence": {},
        "detail_sufficient": False,
        "missing_detail": "The small held object cannot be identified at this resolution.",
        "evidence_need": evidence_need,
        "observer_backend": "api",
        "parse_ok": True,
    }
    if with_detail_target:
        payload["detail_target"] = {"timestamp_s": 73.0}
        payload["detail_query"] = (
            "Inspect the held object's shape without assuming an option."
        )
    return format_v10_observation(payload)


def _memory_output(
    *,
    span: list[float],
    detail_sufficient: bool,
    missing_detail: str,
    target_match: str,
    scope_coverage: str,
    evidence_need: str,
    supports: list[str] | None = None,
    contradicts: list[str] | None = None,
) -> str:
    return format_v10_observation(
        {
            "tool": "frame_verify",
            "scene_id": "S001",
            "window_id": "W001",
            "t_range": span,
            "overall_summary": "Neutral verified observation.",
            "observed_fact": "Neutral verified observation.",
            "target_match": target_match,
            "question_scope": "local_window",
            "scope_coverage": scope_coverage,
            "detail_sufficient": detail_sufficient,
            "missing_detail": missing_detail,
            "evidence_need": evidence_need,
            "supports_options": supports or [],
            "contradicts_options": contradicts or [],
            "observer_backend": "api",
            "parse_ok": True,
        }
    )


class PacketAdaptiveRobustV2Tests(unittest.TestCase):
    def test_multiwindow_decision_sufficiency_preserves_optional_detail(self):
        payload = {
            "decision_sufficient": True,
            "detail_sufficient": False,
            "candidate_coverage_complete": True,
            "candidate_binding_complete": True,
            "target_match": "matched",
            "scope_coverage": "sufficient",
            "supports_options": ["C"],
            "contradicts_options": ["A", "B", "D"],
            "evidence_need": "temporal_coverage",
            "missing_detail": (
                "A direct arrival shot would make the already discriminated motive more explicit."
            ),
        }
        self.assertTrue(
            _apply_reported_decision_completeness(payload, candidate_count=3)
        )
        self.assertFalse(payload["detail_sufficient"])
        self.assertFalse(payload["visual_detail_sufficient"])
        self.assertEqual(payload["evidence_need"], "temporal_coverage")
        self.assertEqual(payload["decision_evidence_need"], "sufficient")
        self.assertTrue(payload["decision_sufficiency_validated"])
        self.assertIn("direct arrival shot", payload["residual_visual_detail"])

    def test_decision_prompt_explains_different_event_option_binding(self):
        source = __import__("inspect").getsource(
            _execute_multiwindow_frame_verify_once
        )
        self.assertIn("belongs to a confirmed different_event", source)
        self.assertIn("use unresolved only when", source)

    def test_multiwindow_decision_sufficiency_rejects_unsafe_inputs(self):
        base = {
            "decision_sufficient": True,
            "detail_sufficient": False,
            "candidate_coverage_complete": True,
            "candidate_binding_complete": True,
            "target_match": "matched",
            "scope_coverage": "sufficient",
            "supports_options": ["C"],
            "contradicts_options": ["A", "B", "D"],
        }
        self.assertFalse(
            _apply_reported_decision_completeness(dict(base), candidate_count=1)
        )
        incomplete = dict(base, candidate_binding_complete=False)
        self.assertFalse(
            _apply_reported_decision_completeness(incomplete, candidate_count=3)
        )
        conflicted = dict(base, option_set_conflict=True)
        self.assertFalse(
            _apply_reported_decision_completeness(conflicted, candidate_count=3)
        )
        uncovered = dict(base, contradicts_options=["A", "B"])
        self.assertFalse(
            _apply_reported_decision_completeness(uncovered, candidate_count=3)
        )

    def test_validated_multiwindow_decision_closes_query_gap_without_hiding_detail(self):
        question = (
            "Why did the girl run downstairs?\n"
            "(A) To leave school\n(B) To meet a friend\n"
            "(C) Because trouble was happening outside\n(D) To get lunch"
        )
        parameters = {
            "candidate_windows": [
                {"candidate_id": "C001", "t_range": [249.0, 257.7]},
                {"candidate_id": "C002", "t_range": [263.0, 279.0]},
                {"candidate_id": "C003", "t_range": [284.3, 300.3]},
            ]
        }
        payload = {
            "tool": "frame_verify",
            "t_range": [249.0, 300.3],
            "decision_sufficient": True,
            "detail_sufficient": False,
            "candidate_coverage_complete": True,
            "candidate_binding_complete": True,
            "target_match": "matched",
            "scope_coverage": "sufficient",
            "supports_options": ["C"],
            "contradicts_options": ["A", "B", "D"],
            "evidence_need": "temporal_coverage",
            "missing_detail": (
                "A direct arrival shot would make the already supported motive explicit."
            ),
            "candidate_assessments": [
                {"candidate_id": "C001", "target_match": "matched"},
                {"candidate_id": "C002", "target_match": "matched"},
                {"candidate_id": "C003", "target_match": "matched"},
            ],
            "timestamp_observations": [
                {"candidate_id": "C001", "timestamp_s": 253.0, "description": "She watches trouble outside."},
                {"candidate_id": "C002", "timestamp_s": 271.0, "description": "She runs downstairs urgently."},
                {"candidate_id": "C003", "timestamp_s": 292.0, "description": "The trouble outside is shown."},
            ],
            "observed_fact": "The three windows bind the observation, urgent run, and outside trouble.",
            "observer_backend": "api",
            "parse_ok": True,
        }
        self.assertTrue(
            _apply_reported_decision_completeness(payload, candidate_count=3)
        )
        memory = init_observation_memory()
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters=parameters,
            output=format_v10_observation(payload),
            use_structured_evidence_state=True,
            question_context=question,
            scope_aware_evidence_memory=True,
            persistent_open_gaps=True,
            semantic_gap_resolution=True,
            preserve_aggregate_verifier_decision=True,
        )

        self.assertFalse(payload["detail_sufficient"])
        self.assertEqual(memory["open_gaps"], [])
        aggregate = [
            item
            for item in memory["structured_evidence"]["evidence_items"]
            if item["refs"]["source_kind"] == "aggregate_verifier_decision"
        ]
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["evidence_level"], "verified")
        self.assertTrue(aggregate[0]["decision_sufficient"])
        self.assertIn("direct arrival shot", aggregate[0]["residual_visual_detail"])
        compact = build_compact_investigation_state(
            memory,
            question=question,
            scope_aware_evidence_memory=True,
        )
        self.assertEqual(compact["missing_context"], [])
        self.assertEqual(compact["answer_status"]["option"], "C")

    def test_verified_window_supersedes_only_covered_candidate_uncertainty(self):
        memory = init_observation_memory()
        question = "What happens?\n(A) fall\n(B) jump\n(C) run\n(D) stop"
        localize_payload = {
            "tool": "localize_qwen",
            "scene_id": "S001",
            "t_range": [10.0, 30.0],
            "timestamp_observations": [
                {
                    "timestamp_s": 15.0,
                    "description": "A possible action is visible.",
                }
            ],
            "missing_detail": "Use frame_verify for the final visual decision.",
            "observer_backend": "local_qwen",
            "parse_ok": True,
        }
        merge_tool_observation(
            memory,
            tool_name="localize_qwen",
            parameters={"start_time": 10.0, "end_time": 30.0},
            output=format_v10_observation(localize_payload),
            use_structured_evidence_state=True,
            question_context=question,
            scope_aware_evidence_memory=True,
            persistent_open_gaps=True,
            semantic_gap_resolution=True,
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 10.0, "end_time": 15.0},
            output=_memory_output(
                span=[10.0, 15.0],
                detail_sufficient=True,
                missing_detail="The rest of the candidate range is not inspected.",
                target_match="matched",
                scope_coverage="partial",
                evidence_need="temporal_coverage",
            ),
            use_structured_evidence_state=True,
            question_context=question,
            scope_aware_evidence_memory=True,
            persistent_open_gaps=True,
            semantic_gap_resolution=True,
        )
        partial = build_compact_investigation_state(
            memory,
            question=question,
            scope_aware_evidence_memory=True,
        )
        self.assertTrue(partial["missing_context"])

        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 15.0, "end_time": 30.0},
            output=_memory_output(
                span=[15.0, 30.0],
                detail_sufficient=True,
                missing_detail="",
                target_match="matched",
                scope_coverage="sufficient",
                evidence_need="sufficient",
            ),
            use_structured_evidence_state=True,
            question_context=question,
            scope_aware_evidence_memory=True,
            persistent_open_gaps=True,
            semantic_gap_resolution=True,
        )
        covered = build_compact_investigation_state(
            memory,
            question=question,
            scope_aware_evidence_memory=True,
        )
        self.assertFalse(
            any(
                item.get("source") == "localize_qwen"
                or str(item.get("source") or "").startswith("E00001")
                for item in covered["missing_context"]
            )
        )
        self.assertTrue(
            any(item.get("source") == "frame_verify" for item in covered["missing_context"])
        )

    def _candidate_projection_memory(self) -> tuple[dict, str]:
        question = (
            "Why does the girl with glasses run downstairs in a hurry?\n"
            "(A) Chasing someone\n(B) Being chased\n"
            "(C) To help another girl out of a situation\n(D) To get something"
        )
        memory = init_observation_memory()
        overview = format_v10_observation(
            {
                "tool": "overview",
                "observer_backend": "api",
                "timestamp_observations": [
                    {
                        "timestamp_s": 263.3,
                        "scene_id": "pe_area_conflict",
                        "description": "The girl with glasses walks along an indoor corridor.",
                    },
                    {
                        "timestamp_s": 271.0,
                        "scene_id": "pe_area_conflict",
                        "description": "She tells a boy 'Follow me!' as they move along a marble wall.",
                    },
                    {
                        "timestamp_s": 277.7,
                        "scene_id": "pe_area_conflict",
                        "description": "A group of girls stand outside watching a tense exchange.",
                    },
                    {
                        "timestamp_s": 410.9,
                        "scene_id": "stairs_letter_scene",
                        "description": "Girls face boys on a stairwell.",
                        "needs_focus": "Verify the hurried movement and reason.",
                    },
                    {
                        "timestamp_s": 420.9,
                        "scene_id": "stairs_letter_scene",
                        "description": "The girl with glasses reads a letter on the landing.",
                        "needs_focus": "Verify the hurried movement and reason.",
                    },
                ],
                "scene_summaries": [
                    {
                        "scene_id": "pe_area_conflict",
                        "t_range": [263.3, 284.8],
                        "summary": "The girl leads a boy and joins girls during a tense exchange.",
                        "possible_evidence": False,
                        "suggest_focus_windows": [[271.0, 277.7]],
                        "missing_detail": "Exact cause of tension is unclear.",
                    },
                    {
                        "scene_id": "stairs_letter_scene",
                        "t_range": [400.9, 430.9],
                        "summary": "The girl moves on stairs and later reads a letter.",
                        "possible_evidence": True,
                        "suggest_focus_windows": [[410.9, 420.9], [420.9, 430.9]],
                        "missing_detail": "Verify the exact hurried movement and reason.",
                    },
                ],
            }
        )
        merge_tool_observation(
            memory,
            tool_name="overview",
            parameters={},
            output=overview,
        )
        for index, span in enumerate(
            ([410.7, 430.7], [402.9, 430.9], [408.7, 426.7], [423.9, 430.9])
        ):
            merge_tool_observation(
                memory,
                tool_name="frame_verify" if index == 0 else "localize_qwen",
                parameters={"start_time": span[0], "end_time": span[1]},
                output=format_v10_observation(
                    {
                        "tool": "frame_verify" if index == 0 else "localize_qwen",
                        "scene_id": "stairs_letter_scene",
                        "t_range": span,
                        "overall_summary": "The letter sequence is visible, but the run cause is not.",
                        "possible_evidence": True,
                        "detail_sufficient": False,
                        "missing_detail": (
                            "The exact shot of her running downstairs is absent; only letter context is visible."
                            if index == 0
                            else "Use frame_verify on a recommended window for the final visual decision."
                        ),
                        "suggest_focus_windows": [span],
                        "observer_backend": "api" if index == 0 else "local_qwen",
                    }
                ),
                use_structured_evidence_state=True,
                question_context=question,
                scope_aware_evidence_memory=True,
            )
        return memory, question

    def test_candidate_projection_retains_scene_facts_and_diverse_targets(self):
        memory, question = self._candidate_projection_memory()
        compact = build_compact_investigation_state(
            memory,
            question=question,
            max_targets=8,
            max_gaps=8,
            scope_aware_evidence_memory=True,
            separate_routing_candidates=True,
            query_relevant_retention=True,
            persistent_candidate_pool=True,
            candidate_evidence_projection=True,
        )
        conflict = next(
            row
            for row in compact["overview_candidates"]
            if row.get("scene_id") == "pe_area_conflict"
        )
        self.assertTrue(
            any("Follow me" in cue.get("description", "") for cue in conflict["timestamp_cues"])
        )
        self.assertTrue(
            any(
                target.get("scene_id") == "pe_area_conflict"
                for target in compact["next_search_targets"]
            )
        )
        self.assertEqual(
            sum(
                target.get("target") == "resolve_missing_context"
                and (target.get("t_range") or [0.0])[0] >= 400.0
                for target in compact["next_search_targets"]
            ),
            1,
        )

        prompt = format_compact_investigation_state_for_prompt(
            memory,
            question=question,
            max_targets=8,
            max_gaps=8,
            scope_aware_evidence_memory=True,
            separate_routing_candidates=True,
            query_relevant_retention=True,
            persistent_candidate_pool=True,
            candidate_evidence_projection=True,
        )
        self.assertIn("scene_facts=", prompt)
        self.assertIn("Follow me", prompt)

    def test_candidate_projection_is_opt_in(self):
        memory, question = self._candidate_projection_memory()
        compact = build_compact_investigation_state(
            memory,
            question=question,
            max_targets=8,
            max_gaps=8,
            scope_aware_evidence_memory=True,
            separate_routing_candidates=True,
            query_relevant_retention=True,
            persistent_candidate_pool=True,
        )
        conflict = next(
            row
            for row in compact["overview_candidates"]
            if row.get("scene_id") == "pe_area_conflict"
        )
        self.assertFalse(
            any("Follow me" in cue.get("description", "") for cue in conflict["timestamp_cues"])
        )
        self.assertGreater(
            sum(
                target.get("target") == "resolve_missing_context"
                and (target.get("t_range") or [0.0])[0] >= 400.0
                for target in compact["next_search_targets"]
            ),
            1,
        )

    def _candidate_context_agent(self) -> VideoSeekAgent:
        agent = VideoSeekAgent.__new__(VideoSeekAgent)
        agent.config = {
            "coseek1_candidate_context_localize_enabled": True,
            "focus_qwen_max_window_s": 20.0,
            "localize_qwen_top_k": 3,
            "localize_qwen_max_top_k": 4,
            "localize_qwen_max_search_windows": 8,
            "localize_qwen_verify_context_margin_s": 4.0,
            "coseek1_persistent_candidate_pool_timestamp_radius_s": 8.0,
            "coseek1_intent_preserving_same_range_overlap": 0.85,
            "coseek1_planner_json_action": True,
        }
        agent.allowed_tool_names = {
            "overview",
            "localize_qwen",
            "frame_verify",
            "answer",
        }
        agent.duration = 490.0
        agent.question = "Where are tourists walking?"
        agent.messages = [{"role": "assistant", "content": "planner thought"}]
        agent._candidate_router_reason = ""
        agent.observation_memory = init_observation_memory()
        agent.observation_memory["candidate_pool"] = [
            {
                "candidate_id": "C_BOUNDARY",
                "status": "unvisited",
                "query_relevance_score": 2,
                "t_range": [411.5, 490.0],
                "suggest_focus_windows": [
                    [411.5, 425.5],
                    [438.2, 490.0],
                    [403.5, 419.5],
                    [430.2, 446.2],
                ],
            }
        ]
        return agent

    def test_short_verify_preserves_adjacent_candidate_boundary(self):
        agent = self._candidate_context_agent()
        action = Action(
            "frame_verify",
            {
                "query": "Neutrally identify the setting around the transition.",
                "start_time": 411.5,
                "end_time": 425.5,
                "mode": "detail_verify",
            },
            "planner_verify",
        )
        routed, thought, changed = (
            agent._VideoSeekAgent__route_short_verify_with_candidate_context(
                [action], step=1, thought="planner thought"
            )
        )
        self.assertTrue(changed)
        self.assertEqual(routed[0].function_name, "localize_qwen")
        self.assertEqual(
            routed[0].parameters["search_windows"],
            [[403.5, 419.5], [411.5, 425.5]],
        )
        self.assertEqual(routed[0].parameters["start_time"], 403.5)
        self.assertEqual(routed[0].parameters["end_time"], 425.5)
        self.assertIn("candidate_context_localize", routed[0].function_id)
        self.assertIn("403.5", thought)

    def test_existing_localize_is_enriched_without_adding_a_call(self):
        agent = self._candidate_context_agent()
        action = Action(
            "localize_qwen",
            {
                "search_windows": [[411.5, 425.5]],
                "localization_goal": "Find people walking and identify the setting.",
                "evidence_profile": "generic",
                "top_k": 3,
            },
            "planner_localize",
        )
        routed = agent._VideoSeekAgent__enrich_localize_with_candidate_context(
            [action]
        )
        self.assertEqual(len(routed), 1)
        self.assertEqual(routed[0].function_name, "localize_qwen")
        self.assertEqual(routed[0].function_id, "planner_localize")
        self.assertEqual(
            routed[0].parameters["search_windows"],
            [[403.5, 419.5], [411.5, 425.5]],
        )
        self.assertEqual(
            routed[0].parameters["source_planner_search_windows"],
            [[411.5, 425.5]],
        )
        self.assertIn("no extra tool call", agent._candidate_router_reason)

    def test_multiple_planner_localize_windows_are_not_rewritten(self):
        agent = self._candidate_context_agent()
        action = Action(
            "localize_qwen",
            {
                "search_windows": [[20.0, 30.0], [80.0, 90.0]],
                "localization_goal": "Compare two candidates.",
                "evidence_profile": "generic",
                "top_k": 3,
            },
            "planner_localize",
        )
        routed = agent._VideoSeekAgent__enrich_localize_with_candidate_context(
            [action]
        )
        self.assertIs(routed[0], action)

    def test_single_candidate_window_keeps_direct_verify(self):
        agent = self._candidate_context_agent()
        agent.observation_memory["candidate_pool"][0]["suggest_focus_windows"] = [
            [411.5, 425.5]
        ]
        action = Action(
            "frame_verify",
            {"query": "verify", "start_time": 411.5, "end_time": 425.5},
            "planner_verify",
        )
        routed, _, changed = (
            agent._VideoSeekAgent__route_short_verify_with_candidate_context(
                [action], step=1, thought="planner thought"
            )
        )
        self.assertFalse(changed)
        self.assertIs(routed[0], action)

    def test_unrelated_candidate_context_keeps_direct_verify(self):
        agent = self._candidate_context_agent()
        action = Action(
            "frame_verify",
            {"query": "verify", "start_time": 100.0, "end_time": 112.0},
            "planner_verify",
        )
        routed, _, changed = (
            agent._VideoSeekAgent__route_short_verify_with_candidate_context(
                [action], step=1, thought="planner thought"
            )
        )
        self.assertFalse(changed)
        self.assertIs(routed[0], action)

    def test_candidate_context_is_not_relocalized_after_usable_observation(self):
        agent = self._candidate_context_agent()
        agent.observation_memory["tool_observations"] = [
            {
                "tool": "localize_qwen",
                "parameters": {"start_time": 403.5, "end_time": 425.5},
                "observer_backend": "local_qwen",
                "parse_ok": True,
            }
        ]
        action = Action(
            "frame_verify",
            {"query": "verify", "start_time": 411.5, "end_time": 425.5},
            "planner_verify",
        )
        routed, _, changed = (
            agent._VideoSeekAgent__route_short_verify_with_candidate_context(
                [action], step=2, thought="planner thought"
            )
        )
        self.assertFalse(changed)
        self.assertIs(routed[0], action)

    def test_localize_observer_health_distinguishes_unavailable_from_no_hit(self):
        unavailable = {
            "observer_backend": "local_qwen_error",
            "parse_ok": False,
            "num_frames": 0,
            "timestamp_observations": [],
        }
        healthy_no_hit = {
            "observer_backend": "local_qwen",
            "parse_ok": True,
            "num_frames": 8,
            "timestamp_observations": [],
        }
        self.assertEqual(_observer_health([unavailable]), ("unavailable", 1))
        self.assertEqual(
            _observer_health([unavailable, healthy_no_hit]),
            ("partial", 1),
        )
        self.assertEqual(_observer_health([healthy_no_hit]), ("complete", 0))

    def test_density_skim_stops_after_backend_failure(self):
        failed = format_v10_observation(
            {
                "window_id": "W001",
                "scene_id": "S001",
                "t_range": [0.0, 8.0],
                "timestamp_observations": [],
                "possible_evidence": False,
                "observer_backend": "local_qwen_error",
                "observer_wall_s": 0.1,
                "num_frames": 0,
                "parse_ok": False,
            }
        )
        config = {
            "skim_qwen_temporal_density_enabled": True,
            "skim_qwen_target_fps": 1.0,
            "skim_qwen_density_batch_frames": 8,
            "skim_qwen_density_max_frames": 32,
        }
        with patch(
            "videoseek.tools.skim_qwen.execute_skim_qwen",
            return_value=failed,
        ) as child:
            output = _execute_density_batched_skim_qwen(
                config,
                {},
                start_time=0.0,
                end_time=32.0,
                scene_id="S001",
                window_id="W001",
                query="locate a visible event",
            )
        payload = extract_v10_payload(output)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(payload["observer_status"], "unavailable")
        self.assertEqual(payload["observer_batch_count"], 1)
        self.assertEqual(payload["observer_planned_batch_count"], 4)
        self.assertFalse(payload["parse_ok"])

    def test_modified_video_phrase_preserves_global_scope(self):
        self.assertEqual(
            _question_scope_hint(
                "Are there any irregularities in this surveillance video?"
            ),
            "global_video",
        )

    def test_verify_context_margin_is_bounded_by_source_search_window(self):
        self.assertEqual(
            _expand_verify_window(
                [10.0, 22.0],
                source_window=[8.5, 23.0],
                margin_s=2.0,
            ),
            [8.5, 23.0],
        )

    def test_search_context_expands_without_merging_independent_windows(self):
        self.assertEqual(
            _expand_search_windows(
                [[10.0, 20.0], [24.0, 30.0]],
                duration=40.0,
                margin_s=8.0,
            ),
            [[2.0, 22.0], [22.0, 38.0]],
        )

    def test_search_context_is_disabled_by_default_semantics(self):
        windows = [[0.0, 5.0], [95.0, 100.0]]
        self.assertEqual(
            _expand_search_windows(windows, duration=100.0, margin_s=0.0),
            windows,
        )

    def test_adaptive_search_context_tracks_overview_sampling_interval(self):
        config = {
            "localize_qwen_search_context_margin_s": 8.0,
            "localize_qwen_adaptive_search_context": True,
            "frame_sampling_factor": 4,
            "overview_base": 16,
        }
        self.assertAlmostEqual(
            _effective_search_context_margin(config, duration=189.0),
            3.0,
        )
        self.assertEqual(
            _effective_search_context_margin(config, duration=630.0),
            8.0,
        )

    def test_verify_context_boundaries_become_sampling_anchors(self):
        anchors = _verification_anchors(
            [{"timestamp_s": 100.0}, {"timestamp_s": 102.2}],
            localized_window=[93.2, 102.2],
            verify_window=[93.2, 106.2],
        )
        self.assertEqual(anchors[0], 93.2)
        self.assertEqual(anchors[-1], 106.2)
        self.assertIn(102.2, anchors)

    def test_many_verify_anchors_are_evenly_limited_without_losing_boundaries(self):
        rows = [
            {"timestamp_s": float(timestamp)}
            for timestamp in range(90, 111)
        ]
        anchors = _verification_anchors(
            rows,
            localized_window=[93.0, 107.0],
            verify_window=[90.0, 110.0],
            limit=12,
        )
        self.assertLessEqual(len(anchors), 12)
        self.assertEqual(anchors[0], 90.0)
        self.assertEqual(anchors[-1], 110.0)

    def test_packet_detail_anchors_span_localized_evidence(self):
        positioned_rows = [
            (index, float(index), object()) for index in range(10)
        ]
        positions, source = _packet_detail_anchor_positions(
            candidate={
                "timestamp_anchors": [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 9.0],
                "localized_timestamp_anchors": [2.0, 4.0, 6.0, 8.0],
            },
            positioned_rows=positioned_rows,
            limit=2,
        )
        self.assertEqual(positions, [2, 8])
        self.assertEqual(source, "localized_span_even")

    def test_packet_detail_anchors_fall_back_to_candidate_span(self):
        positioned_rows = [
            (index, float(index), object()) for index in range(8)
        ]
        positions, source = _packet_detail_anchor_positions(
            candidate={"timestamp_anchors": []},
            positioned_rows=positioned_rows,
            limit=2,
        )
        self.assertEqual(positions, [0, 7])
        self.assertEqual(source, "uniform_span_even")

    def test_temporal_anchor_spread_is_disabled_by_default(self):
        from config import general_config

        self.assertFalse(
            general_config.get(
                "grounded_verify_packet_temporal_anchor_spread_enabled"
            )
        )

    def test_normalized_candidate_score_does_not_reward_sampling_density(self):
        sparse = {
            "relevance": 0.7,
            "timestamp_observations": [
                {"timestamp_s": 1.0, "confidence": "high"},
                {"timestamp_s": 2.0, "confidence": "high"},
                {"timestamp_s": 3.0, "confidence": "low"},
                {"timestamp_s": 4.0, "confidence": "low"},
            ],
        }
        dense = {
            "relevance": 0.7,
            "timestamp_observations": [
                {"timestamp_s": float(index), "confidence": confidence}
                for index, confidence in enumerate(
                    ["high", "high", "high", "high", "low", "low", "low", "low"],
                    start=1,
                )
            ],
        }
        sparse_score = _candidate_score(sparse, [0.0, 20.0], 0, normalized=True)
        dense_score = _candidate_score(dense, [0.0, 20.0], 0, normalized=True)
        self.assertAlmostEqual(sparse_score, dense_score, places=4)

    def test_normalized_candidate_score_rewards_relevant_fraction_and_continuity(self):
        strong = {
            "relevance": 0.7,
            "timestamp_observations": [
                {"timestamp_s": 1.0, "query_relevance": "high"},
                {"timestamp_s": 2.0, "query_relevance": "high"},
                {"timestamp_s": 3.0, "query_relevance": "high"},
                {"timestamp_s": 4.0, "query_relevance": "low"},
            ],
        }
        weak = {
            "relevance": 0.7,
            "timestamp_observations": [
                {"timestamp_s": 1.0, "confidence": "high"},
                {"timestamp_s": 2.0, "confidence": "low"},
                {"timestamp_s": 3.0, "confidence": "low"},
                {"timestamp_s": 4.0, "confidence": "low"},
            ],
        }
        strong_parts = _candidate_score_components(strong, [0.0, 5.0], 0)
        weak_parts = _candidate_score_components(weak, [0.0, 5.0], 0)
        self.assertGreater(strong_parts["score"], weak_parts["score"])
        self.assertEqual(strong_parts["relevant_row_count"], 3)

    def test_normalized_candidate_score_is_disabled_by_default(self):
        from config import general_config

        self.assertFalse(
            general_config.get("localize_qwen_normalized_candidate_score_enabled")
        )

    def test_event_binding_and_consistency_repairs_are_disabled_by_default(self):
        from config import general_config

        self.assertFalse(
            general_config.get("grounded_candidate_event_binding_enabled")
        )
        self.assertFalse(
            general_config.get("packet_decision_consistency_normalization_enabled")
        )

    def test_temporally_remote_candidates_form_distinct_event_groups(self):
        groups = _candidate_event_groups(
            [
                {"candidate_id": "LQ001", "t_range": [263.3, 271.0]},
                {"candidate_id": "LQ002", "t_range": [270.3, 292.8]},
                {"candidate_id": "LQ003", "t_range": [396.6, 412.1]},
                {"candidate_id": "LQ004", "t_range": [411.6, 431.6]},
            ],
            max_gap_s=24.0,
        )
        self.assertEqual(groups["LQ001"], groups["LQ002"])
        self.assertEqual(groups["LQ003"], groups["LQ004"])
        self.assertNotEqual(groups["LQ001"], groups["LQ003"])

    def test_event_binding_applies_only_across_disconnected_groups(self):
        one_group = _candidate_event_groups(
            [
                {"candidate_id": "LQ001", "t_range": [403.5, 416.4]},
                {"candidate_id": "LQ002", "t_range": [411.5, 425.5]},
            ],
            max_gap_s=24.0,
        )
        two_groups = _candidate_event_groups(
            [
                {"candidate_id": "LQ001", "t_range": [263.3, 271.0]},
                {"candidate_id": "LQ002", "t_range": [411.6, 431.6]},
            ],
            max_gap_s=24.0,
        )
        self.assertFalse(
            _should_apply_candidate_event_binding(
                requested=True,
                event_groups=one_group,
            )
        )
        self.assertTrue(
            _should_apply_candidate_event_binding(
                requested=True,
                event_groups=two_groups,
            )
        )
        self.assertFalse(
            _should_apply_candidate_event_binding(
                requested=False,
                event_groups=two_groups,
            )
        )

    def test_recovery_keeps_single_event_binding_disabled(self):
        candidates = [
            {
                "candidate_id": "LQ001",
                "t_range": [10.0, 16.0],
                "summary": "One continuous local event.",
            }
        ]
        initial = format_v10_observation(
            {
                "tool": "frame_verify",
                "timestamp_observations": [],
                "candidate_assessments": [],
                "parse_ok": True,
            }
        )
        recovered = format_v10_observation(
            {
                "tool": "frame_verify",
                "timestamp_observations": [
                    {
                        "candidate_id": "LQ001",
                        "timestamp_s": 13.0,
                        "description": "The target event is visible.",
                    }
                ],
                "candidate_assessments": [
                    {
                        "candidate_id": "LQ001",
                        "target_match": "matched",
                        "observed_fact": "The target event is visible.",
                        "supports_options": ["A"],
                        "contradicts_options": ["B", "C", "D"],
                    }
                ],
                "target_match": "matched",
                "scope_coverage": "sufficient",
                "detail_sufficient": True,
                "supports_options": ["A"],
                "contradicts_options": ["B", "C", "D"],
                "parse_ok": True,
            }
        )
        config = _config()
        config.update(
            {
                "grounded_candidate_binding_enabled": True,
                "grounded_candidate_event_binding_enabled": True,
                "multiwindow_verify_recovery_enabled": True,
                "multiwindow_verify_recovery_max_calls": 1,
                "multiwindow_verify_recovery_batch_size": 1,
                "packet_detail_grounding_escalation_enabled": False,
            }
        )
        parameters = {
            **_direct_parameters(),
            "candidate_windows": candidates,
            "start_time": 10.0,
            "end_time": 16.0,
        }
        with patch(
            "videoseek.tools.frame_verify._execute_multiwindow_frame_verify_once",
            side_effect=[initial, recovered],
        ):
            payload = extract_v10_payload(
                execute_multiwindow_frame_verify(config, parameters)
            )
        self.assertEqual(payload["supports_options"], ["A"])
        self.assertNotIn("target_event_match", payload)

    def test_same_entity_different_event_support_stays_context_only(self):
        candidates = [
            {"candidate_id": "LQ001", "t_range": [263.3, 271.0]},
            {"candidate_id": "LQ002", "t_range": [411.6, 431.6]},
        ]
        assessments, complete = _normalize_candidate_assessments(
            [
                {
                    "candidate_id": "LQ001",
                    "target_match": "partial",
                    "event_match": "ambiguous",
                    "observed_fact": "The girl walks through a corridor.",
                    "supports_options": [],
                },
                {
                    "candidate_id": "LQ002",
                    "target_match": "matched",
                    "event_match": "different_event",
                    "observed_fact": "The same girl later receives a letter.",
                    "supports_options": ["D"],
                },
            ],
            candidates=candidates,
            event_binding_enabled=True,
        )
        self.assertTrue(complete)
        payload = {
            "candidate_assessments": assessments,
            "supports_options": ["D"],
            "contradicts_options": ["A", "B", "C"],
            "option_evidence": {
                "D": {"status": "support", "reason": "A letter is visible later."}
            },
            "detail_sufficient": True,
            "scope_coverage": "sufficient",
        }
        _apply_event_binding_option_scope(payload)
        self.assertEqual(payload["supports_options"], [])
        self.assertEqual(payload["contradicts_options"], [])
        self.assertEqual(payload["context_supports_options"], ["D"])
        self.assertEqual(payload["target_event_match"], "unresolved")
        self.assertFalse(payload["detail_sufficient"])
        self.assertEqual(payload["evidence_need"], "temporal_coverage")
        self.assertEqual(payload["option_evidence"]["D"]["status"], "unresolved")

    def test_only_direct_event_candidate_contributes_option_support(self):
        payload = {
            "candidate_assessments": [
                {
                    "candidate_id": "LQ001",
                    "event_match": "direct",
                    "supports_options": ["C"],
                    "contradicts_options": ["A", "B", "D"],
                },
                {
                    "candidate_id": "LQ002",
                    "event_match": "different_event",
                    "supports_options": ["D"],
                    "contradicts_options": ["C"],
                },
            ],
            "supports_options": ["C", "D"],
            "contradicts_options": ["A", "B", "C", "D"],
            "option_evidence": {
                "C": {"status": "support", "reason": "Direct event."},
                "D": {"status": "support", "reason": "Later event."},
            },
        }
        _apply_event_binding_option_scope(payload)
        self.assertEqual(payload["supports_options"], ["C"])
        self.assertEqual(payload["contradicts_options"], ["A", "B", "D"])
        self.assertEqual(payload["context_supports_options"], ["D"])
        self.assertEqual(payload["option_evidence"]["C"]["status"], "support")
        self.assertEqual(payload["option_evidence"]["D"]["status"], "unresolved")

    def test_complete_option_comparison_repairs_false_insufficient(self):
        payload = {
            "detail_sufficient": False,
            "target_match": "matched",
            "target_event_match": "direct",
            "scope_coverage": "sufficient",
            "supports_options": ["D"],
            "contradicts_options": ["A", "B", "C"],
            "evidence_need": "temporal_coverage",
            "missing_detail": "None; evidence is clear.",
        }
        self.assertTrue(_normalize_decision_completeness(payload))
        self.assertTrue(payload["detail_sufficient"])
        self.assertEqual(payload["evidence_need"], "sufficient")
        self.assertTrue(payload["decision_consistency_repaired"])

    def test_consistency_repair_keeps_real_missing_detail_open(self):
        payload = {
            "detail_sufficient": False,
            "target_match": "matched",
            "target_event_match": "direct",
            "scope_coverage": "sufficient",
            "supports_options": ["D"],
            "contradicts_options": ["A", "B", "C"],
            "evidence_need": "temporal_coverage",
            "missing_detail": "The exact action frame is not shown.",
        }
        self.assertFalse(_normalize_decision_completeness(payload))
        self.assertFalse(payload["detail_sufficient"])

    def test_memory_prompt_exposes_candidate_event_match(self):
        memory = init_observation_memory()
        output = format_v10_observation(
            {
                "tool": "frame_verify",
                "scene_id": "multi_scene",
                "t_range": [10.0, 80.0],
                "observer_backend": "api",
                "timestamp_observations": [
                    {
                        "candidate_id": "LQ001",
                        "timestamp_s": 72.0,
                        "description": "The same person appears in another event.",
                        "target_match": "matched",
                        "event_match": "different_event",
                    }
                ],
                "candidate_assessments": [
                    {
                        "candidate_id": "LQ001",
                        "target_match": "matched",
                        "event_match": "different_event",
                        "observed_fact": "The same person appears in another event.",
                        "target_binding_reason": "Identity matches but action does not.",
                        "supports_options": [],
                        "contradicts_options": [],
                    }
                ],
                "target_match": "matched",
                "target_event_match": "unresolved",
                "scope_coverage": "partial",
                "detail_sufficient": False,
                "missing_detail": "The queried event is not visible.",
            }
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={
                "candidate_windows": [
                    {"candidate_id": "LQ001", "t_range": [70.0, 75.0]}
                ]
            },
            output=output,
            question_context="Why did the person run?\n(A) help\n(B) chase\n(C) leave\n(D) fetch",
        )
        text = build_compact_investigation_state(
            memory,
            question="Why did the person run?",
        )
        rendered = str(text)
        self.assertEqual(
            memory["candidate_binding_memory"][-1]["event_match"],
            "different_event",
        )
        self.assertIn("different_event", rendered)

    def test_ranked_candidates_preserve_one_per_search_window(self):
        first = {
            "relevance": 0.7,
            "timestamp_observations": [
                {"timestamp_s": 2.0, "confidence": "high"},
                {"timestamp_s": 3.0, "confidence": "high"},
                {"timestamp_s": 15.0, "confidence": "high"},
                {"timestamp_s": 16.0, "confidence": "high"},
            ],
        }
        second = {
            "relevance": 0.4,
            "timestamp_observations": [
                {"timestamp_s": 32.0, "confidence": "medium"},
                {"timestamp_s": 33.0, "confidence": "medium"},
            ],
        }
        candidates = _rank_coarse_candidates(
            [([0.0, 20.0], first), ([30.0, 40.0], second)],
            top_k=2,
            candidate_window_s=6.0,
            row_candidates_enabled=True,
            normalized_score_enabled=True,
            preserve_source_coverage=True,
        )
        self.assertEqual({item["source_index"] for item in candidates}, {0, 1})

    def test_false_sufficient_with_missing_frames_normalizes_to_temporal(self):
        self.assertEqual(
            _normalize_evidence_need(
                {
                    "evidence_need": "sufficient",
                    "detail_sufficient": False,
                    "target_match": "matched",
                    "question_scope": "local_window",
                    "scope_coverage": "sufficient",
                    "missing_detail": "The exact contact frame is missing.",
                }
            ),
            "temporal_coverage",
        )

    def test_no_supported_option_normalizes_to_semantic_binding(self):
        self.assertEqual(
            _normalize_evidence_need(
                {
                    "evidence_need": "temporal_coverage",
                    "detail_sufficient": False,
                    "target_match": "matched",
                    "question_scope": "global_video",
                    "scope_coverage": "partial",
                    "supports_options": [],
                    "option_evidence": {
                        letter: {"status": "unresolved"}
                        for letter in ("A", "B", "C", "D")
                    },
                }
            ),
            "semantic_binding",
        )

    def test_partial_global_contradictions_remain_local(self):
        payload = {
            "question_scope": "global_video",
            "scope_coverage": "partial",
            "detail_sufficient": False,
            "supports_options": ["B"],
            "contradicts_options": ["A", "C"],
            "option_evidence": {
                "A": {"status": "contradicted", "reason": "No event here."},
                "B": {"status": "weak_support", "reason": "This window is calm."},
            },
        }
        _demote_partial_global_option_claims(payload)
        self.assertEqual(payload["supports_options"], [])
        self.assertEqual(payload["contradicts_options"], [])
        self.assertEqual(payload["local_supports_options"], ["B"])
        self.assertEqual(payload["local_contradicts_options"], ["A", "C"])
        self.assertEqual(payload["option_evidence"]["A"]["status"], "unresolved")

    def test_global_temporal_need_can_also_request_local_spatial_detail(self):
        payload = {
            "question_scope": "global_video",
            "scope_coverage": "partial",
            "detail_sufficient": False,
            "target_match": "partial",
            "detail_target": {"timestamp_s": 119.6},
            "detail_query": "Inspect the small object in the person's hand.",
        }
        self.assertEqual(
            _normalize_evidence_needs(payload),
            ["temporal_coverage", "spatial_detail"],
        )

    def test_direct_packet_spatial_need_uses_one_detail_escalation(self):
        config = _config()
        config.update(
            {
                "grounded_direct_verify_packet_enabled": True,
                "packet_adaptive_evidence_need_enabled": True,
                "packet_detail_grounding_direct_escalation_enabled": True,
            }
        )
        detail_payload = extract_v10_payload(_detail_response())
        detail_payload["parse_ok"] = True
        for row in detail_payload.get("candidate_assessments") or []:
            row["candidate_id"] = "DIRECT001"
        for row in detail_payload.get("timestamp_observations") or []:
            row["candidate_id"] = "DIRECT001"
        with patch(
            "videoseek.tools.frame_verify.execute_focus",
            return_value=_direct_initial(evidence_need="spatial_detail"),
        ), patch(
            "videoseek.tools.frame_verify._execute_multiwindow_frame_verify_once",
            return_value=format_v10_observation(detail_payload),
        ) as detail_verify:
            output = execute_frame_verify(config, _direct_parameters())

        payload = extract_v10_payload(output)
        self.assertEqual(detail_verify.call_count, 1)
        self.assertTrue(payload["packet_detail_escalation_used"])
        self.assertTrue(payload["direct_packet_detail_escalation"])
        self.assertEqual(payload["scene_id"], "S007")
        self.assertEqual(payload["supports_options"], ["B"])

    def test_direct_packet_temporal_need_does_not_crop(self):
        config = _config()
        config.update(
            {
                "grounded_direct_verify_packet_enabled": True,
                "packet_adaptive_evidence_need_enabled": True,
                "packet_detail_grounding_direct_escalation_enabled": True,
            }
        )
        with patch(
            "videoseek.tools.frame_verify.execute_focus",
            return_value=_direct_initial(
                evidence_need="temporal_coverage", with_detail_target=False
            ),
        ), patch(
            "videoseek.tools.frame_verify._execute_multiwindow_frame_verify_once"
        ) as detail_verify:
            output = execute_frame_verify(config, _direct_parameters())

        payload = extract_v10_payload(output)
        detail_verify.assert_not_called()
        self.assertEqual(payload["evidence_need"], "temporal_coverage")
        self.assertNotIn("direct_packet_detail_escalation", payload)

    def test_semantic_gap_resolution_keeps_unresolved_overlap_open(self):
        memory = init_observation_memory()
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 10.0, "end_time": 20.0},
            output=_memory_output(
                span=[10.0, 20.0],
                detail_sufficient=False,
                missing_detail="Target identity remains unclear.",
                target_match="partial",
                scope_coverage="sufficient",
                evidence_need="semantic_binding",
            ),
            persistent_open_gaps=True,
            semantic_gap_resolution=True,
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 12.0, "end_time": 18.0},
            output=_memory_output(
                span=[12.0, 18.0],
                detail_sufficient=True,
                missing_detail="More surrounding time is needed.",
                target_match="partial",
                scope_coverage="partial",
                evidence_need="temporal_coverage",
            ),
            persistent_open_gaps=True,
            semantic_gap_resolution=True,
        )
        self.assertEqual(memory["open_gaps"][0]["status"], "open")

        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 12.0, "end_time": 18.0},
            output=_memory_output(
                span=[12.0, 18.0],
                detail_sufficient=True,
                missing_detail="",
                target_match="matched",
                scope_coverage="sufficient",
                evidence_need="sufficient",
            ),
            persistent_open_gaps=True,
            semantic_gap_resolution=True,
        )
        self.assertEqual(memory["open_gaps"][0]["status"], "resolved")

    def test_compact_state_exposes_verified_observer_conflict(self):
        memory = init_observation_memory()
        memory["runtime_config"] = {
            "coseek1_observer_conflict_memory_enabled": True
        }
        question = "What happened?\n(A) shooting\n(B) normal\n(C) stealing\n(D) fighting"
        for supports, contradicts, span in (
            (["A"], [], [50.0, 60.0]),
            ([], ["A"], [52.0, 58.0]),
        ):
            merge_tool_observation(
                memory,
                tool_name="frame_verify",
                parameters={"start_time": span[0], "end_time": span[1]},
                output=_memory_output(
                    span=span,
                    detail_sufficient=True,
                    missing_detail="",
                    target_match="matched",
                    scope_coverage="sufficient",
                    evidence_need="sufficient",
                    supports=supports,
                    contradicts=contradicts,
                ),
                use_structured_evidence_state=True,
                question_context=question,
                scope_aware_evidence_memory=True,
            )

        compact = build_compact_investigation_state(
            memory,
            question=question,
            scope_aware_evidence_memory=True,
        )
        self.assertEqual(len(compact["observer_conflicts"]), 1)
        self.assertEqual(compact["observer_conflicts"][0]["option"], "A")
        self.assertEqual(compact["answer_status"]["status"], "no_verified_answer_yet")

    def test_complete_timestamped_verify_preserves_aggregate_option_decision(self):
        memory = init_observation_memory()
        question = (
            "How did she feel?\n"
            "(A) Sad\n(B) Angry\n(C) Happy\n(D) Afraid"
        )
        output = format_v10_observation(
            {
                "tool": "frame_verify",
                "scene_id": "S004",
                "window_id": "multi_candidate_verify",
                "t_range": [249.0, 297.0],
                "timestamp_observations": [
                    {
                        "candidate_id": "LQ001",
                        "timestamp_s": 263.0,
                        "description": "She smiles broadly after the event.",
                    },
                    {
                        "candidate_id": "LQ002",
                        "timestamp_s": 289.0,
                        "description": "She remains visibly pleased.",
                    },
                ],
                "overall_summary": "Both views show a positive reaction.",
                "observed_fact": "The target woman smiles and appears happy.",
                "target_match": "matched",
                "target_event_match": "direct",
                "question_scope": "local_window",
                "scope_coverage": "sufficient",
                "detail_sufficient": True,
                "evidence_need": "sufficient",
                "evidence_needs": ["sufficient"],
                "missing_detail": (
                    "None; immediate reason is established by dialogue and "
                    "destination context."
                ),
                "candidate_assessments": [
                    {
                        "candidate_id": "LQ001",
                        "target_match": "matched",
                        "observed_fact": "She smiles after the target event.",
                        "supports_options": ["C"],
                        "contradicts_options": ["A", "B", "D"],
                    },
                    {
                        "candidate_id": "LQ002",
                        "target_match": "matched",
                        "observed_fact": "A second view confirms her reaction.",
                        "supports_options": ["C"],
                        "contradicts_options": ["A", "B", "D"],
                    },
                ],
                "candidate_binding_complete": True,
                "supports_options": ["C"],
                "contradicts_options": ["A", "B", "D"],
                "option_set_conflict": False,
                "observer_backend": "api",
                "parse_ok": True,
            }
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={
                "candidate_windows": [
                    {"candidate_id": "LQ001", "t_range": [259.0, 267.0]},
                    {"candidate_id": "LQ002", "t_range": [285.0, 293.0]},
                ]
            },
            output=output,
            use_structured_evidence_state=True,
            question_context=question,
            scope_aware_evidence_memory=True,
            preserve_aggregate_verifier_decision=True,
        )

        evidence = memory["structured_evidence"]
        aggregate = [
            item
            for item in evidence["evidence_items"]
            if item["refs"]["source_kind"] == "aggregate_verifier_decision"
        ]
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["evidence_level"], "verified")
        self.assertEqual(aggregate[0]["evidence_scope"], "query_sufficient")
        support = {
            row["option"]: row for row in evidence["option_support"]
        }
        self.assertIn(aggregate[0]["evidence_id"], support["C"]["supports"])

        compact = build_compact_investigation_state(
            memory,
            question=question,
            scope_aware_evidence_memory=True,
        )
        self.assertEqual(
            compact["answer_status"]["status"],
            "verified_candidate_available",
        )
        self.assertEqual(compact["answer_status"]["option"], "C")

    def test_aggregate_verifier_decision_is_opt_in(self):
        memory = init_observation_memory()
        output = _memory_output(
            span=[10.0, 18.0],
            detail_sufficient=True,
            missing_detail="",
            target_match="matched",
            scope_coverage="sufficient",
            evidence_need="sufficient",
            supports=["B"],
            contradicts=["A", "C", "D"],
        )
        payload = extract_v10_payload(output)
        payload["timestamp_observations"] = [
            {"timestamp_s": 14.0, "description": "The verified target is visible."}
        ]
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 10.0, "end_time": 18.0},
            output=format_v10_observation(payload),
            use_structured_evidence_state=True,
            question_context="What is shown?\n(A) cup\n(B) book\n(C) phone\n(D) key",
        )
        source_kinds = {
            item["refs"]["source_kind"]
            for item in memory["structured_evidence"]["evidence_items"]
        }
        self.assertNotIn("aggregate_verifier_decision", source_kinds)

    def test_incomplete_or_conflicted_verify_is_not_aggregated(self):
        question = "What happened?\n(A) fall\n(B) jump\n(C) run\n(D) stop"
        for overrides in (
            {"scope_coverage": "partial", "evidence_need": "temporal_coverage"},
            {"option_set_conflict": True},
        ):
            memory = init_observation_memory()
            payload = extract_v10_payload(
                _memory_output(
                    span=[20.0, 28.0],
                    detail_sufficient=True,
                    missing_detail="",
                    target_match="matched",
                    scope_coverage="sufficient",
                    evidence_need="sufficient",
                    supports=["A"],
                    contradicts=["B", "C", "D"],
                )
            )
            payload.update(overrides)
            payload["timestamp_observations"] = [
                {"timestamp_s": 24.0, "description": "A person changes posture."}
            ]
            merge_tool_observation(
                memory,
                tool_name="frame_verify",
                parameters={"start_time": 20.0, "end_time": 28.0},
                output=format_v10_observation(payload),
                use_structured_evidence_state=True,
                question_context=question,
                preserve_aggregate_verifier_decision=True,
            )
            source_kinds = {
                item["refs"]["source_kind"]
                for item in memory["structured_evidence"]["evidence_items"]
            }
            self.assertNotIn("aggregate_verifier_decision", source_kinds)

    def test_decision_mode_does_not_aggregate_an_uncovered_competitor(self):
        memory = init_observation_memory()
        question = (
            "Why did she run?\n(A) To leave\n(B) To chase someone\n"
            "(C) To help another girl\n(D) To get an item"
        )
        payload = extract_v10_payload(
            _memory_output(
                span=[260.0, 420.0],
                detail_sufficient=True,
                missing_detail="",
                target_match="matched",
                scope_coverage="sufficient",
                evidence_need="sufficient",
                supports=["C"],
                contradicts=["A", "B"],
            )
        )
        payload.update(
            {
                "decision_sufficient": True,
                "decision_sufficiency_requested": True,
                "candidate_binding_complete": True,
                "candidate_coverage_complete": True,
                "candidate_assessments": [
                    {"candidate_id": "C001", "target_match": "matched"},
                    {"candidate_id": "C002", "target_match": "matched"},
                ],
                "option_evidence": {
                    "A": {"status": "contradicted", "reason": "not shown"},
                    "B": {"status": "contradicted", "reason": "not shown"},
                    "C": {"status": "supported", "reason": "direct event"},
                    "D": {"status": "unresolved", "reason": "another event"},
                },
                "timestamp_observations": [
                    {"timestamp_s": 270.0, "description": "She runs urgently."},
                    {"timestamp_s": 410.0, "description": "A separate item scene."},
                ],
            }
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={
                "candidate_windows": [
                    {"candidate_id": "C001", "t_range": [260.0, 280.0]},
                    {"candidate_id": "C002", "t_range": [400.0, 420.0]},
                ]
            },
            output=format_v10_observation(payload),
            use_structured_evidence_state=True,
            question_context=question,
            preserve_aggregate_verifier_decision=True,
        )
        aggregate = [
            item
            for item in memory["structured_evidence"]["evidence_items"]
            if item["refs"]["source_kind"] == "aggregate_verifier_decision"
        ]
        self.assertEqual(aggregate, [])

    def test_decision_mode_missing_decision_field_cannot_use_legacy_aggregate(self):
        memory = init_observation_memory()
        question = "What happened?\n(A) fall\n(B) jump\n(C) run\n(D) stop"
        payload = extract_v10_payload(
            _memory_output(
                span=[10.0, 30.0],
                detail_sufficient=True,
                missing_detail="",
                target_match="matched",
                scope_coverage="sufficient",
                evidence_need="sufficient",
                supports=["C"],
                contradicts=["A", "B", "D"],
            )
        )
        payload.update(
            {
                "decision_sufficiency_requested": True,
                "candidate_binding_complete": True,
                "candidate_assessments": [
                    {"candidate_id": "C001", "target_match": "matched"},
                    {"candidate_id": "C002", "target_match": "matched"},
                ],
                "timestamp_observations": [
                    {"timestamp_s": 14.0, "description": "First view."},
                    {"timestamp_s": 26.0, "description": "Second view."},
                ],
            }
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={
                "candidate_windows": [
                    {"candidate_id": "C001", "t_range": [10.0, 18.0]},
                    {"candidate_id": "C002", "t_range": [22.0, 30.0]},
                ]
            },
            output=format_v10_observation(payload),
            use_structured_evidence_state=True,
            question_context=question,
            preserve_aggregate_verifier_decision=True,
        )
        source_kinds = {
            item["refs"]["source_kind"]
            for item in memory["structured_evidence"]["evidence_items"]
        }
        self.assertNotIn("aggregate_verifier_decision", source_kinds)

    def test_single_narrow_verify_stays_local_even_when_self_reported_complete(self):
        memory = init_observation_memory()
        payload = extract_v10_payload(
            _memory_output(
                span=[410.5, 415.0],
                detail_sufficient=True,
                missing_detail="",
                target_match="matched",
                scope_coverage="sufficient",
                evidence_need="sufficient",
                supports=["A"],
                contradicts=["B", "C", "D"],
            )
        )
        payload["timestamp_observations"] = [
            {
                "timestamp_s": 412.0,
                "description": "A narrow view shows a paved street.",
            }
        ]
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 410.5, "end_time": 415.0},
            output=format_v10_observation(payload),
            use_structured_evidence_state=True,
            question_context=(
                "Where are they?\n(A) city\n(B) forest\n"
                "(C) fishing village\n(D) beach"
            ),
            scope_aware_evidence_memory=True,
            preserve_aggregate_verifier_decision=True,
        )
        source_kinds = {
            item["refs"]["source_kind"]
            for item in memory["structured_evidence"]["evidence_items"]
        }
        self.assertNotIn("aggregate_verifier_decision", source_kinds)


if __name__ == "__main__":
    unittest.main()
