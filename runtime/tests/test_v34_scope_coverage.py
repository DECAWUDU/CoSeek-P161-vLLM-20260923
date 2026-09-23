import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from videoseek.core.memory import (
    build_compact_investigation_state,
    format_compact_investigation_state_for_prompt,
    format_memory_for_prompt,
    init_observation_memory,
    merge_tool_observation,
)
from videoseek.agent import VideoSeekAgent
from videoseek.tools import focus_qwen as focus_qwen_module
from videoseek.tools.answer import execute_answer
from videoseek.tools.overview import (
    _annotate_frame_timestamp,
    _normalize_timestamp_observations,
)
from videoseek.tools.v10_format import format_v10_observation
from videoseek.utils import extract_json_object


def observation(payload):
    return format_v10_observation(payload, fallback_text=payload.get("overall_summary", ""))


class V34ScopeCoverageTests(unittest.TestCase):
    def test_query_relevant_retention_keeps_early_compound_match(self):
        memory = init_observation_memory()
        timestamp_rows = [
            {
                "timestamp_s": 53.0,
                "scene_id": "yard_walk",
                "description": "Three uniformed men walk beside mesh stairs.",
            }
        ]
        timestamp_rows.extend(
            {
                "timestamp_s": 100.0 + index,
                "scene_id": f"late_{index}",
                "description": "Unrelated room conversation and furniture.",
            }
            for index in range(24)
        )
        merge_tool_observation(
            memory,
            tool_name="overview",
            parameters={},
            output=observation(
                {
                    "observer_backend": "api",
                    "timestamp_observations": timestamp_rows,
                    "scene_summaries": [
                        {
                            "scene_id": "yard_walk",
                            "t_range": [53.0, 60.6],
                            "summary": "Men pass mesh stairs in a snowy yard.",
                            "possible_evidence": False,
                            "suggest_focus_windows": [[53.0, 60.6]],
                            "missing_detail": "Only background stairs are visible.",
                        }
                    ],
                }
            ),
        )
        question = (
            "How many people are interacting at the staircase?\n"
            "(A) Four\n(B) Two\n(C) Five\n(D) Three"
        )
        compact = build_compact_investigation_state(
            memory,
            question=question,
            max_facts=4,
            separate_routing_candidates=True,
            query_relevant_retention=True,
        )
        self.assertIn(53.0, [item["time"] for item in compact["visual_facts"]])
        self.assertTrue(
            any(
                item.get("scene_id") == "yard_walk"
                for item in compact["next_search_targets"]
            )
        )
        prompt = format_memory_for_prompt(
            memory,
            question=question,
            max_observations=4,
            include_compact_planner_state=True,
            separate_routing_candidates=True,
            query_relevant_retention=True,
        )
        self.assertIn("53.0s", prompt)
        self.assertIn("mesh stairs", prompt)

    def test_overview_timestamp_is_embedded_in_each_cell(self):
        frame = np.full((256, 480, 3), 127, dtype=np.uint8)
        annotated = _annotate_frame_timestamp(frame, 53.0)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertFalse(np.array_equal(annotated[:40, :140], frame[:40, :140]))

    @patch("videoseek.tools.answer.call_llm_api")
    def test_answer_digest_preserves_scope_and_routing_semantics(self, call_api):
        call_api.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="B"))]
        )
        memory = init_observation_memory()
        merge_tool_observation(
            memory,
            tool_name="overview",
            parameters={},
            output=observation(
                {
                    "observer_backend": "api",
                    "scene_summaries": [
                        {
                            "scene_id": "S001",
                            "t_range": [0.0, 20.0],
                            "summary": "possible dog enclosure",
                            "possible_evidence": True,
                        },
                        {
                            "scene_id": "S002",
                            "t_range": [80.0, 100.0],
                            "summary": "possible sheep enclosure",
                            "possible_evidence": True,
                        },
                    ],
                }
            ),
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 0.0, "end_time": 20.0},
            output=observation(
                {
                    "observer_backend": "api",
                    "scene_id": "S001",
                    "t_range": [0.0, 20.0],
                    "overall_summary": "A dog is visible in this local enclosure.",
                    "detail_sufficient": True,
                    "supports_options": ["B"],
                    "contradicts_options": ["A", "C", "D"],
                }
            ),
            use_structured_evidence_state=True,
            question_context="Which animal?\n(A) Sheep\n(B) Dog\n(C) Cat\n(D) Pig",
            scope_aware_evidence_memory=True,
        )

        config = {
            "use_evidence_reducer": True,
            "structured_evidence_include_planner_state": True,
            "use_structured_evidence_state": True,
            "structured_evidence_max_items": 16,
            "structured_evidence_answer_max_items": 16,
            "structured_evidence_include_timeline_view": False,
            "structured_evidence_include_object_state_view": False,
            "coseek1_scope_aware_evidence_memory": True,
            "coseek1_separate_routing_candidates": True,
            "model_name": "test",
            "api_base": "test",
            "api_key": "test",
            "api_version": None,
            "max_tokens": 100,
            "reasoning_effort": "low",
            "seed": 1,
            "temperature": 0.0,
        }
        execute_answer(
            config,
            {
                "question": "Which animal?\n(A) Sheep\n(B) Dog\n(C) Cat\n(D) Pig",
                "messages": [],
                "memory": memory,
            },
        )

        compact = memory["compact_investigation_state"]
        self.assertEqual(len(compact["overview_candidates"]), 2)
        self.assertNotIn(
            "overview", {item["tool"] for item in compact["inspected_windows"]}
        )
        self.assertEqual(
            compact["answer_status"]["status"], "local_verified_candidate_available"
        )
        self.assertTrue(compact["answer_status"]["uninspected_alternatives"])

    def test_overview_timestamp_rows_are_aligned_and_omissions_recorded(self):
        parsed = {
            "timestamp_observations": [
                {"timestamp_s": 0.02, "description": "opening room"},
                {"timestamp_s": 20.1, "description": "pasture"},
            ]
        }
        _normalize_timestamp_observations(
            parsed,
            allowed_timestamps=[0.0, 10.0, 20.0],
            fill_omitted=True,
        )
        self.assertEqual(
            [item["timestamp_s"] for item in parsed["timestamp_observations"]],
            [0.0, 10.0, 20.0],
        )
        self.assertEqual(parsed["overview_observed_timestamp_count"], 2)
        self.assertEqual(parsed["overview_omitted_timestamps"], [10.0])
        self.assertIn(
            "overview_omitted_frame",
            parsed["timestamp_observations"][1]["event_tags"],
        )

    def test_disjoint_short_windows_are_not_routed_as_one_long_skim(self):
        agent = object.__new__(VideoSeekAgent)
        agent.config = {
            "focus_qwen_max_window_s": 20.0,
            "focus_qwen_multi_window_max_windows": 4,
        }
        agent.duration = 100.0
        params = agent._VideoSeekAgent__planner_json_parameters(
            {
                "parameters": {
                    "query": "compare three staircase candidates",
                    "windows": [[10, 18], [52, 60], [80, 92]],
                    "mode": "normal",
                }
            }
        )
        self.assertEqual(params["start_time"], 10.0)
        self.assertEqual(params["end_time"], 92.0)
        self.assertFalse(
            agent._VideoSeekAgent__coseek1_focus_qwen_window_too_long(params)
        )

    def test_local_window_contradictions_are_not_global(self):
        memory = init_observation_memory()
        payload = {
            "observer_backend": "api",
            "scene_id": "S001",
            "t_range": [10.0, 20.0],
            "overall_summary": "This enclosure contains a dog.",
            "detail_sufficient": True,
            "supports_options": ["B"],
            "contradicts_options": ["A", "C", "D"],
            "scene_summaries": [
                {
                    "scene_id": "S001",
                    "t_range": [10.0, 20.0],
                    "summary": "This enclosure contains a dog.",
                }
            ],
        }
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 10.0, "end_time": 20.0},
            output=observation(payload),
            use_structured_evidence_state=True,
            question_context=(
                "Which animal is shown?\n(A) Sheep\n(B) Dog\n(C) Horse\n(D) Cow"
            ),
            scope_aware_evidence_memory=True,
        )

        rows = {
            row["option"]: row
            for row in memory["structured_evidence"]["option_support"]
        }
        self.assertTrue(rows["B"]["supports"])
        for option in ("A", "C", "D"):
            self.assertEqual(rows[option]["contradicts"], [])
            self.assertTrue(rows[option]["local_contradicts"])

    def test_overview_is_routing_map_not_inspected_evidence(self):
        memory = init_observation_memory()
        overview = {
            "observer_backend": "api",
            "global_summary": "Two candidate regions.",
            "scene_summaries": [
                {
                    "scene_id": "S001",
                    "t_range": [0.0, 100.0],
                    "summary": "possible animal enclosure",
                    "possible_evidence": True,
                }
            ],
        }
        merge_tool_observation(
            memory,
            tool_name="overview",
            parameters={},
            output=observation(overview),
        )
        skim = {
            "observer_backend": "local_qwen",
            "scene_id": "S001",
            "t_range": [0.0, 100.0],
            "overall_summary": "No target animal is visible.",
            "possible_evidence": False,
            "detail_sufficient": False,
            "scene_summaries": [
                {
                    "scene_id": "S001",
                    "t_range": [0.0, 100.0],
                    "summary": "No target animal is visible.",
                    "possible_evidence": False,
                }
            ],
        }
        merge_tool_observation(
            memory,
            tool_name="skim_qwen",
            parameters={"start_time": 0.0, "end_time": 100.0},
            output=observation(skim),
        )

        compact = build_compact_investigation_state(
            memory,
            separate_routing_candidates=True,
        )
        self.assertEqual(len(compact["overview_candidates"]), 1)
        self.assertTrue(compact["inspected_windows"])
        self.assertNotIn(
            "overview", {item["tool"] for item in compact["inspected_windows"]}
        )
        self.assertEqual(compact["scene_coverage"][0]["coverage_ratio"], 1.0)
        self.assertGreater(compact["scene_coverage"][0]["negative_observations"], 0)

    def test_local_verified_candidate_keeps_uninspected_overview_target(self):
        memory = init_observation_memory()
        overview = {
            "observer_backend": "api",
            "scene_summaries": [
                {
                    "scene_id": "S001",
                    "t_range": [0.0, 20.0],
                    "summary": "dog enclosure",
                    "possible_evidence": True,
                },
                {
                    "scene_id": "S002",
                    "t_range": [80.0, 100.0],
                    "summary": "another animal enclosure",
                    "possible_evidence": True,
                },
            ],
        }
        merge_tool_observation(memory, tool_name="overview", parameters={}, output=observation(overview))
        verified = {
            "observer_backend": "api",
            "scene_id": "S001",
            "t_range": [0.0, 20.0],
            "overall_summary": "A dog is visible in this enclosure.",
            "detail_sufficient": True,
            "supports_options": ["B"],
            "scene_summaries": [
                {
                    "scene_id": "S001",
                    "t_range": [0.0, 20.0],
                    "summary": "A dog is visible in this enclosure.",
                    "possible_evidence": True,
                }
            ],
        }
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 0.0, "end_time": 20.0},
            output=observation(verified),
            use_structured_evidence_state=True,
            question_context=(
                "Which animal is shown?\n(A) Sheep\n(B) Dog\n(C) Horse\n(D) Cow"
            ),
            scope_aware_evidence_memory=True,
        )

        compact = build_compact_investigation_state(
            memory,
            question="Which animal is shown?",
            scope_aware_evidence_memory=True,
            separate_routing_candidates=True,
        )
        self.assertEqual(
            compact["answer_status"]["status"],
            "local_verified_candidate_available",
        )
        self.assertEqual(
            compact["answer_status"]["uncovered_options"],
            ["A", "C", "D"],
        )
        self.assertTrue(
            any(
                target.get("scene_id") == "S002"
                for target in compact["next_search_targets"]
            )
        )
        prompt = format_compact_investigation_state_for_prompt(
            memory,
            question="Which animal is shown?",
            scope_aware_evidence_memory=True,
            separate_routing_candidates=True,
        )
        self.assertIn("provisional_local_candidate=B", prompt)
        self.assertNotIn("verified_candidate=B", prompt)
        self.assertIn("Uninspected competing routing candidates", prompt)

    def test_hyphenated_competitor_keeps_unlikely_scene_searchable(self):
        memory = init_observation_memory()
        overview = {
            "observer_backend": "api",
            "scene_summaries": [
                {
                    "scene_id": "S005",
                    "t_range": [481.0, 575.6],
                    "summary": "Rural pastures, barns, and livestock.",
                    "possible_evidence": False,
                    "suggest_focus_windows": [[481.0, 540.3]],
                    "missing_detail": "Confirm non-dog livestock presence.",
                }
            ],
        }
        merge_tool_observation(
            memory,
            tool_name="overview",
            parameters={},
            output=observation(overview),
        )
        compact = build_compact_investigation_state(
            memory,
            question=(
                "What kind of animals are being kept?\n"
                "(A) Sheep\n(B) Dog\n(C) Cat\n(D) Pig"
            ),
            separate_routing_candidates=True,
        )
        target = next(
            item
            for item in compact["next_search_targets"]
            if item.get("scene_id") == "S005"
        )
        self.assertEqual(target["possible_evidence"], False)
        self.assertEqual(target["t_range"], [481.0, 540.3])

    def test_persistent_gaps_ignore_none_and_resolve_by_window(self):
        memory = init_observation_memory()
        uncertain = {
            "observer_backend": "local_qwen",
            "scene_id": "S001",
            "t_range": [30.0, 40.0],
            "overall_summary": "An animal is visible, species unclear.",
            "detail_sufficient": False,
            "missing_detail": "None; species and enclosure type are clear.",
            "scene_summaries": [
                {
                    "scene_id": "S001",
                    "t_range": [30.0, 40.0],
                    "summary": "An animal is visible.",
                    "missing_detail": "verify the animal species",
                }
            ],
        }
        merge_tool_observation(
            memory,
            tool_name="focus_qwen",
            parameters={"start_time": 30.0, "end_time": 40.0},
            output=observation(uncertain),
            persistent_open_gaps=True,
        )
        self.assertEqual(len(memory["open_gaps"]), 1)
        self.assertEqual(memory["open_gaps"][0]["status"], "open")

        verified = {
            "observer_backend": "api",
            "scene_id": "S001",
            "t_range": [30.0, 40.0],
            "overall_summary": "The animal is a sheep.",
            "detail_sufficient": True,
        }
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 30.0, "end_time": 40.0},
            output=observation(verified),
            persistent_open_gaps=True,
        )
        self.assertEqual(memory["open_gaps"][0]["status"], "resolved")

    def test_overview_gaps_stay_on_routing_candidates(self):
        memory = init_observation_memory()
        payload = {
            "observer_backend": "api",
            "scene_summaries": [
                {
                    "scene_id": "S005",
                    "t_range": [481.0, 575.6],
                    "summary": "Rural pastures and livestock.",
                    "possible_evidence": False,
                    "missing_detail": "Confirm which livestock is kept here.",
                }
            ],
        }
        merge_tool_observation(
            memory,
            tool_name="overview",
            parameters={},
            output=observation(payload),
            use_structured_evidence_state=True,
            scope_aware_evidence_memory=True,
            persistent_open_gaps=True,
        )
        self.assertEqual(memory["open_gaps"], [])
        self.assertEqual(memory["structured_evidence"]["uncertainties"], [])
        self.assertEqual(
            memory["scene_memory"][0]["missing_detail"],
            "Confirm which livestock is kept here.",
        )

    def test_multi_window_focus_aggregates_child_observations(self):
        calls = []

        def fake_single(_config, parameters):
            calls.append([parameters["start_time"], parameters["end_time"]])
            index = len(calls)
            payload = {
                "scene_id": f"S{index:03d}",
                "window_id": "focus_window",
                "t_range": calls[-1],
                "overall_summary": f"candidate {index}",
                "possible_evidence": index == 2,
                "detail_sufficient": False,
                "observer_backend": "local_qwen",
                "observer_wall_s": 2.0,
                "num_frames": 8,
                "parse_ok": True,
                "timestamp_observations": [
                    {
                        "timestamp_s": parameters["start_time"],
                        "scene_id": f"S{index:03d}",
                        "description": f"frame {index}",
                    }
                ],
            }
            return observation(payload)

        parameters = {
            "query": "locate the staircase",
            "start_time": 10.0,
            "end_time": 70.0,
            "windows": [[10.0, 18.0], [52.0, 60.0]],
            "mode": "detail_verify",
            "duration": 100.0,
            "vr": object(),
        }
        with patch.object(focus_qwen_module, "_execute_focus_qwen_single", fake_single):
            output = focus_qwen_module.execute_focus_qwen(
                {
                    "focus_qwen_multi_window_enabled": True,
                    "focus_qwen_multi_window_max_windows": 4,
                },
                parameters,
            )

        payload = extract_json_object(output)
        self.assertEqual(calls, [[10.0, 18.0], [52.0, 60.0]])
        self.assertTrue(payload["multi_window_focus"])
        self.assertEqual(len(payload["window_observations"]), 2)
        self.assertEqual(len(payload["timestamp_observations"]), 2)
        self.assertEqual(payload["num_frames"], 16)
        self.assertTrue(payload["possible_evidence"])


if __name__ == "__main__":
    unittest.main()
