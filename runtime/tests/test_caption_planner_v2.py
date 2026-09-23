import unittest
from unittest.mock import patch

from videoseek.core.memory import extract_v10_payload, init_observation_memory
from videoseek.core.planner_state import (
    format_caption_planner_state,
    merge_caption_planner_payload,
    resolve_candidate_windows,
)
from videoseek.tools.focus_qwen import execute_focus_qwen
from videoseek.tools import DEFAULT_TOOL_REGISTRY
from videoseek.tools.v10_format import format_v10_observation


class CaptionPlannerStateTest(unittest.TestCase):
    def test_planner_candidates_are_persistent_and_resolvable(self):
        memory = init_observation_memory()
        merge_caption_planner_payload(
            memory,
            {
                "state": "two plausible doorway moments",
                "localization_status": "ambiguous",
                "missing_evidence": "dress color",
                "candidate_updates": [
                    {
                        "candidate_id": "C001",
                        "t_range": [8.0, 14.0],
                        "query": "woman entering doorway",
                        "reason": "9.0s caption",
                        "priority": 2,
                    },
                    {
                        "candidate_id": "C002",
                        "t_range": [53.0, 61.0],
                        "query": "woman entering while men cook",
                        "reason": "57.6s caption plus cooking context",
                        "priority": 1,
                    },
                ],
            },
            duration=80.0,
        )

        windows = resolve_candidate_windows(memory, ["C002", "C001"], max_windows=2)
        self.assertEqual([item["candidate_id"] for item in windows], ["C002", "C001"])
        self.assertEqual(windows[0]["t_range"], [53.0, 61.0])
        rendered = format_caption_planner_state(memory)
        self.assertIn("C002", rendered)
        self.assertIn("missing_evidence=dress color", rendered)

    def test_invalid_candidate_window_is_ignored(self):
        memory = init_observation_memory()
        merge_caption_planner_payload(
            memory,
            {"candidate_updates": [{"candidate_id": "C001", "t_range": [7, 7]}]},
        )
        self.assertEqual(memory["caption_planner_state"]["candidates"], [])

    def test_reused_id_does_not_overwrite_a_different_window(self):
        memory = init_observation_memory()
        merge_caption_planner_payload(
            memory,
            {"candidate_updates": [{"candidate_id": "C001", "t_range": [8, 14]}]},
        )
        merge_caption_planner_payload(
            memory,
            {
                "candidate_updates": [{"candidate_id": "C001", "t_range": [53, 61]}],
                "ranked_verify_windows": [{"candidate_id": "C001", "t_range": [53, 61]}],
            },
        )
        candidates = memory["caption_planner_state"]["candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertEqual({tuple(item["t_range"]) for item in candidates}, {(8.0, 14.0), (53.0, 61.0)})
        self.assertEqual(
            memory["caption_planner_state"]["ranked_verify_windows"][0]["candidate_id"],
            "C002",
        )


class MultiCandidateFocusTest(unittest.TestCase):
    def test_focus_candidates_are_observed_independently(self):
        def fake_single(_config, parameters):
            start = float(parameters["start_time"])
            return format_v10_observation(
                {
                    "scene_id": "S001",
                    "window_id": "W",
                    "t_range": [start, float(parameters["end_time"])],
                    "overall_summary": f"frame near {start:.1f}s",
                    "timestamp_observations": [
                        {"timestamp_s": start, "description": "visible frame"}
                    ],
                    "observer_backend": "local_qwen",
                    "observer_wall_s": 1.0,
                    "num_frames": 1,
                    "parse_ok": True,
                }
            )

        parameters = {
            "query": "doorway",
            "start_time": 8.0,
            "end_time": 61.0,
            "candidate_windows": [
                {"candidate_id": "C001", "t_range": [8.0, 14.0], "query": "first"},
                {"candidate_id": "C002", "t_range": [53.0, 61.0], "query": "second"},
            ],
        }
        with patch("videoseek.tools.focus_qwen._execute_focus_qwen_single", side_effect=fake_single):
            output = execute_focus_qwen(
                {"coseek1_caption_planner_v2": True, "focus_qwen_multi_candidate_max": 3},
                parameters,
            )
        payload = extract_v10_payload(output)
        self.assertEqual(payload["observer_batch_count"], 2)
        self.assertTrue(payload["independent_candidate_windows"])
        self.assertEqual(
            [item["candidate_id"] for item in payload["candidate_observations"]],
            ["C001", "C002"],
        )


class CaptionPlannerActionTest(unittest.TestCase):
    def test_candidate_ids_resolve_to_one_multi_focus_action(self):
        from videoseek.agent import VideoSeekAgent

        agent = object.__new__(VideoSeekAgent)
        agent.config = {
            "coseek1_caption_planner_v2": True,
            "coseek1_structured_planner": True,
            "coseek1_planner_json_action": True,
            "focus_qwen_multi_candidate_max": 3,
            "focus_qwen_max_window_s": 20.0,
        }
        agent.tool_registry = DEFAULT_TOOL_REGISTRY
        agent.allowed_tool_names = {"overview", "skim_qwen", "focus_qwen", "frame_verify", "answer"}
        agent.duration = 80.0
        agent.question = "What happens?"
        agent.observation_memory = init_observation_memory()
        merge_caption_planner_payload(
            agent.observation_memory,
            {
                "candidate_updates": [
                    {"candidate_id": "C001", "t_range": [8, 14], "query": "first"},
                    {"candidate_id": "C002", "t_range": [53, 61], "query": "second"},
                ]
            },
        )
        thought = (
            '{"state":"compare candidates","action":{"tool":"focus_qwen",'
            '"parameters":{"candidate_ids":["C002","C001"],"query":"doorway",'
            '"mode":"normal"}},"why":"compare"}'
        )
        actions = agent._VideoSeekAgent__extract_planner_json_actions(thought)
        actions = agent._VideoSeekAgent__resolve_caption_planner_actions(actions)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].parameters["start_time"], 8.0)
        self.assertEqual(actions[0].parameters["end_time"], 61.0)
        self.assertEqual(
            [item["candidate_id"] for item in actions[0].parameters["candidate_windows"]],
            ["C002", "C001"],
        )

    def test_long_focus_candidate_uses_skim_semantics(self):
        from videoseek.agent import VideoSeekAgent

        agent = object.__new__(VideoSeekAgent)
        agent.config = {
            "coseek1_caption_planner_v2": True,
            "focus_qwen_multi_candidate_max": 3,
            "focus_qwen_max_window_s": 20.0,
        }
        agent.duration = 100.0
        agent.question = "What happens?"
        agent.observation_memory = init_observation_memory()
        merge_caption_planner_payload(
            agent.observation_memory,
            {
                "candidate_updates": [
                    {"candidate_id": "C001", "t_range": [20, 70], "query": "find event"}
                ]
            },
        )
        action = type("ActionLike", (), {
            "function_name": "focus_qwen",
            "parameters": {"candidate_ids": ["C001"], "query": "find event", "mode": "normal"},
            "function_id": "test",
        })()
        resolved = agent._VideoSeekAgent__resolve_caption_planner_actions([action])
        self.assertEqual(resolved[0].function_name, "skim_qwen")
        self.assertEqual(resolved[0].parameters["start_time"], 20.0)
        self.assertEqual(resolved[0].parameters["end_time"], 70.0)


if __name__ == "__main__":
    unittest.main()
