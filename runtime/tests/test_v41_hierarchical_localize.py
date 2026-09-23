import tempfile
import unittest
from unittest.mock import patch

from videoseek.agent import VideoSeekAgent
from videoseek.core import Action
from videoseek.core.evidence_state import _initial_level
from videoseek.core.memory import extract_v10_payload
from videoseek.tools import DEFAULT_TOOL_REGISTRY
from videoseek.tools.localize_qwen import execute_localize_qwen
from videoseek.tools.v10_format import format_v10_observation


class _VideoReaderStub:
    def __len__(self):
        return 3000

    def get_avg_fps(self):
        return 30.0


class _RegistryStub:
    def __init__(self, answer_result="fallback-answer"):
        self.answer_result = answer_result
        self.answer_calls = 0

    def has_tool(self, name):
        return name == "answer"

    def get_function(self, name):
        def execute(*, config, parameters):
            self.answer_calls += 1
            return self.answer_result

        return execute


class V41DirectPlannerAnswerTests(unittest.TestCase):
    def _agent(self):
        agent = VideoSeekAgent.__new__(VideoSeekAgent)
        agent.duration = 100.0
        agent.max_steps = 20
        agent.question = "Question\nA. one\nB. two\nC. three\nD. four"
        agent.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        agent.observation_memory = {}
        agent.config = {
            "coseek1_structured_planner": True,
            "coseek1_direct_planner_answer": True,
            "coseek1_localize_qwen_enabled": True,
            "coseek1_planner_json_action": True,
            "localize_qwen_max_search_windows": 8,
            "localize_qwen_max_top_k": 4,
        }
        agent.tool_registry = DEFAULT_TOOL_REGISTRY
        agent.allowed_tool_names = {"overview", "localize_qwen", "frame_verify", "answer"}
        return agent

    def test_parser_preserves_direct_answer_and_support_refs(self):
        agent = self._agent()
        thought = (
            '{"state":"done","action":{"tool":"answer","answer":"D",'
            '"support_refs":["E00128"]},"why":"verified"}'
        )
        actions = agent._VideoSeekAgent__extract_planner_json_actions(thought)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].parameters, {"answer": "D", "support_refs": ["E00128"]})

    def test_valid_direct_answer_skips_answer_executor(self):
        agent = self._agent()
        registry = _RegistryStub()
        agent.tool_registry = registry
        outcome = agent._VideoSeekAgent__exec_action(
            Action("answer", {"answer": "C", "support_refs": ["E1"]}, "answer")
        )
        self.assertEqual(outcome, "C")
        self.assertEqual(registry.answer_calls, 0)

    def test_invalid_direct_answer_falls_back_to_v40_answer_tool(self):
        agent = self._agent()
        registry = _RegistryStub()
        agent.tool_registry = registry
        outcome = agent._VideoSeekAgent__exec_action(
            Action("answer", {"answer": "unknown"}, "answer")
        )
        self.assertEqual(outcome, "fallback-answer")
        self.assertEqual(registry.answer_calls, 1)

    def test_planner_prompt_documents_both_v41_contracts(self):
        agent = self._agent()
        prompt = agent._VideoSeekAgent__format_planner_step_prompt(step=1, memory_text="memory")
        self.assertIn("localize_qwen", prompt)
        self.assertIn('"support_refs"', prompt)
        self.assertIn('"search_windows"', prompt)


class V41LocalizeQwenTests(unittest.TestCase):
    def test_tool_is_registered_without_removing_old_tools(self):
        for name in ("localize_qwen", "skim_qwen", "focus_qwen"):
            self.assertTrue(DEFAULT_TOOL_REGISTRY.has_tool(name))

    def test_localize_runs_coarse_then_fine_and_returns_one_candidate_payload(self):
        skim_calls = []
        focus_calls = []

        def fake_skim(config, parameters):
            skim_calls.append((dict(config), dict(parameters)))
            return format_v10_observation(
                {
                    "t_range": [parameters["start_time"], parameters["end_time"]],
                    "suggest_focus_windows": [[12.0, 22.0], [70.0, 80.0]],
                    "timestamp_observations": [
                        {"timestamp_s": 16.0, "description": "target event begins", "confidence": "high"},
                        {"timestamp_s": 74.0, "description": "possible repeat", "confidence": "medium"},
                    ],
                    "possible_evidence": True,
                    "relevance": 0.8,
                    "observer_backend": "local_qwen",
                    "observer_wall_s": 3.0,
                    "num_frames": 20,
                    "observer_batch_count": 3,
                    "parse_ok": True,
                }
            )

        def fake_focus(config, parameters):
            focus_calls.append((dict(config), dict(parameters)))
            rows = [
                {"timestamp_s": 15.0, "description": "actor performs target action"},
                {"timestamp_s": 75.0, "description": "different actor walks away"},
            ]
            return format_v10_observation(
                {
                    "t_range": [12.0, 80.0],
                    "timestamp_observations": rows,
                    "overall_summary": "localized two alternatives",
                    "observer_backend": "local_qwen",
                    "observer_wall_s": 2.0,
                    "num_frames": 12,
                    "observer_batch_count": 2,
                    "parse_ok": True,
                }
            )

        with tempfile.TemporaryDirectory() as output_dir, patch(
            "videoseek.tools.localize_qwen.execute_skim_qwen", side_effect=fake_skim
        ), patch(
            "videoseek.tools.localize_qwen.execute_focus_qwen", side_effect=fake_focus
        ):
            output = execute_localize_qwen(
                {
                    "localize_qwen_coarse_max_frames": 32,
                    "localize_qwen_fine_max_frames": 16,
                    "localize_qwen_verify_window_s": 10.0,
                    "localize_qwen_trace_enabled": True,
                    "localize_qwen_max_top_k": 4,
                    "localize_qwen_max_search_windows": 8,
                    "skim_qwen_candidate_window_s": 18.0,
                },
                {
                    "vr": _VideoReaderStub(),
                    "duration": 100.0,
                    "subtitles": [],
                    "output_dir": output_dir,
                    "search_windows": [[0.0, 100.0]],
                    "localization_goal": "find the target action",
                    "evidence_profile": "event_boundary",
                    "top_k": 2,
                },
            )

        payload = extract_v10_payload(output)
        self.assertEqual(len(skim_calls), 1)
        self.assertEqual(len(focus_calls), 1)
        self.assertEqual(len(focus_calls[0][1]["windows"]), 2)
        self.assertEqual(payload["tool"], "localize_qwen")
        self.assertEqual(payload["evidence_level"], "candidate")
        self.assertEqual(len(payload["ranked_candidates"]), 2)
        self.assertEqual(payload["internal_qwen_calls"], 5)
        self.assertTrue(payload["trace_path"])

    def test_localize_never_promotes_qwen_output_to_verified(self):
        self.assertEqual(
            _initial_level("localize_qwen", {"detail_sufficient": True, "parse_ok": True}),
            "candidate",
        )

    def test_row_relevance_recovers_candidate_ignored_by_suggested_window(self):
        focus_calls = []

        def fake_skim(_config, parameters):
            return format_v10_observation(
                {
                    "t_range": [parameters["start_time"], parameters["end_time"]],
                    "suggest_focus_windows": [[385.7, 392.6]],
                    "timestamp_observations": [
                        {"timestamp_s": 373.0, "description": "two men at staircase", "confidence": "high"},
                        {"timestamp_s": 375.2, "description": "staircase interaction", "confidence": "high"},
                        {"timestamp_s": 390.3, "description": "men walking on street", "confidence": "low"},
                    ],
                    "possible_evidence": True,
                    "relevance": 0.7,
                    "observer_backend": "local_qwen",
                    "num_frames": 3,
                    "parse_ok": True,
                }
            )

        def fake_focus(_config, parameters):
            focus_calls.append(dict(parameters))
            return format_v10_observation(
                {
                    "timestamp_observations": [
                        {"timestamp_s": 374.0, "description": "staircase interaction"}
                    ],
                    "observer_backend": "local_qwen",
                    "num_frames": 3,
                    "parse_ok": True,
                }
            )

        with patch(
            "videoseek.tools.localize_qwen.execute_skim_qwen", side_effect=fake_skim
        ), patch(
            "videoseek.tools.localize_qwen.execute_focus_qwen", side_effect=fake_focus
        ):
            output = execute_localize_qwen(
                {
                    "localize_qwen_coarse_max_frames": 8,
                    "localize_qwen_fine_max_frames": 4,
                    "localize_qwen_verify_window_s": 10.0,
                    "localize_qwen_trace_enabled": False,
                    "localize_qwen_max_top_k": 4,
                    "localize_qwen_max_search_windows": 8,
                    "localize_qwen_row_candidate_windows": True,
                    "skim_qwen_candidate_window_s": 18.0,
                },
                {
                    "vr": _VideoReaderStub(),
                    "duration": 400.0,
                    "subtitles": [],
                    "search_windows": [[364.9, 392.6]],
                    "localization_goal": "find staircase interaction",
                    "evidence_profile": "relation",
                    "top_k": 1,
                },
            )

        payload = extract_v10_payload(output)
        self.assertEqual(len(focus_calls), 1)
        self.assertLess(focus_calls[0]["windows"][0][1], 385.7)
        self.assertLess(payload["ranked_candidates"][0]["t_range"][1], 385.7)

    def test_search_window_coverage_expands_effective_top_k(self):
        focus_calls = []

        def fake_skim(_config, parameters):
            start = float(parameters["start_time"])
            return format_v10_observation(
                {
                    "t_range": [start, parameters["end_time"]],
                    "suggest_focus_windows": [[start + 2.0, start + 6.0]],
                    "timestamp_observations": [
                        {"timestamp_s": start + 3.0, "description": "candidate", "confidence": "high"}
                    ],
                    "possible_evidence": True,
                    "observer_backend": "local_qwen",
                    "num_frames": 4,
                    "parse_ok": True,
                }
            )

        def fake_focus(_config, parameters):
            focus_calls.append(dict(parameters))
            return format_v10_observation(
                {
                    "timestamp_observations": [
                        {"timestamp_s": window[0] + 1.0, "description": "candidate"}
                        for window in parameters["windows"]
                    ],
                    "observer_backend": "local_qwen",
                    "num_frames": 6,
                    "parse_ok": True,
                }
            )

        with patch(
            "videoseek.tools.localize_qwen.execute_skim_qwen", side_effect=fake_skim
        ), patch(
            "videoseek.tools.localize_qwen.execute_focus_qwen", side_effect=fake_focus
        ):
            output = execute_localize_qwen(
                {
                    "localize_qwen_coarse_max_frames": 12,
                    "localize_qwen_fine_max_frames": 8,
                    "localize_qwen_verify_window_s": 8.0,
                    "localize_qwen_trace_enabled": False,
                    "localize_qwen_max_top_k": 3,
                    "localize_qwen_max_search_windows": 8,
                    "localize_qwen_cover_search_windows": True,
                    "skim_qwen_candidate_window_s": 12.0,
                },
                {
                    "vr": _VideoReaderStub(),
                    "duration": 100.0,
                    "subtitles": [],
                    "search_windows": [[0.0, 20.0], [40.0, 60.0], [80.0, 100.0]],
                    "localization_goal": "find repeated action",
                    "evidence_profile": "event_boundary",
                    "top_k": 1,
                },
            )

        payload = extract_v10_payload(output)
        self.assertEqual(len(focus_calls[0]["windows"]), 3)
        self.assertEqual(payload["coverage"]["requested_top_k"], 1)
        self.assertEqual(payload["coverage"]["effective_top_k"], 3)
        self.assertEqual(len(payload["ranked_candidates"]), 3)


if __name__ == "__main__":
    unittest.main()
