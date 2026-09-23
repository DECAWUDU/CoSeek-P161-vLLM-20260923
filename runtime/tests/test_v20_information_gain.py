import json
import unittest

from videoseek.agent import VideoSeekAgent
from videoseek.core import Action
from videoseek.core.memory import extract_v10_payload
from videoseek.tools import skim_qwen as skim_module
from videoseek.tools.skim_qwen import (
    _density_timestamps,
    _execute_density_batched_skim_qwen,
    _skim_frame_budget,
    _temporal_density_target_fps,
)
from videoseek.tools.v10_format import format_v10_observation


class TemporalDensitySamplingTest(unittest.TestCase):
    def test_adaptive_density_budget_tracks_window_duration(self):
        config = {
            "skim_qwen_temporal_density_enabled": True,
            "skim_qwen_adaptive_density_enabled": True,
            "skim_qwen_density_short_fps": 1.0,
            "skim_qwen_density_mid_fps": 0.75,
            "skim_qwen_density_long_fps": 0.5,
            "skim_qwen_density_max_frames": 48,
        }
        expected = {10.0: 10, 20.0: 15, 40.0: 20, 90.0: 45, 120.0: 48}
        for window_s, budget in expected.items():
            with self.subTest(window_s=window_s):
                self.assertEqual(
                    _skim_frame_budget(
                        config,
                        window_s=window_s,
                        query_aware_probe=False,
                    ),
                    budget,
                )
        self.assertEqual(_temporal_density_target_fps(config, window_s=10.0), 1.0)
        self.assertEqual(_temporal_density_target_fps(config, window_s=20.0), 0.75)
        self.assertEqual(_temporal_density_target_fps(config, window_s=40.0), 0.5)

    def test_density_timestamp_count_is_exact(self):
        timestamps = _density_timestamps(
            start_time=41.0,
            end_time=47.0,
            target_fps=1.0,
            frame_count=8,
        )
        self.assertEqual(len(timestamps), 8)
        self.assertEqual(timestamps[0], 41.0)
        self.assertEqual(timestamps[-1], 47.0)

    def test_one_fps_budget_uses_window_duration(self):
        config = {
            "skim_qwen_temporal_density_enabled": True,
            "skim_qwen_target_fps": 1.0,
        }
        self.assertEqual(
            _skim_frame_budget(
                config,
                window_s=49.3,
                query_aware_probe=False,
            ),
            50,
        )
        self.assertEqual(
            _skim_frame_budget(
                {**config, "skim_qwen_target_fps": 2.0},
                window_s=49.3,
                query_aware_probe=False,
            ),
            99,
        )

    def test_density_timestamps_are_global_and_regular(self):
        timestamps = _density_timestamps(
            start_time=320.6,
            end_time=325.1,
            target_fps=1.0,
        )
        self.assertEqual(timestamps, [320.6, 321.6, 322.6, 323.6, 324.6])

    def test_logical_skim_aggregates_micro_batches(self):
        original_execute = skim_module.execute_skim_qwen

        def fake_execute(config, parameters):
            start = float(parameters["start_time"])
            end = float(parameters["end_time"])
            timestamps = _density_timestamps(
                start_time=start,
                end_time=end,
                target_fps=float(config["skim_qwen_target_fps"]),
                frame_count=int(parameters.get("density_frame_budget") or 1),
            )
            rows = [
                {
                    "frame_id": f"F{idx + 1:02d}",
                    "timestamp_s": ts,
                    "caption": f"visible event at {ts:.1f}s",
                    "query_relevance": "medium",
                    "scene_id": "S000",
                }
                for idx, ts in enumerate(timestamps)
            ]
            return format_v10_observation(
                {
                    "window_id": "S000_skim_qwen_window",
                    "scene_id": "S000",
                    "t_range": [start, end],
                    "timestamp_observations": [
                        {
                            "timestamp_s": row["timestamp_s"],
                            "scene_id": "S000",
                            "description": row["caption"],
                            "confidence": row["query_relevance"],
                            "frame_ids": [row["frame_id"]],
                        }
                        for row in rows
                    ],
                    "observed_event": "visible event",
                    "possible_evidence": True,
                    "relevance": 0.4,
                    "suggest_focus_windows": [[start, min(end, start + 10.0)]],
                    "missing_detail": "",
                    "observer_backend": "local_qwen",
                    "observer_wall_s": 1.0,
                    "parse_ok": True,
                }
            )

        skim_module.execute_skim_qwen = fake_execute
        try:
            output = _execute_density_batched_skim_qwen(
                {
                    "skim_qwen_temporal_density_enabled": True,
                    "skim_qwen_target_fps": 1.0,
                    "skim_qwen_density_batch_frames": 24,
                    "skim_qwen_max_candidate_windows": 4,
                },
                {"query": "find event"},
                start_time=0.0,
                end_time=49.3,
                scene_id="S000",
                window_id="S000_skim_qwen_window",
                query="find event",
            )
        finally:
            skim_module.execute_skim_qwen = original_execute

        payload = extract_v10_payload(output)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["observer_batch_count"], 3)
        self.assertEqual(payload["num_frames"], 50)
        self.assertEqual(payload["sampled_timestamps"][0], 0.0)
        self.assertEqual(payload["sampled_timestamps"][-1], 49.3)


class VisitedWindowRoutingTest(unittest.TestCase):
    def _agent(self):
        agent = object.__new__(VideoSeekAgent)
        agent.config = {
            "coseek1_visited_window_information_gain": True,
            "coseek1_visited_window_overlap": 0.85,
            "focus_qwen_max_window_s": 20.0,
            "coseek1_requirement_candidate_max_window_s": 24.0,
            "coseek1_planner_json_action": True,
        }
        agent.duration = 200.0
        agent.max_steps = 20
        agent.question = "What happened?\n(A) A\n(B) B\n(C) C\n(D) D"
        agent.messages = [{"role": "assistant", "content": "{}"}]
        agent.observation_memory = {
            "tool_observations": [
                {
                    "tool": "skim_qwen",
                    "parameters": {"start_time": 0.0, "end_time": 60.0},
                    "parsed": True,
                    "observer_backend": "local_qwen",
                }
            ],
            "scene_memory": [],
            "timestamped_observations": [],
        }
        return agent

    def test_repeat_advances_to_unvisited_focus_candidate(self):
        agent = self._agent()
        agent.observation_memory["scene_memory"] = [
            {
                "source_tool": "skim_qwen",
                "scene_id": "S000",
                "t_range": [0.0, 60.0],
                "summary": "candidate event",
                "possible_evidence": True,
                "suggest_focus_windows": [[32.0, 48.0]],
                "observer_backend": "local_qwen",
            }
        ]
        action = Action(
            function_name="skim_qwen",
            parameters={
                "query": "find event",
                "start_time": 0.0,
                "end_time": 60.0,
                "mode": "normal",
            },
            function_id="repeat",
        )
        routed, _ = agent._VideoSeekAgent__route_repeated_skim_for_information_gain(
            [action],
            step=4,
            thought="{}",
        )
        self.assertEqual(routed[0].function_name, "focus_qwen")
        self.assertEqual(routed[0].parameters["start_time"], 32.0)
        self.assertEqual(routed[0].parameters["end_time"], 48.0)

    def test_repeat_without_focus_moves_to_next_overview_scene(self):
        agent = self._agent()
        agent.observation_memory["scene_memory"] = [
            {
                "source_tool": "overview",
                "scene_id": "S000",
                "t_range": [0.0, 60.0],
                "summary": "first candidate",
                "possible_evidence": True,
                "suggest_focus_windows": [],
            },
            {
                "source_tool": "overview",
                "scene_id": "S001",
                "t_range": [100.0, 150.0],
                "summary": "second candidate with interaction",
                "possible_evidence": True,
                "suggest_focus_windows": [],
            },
        ]
        action = Action(
            function_name="skim_qwen",
            parameters={
                "query": "find event",
                "start_time": 0.0,
                "end_time": 60.0,
                "mode": "normal",
            },
            function_id="repeat",
        )
        routed, _ = agent._VideoSeekAgent__route_repeated_skim_for_information_gain(
            [action],
            step=4,
            thought="{}",
        )
        self.assertEqual(routed[0].function_name, "skim_qwen")
        self.assertEqual(routed[0].parameters["start_time"], 100.0)
        self.assertEqual(routed[0].parameters["end_time"], 150.0)


if __name__ == "__main__":
    unittest.main()
