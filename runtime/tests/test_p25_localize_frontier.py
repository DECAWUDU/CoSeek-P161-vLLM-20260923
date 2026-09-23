import unittest
from unittest.mock import patch

from videoseek.agent import VideoSeekAgent
from videoseek.core import Action
from videoseek.core.local_observation_cache import reset_local_observation_cache
from videoseek.core.memory import extract_v10_payload
from videoseek.tools.localize_qwen import (
    _allocate_duration_budgets,
    execute_localize_qwen,
)
from videoseek.tools.v10_format import format_v10_observation


class _VideoReaderStub:
    def __len__(self):
        return 9000

    def get_avg_fps(self):
        return 30.0


def _rows(start, end, count, *, high_at=None, prefix="frame"):
    if count == 1:
        timestamps = [(start + end) / 2.0]
    else:
        timestamps = [start + (end - start) * index / (count - 1) for index in range(count)]
    rows = []
    for index, timestamp in enumerate(timestamps):
        confidence = "high" if high_at is not None and abs(timestamp - high_at) <= 0.8 else "low"
        rows.append(
            {
                "timestamp_s": round(timestamp, 3),
                "description": f"{prefix} {index}",
                "confidence": confidence,
            }
        )
    return rows


class P25LocalizeFrontierTests(unittest.TestCase):
    def setUp(self):
        reset_local_observation_cache()

    def test_duration_budget_does_not_spend_full_cap_on_short_windows(self):
        budgets = _allocate_duration_budgets(
            [[0.0, 4.0], [20.0, 24.0], [40.0, 44.0]],
            total_budget=96,
            minimum=8,
            target_fps=2.0,
            per_window_max=32,
        )
        self.assertEqual(budgets, [9, 9, 9])
        self.assertLess(sum(budgets), 96)

    def test_second_identical_localize_reuses_caption_cache(self):
        skim_calls = []
        focus_calls = []

        def fake_skim(_config, parameters):
            skim_calls.append(dict(parameters))
            frame_count = int(_config["skim_qwen_density_max_frames"])
            rows = _rows(0.0, 10.0, frame_count, high_at=5.0, prefix="coarse")
            return format_v10_observation(
                {
                    "t_range": [0.0, 10.0],
                    "timestamp_observations": rows,
                    "possible_evidence": True,
                    "relevance": 0.8,
                    "observer_backend": "local_qwen",
                    "observer_wall_s": 2.0,
                    "num_frames": frame_count,
                    "observer_batch_count": 2,
                    "parse_ok": True,
                }
            )

        def fake_focus(_config, parameters):
            focus_calls.append(dict(parameters))
            window = parameters["windows"][0]
            rows = _rows(window[0], window[1], 8, high_at=5.0, prefix="fine")
            return format_v10_observation(
                {
                    "t_range": list(window),
                    "timestamp_observations": rows,
                    "observer_backend": "local_qwen",
                    "observer_wall_s": 1.0,
                    "num_frames": 8,
                    "observer_batch_count": 1,
                    "parse_ok": True,
                }
            )

        config = {
            "localize_qwen_coarse_max_frames": 96,
            "localize_qwen_fine_max_frames": 8,
            "localize_qwen_verify_window_s": 8.0,
            "localize_qwen_trace_enabled": False,
            "localize_qwen_max_top_k": 3,
            "localize_qwen_max_search_windows": 8,
            "localize_qwen_row_candidate_windows": True,
            "localize_qwen_normalized_candidate_score_enabled": True,
            "localize_qwen_duration_budget_enabled": True,
            "localize_qwen_duration_budget_min_frames": 8,
            "localize_qwen_duration_budget_fps": 2.0,
            "localize_qwen_duration_budget_max_frames": 32,
            "localize_qwen_caption_cache_enabled": True,
            "localize_qwen_caption_cache_goal_similarity": 0.72,
            "localize_qwen_caption_cache_coverage_ratio": 0.70,
            "localize_qwen_candidate_novelty_enabled": True,
            "skim_qwen_candidate_window_s": 8.0,
        }
        parameters = {
            "vr": _VideoReaderStub(),
            "video_path": "/tmp/p25_cache_video.mp4",
            "duration": 100.0,
            "subtitles": [],
            "search_windows": [[0.0, 10.0]],
            "localization_goal": "find the person opening the door",
            "evidence_profile": "event_boundary",
            "top_k": 1,
        }

        with patch(
            "videoseek.tools.localize_qwen.execute_skim_qwen", side_effect=fake_skim
        ), patch(
            "videoseek.tools.localize_qwen.execute_focus_qwen", side_effect=fake_focus
        ):
            first = extract_v10_payload(execute_localize_qwen(config, parameters))
            second = extract_v10_payload(execute_localize_qwen(config, parameters))

        self.assertEqual(first["localization_progress"], "new_candidates")
        self.assertEqual(
            second["localization_progress"],
            "no_new_information",
            msg=str(
                {
                    "coverage": second.get("coverage"),
                    "skim_calls": len(skim_calls),
                    "focus_calls": len(focus_calls),
                    "candidates": second.get("ranked_candidates"),
                }
            ),
        )
        self.assertEqual(len(skim_calls), 1)
        self.assertEqual(len(focus_calls), 1)
        self.assertEqual(second["coverage"]["caption_cache_hits"], 2)
        self.assertGreater(second["coverage"]["reused_caption_frames"], 0)
        self.assertEqual(second["num_frames"], 0)

    def test_boundary_expansion_only_scans_relevant_edge(self):
        skim_calls = []

        def fake_skim(_config, parameters):
            skim_calls.append(dict(parameters))
            start = float(parameters["start_time"])
            end = float(parameters["end_time"])
            high_at = 9.5 if start == 0.0 else None
            return format_v10_observation(
                {
                    "t_range": [start, end],
                    "timestamp_observations": _rows(
                        start,
                        end,
                        int(_config["skim_qwen_density_max_frames"]),
                        high_at=high_at,
                    ),
                    "possible_evidence": high_at is not None,
                    "observer_backend": "local_qwen",
                    "num_frames": int(_config["skim_qwen_density_max_frames"]),
                    "observer_batch_count": 1,
                    "parse_ok": True,
                }
            )

        def fake_focus(_config, parameters):
            return format_v10_observation(
                {
                    "timestamp_observations": [
                        {"timestamp_s": 9.5, "description": "target begins", "confidence": "high"}
                    ],
                    "observer_backend": "local_qwen",
                    "num_frames": 1,
                    "observer_batch_count": 1,
                    "parse_ok": True,
                }
            )

        with patch(
            "videoseek.tools.localize_qwen.execute_skim_qwen", side_effect=fake_skim
        ), patch(
            "videoseek.tools.localize_qwen.execute_focus_qwen", side_effect=fake_focus
        ):
            payload = extract_v10_payload(
                execute_localize_qwen(
                    {
                        "localize_qwen_coarse_max_frames": 32,
                        "localize_qwen_fine_max_frames": 8,
                        "localize_qwen_verify_window_s": 8.0,
                        "localize_qwen_trace_enabled": False,
                        "localize_qwen_max_top_k": 3,
                        "localize_qwen_max_search_windows": 8,
                        "localize_qwen_row_candidate_windows": True,
                        "localize_qwen_duration_budget_enabled": True,
                        "localize_qwen_boundary_expansion_enabled": True,
                        "localize_qwen_boundary_context_margin_s": 8.0,
                        "localize_qwen_boundary_trigger_s": 2.0,
                        "localize_qwen_boundary_expansion_max_frames": 8,
                        "skim_qwen_candidate_window_s": 8.0,
                    },
                    {
                        "vr": _VideoReaderStub(),
                        "duration": 100.0,
                        "subtitles": [],
                        "search_windows": [[0.0, 10.0]],
                        "localization_goal": "find target transition",
                        "evidence_profile": "event_boundary",
                        "top_k": 1,
                    },
                )
            )

        self.assertEqual(len(skim_calls), 2)
        self.assertEqual(skim_calls[1]["start_time"], 10.0)
        self.assertEqual(skim_calls[1]["end_time"], 18.0)
        self.assertEqual(payload["coverage"]["boundary_expansion_count"], 1)
        self.assertEqual(payload["searched_windows"], [[0.0, 18.0]])

    def test_paraphrased_goal_reuses_stable_question_cache(self):
        skim_calls = []
        focus_calls = []

        def fake_skim(_config, parameters):
            skim_calls.append(dict(parameters))
            rows = _rows(0.0, 12.0, 25, high_at=6.0, prefix="coarse")
            return format_v10_observation(
                {
                    "t_range": [0.0, 12.0],
                    "timestamp_observations": rows,
                    "possible_evidence": True,
                    "observer_backend": "local_qwen",
                    "num_frames": len(rows),
                    "parse_ok": True,
                }
            )

        def fake_focus(_config, parameters):
            focus_calls.append(dict(parameters))
            window = parameters["windows"][0]
            rows = _rows(window[0], window[1], 8, high_at=6.0, prefix="fine")
            return format_v10_observation(
                {
                    "t_range": list(window),
                    "timestamp_observations": rows,
                    "observer_backend": "local_qwen",
                    "num_frames": len(rows),
                    "parse_ok": True,
                }
            )

        config = {
            "localize_qwen_coarse_max_frames": 96,
            "localize_qwen_fine_max_frames": 8,
            "localize_qwen_verify_window_s": 8.0,
            "localize_qwen_trace_enabled": False,
            "localize_qwen_max_top_k": 3,
            "localize_qwen_max_search_windows": 8,
            "localize_qwen_row_candidate_windows": True,
            "localize_qwen_normalized_candidate_score_enabled": True,
            "localize_qwen_duration_budget_enabled": True,
            "localize_qwen_duration_budget_min_frames": 8,
            "localize_qwen_duration_budget_fps": 2.0,
            "localize_qwen_duration_budget_max_frames": 32,
            "localize_qwen_caption_cache_enabled": True,
            "localize_qwen_caption_cache_goal_similarity": 0.30,
            "localize_qwen_caption_cache_coverage_ratio": 0.70,
            "localize_qwen_candidate_novelty_enabled": True,
            "skim_qwen_candidate_window_s": 8.0,
        }
        common = {
            "vr": _VideoReaderStub(),
            "video_path": "/tmp/p26_semantic_cache.mp4",
            "question": "How many people were physically interacting at the staircase?",
            "duration": 100.0,
            "subtitles": [],
            "search_windows": [[0.0, 12.0]],
            "evidence_profile": "event_boundary",
            "top_k": 1,
        }
        first_parameters = {
            **common,
            "localization_goal": "Find staircase shots with all interacting people visible together",
        }
        second_parameters = {
            **common,
            "localization_goal": "Locate exact moments showing people physically interacting on the staircase",
        }

        with patch(
            "videoseek.tools.localize_qwen.execute_skim_qwen", side_effect=fake_skim
        ), patch(
            "videoseek.tools.localize_qwen.execute_focus_qwen", side_effect=fake_focus
        ):
            first = extract_v10_payload(execute_localize_qwen(config, first_parameters))
            second = extract_v10_payload(execute_localize_qwen(config, second_parameters))

        self.assertEqual(first["localization_progress"], "new_candidates")
        self.assertEqual(second["localization_progress"], "no_new_information")
        self.assertEqual(len(skim_calls), 1)
        self.assertEqual(len(focus_calls), 1)
        self.assertEqual(second["coverage"]["caption_cache_hits"], 2)

    def test_strict_status_keeps_partial_context_ambiguous(self):
        def fake_skim(_config, parameters):
            start = float(parameters["start_time"])
            end = float(parameters["end_time"])
            rows = _rows(start, end, 12, prefix="coarse")
            for row in rows:
                row["confidence"] = "medium"
            return format_v10_observation(
                {
                    "t_range": [start, end],
                    "timestamp_observations": rows,
                    "possible_evidence": True,
                    "observer_backend": "local_qwen",
                    "num_frames": len(rows),
                    "parse_ok": True,
                }
            )

        def fake_focus(_config, parameters):
            window = parameters["windows"][0]
            rows = _rows(window[0], window[1], 8, prefix="fine")
            for row in rows:
                row["confidence"] = "medium"
                row["description"] = "Two people near a railing; no physical contact visible"
            return format_v10_observation(
                {
                    "t_range": list(window),
                    "timestamp_observations": rows,
                    "observer_backend": "local_qwen",
                    "num_frames": len(rows),
                    "parse_ok": True,
                }
            )

        with patch(
            "videoseek.tools.localize_qwen.execute_skim_qwen", side_effect=fake_skim
        ), patch(
            "videoseek.tools.localize_qwen.execute_focus_qwen", side_effect=fake_focus
        ):
            payload = extract_v10_payload(
                execute_localize_qwen(
                    {
                        "localize_qwen_coarse_max_frames": 32,
                        "localize_qwen_fine_max_frames": 8,
                        "localize_qwen_verify_window_s": 8.0,
                        "localize_qwen_trace_enabled": False,
                        "localize_qwen_max_top_k": 3,
                        "localize_qwen_max_search_windows": 8,
                        "localize_qwen_row_candidate_windows": True,
                        "localize_qwen_strict_evidence_status_enabled": True,
                        "skim_qwen_candidate_window_s": 8.0,
                    },
                    {
                        "vr": _VideoReaderStub(),
                        "duration": 100.0,
                        "subtitles": [],
                        "search_windows": [[0.0, 12.0]],
                        "localization_goal": "Find people physically interacting at a staircase",
                        "top_k": 1,
                    },
                )
            )

        self.assertEqual(payload["localization_progress"], "ambiguous_candidates")
        self.assertFalse(payload["possible_evidence"])
        self.assertEqual(payload["suggest_focus_windows"], [])
        self.assertEqual(
            payload["ranked_candidates"][0]["localization_status"], "ambiguous"
        )

    def test_repeated_frontier_does_not_auto_verify_again(self):
        agent = VideoSeekAgent.__new__(VideoSeekAgent)
        agent.config = {
            "coseek1_localize_inline_verify": True,
            "localize_qwen_candidate_novelty_enabled": True,
            "localize_inline_verify_max_windows": 3,
        }
        output = format_v10_observation(
            {
                "parse_ok": True,
                "localization_progress": "no_new_information",
                "ranked_candidates": [
                    {
                        "candidate_id": "LQ001",
                        "localization_status": "found",
                        "novelty_status": "repeated",
                        "recommended_verify_window": [10.0, 18.0],
                    }
                ],
            }
        )
        action = Action(
            "localize_qwen",
            {
                "localization_goal": "find target",
                "evidence_profile": "generic",
            },
            "localize_1",
        )
        result = agent._VideoSeekAgent__inline_localize_verify_action(
            localize_action=action,
            localize_output=output,
            step=2,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
