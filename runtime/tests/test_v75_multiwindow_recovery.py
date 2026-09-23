import json
import unittest
from unittest.mock import patch

import numpy as np

from videoseek.core.memory import extract_v10_payload
from videoseek.tools.frame_verify import execute_frame_verify


class _Batch:
    def __init__(self, frames):
        self._frames = frames

    def asnumpy(self):
        return self._frames


class _VideoReaderStub:
    def __len__(self):
        return 3000

    def get_avg_fps(self):
        return 30.0

    def get_batch(self, indices):
        return _Batch(np.zeros((len(indices), 24, 32, 3), dtype=np.uint8))


def _timestamps(**kwargs):
    start = float(kwargs["start_time"])
    end = float(kwargs["end_time"])
    middle = (start + end) / 2.0
    return [start + 1.0, middle, end - 1.0]


def _response(*observations, detail_sufficient=True):
    return json.dumps(
        {
            "timestamp_observations": list(observations),
            "overall_summary": "Candidate windows were inspected.",
            "target_match": "matched",
            "target_binding_reason": "The returned facts come from the requested frames.",
            "observed_fact": "The candidate windows contain visible evidence.",
            "supports_options": [],
            "contradicts_options": [],
            "option_evidence": {},
            "detail_sufficient": detail_sufficient,
            "missing_detail": "",
        }
    )


def _parameters(candidate_windows):
    return {
        "vr": _VideoReaderStub(),
        "video_path": "/tmp/video.mp4",
        "duration": 100.0,
        "output_dir": "/tmp",
        "question": "Which event happened first?\n(A) one\n(B) two\n(C) three\n(D) four",
        "query": "Inspect every candidate event neutrally.",
        "mode": "temporal_strip",
        "candidate_windows": candidate_windows,
    }


class V75MultiwindowRecoveryTests(unittest.TestCase):
    def test_partial_response_retries_only_missing_candidate(self):
        calls = []
        responses = iter(
            [
                _response(
                    {
                        "candidate_id": "LQ001",
                        "timestamp_s": 11.0,
                        "description": "first event is visible",
                    }
                ),
                _response(
                    {
                        "candidate_id": "LQ002",
                        "timestamp_s": 72.0,
                        "description": "second event is visible",
                    }
                ),
            ]
        )

        def fake_observe(
            config, *, content, tool_name, tool_mode, output_dir, return_json=False
        ):
            calls.append(content)
            return next(responses), "api"

        with patch(
            "videoseek.tools.frame_verify.scene_aware_timestamps",
            side_effect=_timestamps,
        ), patch(
            "videoseek.tools.frame_verify.observe_content",
            side_effect=fake_observe,
        ):
            output = execute_frame_verify(
                {
                    "localize_inline_verify_max_windows": 3,
                    "localize_inline_verify_max_frames": 6,
                    "multiwindow_verify_recovery_enabled": True,
                    "multiwindow_verify_recovery_max_calls": 2,
                    "multiwindow_verify_recovery_batch_size": 2,
                },
                _parameters(
                    [
                        {"candidate_id": "LQ001", "t_range": [10.0, 14.0]},
                        {"candidate_id": "LQ002", "t_range": [70.0, 75.0]},
                    ]
                ),
            )

        payload = extract_v10_payload(output)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("LQ001 |", json.dumps(calls[1]))
        self.assertIn("LQ002 |", json.dumps(calls[1]))
        self.assertEqual(payload["verified_candidate_ids"], ["LQ001", "LQ002"])
        self.assertEqual(payload["missing_candidate_ids"], [])
        self.assertTrue(payload["candidate_coverage_complete"])
        self.assertTrue(payload["multiwindow_recovery_used"])
        self.assertEqual(payload["multiwindow_recovery_calls"], 1)

    def test_empty_initial_response_recovers_four_candidates_in_two_batches(self):
        calls = []
        responses = iter(
            [
                None,
                _response(
                    {"candidate_id": "EVT01", "timestamp_s": 11.0, "description": "event one"},
                    {"candidate_id": "EVT02", "timestamp_s": 31.0, "description": "event two"},
                ),
                _response(
                    {"candidate_id": "EVT03", "timestamp_s": 51.0, "description": "event three"},
                    {"candidate_id": "EVT04", "timestamp_s": 71.0, "description": "event four"},
                ),
            ]
        )

        def fake_observe(
            config, *, content, tool_name, tool_mode, output_dir, return_json=False
        ):
            calls.append(content)
            return next(responses), "api"

        parameters = _parameters(
            [
                {"candidate_id": "EVT01", "t_range": [10.0, 14.0]},
                {"candidate_id": "EVT02", "t_range": [30.0, 34.0]},
                {"candidate_id": "EVT03", "t_range": [50.0, 54.0]},
                {"candidate_id": "EVT04", "t_range": [70.0, 74.0]},
            ]
        )
        parameters["inline_event_coverage"] = True
        parameters["event_coverage_max_windows"] = 4
        with patch(
            "videoseek.tools.frame_verify.scene_aware_timestamps",
            side_effect=_timestamps,
        ), patch(
            "videoseek.tools.frame_verify.observe_content",
            side_effect=fake_observe,
        ):
            output = execute_frame_verify(
                {
                    "localize_inline_verify_max_windows": 3,
                    "localize_inline_verify_max_frames": 8,
                    "multiwindow_verify_recovery_enabled": True,
                    "multiwindow_verify_recovery_max_calls": 2,
                    "multiwindow_verify_recovery_batch_size": 2,
                },
                parameters,
            )

        payload = extract_v10_payload(output)
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            payload["verified_candidate_ids"],
            ["EVT01", "EVT02", "EVT03", "EVT04"],
        )
        self.assertEqual(payload["missing_candidate_ids"], [])
        self.assertTrue(payload["candidate_coverage_complete"])
        self.assertEqual(payload["multiwindow_recovery_calls"], 2)

    def test_recovery_can_be_disabled_for_ablation(self):
        calls = []

        def fake_observe(
            config, *, content, tool_name, tool_mode, output_dir, return_json=False
        ):
            calls.append(content)
            return _response(
                {
                    "candidate_id": "LQ001",
                    "timestamp_s": 11.0,
                    "description": "only the first candidate was described",
                }
            ), "api"

        with patch(
            "videoseek.tools.frame_verify.scene_aware_timestamps",
            side_effect=_timestamps,
        ), patch(
            "videoseek.tools.frame_verify.observe_content",
            side_effect=fake_observe,
        ):
            output = execute_frame_verify(
                {
                    "localize_inline_verify_max_windows": 3,
                    "localize_inline_verify_max_frames": 6,
                    "multiwindow_verify_recovery_enabled": False,
                },
                _parameters(
                    [
                        {"candidate_id": "LQ001", "t_range": [10.0, 14.0]},
                        {"candidate_id": "LQ002", "t_range": [70.0, 75.0]},
                    ]
                ),
            )

        payload = extract_v10_payload(output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(payload["verified_candidate_ids"], ["LQ001"])
        self.assertEqual(payload["missing_candidate_ids"], ["LQ002"])
        self.assertFalse(payload["candidate_coverage_complete"])


if __name__ == "__main__":
    unittest.main()
