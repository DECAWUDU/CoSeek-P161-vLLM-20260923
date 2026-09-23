from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from videoseek.core.memory import extract_v10_payload
from videoseek.tools.frame_verify import _frame_data_url, execute_frame_verify


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
        return _Batch(np.zeros((len(indices), 48, 64, 3), dtype=np.uint8))


def _parameters():
    return {
        "vr": _VideoReaderStub(),
        "video_path": "/tmp/video.mp4",
        "duration": 100.0,
        "output_dir": "/tmp",
        "question": "What is held?\n(A) cup\n(B) flower\n(C) book\n(D) phone",
        "query": "Neutrally inspect what the person holds.",
        "mode": "option_verify",
        "candidate_windows": [
            {
                "candidate_id": "LQ001",
                "t_range": [10.0, 14.0],
                "summary": "person near a cup",
                "timestamp_anchors": [12.0],
            },
            {
                "candidate_id": "LQ002",
                "t_range": [70.0, 75.0],
                "summary": "person holds a small object",
                "timestamp_anchors": [72.5],
            },
        ],
    }


def _initial_response(*, request_detail: bool) -> str:
    return json.dumps(
        {
            "candidate_assessments": [
                {
                    "candidate_id": "LQ001",
                    "target_match": "mismatch",
                    "observed_fact": "A cup remains on the table.",
                    "target_binding_reason": "No hand-cup contact is visible.",
                    "best_timestamp_s": 12.0,
                    "supports_options": [],
                    "contradicts_options": ["A"],
                    "option_set_conflict": False,
                },
                {
                    "candidate_id": "LQ002",
                    "target_match": "partial",
                    "observed_fact": "The hand grasps a small stem-like object.",
                    "target_binding_reason": "The hand and object are co-visible.",
                    "best_timestamp_s": 72.5,
                    "supports_options": [],
                    "contradicts_options": [],
                    "option_set_conflict": False,
                },
            ],
            "timestamp_observations": [
                {"candidate_id": "LQ001", "timestamp_s": 12.0, "description": "Cup on table."},
                {"candidate_id": "LQ002", "timestamp_s": 72.5, "description": "Small held object."},
            ],
            "overall_summary": "LQ002 is the best bound candidate, but its small object is unclear.",
            "target_match": "partial",
            "scope_coverage": "sufficient",
            "supports_options": [],
            "contradicts_options": ["A"],
            "option_evidence": {},
            "detail_sufficient": False,
            "missing_detail": "The small held object needs a local high-resolution view.",
            "detail_escalation": {
                "needed": request_detail,
                "candidate_id": "LQ002" if request_detail else "",
                "timestamp_s": 72.5 if request_detail else None,
                "query": "Inspect only the held object's shape and petals." if request_detail else "",
                "reason": "Object is too small in the full anchor." if request_detail else "",
            },
        }
    )


def _detail_response() -> str:
    return json.dumps(
        {
            "candidate_assessments": [
                {
                    "candidate_id": "LQ002",
                    "target_match": "matched",
                    "observed_fact": "The crop shows flower petals attached to held stems.",
                    "target_binding_reason": "The cropped petals connect to stems in the hand.",
                    "best_timestamp_s": 72.5,
                    "supports_options": ["B"],
                    "contradicts_options": ["A", "C", "D"],
                    "option_set_conflict": False,
                }
            ],
            "timestamp_observations": [
                {"candidate_id": "LQ002", "timestamp_s": 72.5, "description": "Held flower petals and stems."}
            ],
            "overall_summary": "The local crop resolves the held object as flowers.",
            "observed_fact": "The person holds flowers.",
            "target_match": "matched",
            "scope_coverage": "sufficient",
            "supports_options": ["B"],
            "contradicts_options": ["A", "C", "D"],
            "option_evidence": {"B": {"status": "support", "reason": "Visible petals and stems."}},
            "detail_sufficient": True,
            "missing_detail": "",
        }
    )


def _config():
    return {
        "localize_inline_verify_max_windows": 3,
        "localize_inline_verify_max_frames": 6,
        "multiwindow_verify_recovery_enabled": False,
        "grounded_candidate_binding_enabled": True,
        "grounded_frame_verify_enabled": False,
        "grounded_verify_packet_enabled": True,
        "grounded_verify_packet_anchors_per_candidate": 1,
        "packet_detail_grounding_escalation_enabled": True,
        "packet_detail_grounding_escalation_frames": 4,
    }


def _timestamps(**kwargs):
    start, end = float(kwargs["start_time"]), float(kwargs["end_time"])
    return [start, (start + end) / 2.0, end]


class PacketAdaptiveGroundingTests(unittest.TestCase):
    def test_no_explicit_request_does_not_run_grounding(self):
        calls = []

        def fake_observe(*args, **kwargs):
            calls.append(kwargs["content"])
            return _initial_response(request_detail=False), "api"

        with patch("videoseek.tools.frame_verify.scene_aware_timestamps", side_effect=_timestamps), patch(
            "videoseek.tools.frame_verify.observe_content", side_effect=fake_observe
        ), patch("videoseek.tools.frame_verify.prepare_grounded_frames") as grounding:
            output = execute_frame_verify(_config(), _parameters())

        self.assertEqual(len(calls), 1)
        grounding.assert_not_called()
        self.assertIsNotNone(extract_v10_payload(output))

    def test_valid_crop_runs_one_candidate_one_anchor_retry(self):
        responses = iter([_initial_response(request_detail=True), _detail_response()])
        api_calls = []
        grounding_calls = []

        def fake_observe(*args, **kwargs):
            api_calls.append(kwargs["content"])
            return next(responses), "api"

        crop = {
            "type": "image_url",
            "image_url": {
                "url": _frame_data_url(np.zeros((24, 32, 3), dtype=np.uint8)),
                "detail": "high",
            },
        }

        def fake_grounding(config, **kwargs):
            grounding_calls.append((dict(config), kwargs))
            return (
                {1: [{"type": "text", "text": "crop"}, crop]},
                {
                    "enabled": True,
                    "crop_count": 1,
                    "records": [
                        {"position": 1, "candidate_ids": ["LQ002"], "packet_mode": "crop_only"}
                    ],
                },
            )

        with patch("videoseek.tools.frame_verify.scene_aware_timestamps", side_effect=_timestamps), patch(
            "videoseek.tools.frame_verify.observe_content", side_effect=fake_observe
        ), patch(
            "videoseek.tools.frame_verify.prepare_grounded_frames", side_effect=fake_grounding
        ):
            output = execute_frame_verify(_config(), _parameters())

        payload = extract_v10_payload(output)
        self.assertEqual(len(api_calls), 2)
        self.assertEqual(len(grounding_calls), 1)
        grounding_config, grounding_kwargs = grounding_calls[0]
        self.assertEqual(grounding_config["grounded_frame_verify_max_anchors"], 1)
        self.assertEqual(list(grounding_kwargs["candidate_anchor_indices"].values()), [["LQ002"]])
        self.assertTrue(payload["packet_detail_escalation_used"])
        by_id = {row["candidate_id"]: row for row in payload["candidate_assessments"]}
        self.assertEqual(by_id["LQ001"]["target_match"], "mismatch")
        self.assertEqual(by_id["LQ002"]["target_match"], "matched")
        self.assertEqual(payload["supports_options"], ["B"])

    def test_invalid_crop_skips_second_remote_verify(self):
        api_calls = []

        def fake_observe(*args, **kwargs):
            api_calls.append(kwargs["content"])
            return _initial_response(request_detail=True), "api"

        no_crop_audit = {
            "enabled": True,
            "crop_count": 0,
            "records": [
                {"position": 1, "candidate_ids": ["LQ002"], "packet_mode": "full_fallback"}
            ],
        }
        with patch("videoseek.tools.frame_verify.scene_aware_timestamps", side_effect=_timestamps), patch(
            "videoseek.tools.frame_verify.observe_content", side_effect=fake_observe
        ), patch(
            "videoseek.tools.frame_verify.prepare_grounded_frames",
            return_value=({}, no_crop_audit),
        ):
            output = execute_frame_verify(_config(), _parameters())

        payload = extract_v10_payload(output)
        self.assertEqual(len(api_calls), 1)
        self.assertFalse(payload["packet_detail_escalation_used"])
        self.assertTrue(payload["packet_detail_escalation_attempted"])
        self.assertEqual(payload["packet_detail_escalation_skip_reason"], "no_valid_local_crop")
        self.assertEqual(payload["candidate_search_status"], "unresolved")


if __name__ == "__main__":
    unittest.main()
