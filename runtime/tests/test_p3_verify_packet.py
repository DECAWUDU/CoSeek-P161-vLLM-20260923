from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from videoseek.core.memory import extract_v10_payload
from videoseek.tools.frame_verify import (
    _bind_candidate_observations,
    _frame_data_url,
    execute_frame_verify,
)
from videoseek.tools.spatial_grounding import (
    build_answer_blind_grounding_prompt,
    prepare_grounded_frames,
    select_grounding_positions,
)


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
        frames = np.zeros((len(indices), 240, 320, 3), dtype=np.uint8)
        return _Batch(frames)


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
                "summary": "person may hold flowers",
                "timestamp_anchors": [72.5],
            },
        ],
    }


def _response_without_duplicate_observations():
    return json.dumps(
        {
            "candidate_assessments": [
                {
                    "candidate_id": "LQ001",
                    "target_match": "mismatch",
                    "observed_fact": "The cup remains on the table.",
                    "target_binding_reason": "No hand-cup contact is visible.",
                    "best_timestamp_s": 12.0,
                    "supports_options": [],
                    "contradicts_options": ["A"],
                    "option_set_conflict": False,
                },
                {
                    "candidate_id": "LQ002",
                    "target_match": "matched",
                    "observed_fact": "The person visibly grasps flower stems.",
                    "target_binding_reason": "The hand and stems are co-visible.",
                    "best_timestamp_s": 72.5,
                    "supports_options": ["B"],
                    "contradicts_options": ["A", "C", "D"],
                    "option_set_conflict": False,
                },
            ],
            "overall_summary": "Only LQ002 shows the held object.",
            "target_match": "matched",
            "scope_coverage": "sufficient",
            "supports_options": ["B"],
            "contradicts_options": ["A", "C", "D"],
            "option_evidence": {},
            "detail_sufficient": True,
            "missing_detail": "",
        }
    )


class VerifyPacketP3Tests(unittest.TestCase):
    def test_balanced_selection_keeps_temporal_backup_per_candidate(self):
        positions = select_grounding_positions(
            frame_indices=[10, 20, 30, 40, 50, 60],
            mandatory_anchor_indices=set(),
            candidate_anchor_indices={
                10: ["C01"],
                20: ["C01"],
                30: ["C02"],
                40: ["C02"],
                50: ["C03"],
                60: ["C03"],
            },
            max_anchors=6,
            candidate_balanced=True,
            max_per_candidate=2,
        )
        self.assertEqual(positions, [0, 1, 2, 3, 4, 5])

    def test_candidate_anchor_cap_prefers_temporal_endpoints(self):
        positions = select_grounding_positions(
            frame_indices=[10, 20, 30, 40],
            mandatory_anchor_indices=set(),
            candidate_anchor_indices={
                10: ["C01"],
                20: ["C01"],
                30: ["C01"],
                40: ["C02"],
            },
            max_anchors=4,
            candidate_balanced=True,
            max_per_candidate=2,
        )
        self.assertEqual(positions, [0, 2, 3])

    def test_high_recall_prompt_proposes_referent_without_answer_choices(self):
        prompt = build_answer_blind_grounding_prompt(
            query="Inspect the bottle on the table and determine its color.",
            question="What color is the bottle?\n(A) Red\n(B) Green",
            high_recall=True,
        )
        self.assertIn("most plausible visible referent", prompt)
        self.assertIn("do not reject a plausible region", prompt)
        self.assertNotIn("(A) Red", prompt)

    def test_packet_grounding_emits_crop_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            config = {
                "grounded_frame_verify_max_anchors": 1,
                "grounded_frame_verify_crop_area_threshold": 0.5,
                "grounded_frame_verify_padding_ratio": 0.18,
                "grounded_frame_verify_min_crop_side_ratio": 0.12,
                "grounded_frame_verify_context_short_side": 64,
                "grounded_frame_verify_max_new_tokens": 160,
                "grounded_frame_verify_cache_enabled": False,
                "grounded_frame_verify_cache_dir": str(Path(tmp)),
                "grounded_verify_packet_enabled": True,
                "grounded_high_recall_proposal_enabled": True,
                "grounded_verify_packet_crop_max_side": 128,
            }
            with patch(
                "videoseek.tools.spatial_grounding.call_local_qwen_parts",
                return_value="VISIBLE yes\nBOX 400 300 600 700\nLABEL target\nCONFIDENCE 85",
            ):
                replacements, audit = prepare_grounded_frames(
                    config,
                    frames=np.asarray([frame]),
                    frame_indices=[10],
                    timestamps=[1.0],
                    mandatory_anchor_indices=set(),
                    candidate_anchor_indices={10: ["LQ001"]},
                    query="Inspect the bottle.",
                    question="What color is the bottle?",
                    output_dir=tmp,
                )

        self.assertEqual(audit["protocol"], "answer_blind_high_recall_bbox_v2")
        self.assertEqual(audit["crop_only_count"], 1)
        self.assertEqual(audit["full_fallback_count"], 0)
        self.assertEqual(
            sum(item["type"] == "image_url" for item in replacements[0]),
            1,
        )

    def test_candidate_packet_uses_two_images_per_candidate(self):
        observed_content = []

        def fake_observe(
            config, *, content, tool_name, tool_mode, output_dir, return_json=False
        ):
            observed_content.extend(content)
            return _response_without_duplicate_observations(), "api"

        crop = {
            "type": "image_url",
            "image_url": {
                "url": _frame_data_url(np.zeros((24, 32, 3), dtype=np.uint8)),
                "detail": "high",
            },
        }
        grounding_audit = {
            "enabled": True,
            "records": [
                {"position": 1, "candidate_ids": ["LQ001"], "packet_mode": "crop_only"},
                {"position": 4, "candidate_ids": ["LQ002"], "packet_mode": "crop_only"},
            ],
            "crop_count": 2,
            "full_fallback_count": 0,
        }
        timestamp_rows = iter([[10.0, 12.0, 14.0], [70.0, 72.5, 75.0]])
        with patch(
            "videoseek.tools.frame_verify.scene_aware_timestamps",
            side_effect=lambda **kwargs: next(timestamp_rows),
        ), patch(
            "videoseek.tools.frame_verify.prepare_grounded_frames",
            return_value=(
                {
                    1: [{"type": "text", "text": "crop"}, crop],
                    4: [{"type": "text", "text": "crop"}, crop],
                },
                grounding_audit,
            ),
        ), patch(
            "videoseek.tools.frame_verify.observe_content",
            side_effect=fake_observe,
        ):
            output = execute_frame_verify(
                {
                    "localize_inline_verify_max_windows": 3,
                    "localize_inline_verify_max_frames": 6,
                    "multiwindow_verify_recovery_enabled": False,
                    "grounded_candidate_binding_enabled": True,
                    "grounded_frame_verify_enabled": True,
                    "grounded_verify_packet_enabled": True,
                },
                _parameters(),
            )

        images = [item for item in observed_content if item.get("type") == "image_url"]
        self.assertEqual(len(images), 4)
        self.assertEqual(sum(item["image_url"].get("detail") == "low" for item in images), 2)
        payload = extract_v10_payload(output)
        self.assertEqual(payload["visual_layout"], "candidate_verify_packets")
        self.assertEqual(payload["visual_packet_audit"]["api_image_count"], 4)
        self.assertEqual(payload["visual_packet_audit"]["detail_crop_count"], 2)
        self.assertEqual(len(payload["timestamp_observations"]), 2)
        self.assertTrue(payload["candidate_coverage_complete"])

    def test_packet_without_grounding_keeps_two_full_anchors_per_candidate(self):
        observed_content = []

        def fake_observe(
            config, *, content, tool_name, tool_mode, output_dir, return_json=False
        ):
            observed_content.extend(content)
            return _response_without_duplicate_observations(), "api"

        timestamp_rows = iter([[10.0, 12.0, 14.0], [70.0, 72.5, 75.0]])
        parameters = _parameters()
        parameters["candidate_windows"][0]["timestamp_anchors"] = [10.0, 14.0]
        parameters["candidate_windows"][1]["timestamp_anchors"] = [70.0, 75.0]
        with patch(
            "videoseek.tools.frame_verify.scene_aware_timestamps",
            side_effect=lambda **kwargs: next(timestamp_rows),
        ), patch(
            "videoseek.tools.frame_verify.observe_content",
            side_effect=fake_observe,
        ):
            output = execute_frame_verify(
                {
                    "localize_inline_verify_max_windows": 3,
                    "localize_inline_verify_max_frames": 6,
                    "multiwindow_verify_recovery_enabled": False,
                    "grounded_candidate_binding_enabled": True,
                    "grounded_frame_verify_enabled": False,
                    "grounded_verify_packet_enabled": True,
                    "grounded_verify_packet_anchors_per_candidate": 2,
                },
                parameters,
            )

        images = [item for item in observed_content if item.get("type") == "image_url"]
        self.assertEqual(len(images), 4)
        payload = extract_v10_payload(output)
        packets = payload["visual_packet_audit"]["packets"]
        self.assertEqual([row["detail_tile_count"] for row in packets], [2, 2])
        self.assertEqual([row["detail_mode"] for row in packets], ["full_anchor", "full_anchor"])

    def test_assessments_materialize_timestamp_observations(self):
        candidates = _parameters()["candidate_windows"]
        assessments = json.loads(_response_without_duplicate_observations())[
            "candidate_assessments"
        ]
        payload = {}
        _bind_candidate_observations(
            payload,
            candidates=candidates,
            assessments=assessments,
            allowed_by_candidate={"LQ001": [10.0, 12.0, 14.0], "LQ002": [70.0, 72.5, 75.0]},
        )
        self.assertEqual(
            [row["candidate_id"] for row in payload["timestamp_observations"]],
            ["LQ001", "LQ002"],
        )
        self.assertEqual(
            [row["timestamp_s"] for row in payload["timestamp_observations"]],
            [12.0, 72.5],
        )


if __name__ == "__main__":
    unittest.main()
