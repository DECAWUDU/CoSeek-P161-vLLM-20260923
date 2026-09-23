from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from videoseek.tools.spatial_grounding import (
    build_answer_blind_grounding_prompt,
    parse_grounding_box,
    prepare_grounded_frames,
    select_grounding_positions,
)
from videoseek.tools.frame_verify import _normalize_candidate_windows


class SpatialGroundingP1Test(unittest.TestCase):
    def test_prompt_marks_answer_values_unknown(self):
        prompt = build_answer_blind_grounding_prompt(
            query="Check whether the bottle is red or green.",
            question="What color is the bottle on the table?\n(A) Red\n(B) Green",
        )
        self.assertIn("unknown candidate values", prompt)
        self.assertIn("What color is the bottle on the table?", prompt)
        self.assertNotIn("(A) Red", prompt)

    def test_parser_clamps_coordinates_and_discards_absent_box(self):
        parsed = parse_grounding_box("VISIBLE yes\nBOX -5 10 1010 900")
        self.assertTrue(parsed["box_valid"])
        self.assertEqual(parsed["box"], [0.0, 10.0, 1000.0, 900.0])
        self.assertTrue(parsed["coordinates_clamped"])
        absent = parse_grounding_box("VISIBLE no\nBOX NONE")
        self.assertTrue(absent["parse_ok"])
        self.assertFalse(absent["box_valid"])

    def test_parser_accepts_prefixless_four_line_response(self):
        parsed = parse_grounding_box(
            "yes\n182 0 678 521\nwoman arranging flowers\n85"
        )
        self.assertTrue(parsed["parse_ok"])
        self.assertTrue(parsed["box_valid"])
        self.assertEqual(parsed["box"], [182.0, 0.0, 678.0, 521.0])

    def test_anchor_selection_prefers_explicit_candidates(self):
        positions = select_grounding_positions(
            frame_indices=[10, 20, 30, 40, 50],
            mandatory_anchor_indices={20},
            candidate_anchor_indices={40: ["C2"]},
            max_anchors=2,
        )
        self.assertEqual(positions, [1, 3])

    def test_candidate_balanced_selection_keeps_one_anchor_per_candidate(self):
        positions = select_grounding_positions(
            frame_indices=[10, 20, 30, 40, 50, 60],
            mandatory_anchor_indices=set(),
            candidate_anchor_indices={
                10: ["LQ001"],
                20: ["LQ001"],
                30: ["LQ002"],
                40: ["LQ002"],
                50: ["LQ003"],
                60: ["LQ003"],
            },
            max_anchors=3,
            candidate_balanced=True,
        )
        self.assertEqual(positions, [1, 3, 5])

    def test_candidate_normalization_keeps_in_window_timestamp_anchors(self):
        candidates = _normalize_candidate_windows(
            [
                {
                    "candidate_id": "LQ001",
                    "recommended_verify_window": [310.5, 312.5],
                    "timestamp_anchors": [309.7, 310.9, 312.1, 313.3, "bad"],
                }
            ],
            duration=740.0,
            max_windows=3,
        )
        self.assertEqual(candidates[0]["timestamp_anchors"], [310.9, 312.1])

    def _config(self, root: Path) -> dict:
        return {
            "grounded_frame_verify_max_anchors": 1,
            "grounded_frame_verify_crop_area_threshold": 0.5,
            "grounded_frame_verify_padding_ratio": 0.18,
            "grounded_frame_verify_min_crop_side_ratio": 0.12,
            "grounded_frame_verify_context_short_side": 64,
            "grounded_frame_verify_max_new_tokens": 160,
            "grounded_frame_verify_cache_enabled": False,
            "grounded_frame_verify_cache_dir": str(root),
        }

    def test_small_box_builds_context_crop_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = np.zeros((120, 200, 3), dtype=np.uint8)
            with patch(
                "videoseek.tools.spatial_grounding.call_local_qwen_parts",
                return_value="VISIBLE yes\nBOX 400 300 600 700",
            ):
                replacements, audit = prepare_grounded_frames(
                    self._config(Path(tmp)),
                    frames=np.asarray([frame]),
                    frame_indices=[10],
                    timestamps=[1.0],
                    mandatory_anchor_indices=set(),
                    candidate_anchor_indices={},
                    query="Inspect the bottle color.",
                    question="What color is the bottle?",
                    output_dir=tmp,
                )
            self.assertIn(0, replacements)
            self.assertEqual(audit["context_crop_count"], 1)
            self.assertEqual(audit["records"][0]["packet_mode"], "context_crop")
            self.assertEqual(
                sum(item["type"] == "image_url" for item in replacements[0]),
                2,
            )

    def test_large_or_missing_box_keeps_full_frame(self):
        for raw, reason in (
            ("VISIBLE yes\nBOX 10 10 990 990", "crop_too_large"),
            ("VISIBLE no\nBOX NONE", "target_not_grounded"),
        ):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as tmp:
                frame = np.zeros((120, 200, 3), dtype=np.uint8)
                with patch(
                    "videoseek.tools.spatial_grounding.call_local_qwen_parts",
                    return_value=raw,
                ):
                    replacements, audit = prepare_grounded_frames(
                        self._config(Path(tmp)),
                        frames=np.asarray([frame]),
                        frame_indices=[10],
                        timestamps=[1.0],
                        mandatory_anchor_indices=set(),
                        candidate_anchor_indices={},
                        query="Inspect the target.",
                        question="What is visible?",
                        output_dir=tmp,
                    )
                self.assertEqual(replacements, {})
                self.assertEqual(audit["records"][0]["fallback_reason"], reason)


if __name__ == "__main__":
    unittest.main()
