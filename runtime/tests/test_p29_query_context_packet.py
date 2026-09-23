import unittest

import numpy as np

from videoseek.tools.frame_verify import _packet_detail_anchor_positions


class P29QueryContextPacketTests(unittest.TestCase):
    def test_role_split_keeps_query_peak_and_visual_boundary(self):
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        gray = np.full((32, 32, 3), 32, dtype=np.uint8)
        white = np.full((32, 32, 3), 255, dtype=np.uint8)
        rows = [
            (0, 0.0, white),
            (1, 4.0, black),
            (2, 6.0, black),
            (3, 10.0, gray),
        ]
        candidate = {
            "localized_verify_window": [4.0, 8.0],
            "localized_evidence_anchors": [
                {"timestamp_s": 4.0, "confidence": "high"},
                {"timestamp_s": 6.0, "confidence": "high"},
            ],
        }
        positions, source = _packet_detail_anchor_positions(
            candidate=candidate,
            positioned_rows=rows,
            limit=2,
            query_context_roles_enabled=True,
        )
        self.assertEqual(positions, [2, 0])
        self.assertEqual(source, "query_peak_plus_event_context")

    def test_disabled_role_split_preserves_temporal_spread(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        rows = [(0, 0.0, frame), (1, 4.0, frame), (2, 6.0, frame), (3, 10.0, frame)]
        candidate = {
            "localized_verify_window": [4.0, 8.0],
            "localized_evidence_anchors": [
                {"timestamp_s": 4.0, "confidence": "high"},
                {"timestamp_s": 6.0, "confidence": "high"},
            ],
        }
        positions, source = _packet_detail_anchor_positions(
            candidate=candidate,
            positioned_rows=rows,
            limit=2,
        )
        self.assertEqual(positions, [1, 2])
        self.assertEqual(source, "query_confidence_then_context")


if __name__ == "__main__":
    unittest.main()
