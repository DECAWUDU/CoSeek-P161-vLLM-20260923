from __future__ import annotations

import unittest

import numpy as np

from videoseek.agent import VideoSeekAgent
from videoseek.core import Action
from videoseek.core.memory import init_observation_memory
from videoseek.core.tool_evidence_handoff import collect_memory_anchor_timestamps
from videoseek.tools.frame_verify import (
    _assign_candidate_context_groups,
    _normalize_candidate_windows,
    _packet_detail_anchor_positions,
)
from videoseek.tools.localize_qwen import _expand_verify_window_to_nearby_anchors
from videoseek.tools.v10_format import format_v10_observation


class P37ToolEventIntegrityTests(unittest.TestCase):
    def test_nearby_memory_anchor_extends_actual_verify_window(self):
        window = _expand_verify_window_to_nearby_anchors(
            [364.9, 382.9],
            source_window=[364.9, 384.6],
            anchors=[372.9, 384.6],
            max_gap_s=2.0,
            pad_s=0.25,
        )
        self.assertEqual(window, [364.9, 384.6])

    def test_distant_memory_anchor_does_not_expand_verify_window(self):
        window = _expand_verify_window_to_nearby_anchors(
            [73.6, 79.6],
            source_window=[73.6, 101.6],
            anchors=[74.5, 83.0, 87.6, 97.0],
            max_gap_s=2.0,
            pad_s=0.25,
        )
        self.assertEqual(window, [73.6, 79.6])

    def test_inline_verify_preserves_recollected_memory_anchors(self):
        agent = VideoSeekAgent.__new__(VideoSeekAgent)
        agent.question = "How many separate times does the dance occur?"
        agent.config = {
            "coseek1_localize_inline_verify": True,
            "localize_inline_verify_max_windows": 3,
            "coseek1_event_coverage_inline_verify": False,
        }
        localize_output = format_v10_observation(
            {
                "parse_ok": True,
                "ranked_candidates": [
                    {
                        "candidate_id": "LQ003",
                        "rank": 1,
                        "recommended_verify_window": [317.1, 328.9],
                        "localization_status": "found",
                        "timestamp_anchors": [319.1, 324.8],
                        "localized_timestamp_anchors": [319.1, 324.8],
                        "memory_timestamp_anchors": [312.8, 319.1, 329.1],
                    }
                ],
            }
        )
        action = agent._VideoSeekAgent__inline_localize_verify_action(
            localize_action=Action(
                "localize_qwen",
                {
                    "localization_goal": "Find distinct dance occurrences.",
                    "evidence_profile": "event_boundary",
                },
                "localize-1",
            ),
            localize_output=localize_output,
            step=1,
        )
        self.assertIsNotNone(action)
        self.assertEqual(
            action.parameters["candidate_windows"][0]["memory_timestamp_anchors"],
            [312.8, 319.1, 329.1],
        )

    def test_expanded_window_recollects_overview_boundary_anchor(self):
        memory = init_observation_memory()
        memory["timestamped_observations"] = [
            {
                "source_tool": "overview",
                "timestamp_s": 319.1,
                "description": "three women begin a synchronized routine",
                "needs_focus": "",
            },
            {
                "source_tool": "overview",
                "timestamp_s": 327.6,
                "description": "hard cut to a larger dance group",
                "needs_focus": "verify the transition",
            },
        ]
        narrow = collect_memory_anchor_timestamps(
            memory,
            [[318.8, 321.2]],
            margin_s=2.0,
            per_window=4,
        )
        expanded = collect_memory_anchor_timestamps(
            memory,
            [[310.8, 329.2]],
            margin_s=0.0,
            per_window=4,
        )
        self.assertNotIn(327.6, narrow[0])
        self.assertIn(327.6, expanded[0])

    def test_packet_uses_query_peak_and_recollected_post_boundary(self):
        candidates = _normalize_candidate_windows(
            [
                {
                    "candidate_id": "LQ004",
                    "source_index": 3,
                    "t_range": [314.0, 329.2],
                    "localized_verify_window": [318.0, 327.6],
                    "source_search_window": [310.8, 329.2],
                    "localized_evidence_anchors": [
                        {"timestamp_s": 319.1, "confidence": "high"},
                        {"timestamp_s": 322.6, "confidence": "high"},
                    ],
                    "memory_timestamp_anchors": [319.1, 327.6],
                }
            ],
            duration=500.0,
            max_windows=8,
        )
        rows = [
            (0, 314.0, np.zeros((8, 8, 3), dtype=np.uint8)),
            (1, 319.1, np.ones((8, 8, 3), dtype=np.uint8) * 32),
            (2, 322.6, np.ones((8, 8, 3), dtype=np.uint8) * 64),
            (3, 324.8, np.ones((8, 8, 3), dtype=np.uint8) * 96),
            (4, 327.6, np.ones((8, 8, 3), dtype=np.uint8) * 192),
            (5, 329.2, np.ones((8, 8, 3), dtype=np.uint8) * 255),
        ]
        positions, source = _packet_detail_anchor_positions(
            candidate=candidates[0],
            positioned_rows=rows,
            limit=2,
            query_context_roles_enabled=True,
            boundary_context_enabled=True,
        )
        self.assertEqual(source, "query_peak_plus_boundary_context")
        self.assertEqual(positions, [2, 4])

    def test_disjoint_packet_groups_do_not_preassign_event_identity(self):
        candidates = [
            {"candidate_id": "LQ005", "t_range": [42.7, 50.2]},
            {"candidate_id": "LQ002", "t_range": [62.8, 82.2]},
        ]
        groups = _assign_candidate_context_groups(
            candidates,
            max_gap_s=24.0,
            decouple_event_identity=True,
        )
        self.assertEqual(len(set(groups.values())), 2)
        self.assertEqual(candidates[0]["packet_group_id"], "PG001")
        self.assertEqual(candidates[1]["packet_group_id"], "PG002")
        self.assertNotIn("event_group_id", candidates[0])
        self.assertNotIn("event_group_id", candidates[1])

    def test_legacy_grouping_remains_available_when_repair_is_disabled(self):
        candidates = [
            {"candidate_id": "LQ005", "t_range": [42.7, 50.2]},
            {"candidate_id": "LQ002", "t_range": [62.8, 82.2]},
        ]
        groups = _assign_candidate_context_groups(
            candidates,
            max_gap_s=24.0,
            decouple_event_identity=False,
        )
        self.assertEqual(len(set(groups.values())), 1)
        self.assertEqual(candidates[0]["event_group_id"], "EG001")
        self.assertEqual(candidates[1]["event_group_id"], "EG001")


if __name__ == "__main__":
    unittest.main()
