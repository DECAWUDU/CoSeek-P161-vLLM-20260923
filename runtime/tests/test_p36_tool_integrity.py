from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from videoseek.core.memory import init_observation_memory, merge_tool_observation
from videoseek.core.tool_evidence_handoff import (
    collect_memory_anchor_timestamps,
    expand_windows_to_anchors,
)
from videoseek.tools.frame_verify import (
    _normalize_candidate_windows,
    _packet_detail_anchor_positions,
    execute_frame_verify,
)
from videoseek.tools.localize_qwen import _rank_coarse_candidates
from videoseek.tools.v10_format import format_v10_observation


class P36ToolEvidenceHandoffTests(unittest.TestCase):
    def test_near_boundary_overview_anchor_is_preserved_and_expands_search(self):
        memory = init_observation_memory()
        memory["timestamped_observations"] = [
            {
                "source_tool": "overview",
                "timestamp_s": 319.1,
                "description": "small group stretching",
                "needs_focus": "",
            },
            {
                "source_tool": "overview",
                "timestamp_s": 327.6,
                "description": "women perform a dance routine",
                "confidence": "routing_only",
                "needs_focus": "verify this candidate",
            },
            {
                "source_tool": "overview",
                "timestamp_s": 329.1,
                "description": "baboons in grass",
                "needs_focus": "",
            },
        ]
        windows = [[311.1, 327.1]]
        anchors = collect_memory_anchor_timestamps(
            memory,
            windows,
            margin_s=2.0,
            per_window=4,
        )
        self.assertIn(327.6, anchors[0])
        expanded = expand_windows_to_anchors(
            windows,
            anchors,
            duration=500.0,
            pad_s=0.25,
        )
        self.assertGreaterEqual(expanded[0][1], 327.85)

    def test_verified_and_qwen_exact_anchors_replace_no_frame_budget(self):
        memory = init_observation_memory()
        memory["timestamped_observations"] = [
            {
                "source_tool": "localize_qwen",
                "timestamp_s": 16.9,
                "description": "hands thread a bead",
                "confidence": "high",
            }
        ]
        memory["candidate_binding_memory"] = [
            {
                "id": "bind_00001",
                "source_tool": "frame_verify",
                "best_timestamp_s": 16.9,
                "target_match": "matched",
                "event_match": "direct",
                "observed_fact": "Hands actively assemble jewelry.",
            }
        ]
        anchors = collect_memory_anchor_timestamps(
            memory,
            [[11.9, 22.0]],
            per_window=2,
        )
        self.assertEqual(anchors, [[16.9]])

    def test_anchor_is_assigned_to_containing_window_not_neighbor_margin(self):
        memory = init_observation_memory()
        memory["timestamped_observations"] = [
            {
                "source_tool": "overview",
                "timestamp_s": 270.2,
                "description": "hands use pliers",
                "needs_focus": "verify",
            }
        ]
        anchors = collect_memory_anchor_timestamps(
            memory,
            [[256.3, 268.3], [269.8, 270.2]],
            margin_s=2.0,
            per_window=4,
        )
        self.assertEqual(anchors, [[], [270.2]])

    def test_ranker_can_retain_one_candidate_per_requested_source(self):
        payloads = []
        for index in range(7):
            start = float(index * 20)
            payloads.append(
                (
                    [start, start + 10.0],
                    {
                        "suggest_focus_windows": [[start + 2.0, start + 6.0]],
                        "overall_summary": f"candidate {index}",
                        "possible_evidence": True,
                        "relevance": 0.9 - index * 0.05,
                    },
                )
            )
        ranked = _rank_coarse_candidates(
            payloads,
            top_k=7,
            candidate_window_s=12.0,
            preserve_source_coverage=True,
        )
        self.assertEqual(len(ranked), 7)
        self.assertEqual({item["source_index"] for item in ranked}, set(range(7)))

    def test_direct_frame_verify_receives_prior_exact_timestamp(self):
        memory = init_observation_memory()
        memory["timestamped_observations"] = [
            {
                "source_tool": "localize_qwen",
                "timestamp_s": 16.9,
                "description": "hands thread a bead",
                "confidence": "high",
            }
        ]
        captured = {}

        def fake_focus(config, parameters):
            captured.update(parameters)
            return format_v10_observation(
                {
                    "t_range": [11.9, 22.0],
                    "target_match": "matched",
                    "target_event_match": "direct",
                    "observer_backend": "api",
                    "parse_ok": True,
                }
            )

        with patch("videoseek.tools.frame_verify.execute_focus", side_effect=fake_focus):
            execute_frame_verify(
                {
                    "coseek1_tool_integrity_repair_enabled": True,
                    "coseek1_tool_memory_anchors_per_window": 4,
                },
                {
                    "query": "Is jewelry actively being made?",
                    "start_time": 11.9,
                    "end_time": 22.0,
                    "mode": "option_verify",
                    "memory": memory,
                },
            )
        self.assertIn(16.9, captured["mandatory_timestamps"])

    def test_localized_window_survives_normalization_and_selects_event_peak(self):
        candidates = _normalize_candidate_windows(
            [
                {
                    "candidate_id": "LQ003",
                    "source_index": 4,
                    "t_range": [407.65, 424.4],
                    "localized_verify_window": [409.2, 420.4],
                    "source_search_window": [407.65, 425.5],
                    "localized_evidence_anchors": [
                        {"timestamp_s": 409.2, "confidence": "high"},
                        {"timestamp_s": 414.8, "confidence": "high"},
                    ],
                }
            ],
            duration=500.0,
            max_windows=8,
        )
        self.assertEqual(candidates[0]["localized_verify_window"], [409.2, 420.4])
        self.assertEqual(candidates[0]["source_index"], 4)
        rows = [
            (0, 407.7, np.zeros((8, 8, 3), dtype=np.uint8)),
            (1, 409.2, np.zeros((8, 8, 3), dtype=np.uint8)),
            (2, 414.8, np.ones((8, 8, 3), dtype=np.uint8) * 128),
            (3, 420.4, np.zeros((8, 8, 3), dtype=np.uint8)),
        ]
        positions, source = _packet_detail_anchor_positions(
            candidate=candidates[0],
            positioned_rows=rows,
            limit=2,
            query_context_roles_enabled=True,
        )
        self.assertEqual(source, "query_peak_plus_event_context")
        self.assertEqual(positions[0], 2)

    def test_count138_localized_window_selects_late_action_anchor(self):
        candidates = _normalize_candidate_windows(
            [
                {
                    "candidate_id": "LQ001",
                    "source_index": 0,
                    "t_range": [6.8, 24.8],
                    "localized_verify_window": [10.8, 22.0],
                    "source_search_window": [6.8, 24.8],
                    "localized_evidence_anchors": [
                        {"timestamp_s": 10.8, "confidence": "high"},
                        {"timestamp_s": 18.8, "confidence": "high"},
                    ],
                }
            ],
            duration=441.0,
            max_windows=8,
        )
        rows = [
            (0, 6.8, np.zeros((8, 8, 3), dtype=np.uint8)),
            (1, 10.8, np.zeros((8, 8, 3), dtype=np.uint8)),
            (2, 18.8, np.ones((8, 8, 3), dtype=np.uint8) * 128),
            (3, 24.8, np.zeros((8, 8, 3), dtype=np.uint8)),
        ]
        positions, source = _packet_detail_anchor_positions(
            candidate=candidates[0],
            positioned_rows=rows,
            limit=2,
            query_context_roles_enabled=True,
        )
        self.assertEqual(source, "query_peak_plus_event_context")
        self.assertEqual(positions[0], 2)


class P36MemoryIntegrityTests(unittest.TestCase):
    @staticmethod
    def _candidate_output(*, event_match: str, timestamp: float) -> str:
        return format_v10_observation(
            {
                "t_range": [timestamp - 4.0, timestamp + 4.0],
                "candidate_assessments": [
                    {
                        "candidate_id": "LQ001",
                        "event_group_id": "EG001",
                        "target_match": "matched",
                        "event_match": event_match,
                        "observed_fact": "The target action is directly visible.",
                        "best_timestamp_s": timestamp,
                    }
                ],
                "observer_backend": "api",
                "parse_ok": True,
            }
        )

    @staticmethod
    def _candidate_parameters(timestamp: float) -> dict:
        return {
            "candidate_windows": [
                {
                    "candidate_id": "LQ001",
                    "event_group_id": "EG001",
                    "t_range": [timestamp - 4.0, timestamp + 4.0],
                }
            ]
        }

    def test_event_groups_are_globally_namespaced_when_repair_is_enabled(self):
        memory = init_observation_memory()
        memory["runtime_config"] = {"coseek1_tool_integrity_repair_enabled": True}
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters=self._candidate_parameters(16.9),
            output=self._candidate_output(event_match="direct", timestamp=16.9),
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters=self._candidate_parameters(106.0),
            output=self._candidate_output(event_match="direct", timestamp=106.0),
        )
        groups = [item["event_group_id"] for item in memory["candidate_binding_memory"]]
        self.assertEqual(groups, ["verify_00001:EG001", "verify_00002:EG001"])
        self.assertNotEqual(groups[0], groups[1])

    def test_overlapping_direct_and_context_observations_are_audited(self):
        memory = init_observation_memory()
        memory["runtime_config"] = {"coseek1_tool_integrity_repair_enabled": True}
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters=self._candidate_parameters(16.9),
            output=self._candidate_output(event_match="direct", timestamp=16.9),
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 11.9, "end_time": 22.0},
            output=format_v10_observation(
                {
                    "t_range": [11.9, 22.0],
                    "target_match": "partial",
                    "target_event_match": "context_only",
                    "observed_fact": "Only a jewelry kit is visible.",
                    "observer_backend": "api",
                    "parse_ok": True,
                }
            ),
        )
        self.assertEqual(len(memory["observation_conflicts"]), 1)
        self.assertEqual(
            memory["observation_conflicts"][0]["conflict_type"],
            "overlapping_direct_vs_non_direct",
        )
        self.assertTrue(
            memory["tool_observations"][-1].get("observation_conflict_ids")
        )

    def test_disabled_repair_preserves_legacy_event_group(self):
        memory = init_observation_memory()
        memory["runtime_config"] = {"coseek1_tool_integrity_repair_enabled": False}
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters=self._candidate_parameters(16.9),
            output=self._candidate_output(event_match="direct", timestamp=16.9),
        )
        self.assertEqual(
            memory["candidate_binding_memory"][0]["event_group_id"], "EG001"
        )
        self.assertEqual(memory["verification_facts"], [])


if __name__ == "__main__":
    unittest.main()
