import unittest
from unittest.mock import patch

from videoseek.core.memory import build_temporal_evidence_ledger, extract_v10_payload
from videoseek.tools.focus_qwen import _parse_caption_only_rows as parse_focus_rows
from videoseek.tools.frame_verify import execute_multiwindow_frame_verify
from videoseek.tools.localize_qwen import (
    _candidate_evidence,
    _rank_coarse_candidates,
)
from videoseek.tools.skim_qwen import _parse_caption_only_rows as parse_skim_rows
from videoseek.tools.v10_format import format_v10_observation


def row(timestamp, relevance, description, target="unknown", event="unknown"):
    return {
        "timestamp_s": float(timestamp),
        "confidence": relevance,
        "description": description,
        "target_match": target,
        "event_match": event,
    }


def payload(rows, suggestions):
    return {
        "timestamp_observations": rows,
        "suggest_focus_windows": suggestions,
        "relevance": 0.7,
        "overall_summary": "frozen replay",
    }


class P124EvidencePreservingSelectionTest(unittest.TestCase):
    def rank(self, coarse_payloads, top_k=2):
        return _rank_coarse_candidates(
            coarse_payloads,
            top_k=top_k,
            candidate_window_s=18.0,
            row_candidates_enabled=False,
            normalized_score_enabled=True,
            evidence_preserving=True,
            ambiguous_reserve=2,
        )

    @staticmethod
    def positive_timestamps(candidates):
        return {
            float(anchor["timestamp_s"])
            for candidate in candidates
            if candidate.get("mandatory_positive")
            for anchor in candidate.get("coarse_positive_anchors") or []
        }

    def test_r08_positive_set_is_partition_invariant(self):
        rows = [
            row(68.0, "high", "person rides bicycle"),
            row(346.0, "high", "person holds red paraglider against sky"),
            row(384.5, "medium", "distant object over water"),
            row(392.2, "medium", "distant object over water"),
        ]
        suggestions = [[60.2, 78.2], [383.2, 401.2], [337.0, 355.0]]
        single = self.rank([([0.0, 646.0], payload(rows, suggestions))])
        partitioned = self.rank(
            [
                ([51.3, 111.1], payload(rows[:1], [suggestions[0]])),
                ([323.0, 461.4], payload(rows[1:], suggestions[1:])),
            ]
        )
        self.assertIn(346.0, self.positive_timestamps(single))
        self.assertEqual(
            self.positive_timestamps(single),
            self.positive_timestamps(partitioned),
        )

    def test_all_explicit_positives_survive_above_top_k(self):
        rows = [
            row(
                10.0 + 20.0 * index,
                "high",
                f"direct target event {index}",
                target="matched",
                event="direct",
            )
            for index in range(6)
        ]
        candidates = self.rank([([0.0, 130.0], payload(rows, []))], top_k=1)
        positives = [item for item in candidates if item.get("mandatory_positive")]
        self.assertEqual(len(positives), 6)

    def test_high_relevance_explicit_mismatch_is_not_mandatory(self):
        evidence = _candidate_evidence(
            payload(
                [
                    row(
                        68.0,
                        "high",
                        "person rides bicycle",
                        target="not_matched",
                        event="not_matched",
                    )
                ],
                [],
            ),
            [60.0, 78.0],
        )
        self.assertEqual(evidence["evidence_class"], "ambiguous")
        self.assertEqual(evidence["positive_anchors"], [])

    def test_overlapping_suggestions_with_same_anchor_form_one_event_candidate(self):
        rows = [
            row(107.7, "high", "sausages cooking", "matched", "direct"),
            row(108.8, "high", "sausages cooking", "matched", "direct"),
            row(116.7, "high", "sausages cooking", "matched", "direct"),
        ]
        candidates = self.rank(
            [
                (
                    [101.1, 124.5],
                    payload(
                        rows,
                        [[101.1, 108.8], [105.7, 118.7], [116.7, 124.5]],
                    ),
                )
            ]
        )
        positives = [item for item in candidates if item.get("mandatory_positive")]
        self.assertEqual(len(positives), 1)
        self.assertEqual(
            {anchor["timestamp_s"] for anchor in positives[0]["coarse_positive_anchors"]},
            {107.7, 108.8, 116.7},
        )

    def test_local_caption_protocol_keeps_target_and_event_labels(self):
        raw = (
            "F01: red paraglider in sky | relevance=high | "
            "target_match=matched | event_match=direct"
        )
        for parser in (parse_skim_rows, parse_focus_rows):
            rows = parser(
                raw,
                allowed_timestamps=[346.0],
                scene_id="S1",
            )
            self.assertEqual(rows[0]["target_match"], "matched")
            self.assertEqual(rows[0]["event_match"], "direct")

    def test_multiwindow_call_range_cannot_become_stable_event(self):
        memory = {
            "tool_observations": [
                {
                    "tool": "frame_verify",
                    "verification_id": "verify_00001",
                    "parameters": {
                        "candidate_windows": [
                            {"candidate_id": "LQ001", "t_range": [65.5, 88.0]},
                            {"candidate_id": "LQ002", "t_range": [101.1, 124.5]},
                        ]
                    },
                    "t_range": [65.5, 124.5],
                    "target_match": "matched",
                    "target_event_match": "direct",
                    "supports_options": ["C"],
                    "candidate_assessments": [],
                }
            ],
            "candidate_binding_memory": [],
        }
        ledger = build_temporal_evidence_ledger(memory)
        self.assertEqual(ledger["stable_event_span_count"], 0)

    def test_mandatory_candidates_are_verified_in_bounded_batches(self):
        candidates = [
            {
                "candidate_id": f"LQ{index + 1:03d}",
                "t_range": [20.0 * index, 20.0 * index + 8.0],
                "summary": "direct target lead",
            }
            for index in range(5)
        ]

        def fake_verify_once(_config, parameters):
            batch = parameters["candidate_windows"]
            return format_v10_observation(
                {
                    "tool": "frame_verify",
                    "window_id": "inline_multiwindow_verify",
                    "scene_id": "multi_scene",
                    "t_range": [batch[0]["t_range"][0], batch[-1]["t_range"][1]],
                    "requested_candidate_ids": [row["candidate_id"] for row in batch],
                    "verified_candidate_ids": [row["candidate_id"] for row in batch],
                    "candidate_coverage_complete": True,
                    "candidate_binding_complete": True,
                    "candidate_assessments": [
                        {
                            "candidate_id": row["candidate_id"],
                            "target_match": "matched",
                            "event_match": "direct",
                            "best_timestamp_s": row["t_range"][0] + 1.0,
                            "event_span": [
                                row["t_range"][0] + 0.5,
                                row["t_range"][0] + 1.5,
                            ],
                            "observed_fact": "direct event",
                            "supports_options": [],
                            "contradicts_options": [],
                        }
                        for row in batch
                    ],
                    "timestamp_observations": [
                        {
                            "candidate_id": row["candidate_id"],
                            "timestamp_s": row["t_range"][0] + 1.0,
                            "description": "direct event",
                        }
                        for row in batch
                    ],
                    "scope_coverage": "partial",
                    "observer_backend": "api",
                    "parse_ok": True,
                }
            )

        config = {
            "localize_inline_verify_max_windows": 16,
            "localize_inline_verify_batch_enabled": True,
            "localize_inline_verify_batch_size": 4,
            "grounded_candidate_binding_enabled": True,
            "grounded_candidate_event_binding_enabled": True,
            "grounded_candidate_event_binding_force_enabled": True,
            "grounded_candidate_event_group_max_gap_s": 24.0,
            "coseek1_tool_event_identity_repair_enabled": True,
            "multiwindow_verify_recovery_enabled": False,
        }
        with patch(
            "videoseek.tools.frame_verify._execute_multiwindow_frame_verify_once",
            side_effect=fake_verify_once,
        ) as mocked:
            output = execute_multiwindow_frame_verify(
                config,
                {
                    "vr": object(),
                    "duration": 120.0,
                    "candidate_windows": candidates,
                },
            )
        result = extract_v10_payload(output)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(len(result["candidate_assessments"]), 5)
        self.assertTrue(result["candidate_coverage_complete"])


if __name__ == "__main__":
    unittest.main()
