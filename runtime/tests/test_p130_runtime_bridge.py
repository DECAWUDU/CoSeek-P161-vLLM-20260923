from __future__ import annotations

import unittest
import json

from videoseek.core.p130_runtime import (
    append_p130_observation,
    init_p130_state,
    initial_search_windows,
    next_recovery_spec,
)


def format_v10_observation(payload: dict) -> str:
    return "V10_OBSERVATION_JSON:\n```json\n" + json.dumps(payload) + "\n```"


R09 = """Throughout this video, what is the total count of occurrences for the scene featuring the 'cooking sausages' action
(A) 1
(B) 0
(C) 4
(D) 3
Please directly answer with the best option's letter from the given choices directly (A, B, C, or D)."""

R10 = """Arrange the following events from the video in the correct chronological order: (1)a young and a kid are doing balance in a balance rope; (2)people are walking in a bridge to see a competition of men doing tricks on top of a balance rope; (3)a man is jumping and doing tricks in a balance rope above a cold river; (4)the boy is in a competition in snowy path doing tricks on a balance rope with people behind a fence watching him.
(A) 2->1->3->4
(B) 1->2->3->4
(C) 4->3->2->1
(D) 3->2->1->4
Please directly answer with the best option's letter from the given choices directly (A, B, C, or D)."""


class P130RuntimeBridgeTests(unittest.TestCase):
    def test_full_range_localize_is_routing_only(self) -> None:
        memory = {}
        init_p130_state(memory, question=R09, duration=490.0)
        output = format_v10_observation(
            {
                "tool": "localize_qwen",
                "t_range": [0.0, 490.0],
                "parse_ok": True,
                "ranked_candidates": [
                    {
                        "candidate_id": "LQ001",
                        "recommended_verify_window": [70.0, 82.0],
                        "fine_summary": "sausages cooking",
                        "localization_status": "found",
                    }
                ],
                "timestamp_observations": [],
            }
        )

        snapshot = append_p130_observation(
            memory,
            tool_name="localize_qwen",
            parameters={"search_windows": [[0.0, 490.0]]},
            output=output,
            action_id="localize-1",
            question=R09,
            duration=490.0,
        )

        self.assertEqual(snapshot["observed_count_lower_bound"], 0)
        self.assertEqual(snapshot["coverage_status"], "open")
        self.assertFalse(snapshot["enumeration_complete"])
        self.assertFalse(snapshot["decision_sufficient"])
        self.assertEqual(len(memory["p130_global"]["routing_candidates"]), 1)

    def test_count_receipts_ignore_local_option_claims_and_block_three(self) -> None:
        memory = {}
        init_p130_state(memory, question=R09, duration=490.0)
        output = format_v10_observation(
            {
                "tool": "frame_verify",
                "parse_ok": True,
                "t_range": [70.0, 330.0],
                "supports_options": ["D"],
                "candidate_assessments": [
                    {
                        "candidate_id": f"LQ00{index}",
                        "packet_group_id": f"PG00{index}",
                        "target_match": "matched",
                        "event_match": "direct",
                        "event_span": [start, start + 8.0],
                        "best_timestamp_s": start + 2.0,
                        "observed_fact": "sausages cooking",
                        "supports_options": ["D"],
                    }
                    for index, start in enumerate((72.0, 108.0, 311.0), start=1)
                ],
            }
        )

        snapshot = append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 70.0, "end_time": 330.0},
            output=output,
            action_id="verify-1",
            question=R09,
            duration=490.0,
        )

        self.assertEqual(snapshot["observed_count_lower_bound"], 3)
        self.assertEqual(snapshot["viable_options"], ["C", "D"])
        self.assertFalse(snapshot["decision_sufficient"])
        for receipt in memory["p130_global"]["evidence_receipts"]:
            self.assertEqual(receipt["supports_options"], [])
            self.assertEqual(receipt["contradicts_options"], [])

    def test_frame_evidence_requires_explicit_true_parse_ok(self) -> None:
        for label, parse_ok in (("missing", ...), ("false", False), ("null", None)):
            with self.subTest(parse_ok=label):
                memory = {}
                init_p130_state(memory, question=R09, duration=490.0)
                payload = {
                    "tool": "frame_verify",
                    "candidate_assessments": [
                        {
                            "candidate_id": "LQ001",
                            "packet_group_id": "PG001",
                            "target_match": "matched",
                            "event_match": "direct",
                            "event_span": [72.0, 80.0],
                            "observed_fact": "sausages cooking",
                        }
                    ],
                }
                if parse_ok is not ...:
                    payload["parse_ok"] = parse_ok

                snapshot = append_p130_observation(
                    memory,
                    tool_name="frame_verify",
                    parameters={"start_time": 70.0, "end_time": 82.0},
                    output=format_v10_observation(payload),
                    action_id=f"verify-{label}",
                    question=R09,
                    duration=490.0,
                )

                self.assertFalse(memory["p130_global"]["tool_receipts"][-1]["parse_ok"])
                self.assertEqual(memory["p130_global"]["evidence_receipts"], [])
                self.assertEqual(snapshot["observed_count_lower_bound"], 0)

    def test_aggregate_payload_without_candidate_assessments_is_not_local_evidence(self) -> None:
        memory = {}
        init_p130_state(memory, question=R09, duration=490.0)
        output = format_v10_observation(
            {
                "tool": "frame_verify",
                "parse_ok": True,
                "target_match": "matched",
                "event_match": "direct",
                "t_range": [70.0, 82.0],
                "best_timestamp_s": 76.0,
                "observed_fact": "aggregate summary says sausages cooking",
            }
        )

        snapshot = append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 70.0, "end_time": 82.0},
            output=output,
            action_id="verify-aggregate",
            question=R09,
            duration=490.0,
        )

        self.assertEqual(memory["p130_global"]["evidence_receipts"], [])
        self.assertEqual(snapshot["observed_count_lower_bound"], 0)

    def test_explicit_non_direct_event_match_overrides_matched_target(self) -> None:
        memory = {}
        init_p130_state(memory, question=R09, duration=490.0)
        labels = ("ambiguous", "partial", "context", "negative")
        output = format_v10_observation(
            {
                "tool": "frame_verify",
                "parse_ok": True,
                "candidate_assessments": [
                    {
                        "candidate_id": f"LQ00{index}",
                        "packet_group_id": f"PG00{index}",
                        "target_match": "matched",
                        "event_match": label,
                        "event_span": [start, start + 5.0],
                        "best_timestamp_s": start + 1.0,
                        "observed_fact": f"assessment is {label}",
                    }
                    for index, (label, start) in enumerate(
                        zip(labels, (70.0, 100.0, 130.0, 160.0)), start=1
                    )
                ],
            }
        )

        snapshot = append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 65.0, "end_time": 170.0},
            output=output,
            action_id="verify-precedence",
            question=R09,
            duration=490.0,
        )

        self.assertEqual(
            [row["match"] for row in memory["p130_global"]["evidence_receipts"]],
            ["ambiguous", "ambiguous", "context", "negative"],
        )
        self.assertEqual(snapshot["observed_count_lower_bound"], 0)

    def test_direct_count_requires_local_span_and_fact_or_anchor(self) -> None:
        memory = {}
        init_p130_state(memory, question=R09, duration=490.0)
        output = format_v10_observation(
            {
                "tool": "frame_verify",
                "parse_ok": True,
                # This aggregate request range must never stand in for a local
                # event span on the first candidate.
                "t_range": [0.0, 490.0],
                "candidate_assessments": [
                    {
                        "candidate_id": "LQ001",
                        "packet_group_id": "PG001",
                        "event_match": "direct",
                        "target_match": "matched",
                        "observed_fact": "fact without local span",
                    },
                    {
                        "candidate_id": "LQ002",
                        "packet_group_id": "PG002",
                        "event_match": "direct",
                        "target_match": "matched",
                        "event_span": [100.0, 108.0],
                    },
                    {
                        "candidate_id": "LQ003",
                        "packet_group_id": "PG003",
                        "event_match": "direct",
                        "target_match": "matched",
                        "event_span": [130.0, 138.0],
                        "observed_fact": "sausages cooking",
                    },
                    {
                        "candidate_id": "LQ004",
                        "packet_group_id": "PG004",
                        "event_match": "direct",
                        "target_match": "matched",
                        "event_span": [160.0, 168.0],
                        "anchor_timestamps": [164.0],
                    },
                ],
            }
        )

        snapshot = append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 0.0, "end_time": 490.0},
            output=output,
            action_id="verify-local-proof",
            question=R09,
            duration=490.0,
        )

        receipts = memory["p130_global"]["evidence_receipts"]
        self.assertEqual([row["packet_group_id"] for row in receipts], ["PG003", "PG004"])
        self.assertEqual([row["event_span"] for row in receipts], [[130.0, 138.0], [160.0, 168.0]])
        self.assertEqual(snapshot["observed_count_lower_bound"], 2)

    def test_count_packet_group_is_audit_only_and_temporal_dedup_is_semantic(self) -> None:
        memory = {}
        init_p130_state(memory, question=R09, duration=490.0)
        output = format_v10_observation(
            {
                "tool": "frame_verify",
                "parse_ok": True,
                "candidate_assessments": [
                    {
                        "candidate_id": "LQ001",
                        "packet_group_id": "PG001",
                        "event_match": "direct",
                        "target_match": "matched",
                        "event_span": [70.0, 78.0],
                        "observed_fact": "left side of one sausage scene",
                    },
                    {
                        "candidate_id": "LQ002",
                        "packet_group_id": "PG001",
                        "event_match": "direct",
                        "target_match": "matched",
                        "event_span": [74.0, 82.0],
                        "observed_fact": "right side of the same sausage scene",
                    },
                    {
                        "candidate_id": "LQ003",
                        "packet_group_id": "PG002",
                        "event_match": "direct",
                        "target_match": "matched",
                        "event_span": [108.0, 116.0],
                        "observed_fact": "a separate sausage scene",
                    },
                ],
            }
        )

        snapshot = append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 65.0, "end_time": 120.0},
            output=output,
            action_id="verify-packets",
            question=R09,
            duration=490.0,
        )

        receipts = memory["p130_global"]["evidence_receipts"]
        self.assertEqual(len(receipts), 3)
        first_packet = [row for row in receipts if row["packet_group_id"] == "PG001"]
        self.assertEqual(
            [row["source_candidate_ids"] for row in first_packet],
            [["LQ001"], ["LQ002"]],
        )
        self.assertTrue(all("event_key" not in row for row in receipts))
        self.assertEqual(snapshot["observed_count_lower_bound"], 2)

    def test_order_packet_rows_with_same_typed_id_become_one_event(self) -> None:
        memory = {}
        init_p130_state(memory, question=R10, duration=695.0)
        rows = [
            ("LQ001", "PG001", "1", 26.0),
            ("LQ002", "PG001", "1", 32.0),
            ("LQ003", "PG002", "2", 88.0),
            ("LQ004", "PG003", "3", 110.0),
            ("LQ005", "PG004", "4", 160.0),
        ]
        output = format_v10_observation(
            {
                "tool": "frame_verify",
                "parse_ok": True,
                "candidate_assessments": [
                    {
                        "candidate_id": candidate,
                        "packet_group_id": packet,
                        "target_match": "matched",
                        "event_match": "direct",
                        "target_event_id": target,
                        "event_index_candidates": [int(target)],
                        "event_span": [start, start + 5.0],
                        "best_timestamp_s": start + 1.0,
                        "observed_fact": f"event {target}",
                    }
                    for candidate, packet, target, start in rows
                ],
            }
        )

        snapshot = append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 20.0, "end_time": 180.0},
            output=output,
            action_id="verify-order",
            question=R10,
            duration=695.0,
        )

        self.assertEqual(snapshot["bound_event_ids"], ["1", "2", "3", "4"])
        self.assertEqual(snapshot["validated_option"], "B")
        self.assertTrue(snapshot["decision_sufficient"])
        self.assertEqual(len(snapshot["event_table"]), 4)

    def test_ambiguous_count_candidate_remains_first_recovery_target(self) -> None:
        memory = {}
        init_p130_state(memory, question=R09, duration=490.0)
        localize = format_v10_observation(
            {
                "tool": "localize_qwen",
                "parse_ok": True,
                "ranked_candidates": [
                    {
                        "candidate_id": "LQ004",
                        "recommended_verify_window": [416.0, 436.0],
                        "fine_summary": "bright flame obscures plate contents",
                        "selection_class": "ambiguous",
                        "localization_status": "ambiguous",
                    }
                ],
            }
        )
        append_p130_observation(
            memory,
            tool_name="localize_qwen",
            parameters={"search_windows": [[400.0, 450.0]]},
            output=localize,
            action_id="localize-1",
            question=R09,
            duration=490.0,
        )
        verify = format_v10_observation(
            {
                "tool": "frame_verify",
                "parse_ok": True,
                "candidate_assessments": [
                    {
                        "candidate_id": "LQ004",
                        "target_match": "not_visible",
                        "event_match": "different_event",
                        "event_span": [416.0, 436.0],
                        "observed_fact": "flame, target hidden",
                    }
                ],
            }
        )
        append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={"source_localize_action_id": "localize-1"},
            output=verify,
            action_id="verify-1",
            question=R09,
            duration=490.0,
        )

        spec = next_recovery_spec(
            memory,
            question=R09,
            overview_output="",
            duration=490.0,
            attempt_index=0,
        )

        self.assertEqual(spec["kind"], "frame_verify")
        self.assertEqual(spec["reason"], "residual_count_cursor")
        self.assertTrue(all(row["summary"] == "" for row in spec["candidate_windows"]))

    def test_overview_relevance_selects_order_domains(self) -> None:
        overview = format_v10_observation(
            {
                "scene_summaries": [
                    {
                        "t_range": [30.0, 210.0],
                        "summary": "snowy slackline competition beside a bridge and river",
                        "possible_evidence": True,
                    },
                    {
                        "t_range": [230.0, 690.0],
                        "summary": "bird wildlife documentary",
                        "possible_evidence": False,
                    },
                ],
                "timestamp_observations": [],
            }
        )

        windows = initial_search_windows(R10, overview, duration=695.0)

        self.assertEqual(windows, [[22.0, 218.0]])


if __name__ == "__main__":
    unittest.main()
