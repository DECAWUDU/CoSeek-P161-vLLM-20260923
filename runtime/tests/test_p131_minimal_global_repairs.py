from __future__ import annotations

import json
from pathlib import Path
import unittest

from videoseek.core.p130_runtime import (
    append_p130_observation,
    count_recovery_spec,
    init_p130_state,
    initial_count_verify_spec,
    initial_search_windows,
    normalize_p131_order_payload,
    order_chapter_spec,
    order_recovery_spec,
)
from videoseek.core.minimal_global_fsm import reduce_order


def observation(payload: dict) -> str:
    return "V10_OBSERVATION_JSON:\n```json\n" + json.dumps(payload) + "\n```"


R09 = """Throughout this video, what is the total count of occurrences for the scene featuring the 'cooking sausages' action?
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


class P131InitialSelectionTests(unittest.TestCase):
    def test_overview_rows_cluster_before_cap_and_keep_late_unique_lead(self) -> None:
        rows = [
            {"timestamp_s": value, "description": "sausages cooking in a pan"}
            for value in (10.0, 10.5, 11.0, 11.5, 12.0)
        ] + [
            {"timestamp_s": 100.0, "description": "sausages cooking in a pan"},
            {"timestamp_s": 200.0, "description": "sausages cooking in a pan"},
        ]
        overview = observation({"parse_ok": True, "timestamp_observations": rows})

        windows = initial_search_windows(R09, overview, duration=240.0, max_windows=3)
        self.assertEqual(len(windows), 3)
        self.assertTrue(any(start <= 100.0 <= end for start, end in windows))
        self.assertTrue(any(start <= 200.0 <= end for start, end in windows))

        memory: dict = {}
        init_p130_state(memory, question=R09, duration=240.0)
        spec = initial_count_verify_spec(
            memory,
            question=R09,
            overview_output=overview,
            duration=240.0,
            max_candidates=3,
        )
        self.assertIsNotNone(spec)
        self.assertEqual(len(spec["candidate_windows"]), 3)
        self.assertTrue(all(row["summary"] == "" for row in spec["candidate_windows"]))
        first = spec["candidate_windows"][0]
        self.assertTrue(
            {10.0, 10.5, 11.0, 11.5, 12.0}.issubset(
                set(first["timestamp_anchors"])
            )
        )
        self.assertEqual(first["timestamp_anchors"][0], first["t_range"][0])
        self.assertEqual(first["timestamp_anchors"][-1], first["t_range"][1])

    def test_localizer_candidate_uses_local_window_and_no_global_summary(self) -> None:
        memory: dict = {}
        init_p130_state(memory, question=R09, duration=490.0)
        append_p130_observation(
            memory,
            tool_name="localize_qwen",
            parameters={"search_windows": [[245.0, 490.0]]},
            output=observation(
                {
                    "parse_ok": True,
                    "timestamp_observations": [
                        {"timestamp_s": 313.0, "description": "sausages cooking"}
                    ],
                    "ranked_candidates": [
                        {
                            "candidate_id": "LQ001",
                            "t_range": [245.0, 490.0],
                            "recommended_verify_window": [446.0, 464.0],
                            "fine_summary": "global summary repeats sausage evidence",
                            "fine_evidence_class": "negative",
                            "selection_class": "mandatory_positive",
                            "localized_timestamp_anchors": [313.0, 455.0],
                        }
                    ],
                }
            ),
            action_id="localize-late",
            question=R09,
            duration=490.0,
        )
        route = memory["p130_global"]["routing_candidates"][0]
        self.assertEqual(route["t_range"], [446.0, 464.0])
        self.assertEqual(route["summary"], "")
        self.assertEqual(route["timestamp_anchors"], [455.0])
        self.assertEqual(route["selection_class"], "ambiguous")

    def test_count_anchor_clustering_cannot_chain_past_twelve_seconds(self) -> None:
        memory: dict = {}
        init_p130_state(memory, question=R09, duration=140.0)
        overview = observation(
            {
                "parse_ok": True,
                "timestamp_observations": [
                    {
                        "timestamp_s": value,
                        "description": "sausages cooking in a pan",
                    }
                    for value in (70.0, 81.0, 92.0)
                ],
            }
        )
        spec = initial_count_verify_spec(
            memory,
            question=R09,
            overview_output=overview,
            duration=140.0,
            max_candidates=8,
        )
        self.assertEqual(len(spec["candidate_windows"]), 2)
        self.assertFalse(
            any(
                row["t_range"][0] <= 70.0 and row["t_range"][1] >= 92.0
                for row in spec["candidate_windows"]
            )
        )

    def test_source_provenance_prevents_suffix_writeback_broadcast(self) -> None:
        memory: dict = {}
        init_p130_state(memory, question=R09, duration=490.0)
        for action_id, window in (("loc-1", [70.0, 82.0]), ("loc-2", [420.0, 438.0])):
            append_p130_observation(
                memory,
                tool_name="localize_qwen",
                parameters={"search_windows": [window]},
                output=observation(
                    {
                        "parse_ok": True,
                        "ranked_candidates": [
                            {"candidate_id": "LQ001", "recommended_verify_window": window}
                        ],
                    }
                ),
                action_id=action_id,
                question=R09,
                duration=490.0,
            )
        append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={"source_localize_action_id": "loc-2"},
            output=observation(
                {
                    "parse_ok": True,
                    "candidate_assessments": [
                        {
                            "candidate_id": "LQ001",
                            "event_match": "negative",
                            "event_span": [422.0, 430.0],
                            "observed_fact": "different food",
                        }
                    ],
                }
            ),
            action_id="verify-2",
            question=R09,
            duration=490.0,
        )
        routes = {
            row["routing_key"]: row for row in memory["p130_global"]["routing_candidates"]
        }
        self.assertIsNone(routes["loc-1:LQ001"]["verified_match"])
        self.assertEqual(routes["loc-2:LQ001"]["verified_match"], "negative")


class P131OrderTests(unittest.TestCase):
    def test_distant_duplicate_typed_id_is_a_binding_conflict(self) -> None:
        def direct(event_id: str, timestamp: float, receipt_id: str) -> dict:
            return {
                "receipt_id": receipt_id,
                "event_span": [timestamp - 2.0, timestamp + 2.0],
                "anchor_timestamps": [timestamp],
                "match": "direct",
                "target_event_ids": [event_id],
                "binding_confidence": "direct",
                "fact": f"event {event_id}",
                "event_key": f"order:event:{event_id}",
                "identity_source": "event_id",
                "p131_temporal_identity_guard": True,
                "source_action_id": receipt_id,
            }

        state = reduce_order(
            R10,
            [
                direct("1", 30.0, "near-1a"),
                direct("1", 45.0, "near-1b"),
                direct("1", 60.0, "chain-must-stop"),
                direct("1", 160.0, "remote-1"),
                direct("2", 180.0, "event-2"),
                direct("3", 200.0, "event-3"),
                direct("4", 220.0, "event-4"),
            ],
        )

        self.assertNotIn("1", state["bound_event_ids"])
        self.assertEqual(
            [row["event_id"] for row in state["binding_conflicts"]],
            ["1"],
        )
        typed_one_events = [
            row for row in state["event_table"] if row["target_event_ids"] == ["1"]
        ]
        self.assertEqual(len(typed_one_events), 3)
        self.assertEqual(len(state["event_table"]), 6)
        self.assertFalse(state["decision_sufficient"])

    def test_one_comparative_packet_merges_a_long_continuous_event_run(self) -> None:
        timestamps = [101.0, 112.0, 128.0, 144.0, 160.0]
        payload = {
            "parse_ok": True,
            "timestamp_observations": [
                {
                    "candidate_id": f"long-{index}",
                    "timestamp_s": timestamp,
                    "event_id": "3",
                    "event_match": "direct",
                    "observed_fact": "the man continues balancing above the cold river",
                }
                for index, timestamp in enumerate(timestamps, start=1)
            ],
        }
        catalog = [
            {"event_id": str(index), "description": f"event {index}"}
            for index in range(1, 5)
        ]
        normalize_p131_order_payload(
            payload,
            catalog=catalog,
            allowed_timestamps=timestamps,
        )
        memory: dict = {}
        init_p130_state(memory, question=R10, duration=240.0)

        snapshot = append_p130_observation(
            memory,
            tool_name="skim",
            parameters={"start_time": 90.0, "end_time": 170.0},
            output=observation(payload),
            action_id="p131_order_chapter_scan",
            question=R10,
            duration=240.0,
        )

        self.assertIn("3", snapshot["bound_event_ids"])
        self.assertEqual(snapshot["binding_conflicts"], [])
        event = next(
            row for row in snapshot["event_table"] if row["target_event_ids"] == ["3"]
        )
        self.assertEqual(event["anchor_timestamps"], timestamps)
        self.assertEqual(event["p131_order_run_ids"], ["event:3:run:1"])

    def test_interleaved_direct_catalog_id_splits_same_packet_event_run(self) -> None:
        sequence = [
            (101.0, "3"),
            (112.0, "3"),
            (120.0, "2"),
            (128.0, "3"),
            (144.0, "3"),
            (160.0, "3"),
        ]
        payload = {
            "parse_ok": True,
            "timestamp_observations": [
                {
                    "candidate_id": f"row-{index}",
                    "timestamp_s": timestamp,
                    "event_id": event_id,
                    "event_match": "direct",
                    "observed_fact": f"visible event {event_id}",
                }
                for index, (timestamp, event_id) in enumerate(sequence, start=1)
            ],
        }
        catalog = [
            {"event_id": str(index), "description": f"event {index}"}
            for index in range(1, 5)
        ]
        normalize_p131_order_payload(
            payload,
            catalog=catalog,
            allowed_timestamps=[timestamp for timestamp, _event_id in sequence],
        )
        memory: dict = {}
        init_p130_state(memory, question=R10, duration=240.0)

        snapshot = append_p130_observation(
            memory,
            tool_name="skim",
            parameters={"start_time": 90.0, "end_time": 170.0},
            output=observation(payload),
            action_id="p131_order_chapter_scan",
            question=R10,
            duration=240.0,
        )

        self.assertNotIn("3", snapshot["bound_event_ids"])
        self.assertEqual(
            [row["event_id"] for row in snapshot["binding_conflicts"]],
            ["3"],
        )
        event_three_rows = [
            row for row in snapshot["event_table"] if row["target_event_ids"] == ["3"]
        ]
        self.assertEqual(len(event_three_rows), 2)
        self.assertEqual(
            [row["p131_order_run_ids"] for row in event_three_rows],
            [["event:3:run:1"], ["event:3:run:2"]],
        )

    def test_omitted_sample_grid_frames_split_distant_same_id_rows(self) -> None:
        allowed_timestamps = [10.0, 50.0, 100.0, 150.0, 200.0, 210.0, 220.0, 230.0]
        sequence = [
            (10.0, "1"),
            (200.0, "1"),
            (210.0, "2"),
            (220.0, "3"),
            (230.0, "4"),
        ]
        payload = {
            "parse_ok": True,
            "timestamp_observations": [
                {
                    "candidate_id": f"sparse-{index}",
                    "timestamp_s": timestamp,
                    "event_id": event_id,
                    "event_match": "direct",
                    "observed_fact": f"visible event {event_id}",
                }
                for index, (timestamp, event_id) in enumerate(sequence, start=1)
            ],
        }
        catalog = [
            {"event_id": str(index), "description": f"event {index}"}
            for index in range(1, 5)
        ]
        normalize_p131_order_payload(
            payload,
            catalog=catalog,
            allowed_timestamps=allowed_timestamps,
        )
        event_one_rows = [
            row for row in payload["timestamp_observations"] if row["event_id"] == "1"
        ]
        self.assertEqual(
            [row["p131_sample_index"] for row in event_one_rows],
            [0, 4],
        )
        self.assertEqual(
            [row["p131_order_run_id"] for row in event_one_rows],
            ["event:1:run:1", "event:1:run:2"],
        )

        memory: dict = {}
        init_p130_state(memory, question=R10, duration=240.0)
        snapshot = append_p130_observation(
            memory,
            tool_name="skim",
            parameters={"start_time": 0.0, "end_time": 240.0},
            output=observation(payload),
            action_id="p131_order_chapter_scan",
            question=R10,
            duration=240.0,
        )

        self.assertNotIn("1", snapshot["bound_event_ids"])
        self.assertEqual(
            [row["event_id"] for row in snapshot["binding_conflicts"]],
            ["1"],
        )
        event_one_events = [
            row for row in snapshot["event_table"] if row["target_event_ids"] == ["1"]
        ]
        self.assertEqual(len(event_one_events), 2)

    def test_comparative_direct_requires_raw_sample_timestamp_fact_and_one_id(self) -> None:
        payload = {
            "parse_ok": True,
            "timestamp_observations": [
                {
                    "candidate_id": "valid",
                    "timestamp_s": 10.0,
                    "event_id": "1",
                    "event_match": "direct",
                    "observed_fact": "adult and child balance together",
                },
                {
                    "candidate_id": "missing-ts",
                    "event_id": "2",
                    "event_match": "direct",
                    "observed_fact": "people walk through a bridge",
                },
                {
                    "candidate_id": "illegal-ts",
                    "timestamp_s": 999.0,
                    "event_id": "3",
                    "event_match": "direct",
                    "observed_fact": "man balances over a river",
                },
                {
                    "candidate_id": "empty-fact",
                    "timestamp_s": 20.0,
                    "event_id": "4",
                    "event_match": "direct",
                    "observed_fact": "",
                    "description": "description fallback must not prove a direct fact",
                },
                {
                    "candidate_id": "conflicting-id-aliases",
                    "timestamp_s": 20.0,
                    "event_id": "1",
                    "target_event_id": "2",
                    "event_match": "direct",
                    "observed_fact": "one visible scene",
                },
            ],
        }
        catalog = [
            {"event_id": str(index), "description": f"event {index}"}
            for index in range(1, 5)
        ]

        normalize_p131_order_payload(
            payload,
            catalog=catalog,
            allowed_timestamps=[10.0, 20.0],
        )
        rows = {
            row["candidate_id"]: row for row in payload["timestamp_observations"]
        }
        # This is the generic V10 fallback performed later by skim.py.  The raw
        # validation flag and demoted match must survive that synthesized value.
        rows["missing-ts"]["timestamp_s"] = 10.0
        self.assertTrue(rows["valid"]["p131_direct_evidence_valid"])
        self.assertEqual(rows["valid"]["event_match"], "direct")
        for candidate_id in (
            "missing-ts",
            "illegal-ts",
            "empty-fact",
            "conflicting-id-aliases",
        ):
            self.assertFalse(rows[candidate_id]["p131_direct_evidence_valid"])
            self.assertEqual(rows[candidate_id]["event_match"], "ambiguous")
        self.assertEqual(rows["missing-ts"]["timestamp_s"], 10.0)
        self.assertTrue(rows["conflicting-id-aliases"]["p131_id_alias_conflict"])

        memory: dict = {}
        init_p130_state(memory, question=R10, duration=240.0)
        snapshot = append_p130_observation(
            memory,
            tool_name="skim",
            parameters={"start_time": 0.0, "end_time": 240.0},
            output=observation(payload),
            action_id="p131_order_chapter_scan",
            question=R10,
            duration=240.0,
        )
        self.assertEqual(
            [row["source_candidate_ids"] for row in memory["p130_global"]["evidence_receipts"]],
            [["valid"]],
        )
        self.assertEqual(snapshot["bound_event_ids"], ["1"])

    def test_replayed_comparative_direct_without_validation_flags_is_rejected(self) -> None:
        memory: dict = {}
        init_p130_state(memory, question=R10, duration=240.0)
        payload = {
            "parse_ok": True,
            "p131_order_comparative": True,
            "timestamp_observations": [
                {
                    "candidate_id": "unvalidated",
                    "timestamp_s": 10.0,
                    "event_id": "1",
                    "event_match": "direct",
                    "observed_fact": "adult and child balance together",
                }
            ],
        }

        snapshot = append_p130_observation(
            memory,
            tool_name="skim",
            parameters={"start_time": 0.0, "end_time": 240.0},
            output=observation(payload),
            action_id="p131_order_chapter_scan",
            question=R10,
            duration=240.0,
        )

        self.assertEqual(memory["p130_global"]["evidence_receipts"], [])
        self.assertEqual(snapshot["bound_event_ids"], [])

    def test_comparative_skim_emits_only_unique_typed_direct_receipts(self) -> None:
        memory: dict = {}
        init_p130_state(memory, question=R10, duration=695.0)
        snapshot = append_p130_observation(
            memory,
            tool_name="skim",
            parameters={"start_time": 20.0, "end_time": 210.0},
            output=observation(
                {
                    "parse_ok": True,
                    "timestamp_observations": [
                        {
                            "candidate_id": "S1",
                            "timestamp_s": 33.0,
                            "event_match": "direct",
                            "target_event_id": "1",
                            "observed_fact": "adult and child balance together",
                        },
                        {
                            "candidate_id": "S2",
                            "timestamp_s": 62.0,
                            "event_match": "ambiguous",
                            "target_event_id": "2",
                            "observed_fact": "bridge context",
                        },
                        {
                            "candidate_id": "S23",
                            "timestamp_s": 90.0,
                            "event_match": "direct",
                            "target_event_ids": ["2", "3"],
                            "observed_fact": "cannot distinguish two catalog events",
                        },
                        {
                            "candidate_id": "S3",
                            "timestamp_s": 125.0,
                            "event_match": "direct",
                            "target_event_id": "3",
                            "observed_fact": "man balances above river",
                        },
                    ],
                }
            ),
            action_id="order-chapter",
            question=R10,
            duration=695.0,
        )
        receipts = memory["p130_global"]["evidence_receipts"]
        self.assertEqual([row["event_key"] for row in receipts], ["order:event:1", "order:event:3"])
        self.assertEqual(snapshot["bound_event_ids"], ["1", "3"])
        routes = memory["p130_global"]["routing_candidates"]
        self.assertEqual({row["candidate_id"] for row in routes}, {"S2", "S23"})

    def test_order_chapter_is_one_bounded_continuous_window(self) -> None:
        memory: dict = {}
        init_p130_state(memory, question=R10, duration=695.0)
        overview = observation(
            {
                "parse_ok": True,
                "scene_summaries": [
                    {
                        "t_range": [30.0, 210.0],
                        "summary": "snowy slackline competition, covered bridge and cold river",
                        "possible_evidence": True,
                    },
                    {
                        "t_range": [300.0, 690.0],
                        "summary": "birds and wildlife",
                        "possible_evidence": False,
                    },
                ],
                "timestamp_observations": [
                    {
                        "timestamp_s": timestamp,
                        "description": description,
                        "target_event_id": event_id,
                    }
                    for timestamp, event_id, description in (
                        (33.0, "1", "event 1 adult and child balance"),
                        (62.0, "2", "event 2 people walk on bridge"),
                        (125.0, "3", "event 3 man over cold river"),
                        (171.0, "4", "event 4 boy in snowy competition"),
                    )
                ],
            }
        )
        spec = order_chapter_spec(
            memory, question=R10, overview_output=overview, duration=695.0
        )
        self.assertEqual(spec["kind"], "skim")
        self.assertEqual(len(spec["windows"]), 1)
        self.assertLessEqual(spec["start_time"], 24.0)
        self.assertGreaterEqual(spec["end_time"], 210.0)
        self.assertLessEqual(spec["end_time"] - spec["start_time"], 240.0)
        self.assertLessEqual(len(spec["timestamps"]), 16)
        self.assertEqual([row["event_id"] for row in spec["event_catalog"]], ["1", "2", "3", "4"])

    def test_order_recovery_uses_next_best_candidate_on_second_round(self) -> None:
        memory: dict = {}
        init_p130_state(memory, question=R10, duration=695.0)
        append_p130_observation(
            memory,
            tool_name="skim",
            parameters={"start_time": 20.0, "end_time": 210.0},
            output=observation(
                {
                    "parse_ok": True,
                    "timestamp_observations": [
                        {
                            "candidate_id": "bridge-ambiguous",
                            "timestamp_s": 62.0,
                            "event_match": "ambiguous",
                            "target_event_id": "2",
                            "description": "event 2 people walking on bridge",
                        }
                    ],
                }
            ),
            action_id="skim-order",
            question=R10,
            duration=695.0,
        )
        overview = observation(
            {
                "parse_ok": True,
                "timestamp_observations": [
                    {
                        "timestamp_s": 62.0,
                        "description": "event 2 people walking on bridge",
                        "target_event_id": "2",
                    },
                    {
                        "timestamp_s": 92.0,
                        "description": "event 2 people cross covered bridge",
                        "target_event_id": "2",
                    },
                ],
            }
        )
        first = order_recovery_spec(
            memory,
            question=R10,
            overview_output=overview,
            duration=695.0,
            attempt_index=0,
        )
        self.assertEqual(first["target_event_ids"], ["2"])
        self.assertEqual(first["candidate_windows"][0]["summary"], "")
        memory["p130_global"]["recovery_history"].append(
            {"reason": first["reason"], "windows": first["windows"]}
        )
        second = order_recovery_spec(
            memory,
            question=R10,
            overview_output=overview,
            duration=695.0,
            attempt_index=1,
        )
        self.assertGreater(second["windows"][0][0], first["windows"][0][1])


class P131CountReducerAndRecoveryTests(unittest.TestCase):
    def test_cross_round_overlapping_count_receipts_deduplicate(self) -> None:
        memory: dict = {}
        init_p130_state(memory, question=R09, duration=490.0)
        for index, (span, anchor) in enumerate(
            (([70.0, 78.0], 74.0), ([72.0, 80.0], 75.0), ([108.0, 116.0], 112.0)),
            start=1,
        ):
            snapshot = append_p130_observation(
                memory,
                tool_name="frame_verify",
                parameters={"start_time": span[0], "end_time": span[1]},
                output=observation(
                    {
                        "parse_ok": True,
                        "candidate_assessments": [
                            {
                                "candidate_id": f"round-{index}",
                                "packet_group_id": "same-packet-label",
                                "event_match": "direct",
                                "event_span": span,
                                "best_timestamp_s": anchor,
                                "observed_fact": "sausages cooking",
                            }
                        ],
                    }
                ),
                action_id=f"verify-{index}",
                question=R09,
                duration=490.0,
            )
        self.assertEqual(snapshot["observed_count_lower_bound"], 2)
        self.assertEqual(len(memory["p130_global"]["evidence_receipts"]), 3)

    def test_residual_rounds_are_disjoint_and_exclude_direct_spans_with_guard(self) -> None:
        memory: dict = {}
        init_p130_state(memory, question=R09, duration=160.0)
        append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={},
            output=observation(
                {
                    "parse_ok": True,
                    "candidate_assessments": [
                        {
                            "candidate_id": "known",
                            "event_match": "direct",
                            "event_span": [40.0, 50.0],
                            "best_timestamp_s": 45.0,
                            "observed_fact": "sausages cooking",
                        }
                    ],
                }
            ),
            action_id="known-event",
            question=R09,
            duration=160.0,
        )
        first = count_recovery_spec(
            memory,
            question=R09,
            overview_output="",
            duration=160.0,
            attempt_index=0,
            max_candidates=4,
        )
        self.assertEqual(first["reason"], "residual_count_cursor")
        self.assertTrue(
            all(end <= 38.0 or start >= 52.0 for start, end in first["windows"])
        )
        memory["p130_global"]["recovery_history"].append(
            {"reason": first["reason"], "windows": first["windows"]}
        )
        second = count_recovery_spec(
            memory,
            question=R09,
            overview_output="",
            duration=160.0,
            attempt_index=1,
            max_candidates=4,
        )
        self.assertEqual(second["reason"], "residual_count_cursor")
        for left in first["windows"]:
            for right in second["windows"]:
                overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
                self.assertEqual(overlap, 0.0)


class P131SavedOverviewReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "tests"
            / "fixtures"
            / "p131_real_overview_replay.json"
        )
        cls.fixture = json.loads(fixture_path.read_text())

    def test_r09_saved_overview_keeps_four_independent_event_hypotheses(self) -> None:
        case = self.fixture["R09_count_166"]
        memory: dict = {}
        init_p130_state(memory, question=case["question"], duration=case["duration"])
        spec = initial_count_verify_spec(
            memory,
            question=case["question"],
            overview_output=observation(case["overview_payload"]),
            duration=case["duration"],
            max_candidates=8,
        )
        candidates = spec["candidate_windows"]
        self.assertEqual(len(candidates), 4)
        owners = {
            target: [
                index
                for index, row in enumerate(candidates)
                if row["t_range"][0] <= target <= row["t_range"][1]
            ]
            for target in (73.5, 108.9, 318.6, 429.5)
        }
        self.assertTrue(all(len(indices) == 1 for indices in owners.values()))
        self.assertNotEqual(owners[73.5], owners[108.9])
        late = candidates[owners[429.5][0]]
        self.assertIn(429.5, late["timestamp_anchors"])
        self.assertTrue(any(422.0 <= value <= 424.0 for value in late["timestamp_anchors"]))

    def test_r10_saved_overview_schedule_reaches_late_chapter(self) -> None:
        case = self.fixture["R10_order_2"]
        memory: dict = {}
        init_p130_state(memory, question=case["question"], duration=case["duration"])
        spec = order_chapter_spec(
            memory,
            question=case["question"],
            overview_output=observation(case["overview_payload"]),
            duration=case["duration"],
            max_frames=16,
        )
        timestamps = spec["timestamps"]
        self.assertEqual(len(timestamps), 16)
        self.assertLessEqual(timestamps[0], 5.0)
        self.assertGreaterEqual(timestamps[-1], 180.0)
        self.assertGreaterEqual(timestamps[-1] - timestamps[0], 175.0)


if __name__ == "__main__":
    unittest.main()
