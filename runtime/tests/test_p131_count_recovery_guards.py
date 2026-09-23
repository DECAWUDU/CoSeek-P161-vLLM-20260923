from __future__ import annotations

import json
import unittest

from videoseek.core.p130_runtime import (
    append_p130_observation,
    coalesce_p131_count_receipts,
    count_recovery_spec,
    init_p130_state,
    merge_p131_count_occurrences,
    recovery_observation_meaningful,
)
COUNT_QUESTION = """Throughout this video, what is the total count of occurrences for the scene featuring the 'cooking sausages' action?
(A) 1
(B) 0
(C) 4
(D) 3
Please directly answer with the best option's letter from the given choices directly (A, B, C, or D)."""


def observation(payload: dict) -> str:
    return "V10_OBSERVATION_JSON:\n```json\n" + json.dumps(payload) + "\n```"


class P131RecoverySchedulingTests(unittest.TestCase):
    def test_fifo_rounds_do_not_consume_two_residual_parities(self) -> None:
        memory: dict = {}
        init_p130_state(
            memory,
            question=COUNT_QUESTION,
            duration=160.0,
            p131_enabled=True,
        )
        state = memory["p130_global"]
        state["routing_candidates"] = [
            {
                "routing_key": f"loc:C{index}",
                "source_action_id": "loc",
                "candidate_id": f"C{index}",
                "t_range": [index * 5.0, index * 5.0 + 3.0],
                "timestamp_anchors": [index * 5.0 + 1.0],
                "verified_match": None,
            }
            for index in range(1, 7)
        ]

        first = count_recovery_spec(
            memory,
            question=COUNT_QUESTION,
            overview_output="",
            duration=160.0,
            attempt_index=0,
            max_candidates=2,
            p131_schedule=True,
        )
        self.assertEqual(first["reason"], "fifo_unverified_count_leads")
        state["recovery_history"].append(
            {"reason": first["reason"], "windows": first["windows"]}
        )
        second = count_recovery_spec(
            memory,
            question=COUNT_QUESTION,
            overview_output="",
            duration=160.0,
            attempt_index=1,
            max_candidates=2,
            p131_schedule=True,
        )
        self.assertEqual(second["reason"], "fifo_unverified_count_leads")
        self.assertTrue(set(map(tuple, first["windows"])).isdisjoint(map(tuple, second["windows"])))
        state["recovery_history"].append(
            {"reason": second["reason"], "windows": second["windows"]}
        )

        parity_zero = count_recovery_spec(
            memory,
            question=COUNT_QUESTION,
            overview_output="",
            duration=160.0,
            attempt_index=2,
            max_candidates=2,
            p131_schedule=True,
        )
        self.assertEqual(parity_zero["reason"], "residual_count_cursor")
        state["recovery_history"].append(
            {"reason": parity_zero["reason"], "windows": parity_zero["windows"]}
        )
        parity_one = count_recovery_spec(
            memory,
            question=COUNT_QUESTION,
            overview_output="",
            duration=160.0,
            attempt_index=3,
            max_candidates=2,
            p131_schedule=True,
        )
        self.assertEqual(parity_one["reason"], "residual_count_cursor")
        self.assertTrue(
            set(map(tuple, parity_zero["windows"])).isdisjoint(
                map(tuple, parity_one["windows"])
            )
        )
        self.assertEqual(
            sorted(parity_zero["windows"] + parity_one["windows"]),
            sorted(state["p131_count_residual_plan"]),
        )
        state["recovery_history"].append(
            {"reason": parity_one["reason"], "windows": parity_one["windows"]}
        )
        self.assertIsNone(
            count_recovery_spec(
                memory,
                question=COUNT_QUESTION,
                overview_output="",
                duration=160.0,
                attempt_index=4,
                max_candidates=2,
                p131_schedule=True,
            )
        )

    def test_uncommitted_failure_returns_same_recovery_window(self) -> None:
        memory: dict = {}
        init_p130_state(
            memory,
            question=COUNT_QUESTION,
            duration=80.0,
            p131_enabled=True,
        )
        first = count_recovery_spec(
            memory,
            question=COUNT_QUESTION,
            overview_output="",
            duration=80.0,
            attempt_index=0,
            max_candidates=2,
            p131_schedule=True,
        )
        second = count_recovery_spec(
            memory,
            question=COUNT_QUESTION,
            overview_output="",
            duration=80.0,
            attempt_index=0,
            max_candidates=2,
            p131_schedule=True,
        )
        self.assertEqual(first["windows"], second["windows"])
        self.assertEqual(memory["p130_global"]["recovery_history"], [])

    def test_only_structured_meaningful_output_is_committable(self) -> None:
        self.assertFalse(
            recovery_observation_meaningful(
                "frame_verify", "Tool execution failed: TimeoutError"
            )
        )
        self.assertFalse(
            recovery_observation_meaningful(
                "frame_verify",
                observation(
                    {
                        "parse_ok": True,
                        "candidate_assessments": [
                            {"candidate_id": "C1", "assessment_present": True}
                        ],
                    }
                ),
            )
        )
        self.assertTrue(
            recovery_observation_meaningful(
                "frame_verify",
                observation(
                    {
                        "parse_ok": True,
                        "candidate_assessments": [
                            {
                                "candidate_id": "C1",
                                "assessment_present": True,
                                "target_match": "not_visible",
                                "observed_fact": "No sausage is visible.",
                            }
                        ],
                    }
                ),
            )
        )


class P131CountReceiptTests(unittest.TestCase):
    def test_duplicate_candidate_rows_preserve_two_occurrences(self) -> None:
        candidate = {"candidate_id": "WIDE", "t_range": [0.0, 40.0]}
        occurrences = merge_p131_count_occurrences(
            [],
            [
                {
                    "target_match": "matched",
                    "event_match": "direct",
                    "event_span": [4.0, 8.0],
                    "best_timestamp_s": 6.0,
                    "observed_fact": "first sausage cooking scene",
                },
                {
                    "target_match": "matched",
                    "event_match": "direct",
                    "event_span": [26.0, 31.0],
                    "best_timestamp_s": 28.0,
                    "observed_fact": "second sausage cooking scene",
                },
            ],
        )
        self.assertEqual(len(occurrences), 2)
        rows = [
            {
                "candidate_id": "WIDE",
                "target_match": "matched",
                "event_match": "direct",
                "count_occurrences": occurrences,
            }
        ]

        memory: dict = {}
        init_p130_state(
            memory,
            question=COUNT_QUESTION,
            duration=40.0,
            p131_enabled=True,
        )
        snapshot = append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={"candidate_windows": [candidate]},
            output=observation(
                {
                    "parse_ok": True,
                    "candidate_assessments": rows,
                }
            ),
            action_id="wide-tile",
            question=COUNT_QUESTION,
            duration=40.0,
        )
        self.assertEqual(len(memory["p130_global"]["evidence_receipts"]), 2)
        self.assertEqual(snapshot["observed_count_lower_bound"], 2)

    def test_continuous_cross_shot_fragments_coalesce_but_disjoint_events_do_not(self) -> None:
        receipts = [
            {
                "receipt_id": "R1",
                "match": "direct",
                "event_span": [10.0, 15.0],
                "anchor_timestamps": [12.0],
                "source_candidate_ids": ["left-shot"],
                "fact": "sausages cook before the cut",
            },
            {
                "receipt_id": "R2",
                "match": "direct",
                "event_span": [15.4, 20.0],
                "anchor_timestamps": [18.0],
                "source_candidate_ids": ["right-shot"],
                "fact": "same cooking continues after the cut",
            },
            {
                "receipt_id": "R3",
                "match": "direct",
                "event_span": [30.0, 34.0],
                "anchor_timestamps": [32.0],
                "source_candidate_ids": ["same-wide-tile"],
                "fact": "a later separate cooking scene",
            },
        ]
        merged = coalesce_p131_count_receipts(receipts)
        direct = [row for row in merged if row.get("match") == "direct"]
        self.assertEqual(len(direct), 2)
        self.assertEqual(direct[0]["event_span"], [10.0, 20.0])
        self.assertEqual(direct[0]["coalesced_receipt_ids"], ["R1", "R2"])
        self.assertEqual(direct[1]["event_span"], [30.0, 34.0])

    def test_conflicting_count_status_fails_closed(self) -> None:
        memory: dict = {}
        init_p130_state(
            memory,
            question=COUNT_QUESTION,
            duration=20.0,
            p131_enabled=True,
        )
        snapshot = append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={},
            output=observation(
                {
                    "parse_ok": True,
                    "candidate_assessments": [
                        {
                            "candidate_id": "conflict",
                            "target_match": "not_visible",
                            "event_match": "direct",
                            "event_span": [4.0, 8.0],
                            "best_timestamp_s": 6.0,
                            "observed_fact": "conflicting model fields",
                        }
                    ],
                }
            ),
            action_id="conflict",
            question=COUNT_QUESTION,
            duration=20.0,
        )
        self.assertEqual(snapshot["observed_count_lower_bound"], 0)
        self.assertEqual(
            memory["p130_global"]["evidence_receipts"][0]["match"], "ambiguous"
        )

    def test_p131_does_not_write_a_synthetic_evidence_anchor(self) -> None:
        memory: dict = {}
        init_p130_state(
            memory,
            question=COUNT_QUESTION,
            duration=10.0,
            p131_enabled=True,
        )
        snapshot = append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={},
            output=observation(
                {
                    "parse_ok": True,
                    "candidate_assessments": [
                        {
                            "candidate_id": "C1",
                            "target_match": "matched",
                            "event_match": "direct",
                            "event_span": [2.0, 8.0],
                            "observed_fact": "",
                        }
                    ],
                }
            ),
            action_id="no-anchor",
            question=COUNT_QUESTION,
            duration=10.0,
        )
        self.assertEqual(memory["p130_global"]["evidence_receipts"], [])
        self.assertEqual(snapshot["observed_count_lower_bound"], 0)

    def test_parent_negative_dominates_nested_direct_occurrence(self) -> None:
        memory: dict = {}
        init_p130_state(
            memory,
            question=COUNT_QUESTION,
            duration=20.0,
            p131_enabled=True,
        )
        snapshot = append_p130_observation(
            memory,
            tool_name="frame_verify",
            parameters={},
            output=observation(
                {
                    "parse_ok": True,
                    "candidate_assessments": [
                        {
                            "candidate_id": "parent-negative",
                            "target_match": "not_visible",
                            "event_match": "ambiguous",
                            "count_occurrences": [
                                {
                                    "target_match": "matched",
                                    "event_match": "direct",
                                    "event_span": [5.0, 9.0],
                                    "best_timestamp_s": 7.0,
                                    "observed_fact": "contradictory nested occurrence",
                                }
                            ],
                        }
                    ],
                }
            ),
            action_id="nested-conflict",
            question=COUNT_QUESTION,
            duration=20.0,
        )
        self.assertEqual(snapshot["observed_count_lower_bound"], 0)
        self.assertEqual(
            memory["p130_global"]["evidence_receipts"][0]["match"],
            "ambiguous",
        )


if __name__ == "__main__":
    unittest.main()
