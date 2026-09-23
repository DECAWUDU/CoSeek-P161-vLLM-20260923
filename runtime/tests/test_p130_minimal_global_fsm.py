from __future__ import annotations

from copy import deepcopy
import unittest

from videoseek.core.minimal_global_fsm import (
    build_event_table,
    detect_global_mode,
    event_table_version,
    force_answer,
    make_recovery_signature,
    parse_global_question,
    recovery_allowed,
    reduce_count,
    reduce_global,
    reduce_order,
    should_use_minimal_global_fsm,
    validate_answer,
)


R09_QUESTION = """Throughout this video, what is the total count of occurrences for the scene featuring the 'cooking sausages' action
(A) 1
(B) 0
(C) 4
(D) 3
Please directly answer with the best option's letter from the given choices directly (A, B, C, or D)."""

R10_QUESTION = """Arrange the following events from the video in the correct chronological order: (1)a young and a kid are doing balance in a balance rope; (2)people are walking in a bridge to see a competition of men doing tricks on top of a balance rope; (3)a man is jumping and doing tricks in a balance rope above a cold river; (4)the boy is in a competition in snowy path doing tricks on a balance rope with people behind a fence watching him.
(A) 2->1->3->4
(B) 1->2->3->4
(C) 4->3->2->1
(D) 3->2->1->4
Please directly answer with the best option's letter from the given choices directly (A, B, C, or D)."""


def _count_event(index: int, start: float) -> dict:
    return {
        "event_id": f"TE{index:03d}",
        "event_match": "direct",
        "t_range": [start, start + 5.0],
        "anchor_timestamps": [start + 2.0],
        "fact": "Sausages are visibly cooking in a pan.",
        # P130 must ignore these legacy whole-answer claims.
        "supports_options": ["A"],
        "contradicts_options": ["B", "C", "D"],
    }


def _order_event(
    occurrence: str,
    target_id: str | list[str],
    start: float,
    *,
    match: str = "direct",
    confidence: str = "direct",
) -> dict:
    ids = target_id if isinstance(target_id, list) else [target_id]
    return {
        "event_id": occurrence,
        "t_range": [start, start + 4.0],
        "anchor_timestamps": [start + 1.0],
        "event_match": match,
        "matched_event_ids": ids,
        "binding_confidence": confidence,
        "fact": f"Local fact for event {target_id}.",
    }


class QuestionParsingTests(unittest.TestCase):
    def test_strict_route_excludes_local_cardinality_questions(self) -> None:
        self.assertFalse(
            should_use_minimal_global_fsm(
                "How many people are interacting at the staircase?\n(A) 1\n(B) 2\n(C) 3\n(D) 4"
            )
        )
        self.assertFalse(
            should_use_minimal_global_fsm(
                "How many picture frames were on the wall?\n(A) 1\n(B) 2\n(C) 3\n(D) 4"
            )
        )
        self.assertTrue(should_use_minimal_global_fsm(R09_QUESTION))
        self.assertTrue(should_use_minimal_global_fsm(R10_QUESTION))

    def test_parses_r09_numeric_options_without_trailing_instruction(self) -> None:
        parsed = parse_global_question(R09_QUESTION)

        self.assertEqual(detect_global_mode(R09_QUESTION), "count")
        self.assertEqual(parsed["mode"], "count")
        self.assertEqual(
            [(row["option"], row["numeric_value"]) for row in parsed["options"]],
            [("A", 1), ("B", 0), ("C", 4), ("D", 3)],
        )
        self.assertEqual(parsed["options"][-1]["text"], "3")
        self.assertEqual(parsed["parse_errors"], [])

    def test_parses_r10_required_events_and_sequences(self) -> None:
        parsed = parse_global_question(R10_QUESTION)

        self.assertEqual(parsed["mode"], "order")
        self.assertEqual(parsed["required_event_ids"], ["1", "2", "3", "4"])
        self.assertIn("young", parsed["required_events"][0]["description"])
        self.assertEqual(
            {row["option"]: row["sequence"] for row in parsed["options"]},
            {
                "A": ["2", "1", "3", "4"],
                "B": ["1", "2", "3", "4"],
                "C": ["4", "3", "2", "1"],
                "D": ["3", "2", "1", "4"],
            },
        )
        self.assertEqual(parsed["parse_errors"], [])

    def test_other_mode_is_explicitly_delegated_to_p128(self) -> None:
        state = reduce_global("What color is the hat?\n(A) red\n(B) blue", [])

        self.assertEqual(state["mode"], "other")
        self.assertTrue(state["delegated_to_p128"])
        self.assertFalse(state["decision_sufficient"])


class CanonicalEventTableTests(unittest.TestCase):
    def test_same_explicit_group_merges_but_distinct_event_ids_do_not(self) -> None:
        receipts = [
            {
                "evidence_id": "E1",
                "event_group_id": "verify:EG1",
                "event_span": [70.0, 80.0],
                "best_timestamp_s": 75.0,
                "event_match": "direct",
                "observed_fact": "Cooking begins.",
                "supports_options": ["A"],
            },
            {
                "evidence_id": "E2",
                "event_group_id": "verify:EG1",
                "event_span": [76.0, 82.0],
                "best_timestamp_s": 79.0,
                "event_match": "direct",
                "observed_fact": "The same cooking continues.",
                "contradicts_options": ["D"],
            },
            {
                "event_id": "TE002",
                "t_range": [77.0, 81.0],
                "anchor_timestamps": [78.0],
                "fact": "A separately identified ledger event.",
            },
        ]

        table = build_event_table(receipts)

        self.assertEqual(len(table), 2)
        grouped = next(row for row in table if row["event_key"] == "group:verify:EG1")
        self.assertEqual(grouped["receipt_ids"], ["E1", "E2"])
        self.assertEqual(grouped["t_range"], [70.0, 82.0])
        self.assertNotIn("supports_options", grouped)
        self.assertNotIn("contradicts_options", grouped)

    def test_temporal_ledger_is_preferred_over_parallel_option_memory(self) -> None:
        memory = {
            "temporal_evidence_ledger": {
                "stable_event_span_count": 1,
                "events": [_count_event(1, 10.0)],
            },
            "structured_evidence": {
                "evidence_items": [
                    {
                        "evidence_id": "BAD",
                        "event_match": "direct",
                        "t_range": [100.0, 110.0],
                        "supports_options": ["B"],
                    }
                ]
            },
        }

        table = build_event_table(memory)

        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]["event_key"], "event_id:TE001")

    def test_event_version_is_order_independent_and_changes_with_new_evidence(self) -> None:
        first = _count_event(1, 10.0)
        second = _count_event(2, 30.0)

        self.assertEqual(
            event_table_version([first, second]),
            event_table_version([second, first]),
        )
        self.assertNotEqual(
            event_table_version([first]),
            event_table_version([first, second]),
        )


class CountReducerTests(unittest.TestCase):
    def test_r09_three_events_remain_a_lower_bound_despite_nominal_full_range(self) -> None:
        memory = {
            "temporal_evidence_ledger": {
                "stable_event_span_count": 3,
                "events": [
                    _count_event(1, 72.0),
                    _count_event(2, 108.0),
                    _count_event(3, 311.0),
                ],
            },
            # This was the P129 false-closure trigger.  P130 does not consume it.
            "compact_investigation_state": {
                "inspected_windows": [{"t_range": [0.0, 490.1]}],
                "coverage_ratio": 1.0,
            },
        }
        before = deepcopy(memory)

        state = reduce_count(R09_QUESTION, memory)

        self.assertEqual(memory, before)
        self.assertEqual(state["observed_count_lower_bound"], 3)
        self.assertEqual(state["viable_options"], ["C", "D"])
        self.assertEqual(state["eliminated_options"], ["A", "B"])
        self.assertIsNone(state["possible_count_upper_bound"])
        self.assertFalse(state["enumeration_complete"])
        self.assertFalse(state["decision_sufficient"])
        self.assertEqual(state["recovery_need"]["kind"], "residual_count_search")
        self.assertEqual(state["recovery_need"]["next_offered_count"], 4)

    def test_explicit_enumeration_certificate_closes_exact_three(self) -> None:
        state = reduce_count(
            R09_QUESTION,
            [_count_event(1, 72.0), _count_event(2, 108.0), _count_event(3, 311.0)],
            enumeration_complete=True,
        )

        self.assertTrue(state["enumeration_complete"])
        self.assertEqual(state["possible_count_upper_bound"], 3)
        self.assertEqual(state["viable_options"], ["D"])
        self.assertEqual(state["validated_option"], "D")
        self.assertTrue(state["decision_sufficient"])

    def test_four_verified_events_make_r09_option_c_the_only_offered_value(self) -> None:
        state = reduce_count(
            R09_QUESTION,
            [
                _count_event(1, 72.0),
                _count_event(2, 108.0),
                _count_event(3, 311.0),
                _count_event(4, 430.0),
            ],
        )

        self.assertEqual(state["observed_count_lower_bound"], 4)
        self.assertEqual(state["viable_options"], ["C"])
        self.assertTrue(state["remaining_uncertainty_maps_same_option"])
        self.assertTrue(state["decision_sufficient"])
        self.assertEqual(state["validated_option"], "C")

    def test_context_and_negative_receipts_never_raise_the_count(self) -> None:
        receipts = [
            _count_event(1, 72.0),
            {
                "evidence_id": "context",
                "event_match": "context_only",
                "t_range": [300.0, 310.0],
                "observed_fact": "A kitchen is visible.",
            },
            {
                "evidence_id": "negative",
                "event_match": "not_visible",
                "t_range": [400.0, 410.0],
                "observed_fact": "Wildlife only.",
            },
        ]

        state = reduce_count(R09_QUESTION, receipts)

        self.assertEqual(state["observed_count_lower_bound"], 1)
        self.assertEqual(len(state["event_table"]), 3)


class OrderReducerTests(unittest.TestCase):
    def test_r10_explicit_bindings_reduce_to_b(self) -> None:
        receipts = [
            _order_event("TE001", "1", 26.7),
            _order_event("TE002", "2", 69.3),
            _order_event("TE003", "3", 113.5),
            _order_event("TE004", "4", 157.7),
        ]

        state = reduce_order(R10_QUESTION, receipts)

        self.assertEqual(state["bound_event_ids"], ["1", "2", "3", "4"])
        self.assertEqual(state["unbound_event_ids"], [])
        self.assertEqual(
            state["precedence_edges"],
            [["1", "2"], ["1", "3"], ["1", "4"], ["2", "3"], ["2", "4"], ["3", "4"]],
        )
        self.assertEqual(state["viable_options"], ["B"])
        self.assertEqual(state["validated_option"], "B")
        self.assertTrue(state["decision_sufficient"])

    def test_partial_bindings_can_eliminate_but_cannot_validate_order(self) -> None:
        state = reduce_order(
            R10_QUESTION,
            [_order_event("TE001", "1", 26.7), _order_event("TE002", "2", 88.3)],
        )

        self.assertEqual(state["viable_options"], ["B"])
        self.assertEqual(state["unbound_event_ids"], ["3", "4"])
        self.assertFalse(state["coverage_closed"])
        self.assertFalse(state["decision_sufficient"])
        self.assertIsNone(state["validated_option"])
        self.assertEqual(state["recovery_need"]["target_event_ids"], ["3", "4"])

    def test_bridge_like_context_cannot_bind_event_two(self) -> None:
        receipts = [
            _order_event("TE001", "1", 26.7),
            _order_event("TC001", "2", 12.6, match="context_only"),
            _order_event("TE003", "3", 113.5),
            _order_event("TE004", "4", 157.7),
        ]

        state = reduce_order(R10_QUESTION, receipts)

        self.assertNotIn("2", state["bound_event_ids"])
        self.assertIn("2", state["unbound_event_ids"])
        self.assertFalse(state["decision_sufficient"])
        self.assertIsNone(state["validated_option"])

    def test_multiple_ids_and_free_text_evt_labels_are_not_guessed(self) -> None:
        receipts = [
            _order_event("ambiguous", ["1", "2"], 20.0),
            {
                "event_id": "prose-only",
                "event_match": "direct",
                "t_range": [50.0, 55.0],
                "fact": "This prose claims EVT03 but has no structured target ID.",
            },
        ]

        state = reduce_order(R10_QUESTION, receipts)

        self.assertEqual(state["bound_event_ids"], [])
        self.assertEqual(state["unbound_event_ids"], ["1", "2", "3", "4"])
        self.assertEqual(len(state["ambiguous_receipts"]), 1)

    def test_repeated_typed_order_id_has_one_canonical_event(self) -> None:
        receipts = [
            _order_event("occurrence-a", "1", 10.0),
            _order_event("occurrence-b", "1", 100.0),
            _order_event("occurrence-c", "2", 120.0),
        ]

        state = reduce_order(R10_QUESTION, receipts)

        self.assertEqual(state["bound_event_ids"], ["1", "2"])
        self.assertEqual(state["binding_conflicts"], [])
        self.assertEqual(len(state["event_table"]), 2)
        self.assertFalse(state["decision_sufficient"])


class RecoveryAndDecisionTests(unittest.TestCase):
    def test_recovery_signature_ignores_order_but_changes_with_evidence_version(self) -> None:
        first = make_recovery_signature(
            "order",
            target_event_ids=["2", "1"],
            windows=[[31.6, 17.6], [74.3, 104.4]],
            evidence_version="v1",
        )
        reordered = make_recovery_signature(
            "ORDER",
            target_event_ids=["1", "2"],
            windows=[[74.3, 104.4], [17.6, 31.6]],
            evidence_version="v1",
        )
        updated = make_recovery_signature(
            "order",
            target_event_ids=["1", "2"],
            windows=[[17.6, 31.6], [74.3, 104.4]],
            evidence_version="v2",
        )

        self.assertEqual(first, reordered)
        self.assertNotEqual(first, updated)
        self.assertFalse(recovery_allowed(first, [first]))
        self.assertFalse(recovery_allowed(first, [{"signature": first}]))
        self.assertTrue(recovery_allowed(updated, [first]))

    def test_validated_answer_can_override_planner_but_is_never_forced(self) -> None:
        state = reduce_order(
            R10_QUESTION,
            [
                _order_event("TE001", "1", 26.7),
                _order_event("TE002", "2", 88.3),
                _order_event("TE003", "3", 113.5),
                _order_event("TE004", "4", 157.7),
            ],
        )

        decision = validate_answer("A", state)

        self.assertEqual(decision["status"], "validated")
        self.assertEqual(decision["proposed_option"], "A")
        self.assertEqual(decision["selected_option"], "B")
        self.assertTrue(decision["overrode_proposal"])
        self.assertTrue(decision["validated"])
        self.assertFalse(decision["forced"])

    def test_insufficient_answer_is_blocked_and_terminal_choice_stays_forced(self) -> None:
        state = reduce_count(
            R09_QUESTION,
            [_count_event(1, 72.0), _count_event(2, 108.0), _count_event(3, 311.0)],
        )

        blocked = validate_answer("D", state)
        forced = force_answer("D", state, reason="soft_limit_grace_exhausted")
        invalid = force_answer("A", state, reason="hard_investigation_limit")

        self.assertEqual(blocked["status"], "blocked")
        self.assertFalse(blocked["validated"])
        self.assertEqual(forced["status"], "forced")
        self.assertEqual(forced["selected_option"], "D")
        self.assertFalse(forced["validated"])
        self.assertTrue(forced["forced"])
        self.assertEqual(invalid["status"], "forced_unresolved")
        self.assertIsNone(invalid["selected_option"])
        self.assertFalse(invalid["forced"])
        self.assertEqual(invalid["viable_options"], ["C", "D"])


if __name__ == "__main__":
    unittest.main()
