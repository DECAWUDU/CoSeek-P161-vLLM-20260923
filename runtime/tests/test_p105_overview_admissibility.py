from __future__ import annotations

from copy import deepcopy
import json
import unittest

from videoseek.core.overview_admissibility import (
    BLOCKED_FEEDBACK_PREFIX,
    STATE_KEY,
    canonical_action_signature,
    format_blocked_overview_feedback,
    has_valid_overview_receipt,
    record_blocked_overview,
    record_overview_attempt,
)
from videoseek.tools.v10_format import format_v10_observation


def _payload(*, count: int = 64, observed: int | None = None, coverage: float = 1.0) -> dict:
    observed = count if observed is None else observed
    return {
        "global_summary": "A complete global map of the video.",
        "timestamp_observations": [
            {
                "timestamp_s": float(index),
                "scene_id": "S000",
                "description": f"Visible frame {index}.",
            }
            for index in range(count)
        ],
        "scene_summaries": [
            {
                "scene_id": "S000",
                "t_range": [0.0, 63.0],
                "summary": "A valid scene-level routing summary.",
                "possible_evidence": True,
            }
        ],
        "overview_expected_timestamp_count": 64,
        "overview_observed_timestamp_count": observed,
        "overview_timestamp_coverage": coverage,
        "observer_backend": "api",
    }


def _output(payload: dict) -> str:
    return format_v10_observation(payload)


class OverviewReceiptTest(unittest.TestCase):
    def test_read_only_check_does_not_create_state(self) -> None:
        memory: dict = {}
        self.assertIs(has_valid_overview_receipt(memory), False)
        self.assertNotIn(STATE_KEY, memory)

    def test_valid_full_coverage_overview_creates_first_receipt(self) -> None:
        memory: dict = {}
        attempt = record_overview_attempt(memory, _output(_payload()), True, 1)

        self.assertEqual(attempt["status"], "valid")
        self.assertIs(attempt["receipt_created"], True)
        self.assertEqual(attempt["timestamp_count"], 64)
        self.assertEqual(attempt["valid_scene_summary_count"], 1)
        self.assertEqual(attempt["expected_timestamp_count"], 64)
        self.assertEqual(attempt["observed_timestamp_count"], 64)
        self.assertEqual(attempt["timestamp_coverage"], 1.0)
        self.assertEqual(len(attempt["payload_sha256"]), 64)
        self.assertIs(has_valid_overview_receipt(memory), True)
        self.assertEqual(
            memory[STATE_KEY]["valid_receipt"]["attempt_id"],
            "OA0001",
        )

    def test_empty_malformed_and_incomplete_outputs_never_latch(self) -> None:
        cases = {
            "empty": "Overview observation failed: model response is empty.",
            "malformed": 'V10_OBSERVATION_JSON:\n```json\n{"broken":\n```',
            "empty_summary": _output({**_payload(), "global_summary": "  "}),
            "no_timestamp": _output({**_payload(), "timestamp_observations": []}),
            "no_valid_scene": _output(
                {
                    **_payload(),
                    "scene_summaries": [
                        {"summary": "", "t_range": [0.0, 10.0]},
                        {"summary": "text", "t_range": [10.0, 10.0]},
                    ],
                }
            ),
            "short_observed": _output(_payload(observed=63)),
            "short_rendered": _output(_payload(count=63, observed=64)),
            "low_coverage": _output(_payload(coverage=0.998)),
        }
        for label, output in cases.items():
            with self.subTest(label=label):
                memory: dict = {}
                attempt = record_overview_attempt(memory, output, True, 1)
                self.assertEqual(attempt["status"], "failed")
                self.assertTrue(attempt["failure_reasons"])
                self.assertIs(has_valid_overview_receipt(memory), False)
                self.assertIsNone(memory[STATE_KEY]["valid_receipt"])

    def test_non_require_all_still_needs_positive_observed_and_coverage(self) -> None:
        invalid = _payload(observed=0, coverage=0.0)
        invalid.pop("overview_expected_timestamp_count")
        valid = _payload(observed=1, coverage=0.01)
        valid.pop("overview_expected_timestamp_count")

        memory: dict = {}
        first = record_overview_attempt(memory, _output(invalid), False, 1)
        second = record_overview_attempt(memory, _output(valid), False, 2)
        self.assertEqual(first["status"], "failed")
        self.assertEqual(second["status"], "valid")
        self.assertIs(has_valid_overview_receipt(memory), True)

    def test_failure_then_success_and_first_valid_is_immutable(self) -> None:
        memory: dict = {}
        failed = record_overview_attempt(memory, "empty", True, 1)
        first_valid = record_overview_attempt(memory, _output(_payload()), True, 2)
        frozen_receipt = deepcopy(memory[STATE_KEY]["valid_receipt"])
        changed = _payload()
        changed["global_summary"] = "A different but valid later result."
        later_valid = record_overview_attempt(memory, _output(changed), True, 3)

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(first_valid["status"], "valid")
        self.assertIs(first_valid["receipt_created"], True)
        self.assertEqual(later_valid["status"], "valid")
        self.assertIs(later_valid["receipt_created"], False)
        self.assertEqual(memory[STATE_KEY]["valid_receipt"], frozen_receipt)
        self.assertEqual(memory[STATE_KEY]["attempt_count"], 3)
        self.assertEqual(memory[STATE_KEY]["failed_attempt_count"], 1)
        self.assertEqual(memory[STATE_KEY]["valid_attempt_count"], 2)

    def test_block_record_and_feedback_are_non_visual_and_deterministic(self) -> None:
        memory: dict = {}
        record_overview_attempt(memory, _output(_payload()), True, 1)
        signature = canonical_action_signature("overview", {})
        blocked = record_blocked_overview(memory, 2, signature)
        first = format_blocked_overview_feedback(memory)
        second = format_blocked_overview_feedback(memory)

        self.assertEqual(blocked["block_id"], "OB0001")
        self.assertIs(blocked["visual_observer_called"], False)
        self.assertIs(blocked["merged_as_visual_evidence"], False)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith(BLOCKED_FEEDBACK_PREFIX))
        self.assertNotIn("V10_OBSERVATION_JSON", first)
        feedback = json.loads(first[len(BLOCKED_FEEDBACK_PREFIX) :])
        self.assertEqual(feedback["status"], "rejected")
        self.assertIs(feedback["executed"], False)
        self.assertEqual(
            feedback["reason"],
            "valid_overview_already_completed",
        )

    def test_block_before_receipt_is_rejected_without_fabricating_receipt(self) -> None:
        memory: dict = {}
        with self.assertRaises(ValueError):
            record_blocked_overview(memory, 1, "signature")
        self.assertIs(has_valid_overview_receipt(memory), False)


class CanonicalActionSignatureTest(unittest.TestCase):
    def test_equivalent_reordered_windows_and_numeric_types_match(self) -> None:
        left = canonical_action_signature(
            "LOCALIZE_QWEN",
            {
                "search_windows": [[0, 10], [20.0, 30.0]],
                "localization_goal": "find the target action",
                "evidence_profile": "count_occurrence",
                "top_k": 3,
                "mode": "normal",
            },
        )
        right = canonical_action_signature(
            "localize_qwen",
            {
                "search_windows": [[20, 30], [0.0, 10.0]],
                "localization_goal": "find   the target action",
                "evidence_profile": "count_occurrence",
                "top_k": 3.0,
                "mode": "normal",
            },
        )
        self.assertEqual(left, right)

    def test_candidate_window_reordering_preserves_identity(self) -> None:
        first = {
            "candidate_id": "C1",
            "event_id": "E1",
            "t_range": [1, 2],
            "criterion": "person enters",
        }
        second = {
            "candidate_id": "C2",
            "event_id": "E2",
            "t_range": [8.0, 9.0],
            "criterion": "person exits",
        }
        left = canonical_action_signature(
            "frame_verify",
            {"candidate_windows": [first, second], "mode": "compare", "query": "test both"},
        )
        right = canonical_action_signature(
            "frame_verify",
            {"candidate_windows": [second, first], "mode": "compare", "query": "test both"},
        )
        self.assertEqual(left, right)

    def test_different_windows_goal_mode_and_identity_remain_distinct(self) -> None:
        base = {
            "search_windows": [[0, 10], [20, 30]],
            "localization_goal": "find event",
            "mode": "normal",
            "candidate_id": "C1",
        }
        signatures = {
            canonical_action_signature("localize_qwen", base),
            canonical_action_signature(
                "localize_qwen", {**base, "search_windows": [[0, 10], [40, 50]]}
            ),
            canonical_action_signature(
                "localize_qwen", {**base, "localization_goal": "find another event"}
            ),
            canonical_action_signature(
                "localize_qwen", {**base, "mode": "detail_verify"}
            ),
            canonical_action_signature(
                "localize_qwen", {**base, "candidate_id": "C2"}
            ),
        }
        self.assertEqual(len(signatures), 5)

    def test_invalid_parameters_never_collapse_to_none_none(self) -> None:
        signatures = {
            canonical_action_signature(
                "localize_qwen", {"search_windows": [[None, None]], "localization_goal": "x"}
            ),
            canonical_action_signature(
                "localize_qwen", {"search_windows": [[None]], "localization_goal": "x"}
            ),
            canonical_action_signature(
                "localize_qwen", {"search_windows": ["invalid"], "localization_goal": "x"}
            ),
            canonical_action_signature("localize_qwen", {}),
            canonical_action_signature("localize_qwen", None),
        }
        self.assertEqual(len(signatures), 5)
        for signature in signatures:
            self.assertTrue(signature.startswith("p105_action_signature_v1:"))
            self.assertNotEqual(signature, "localize_qwen:None:None")


if __name__ == "__main__":
    unittest.main()
