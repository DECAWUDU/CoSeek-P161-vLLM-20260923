from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from videoseek.core.memory import extract_v10_payload
from videoseek.tools.frame_verify import (
    _apply_p130_local_receipt_contract,
    _merge_anchor_audit_into_candidate_assessments,
    _multiwindow_verifier_instruction,
    _normalize_anchor_assessments,
    _normalize_candidate_assessments,
    _p130_global_context,
    execute_frame_verify,
)
from videoseek.tools.v10_format import format_v10_observation


COUNT_QUESTION = """Throughout this video, what is the total count of occurrences for the scene featuring cooking sausages?
(A) 1
(B) 0
(C) 4
(D) 3"""

ORDER_QUESTION = """Arrange the following events from the video in the correct chronological order: (1)a child balances with an adult; (2)people walk on a bridge; (3)a man performs above a cold river; (4)a boy competes while spectators stand behind a fence.
(A) 2->1->3->4
(B) 1->2->3->4
(C) 4->3->2->1
(D) 3->2->1->4"""


def _candidate() -> list[dict]:
    return [{"candidate_id": "LQ001", "t_range": [10.0, 20.0]}]


class _Batch:
    def __init__(self, frames):
        self._frames = frames

    def asnumpy(self):
        return self._frames


class _VideoReader:
    def __len__(self):
        return 3000

    def get_avg_fps(self):
        return 30.0

    def get_batch(self, indices):
        return _Batch(np.zeros((len(indices), 24, 32, 3), dtype=np.uint8))


class P130FrameVerifyContractTests(unittest.TestCase):
    def test_contract_is_gated_by_flag_and_global_mode(self) -> None:
        self.assertFalse(
            _p130_global_context(
                {"p130_minimal_global_fsm_enabled": False},
                {"question": ORDER_QUESTION},
            )["active"]
        )
        self.assertFalse(
            _p130_global_context(
                {"p130_minimal_global_fsm_enabled": True},
                {"question": "What color is the hat?\n(A) red\n(B) blue"},
            )["active"]
        )
        self.assertFalse(
            _p130_global_context(
                {"p130_minimal_global_fsm_enabled": True},
                {
                    "question": (
                        "How many picture frames were on the wall?\n"
                        "(A) 1\n(B) 2\n(C) 3\n(D) 4"
                    )
                },
            )["active"]
        )
        self.assertTrue(
            _p130_global_context(
                {"p130_minimal_global_fsm_enabled": True},
                {"question": COUNT_QUESTION},
            )["active"]
        )

    def test_order_normalizer_accepts_one_direct_whitelisted_alias(self) -> None:
        rows, complete = _normalize_candidate_assessments(
            [
                {
                    "candidate_id": "LQ001",
                    "target_match": "matched",
                    "event_match": "direct",
                    "event_index": "event 2",
                    "candidate_event_ids": ["2"],
                    "best_timestamp_s": 15.0,
                    "observed_fact": "People visibly walk across the bridge.",
                }
            ],
            candidates=_candidate(),
            allowed_by_candidate={"LQ001": [15.0]},
            event_binding_enabled=True,
            required_event_ids={"1", "2", "3", "4"},
        )

        self.assertTrue(complete)
        self.assertEqual(rows[0]["target_event_id"], "2")
        self.assertEqual(rows[0]["event_index"], "2")
        self.assertEqual(rows[0]["matched_event_ids"], ["2"])
        self.assertEqual(rows[0]["binding_confidence"], "direct")
        self.assertFalse(rows[0]["event_id_conflict"])

    def test_non_direct_multiple_or_invalid_ids_never_become_bindings(self) -> None:
        rows, _ = _normalize_candidate_assessments(
            [
                {
                    "candidate_id": "LQ001",
                    "target_match": "ambiguous",
                    "event_match": "context_only",
                    "target_event_id": "2",
                    "candidate_event_ids": ["2", "9"],
                }
            ],
            candidates=_candidate(),
            event_binding_enabled=True,
            required_event_ids={"1", "2", "3", "4"},
        )

        self.assertEqual(rows[0]["candidate_event_ids"], ["2"])
        self.assertEqual(rows[0]["target_event_id"], "")
        self.assertEqual(rows[0]["matched_event_ids"], [])
        self.assertEqual(rows[0]["binding_confidence"], "ambiguous")
        self.assertTrue(rows[0]["event_id_conflict"])
        self.assertEqual(rows[0]["invalid_event_ids"], ["9"])

    def test_conflicting_direct_anchor_clears_candidate_exact_binding(self) -> None:
        assessments, _ = _normalize_candidate_assessments(
            [
                {
                    "candidate_id": "LQ001",
                    "target_match": "matched",
                    "event_match": "direct",
                    "target_event_id": "1",
                    "candidate_event_ids": ["1"],
                }
            ],
            candidates=_candidate(),
            event_binding_enabled=True,
            required_event_ids={"1", "2", "3", "4"},
        )
        anchors, complete = _normalize_anchor_assessments(
            [
                {
                    "candidate_id": "LQ001",
                    "timestamp_s": 15.0,
                    "target_match": "matched",
                    "event_match": "direct",
                    "target_event_id": "2",
                    "candidate_event_ids": ["2"],
                    "observed_fact": "The anchor resembles event two.",
                }
            ],
            allowed_by_candidate={"LQ001": [15.0]},
            event_binding_enabled=True,
            required_event_ids={"1", "2", "3", "4"},
        )
        _merge_anchor_audit_into_candidate_assessments(
            assessments,
            anchors,
            event_binding_enabled=True,
            required_event_ids={"1", "2", "3", "4"},
        )

        self.assertTrue(complete)
        self.assertEqual(set(assessments[0]["candidate_event_ids"]), {"1", "2"})
        self.assertEqual(assessments[0]["matched_event_ids"], [])
        self.assertTrue(assessments[0]["event_id_conflict"])

    def test_global_receipt_isolation_preserves_local_audit_only(self) -> None:
        context = _p130_global_context(
            {"p130_minimal_global_fsm_enabled": True},
            {"question": COUNT_QUESTION},
        )
        payload = {
            "supports_options": ["D"],
            "contradicts_options": ["A"],
            "option_evidence": {"D": {"status": "support"}},
            "decision_sufficient": True,
            "scope_coverage": "sufficient",
            "candidate_set_aggregate_supports_options": ["D"],
            "candidate_set_aggregate_decision_preserved": True,
            "candidate_assessments": [
                {
                    "candidate_id": "LQ001",
                    "event_match": "direct",
                    "supports_options": ["D"],
                    "contradicts_options": ["A"],
                    "option_evidence": {"D": {"status": "support"}},
                }
            ],
        }

        _apply_p130_local_receipt_contract(payload, context=context)

        self.assertEqual(payload["supports_options"], [])
        self.assertEqual(payload["contradicts_options"], [])
        self.assertEqual(payload["option_evidence"], {})
        self.assertEqual(payload["local_supports_options"], ["D"])
        self.assertEqual(payload["local_option_evidence"]["D"]["status"], "support")
        self.assertFalse(payload["decision_sufficient"])
        self.assertTrue(payload["local_decision_sufficient"])
        self.assertEqual(payload["scope_coverage"], "partial")
        self.assertEqual(payload["local_scope_coverage"], "sufficient")
        self.assertEqual(
            payload["candidate_assessments"][0]["supports_options"], []
        )
        self.assertEqual(
            payload["candidate_assessments"][0]["local_supports_options"], ["D"]
        )

    def test_order_prompt_withholds_choices_and_lists_event_catalog(self) -> None:
        context = _p130_global_context(
            {"p130_minimal_global_fsm_enabled": True},
            {"question": ORDER_QUESTION},
        )
        prompt = _multiwindow_verifier_instruction(
            question=ORDER_QUESTION,
            query="Identify the visible local event.",
            candidate_text="- LQ001 [10, 20]",
            p130_context=context,
        )

        self.assertIn("event_id=1", prompt)
        self.assertIn("event_id=4", prompt)
        self.assertIn("answer choices intentionally withheld", prompt)
        self.assertNotIn("(B) 1->2->3->4", prompt)

    def test_direct_wrapper_applies_contract_before_return(self) -> None:
        raw = format_v10_observation(
            {
                "tool": "focus",
                "target_match": "matched",
                "target_event_match": "direct",
                "target_event_id": "2",
                "candidate_event_ids": ["2"],
                "supports_options": ["B"],
                "contradicts_options": ["A", "C", "D"],
                "option_evidence": {"B": {"status": "support"}},
                "decision_sufficient": True,
                "scope_coverage": "sufficient",
            }
        )
        with patch(
            "videoseek.tools.frame_verify.execute_focus", return_value=raw
        ) as execute:
            output = execute_frame_verify(
                {"p130_minimal_global_fsm_enabled": True},
                {"question": ORDER_QUESTION, "query": "Inspect this local event."},
            )

        payload = extract_v10_payload(output)
        self.assertFalse(execute.call_args.args[0]["observer_backend"] == "local_qwen")
        self.assertFalse(execute.call_args.args[1]["force_option_evidence"])
        self.assertEqual(payload["matched_event_ids"], ["2"])
        self.assertEqual(payload["supports_options"], [])
        self.assertEqual(payload["local_supports_options"], ["B"])
        self.assertFalse(payload["decision_sufficient"])
        self.assertEqual(payload["scope_coverage"], "partial")

    def test_multiwindow_path_emits_typed_local_receipts(self) -> None:
        captured_content = []

        def fake_observe(
            config, *, content, tool_name, tool_mode, output_dir, return_json=False
        ):
            captured_content.extend(content)
            return (
                """{
                  "timestamp_observations": [{"candidate_id": "LQ001", "timestamp_s": 15.0, "description": "People walk on a bridge."}],
                  "candidate_assessments": [{"candidate_id": "LQ001", "target_match": "matched", "event_match": "direct", "target_event_id": "2", "candidate_event_ids": ["2"], "observed_fact": "People visibly walk on a bridge.", "best_timestamp_s": 15.0, "event_span": [12.0, 18.0], "supports_options": ["B"], "contradicts_options": ["A", "C", "D"]}],
                  "target_match": "matched",
                  "scope_coverage": "sufficient",
                  "supports_options": ["B"],
                  "contradicts_options": ["A", "C", "D"],
                  "option_evidence": {"B": {"status": "support"}},
                  "decision_sufficient": true,
                  "detail_sufficient": true
                }""",
                "api",
            )

        parameters = {
            "vr": _VideoReader(),
            "video_path": "/tmp/video.mp4",
            "duration": 100.0,
            "output_dir": "/tmp",
            "question": ORDER_QUESTION,
            "query": "Identify the local event shown.",
            "mode": "timeline",
            "candidate_windows": [
                {
                    "candidate_id": "LQ001",
                    "t_range": [10.0, 20.0],
                    "summary": "bridge scene",
                    "timestamp_anchors": [15.0],
                }
            ],
        }
        with patch(
            "videoseek.tools.frame_verify.scene_aware_timestamps",
            return_value=[10.0, 15.0, 20.0],
        ), patch(
            "videoseek.tools.frame_verify.observe_content", side_effect=fake_observe
        ):
            payload = extract_v10_payload(
                execute_frame_verify(
                    {
                        "p130_minimal_global_fsm_enabled": True,
                        "localize_inline_verify_max_windows": 2,
                        "localize_inline_verify_max_frames": 3,
                        "multiwindow_verify_recovery_enabled": False,
                    },
                    parameters,
                )
            )

        row = payload["candidate_assessments"][0]
        self.assertEqual(row["matched_event_ids"], ["2"])
        self.assertEqual(row["supports_options"], [])
        self.assertEqual(row["local_supports_options"], ["B"])
        self.assertEqual(
            payload["timestamp_observations"][0]["matched_event_ids"], ["2"]
        )
        self.assertEqual(payload["supports_options"], [])
        self.assertFalse(payload["decision_sufficient"])
        self.assertEqual(payload["scope_coverage"], "partial")
        prompt = "\n".join(
            str(item.get("text") or "")
            for item in captured_content
            if item.get("type") == "text"
        )
        self.assertIn("event_id=2", prompt)
        self.assertNotIn("(B) 1->2->3->4", prompt)

    def test_flag_off_keeps_p128_option_contract(self) -> None:
        raw = format_v10_observation(
            {
                "supports_options": ["B"],
                "decision_sufficient": True,
                "scope_coverage": "sufficient",
            }
        )
        with patch(
            "videoseek.tools.frame_verify.execute_focus", return_value=raw
        ) as execute:
            payload = extract_v10_payload(
                execute_frame_verify(
                    {"p130_minimal_global_fsm_enabled": False},
                    {"question": ORDER_QUESTION, "query": "Inspect this event."},
                )
            )

        self.assertTrue(execute.call_args.args[1]["force_option_evidence"])
        self.assertEqual(payload["supports_options"], ["B"])
        self.assertTrue(payload["decision_sufficient"])
        self.assertEqual(payload["scope_coverage"], "sufficient")


if __name__ == "__main__":
    unittest.main()
