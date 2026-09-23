import json
import unittest
from unittest.mock import patch

import numpy as np

from videoseek.core.memory import (
    extract_v10_payload,
    format_memory_for_prompt,
    merge_tool_observation,
)
from videoseek.tools.frame_verify import (
    _merge_candidate_sampling_timestamps,
    _normalize_candidate_assessments,
    execute_frame_verify,
)
from videoseek.tools.v10_format import format_v10_observation


class _Batch:
    def __init__(self, frames):
        self._frames = frames

    def asnumpy(self):
        return self._frames


class _VideoReaderStub:
    def __len__(self):
        return 3000

    def get_avg_fps(self):
        return 30.0

    def get_batch(self, indices):
        return _Batch(np.zeros((len(indices), 24, 32, 3), dtype=np.uint8))


def _parameters():
    return {
        "vr": _VideoReaderStub(),
        "video_path": "/tmp/video.mp4",
        "duration": 100.0,
        "output_dir": "/tmp",
        "question": "What is held?\n(A) cup\n(B) flower\n(C) book\n(D) phone",
        "query": "Compare what the person holds without assuming either candidate is correct.",
        "mode": "option_verify",
        "candidate_windows": [
            {
                "candidate_id": "LQ001",
                "t_range": [10.0, 14.0],
                "summary": "person near a cup",
                "timestamp_anchors": [11.0],
            },
            {
                "candidate_id": "LQ002",
                "t_range": [70.0, 75.0],
                "summary": "person may hold flowers",
                "timestamp_anchors": [72.5],
            },
        ],
    }


def _response(include_second=True):
    assessments = [
        {
            "candidate_id": "LQ001",
            "target_match": "mismatch",
            "observed_fact": "A cup is on the table, not held.",
            "target_binding_reason": "The hand does not contact the cup.",
            "best_timestamp_s": 11.3,
            "supports_options": [],
            "contradicts_options": ["A"],
            "option_set_conflict": False,
        }
    ]
    observations = [
        {
            "candidate_id": "LQ001",
            "timestamp_s": 11.3,
            "description": "The cup remains on the table.",
        }
    ]
    if include_second:
        assessments.append(
            {
                "candidate_id": "LQ002",
                "target_match": "matched",
                "observed_fact": "The person holds flowers in one hand.",
                "target_binding_reason": "The hand and flower stems are co-visible.",
                "best_timestamp_s": 72.7,
                "supports_options": ["B"],
                "contradicts_options": ["A", "C", "D"],
                "option_set_conflict": False,
            }
        )
        observations.append(
            {
                "candidate_id": "LQ002",
                "timestamp_s": 72.7,
                "description": "Flower stems are visibly grasped in the hand.",
            }
        )
    return json.dumps(
        {
            "timestamp_observations": observations,
            "candidate_assessments": assessments,
            "overall_summary": "The candidates show different object relations.",
            "target_entity_or_event": "held object",
            "target_match": "matched",
            "target_binding_reason": "candidate-specific assessments provided",
            "scope_coverage": "sufficient",
            "observed_fact": "Only LQ002 visibly binds the hand to flowers.",
            "supports_options": ["B"],
            "contradicts_options": ["A", "C", "D"],
            "option_evidence": {},
            "detail_sufficient": True,
            "missing_detail": "",
        }
    )


class CandidateBindingP2Tests(unittest.TestCase):
    def test_exact_timestamp_anchors_replace_sparse_base_samples(self):
        timestamps = _merge_candidate_sampling_timestamps(
            [0.0, 5.0, 10.0],
            [2.7, 6.1, 10.9],
            start=0.0,
            end=10.9,
            budget=3,
        )
        self.assertEqual(timestamps, [2.7, 6.1, 10.9])

    def test_normalizer_keeps_distinct_status_and_candidate_timestamp(self):
        candidates = _parameters()["candidate_windows"]
        rows, complete = _normalize_candidate_assessments(
            json.loads(_response())["candidate_assessments"],
            candidates=candidates,
            allowed_by_candidate={"LQ001": [10.0, 12.0], "LQ002": [70.0, 72.5, 75.0]},
        )
        self.assertTrue(complete)
        self.assertEqual([row["target_match"] for row in rows], ["mismatch", "matched"])
        self.assertEqual(rows[0]["best_timestamp_s"], 12.0)
        self.assertEqual(rows[1]["best_timestamp_s"], 72.5)

    def test_one_verify_preserves_candidate_specific_binding(self):
        calls = []

        def fake_observe(
            config, *, content, tool_name, tool_mode, output_dir, return_json=False
        ):
            self.assertTrue(return_json)
            calls.append(content)
            return _response(), "api"

        timestamps = iter([[10.0, 12.0, 14.0], [70.0, 72.5, 75.0]])
        with patch(
            "videoseek.tools.frame_verify.scene_aware_timestamps",
            side_effect=lambda **kwargs: next(timestamps),
        ), patch(
            "videoseek.tools.frame_verify.observe_content",
            side_effect=fake_observe,
        ):
            output = execute_frame_verify(
                {
                    "localize_inline_verify_max_windows": 3,
                    "localize_inline_verify_max_frames": 6,
                    "multiwindow_verify_recovery_enabled": True,
                    "grounded_candidate_binding_enabled": True,
                },
                _parameters(),
            )

        payload = extract_v10_payload(output)
        self.assertEqual(len(calls), 1)
        self.assertTrue(payload["candidate_binding_complete"])
        self.assertEqual(payload["matched_candidate_ids"], ["LQ002"])
        self.assertEqual(payload["mismatched_candidate_ids"], ["LQ001"])
        self.assertEqual(payload["candidate_search_status"], "matched_candidate_found")
        by_id = {row["candidate_id"]: row for row in payload["timestamp_observations"]}
        self.assertEqual(by_id["LQ001"]["target_match"], "mismatch")
        self.assertEqual(by_id["LQ002"]["target_match"], "matched")
        self.assertEqual(by_id["LQ002"]["timestamp_s"], 72.5)

    def test_missing_binding_assessment_recovers_only_missing_candidate(self):
        calls = []
        responses = iter([_response(include_second=False), _response(include_second=True)])

        def fake_observe(
            config, *, content, tool_name, tool_mode, output_dir, return_json=False
        ):
            self.assertTrue(return_json)
            calls.append(content)
            return next(responses), "api"

        def timestamps(**kwargs):
            start, end = float(kwargs["start_time"]), float(kwargs["end_time"])
            return [start, (start + end) / 2.0, end]

        with patch(
            "videoseek.tools.frame_verify.scene_aware_timestamps",
            side_effect=timestamps,
        ), patch(
            "videoseek.tools.frame_verify.observe_content",
            side_effect=fake_observe,
        ):
            output = execute_frame_verify(
                {
                    "localize_inline_verify_max_windows": 3,
                    "localize_inline_verify_max_frames": 6,
                    "multiwindow_verify_recovery_enabled": True,
                    "multiwindow_verify_recovery_max_calls": 1,
                    "multiwindow_verify_recovery_batch_size": 1,
                    "grounded_candidate_binding_enabled": True,
                },
                _parameters(),
            )

        payload = extract_v10_payload(output)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("LQ001 |", json.dumps(calls[1]))
        self.assertIn("LQ002 |", json.dumps(calls[1]))
        self.assertTrue(payload["candidate_binding_complete"])

    def test_memory_and_structured_evidence_keep_both_bindings(self):
        payload = json.loads(_response())
        payload.update(
            {
                "tool": "frame_verify",
                "observer_backend": "api",
                "scene_id": "multi_scene",
                "t_range": [10.0, 75.0],
                "candidate_binding_complete": True,
                "candidate_search_status": "matched_candidate_found",
            }
        )
        memory = {}
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters=_parameters(),
            output=format_v10_observation(payload),
            use_structured_evidence_state=True,
            question_context=_parameters()["question"],
        )

        bindings = memory["candidate_binding_memory"]
        self.assertEqual([item["target_match"] for item in bindings], ["mismatch", "matched"])
        evidence = memory["structured_evidence"]["evidence_items"]
        by_candidate = {item.get("candidate_id"): item for item in evidence}
        self.assertEqual(by_candidate["LQ001"]["target_match"], "mismatch")
        self.assertEqual(by_candidate["LQ002"]["target_match"], "matched")
        prompt = format_memory_for_prompt(memory, question=_parameters()["question"])
        self.assertIn("Candidate target bindings:", prompt)
        self.assertIn("LQ001 [10.0, 14.0]: target_match=mismatch", prompt)
        self.assertIn("LQ002 [70.0, 75.0]: target_match=matched", prompt)


if __name__ == "__main__":
    unittest.main()
