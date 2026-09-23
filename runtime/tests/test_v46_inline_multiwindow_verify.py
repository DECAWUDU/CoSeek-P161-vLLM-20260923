import json
import unittest
from unittest.mock import patch

import numpy as np

from videoseek.agent import VideoSeekAgent
from videoseek.core import Action
from videoseek.core.memory import extract_v10_payload
from videoseek.tools.frame_verify import execute_frame_verify
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


class V46InlineVerifyActionTests(unittest.TestCase):
    def _agent(self):
        agent = VideoSeekAgent.__new__(VideoSeekAgent)
        agent.question = "What is held?\n(A) cup\n(B) flower\n(C) book\n(D) phone"
        agent.config = {
            "coseek1_localize_inline_verify": True,
            "localize_inline_verify_max_windows": 3,
        }
        return agent

    def test_builds_one_multiwindow_verify_action_from_found_candidates(self):
        agent = self._agent()
        localize_action = Action(
            "localize_qwen",
            {
                "localization_goal": "Compare the held object neutrally.",
                "evidence_profile": "relation",
            },
            "localize-1",
        )
        output = format_v10_observation(
            {
                "parse_ok": True,
                "suggest_frame_verify_query": "Compare the held object neutrally.",
                "ranked_candidates": [
                    {
                        "candidate_id": "LQ001",
                        "rank": 1,
                        "recommended_verify_window": [10.0, 14.0],
                        "fine_summary": "person holds a pale object",
                        "localization_status": "found",
                        "timestamp_anchors": [11.0, 12.0],
                    },
                    {
                        "candidate_id": "LQ002",
                        "rank": 2,
                        "recommended_verify_window": [70.0, 75.0],
                        "fine_summary": "person holds flowers",
                        "localization_status": "found",
                    },
                ],
            }
        )
        action = agent._VideoSeekAgent__inline_localize_verify_action(
            localize_action=localize_action,
            localize_output=output,
            step=1,
        )
        self.assertIsNotNone(action)
        self.assertEqual(action.function_name, "frame_verify")
        self.assertEqual(len(action.parameters["candidate_windows"]), 2)
        self.assertEqual(action.parameters["windows"], [[10.0, 14.0], [70.0, 75.0]])
        self.assertEqual(action.parameters["start_time"], 10.0)
        self.assertEqual(action.parameters["end_time"], 75.0)
        self.assertEqual(
            action.parameters["candidate_windows"][0]["timestamp_anchors"],
            [11.0, 12.0],
        )

    def test_does_not_verify_ambiguous_or_empty_localization(self):
        output = format_v10_observation(
            {
                "parse_ok": True,
                "ranked_candidates": [
                    {
                        "candidate_id": "LQ001",
                        "recommended_verify_window": [10.0, 14.0],
                        "localization_status": "ambiguous",
                    }
                ],
            }
        )
        action = self._agent()._VideoSeekAgent__inline_localize_verify_action(
            localize_action=Action(
                "localize_qwen",
                {"localization_goal": "find event", "evidence_profile": "generic"},
                "localize-1",
            ),
            localize_output=output,
            step=1,
        )
        self.assertIsNone(action)

    def test_candidate_budget_can_keep_six_localized_windows(self):
        agent = self._agent()
        agent.config["localize_inline_verify_max_windows"] = 6
        output = format_v10_observation(
            {
                "parse_ok": True,
                "ranked_candidates": [
                    {
                        "candidate_id": f"LQ{index + 1:03d}",
                        "rank": index + 1,
                        "recommended_verify_window": [index * 20.0, index * 20.0 + 6.0],
                        "fine_summary": f"candidate {index + 1}",
                        "localization_status": "found",
                    }
                    for index in range(6)
                ],
            }
        )
        action = agent._VideoSeekAgent__inline_localize_verify_action(
            localize_action=Action(
                "localize_qwen",
                {"localization_goal": "compare all occurrences", "evidence_profile": "generic"},
                "localize-coverage",
            ),
            localize_output=output,
            step=1,
        )
        self.assertIsNotNone(action)
        self.assertEqual(len(action.parameters["candidate_windows"]), 6)


class V46MultiwindowFrameVerifyTests(unittest.TestCase):
    def test_one_observer_request_compares_all_candidate_windows(self):
        calls = []

        def fake_observe(
            config, *, content, tool_name, tool_mode, output_dir, return_json=False
        ):
            calls.append(content)
            return (
                json.dumps(
                    {
                        "timestamp_observations": [
                            {
                                "candidate_id": "LQ001",
                                "timestamp_s": 11.0,
                                "description": "the object is not identifiable",
                            },
                            {
                                "candidate_id": "LQ002",
                                "timestamp_s": 72.5,
                                "description": "person holds flowers",
                            },
                        ],
                        "overall_summary": "The second candidate clearly shows flowers.",
                        "target_entity_or_event": "held object",
                        "target_match": "matched",
                        "target_binding_reason": "hand and flowers are visible together",
                        "scope_coverage": "sufficient",
                        "scope_coverage_reason": "the target relation is visible",
                        "observed_fact": "The person holds flowers.",
                        "supports_options": [{"option": "B", "reason": "flowers visible"}],
                        "contradicts_options": [
                            {"option": "A"},
                            {"option": "C"},
                            {"option": "D"},
                        ],
                        "option_evidence": {
                            "A": {"status": "contradict", "reason": "not a cup"},
                            "B": {"status": "support", "reason": "flowers visible"},
                            "C": {"status": "contradict", "reason": "not a book"},
                            "D": {"status": "contradict", "reason": "not a phone"},
                        },
                        "detail_sufficient": True,
                        "missing_detail": "",
                    }
                ),
                "api",
            )

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
                },
                {
                    "vr": _VideoReaderStub(),
                    "video_path": "/tmp/video.mp4",
                    "duration": 100.0,
                    "output_dir": "/tmp",
                    "question": "What is held?\n(A) cup\n(B) flower\n(C) book\n(D) phone",
                    "query": "Compare what the person holds.",
                    "mode": "option_verify",
                    "candidate_windows": [
                        {"candidate_id": "LQ001", "t_range": [10.0, 14.0], "summary": "object"},
                        {"candidate_id": "LQ002", "t_range": [70.0, 75.0], "summary": "flowers"},
                    ],
                },
            )

        self.assertEqual(len(calls), 1)
        image_count = sum(item.get("type") == "image_url" for item in calls[0])
        self.assertEqual(image_count, 6)
        payload = extract_v10_payload(output)
        self.assertEqual(payload["verified_candidate_ids"], ["LQ001", "LQ002"])
        self.assertEqual(payload["supports_options"], ["B"])
        self.assertTrue(payload["inline_localize_verify"])

    def test_contact_sheet_layout_keeps_one_image_per_candidate(self):
        calls = []

        def fake_observe(
            config, *, content, tool_name, tool_mode, output_dir, return_json=False
        ):
            calls.append(content)
            return (
                json.dumps(
                    {
                        "timestamp_observations": [
                            {"candidate_id": "LQ001", "timestamp_s": 11.0, "description": "first action"},
                            {"candidate_id": "LQ002", "timestamp_s": 72.5, "description": "second action"},
                        ],
                        "overall_summary": "Both candidates were inspected.",
                        "target_match": "matched",
                        "supports_options": [],
                        "contradicts_options": [],
                        "option_evidence": {},
                        "detail_sufficient": True,
                        "missing_detail": "",
                    }
                ),
                "api",
            )

        timestamps = iter([[10.0, 11.0, 12.0], [70.0, 72.5, 75.0]])
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
                    "localize_inline_verify_contact_sheets": True,
                },
                {
                    "vr": _VideoReaderStub(),
                    "video_path": "/tmp/video.mp4",
                    "duration": 100.0,
                    "output_dir": "/tmp",
                    "question": "How many actions occur?\n(A) one\n(B) two\n(C) three\n(D) four",
                    "query": "Compare both action windows.",
                    "mode": "count_occurrence",
                    "candidate_windows": [
                        {"candidate_id": "LQ001", "t_range": [10.0, 12.0], "summary": "first"},
                        {"candidate_id": "LQ002", "t_range": [70.0, 75.0], "summary": "second"},
                    ],
                },
            )

        image_items = [item for item in calls[0] if item.get("type") == "image_url"]
        self.assertEqual(len(image_items), 2)
        self.assertTrue(
            all(item["image_url"]["url"].startswith("data:image/jpeg;base64,") for item in image_items)
        )
        payload = extract_v10_payload(output)
        self.assertEqual(payload["visual_layout"], "candidate_contact_sheets")
        self.assertEqual(payload["num_frames"], 6)


if __name__ == "__main__":
    unittest.main()
