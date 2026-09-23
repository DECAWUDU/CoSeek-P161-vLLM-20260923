import unittest

from videoseek.agent import VideoSeekAgent
from videoseek.core import Action
from videoseek.core.memory import init_observation_memory
from videoseek.tools.v10_format import format_v10_observation


ORDER_QUESTION = """Arrange the following events from the video in the correct chronological order: (1)a guy sits and talks inside; (2)a man surfs on a body of water; (3)a lady spins on skates; (4)the credits of the video are shown.
(A) 1->2->3->4
(B) 2->1->3->4
(C) 4->3->2->1
(D) 3->2->1->4"""


def _evidence(eid, timestamp, description):
    return {
        "evidence_id": eid,
        "evidence_level": "routing",
        "timestamp_s": timestamp,
        "description": description,
        "source_tool": "overview",
        "backend": "api",
    }


class V73EventBatchVerifyTests(unittest.TestCase):
    def _agent(self, question=ORDER_QUESTION):
        agent = VideoSeekAgent.__new__(VideoSeekAgent)
        agent.question = question
        agent.duration = 665.8
        agent.config = {
            "coseek1_localize_inline_verify": True,
            "localize_inline_verify_max_windows": 4,
            "coseek1_event_coverage_inline_verify": True,
            "coseek1_event_coverage_inline_verify_max_windows": 4,
            "coseek1_event_coverage_inline_verify_radius_s": 5.0,
            "coseek1_compact_event_coverage_max_items": 8,
        }
        agent.observation_memory = init_observation_memory()
        agent.observation_memory["structured_evidence"]["evidence_items"] = [
            _evidence("E01", 225.8, "Indoor interview shot of a seated young man."),
            _evidence("E02", 243.8, "A surfer rides a wave in open water."),
            _evidence("E03", 388.4, "A woman spins on skates."),
            _evidence("E04", 432.6, "Yellow video credits list contributors."),
        ]
        return agent

    def test_batches_all_explicit_event_anchors_in_one_neutral_verify(self):
        output = format_v10_observation(
            {
                "parse_ok": True,
                "ranked_candidates": [
                    {
                        "candidate_id": "LQ001",
                        "recommended_verify_window": [383.0, 397.0],
                        "fine_summary": "woman spins on skates",
                        "localization_status": "found",
                    }
                ],
            }
        )
        action = self._agent()._VideoSeekAgent__inline_localize_verify_action(
            localize_action=Action(
                "localize_qwen",
                {"localization_goal": "find the events", "evidence_profile": "temporal_order"},
                "localize-1",
            ),
            localize_output=output,
            step=1,
        )
        self.assertIsNotNone(action)
        self.assertEqual(action.parameters["mode"], "temporal_strip")
        self.assertTrue(action.parameters["inline_event_coverage"])
        self.assertEqual(
            [item["candidate_id"] for item in action.parameters["candidate_windows"]],
            ["LQ001"],
        )
        self.assertEqual(action.parameters["windows"], [[383.0, 397.0]])
        self.assertEqual(action.parameters["event_coverage_max_windows"], 4)
        self.assertIn("neutrally identify", action.parameters["query"])
        self.assertIn("do not assume", action.parameters["query"])

    def test_single_event_question_preserves_v46_localize_candidates(self):
        question = "What color is the bottle?\n(A) red\n(B) blue\n(C) green\n(D) black"
        agent = self._agent(question)
        output = format_v10_observation(
            {
                "parse_ok": True,
                "ranked_candidates": [
                    {
                        "candidate_id": "LQ001",
                        "recommended_verify_window": [10.0, 14.0],
                        "fine_summary": "bottle on table",
                        "localization_status": "found",
                    }
                ],
            }
        )
        action = agent._VideoSeekAgent__inline_localize_verify_action(
            localize_action=Action(
                "localize_qwen",
                {"localization_goal": "find bottle", "evidence_profile": "attribute"},
                "localize-1",
            ),
            localize_output=output,
            step=1,
        )
        self.assertFalse(action.parameters["inline_event_coverage"])
        self.assertEqual(action.parameters["windows"], [[10.0, 14.0]])
        self.assertEqual(action.parameters["query"], "find bottle")


if __name__ == "__main__":
    unittest.main()
