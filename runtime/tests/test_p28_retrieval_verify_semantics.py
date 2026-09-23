import unittest
from unittest.mock import patch

from videoseek.agent import VideoSeekAgent
from videoseek.core import Action
from videoseek.tools.frame_verify import _multiwindow_verifier_instruction
from videoseek.tools.v10_format import format_v10_observation


QUESTION = "What is held?\n(A) cup\n(B) flower\n(C) book\n(D) phone"
RETRIEVAL_GOAL = "Only accept a clearly visible white flower held in the right hand."


def _localize_output():
    return format_v10_observation(
        {
            "parse_ok": True,
            "suggest_frame_verify_query": RETRIEVAL_GOAL,
            "ranked_candidates": [
                {
                    "candidate_id": "LQ001",
                    "rank": 1,
                    "recommended_verify_window": [10.0, 14.0],
                    "fine_summary": "person holds a pale object",
                    "localization_status": "found",
                    "timestamp_anchors": [11.0, 12.0],
                }
            ],
        }
    )


def _agent(*, semantic_decoupling_enabled):
    agent = VideoSeekAgent.__new__(VideoSeekAgent)
    agent.question = QUESTION
    agent.config = {
        "coseek1_localize_inline_verify": True,
        "localize_inline_verify_max_windows": 3,
        "coseek1_retrieval_verify_semantic_decoupling_enabled": (
            semantic_decoupling_enabled
        ),
    }
    return agent


class P28RetrievalVerifySemanticsTests(unittest.TestCase):
    def _build_action(self, *, enabled):
        return _agent(
            semantic_decoupling_enabled=enabled
        )._VideoSeekAgent__inline_localize_verify_action(
            localize_action=Action(
                "localize_qwen",
                {
                    "localization_goal": RETRIEVAL_GOAL,
                    "evidence_profile": "relation",
                },
                "localize-1",
            ),
            localize_output=_localize_output(),
            step=1,
        )

    def test_disabled_switch_preserves_p27_query_semantics(self):
        action = self._build_action(enabled=False)
        self.assertEqual(action.parameters["query"], RETRIEVAL_GOAL)
        self.assertNotIn("retrieval_hint", action.parameters)
        self.assertNotIn("verification_objective_source", action.parameters)

    def test_enabled_switch_keeps_retrieval_goal_as_untrusted_hint(self):
        action = self._build_action(enabled=True)
        self.assertEqual(action.parameters["query"], QUESTION)
        self.assertEqual(action.parameters["retrieval_hint"], RETRIEVAL_GOAL)
        self.assertEqual(
            action.parameters["verification_objective_source"], "original_question"
        )

    def test_event_coverage_labels_remain_retrieval_only(self):
        event_candidates = [
            {
                "candidate_id": "EV001",
                "rank": 1,
                "t_range": [10.0, 14.0],
                "summary": "first event candidate",
                "event": "person starts cooking",
                "timestamp_anchors": [12.0],
            },
            {
                "candidate_id": "EV002",
                "rank": 2,
                "t_range": [40.0, 44.0],
                "summary": "second event candidate",
                "event": "person leaves the room",
                "timestamp_anchors": [42.0],
            },
        ]
        agent = _agent(semantic_decoupling_enabled=True)
        with patch.object(
            VideoSeekAgent,
            "_VideoSeekAgent__event_coverage_inline_candidates",
            return_value=event_candidates,
        ):
            action = agent._VideoSeekAgent__inline_localize_verify_action(
                localize_action=Action(
                    "localize_qwen",
                    {
                        "localization_goal": "find all events",
                        "evidence_profile": "temporal_order",
                    },
                    "localize-events",
                ),
                localize_output=_localize_output(),
                step=1,
            )
        self.assertEqual(action.parameters["query"], QUESTION)
        self.assertIn("person starts cooking", action.parameters["retrieval_hint"])
        self.assertIn("person leaves the room", action.parameters["retrieval_hint"])
        self.assertTrue(action.parameters["inline_event_coverage"])

    def test_semantic_contract_prompt_hides_retrieval_goal(self):
        prompt = _multiwindow_verifier_instruction(
            question=QUESTION,
            query=QUESTION,
            retrieval_hint=RETRIEVAL_GOAL,
            candidate_text="- LQ001 [10.0, 14.0]: person holds a pale object",
            semantic_decoupling_enabled=True,
        )
        self.assertEqual(prompt.count(QUESTION), 1)
        self.assertIn("Original multiple-choice question (authoritative)", prompt)
        self.assertIn("retrieval objective is intentionally not", prompt)
        self.assertNotIn(RETRIEVAL_GOAL, prompt)
        self.assertNotIn("Verification objective:\n" + RETRIEVAL_GOAL, prompt)


if __name__ == "__main__":
    unittest.main()
