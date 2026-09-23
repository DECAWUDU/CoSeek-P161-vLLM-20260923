import unittest

from videoseek.core.memory import (
    build_compact_investigation_state,
    format_compact_investigation_state_for_prompt,
    init_observation_memory,
    merge_tool_observation,
)
from videoseek.tools.v10_format import format_v10_observation
from videoseek.tools.focus import _normalize_scope_coverage, _question_scope_hint


def observation(payload):
    return format_v10_observation(
        payload,
        fallback_text=payload.get("overall_summary", ""),
    )


class V36CandidatePoolTests(unittest.TestCase):
    question = (
        "What kind of animals are being kept in the video?\n"
        "(A) Sheep\n(B) Dog\n(C) Cat\n(D) Pig"
    )

    def _overview_memory(self):
        memory = init_observation_memory()
        merge_tool_observation(
            memory,
            tool_name="overview",
            parameters={},
            output=observation(
                {
                    "observer_backend": "api",
                    "timestamp_observations": [
                        {
                            "timestamp_s": 10.0,
                            "scene_id": "dog_closeup",
                            "description": "A dog is visible beside a dark kennel.",
                            "event_tags": ["dog", "kennel"],
                            "needs_focus": "Determine whether this is the animal being kept.",
                        },
                        {
                            "timestamp_s": 90.0,
                            "scene_id": "animal_pen",
                            "description": "Several sheep stand inside a fenced animal pen.",
                            "event_tags": ["sheep", "animal_pen"],
                            "needs_focus": "Verify the animals enclosed in the pen.",
                        },
                    ],
                    "scene_summaries": [
                        {
                            "scene_id": "dog_closeup",
                            "t_range": [0.0, 20.0],
                            "summary": "A dog appears near a kennel.",
                            "possible_evidence": True,
                            "suggest_focus_windows": [[6.0, 14.0]],
                            "missing_detail": "Whether the visible dog is the target animal.",
                        }
                    ],
                }
            ),
        )
        return memory

    def test_question_scope_comes_from_original_question(self):
        self.assertEqual(_question_scope_hint(self.question), "global_video")
        self.assertEqual(
            _question_scope_hint("What did she do after entering the room?"),
            "event_instance",
        )
        self.assertEqual(
            _question_scope_hint("What color is the bottle on the table?"),
            "local_window",
        )

    def test_short_local_window_is_partial_for_global_question(self):
        coverage, reason = _normalize_scope_coverage(
            question_scope="global_video",
            reported_coverage="sufficient",
            reported_reason="The dog is clearly visible.",
            start_time=723.0,
            end_time=731.0,
            duration=740.0,
        )
        self.assertEqual(coverage, "partial")
        self.assertIn("other Candidate Pool locations", reason)
        self.assertIn("dog is clearly visible", reason)

    def test_timestamp_cue_survives_missing_scene_summary(self):
        memory = self._overview_memory()
        compact = build_compact_investigation_state(
            memory,
            question=self.question,
            separate_routing_candidates=True,
            query_relevant_retention=True,
            persistent_candidate_pool=True,
            persistent_candidate_pool_max_items=20,
            persistent_candidate_pool_timestamp_radius_s=8.0,
        )

        pool = compact["overview_candidates"]
        sheep = [item for item in pool if item.get("scene_id") == "animal_pen"]
        self.assertEqual(len(sheep), 1)
        self.assertEqual(sheep[0]["source"], "overview_timestamp")
        self.assertEqual(sheep[0]["anchor_timestamps"], [90.0])
        self.assertEqual(sheep[0]["status"], "unvisited")
        self.assertTrue(
            any(
                target.get("scene_id") == "animal_pen"
                for target in compact["next_search_targets"]
            )
        )

    def test_candidate_identity_is_stable_and_inspection_updates_status(self):
        memory = self._overview_memory()
        first = build_compact_investigation_state(
            memory,
            question=self.question,
            separate_routing_candidates=True,
            persistent_candidate_pool=True,
        )
        first_ids = {
            item["scene_id"]: item["candidate_id"]
            for item in first["overview_candidates"]
        }

        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 6.0, "end_time": 14.0},
            output=observation(
                {
                    "observer_backend": "api",
                    "scene_id": "dog_closeup",
                    "t_range": [6.0, 14.0],
                    "overall_summary": "A dog is visible in this local window.",
                    "observed_fact": "A dog stands beside a kennel.",
                    "target_entity_or_event": "animals being kept at the location",
                    "target_match": "ambiguous",
                    "target_binding_reason": (
                        "The dog is visible, but this window does not establish that it "
                        "represents all animals being kept."
                    ),
                    "question_scope": "global_video",
                    "scope_coverage": "partial",
                    "scope_coverage_reason": (
                        "The local kennel window does not cover other animal locations."
                    ),
                    "detail_sufficient": True,
                    "supports_options": ["B"],
                    "contradicts_options": [],
                    "scene_summaries": [
                        {
                            "scene_id": "dog_closeup",
                            "t_range": [6.0, 14.0],
                            "summary": "A dog is locally visible beside a kennel.",
                            "possible_evidence": True,
                        }
                    ],
                }
            ),
            use_structured_evidence_state=True,
            question_context=self.question,
            scope_aware_evidence_memory=True,
        )
        second = build_compact_investigation_state(
            memory,
            question=self.question,
            separate_routing_candidates=True,
            persistent_candidate_pool=True,
        )
        second_ids = {
            item["scene_id"]: item["candidate_id"]
            for item in second["overview_candidates"]
        }

        self.assertEqual(first_ids, second_ids)
        dog = next(
            item
            for item in second["overview_candidates"]
            if item.get("scene_id") == "dog_closeup"
        )
        self.assertEqual(dog["status"], "verified")
        self.assertEqual(dog["target_bindings"][0]["target_match"], "ambiguous")
        self.assertEqual(dog["target_bindings"][0]["scope_coverage"], "partial")
        self.assertEqual(second["target_bindings"][0]["target_match"], "ambiguous")

        prompt = format_compact_investigation_state_for_prompt(
            memory,
            question=self.question,
            separate_routing_candidates=True,
            persistent_candidate_pool=True,
        )
        self.assertIn("Persistent Candidate Pool", prompt)
        self.assertIn("match=ambiguous", prompt)
        self.assertIn("scope_coverage=partial", prompt)
        self.assertIn("animal_pen", prompt)


if __name__ == "__main__":
    unittest.main()
