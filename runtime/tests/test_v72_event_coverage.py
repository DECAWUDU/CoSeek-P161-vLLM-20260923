import unittest

from videoseek.core.memory import (
    build_event_coverage,
    format_event_coverage_for_prompt,
    format_memory_for_prompt,
    init_observation_memory,
)


ORDER_QUESTION = """Arrange the following events from the video in the correct chronological order: (1)a guy sits and talks inside; (2)a man surfs on a body of water; (3)a lady spins on skates; (4)the credits of the video are shown.
(A) 1->2->3->4
(B) 2->1->3->4
(C) 4->3->2->1
(D) 3->2->1->4"""


def _evidence(eid, level, timestamp, description, tool="overview"):
    return {
        "evidence_id": eid,
        "evidence_level": level,
        "timestamp_s": timestamp,
        "description": description,
        "source_tool": tool,
        "backend": "api",
        "detail_sufficient": level == "verified",
    }


class V72EventCoverageTests(unittest.TestCase):
    def _memory(self):
        memory = init_observation_memory()
        memory["structured_evidence"]["evidence_items"] = [
            _evidence("E01", "routing", 225.8, "Indoor interview shot of a seated young man."),
            _evidence("E02", "verified", 262.1, "A man is seated for an indoor interview.", "frame_verify"),
            _evidence("E03", "routing", 243.8, "A surfer rides a wave in open water."),
            _evidence("E04", "verified", 248.5, "A surfer rides a large wave on open water.", "frame_verify"),
            _evidence("E05", "candidate", 243.6, "No people or skates are visible.", "localize_qwen"),
            _evidence("E06", "verified", 388.4, "A woman spins on skates.", "frame_verify"),
            _evidence("E07", "verified", 397.4, "No credits are shown in this window.", "frame_verify"),
            _evidence("E08", "routing", 432.6, "Yellow video credits list contributors."),
        ]
        return memory

    def test_keeps_earliest_anchor_separate_from_strongest_evidence(self):
        coverage = build_event_coverage(self._memory(), question=ORDER_QUESTION)
        self.assertEqual(len(coverage), 4)
        self.assertEqual(coverage[0]["earliest_evidence_ref"], "E01")
        self.assertEqual(coverage[0]["evidence_ref"], "E02")
        self.assertEqual(coverage[1]["earliest_evidence_ref"], "E03")
        self.assertEqual(coverage[1]["evidence_ref"], "E04")

    def test_explicit_negation_is_not_positive_event_evidence(self):
        coverage = build_event_coverage(self._memory(), question=ORDER_QUESTION)
        self.assertEqual(coverage[2]["earliest_evidence_ref"], "E06")
        self.assertEqual(coverage[3]["earliest_evidence_ref"], "E08")
        self.assertNotEqual(coverage[3]["evidence_ref"], "E07")

    def test_arrow_choices_supply_facets_when_stem_is_not_numbered(self):
        question = "Which event order is correct?\n(A) jumps -> runs -> sits\n(B) runs -> jumps -> sits\n(C) sits -> runs -> jumps\n(D) jumps -> sits -> runs"
        memory = init_observation_memory()
        memory["structured_evidence"]["evidence_items"] = [
            _evidence("E11", "routing", 1.0, "The athlete jumps."),
            _evidence("E12", "routing", 2.0, "The athlete runs."),
            _evidence("E13", "routing", 3.0, "The athlete sits."),
        ]
        coverage = build_event_coverage(memory, question=question)
        self.assertEqual([item["event"] for item in coverage], ["jumps", "runs", "sits"])

    def test_single_event_question_leaves_v46_prompt_unchanged(self):
        memory = self._memory()
        question = "What color is the bottle?\n(A) red\n(B) blue\n(C) green\n(D) black"
        self.assertEqual(build_event_coverage(memory, question=question), [])
        self.assertEqual(format_event_coverage_for_prompt(memory, question=question), "")
        prompt = format_memory_for_prompt(
            memory,
            question=question,
            include_compact_event_coverage=True,
            include_compact_planner_state=False,
            include_structured_evidence=False,
            include_timeline_view=False,
            include_object_state_view=False,
        )
        self.assertNotIn("Compact Event Coverage", prompt)

    def test_prompt_exposes_refs_levels_and_timestamps(self):
        text = format_event_coverage_for_prompt(self._memory(), question=ORDER_QUESTION)
        self.assertIn("earliest=routing@225.8s ref=E01", text)
        self.assertIn("strongest=verified@262.1s ref=E02", text)
        self.assertIn("earliest=routing@432.6s ref=E08", text)


if __name__ == "__main__":
    unittest.main()
