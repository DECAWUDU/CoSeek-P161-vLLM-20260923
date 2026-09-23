from __future__ import annotations

from copy import deepcopy
import json
import unittest

from config import general_config
from videoseek.core.planner_capsule import (
    CAPSULE_HEADER,
    build_decision_neutral_planner_capsule,
    build_planner_evidence_capsule,
    estimate_tokens,
)


def _evidence(
    evidence_id: str,
    *,
    fact: str,
    start: float,
    end: float,
    support: list[str] | None = None,
    contradict: list[str] | None = None,
    mode: str = "object_attribute",
) -> dict:
    return {
        "evidence_id": evidence_id,
        "source_tool": "frame_verify",
        "backend": "api",
        "scene_id": "S1",
        "window_id": f"W-{mode}",
        "candidate_id": "",
        "timestamp_s": start,
        "t_range": [start, end],
        "evidence_level": "verified",
        "description": fact,
        "observed_fact": fact,
        "supports_options": support or [],
        "contradicts_options": contradict or [],
        "detail_sufficient": True,
        "decision_sufficient": bool(support),
        "target_entity_or_event": "the queried person's sweater",
        "target_match": "matched",
        "event_match": "direct",
        "question_scope": "local_window",
        "scope_coverage": "sufficient",
        "evidence_scope": "local_window",
        "refs": {
            "source_kind": "timestamp_observation",
            "source_index": int(evidence_id[1:]),
            "parameters": {
                "start_time": start,
                "end_time": end,
                "mode": mode,
            },
        },
    }


def _memory() -> dict:
    fact = "The older man is wearing a red knitted sweater."
    evidence = [
        _evidence("E00001", fact=fact, start=5.0, end=8.0, support=["A"]),
        _evidence("E00002", fact=fact, start=5.0, end=8.0, support=["A"]),
        _evidence(
            "E00003",
            fact="A second source reports that the garment appears blue.",
            start=6.0,
            end=9.0,
            support=["B"],
            contradict=["A"],
            mode="conflict_check",
        ),
        _evidence(
            "E00004",
            fact="A nearby but independently verified scene also contains a red sweater.",
            start=6.5,
            end=9.5,
            support=["A"],
            mode="independent_scene",
        ),
    ]
    return {
        "version": "fixture",
        "structured_evidence": {
            "evidence_items": evidence,
            "option_evidence": [],
        },
        "compact_investigation_state": {
            "answer_status": {
                "status": "conflicted_verified_candidate_available",
                "option": "A",
                "support_refs": ["E00002"],
                "uncovered_options": ["C", "D"],
                "search_coverage": {
                    "overview_scene_count": 2,
                    "materially_inspected_scene_count": 1,
                },
            },
            "option_hypotheses": [
                {
                    "option": "A",
                    "verified_support": ["E00002"],
                    "verified_contradict": ["E00003"],
                    "evidence_conflict": True,
                    "support": ["E00002"],
                },
                {
                    "option": "B",
                    "verified_support": ["E00003"],
                    "verified_contradict": ["E00002"],
                    "evidence_conflict": True,
                    "support": ["E00003"],
                },
            ],
        },
        "candidate_pool": [
            {
                "candidate_id": "CAND1",
                "scene_id": "S1",
                "t_range": [4.0, 10.0],
                "status": "inspected_partial",
                "missing_detail": "Resolve the red-versus-blue conflict.",
                "possible_evidence": True,
                "query_relevance_score": 1,
            },
            {
                "candidate_id": "CAND1",
                "scene_id": "S1",
                "t_range": [4.0, 10.0],
                "status": "inspected_partial",
                "missing_detail": "Resolve the red-versus-blue conflict.",
                "possible_evidence": True,
                "query_relevance_score": 1,
            },
            {
                "candidate_id": "CAND2",
                "scene_id": "S2",
                "t_range": [30.0, 40.0],
                "status": "unvisited",
                "missing_detail": "Check a temporally distinct candidate.",
                "possible_evidence": True,
                "query_relevance_score": 1,
            },
        ],
        "scene_coverage": [
            {
                "scene_id": "S1",
                "covered_intervals": [[4.0, 7.0], [6.0, 10.0]],
                "coverage_ratio": 0.75,
                "evidence_found": True,
            },
            {
                "scene_id": "S2",
                "covered_intervals": [],
                "coverage_ratio": 0.0,
                "evidence_found": False,
            },
        ],
        "open_gaps": [
            {
                "gap_id": "G1",
                "scene_id": "S1",
                "gap": "Resolve the color conflict with a discriminating view.",
                "suggested_window": [4.0, 10.0],
                "status": "open",
            },
            {
                "gap_id": "G2",
                "scene_id": "S2",
                "gap": "This old gap is complete.",
                "suggested_window": [30.0, 40.0],
                "status": "resolved",
            },
        ],
        "observation_conflicts": [],
        "candidate_binding_memory": [],
        "tool_observations": [{"tool": "frame_verify"}],
        "timestamped_observations": [],
        "scene_memory": [],
    }


def _payload(text: str) -> dict:
    assert text.startswith(CAPSULE_HEADER)
    return json.loads(text[len(CAPSULE_HEADER) :])


class PlannerEvidenceCapsuleTest(unittest.TestCase):
    def test_default_is_control_off(self) -> None:
        self.assertIs(general_config["coseek1_planner_evidence_capsule_enabled"], False)
        self.assertIs(
            general_config["coseek1_planner_decision_neutral_capsule_enabled"],
            False,
        )
        self.assertEqual(general_config["coseek1_planner_capsule_token_budget"], 4000)

    def test_neutral_projection_removes_policy_before_fitting(self) -> None:
        memory = _memory()
        frozen = deepcopy(memory)
        result = build_decision_neutral_planner_capsule(
            memory, question="What color is the sweater?", full_memory_text="full " * 3000,
            token_budget=4000,
        )
        payload = _payload(result.text)
        self.assertEqual(memory, frozen)
        self.assertFalse(result.audit["no_backfill"])
        self.assertFalse(result.audit["invalid_references"])
        self.assertLessEqual(estimate_tokens(result.text)[0], 4000)
        for key in ("leading_option", "strongest_competitor", "uncovered_options"):
            self.assertNotIn(key, payload["answer_state"])
        self.assertFalse(any(str(row["discriminator"]).startswith("Distinguish leading option")
                             for row in payload["unresolved_obligations"]))

    def test_p100_empty_memory_and_parent_fail_open_are_unchanged(self) -> None:
        empty = {
            "structured_evidence": {"version": "initialized", "evidence_items": []},
            "tool_observations": [],
        }
        untouched = build_decision_neutral_planner_capsule(
            empty,
            question="Question",
            full_memory_text="NO OBSERVATION YET",
        )
        self.assertEqual(untouched.text, "NO OBSERVATION YET")
        self.assertIs(untouched.audit["decision_neutral_activated"], False)

        full = "ORIGINAL FULL MEMORY"
        overflow = build_decision_neutral_planner_capsule(
            _memory(),
            question="What color is the sweater?",
            full_memory_text=full,
            token_budget=8,
            max_verified=1,
            max_candidates=1,
            max_obligations=1,
            max_conflicts=1,
        )
        self.assertEqual(overflow.text, full)
        self.assertIs(overflow.audit["fail_open"], True)
        self.assertIs(overflow.audit["decision_neutral_activated"], False)

    def test_capsule_is_deterministic_bounded_read_only_and_keeps_required_refs(self) -> None:
        memory = _memory()
        frozen = deepcopy(memory)
        kwargs = {
            "question": "What color is the older man's sweater?\n(A) Red\n(B) Blue\n(C) White\n(D) Yellow",
            "full_memory_text": "full memory " * 3000,
            "token_budget": 4000,
            "max_verified": 2,
            "max_candidates": 2,
            "max_obligations": 4,
            "max_conflicts": 2,
        }
        first = build_planner_evidence_capsule(memory, **kwargs)
        second = build_planner_evidence_capsule(memory, **kwargs)

        self.assertEqual(first.text, second.text)
        self.assertEqual(first.audit["capsule_hash"], second.audit["capsule_hash"])
        self.assertEqual(memory, frozen)
        self.assertIs(first.audit["memory_mutated"], False)
        self.assertLessEqual(estimate_tokens(first.text)[0], 4000)
        self.assertIs(first.audit["fail_open"], False)
        self.assertEqual(first.audit["invalid_references"], [])
        self.assertEqual(first.audit["omitted_decisive_support_ids"], [])
        self.assertEqual(first.audit["omitted_strong_conflict_evidence_ids"], [])
        self.assertEqual(first.audit["duplicate_fact_bodies"], 0)
        self.assertEqual(
            first.text.count("The older man is wearing a red knitted sweater."), 1
        )

        payload = _payload(first.text)
        included = {
            evidence_id
            for row in payload["verified_evidence"]
            for evidence_id in row["evidence_ids"]
        }
        self.assertLessEqual({"E00002", "E00003"}, included)
        self.assertEqual(
            {row["candidate_id"] for row in payload["candidate_frontier"]},
            {"CAND1", "CAND2"},
        )
        self.assertTrue(
            all(
                "G2" not in row.get("source_gap_ids", [])
                for row in payload["unresolved_obligations"]
            )
        )

    def test_overlapping_independent_source_calls_are_not_merged_as_one_event(self) -> None:
        result = build_planner_evidence_capsule(
            _memory(),
            question="What color is the sweater?",
            full_memory_text="full " * 1000,
            max_verified=8,
        )
        payload = _payload(result.text)
        event_by_evidence = {
            evidence_id: row["event_id"]
            for row in payload["verified_evidence"]
            for evidence_id in row["evidence_ids"]
        }
        self.assertNotEqual(event_by_evidence["E00003"], event_by_evidence["E00004"])

    def test_required_groups_may_soft_overflow_item_limit_but_not_token_budget(self) -> None:
        result = build_planner_evidence_capsule(
            _memory(),
            question="What color is the sweater?",
            full_memory_text="full " * 1000,
            max_verified=1,
            token_budget=4000,
        )
        self.assertGreaterEqual(result.audit["required_group_limit_overflow"], 1)
        self.assertEqual(result.audit["omitted_decisive_support_ids"], [])
        self.assertEqual(result.audit["omitted_strong_conflict_evidence_ids"], [])
        self.assertIs(result.audit["fail_open"], False)

    def test_empty_memory_preserves_original_text_byte_for_byte(self) -> None:
        memory = {
            "structured_evidence": {"version": "initialized", "evidence_items": []},
            "event_coverage": [{"event_id": "question-derived-only"}],
            "tool_observations": [],
        }
        result = build_planner_evidence_capsule(
            memory,
            question="Question",
            full_memory_text="Question-derived state exists, but no tool has observed pixels.",
        )
        self.assertEqual(
            result.text,
            "Question-derived state exists, but no tool has observed pixels.",
        )
        self.assertIs(result.audit["activated"], False)
        self.assertIs(result.audit["memory_mutated"], False)

    def test_required_content_over_budget_fails_open_to_full_memory(self) -> None:
        full = "ORIGINAL FULL MEMORY"
        result = build_planner_evidence_capsule(
            _memory(),
            question="What color is the sweater?",
            full_memory_text=full,
            token_budget=8,
            max_verified=1,
            max_candidates=1,
            max_obligations=1,
            max_conflicts=1,
        )
        self.assertEqual(result.text, full)
        self.assertIs(result.audit["fail_open"], True)
        self.assertIs(result.audit["over_budget"], True)


if __name__ == "__main__":
    unittest.main()
