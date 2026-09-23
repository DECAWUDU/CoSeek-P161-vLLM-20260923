from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from videoseek.agent import VideoSeekAgent
from videoseek.core.evidence_state import update_structured_evidence_state
from videoseek.core.planner_capsule import (
    CAPSULE_HEADER,
    build_decision_neutral_planner_capsule,
)
from videoseek.tools.frame_verify import _normalize_packet_detail_request


def _capsule_payload(memory: dict, question: str) -> tuple[dict, dict]:
    result = build_decision_neutral_planner_capsule(
        memory,
        question=question,
        full_memory_text="FULL MEMORY",
        token_budget=4000,
        max_verified=8,
        max_candidates=8,
        max_obligations=4,
        max_conflicts=2,
    )
    if not result.text.startswith(CAPSULE_HEADER):
        raise AssertionError(result.audit)
    return json.loads(result.text[len(CAPSULE_HEADER) :]), result.audit


class P125VerifiedLedgerHandoffTest(unittest.TestCase):
    def test_agent_records_p125_switches_in_online_memory(self) -> None:
        agent = VideoSeekAgent.__new__(VideoSeekAgent)
        agent.config = {
            "p125_per_candidate_direct_evidence_enabled": True,
            "p125_ledger_capsule_bridge_enabled": True,
            "p127_strict_per_anchor_evidence_enabled": True,
            "p127_ambiguous_mandatory_detail_recovery_enabled": True,
        }
        agent.observation_memory = {}
        agent._VideoSeekAgent__record_v34_runtime_config()
        self.assertIs(
            agent.observation_memory["runtime_config"][
                "p125_per_candidate_direct_evidence_enabled"
            ],
            True,
        )
        self.assertIs(
            agent.observation_memory["runtime_config"][
                "p125_ledger_capsule_bridge_enabled"
            ],
            True,
        )
        self.assertIs(
            agent.observation_memory["runtime_config"][
                "p127_strict_per_anchor_evidence_enabled"
            ],
            True,
        )
        self.assertIs(
            agent.observation_memory["runtime_config"][
                "p127_ambiguous_mandatory_detail_recovery_enabled"
            ],
            True,
        )

    def test_p127_ambiguous_mandatory_candidate_forces_detail_retry(self) -> None:
        candidates = [
            {
                "candidate_id": "LQ002",
                "t_range": [373.9, 395.6],
                "mandatory_positive": True,
                "timestamp_anchors": [377.7, 387.9],
            }
        ]
        payload = {
            "detail_sufficient": True,
            "evidence_need": "sufficient",
            "candidate_assessments": [
                {
                    "candidate_id": "LQ002",
                    "target_match": "ambiguous",
                    "event_match": "context_only",
                    "best_timestamp_s": 387.9,
                }
            ],
        }
        self.assertIsNone(
            _normalize_packet_detail_request(payload, candidates=candidates)
        )
        request = _normalize_packet_detail_request(
            payload,
            candidates=candidates,
            allow_evidence_need=True,
            allow_ambiguous_mandatory=True,
        )
        self.assertEqual(request["candidate_id"], "LQ002")
        self.assertEqual(request["timestamp_s"], 387.9)
        self.assertIn("highest available resolution", request["query"])

    def test_exact_r08_p124_trace_projects_two_canonical_events(self) -> None:
        experiments = Path(__file__).resolve().parents[3]
        trace = (
            experiments
            / "p124_evidence_preserving_candidate_20260903"
            / "smoke"
            / "runs"
            / "R08_count_233_p124_optimized"
            / "trajectory.json"
        )
        trajectory = json.loads(trace.read_text())
        memory = deepcopy(trajectory["memory"])
        memory.setdefault("runtime_config", {}).update(
            {
                "p125_ledger_capsule_bridge_enabled": True,
                "p125_per_candidate_direct_evidence_enabled": True,
            }
        )

        payload, audit = _capsule_payload(memory, trajectory["question"])
        verified = payload["verified_evidence"]

        self.assertEqual(len(verified), 2)
        self.assertEqual(audit["ledger_bridge_event_count"], 2)
        self.assertEqual(audit["invalid_references"], [])
        self.assertEqual(
            {row["event_identity_basis"] for row in verified},
            {"temporal_evidence_ledger"},
        )
        self.assertIn(["E00104"], [row["evidence_ids"] for row in verified])
        self.assertIn(
            ["E00106", "E00108"],
            [row["evidence_ids"] for row in verified],
        )
        projected_ids = {
            evidence_id
            for row in verified
            for evidence_id in row["evidence_ids"]
        }
        self.assertTrue({"E00105", "E00107", "E00109"}.isdisjoint(projected_ids))
        facts = " ".join(row["text"] for row in payload["fact_catalog"])
        self.assertIn("Red-and-yellow paraglider", facts)
        self.assertIn("small white canopy", facts)

    @staticmethod
    def _frame_verify_memory(enabled: bool, strict: bool = False) -> dict:
        return {
            "runtime_config": {
                "p125_per_candidate_direct_evidence_enabled": enabled,
                "p127_strict_per_anchor_evidence_enabled": strict,
            }
        }

    @staticmethod
    def _frame_verify_payload(detail_sufficient: bool = False) -> dict:
        return {
            "candidate_binding_complete": True,
            "detail_sufficient": detail_sufficient,
            "candidate_assessments": [
                {
                    "candidate_id": "LQ002",
                    "assessment_present": True,
                    "target_match": "matched",
                    "event_match": "direct",
                    "best_timestamp_s": 10.0,
                    "event_span": [9.0, 11.0],
                    "observed_fact": "A paraglider is directly visible.",
                    "supports_options": ["A"],
                    "contradicts_options": ["B"],
                    "option_set_conflict": False,
                }
            ],
            "timestamp_observations": [
                {
                    "candidate_id": "LQ002",
                    "timestamp_s": 10.0,
                    "description": "A paraglider is directly visible.",
                },
                {
                    "candidate_id": "LQ002",
                    "timestamp_s": 11.0,
                    "description": "An indoor audience is visible.",
                },
            ],
            "observer_backend": "api",
            "parse_ok": True,
        }

    def _compile_frame_verify(
        self,
        enabled: bool,
        *,
        strict: bool = False,
        detail_sufficient: bool = False,
    ) -> list[dict]:
        memory = self._frame_verify_memory(enabled, strict)
        update_structured_evidence_state(
            memory,
            tool_name="frame_verify",
            parameters={
                "candidate_windows": [
                    {"candidate_id": "LQ002", "t_range": [8.0, 12.0]}
                ]
            },
            payload=self._frame_verify_payload(detail_sufficient),
        )
        return memory["structured_evidence"]["evidence_items"]

    def test_only_explicit_best_direct_anchor_is_promoted(self) -> None:
        rows = self._compile_frame_verify(True)
        self.assertEqual(
            [(row["timestamp_s"], row["evidence_level"]) for row in rows],
            [(10.0, "verified"), (11.0, "candidate")],
        )

    def test_flag_off_preserves_p124_batch_level_behavior(self) -> None:
        rows = self._compile_frame_verify(False)
        self.assertEqual(
            [(row["timestamp_s"], row["evidence_level"]) for row in rows],
            [(10.0, "candidate"), (11.0, "candidate")],
        )

    def test_p127_strict_anchor_blocks_context_in_sufficient_batch(self) -> None:
        rows = self._compile_frame_verify(
            True,
            strict=True,
            detail_sufficient=True,
        )
        self.assertEqual(
            [(row["timestamp_s"], row["evidence_level"]) for row in rows],
            [(10.0, "verified"), (11.0, "candidate")],
        )


if __name__ == "__main__":
    unittest.main()
