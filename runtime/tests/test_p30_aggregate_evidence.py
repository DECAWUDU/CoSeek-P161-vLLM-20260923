import unittest

from videoseek.tools.frame_verify import _restore_resolved_candidate_set_decision


def _payload(event_matches):
    return {
        "candidate_binding_complete": True,
        "candidate_assessments": [
            {
                "candidate_id": f"LQ{index + 1:03d}",
                "event_match": event_match,
                "option_set_conflict": False,
            }
            for index, event_match in enumerate(event_matches)
        ],
        "supports_options": [],
        "contradicts_options": [],
        "option_evidence": {},
    }


def _snapshot():
    return {
        "supports_options": ["B"],
        "contradicts_options": ["A", "C", "D"],
        "option_evidence": {
            "A": {"status": "contradict", "reason": "not four"},
            "B": {"status": "support", "reason": "five resolved candidates"},
            "C": {"status": "contradict", "reason": "not two"},
            "D": {"status": "contradict", "reason": "not six"},
        },
        "decision_sufficient": True,
    }


class P30AggregateEvidenceTests(unittest.TestCase):
    def test_preserves_unique_decision_when_candidate_set_is_resolved(self):
        payload = _payload(["direct", "direct", "direct", "direct", "direct"])
        restored = _restore_resolved_candidate_set_decision(
            payload,
            aggregate_snapshot=_snapshot(),
            candidate_coverage_complete=True,
        )
        self.assertTrue(restored)
        self.assertEqual(payload["supports_options"], ["B"])
        self.assertEqual(payload["candidate_set_aggregate_supports_options"], ["B"])
        self.assertTrue(payload["candidate_set_aggregate_decision_preserved"])

    def test_ambiguous_candidate_does_not_restore_aggregate_decision(self):
        payload = _payload(["direct", "ambiguous", "different_event"])
        restored = _restore_resolved_candidate_set_decision(
            payload,
            aggregate_snapshot=_snapshot(),
            candidate_coverage_complete=True,
        )
        self.assertFalse(restored)
        self.assertEqual(payload["supports_options"], [])
        self.assertEqual(
            payload["candidate_set_aggregate_decision_reason"],
            "ambiguous_candidate_binding",
        )

    def test_context_only_candidate_is_not_treated_as_resolved_target(self):
        payload = _payload(["direct", "context_only"])
        restored = _restore_resolved_candidate_set_decision(
            payload,
            aggregate_snapshot=_snapshot(),
            candidate_coverage_complete=True,
        )
        self.assertFalse(restored)
        self.assertEqual(
            payload["candidate_set_aggregate_decision_reason"],
            "ambiguous_candidate_binding",
        )

    def test_missing_candidate_coverage_never_restores_decision(self):
        payload = _payload(["direct", "direct"])
        restored = _restore_resolved_candidate_set_decision(
            payload,
            aggregate_snapshot=_snapshot(),
            candidate_coverage_complete=False,
        )
        self.assertFalse(restored)
        self.assertEqual(
            payload["candidate_set_aggregate_decision_reason"],
            "incomplete_candidate_coverage",
        )


if __name__ == "__main__":
    unittest.main()
