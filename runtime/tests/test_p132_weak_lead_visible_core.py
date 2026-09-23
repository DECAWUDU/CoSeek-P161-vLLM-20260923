from __future__ import annotations

import json
import unittest

from videoseek.core.minimal_global_fsm import parse_global_question
from videoseek.core.p130_runtime import (
    _p132_count_target_tokens,
    _p132_count_negative,
    init_p130_state,
    initial_count_verify_spec,
)


R09 = """Throughout this video, what is the total count of occurrences for the scene featuring the 'cooking sausages' action
(A) 1
(B) 0
(C) 4
(D) 3
Please directly answer with the best option's letter from the given choices directly (A, B, C, or D)."""


def observation(payload: dict) -> str:
    return "V10_OBSERVATION_JSON:\n```json\n" + json.dumps(payload) + "\n```"


def r09_real_overview_excerpt(*, extra_rows: list[dict] | None = None) -> str:
    """Rows copied verbatim from the P131 R09 real Overview trajectory."""

    rows = [
        {
            "timestamp_s": 73.5,
            "scene_id": "s3",
            "description": "Indoor kitchen scene shows a frying pan with browned sausages and a utensil stirring them; no full person visible.",
            "event_tags": ["indoor", "cooking", "sausages"],
            "needs_focus": "This is the clearest query-relevant cooking shot; verify if sausages are being actively cooked.",
        },
        {
            "timestamp_s": 77.8,
            "scene_id": "s3",
            "description": "Close view of sausages bubbling in a pan on a stovetop, with a utensil moving them; no people fully visible.",
            "event_tags": ["indoor", "cooking", "sausages"],
            "needs_focus": "Confirm whether this same cooking action repeats or is a separate occurrence.",
        },
        {
            "timestamp_s": 101.1,
            "scene_id": "s3",
            "description": "Aerial landscape with a bird gliding over a vast plain; no people visible and no cooking in frame.",
            "event_tags": ["wildlife", "bird", "aerial"],
            "needs_focus": "No obvious query evidence here; just transition footage.",
        },
        {
            "timestamp_s": 108.9,
            "scene_id": "s3",
            "description": "Indoor stovetop again shows sausages frying in a pan, glossy with oil; no full person visible.",
            "event_tags": ["indoor", "cooking", "sausages"],
            "needs_focus": "Possible second view of the same sausage-cooking action.",
        },
        {
            "timestamp_s": 318.6,
            "scene_id": "s11",
            "description": "Indoor kitchen with an older man standing at a stove, cooking in a pan under hanging utensils; one person visible.",
            "event_tags": ["indoor", "person", "cooking"],
            "needs_focus": "This confirms a human cooking scene; check pan contents.",
        },
        {
            "timestamp_s": 342.3,
            "scene_id": "s12",
            "description": "Several lion cubs walk together in grass, energetic and alert, with adult lion nearby; no people visible.",
            "event_tags": ["wildlife", "lion", "cub"],
            "needs_focus": "Verify cub group count.",
        },
        {
            "timestamp_s": 429.5,
            "scene_id": "s15",
            "description": "Nighttime or dim indoor scene shows a flame rising from a bowl or dish on a table; one person’s hand is partly visible.",
            "event_tags": ["indoor", "fire", "person"],
            "needs_focus": "Not wildlife-related; likely a symbolic or transition shot.",
        },
    ]
    rows.extend(extra_rows or [])
    return observation(
        {
            "parse_ok": True,
            "timestamp_observations": rows,
            "scene_summaries": [
                {
                    "scene_id": "s3",
                    "t_range": [73.5, 108.9],
                    "summary": "A brief indoor cooking sequence shows sausages frying in a pan, contrasted with aerial wildlife footage.",
                    "possible_evidence": True,
                    "suggest_focus_windows": [[73.5, 77.8], [108.9, 108.9]],
                    "missing_detail": "Confirm whether the sausage shot is the only cooking occurrence.",
                },
                {
                    "scene_id": "s11",
                    "t_range": [318.6, 326.3],
                    "summary": "A human cooking shot appears in a kitchen, then cuts to an aerial river landscape.",
                    "possible_evidence": True,
                    "suggest_focus_windows": [[318.6, 318.6]],
                    "missing_detail": "Check pan contents.",
                },
                {
                    "scene_id": "s15",
                    "t_range": [416.6, 435.6],
                    "summary": "The leopard continues moving through grass, with brief non-wildlife flame imagery and nearby small mammals.",
                    "possible_evidence": False,
                    "suggest_focus_windows": [[416.6, 420.1], [435.6, 435.6]],
                    "missing_detail": "Confirm whether the animal is still tracking prey.",
                },
            ],
        }
    )


def initial_spec(overview: str, *, p132: bool) -> tuple[dict, dict]:
    memory: dict = {}
    init_p130_state(memory, question=R09, duration=490.06, p131_enabled=True)
    spec = initial_count_verify_spec(
        memory,
        question=R09,
        overview_output=overview,
        duration=490.06,
        max_candidates=8,
        p132_weak_lead_visible_core=p132,
    )
    assert spec is not None
    return memory, spec


def exact_anchors(spec: dict) -> set[float]:
    return {
        float(value)
        for row in spec["candidate_windows"]
        for value in row["timestamp_anchors"]
    }


class P132WeakLeadVisibleCoreTests(unittest.TestCase):
    def test_real_r09_replay_replaces_count_pollution_with_weak_fire_lead(self) -> None:
        memory, spec = initial_spec(r09_real_overview_excerpt(), p132=True)

        anchors = exact_anchors(spec)
        self.assertIn(429.5, anchors)
        self.assertNotIn(342.3, anchors)
        self.assertEqual(
            [row["t_range"] for row in spec["candidate_windows"]],
            [
                [64.5, 86.8],
                [99.9, 117.9],
                [309.6, 327.6],
                [417.5, 438.5],
            ],
        )
        late = spec["candidate_windows"][-1]
        self.assertIn(422.75, late["timestamp_anchors"])

        # Scheduling remains routing-only.  The weak lead cannot increment the
        # Count lower bound until frame_verify emits a direct local receipt.
        self.assertEqual(memory["p130_global"]["evidence_receipts"], [])
        self.assertEqual(
            memory["p130_global"]["snapshot"]["observed_count_lower_bound"], 0
        )

    def test_flag_off_preserves_p131_selection(self) -> None:
        _memory, spec = initial_spec(r09_real_overview_excerpt(), p132=False)
        anchors = exact_anchors(spec)
        self.assertIn(342.3, anchors)
        self.assertNotIn(429.5, anchors)

    def test_negative_target_sentence_does_not_train_wildlife_signature(self) -> None:
        overview = r09_real_overview_excerpt(
            extra_rows=[
                {
                    "timestamp_s": 200.0,
                    "scene_id": "birds_only",
                    "description": "A bird flies over a river with no food preparation visible.",
                    "event_tags": ["wildlife", "bird", "aerial"],
                    "needs_focus": "Count the birds in the group.",
                }
            ]
        )
        _memory, spec = initial_spec(overview, p132=True)
        self.assertFalse(
            any(start <= 200.0 <= end for start, end in spec["windows"])
        )

    def test_one_signature_tag_is_insufficient_but_coherent_pair_is_kept(self) -> None:
        overview = r09_real_overview_excerpt(
            extra_rows=[
                {
                    "timestamp_s": 400.0,
                    "scene_id": "indoor_only",
                    "description": "An empty indoor hallway.",
                    "event_tags": ["indoor"],
                    "needs_focus": "Inspect the doorway.",
                }
            ]
        )
        _memory, spec = initial_spec(overview, p132=True)
        self.assertFalse(
            any(start <= 400.0 <= end for start, end in spec["windows"])
        )
        self.assertTrue(
            any(start <= 429.5 <= end for start, end in spec["windows"])
        )

    def test_count_target_extraction_uses_quoted_visible_action(self) -> None:
        parsed = parse_global_question(R09)
        self.assertEqual(_p132_count_target_tokens(parsed, R09), {"cook", "sausag"})
        other = (
            "In this video, how many times does the scene of the 'making jewelry' "
            "action appear in total?\n(A) 1\n(B) 2"
        )
        self.assertEqual(
            _p132_count_target_tokens(parse_global_question(other), other),
            {"jewelry", "making"},
        )

    def test_negation_uses_the_question_target_for_unrelated_domains(self) -> None:
        self.assertTrue(_p132_count_negative("There is no jewelry making visible.", {"jewelry", "making"}))
        self.assertFalse(_p132_count_negative("No full person visible; hands are making jewelry.", {"jewelry", "making"}))
        self.assertTrue(_p132_count_negative("A bird flies; no cooking in frame.", {"cook", "sausag"}))


if __name__ == "__main__":
    unittest.main()
