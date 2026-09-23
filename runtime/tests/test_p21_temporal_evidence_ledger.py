from __future__ import annotations

import inspect
import unittest

from videoseek.agent import VideoSeekAgent
from videoseek.core import Action
from videoseek.core.memory import (
    build_temporal_evidence_ledger,
    format_memory_for_prompt,
    init_observation_memory,
    merge_tool_observation,
)
from videoseek.core.observation_reuse import (
    find_reusable_observation,
    make_cache_entry,
)
from videoseek.tools.frame_verify import _execute_multiwindow_frame_verify_once
from videoseek.tools.frame_verify import (
    _merge_anchor_audit_into_candidate_assessments,
    _merge_candidate_sampling_timestamps,
    _normalize_anchor_assessments,
    _normalize_candidate_windows,
    _packet_detail_anchor_positions,
)
from videoseek.tools.v10_format import format_v10_observation
from videoseek.tools.temporal_sampling import (
    merge_mandatory_timestamps,
    supplement_frame_indices,
)


def _multiwindow_parameters() -> dict:
    return {
        "query": "Find each distinct jewelry-making event.",
        "candidate_windows": [
            {
                "candidate_id": "LQ001",
                "event_group_id": "EG001",
                "t_range": [10.0, 15.0],
                "summary": "hands use pliers",
            },
            {
                "candidate_id": "LQ002",
                "event_group_id": "EG001",
                "t_range": [18.0, 23.0],
                "summary": "same work continues",
            },
            {
                "candidate_id": "LQ003",
                "event_group_id": "EG002",
                "t_range": [70.0, 75.0],
                "summary": "later threading scene",
            },
        ],
    }


def _multiwindow_output() -> str:
    return format_v10_observation(
        {
            "tool": "frame_verify",
            "t_range": [10.0, 75.0],
            "candidate_assessments": [
                {
                    "candidate_id": "LQ001",
                    "event_group_id": "EG001",
                    "target_match": "matched",
                    "event_match": "direct",
                    "observed_fact": "Hands use pliers on beads.",
                    "target_binding_reason": "Direct crafting is visible.",
                    "best_timestamp_s": 13.0,
                },
                {
                    "candidate_id": "LQ002",
                    "event_group_id": "EG001",
                    "target_match": "matched",
                    "event_match": "direct",
                    "observed_fact": "The same continuous event as LQ001 continues.",
                    "target_binding_reason": "Identical setup and uninterrupted action.",
                    "best_timestamp_s": 20.0,
                },
                {
                    "candidate_id": "LQ003",
                    "event_group_id": "EG002",
                    "target_match": "matched",
                    "event_match": "direct",
                    "observed_fact": "A later scene shows bead threading.",
                    "target_binding_reason": "A separate setup and edit are visible.",
                    "best_timestamp_s": 72.0,
                },
            ],
            "timestamp_observations": [
                {"candidate_id": "LQ001", "timestamp_s": 13.0, "description": "pliers"},
                {"candidate_id": "LQ002", "timestamp_s": 20.0, "description": "continues"},
                {"candidate_id": "LQ003", "timestamp_s": 72.0, "description": "threading"},
            ],
            "overall_summary": "Two temporally distinct verified events.",
            "target_match": "matched",
            "target_event_match": "direct",
            "detail_sufficient": True,
            "observer_backend": "api",
            "parse_ok": True,
        }
    )


def _direct_output() -> str:
    return format_v10_observation(
        {
            "tool": "frame_verify",
            "t_range": [8.4, 19.9],
            "timestamp_observations": [
                {
                    "timestamp_s": 16.8,
                    "description": "Hands thread a bead onto wire.",
                }
            ],
            "observed_fact": "Hands thread a bead onto wire.",
            "target_match": "matched",
            "target_event_match": "direct",
            "detail_sufficient": True,
            "observer_backend": "api",
            "parse_ok": True,
        }
    )


class _Registry:
    def __init__(self, output: str):
        self.output = output
        self.calls = 0

    def has_tool(self, name):
        return name == "frame_verify"

    def get_function(self, name):
        def execute(*, config, parameters):
            self.calls += 1
            return self.output

        return execute


class P21TemporalEvidenceLedgerTests(unittest.TestCase):
    def test_anchor_audit_preserves_direct_event_from_later_anchor(self):
        anchors, complete = _normalize_anchor_assessments(
            [
                {
                    "candidate_id": "LQ003",
                    "timestamp_s": 10.4,
                    "target_match": "partial",
                    "event_match": "context_only",
                    "observed_fact": "Child presents a jewelry kit box.",
                },
                {
                    "candidate_id": "LQ003",
                    "timestamp_s": 18.0,
                    "target_match": "matched",
                    "event_match": "direct",
                    "observed_fact": "Hands assemble a beaded bracelet with tools.",
                    "target_binding_reason": "The making action is directly visible.",
                },
            ],
            allowed_by_candidate={"LQ003": [10.4, 18.0]},
            event_binding_enabled=True,
        )
        assessments = [
            {
                "candidate_id": "LQ003",
                "target_match": "partial",
                "event_match": "context_only",
                "observed_fact": "Jewelry kit shown.",
                "target_binding_reason": "Only context was selected.",
                "best_timestamp_s": 10.4,
                "supports_options": [],
                "contradicts_options": [],
                "option_set_conflict": False,
            }
        ]
        _merge_anchor_audit_into_candidate_assessments(
            assessments,
            anchors,
            event_binding_enabled=True,
        )
        self.assertTrue(complete)
        self.assertEqual(assessments[0]["event_match"], "direct")
        self.assertEqual(assessments[0]["target_match"], "matched")
        self.assertEqual(assessments[0]["best_timestamp_s"], 18.0)
        self.assertTrue(assessments[0]["anchor_audit_direct_preserved"])

    def test_anchor_audit_does_not_upgrade_context_without_direct_observation(self):
        anchors, complete = _normalize_anchor_assessments(
            [
                {
                    "candidate_id": "LQ003",
                    "timestamp_s": 10.4,
                    "target_match": "partial",
                    "event_match": "context_only",
                    "observed_fact": "Jewelry kit is visible.",
                },
                {
                    "candidate_id": "LQ003",
                    "timestamp_s": 18.0,
                    "target_match": "not_visible",
                    "event_match": "not_visible",
                    "observed_fact": "No hands or tools are visible.",
                },
            ],
            allowed_by_candidate={"LQ003": [10.4, 18.0]},
            event_binding_enabled=True,
        )
        assessments = [
            {
                "candidate_id": "LQ003",
                "target_match": "partial",
                "event_match": "context_only",
                "observed_fact": "Jewelry kit shown.",
                "target_binding_reason": "Context only.",
                "best_timestamp_s": 10.4,
                "supports_options": [],
                "contradicts_options": [],
                "option_set_conflict": False,
            }
        ]
        _merge_anchor_audit_into_candidate_assessments(
            assessments,
            anchors,
            event_binding_enabled=True,
        )
        self.assertTrue(complete)
        self.assertEqual(assessments[0]["event_match"], "context_only")
        self.assertNotIn("anchor_audit_direct_preserved", assessments[0])

    def test_mandatory_upstream_anchor_replaces_uniform_frame_without_growing_budget(self):
        selected = merge_mandatory_timestamps(
            [262.2, 265.2, 268.2, 271.2, 274.2],
            [270.1],
            start_time=262.2,
            end_time=278.2,
            budget=5,
        )
        self.assertEqual(len(selected), 5)
        self.assertIn(270.1, selected)
        self.assertEqual(selected[0], 262.2)
        self.assertEqual(selected[-1], 274.2)

    def test_frame_index_fallback_keeps_converted_anchor(self):
        selected = supplement_frame_indices(
            [300, 450],
            [240, 300, 360, 420, 480],
            budget=5,
        )
        self.assertIn(450, selected)
        self.assertEqual(len(selected), 5)

    def test_query_confidence_anchor_survives_fixed_packet_budget(self):
        candidate = {
            "localized_timestamp_anchors": [10.4, 12.0, 15.7, 18.7],
            "localized_evidence_anchors": [
                {"timestamp_s": 10.4, "confidence": "high"},
                {"timestamp_s": 12.0, "confidence": "medium"},
                {"timestamp_s": 15.7, "confidence": "high"},
                {"timestamp_s": 18.7, "confidence": "high"},
            ],
        }
        positioned_rows = [
            (0, 5.1, object()),
            (1, 10.4, object()),
            (2, 12.0, object()),
            (3, 15.7, object()),
            (4, 18.7, object()),
        ]
        positions, source = _packet_detail_anchor_positions(
            candidate=candidate,
            positioned_rows=positioned_rows,
            limit=2,
        )
        selected_timestamps = [positioned_rows[position][1] for position in positions]
        self.assertEqual(source, "query_confidence_then_context")
        self.assertIn(18.7, selected_timestamps)
        self.assertTrue(any(timestamp in selected_timestamps for timestamp in (10.4, 15.7)))

    def test_query_confidence_anchor_is_sampled_before_uniform_context(self):
        selected = _merge_candidate_sampling_timestamps(
            [5.1, 9.1, 12.0, 16.0, 20.4],
            [5.1, 9.1, 10.4, 12.0, 15.7, 18.7, 20.4],
            priority_anchors=[10.4, 15.7, 18.7],
            start=5.1,
            end=20.4,
            budget=5,
        )
        self.assertEqual(len(selected), 5)
        self.assertTrue({10.4, 15.7, 18.7}.issubset(set(selected)))

    def test_candidate_normalization_preserves_evidence_anchor_confidence(self):
        candidates = _normalize_candidate_windows(
            [
                {
                    "candidate_id": "LQ003",
                    "t_range": [5.1, 20.4],
                    "localized_evidence_anchors": [
                        {"timestamp_s": 18.7, "confidence": "high"}
                    ],
                }
            ],
            duration=30.0,
            max_windows=2,
        )
        self.assertEqual(
            candidates[0]["localized_evidence_anchors"],
            [{"timestamp_s": 18.7, "confidence": "high"}],
        )

    def test_ledger_merges_continuous_candidates_but_keeps_later_event(self):
        memory = init_observation_memory()
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters=_multiwindow_parameters(),
            output=_multiwindow_output(),
        )
        ledger = build_temporal_evidence_ledger(memory)
        self.assertEqual(ledger["stable_event_span_count"], 2)
        self.assertEqual(ledger["events"][0]["t_range"], [10.0, 23.0])
        self.assertEqual(ledger["events"][0]["anchor_timestamps"], [13.0, 20.0])
        self.assertEqual(ledger["events"][1]["t_range"], [70.0, 75.0])
        self.assertEqual(ledger["events"][0]["verification_count"], 1)

    def test_later_ambiguous_observation_does_not_erase_verified_event(self):
        memory = init_observation_memory()
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters=_multiwindow_parameters(),
            output=_multiwindow_output(),
        )
        ambiguous_parameters = {
            "query": "Inspect the same early event.",
            "candidate_windows": [
                {"candidate_id": "A001", "t_range": [11.0, 22.0]}
            ],
        }
        ambiguous_output = format_v10_observation(
            {
                "tool": "frame_verify",
                "t_range": [11.0, 22.0],
                "candidate_assessments": [
                    {
                        "candidate_id": "A001",
                        "target_match": "ambiguous",
                        "event_match": "ambiguous",
                        "observed_fact": "The tool is partly occluded.",
                    }
                ],
                "observer_backend": "api",
                "parse_ok": True,
            }
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters=ambiguous_parameters,
            output=ambiguous_output,
        )
        ledger = build_temporal_evidence_ledger(memory)
        self.assertEqual(ledger["stable_event_span_count"], 2)
        self.assertEqual(len(ledger["boundary_observations"]), 1)
        self.assertEqual(
            ledger["boundary_observations"][0]["overlaps_verified_events"],
            ["TE001"],
        )

    def test_direct_verify_is_projected_when_binding_has_no_candidate_window(self):
        memory = init_observation_memory()
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={
                "query": "Inspect active jewelry making.",
                "start_time": 8.4,
                "end_time": 19.9,
            },
            output=_direct_output(),
        )
        ledger = build_temporal_evidence_ledger(memory)
        self.assertEqual(ledger["stable_event_span_count"], 1)
        self.assertEqual(ledger["events"][0]["t_range"], [8.4, 19.9])

    def test_bound_location_without_requested_event_is_not_an_event(self):
        memory = init_observation_memory()
        output = format_v10_observation(
            {
                "tool": "frame_verify",
                "t_range": [367.7, 380.2],
                "timestamp_observations": [
                    {
                        "timestamp_s": 373.5,
                        "description": "The staircase is visible; nobody interacts.",
                    }
                ],
                "observed_fact": (
                    "A staircase is visible, but no physical interaction occurs "
                    "on or beside it."
                ),
                "target_match": "matched",
                "target_event_match": "context_only",
                "detail_sufficient": True,
                "observer_backend": "api",
                "parse_ok": True,
            }
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={
                "query": "How many people interact at the staircase?",
                "start_time": 372.7,
                "end_time": 375.2,
            },
            output=output,
        )
        ledger = build_temporal_evidence_ledger(memory)
        self.assertEqual(ledger["stable_event_span_count"], 0)
        self.assertEqual(len(ledger["contextual_candidates"]), 1)
        self.assertEqual(
            ledger["plausible_observed_span_range"],
            [0, 1],
        )

    def test_context_only_candidate_does_not_inflate_verified_count(self):
        memory = init_observation_memory()
        output = format_v10_observation(
            {
                "tool": "frame_verify",
                "t_range": [3.5, 26.8],
                "observed_fact": "Jewelry kit contents are handled, but assembly is not visible.",
                "target_match": "matched",
                "target_event_match": "context_only",
                "observer_backend": "api",
                "parse_ok": True,
            }
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={
                "query": "Count jewelry-making scenes.",
                "start_time": 3.5,
                "end_time": 26.8,
            },
            output=output,
        )
        ledger = build_temporal_evidence_ledger(memory)
        self.assertEqual(ledger["stable_event_span_count"], 0)
        self.assertEqual(ledger["plausible_observed_span_range"], [0, 1])
        self.assertEqual(len(ledger["contextual_candidates"]), 1)
        self.assertEqual(len(ledger["boundary_observations"]), 0)

    def test_partially_overlapping_context_is_not_a_distinct_occurrence(self):
        memory = init_observation_memory()
        for start, end, event_match, fact in (
            (10.0, 20.0, "context_only", "The participant approaches the work area."),
            (18.0, 30.0, "direct", "The requested action is directly visible."),
        ):
            merge_tool_observation(
                memory,
                tool_name="frame_verify",
                parameters={"query": "Inspect the event.", "start_time": start, "end_time": end},
                output=format_v10_observation(
                    {
                        "tool": "frame_verify",
                        "t_range": [start, end],
                        "observed_fact": fact,
                        "target_match": "matched",
                        "target_event_match": event_match,
                        "observer_backend": "api",
                        "parse_ok": True,
                    }
                ),
            )
        ledger = build_temporal_evidence_ledger(memory)
        self.assertEqual(ledger["stable_event_span_count"], 1)
        self.assertEqual(ledger["plausible_observed_span_range"], [1, 1])
        self.assertEqual(
            ledger["contextual_candidates"][0]["overlaps_verified_events"],
            ["TE001"],
        )

    def test_legacy_negated_event_fact_is_not_counted_as_occurrence(self):
        memory = init_observation_memory()
        output = format_v10_observation(
            {
                "tool": "frame_verify",
                "t_range": [367.7, 380.2],
                "timestamp_observations": [
                    {"timestamp_s": 373.5, "description": "No interaction."}
                ],
                "observed_fact": "Zero people are physically interacting at the staircase.",
                "target_match": "matched",
                "target_event_match": "unknown",
                "observer_backend": "api",
                "parse_ok": True,
            }
        )
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={
                "query": "How many people interact at the staircase?",
                "start_time": 372.7,
                "end_time": 375.2,
            },
            output=output,
        )
        ledger = build_temporal_evidence_ledger(memory)
        self.assertEqual(ledger["stable_event_span_count"], 0)
        self.assertEqual(
            ledger["boundary_observations"][0]["status"],
            "event_absent",
        )

    def test_reused_observation_is_audited_without_duplicate_evidence(self):
        memory = init_observation_memory()
        parameters = {
            "query": "Inspect explicit jewelry making.",
            "start_time": 8.4,
            "end_time": 19.9,
        }
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters=parameters,
            output=_direct_output(),
        )
        reused = {
            **(__import__("videoseek.core.memory", fromlist=["extract_v10_payload"]).extract_v10_payload(_direct_output()) or {}),
            "observation_reused": True,
            "no_new_visual_information": True,
            "reuse_cache_id": "FVC0001",
            "reuse_count": 1,
        }
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters=parameters,
            output=format_v10_observation(reused),
        )
        self.assertEqual(len(memory["timestamped_observations"]), 1)
        self.assertEqual(len(memory["tool_observations"]), 2)
        self.assertEqual(memory["verified_observation_reuse_stats"]["reuse_count"], 1)
        prompt = format_memory_for_prompt(
            memory,
            include_temporal_evidence_ledger=True,
        )
        self.assertIn("1 repeated visual request", prompt)

    def test_cache_requires_same_pixels_and_semantically_equivalent_query(self):
        parameters = {
            "query": "Do frames show explicit jewelry making, threading beads onto wire or using pliers?",
            "start_time": 8.4,
            "end_time": 19.9,
        }
        cache = [
            make_cache_entry(
                cache_id="FVC0001",
                parameters=parameters,
                payload=__import__("videoseek.core.memory", fromlist=["extract_v10_payload"]).extract_v10_payload(_direct_output()) or {},
            )
        ]
        equivalent = {
            "query": "Are beads threaded onto wire or are pliers used for explicit jewelry-making?",
            "start_time": 8.4,
            "end_time": 19.9,
        }
        hit, _scores = find_reusable_observation(cache, parameters=equivalent)
        self.assertIsNotNone(hit)
        different_detail = {
            "query": "What color is the child's shirt?",
            "start_time": 8.4,
            "end_time": 19.9,
        }
        miss, _scores = find_reusable_observation(cache, parameters=different_detail)
        self.assertIsNone(miss)

    def test_agent_reuses_second_equivalent_frame_verify_without_tool_call(self):
        agent = VideoSeekAgent.__new__(VideoSeekAgent)
        agent.config = {
            "coseek1_verified_observation_reuse_enabled": True,
            "coseek1_verified_observation_reuse_overlap": 0.92,
            "coseek1_verified_observation_reuse_query_similarity": 0.55,
        }
        agent._verified_observation_cache = []
        agent.vr = object()
        agent.subtitles = []
        agent.video_path = "/tmp/video.mp4"
        agent.question = "How many times is jewelry made?"
        agent.duration = 100.0
        agent.output_dir = "/tmp"
        agent.tool_registry = _Registry(_direct_output())
        first = Action(
            "frame_verify",
            {
                "query": "Do frames show explicit jewelry making, threading beads onto wire or using pliers?",
                "start_time": 8.4,
                "end_time": 19.9,
                "mode": "count_occurrence",
            },
        )
        second = Action(
            "frame_verify",
            {
                "query": "Are beads threaded onto wire or are pliers used for explicit jewelry-making?",
                "start_time": 8.4,
                "end_time": 19.9,
                "mode": "count_occurrence",
            },
        )
        execute = agent._VideoSeekAgent__exec_action
        execute(first)
        second_output = execute(second)
        self.assertEqual(agent.tool_registry.calls, 1)
        self.assertIn('"observation_reused": true', second_output)

    def test_cache_matches_actual_context_padded_visual_coverage(self):
        first_parameters = {
            "query": "Count people physically interacting by the staircase railing.",
            "start_time": 384.3,
            "end_time": 385.1,
        }
        payload = {
            **(
                __import__(
                    "videoseek.core.memory",
                    fromlist=["extract_v10_payload"],
                ).extract_v10_payload(_direct_output())
                or {}
            ),
            "t_range": [379.3, 390.1],
        }
        cache = [
            make_cache_entry(
                cache_id="FVC0001",
                parameters=first_parameters,
                payload=payload,
                context_pad_s=5.0,
                duration=477.0,
            )
        ]
        repeated_pixels = {
            "query": "How many people physically interact next to the staircase or railing?",
            "start_time": 383.9,
            "end_time": 384.9,
        }
        hit, scores = find_reusable_observation(
            cache,
            parameters=repeated_pixels,
            context_pad_s=5.0,
            duration=477.0,
        )
        self.assertIsNotNone(hit)
        self.assertGreaterEqual(scores["window_overlap"], 0.92)

    def test_verifier_prompt_uses_continuous_event_semantics(self):
        source = inspect.getsource(_execute_multiwindow_frame_verify_once)
        self.assertIn("temporally continuous event shown across adjacent shots", source)
        self.assertIn("Do not require every participant to be co-visible", source)


if __name__ == "__main__":
    unittest.main()
