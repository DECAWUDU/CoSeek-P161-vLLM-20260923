import threading
import unittest
from unittest.mock import patch

from videoseek.tools import dual_path_overview as dual
from videoseek.tools import overview


def _skeleton():
    return {
        "scenes": [
            {"scene_id": "S000", "t_range": [0.0, 50.0]},
            {"scene_id": "S001", "t_range": [50.0, 100.0]},
        ]
    }


class OverviewDispatchTests(unittest.TestCase):
    def test_disabled_switch_preserves_standard_overview(self):
        with patch.object(
            overview, "_execute_standard_overview", return_value="remote"
        ) as standard:
            output = overview.execute_overview(
                {"dual_path_overview_enabled": False}, {"sentinel": True}
            )

        self.assertEqual(output, "remote")
        standard.assert_called_once_with(
            {"dual_path_overview_enabled": False}, {"sentinel": True}
        )

    def test_enabled_switch_uses_dual_path_wrapper(self):
        with patch.object(
            dual, "execute_dual_path_overview", return_value="dual"
        ) as execute_dual:
            output = overview.execute_overview(
                {
                    "dual_path_overview_enabled": True,
                    "use_skeleton_overview": False,
                },
                {"sentinel": True},
            )

        self.assertEqual(output, "dual")
        self.assertIs(
            execute_dual.call_args.kwargs["remote_runner"],
            overview._execute_standard_overview,
        )


class MinimumRecallBudgetTests(unittest.TestCase):
    def test_remote_completion_does_not_cut_scan_before_minimum(self):
        frames = [
            {
                "timestamp_s": float(index * 2),
                "scene_id": "S000" if index < 16 else "S001",
                "source": "coverage_gap",
                "score": 0.7,
            }
            for index in range(40)
        ]
        remote_done = threading.Event()
        remote_done.set()

        def caption_batch(_config, _parameters, batch):
            rows = [
                {
                    "timestamp_s": item["timestamp_s"],
                    "caption": f"visible event at {item['timestamp_s']:.1f}s",
                }
                for item in batch
            ]
            return rows, "caption batch", 0.01

        config = {
            "dual_path_local_batch_size": 8,
            "dual_path_local_min_processed_frames": 32,
            "dual_path_local_deadline_s": 180,
            "dual_path_local_top_k": 4,
            "dual_path_candidate_window_s": 12,
            "dual_path_choice_aware_retrieval_enabled": False,
        }
        with patch.object(dual, "_local_batch_caption", side_effect=caption_batch):
            result = dual.run_local_supplementary_scan(
                config,
                {"duration": 100.0, "question": "What event is visible?"},
                frames,
                remote_done=remote_done,
            )

        coverage = result["coverage"]
        self.assertEqual(coverage["frames_attempted"], 32)
        self.assertEqual(coverage["frames_processed"], 32)
        self.assertEqual(coverage["batches_completed"], 4)
        self.assertEqual(coverage["stop_reason"], "remote_completed")


class CompactFusionTests(unittest.TestCase):
    def test_query_reserve_replaces_redundant_entity_match_with_event_match(self):
        candidates = [
            {
                "candidate_id": "entity_1",
                "priority": 0.56,
                "information_gain_score": 0.56,
                "matched_query_terms": ["girl", "glass"],
                "t_range": [60.0, 72.0],
            },
            {
                "candidate_id": "entity_2",
                "priority": 0.54,
                "information_gain_score": 0.54,
                "matched_query_terms": ["girl", "glass"],
                "t_range": [308.0, 320.0],
            },
            {
                "candidate_id": "event",
                "priority": 0.52,
                "information_gain_score": 0.52,
                "matched_query_terms": ["hurry", "run"],
                "t_range": [268.7, 280.7],
            },
        ]

        selected = dual.select_local_candidates_for_fusion(
            candidates,
            selective=True,
            limit=2,
            query_target_reserve=True,
        )

        self.assertEqual(
            {item["candidate_id"] for item in selected},
            {"entity_1", "event"},
        )

    def test_query_reserve_off_preserves_priority_ranking(self):
        candidates = [
            {
                "candidate_id": "entity_1",
                "priority": 0.56,
                "information_gain_score": 0.56,
                "matched_query_terms": ["girl", "glass"],
                "t_range": [60.0, 72.0],
            },
            {
                "candidate_id": "entity_2",
                "priority": 0.54,
                "information_gain_score": 0.54,
                "matched_query_terms": ["girl", "glass"],
                "t_range": [308.0, 320.0],
            },
            {
                "candidate_id": "event",
                "priority": 0.52,
                "information_gain_score": 0.52,
                "matched_query_terms": ["hurry", "run"],
                "t_range": [268.7, 280.7],
            },
        ]

        selected = dual.select_local_candidates_for_fusion(
            candidates,
            selective=True,
            limit=2,
            query_target_reserve=False,
        )

        self.assertEqual(
            [item["candidate_id"] for item in selected],
            ["entity_1", "entity_2"],
        )

    def test_query_discrimination_precedes_generic_semantic_novelty(self):
        candidates = [
            {
                "candidate_id": "relevant_1",
                "priority": 0.57,
                "information_gain_score": 0.58,
                "matched_query_terms": ["staircase"],
                "t_range": [50.0, 62.0],
            },
            {
                "candidate_id": "relevant_2",
                "priority": 0.55,
                "information_gain_score": 0.56,
                "matched_query_terms": ["staircase"],
                "t_range": [82.0, 94.0],
            },
            {
                "candidate_id": "novel_generic",
                "priority": 0.52,
                "information_gain_score": 0.91,
                "matched_query_terms": ["are"],
                "t_range": [330.0, 342.0],
            },
        ]

        selected = dual.select_local_candidates_for_fusion(
            candidates,
            selective=True,
            limit=2,
        )

        self.assertEqual(
            [item["candidate_id"] for item in selected],
            ["relevant_1", "relevant_2"],
        )

    def test_only_two_routing_candidates_reach_planner(self):
        remote = {
            "global_summary": "Remote map missed the target event.",
            "timestamp_observations": [
                {
                    "timestamp_s": 10.0,
                    "scene_id": "overview",
                    "description": "Empty corridor.",
                }
            ],
            "scene_summaries": [
                {
                    "scene_id": "overview",
                    "t_range": [5.0, 15.0],
                    "summary": "Empty corridor.",
                    "possible_evidence": False,
                    "suggest_focus_windows": [],
                }
            ],
        }
        rows = [
            {
                "timestamp_s": timestamp,
                "caption": caption,
                "frame_id": f"F{index:02d}",
                "sampling_source": "interframe_bitcost_peak",
            }
            for index, (timestamp, caption) in enumerate(
                [
                    (20.0, "A man stands in a room."),
                    (55.0, "Three men walk beside a staircase."),
                    (80.0, "Two people cross a courtyard."),
                ],
                start=1,
            )
        ]
        candidates = [
            {
                "candidate_id": f"OVL{index:03d}",
                "scene_id": "S000" if timestamp < 50 else "S001",
                "t_range": [timestamp - 6.0, timestamp + 6.0],
                "summary": caption,
                "priority": priority,
                "anchors": [timestamp],
                "recommended_verify_window": [timestamp - 6.0, timestamp + 6.0],
                "candidate_type": "interframe_bitcost_peak",
                "matched_query_terms": matched,
                "matched_choice_terms": [],
                "needs_verify": True,
            }
            for index, (timestamp, caption, priority, matched) in enumerate(
                [
                    (20.0, rows[0]["caption"], 0.45, []),
                    (55.0, rows[1]["caption"], 0.92, ["staircase", "walk"]),
                    (80.0, rows[2]["caption"], 0.70, ["people"]),
                ],
                start=1,
            )
        ]
        local = {
            "coverage": {
                "frames_processed": 32,
                "batches_completed": 4,
                "stop_reason": "remote_completed",
            },
            "frame_rows": rows,
            "candidates": candidates,
        }

        merged = dual.merge_overview_payloads(
            remote,
            local,
            skeleton=_skeleton(),
            timing={"exposed_local_wait_s": 0.0, "latency_hiding_ratio": 1.0},
            integration_mode="selective",
            fused_top_k=2,
            planner_caption_limit=0,
            remote_dedup_overlap=0.35,
        )

        local_scenes = [
            item
            for item in merged["scene_summaries"]
            if item.get("observer_backend") == "local_qwen"
        ]
        local_rows = [
            item
            for item in merged["timestamp_observations"]
            if item.get("observer_backend") == "local_qwen"
        ]
        self.assertEqual(len(local_scenes), 2)
        self.assertEqual(len(local_rows), 2)
        self.assertTrue(all(item["evidence_level"] == "routing_only" for item in local_scenes))
        self.assertTrue(all("Verify" in item["needs_focus"] for item in local_rows))
        self.assertIn("OVL002", {item["window_id"] for item in local_scenes})
        self.assertEqual(
            merged["dual_path_overview"]["local_planner_caption_count"], 2
        )

    def test_remote_semantic_prototype_recovers_missed_local_event(self):
        remote = {
            "global_summary": "Several brief dance-fitness inserts appear.",
            "timestamp_observations": [],
            "scene_summaries": [
                {
                    "scene_id": "dance_late",
                    "t_range": [67.8, 76.2],
                    "summary": "Group performs an energetic dance routine in a studio.",
                    "possible_evidence": True,
                    "suggest_focus_windows": [[67.8, 76.2]],
                },
                {
                    "scene_id": "dance_later",
                    "t_range": [140.0, 150.0],
                    "summary": "People practice coordinated dance steps in a studio.",
                    "possible_evidence": True,
                    "suggest_focus_windows": [[140.0, 150.0]],
                }
            ],
        }
        generic = {
            "candidate_id": "generic",
            "scene_id": "S009",
            "t_range": [250.0, 262.0],
            "summary": "A herd moves across dry terrain.",
            "priority": 0.9,
            "anchors": [256.0],
            "recommended_verify_window": [250.0, 262.0],
            "matched_query_terms": [],
            "matched_choice_terms": [],
            "needs_verify": True,
        }
        missed_event = {
            "candidate_id": "missed_event",
            "scene_id": "S002",
            "t_range": [48.6, 60.6],
            "summary": "A group of women dancing in a studio.",
            "priority": 0.35,
            "anchors": [54.6],
            "recommended_verify_window": [48.6, 60.6],
            "matched_query_terms": [],
            "matched_choice_terms": [],
            "needs_verify": True,
        }
        local = {
            "coverage": {"frames_processed": 32, "batches_completed": 4},
            "frame_rows": [
                {
                    "frame_id": "OVF001",
                    "timestamp_s": 54.6,
                    "caption": missed_event["summary"],
                }
            ],
            "candidates": [generic],
            "candidate_pool": [generic, missed_event],
        }

        merged = dual.merge_overview_payloads(
            remote,
            local,
            skeleton=_skeleton(),
            timing={"exposed_local_wait_s": 0.0},
            integration_mode="selective",
            fused_top_k=1,
            planner_caption_limit=0,
            semantic_prototype_ranking_enabled=True,
        )

        local_scenes = [
            item
            for item in merged["scene_summaries"]
            if item.get("observer_backend") == "local_qwen"
        ]
        self.assertEqual(len(local_scenes), 1)
        self.assertEqual(local_scenes[0]["window_id"], "missed_event")
        self.assertEqual(
            set(local_scenes[0]["matched_remote_prototype_terms"]),
            {"danc", "studio"},
        )
        self.assertGreater(
            local_scenes[0]["remote_prototype_temporal_novelty"], 0.5
        )

    def test_prototype_switch_off_keeps_compact_p31_pool(self):
        remote = {
            "global_summary": "Dance scene.",
            "timestamp_observations": [],
            "scene_summaries": [
                {
                    "scene_id": "dance",
                    "t_range": [70.0, 75.0],
                    "summary": "People dance in a studio.",
                    "possible_evidence": True,
                    "suggest_focus_windows": [[70.0, 75.0]],
                }
            ],
        }
        generic = {
            "candidate_id": "p31_candidate",
            "scene_id": "S001",
            "t_range": [20.0, 32.0],
            "summary": "A generic high-motion scene.",
            "priority": 0.9,
            "anchors": [26.0],
            "recommended_verify_window": [20.0, 32.0],
            "matched_query_terms": [],
            "matched_choice_terms": [],
        }
        hidden = {
            **generic,
            "candidate_id": "pool_only",
            "t_range": [45.0, 57.0],
            "anchors": [51.0],
            "summary": "People dancing in a studio.",
        }
        merged = dual.merge_overview_payloads(
            remote,
            {
                "coverage": {},
                "frame_rows": [],
                "candidates": [generic],
                "candidate_pool": [generic, hidden],
            },
            skeleton=_skeleton(),
            timing={},
            integration_mode="selective",
            fused_top_k=1,
            planner_caption_limit=0,
            semantic_prototype_ranking_enabled=False,
        )
        local_scenes = [
            item
            for item in merged["scene_summaries"]
            if item.get("observer_backend") == "local_qwen"
        ]
        self.assertEqual(local_scenes[0]["window_id"], "p31_candidate")


if __name__ == "__main__":
    unittest.main()
