import json
import unittest

from videoseek.agent import VideoSeekAgent
from videoseek.core import Action
from videoseek.core.candidate_frontier import (
    format_candidate_frontier_for_prompt,
    record_planner_proposal,
    record_routing_audit,
    refresh_candidate_frontier,
)
from videoseek.core.memory import init_observation_memory
from videoseek.core.memory import merge_tool_observation
from videoseek.core.observation import Observation
from videoseek.core.trajectory import TrajectoryStep


class CandidateFrontierTest(unittest.TestCase):
    def test_frontier_preserves_provenance_coverage_and_status(self):
        memory = init_observation_memory()
        memory["scene_memory"] = [
            {
                "source_tool": "overview",
                "scene_id": "S003",
                "t_range": [50.0, 80.0],
                "summary": "two men cook while a woman reaches the doorway",
                "possible_evidence": True,
                "suggest_focus_windows": [[53.0, 61.0]],
                "missing_detail": "dress color",
                "requirement_coverage": {
                    "matched_requirements": ["woman enters", "two men cooking"],
                    "missing_requirements": ["dress color"],
                    "relation_verified": True,
                },
            },
            {
                "source_tool": "frame_verify",
                "scene_id": "S003",
                "t_range": [53.0, 61.0],
                "summary": "woman is visible in the doorway wearing blue",
                "possible_evidence": True,
                "detail_sufficient": True,
                "missing_detail": "",
                "requirement_coverage": {
                    "matched_requirements": ["dress color"],
                    "missing_requirements": [],
                    "relation_verified": True,
                },
            },
        ]
        refresh_candidate_frontier(
            memory,
            question="What color is the woman's dress?",
            duration=80.0,
        )
        record_planner_proposal(
            memory,
            planner_payload={
                "leading_hypothesis": "blue",
                "strongest_competitor": "red",
                "decision_critical_evidence": "verified torso color at the doorway",
            },
        )

        text = format_candidate_frontier_for_prompt(memory)
        self.assertIn("CF001", text)
        self.assertIn("'overview', 'frame_verify'", text)
        self.assertIn("relation_verified=True", text)
        self.assertIn("status=verified", text)
        self.assertIn("strongest_competitor=red", text)
        self.assertIn("Candidate coverage: status_counts=", text)

    def test_routing_audit_keeps_proposed_and_executed_actions(self):
        memory = init_observation_memory()
        proposed = Action("frame_verify", {"start_time": 0.0, "end_time": 60.0}, "p")
        executed = Action("skim_qwen", {"start_time": 0.0, "end_time": 60.0}, "r")
        audit = record_routing_audit(
            memory,
            step=2,
            planner_action=proposed,
            executed_action=executed,
            reason="same-range coarse observation",
        )
        self.assertTrue(audit["router_changed_action"])
        self.assertEqual(audit["planner_proposed_action"]["function"], "frame_verify")
        self.assertEqual(audit["router_executed_action"]["function"], "skim_qwen")

        step = TrajectoryStep(
            2,
            "original planner thought",
            executed,
            Observation(executed, "observation"),
            planner_proposed_action=proposed,
            routing_audit=audit,
        ).to_dict()
        self.assertEqual(step["thought"], "original planner thought")
        self.assertEqual(step["planner_proposed_action"]["function"], "frame_verify")

    def test_verified_padded_window_marks_contained_candidate_inspected(self):
        memory = init_observation_memory()
        memory["scene_memory"] = [
            {
                "source_tool": "overview",
                "scene_id": "S001",
                "t_range": [0.0, 11.8],
                "summary": "opening candidate",
            },
            {
                "source_tool": "frame_verify",
                "scene_id": "S000",
                "t_range": [0.0, 16.8],
                "summary": "verified opening detail",
                "detail_sufficient": True,
            },
        ]
        state = refresh_candidate_frontier(memory, duration=100.0)
        opening = next(item for item in state["candidates"] if item["t_range"] == [0.0, 11.8])
        self.assertEqual(opening["status"], "verified")
        self.assertIsNotNone(opening.get("inspected_by"))

    def test_explicit_timestamp_gap_becomes_frontier_candidate(self):
        memory = init_observation_memory()
        memory["scene_memory"] = [
            {
                "source_tool": "overview",
                "scene_id": "S004",
                "t_range": [339.5, 352.3],
                "summary": "person approaches the register",
                "possible_evidence": True,
            }
        ]
        memory["timestamped_observations"] = [
            {
                "source_tool": "overview",
                "timestamp_s": 313.8,
                "scene_id": "S004",
                "description": "two people have a tense exchange",
                "needs_focus": "determine whether access is requested or demanded",
                "event_tags": ["verbal_interaction"],
            }
        ]

        state = refresh_candidate_frontier(memory, duration=403.5)
        cue = next(
            item
            for item in state["candidates"]
            if item.get("anchor_timestamp_s") == 313.8
        )
        self.assertEqual(cue["t_range"], [307.8, 319.8])
        self.assertEqual(cue["origin_kind"], "timestamp_cue")
        self.assertEqual(cue["evidence_scope"], "local_timestamp")
        self.assertIn("requested or demanded", cue["missing_evidence"])

    def test_nearby_timestamp_cues_are_not_merged_by_padded_windows(self):
        memory = init_observation_memory()
        memory["timestamped_observations"] = [
            {
                "source_tool": "overview",
                "timestamp_s": 47.6,
                "scene_id": "S001",
                "description": "dance cutaway",
                "needs_focus": "confirm event",
            },
            {
                "source_tool": "overview",
                "timestamp_s": 55.6,
                "scene_id": "S002",
                "description": "another dance cutaway",
                "needs_focus": "confirm event",
            },
        ]
        memory["scene_memory"] = [
            {
                "source_tool": "frame_verify",
                "scene_id": "S001",
                "t_range": [41.8, 53.8],
                "summary": "dance workout",
                "detail_sufficient": True,
            },
            {
                "source_tool": "frame_verify",
                "scene_id": "S002",
                "t_range": [49.8, 61.8],
                "summary": "dance workout",
                "detail_sufficient": True,
            },
        ]

        state = refresh_candidate_frontier(memory, duration=100.0)
        cues = [item for item in state["candidates"] if item.get("origin_kind") == "timestamp_cue"]
        self.assertEqual([item["anchor_timestamp_s"] for item in cues], [47.6, 55.6])
        self.assertEqual(len(state["temporal_boundary_checks"]), 1)
        self.assertEqual(
            state["temporal_boundary_checks"][0]["continuity_status"],
            "unresolved",
        )
        text = format_candidate_frontier_for_prompt(memory)
        self.assertIn("padded tool windows is not continuity evidence", text)
        self.assertIn("continuity=unresolved", text)

    def test_evidence_scope_is_attached_without_changing_control_flow(self):
        memory = init_observation_memory()
        payload = {
            "observer_backend": "api",
            "t_range": [10.0, 20.0],
            "detail_sufficient": True,
            "timestamp_observations": [
                {
                    "timestamp_s": 15.0,
                    "scene_id": "S001",
                    "description": "no weapon is visible in this frame",
                }
            ],
            "scene_summaries": [
                {
                    "scene_id": "S001",
                    "t_range": [10.0, 20.0],
                    "summary": "calm interaction in this short window",
                }
            ],
        }
        fence = "`" * 3
        output = "V10_OBSERVATION_JSON:\n" + fence + "json\n" + json.dumps(payload) + "\n" + fence
        merge_tool_observation(
            memory,
            tool_name="frame_verify",
            parameters={"start_time": 10.0, "end_time": 20.0},
            output=output,
            use_structured_evidence_state=True,
            question_context="Is this stealing or robbery?\n(A) Stealing\n(B) Robbery",
            include_evidence_scope=True,
        )
        self.assertEqual(memory["timestamped_observations"][0]["evidence_scope"], "local_timestamp")
        self.assertEqual(memory["scene_memory"][0]["evidence_scope"], "local_window")
        scopes = {
            item.get("evidence_scope")
            for item in memory["structured_evidence"]["evidence_items"]
        }
        self.assertEqual(scopes, {"local_timestamp", "local_window"})

    def test_investigation_frontier_preserves_early_discriminating_cues(self):
        memory = init_observation_memory()
        memory["timestamped_observations"] = [
            {
                "source_tool": "overview",
                "timestamp_s": 313.8,
                "scene_id": "register",
                "description": "two women converse before one steps behind the counter",
                "needs_focus": "authorization or coercion before register access",
            },
            {
                "source_tool": "overview",
                "timestamp_s": 383.3,
                "scene_id": "register",
                "description": "person reaches into an open drawer",
                "needs_focus": "cash removal",
            },
        ]
        state = refresh_candidate_frontier(memory, duration=403.5)
        for index in range(20):
            state["candidates"].append(
                {
                    "candidate_id": f"L{index:02d}",
                    "t_range": [390.0 + index * 0.1, 391.0 + index * 0.1],
                    "status": "api_inspected_uncertain",
                    "possible_evidence": False,
                    "missing_evidence": "fine detail",
                    "summary": "late repeated view",
                }
            )
        text = format_candidate_frontier_for_prompt(
            memory,
            max_candidates=4,
            include_investigation_state=True,
            max_investigation_candidates=8,
        )
        self.assertIn("authorization or coercion", text)
        self.assertIn("cash removal", text)
        self.assertIn("not a gate or forced route", text)

    def test_planner_investigation_binding_is_persisted_without_routing(self):
        memory = init_observation_memory()
        memory["timestamped_observations"] = [
            {
                "source_tool": "overview",
                "timestamp_s": 313.8,
                "description": "interaction before counter access",
                "needs_focus": "authorization",
            }
        ]
        state = refresh_candidate_frontier(memory, duration=403.5)
        candidate_id = state["candidates"][0]["candidate_id"]
        record_planner_proposal(
            memory,
            planner_payload={
                "leading_hypothesis": "Stealing",
                "strongest_competitor": "Robbery",
                "decision_critical_evidence": "authorization or coercion",
                "investigation_target": {
                    "candidate_id": candidate_id,
                    "discriminator": "authorization or coercion",
                    "expected_information": "whether access was permitted",
                },
            },
            include_investigation_state=True,
        )
        binding = memory["candidate_frontier"]["active_investigation"]
        self.assertEqual(binding["candidate_id"], candidate_id)
        self.assertEqual(binding["strongest_competitor"], "Robbery")
        self.assertEqual(
            memory["candidate_frontier"]["candidates"][0]["planner_binding"],
            binding,
        )


class IntentPreservingRouterTest(unittest.TestCase):
    def _agent(self):
        agent = object.__new__(VideoSeekAgent)
        agent.config = {
            "focus_qwen_max_window_s": 20.0,
            "focus_qwen_context_pad_s": 2.0,
            "coseek1_intent_preserving_same_range_overlap": 0.85,
        }
        agent.allowed_tool_names = {"overview", "skim_qwen", "focus_qwen", "frame_verify", "answer"}
        agent.duration = 120.0
        agent.question = "What happens?"
        agent.observation_memory = init_observation_memory()
        agent._candidate_router_reason = ""
        return agent

    def test_broad_verify_becomes_same_range_skim(self):
        agent = self._agent()
        action = Action(
            "frame_verify",
            {
                "query": "compare a fall with a collision",
                "start_time": 20.0,
                "end_time": 80.0,
                "mode": "detail_verify",
            },
            "planner",
        )
        routed, thought = agent._VideoSeekAgent__route_broad_verify_to_same_range_skim(
            [action], step=1, thought="original thought"
        )
        self.assertEqual(thought, "original thought")
        self.assertEqual(routed[0].function_name, "skim_qwen")
        self.assertEqual(routed[0].parameters["start_time"], 20.0)
        self.assertEqual(routed[0].parameters["end_time"], 80.0)
        self.assertEqual(routed[0].parameters["query"], "compare a fall with a collision")

    def test_router_does_not_substitute_another_scene(self):
        agent = self._agent()
        agent.observation_memory["scene_memory"] = [
            {
                "source_tool": "overview",
                "scene_id": "S009",
                "t_range": [90.0, 110.0],
                "summary": "high static relevance candidate",
                "possible_evidence": True,
            }
        ]
        action = Action(
            "frame_verify",
            {"query": "inspect requested scene", "start_time": 0.0, "end_time": 50.0},
            "planner",
        )
        routed, _ = agent._VideoSeekAgent__route_broad_verify_to_same_range_skim(
            [action], step=1, thought="original"
        )
        self.assertEqual(
            [routed[0].parameters["start_time"], routed[0].parameters["end_time"]],
            [0.0, 50.0],
        )

    def test_existing_same_range_skim_preserves_planner_verify(self):
        agent = self._agent()
        agent.observation_memory["tool_observations"] = [
            {
                "tool": "skim_qwen",
                "parameters": {"start_time": 20.0, "end_time": 80.0},
                "parsed": True,
            }
        ]
        action = Action(
            "frame_verify",
            {"query": "verify", "start_time": 20.0, "end_time": 80.0},
            "planner",
        )
        routed, _ = agent._VideoSeekAgent__route_broad_verify_to_same_range_skim(
            [action], step=2, thought="original"
        )
        self.assertIs(routed[0], action)


if __name__ == "__main__":
    unittest.main()
