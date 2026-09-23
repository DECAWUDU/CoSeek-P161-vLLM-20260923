from __future__ import annotations

import unittest

from config import general_config
from run_mlvu import summarize_observer_usage
from videoseek.agent import VideoSeekAgent
from videoseek.core.action import Action
from videoseek.core.overview_admissibility import (
    BLOCKED_FEEDBACK_PREFIX,
    STATE_KEY,
    has_valid_overview_receipt,
    record_overview_attempt,
)
from videoseek.tools.v10_format import format_v10_observation


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _valid_overview() -> str:
    payload = {
        "global_summary": "A complete global routing map.",
        "timestamp_observations": [
            {
                "timestamp_s": float(index),
                "scene_id": "S1",
                "description": f"Frame {index}",
            }
            for index in range(64)
        ],
        "scene_summaries": [
            {
                "scene_id": "S1",
                "t_range": [0.0, 63.0],
                "summary": "One valid routing region.",
            }
        ],
        "overview_expected_timestamp_count": 64,
        "overview_observed_timestamp_count": 64,
        "overview_timestamp_coverage": 1.0,
        "observer_backend": "api",
    }
    return format_v10_observation(payload)


class _Registry:
    def __init__(self) -> None:
        self.call_count = 0

    def has_tool(self, _name: str) -> bool:
        return True

    def get_function(self, _name: str):
        def execute(*, config: dict, parameters: dict) -> str:
            del config, parameters
            self.call_count += 1
            return _valid_overview()

        return execute


def _agent(*, enabled: bool) -> VideoSeekAgent:
    agent = VideoSeekAgent.__new__(VideoSeekAgent)
    agent.config = {
        "coseek1_valid_overview_receipt_guard_enabled": enabled,
        "coseek1_planner_json_action": True,
        "coseek1_structured_planner": True,
        "coseek1_localize_qwen_enabled": True,
        "coseek1_candidate_frontier": False,
        "coseek1_evidence_episode_frontier": False,
        "coseek1_direct_planner_answer": True,
        "overview_require_all_timestamps": True,
        "coseek1_verified_observation_reuse_enabled": False,
        "coseek1_tool_integrity_repair_enabled": False,
    }
    agent.max_steps = 20
    agent.observation_memory = {}
    agent.trajectory_steps = []
    agent.tools = [_tool(name) for name in (
        "overview", "localize_qwen", "frame_verify", "answer"
    )]
    agent.allowed_tool_names = {
        tool["function"]["name"] for tool in agent.tools
    }
    agent.tool_registry = _Registry()
    agent.vr = object()
    agent.subtitles = []
    agent.video_path = "/tmp/video.mp4"
    agent.question = "Question"
    agent.duration = 64.0
    agent.output_dir = "/tmp"
    return agent


class P105AgentIntegrationTest(unittest.TestCase):
    def test_default_flag_is_off(self) -> None:
        self.assertIs(
            general_config["coseek1_valid_overview_receipt_guard_enabled"],
            False,
        )

    def test_flag_off_prompt_retains_original_overview_contract(self) -> None:
        agent = _agent(enabled=False)
        prompt = agent._VideoSeekAgent__format_planner_step_prompt(
            step=1,
            memory_text="memory",
        )
        self.assertIn("overview|localize_qwen|frame_verify|answer", prompt)
        self.assertIn("For overview, parameters must be {}.", prompt)
        self.assertNotIn("valid_overview_completed=true", prompt)
        self.assertEqual(
            [tool["function"]["name"] for tool in agent._VideoSeekAgent__planner_action_tools()],
            ["overview", "localize_qwen", "frame_verify", "answer"],
        )

    def test_legacy_agent_without_adaptive_budget_still_formats_prompt(self) -> None:
        agent = _agent(enabled=False)
        self.assertFalse(hasattr(agent, "_adaptive_token_budget"))

        prompt = agent._VideoSeekAgent__format_planner_step_prompt(
            step=1,
            memory_text="legacy memory",
        )

        self.assertIn("legacy memory", prompt)
        self.assertNotIn("Adaptive token budget:", prompt)

    def test_valid_receipt_masks_prompt_and_fallback_schema(self) -> None:
        agent = _agent(enabled=True)
        record_overview_attempt(
            agent.observation_memory,
            _valid_overview(),
            True,
            1,
        )
        prompt = agent._VideoSeekAgent__format_planner_step_prompt(
            step=1,
            memory_text="memory",
        )
        self.assertIn("localize_qwen|frame_verify|answer", prompt)
        self.assertNotIn("overview|localize_qwen", prompt)
        self.assertNotIn("For overview, parameters must be {}.", prompt)
        self.assertIn("valid_overview_completed=true", prompt)
        self.assertEqual(
            [tool["function"]["name"] for tool in agent._VideoSeekAgent__planner_action_tools()],
            ["localize_qwen", "frame_verify", "answer"],
        )

    def test_executor_blocks_without_call_or_visual_memory_merge(self) -> None:
        agent = _agent(enabled=True)
        record_overview_attempt(
            agent.observation_memory,
            _valid_overview(),
            True,
            1,
        )
        frozen_tool_observations = list(
            agent.observation_memory.get("tool_observations") or []
        )
        output = agent._VideoSeekAgent__exec_action(
            Action(function_name="overview", parameters={}, function_id="repeat")
        )
        self.assertTrue(output.startswith(BLOCKED_FEEDBACK_PREFIX))
        self.assertEqual(agent.tool_registry.call_count, 0)
        self.assertEqual(
            agent.observation_memory.get("tool_observations") or [],
            frozen_tool_observations,
        )
        self.assertEqual(
            agent.observation_memory[STATE_KEY]["blocked_overview_count"],
            1,
        )

    def test_failed_attempt_remains_executable(self) -> None:
        agent = _agent(enabled=True)
        record_overview_attempt(
            agent.observation_memory,
            "Overview observation failed: model response is empty.",
            True,
            1,
        )
        self.assertIs(has_valid_overview_receipt(agent.observation_memory), False)
        output = agent._VideoSeekAgent__exec_action(
            Action(function_name="overview", parameters={}, function_id="retry")
        )
        self.assertIn("V10_OBSERVATION_JSON", output)
        self.assertEqual(agent.tool_registry.call_count, 1)

    def test_blocked_overview_is_not_counted_as_visual_tool_execution(self) -> None:
        usage = summarize_observer_usage(
            {
                "steps": [
                    {
                        "action": {"function": "overview", "parameters": {}},
                        "routing_audit": {
                            "executor_blocked": True,
                            "executor_guard": "valid_overview_already_completed",
                        },
                        "observation": BLOCKED_FEEDBACK_PREFIX + "{}",
                    }
                ]
            }
        )
        self.assertEqual(usage["api_visual_tool_count"], 0)
        self.assertEqual(usage["admissibility_rejection_count"], 1)
        self.assertEqual(usage["post_success_overview_blocked_count"], 1)


if __name__ == "__main__":
    unittest.main()
