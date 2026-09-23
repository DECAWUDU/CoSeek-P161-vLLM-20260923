import unittest

from config import general_config, prompts_config
from videoseek.core.memory import format_memory_for_prompt, init_observation_memory


class V33MemoryOnlyTests(unittest.TestCase):
    def test_episode_layers_are_disabled(self):
        self.assertFalse(general_config["coseek1_evidence_episode_frontier"])
        self.assertFalse(general_config["coseek1_episode_backend_router"])
        self.assertFalse(general_config["coseek1_answer_evidence_audit"])
        self.assertFalse(general_config["coseek1_candidate_frontier"])
        self.assertFalse(general_config["coseek1_frontier_temporal_boundaries"])
        self.assertTrue(general_config["use_structured_evidence_state"])
        self.assertTrue(general_config["structured_evidence_include_planner_state"])

    def test_prompt_uses_timestamped_and_structured_memory(self):
        prompt = prompts_config["SYSTEM_PROMPT"]
        self.assertIn("Timestamped Observation Memory", prompt)
        self.assertIn("Structured Evidence", prompt)
        self.assertNotIn("Evidence Episode Frontier", prompt)
        self.assertNotIn("episode_id", prompt)

    def test_memory_prompt_does_not_render_episode_frontier(self):
        memory = init_observation_memory()
        self.assertNotIn("evidence_episode_frontier", memory)
        self.assertNotIn("answer_evidence_audit", memory)
        memory["timestamped_observations"].append(
            {
                "timestamp_s": 12.0,
                "source_tool": "overview",
                "scene_id": "S001",
                "description": "A person enters the room.",
                "evidence_scope": "local_timestamp",
            }
        )
        text = format_memory_for_prompt(
            memory,
            question="Who enters the room?",
            include_evidence_episode_frontier=False,
            include_candidate_frontier=False,
            include_structured_evidence=True,
        )
        self.assertIn("Observed timeline", text)
        self.assertIn("12.0s", text)
        self.assertNotIn("Evidence Episode Frontier", text)


if __name__ == "__main__":
    unittest.main()
