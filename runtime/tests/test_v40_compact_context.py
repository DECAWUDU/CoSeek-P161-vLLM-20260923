import unittest

from videoseek.agent import VideoSeekAgent


class CompactContextTest(unittest.TestCase):
    def _agent(self):
        agent = VideoSeekAgent.__new__(VideoSeekAgent)
        agent.config = {
            "coseek1_compact_planner_context": True,
            "coseek1_compact_answer_context": True,
        }
        agent.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "user", "content": "old memory"},
            {"role": "assistant", "content": "old thought"},
            {"role": "tool", "content": "large raw observation"},
            {"role": "user", "content": "latest cumulative memory"},
        ]
        return agent

    def test_planner_keeps_only_stable_context_and_latest_memory(self):
        agent = self._agent()
        messages = agent._VideoSeekAgent__planner_api_messages()
        self.assertEqual([item["content"] for item in messages], [
            "system",
            "question",
            "latest cumulative memory",
        ])
        self.assertEqual(len(agent.messages), 6)

    def test_final_answer_keeps_only_stable_context_and_digest(self):
        agent = self._agent()
        messages = agent._VideoSeekAgent__final_answer_api_messages()
        self.assertEqual([item["content"] for item in messages], [
            "system",
            "question",
            "latest cumulative memory",
        ])


if __name__ == "__main__":
    unittest.main()
