import json
import tempfile
import unittest
from pathlib import Path

from run_mlvu import read_api_usage_log, summarize_api_usage


class EndToEndUsageTest(unittest.TestCase):
    def test_per_question_api_usage_summary_splits_visual_and_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "usage.jsonl"
            rows = [
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                    "image_count": 4,
                    "has_images": True,
                },
                {
                    "prompt_tokens": 40,
                    "completion_tokens": 5,
                    "total_tokens": 45,
                    "image_count": 0,
                    "has_images": False,
                },
                {
                    "prompt_tokens": 70,
                    "completion_tokens": 7,
                    "total_tokens": 77,
                    "image_count": 2,
                    "has_images": True,
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n{partial")

            summary = summarize_api_usage(read_api_usage_log(path))

        self.assertEqual(
            summary,
            {
                "api_request_count": 3,
                "api_visual_request_count": 2,
                "api_text_request_count": 1,
                "api_image_count": 6,
                "api_prompt_tokens": 210,
                "api_completion_tokens": 22,
                "api_total_tokens": 232,
                "api_visual_prompt_tokens": 170,
                "api_visual_completion_tokens": 17,
                "api_visual_total_tokens": 187,
                "api_text_prompt_tokens": 40,
                "api_text_completion_tokens": 5,
                "api_text_total_tokens": 45,
            },
        )


if __name__ == "__main__":
    unittest.main()
