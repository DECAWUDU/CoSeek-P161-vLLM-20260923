from __future__ import annotations

import base64
import json
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
from PIL import Image

from videoseek.core.memory import extract_v10_payload
from videoseek.tools.focus import _direct_packet_anchor_positions
from videoseek.tools.frame_verify import execute_frame_verify


class _Batch:
    def __init__(self, frames):
        self._frames = frames

    def asnumpy(self):
        return self._frames


class _VideoReaderStub:
    def __len__(self):
        return 3000

    def get_avg_fps(self):
        return 30.0

    def get_batch(self, indices):
        frames = np.zeros((len(indices), 240, 320, 3), dtype=np.uint8)
        for position in range(len(indices)):
            frames[position, :, :, :] = position * 8
        return _Batch(frames)


def _image_item():
    image = Image.new("RGB", (64, 64), color=(120, 30, 20))
    output = BytesIO()
    image.save(output, format="JPEG")
    url = "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode(
        "utf-8"
    )
    return {"type": "image_url", "image_url": {"url": url, "detail": "high"}}


def _parameters():
    return {
        "vr": _VideoReaderStub(),
        "video_path": "/tmp/video.mp4",
        "duration": 100.0,
        "output_dir": "/tmp",
        "subtitles": [],
        "question": "What is held?\n(A) cup\n(B) flower\n(C) book\n(D) phone",
        "query": "Neutrally inspect what the person holds.",
        "start_time": 10.0,
        "end_time": 20.0,
        "mode": "option_verify",
    }


def _response():
    return json.dumps(
        {
            "timestamp_observations": [
                {
                    "timestamp_s": 15.0,
                    "description": "The person holds flower stems.",
                }
            ],
            "overall_summary": "The held object is visible.",
            "target_match": "matched",
            "scope_coverage": "sufficient",
            "supports_options": ["B"],
            "contradicts_options": ["A", "C", "D"],
            "option_evidence": {},
            "detail_sufficient": True,
            "missing_detail": "",
        }
    )


class DirectVerifyPacketP4Tests(unittest.TestCase):
    def test_anchor_selection_keeps_center_and_change(self):
        frames = np.zeros((8, 16, 16, 3), dtype=np.uint8)
        frames[6:] = 255
        positions = _direct_packet_anchor_positions(frames, 2)
        self.assertEqual(len(positions), 2)
        self.assertIn(3, positions)
        self.assertTrue(any(position >= 6 for position in positions))

    def test_direct_packet_preserves_frames_in_five_api_images(self):
        observed_content = []

        def fake_observe(config, *, content, tool_name, tool_mode, output_dir):
            observed_content.extend(content)
            return _response(), "api"

        crop = _image_item()

        def fake_grounding(config, **kwargs):
            frame_count = len(kwargs["frames"])
            return (
                {position: [crop] for position in range(frame_count)},
                {
                    "enabled": True,
                    "records": [
                        {"position": position, "packet_mode": "crop_only"}
                        for position in range(frame_count)
                    ],
                },
            )

        with patch(
            "videoseek.tools.focus.scene_aware_timestamps",
            return_value=np.linspace(10.0, 20.0, 16).tolist(),
        ), patch(
            "videoseek.tools.focus.get_scene_context_for_window",
            return_value=(None, None),
        ), patch(
            "videoseek.tools.focus.prepare_grounded_frames",
            side_effect=fake_grounding,
        ), patch(
            "videoseek.tools.focus.observe_content",
            side_effect=fake_observe,
        ):
            output = execute_frame_verify(
                {
                    "grounded_frame_verify_enabled": True,
                    "grounded_verify_packet_enabled": True,
                    "grounded_direct_verify_packet_enabled": True,
                    "grounded_direct_verify_packet_group_size": 4,
                    "grounded_direct_verify_packet_anchor_count": 2,
                    "grounded_direct_verify_packet_detail_max_side": 256,
                    "local_qwen_max_images": 24,
                },
                _parameters(),
            )

        images = [item for item in observed_content if item.get("type") == "image_url"]
        self.assertEqual(len(images), 5)
        self.assertEqual(
            sum(item["image_url"].get("detail") == "low" for item in images),
            4,
        )
        self.assertEqual(
            sum(item["image_url"].get("detail") == "high" for item in images),
            1,
        )
        payload = extract_v10_payload(output)
        self.assertEqual(payload["visual_layout"], "direct_verify_packet")
        self.assertEqual(payload["direct_packet_audit"]["source_frame_count"], 16)
        self.assertEqual(payload["direct_packet_audit"]["api_image_count"], 5)
        self.assertEqual(len(payload["direct_packet_audit"]["anchor_timestamps_s"]), 2)


if __name__ == "__main__":
    unittest.main()
