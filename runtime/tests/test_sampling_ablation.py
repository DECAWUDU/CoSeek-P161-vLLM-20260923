from unittest.mock import patch

from videoseek import codec
from videoseek.codec import keyframe_only_timestamps
from videoseek.tools.focus_qwen import focus_qwen_frame_budget


def test_keyframe_only_sampling_spreads_requested_budget():
    keyframes = tuple(float(index) for index in range(100))
    with patch("videoseek.codec.probe_keyframes", return_value=keyframes):
        sampled = keyframe_only_timestamps(
            video_path="unused.mp4",
            start_time=0.0,
            end_time=99.0,
            num_frames=64,
        )

    assert len(sampled) == 64
    assert sampled == sorted(sampled)
    assert set(sampled).issubset(set(keyframes))


def test_probe_keyframes_uses_compressed_packet_flags(tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.touch()
    packet_data = {
        "packets": [
            {"pts_time": "0.0", "flags": "K_"},
            {"pts_time": "0.04", "flags": "__"},
            {"dts_time": "2.0", "flags": "K_"},
        ]
    }
    with patch("videoseek.codec._run_ffprobe", return_value=packet_data):
        codec.probe_keyframes.cache_clear()
        keyframes = codec.probe_keyframes(str(video_path))

    assert keyframes == (0.0, 2.0)


def test_focus_density_uses_one_frame_per_second():
    config = {
        "focus_qwen_max_frames": 24,
        "focus_qwen_temporal_density_enabled": True,
        "focus_qwen_target_fps": 1.0,
    }

    assert focus_qwen_frame_budget(config, window_s=6.2) == 7
    assert focus_qwen_frame_budget(config, window_s=20.0) == 20
    assert focus_qwen_frame_budget(config, window_s=40.0) == 24


def test_focus_baseline_keeps_fixed_cap():
    config = {
        "focus_qwen_max_frames": 8,
        "focus_qwen_temporal_density_enabled": False,
        "focus_qwen_target_fps": 1.0,
    }

    assert focus_qwen_frame_budget(config, window_s=20.0) == 8
