import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


def _ffprobe_binary() -> str:
    env_bin = Path(sys.executable).resolve().parent / "ffprobe"
    if env_bin.exists():
        return str(env_bin)
    return "ffprobe"


def _run_ffprobe(args: list[str], *, timeout: int = 60) -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            [_ffprobe_binary(), "-v", "error", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return json.loads(proc.stdout or "{}")
    except Exception:
        return None


@lru_cache(maxsize=64)
def probe_keyframes(video_path: str) -> tuple[float, ...]:
    if not video_path or not Path(video_path).exists():
        return ()
    data = _run_ffprobe(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,dts_time,flags",
            "-of",
            "json",
            video_path,
        ]
    )
    packets = (data or {}).get("packets") or []
    out: list[float] = []
    for packet in packets:
        if "K" not in str(packet.get("flags") or "").upper():
            continue
        try:
            timestamp = packet.get("pts_time", packet.get("dts_time"))
            out.append(float(timestamp))
        except Exception:
            continue
    return tuple(sorted(set(out)))


@lru_cache(maxsize=64)
def probe_packet_size_peaks(video_path: str) -> tuple[tuple[float, int], ...]:
    if not video_path or not Path(video_path).exists():
        return ()
    data = _run_ffprobe(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,size",
            "-of",
            "json",
            video_path,
        ]
    )
    packets = (data or {}).get("packets") or []
    out: list[tuple[float, int]] = []
    for packet in packets:
        try:
            ts = float(packet["pts_time"])
            size = int(packet.get("size") or 0)
        except Exception:
            continue
        if size > 0:
            out.append((ts, size))
    return tuple(out)


@lru_cache(maxsize=64)
def probe_interframe_packet_sizes(video_path: str) -> tuple[tuple[float, int], ...]:
    """Return timestamped P/B packet sizes as a local motion prior."""

    if not video_path or not Path(video_path).exists():
        return ()
    data = _run_ffprobe(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,dts_time,size,flags",
            "-of",
            "json",
            video_path,
        ]
    )
    packets = (data or {}).get("packets") or []
    out: list[tuple[float, int]] = []
    for packet in packets:
        if "K" in str(packet.get("flags") or "").upper():
            continue
        try:
            timestamp = packet.get("pts_time", packet.get("dts_time"))
            size = int(packet.get("size") or 0)
            if size > 0:
                out.append((float(timestamp), size))
        except Exception:
            continue
    return tuple(out)


def _spread(values: list[float], limit: int) -> list[float]:
    if limit <= 0 or not values:
        return []
    values = sorted(values)
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    indexes = sorted({round(i * (len(values) - 1) / (limit - 1)) for i in range(limit)})
    return [values[i] for i in indexes]


def keyframe_only_timestamps(
    *,
    video_path: str,
    start_time: float,
    end_time: float,
    num_frames: int,
) -> list[float]:
    """Select only real codec I-frame timestamps, spread over the window."""
    start = max(0.0, float(start_time))
    end = max(start, float(end_time))
    keyframes = [
        timestamp
        for timestamp in probe_keyframes(video_path)
        if start <= timestamp <= end
    ]
    return _spread(keyframes, max(0, int(num_frames)))


def _dedupe_sorted(values: list[float], *, min_gap_s: float) -> list[float]:
    out: list[float] = []
    for value in sorted(values):
        if all(abs(value - old) >= min_gap_s for old in out):
            out.append(value)
    return out


def codec_aware_timestamps(
    *,
    video_path: str,
    duration_s: float,
    start_time: float,
    end_time: float,
    num_frames: int,
    include_uniform: bool = True,
) -> list[float]:
    start = max(0.0, float(start_time))
    end = max(start, min(float(end_time), float(duration_s)))
    if num_frames <= 0 or end <= start:
        return []

    uniform_count = max(1, num_frames // 3) if include_uniform else 0
    key_count = max(1, num_frames // 3)
    peak_count = max(1, num_frames - uniform_count - key_count)

    chosen: list[float] = []
    if include_uniform:
        chosen.extend(np.linspace(start, end, uniform_count).astype(float).tolist())

    keyframes = [ts for ts in probe_keyframes(video_path) if start <= ts <= end]
    chosen.extend(_spread(keyframes, key_count))

    packets = [(ts, size) for ts, size in probe_packet_size_peaks(video_path) if start <= ts <= end]
    if packets:
        packets = sorted(packets, key=lambda item: item[1], reverse=True)
        peaks: list[float] = []
        min_gap = max(0.5, (end - start) / max(2, num_frames * 2))
        for ts, _size in packets:
            if all(abs(ts - old) >= min_gap for old in peaks):
                peaks.append(ts)
            if len(peaks) >= peak_count:
                break
        chosen.extend(peaks)

    if len(chosen) < num_frames:
        chosen.extend(np.linspace(start, end, num_frames).astype(float).tolist())

    min_gap = max(0.2, (end - start) / max(4, num_frames * 3))
    deduped = _dedupe_sorted(chosen, min_gap_s=min_gap)
    if len(deduped) < num_frames:
        deduped = _dedupe_sorted(
            deduped + np.linspace(start, end, num_frames).astype(float).tolist(),
            min_gap_s=max(0.05, min_gap / 2.0),
        )
    return _spread(deduped, num_frames)


def timestamps_to_frame_indices(
    *,
    timestamps: list[float],
    fps: float,
    total_frames: int,
) -> np.ndarray:
    if not timestamps or total_frames <= 0:
        return np.array([], dtype=int)
    indexes = [
        min(max(int(round(ts * fps)), 0), max(0, total_frames - 1))
        for ts in timestamps
    ]
    return np.array(sorted(set(indexes)), dtype=int)
