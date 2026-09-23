from __future__ import annotations

import base64
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from videoseek.codec import (
    codec_aware_timestamps,
    probe_keyframes,
    probe_packet_size_peaks,
    timestamps_to_frame_indices,
)
from videoseek.observer import observe_content
from videoseek.utils import extract_json_object


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _video_cache_path(config: dict, video_path: str) -> Path:
    cache_root = Path(config.get("skeleton_cache_dir") or "./runs/_skeleton_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    path = Path(video_path)
    stat = path.stat() if path.exists() else None
    key = {
        "path": str(path.resolve()) if path.exists() else str(path),
        "mtime": int(stat.st_mtime) if stat else 0,
        "size": int(stat.st_size) if stat else 0,
        "scene_max_s": config.get("skeleton_scene_max_s"),
        "scene_min_s": config.get("skeleton_scene_min_s"),
        "max_scenes": config.get("skeleton_max_scenes"),
        "anchors_per_scene": config.get("skeleton_anchors_per_scene"),
        "caption_strategy": config.get("skeleton_caption_strategy"),
        "caption_frames": config.get("skeleton_caption_frames"),
        "caption_frames_per_scene": config.get("skeleton_caption_frames_per_scene"),
        "caption_scenes_per_chunk": config.get("skeleton_caption_scenes_per_chunk"),
        "caption_short_side": config.get("skeleton_caption_short_side"),
        "single_frame_short_side": config.get("skeleton_single_frame_short_side"),
        "single_frame_aggregate": config.get("skeleton_single_frame_aggregate"),
        "mini_sheet_group_size": config.get("skeleton_mini_sheet_group_size"),
        "mini_sheet_short_side": config.get("skeleton_mini_sheet_short_side"),
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_root / f"{path.stem}_{digest}.skeleton.json"


def _dedupe_sorted(values: list[float], *, min_gap_s: float) -> list[float]:
    out: list[float] = []
    for value in sorted(values):
        if all(abs(value - old) >= min_gap_s for old in out):
            out.append(round(float(value), 1))
    return out


def _spread(values: list[float], limit: int) -> list[float]:
    if limit <= 0 or not values:
        return []
    values = sorted(values)
    if len(values) <= limit:
        return [round(float(v), 1) for v in values]
    if limit == 1:
        return [round(float(values[len(values) // 2]), 1)]
    idxs = sorted({round(i * (len(values) - 1) / (limit - 1)) for i in range(limit)})
    return [round(float(values[i]), 1) for i in idxs]


def _select_boundaries(
    *,
    video_path: str,
    duration_s: float,
    min_scene_s: float,
    max_scene_s: float,
    max_scenes: int,
) -> list[float]:
    duration_s = max(0.0, duration_s)
    if duration_s <= 0:
        return [0.0]

    packets = list(probe_packet_size_peaks(video_path))
    keyframes = list(probe_keyframes(video_path))
    candidate_scores: dict[float, float] = {}
    if packets:
        for ts, size in packets:
            ts = round(float(ts), 1)
            if min_scene_s <= ts <= duration_s - min_scene_s:
                candidate_scores[ts] = max(candidate_scores.get(ts, 0.0), float(size))
    for ts in keyframes:
        ts = round(float(ts), 1)
        if min_scene_s <= ts <= duration_s - min_scene_s:
            candidate_scores[ts] = max(candidate_scores.get(ts, 0.0), 1.0)

    candidates = _dedupe_sorted(
        list(candidate_scores.keys()),
        min_gap_s=max(2.0, min_scene_s / 2.0),
    )

    boundaries = [0.0, round(duration_s, 1)]

    # First guarantee broad temporal coverage. Each nominal split is snapped to
    # a nearby codec candidate when possible, preventing a long tail window.
    nominal = max_scene_s
    while nominal < duration_s - min_scene_s and len(boundaries) < max_scenes + 1:
        search_radius = max(max_scene_s / 3.0, min_scene_s)
        nearby = [
            ts
            for ts in candidates
            if abs(ts - nominal) <= search_radius
            and all(abs(ts - old) >= min_scene_s for old in boundaries)
        ]
        split = max(nearby, key=lambda ts: candidate_scores.get(ts, 0.0)) if nearby else round(nominal, 1)
        merged = sorted(boundaries + [split])
        if all((b - a) >= min_scene_s for a, b in zip(merged, merged[1:])):
            boundaries = merged
        nominal += max_scene_s

    # Use remaining budget on the currently longest segment, selecting the
    # highest-scored codec candidate inside that segment. This distributes
    # extra detail instead of over-fragmenting the first high-motion region.
    while len(boundaries) < max_scenes + 1:
        segments = sorted(
            list(zip(boundaries, boundaries[1:])),
            key=lambda pair: pair[1] - pair[0],
            reverse=True,
        )
        if not segments or (segments[0][1] - segments[0][0]) < max(min_scene_s * 2.0, max_scene_s * 0.6):
            break
        start, end = segments[0]
        inside = [
            ts
            for ts in candidates
            if start + min_scene_s <= ts <= end - min_scene_s
            and all(abs(ts - old) >= min_scene_s for old in boundaries)
        ]
        if inside:
            midpoint = (start + end) / 2.0
            split = max(
                inside,
                key=lambda ts: (
                    candidate_scores.get(ts, 0.0),
                    -abs(ts - midpoint),
                ),
            )
        else:
            split = round((start + end) / 2.0, 1)
        merged = sorted(boundaries + [round(float(split), 1)])
        if merged == boundaries or not all((b - a) >= min_scene_s for a, b in zip(merged, merged[1:])):
            break
        boundaries = merged

    return sorted(set(round(float(item), 1) for item in boundaries))


def _scene_motion_peaks(video_path: str, start: float, end: float, limit: int = 4) -> list[float]:
    packets = [
        (ts, size)
        for ts, size in probe_packet_size_peaks(video_path)
        if start <= float(ts) <= end
    ]
    if not packets:
        return []
    peaks: list[float] = []
    min_gap = max(0.5, (end - start) / max(2, limit * 2))
    for ts, _size in sorted(packets, key=lambda item: item[1], reverse=True):
        if all(abs(ts - old) >= min_gap for old in peaks):
            peaks.append(round(float(ts), 1))
        if len(peaks) >= limit:
            break
    return sorted(peaks)


def build_codec_skeleton(config: dict, *, video_path: str, duration_s: float) -> dict[str, Any]:
    min_scene_s = _as_float(config.get("skeleton_scene_min_s"), 12.0)
    max_scene_s = _as_float(config.get("skeleton_scene_max_s"), 90.0)
    max_scenes = _as_int(config.get("skeleton_max_scenes"), 24)
    anchors_per_scene = _as_int(config.get("skeleton_anchors_per_scene"), 5)

    boundaries = _select_boundaries(
        video_path=video_path,
        duration_s=duration_s,
        min_scene_s=min_scene_s,
        max_scene_s=max_scene_s,
        max_scenes=max_scenes,
    )
    if len(boundaries) < 2:
        boundaries = [0.0, round(duration_s, 1)]

    scenes: list[dict[str, Any]] = []
    for idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        anchors = codec_aware_timestamps(
            video_path=video_path,
            duration_s=duration_s,
            start_time=start,
            end_time=end,
            num_frames=anchors_per_scene,
        )
        scenes.append(
            {
                "scene_id": f"S{idx:03d}",
                "t_range": [round(float(start), 1), round(float(end), 1)],
                "anchor_timestamps": [round(float(ts), 1) for ts in anchors],
                "motion_peaks": _scene_motion_peaks(video_path, start, end),
                "scene_caption": "",
                "keyframe_observations": [],
                "objects": [],
                "actions": [],
                "possible_events": [],
            }
        )

    return {
        "version": "coseek_v10_1_codec_skeleton",
        "video_path": str(video_path),
        "duration_s": round(float(duration_s), 1),
        "scene_count": len(scenes),
        "scenes": scenes,
    }


def load_or_build_skeleton(config: dict, *, video_path: str, duration_s: float) -> tuple[dict[str, Any], Path]:
    cache_path = _video_cache_path(config, video_path)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8")), cache_path
        except Exception:
            pass
    skeleton = build_codec_skeleton(config, video_path=video_path, duration_s=duration_s)
    cache_path.write_text(json.dumps(skeleton, ensure_ascii=False, indent=2), encoding="utf-8")
    return skeleton, cache_path


def save_skeleton(config: dict, *, video_path: str, skeleton: dict[str, Any]) -> Path:
    cache_path = _video_cache_path(config, video_path)
    cache_path.write_text(json.dumps(skeleton, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_path


def _range_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _find_scene_for_window(
    skeleton: dict[str, Any],
    *,
    start_time: float,
    end_time: float,
) -> dict[str, Any] | None:
    scenes = [scene for scene in (skeleton.get("scenes") or []) if isinstance(scene, dict)]
    if not scenes:
        return None
    midpoint = (float(start_time) + float(end_time)) / 2.0
    best_scene: dict[str, Any] | None = None
    best_score: tuple[float, float] | None = None
    for scene in scenes:
        try:
            scene_start, scene_end = scene.get("t_range") or [0.0, 0.0]
            scene_start = float(scene_start)
            scene_end = float(scene_end)
        except Exception:
            continue
        overlap = _range_overlap(float(start_time), float(end_time), scene_start, scene_end)
        distance = abs(midpoint - ((scene_start + scene_end) / 2.0))
        score = (overlap, -distance)
        if best_score is None or score > best_score:
            best_scene = scene
            best_score = score
    if not best_scene or not best_score or best_score[0] <= 0:
        return None
    return best_scene


def get_scene_context_for_window(
    config: dict,
    *,
    video_path: str,
    duration_s: float,
    start_time: float,
    end_time: float,
) -> tuple[dict[str, Any] | None, Path | None]:
    if not video_path or not bool(config.get("scene_aware_skim", True)):
        return None, None
    try:
        skeleton, cache_path = load_or_build_skeleton(
            config,
            video_path=video_path,
            duration_s=duration_s,
        )
    except Exception:
        return None, None
    return _find_scene_for_window(
        skeleton,
        start_time=float(start_time),
        end_time=float(end_time),
    ), cache_path


def scene_context_to_prompt_text(scene: dict[str, Any] | None) -> str:
    if not scene:
        return ""
    observations = scene.get("keyframe_observations") or []
    obs_text = "; ".join(
        f"{item.get('timestamp_s')}s: {item.get('description')}"
        for item in observations[:4]
        if isinstance(item, dict)
    )
    return "\n".join(
        [
            f"Matched skeleton scene: {scene.get('scene_id')} {scene.get('t_range')}",
            f"scene_caption: {scene.get('scene_caption') or ''}",
            f"anchor_timestamps: {scene.get('anchor_timestamps') or []}",
            f"motion_peaks: {scene.get('motion_peaks') or []}",
            f"objects: {scene.get('objects') or []}",
            f"actions: {scene.get('actions') or []}",
            f"possible_events: {scene.get('possible_events') or []}",
            f"keyframe_observations: {obs_text}",
        ]
    )


def scene_aware_timestamps(
    *,
    video_path: str,
    duration_s: float,
    start_time: float,
    end_time: float,
    num_frames: int,
    scene: dict[str, Any] | None,
) -> list[float]:
    base = codec_aware_timestamps(
        video_path=video_path,
        duration_s=duration_s,
        start_time=start_time,
        end_time=end_time,
        num_frames=num_frames,
    )
    if not scene:
        return base

    scene_candidates: list[float] = []
    for key in ("motion_peaks", "anchor_timestamps"):
        for value in scene.get(key) or []:
            try:
                ts = round(float(value), 1)
            except Exception:
                continue
            if float(start_time) <= ts <= float(end_time):
                scene_candidates.append(ts)
    for item in scene.get("keyframe_observations") or []:
        if not isinstance(item, dict):
            continue
        try:
            ts = round(float(item.get("timestamp_s")), 1)
        except Exception:
            continue
        if float(start_time) <= ts <= float(end_time):
            scene_candidates.append(ts)

    if not scene_candidates:
        return base
    min_gap = max(0.05, (float(end_time) - float(start_time)) / max(4, num_frames * 3))
    merged = _dedupe_sorted(scene_candidates + base, min_gap_s=min_gap)
    return _spread(merged, num_frames)


def _frame_to_data_url(frame: np.ndarray, *, short_side: int) -> str:
    image = Image.fromarray(frame)
    width, height = image.size
    shortest = max(1, min(width, height))
    scale = float(short_side) / float(shortest)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    image = image.resize(target, Image.BICUBIC)
    output = BytesIO()
    image.save(output, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("utf-8")


def _resize_frame(frame: np.ndarray, *, short_side: int) -> np.ndarray:
    image = Image.fromarray(frame)
    width, height = image.size
    shortest = max(1, min(width, height))
    scale = float(short_side) / float(shortest)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.array(image.resize(target, Image.BICUBIC))


def _array_to_data_url(image_array: np.ndarray) -> str:
    image = Image.fromarray(image_array)
    output = BytesIO()
    image.save(output, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("utf-8")


def _caption_frame_plan(
    skeleton: dict[str, Any],
    *,
    max_frames: int,
    frames_per_scene: int,
) -> list[tuple[str, float]]:
    per_scene: list[list[tuple[str, float]]] = []
    for scene in skeleton.get("scenes") or []:
        scene_id = scene.get("scene_id") or "S?"
        peaks = [round(float(ts), 1) for ts in (scene.get("motion_peaks") or [])]
        anchors = [round(float(ts), 1) for ts in (scene.get("anchor_timestamps") or [])]
        selected: list[float] = []
        peak_slots = min(max(0, frames_per_scene - 1), 3)
        anchor_slots = max(0, frames_per_scene - peak_slots)
        for ts in _spread(peaks, peak_slots):
            if ts not in selected:
                selected.append(ts)
        for ts in _spread(anchors, anchor_slots):
            if ts not in selected:
                selected.append(ts)
        if not selected:
            start, end = scene.get("t_range") or [0.0, 0.0]
            selected = [round((float(start) + float(end)) / 2.0, 1)]
        per_scene.append([(scene_id, ts) for ts in selected[:frames_per_scene]])

    planned: list[tuple[str, float]] = []
    # Round-robin keeps broad scene coverage while giving every scene its most
    # important motion peaks before lower-priority anchors.
    while len(planned) < max_frames and any(per_scene):
        progressed = False
        for items in per_scene:
            if not items or len(planned) >= max_frames:
                continue
            planned.append(items.pop(0))
            progressed = True
        if not progressed:
            break
    if len(planned) <= max_frames:
        return planned
    idxs = sorted({round(i * (len(planned) - 1) / (max_frames - 1)) for i in range(max_frames)})
    return [planned[i] for i in idxs]


def _caption_iframe_sheet_plan(
    skeleton: dict[str, Any],
    *,
    video_path: str,
    max_frames: int,
    frames_per_scene: int,
) -> list[tuple[str, float]]:
    keyframes = [round(float(ts), 1) for ts in probe_keyframes(video_path)]
    per_scene: list[list[tuple[str, float]]] = []
    for scene in skeleton.get("scenes") or []:
        scene_id = scene.get("scene_id") or "S?"
        try:
            start, end = scene.get("t_range") or [0.0, 0.0]
            start = float(start)
            end = float(end)
        except Exception:
            start, end = 0.0, 0.0
        scene_keyframes = [ts for ts in keyframes if start <= ts <= end]
        peaks = [round(float(ts), 1) for ts in (scene.get("motion_peaks") or [])]
        anchors = [round(float(ts), 1) for ts in (scene.get("anchor_timestamps") or [])]

        selected: list[float] = []
        for source in (_spread(scene_keyframes, frames_per_scene), _spread(peaks, frames_per_scene), anchors):
            for ts in source:
                ts = round(float(ts), 1)
                if start <= ts <= end and ts not in selected:
                    selected.append(ts)
                if len(selected) >= frames_per_scene:
                    break
            if len(selected) >= frames_per_scene:
                break
        if not selected:
            selected = [round((start + end) / 2.0, 1)]
        per_scene.append([(scene_id, ts) for ts in selected[:frames_per_scene]])

    planned: list[tuple[str, float]] = []
    while len(planned) < max_frames and any(per_scene):
        progressed = False
        for items in per_scene:
            if not items or len(planned) >= max_frames:
                continue
            planned.append(items.pop(0))
            progressed = True
        if not progressed:
            break
    return planned


def _frame_caption_fallback(
    *,
    scene_id: str,
    timestamp_s: float,
    raw: str | None,
) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    return {
        "scene_id": scene_id,
        "timestamp_s": round(float(timestamp_s), 1),
        "caption": text[:500],
        "objects": [],
        "visible_actions": [],
        "text_on_screen": "",
        "uncertainty": "unknown",
    }


def _deterministic_scene_aggregate(
    skeleton: dict[str, Any],
    frame_observations: list[dict[str, Any]],
) -> tuple[int, str]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for item in frame_observations:
        scene_id = str(item.get("scene_id") or "")
        if scene_id:
            by_scene.setdefault(scene_id, []).append(item)

    captioned = 0
    summaries: list[str] = []
    for scene in skeleton.get("scenes") or []:
        scene_id = scene.get("scene_id")
        items = sorted(
            by_scene.get(scene_id) or [],
            key=lambda item: float(item.get("timestamp_s") or 0.0),
        )
        if not items:
            continue
        observations = []
        objects: list[str] = []
        actions: list[str] = []
        for item in items:
            caption = str(item.get("caption") or item.get("description") or "").strip()
            if caption:
                observations.append(
                    {
                        "timestamp_s": round(float(item.get("timestamp_s") or 0.0), 1),
                        "description": caption,
                    }
                )
            for obj in item.get("objects") or []:
                text = str(obj).strip()
                if text and text not in objects:
                    objects.append(text)
            for act in item.get("visible_actions") or item.get("actions") or []:
                text = str(act).strip()
                if text and text not in actions:
                    actions.append(text)
        caption_text = "; ".join(obs["description"] for obs in observations[:4])
        if caption_text:
            scene["scene_caption"] = caption_text
            scene["keyframe_observations"] = observations
            scene["objects"] = objects[:12]
            scene["actions"] = actions[:12]
            scene["possible_events"] = actions[:8]
            captioned += 1
            summaries.append(f"{scene_id}: {caption_text}")
    return captioned, " ".join(summaries[:12])


def _aggregate_single_frame_captions(
    config: dict,
    *,
    skeleton: dict[str, Any],
    frame_observations: list[dict[str, Any]],
    output_dir: str | None,
) -> tuple[dict[str, Any], str, float]:
    t0 = time.time()
    if not bool(config.get("skeleton_single_frame_aggregate", True)):
        captioned, summary = _deterministic_scene_aggregate(skeleton, frame_observations)
        skeleton["skeleton_summary"] = summary
        return skeleton, "deterministic", time.time() - t0

    scenes = skeleton.get("scenes") or []
    scene_ranges = {
        scene.get("scene_id"): scene.get("t_range")
        for scene in scenes
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in frame_observations:
        scene_id = str(item.get("scene_id") or "")
        if scene_id:
            grouped.setdefault(scene_id, []).append(item)

    lines = []
    for scene in scenes:
        scene_id = scene.get("scene_id")
        items = sorted(
            grouped.get(scene_id) or [],
            key=lambda item: float(item.get("timestamp_s") or 0.0),
        )
        if not items:
            continue
        lines.append(f"{scene_id} {scene_ranges.get(scene_id)}")
        for item in items:
            caption = item.get("caption") or item.get("description") or ""
            objects = item.get("objects") or []
            actions = item.get("visible_actions") or item.get("actions") or []
            lines.append(
                f"- {float(item.get('timestamp_s') or 0.0):.1f}s: {caption} "
                f"objects={objects} actions={actions}"
            )

    content = [
        {
            "type": "text",
            "text": (
                "You are aggregating single-frame captions into an offline video skeleton memory.\n"
                "Do not answer any user question. Merge only the provided timestamped observations.\n"
                "Keep scene summaries concise and factual. Preserve scene_id values exactly.\n\n"
                f"Video duration: {skeleton.get('duration_s')}s\n\n"
                "Timestamped frame captions grouped by scene:\n"
                + "\n".join(lines)
                + "\n\n"
                "Return ONLY valid JSON with this schema:\n"
                "{\n"
                "  \"skeleton_summary\": \"one concise visual summary for the video\",\n"
                "  \"scenes\": [\n"
                "    {\n"
                "      \"scene_id\": \"S000\",\n"
                "      \"scene_caption\": \"what appears to happen in this scene\",\n"
                "      \"keyframe_observations\": [\n"
                "        {\"timestamp_s\": 1.0, \"description\": \"what is visible\"}\n"
                "      ],\n"
                "      \"objects\": [\"object\"],\n"
                "      \"actions\": [\"action\"],\n"
                "      \"possible_events\": [\"event\"]\n"
                "    }\n"
                "  ]\n"
                "}\n"
            ),
        }
    ]
    raw, backend = observe_content(
        config,
        content=content,
        tool_name="skeleton_aggregate",
        tool_mode="single_frame_aggregate",
        output_dir=output_dir,
        return_json=True,
    )
    parsed = extract_json_object(raw) or {}
    if not parsed:
        captioned, summary = _deterministic_scene_aggregate(skeleton, frame_observations)
        skeleton["skeleton_summary"] = summary
        skeleton["caption_warnings"] = [
            "single-frame aggregation output could not be parsed; used deterministic fallback"
        ]
        return skeleton, f"{backend}_unparsed_fallback", time.time() - t0

    if parsed.get("skeleton_summary"):
        skeleton["skeleton_summary"] = str(parsed.get("skeleton_summary"))
    by_id = {
        scene.get("scene_id"): scene
        for scene in parsed.get("scenes") or []
        if isinstance(scene, dict)
    }
    changed_count = 0
    for scene in scenes:
        update = by_id.get(scene.get("scene_id")) or {}
        changed = False
        for key in ("scene_caption", "keyframe_observations", "objects", "actions", "possible_events"):
            if update.get(key) is not None:
                scene[key] = update.get(key)
                changed = True
        if changed and (scene.get("scene_caption") or scene.get("keyframe_observations")):
            changed_count += 1
    if changed_count == 0:
        captioned, summary = _deterministic_scene_aggregate(skeleton, frame_observations)
        skeleton["skeleton_summary"] = skeleton.get("skeleton_summary") or summary
        skeleton["caption_warnings"] = [
            "single-frame aggregation returned no usable scenes; used deterministic fallback"
        ]
    return skeleton, backend, time.time() - t0


def _ensure_skeleton_single_frame_captions(
    config: dict,
    *,
    skeleton: dict[str, Any],
    video_path: str,
    vr: Any,
    output_dir: str | None,
) -> tuple[dict[str, Any], bool]:
    t_total = time.time()
    max_frames = _as_int(config.get("skeleton_caption_frames"), 48)
    frames_per_scene = _as_int(config.get("skeleton_caption_frames_per_scene"), 4)
    short_side = _as_int(config.get("skeleton_single_frame_short_side"), 384)
    plan = _caption_frame_plan(
        skeleton,
        max_frames=max_frames,
        frames_per_scene=frames_per_scene,
    )
    if not plan:
        return skeleton, False

    fps = vr.get_avg_fps()
    total_frames = len(vr)
    frame_indices = np.array(
        [
            min(max(int(round(ts * fps)), 0), max(0, total_frames - 1))
            for _scene_id, ts in plan
        ],
        dtype=int,
    )
    frames = vr.get_batch(frame_indices).asnumpy()
    frame_observations: list[dict[str, Any]] = []
    errors: list[str] = []
    backend = "api"
    t_frames = time.time()
    for (scene_id, ts), frame in zip(plan, frames):
        content = [
            {
                "type": "text",
                "text": (
                    "You are captioning one selected video frame for an offline video skeleton memory.\n"
                    "Do not answer any final question. Describe only the visible frame.\n"
                    f"scene_id: {scene_id}\n"
                    f"timestamp_s: {float(ts):.1f}\n\n"
                    "Return ONLY valid JSON with this schema:\n"
                    "{\n"
                    "  \"scene_id\": \"S000\",\n"
                    "  \"timestamp_s\": 1.0,\n"
                    "  \"caption\": \"one concise factual sentence about visible content\",\n"
                    "  \"objects\": [\"object\"],\n"
                    "  \"visible_actions\": [\"action\"],\n"
                    "  \"text_on_screen\": \"visible text if any, else empty\",\n"
                    "  \"uncertainty\": \"low|medium|high\"\n"
                    "}\n"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": _frame_to_data_url(frame, short_side=short_side)},
            },
        ]
        raw, backend = observe_content(
            config,
            content=content,
            tool_name="skeleton",
            tool_mode="single_frame_caption",
            output_dir=output_dir,
            return_json=False,
        )
        parsed = extract_json_object(raw) or _frame_caption_fallback(
            scene_id=scene_id,
            timestamp_s=ts,
            raw=raw,
        )
        if not parsed:
            errors.append(f"{scene_id}@{float(ts):.1f}s: empty or unparseable frame caption")
            continue
        parsed["scene_id"] = str(parsed.get("scene_id") or scene_id)
        parsed["timestamp_s"] = round(float(parsed.get("timestamp_s") or ts), 1)
        if not str(parsed.get("caption") or "").strip():
            parsed["caption"] = str(parsed.get("description") or "").strip()
        frame_observations.append(parsed)
    frame_wall = time.time() - t_frames

    if not frame_observations:
        skeleton["caption_error"] = "; ".join(errors) or "single-frame skeleton caption failed"
        skeleton["caption_backend"] = backend
        skeleton["caption_strategy"] = "single_frame"
        skeleton["caption_timing"] = {
            "strategy": "single_frame",
            "frame_caption_wall_s": round(frame_wall, 3),
            "aggregate_wall_s": 0.0,
            "total_wall_s": round(time.time() - t_total, 3),
            "frame_caption_calls": len(plan),
            "frame_caption_success": 0,
        }
        save_skeleton(config, video_path=video_path, skeleton=skeleton)
        return skeleton, False

    skeleton["frame_caption_observations"] = frame_observations
    skeleton, aggregate_backend, aggregate_wall = _aggregate_single_frame_captions(
        config,
        skeleton=skeleton,
        frame_observations=frame_observations,
        output_dir=output_dir,
    )
    captioned_scene_count = sum(
        1
        for scene in skeleton.get("scenes") or []
        if (scene.get("scene_caption") or scene.get("keyframe_observations"))
    )
    skeleton.pop("caption_error", None)
    if errors:
        skeleton["caption_warnings"] = (skeleton.get("caption_warnings") or []) + errors
    elif not skeleton.get("caption_warnings"):
        skeleton.pop("caption_warnings", None)
    skeleton["caption_backend"] = backend
    skeleton["caption_aggregate_backend"] = aggregate_backend
    skeleton["caption_strategy"] = "single_frame"
    skeleton["caption_frame_count"] = len(frame_observations)
    skeleton["caption_timing"] = {
        "strategy": "single_frame",
        "frame_caption_wall_s": round(frame_wall, 3),
        "aggregate_wall_s": round(aggregate_wall, 3),
        "total_wall_s": round(time.time() - t_total, 3),
        "frame_caption_calls": len(plan),
        "frame_caption_success": len(frame_observations),
        "captioned_scene_count": captioned_scene_count,
    }
    save_skeleton(config, video_path=video_path, skeleton=skeleton)
    return skeleton, captioned_scene_count > 0


def _as_text_list(value: Any, *, limit: int = 8) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    out: list[str] = []
    for item in raw_items:
        text = str(item).strip()
        if text and text.lower() not in {"none", "n/a", "unknown"} and text not in out:
            out.append(text[:80])
        if len(out) >= limit:
            break
    return out


def _mini_sheet_items_from_parsed(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if not isinstance(parsed, dict):
        return []
    for key in ("frames", "observations", "frame_tags", "cells"):
        value = parsed.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    keyed_items: list[dict[str, Any]] = []
    for key, value in parsed.items():
        if not isinstance(value, dict):
            continue
        if str(key).upper().startswith("F") or str(key).lower().startswith("cell"):
            item = dict(value)
            item.setdefault("cell", key)
            keyed_items.append(item)
    return keyed_items


def _normalize_mini_sheet_observations(
    *,
    parsed: dict[str, Any],
    group_plan: list[tuple[str, float]],
) -> list[dict[str, Any]]:
    plan_by_cell = {
        f"F{idx + 1:02d}": (scene_id, ts)
        for idx, (scene_id, ts) in enumerate(group_plan)
    }
    observations: list[dict[str, Any]] = []
    for idx, item in enumerate(_mini_sheet_items_from_parsed(parsed)):
        cell = str(
            item.get("cell")
            or item.get("frame_id")
            or item.get("frame")
            or item.get("id")
            or f"F{idx + 1:02d}"
        ).strip().upper()
        if cell.startswith("CELL"):
            digits = "".join(ch for ch in cell if ch.isdigit())
            if digits:
                cell = f"F{int(digits):02d}"
        scene_id, ts = plan_by_cell.get(cell, group_plan[min(idx, len(group_plan) - 1)])
        objects = _as_text_list(
            item.get("objects")
            or item.get("object_tags")
            or item.get("visible_objects")
        )
        actions = _as_text_list(
            item.get("actions")
            or item.get("action_tags")
            or item.get("visible_actions")
        )
        relations = _as_text_list(
            item.get("relations")
            or item.get("spatial_relations")
            or item.get("state_relations")
        )
        scene_tags = _as_text_list(
            item.get("scene_tags")
            or item.get("tags")
            or item.get("visual_tags")
        )
        uncertainty = _as_text_list(item.get("uncertainty") or item.get("uncertain"))
        caption = str(item.get("caption") or item.get("description") or "").strip()

        desc_parts = []
        if scene_tags:
            desc_parts.append("tags=" + ",".join(scene_tags))
        if objects:
            desc_parts.append("objects=" + ",".join(objects))
        if actions:
            desc_parts.append("actions=" + ",".join(actions))
        if relations:
            desc_parts.append("relations=" + ",".join(relations))
        if uncertainty:
            desc_parts.append("uncertain=" + ",".join(uncertainty))
        description = "; ".join(desc_parts) or caption
        if not description:
            continue
        observations.append(
            {
                "scene_id": str(item.get("scene_id") or scene_id),
                "timestamp_s": round(float(item.get("timestamp_s") or ts), 1),
                "caption": description,
                "objects": objects,
                "visible_actions": actions,
                "relations": relations,
                "scene_tags": scene_tags,
                "uncertainty": uncertainty,
            }
        )
    return observations


def _aggregate_mini_sheet_tags(
    skeleton: dict[str, Any],
    frame_observations: list[dict[str, Any]],
) -> tuple[int, str]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for item in frame_observations:
        scene_id = str(item.get("scene_id") or "")
        if scene_id:
            by_scene.setdefault(scene_id, []).append(item)

    captioned = 0
    summaries: list[str] = []
    for scene in skeleton.get("scenes") or []:
        scene_id = scene.get("scene_id")
        items = sorted(
            by_scene.get(scene_id) or [],
            key=lambda item: float(item.get("timestamp_s") or 0.0),
        )
        if not items:
            continue

        observations = []
        objects: list[str] = []
        actions: list[str] = []
        relations: list[str] = []
        scene_tags: list[str] = []
        uncertainties: list[str] = []
        for item in items:
            caption = str(item.get("caption") or item.get("description") or "").strip()
            if caption:
                observations.append(
                    {
                        "timestamp_s": round(float(item.get("timestamp_s") or 0.0), 1),
                        "description": caption,
                    }
                )
            for target, source_key in (
                (objects, "objects"),
                (actions, "visible_actions"),
                (relations, "relations"),
                (scene_tags, "scene_tags"),
                (uncertainties, "uncertainty"),
            ):
                for value in item.get(source_key) or []:
                    text = str(value).strip()
                    if text and text not in target:
                        target.append(text)

        top_tags = scene_tags[:8] + actions[:8] + objects[:8]
        caption_bits = []
        if scene_tags:
            caption_bits.append("tags: " + ", ".join(scene_tags[:10]))
        if actions:
            caption_bits.append("actions: " + ", ".join(actions[:8]))
        if objects:
            caption_bits.append("objects: " + ", ".join(objects[:10]))
        if relations:
            caption_bits.append("relations: " + ", ".join(relations[:8]))
        if uncertainties:
            caption_bits.append("uncertain: " + ", ".join(uncertainties[:6]))
        scene_caption = "; ".join(caption_bits)
        if not scene_caption and observations:
            scene_caption = "; ".join(obs["description"] for obs in observations[:4])
        if scene_caption:
            scene["scene_caption"] = scene_caption
            scene["keyframe_observations"] = observations
            scene["objects"] = objects[:12]
            scene["actions"] = actions[:12]
            scene["possible_events"] = top_tags[:12]
            scene["tag_relations"] = relations[:12]
            captioned += 1
            summaries.append(f"{scene_id}: {scene_caption}")
    return captioned, " ".join(summaries[:16])


def _ensure_skeleton_mini_sheet_tags(
    config: dict,
    *,
    skeleton: dict[str, Any],
    video_path: str,
    vr: Any,
    output_dir: str | None,
) -> tuple[dict[str, Any], bool]:
    t_total = time.time()
    max_frames = _as_int(config.get("skeleton_caption_frames"), 48)
    frames_per_scene = _as_int(config.get("skeleton_caption_frames_per_scene"), 4)
    group_size = min(4, max(1, _as_int(config.get("skeleton_mini_sheet_group_size"), 4)))
    short_side = _as_int(config.get("skeleton_mini_sheet_short_side"), 384)
    strategy_name = str(config.get("skeleton_caption_strategy") or "mini_sheet_tags").strip().lower()
    if strategy_name in {"iframe_sheet_tags", "iframe-sheet-tags", "iframe_tags", "i_frame_tags"}:
        strategy_name = "iframe_sheet_tags"
        plan = _caption_iframe_sheet_plan(
            skeleton,
            video_path=video_path,
            max_frames=max_frames,
            frames_per_scene=frames_per_scene,
        )
        sample_note = (
            "The contact sheet cells are selected from I-frames when available, "
            "with motion peaks or anchor frames used only as fillers. "
        )
    else:
        strategy_name = "mini_sheet_tags"
        plan = _caption_frame_plan(
            skeleton,
            max_frames=max_frames,
            frames_per_scene=frames_per_scene,
        )
        sample_note = "The contact sheet cells are selected from scene anchors and motion peaks. "
    if not plan:
        return skeleton, False

    plan_by_scene: dict[str, list[float]] = {}
    for scene_id, ts in plan:
        plan_by_scene.setdefault(scene_id, []).append(ts)

    fps = vr.get_avg_fps()
    total_frames = len(vr)
    attempts = max(1, _as_int(config.get("skeleton_caption_retries"), 1))
    frame_observations: list[dict[str, Any]] = []
    errors: list[str] = []
    backend = "api"
    call_count = 0
    frame_count = 0
    t_tags = time.time()

    for scene in skeleton.get("scenes") or []:
        scene_id = str(scene.get("scene_id") or "")
        timestamps = plan_by_scene.get(scene_id) or []
        if not scene_id or not timestamps:
            continue
        for group_start in range(0, len(timestamps), group_size):
            group_timestamps = timestamps[group_start : group_start + group_size]
            group_plan = [(scene_id, ts) for ts in group_timestamps]
            frame_indices = timestamps_to_frame_indices(
                timestamps=group_timestamps,
                fps=fps,
                total_frames=total_frames,
            )
            if len(frame_indices) != len(group_timestamps):
                frame_indices = np.array(
                    [
                        min(max(int(round(ts * fps)), 0), max(0, total_frames - 1))
                        for ts in group_timestamps
                    ],
                    dtype=int,
                )
            frames = vr.get_batch(frame_indices).asnumpy()
            resized_frames = [_resize_frame(frame, short_side=short_side) for frame in frames]
            if not resized_frames:
                continue

            h, w, c = resized_frames[0].shape
            grid_frames = list(resized_frames)
            while len(grid_frames) < 4:
                grid_frames.append(np.zeros((h, w, c), dtype=np.uint8))
            grid = (
                np.stack(grid_frames[:4], axis=0)
                .reshape(2, 2, h, w, c)
                .transpose(0, 2, 1, 3, 4)
                .reshape(2 * h, 2 * w, c)
            )
            cell_lines = []
            for idx, (_sid, ts) in enumerate(group_plan):
                cell_lines.append(f"F{idx + 1:02d}: scene_id={scene_id}, timestamp_s={ts:.1f}")

            content = [
                {
                    "type": "text",
                    "text": (
                        "You are creating a low-cost visual index for a video QA agent.\n"
                        "The image is a 2x2 contact sheet from ONE scene. Cells are ordered:\n"
                        "F01=top-left, F02=top-right, F03=bottom-left, F04=bottom-right.\n"
                        f"{sample_note}"
                        "Only tag visible evidence. Do not infer the final answer.\n"
                        "Use short tags, not long captions. Preserve scene_id and timestamps.\n\n"
                        f"Scene range: {scene.get('t_range')}\n"
                        "Valid cells:\n"
                        + "\n".join(cell_lines)
                        + "\n\n"
                        "For each valid cell, output objects/actions/relations/scene_tags. "
                        "Include uncertainty tags for small or unclear content.\n"
                        "Return ONLY valid JSON with this schema:\n"
                        "{\n"
                        "  \"frames\": [\n"
                        "    {\n"
                        "      \"cell\": \"F01\",\n"
                        "      \"scene_id\": \"S000\",\n"
                        "      \"timestamp_s\": 1.0,\n"
                        "      \"scene_tags\": [\"indoor\"],\n"
                        "      \"objects\": [\"person\"],\n"
                        "      \"actions\": [\"standing\"],\n"
                        "      \"relations\": [\"person_near_table\"],\n"
                        "      \"uncertainty\": [\"small_object_unclear\"]\n"
                        "    }\n"
                        "  ]\n"
                        "}\n"
                    ),
                },
                {"type": "image_url", "image_url": {"url": _array_to_data_url(grid)}},
            ]
            raw = None
            parsed: dict[str, Any] = {}
            for _attempt in range(attempts):
                call_count += 1
                raw, backend = observe_content(
                    config,
                    content=content,
                    tool_name="skeleton",
                    tool_mode=strategy_name,
                    output_dir=output_dir,
                    return_json=False,
                )
                parsed = extract_json_object(raw) or {}
                if parsed:
                    break
            observations = _normalize_mini_sheet_observations(
                parsed=parsed,
                group_plan=group_plan,
            )
            if not observations:
                errors.append(
                    f"{scene_id}@{group_timestamps[0]:.1f}-{group_timestamps[-1]:.1f}s: "
                    + ("output could not be parsed" if raw else "model response is empty")
                )
                continue
            frame_observations.extend(observations)
            frame_count += len(group_plan)

    tag_wall = time.time() - t_tags
    if not frame_observations:
        skeleton["caption_error"] = "; ".join(errors) or "mini-sheet tag skeleton caption failed"
        skeleton["caption_backend"] = backend
        skeleton["caption_strategy"] = strategy_name
        skeleton["caption_timing"] = {
            "strategy": strategy_name,
            "tag_wall_s": round(tag_wall, 3),
            "total_wall_s": round(time.time() - t_total, 3),
            "tag_calls": call_count,
            "tagged_frames": 0,
        }
        save_skeleton(config, video_path=video_path, skeleton=skeleton)
        return skeleton, False

    skeleton["frame_caption_observations"] = frame_observations
    captioned_scene_count, summary = _aggregate_mini_sheet_tags(skeleton, frame_observations)
    skeleton["skeleton_summary"] = summary
    skeleton.pop("caption_error", None)
    if errors:
        skeleton["caption_warnings"] = errors
    else:
        skeleton.pop("caption_warnings", None)
    skeleton["caption_backend"] = backend
    skeleton["caption_aggregate_backend"] = "deterministic"
    skeleton["caption_strategy"] = strategy_name
    skeleton["caption_frame_count"] = frame_count
    skeleton["caption_timing"] = {
        "strategy": strategy_name,
        "tag_wall_s": round(tag_wall, 3),
        "aggregate_wall_s": 0.0,
        "total_wall_s": round(time.time() - t_total, 3),
        "tag_calls": call_count,
        "tagged_frames": len(frame_observations),
        "captioned_scene_count": captioned_scene_count,
    }
    save_skeleton(config, video_path=video_path, skeleton=skeleton)
    return skeleton, captioned_scene_count > 0


def ensure_skeleton_captions(
    config: dict,
    *,
    skeleton: dict[str, Any],
    video_path: str,
    vr: Any,
    output_dir: str | None,
) -> tuple[dict[str, Any], bool]:
    if not bool(config.get("skeleton_caption_enabled", True)):
        return skeleton, False
    if any((scene.get("scene_caption") or "").strip() for scene in skeleton.get("scenes") or []):
        return skeleton, False
    if skeleton.get("caption_error") and not bool(config.get("skeleton_caption_retry_failed_cache", False)):
        return skeleton, False

    strategy = str(config.get("skeleton_caption_strategy") or "contact_sheet").strip().lower()
    if strategy in {"single_frame", "single-frame", "frame"}:
        return _ensure_skeleton_single_frame_captions(
            config,
            skeleton=skeleton,
            video_path=video_path,
            vr=vr,
            output_dir=output_dir,
        )
    if strategy in {
        "mini_sheet_tags",
        "mini-sheet-tags",
        "mini_tags",
        "short_tags",
        "iframe_sheet_tags",
        "iframe-sheet-tags",
        "iframe_tags",
        "i_frame_tags",
    }:
        return _ensure_skeleton_mini_sheet_tags(
            config,
            skeleton=skeleton,
            video_path=video_path,
            vr=vr,
            output_dir=output_dir,
        )

    max_frames = _as_int(config.get("skeleton_caption_frames"), 48)
    frames_per_scene = _as_int(config.get("skeleton_caption_frames_per_scene"), 4)
    scenes_per_chunk = max(1, _as_int(config.get("skeleton_caption_scenes_per_chunk"), 7))
    short_side = _as_int(config.get("skeleton_caption_short_side"), 224)
    plan = _caption_frame_plan(
        skeleton,
        max_frames=max_frames,
        frames_per_scene=frames_per_scene,
    )
    if not plan:
        return skeleton, False

    fps = vr.get_avg_fps()
    total_frames = len(vr)
    attempts = max(1, _as_int(config.get("skeleton_caption_retries"), 2))

    scenes = list(skeleton.get("scenes") or [])
    scene_ranges = {scene.get("scene_id"): scene.get("t_range") for scene in scenes}
    plan_by_scene: dict[str, list[float]] = {}
    for scene_id, ts in plan:
        plan_by_scene.setdefault(scene_id, []).append(ts)

    captioned_scene_count = 0
    caption_frame_count = 0
    summaries: list[str] = []
    errors: list[str] = []
    backend = "api"

    for chunk_start in range(0, len(scenes), scenes_per_chunk):
        chunk_scenes = scenes[chunk_start : chunk_start + scenes_per_chunk]
        chunk_ids = [scene.get("scene_id") or "" for scene in chunk_scenes]
        chunk_plan = [
            (scene_id, ts)
            for scene_id in chunk_ids
            for ts in plan_by_scene.get(scene_id, [])
        ]
        if not chunk_plan:
            continue

        frame_indices = timestamps_to_frame_indices(
            timestamps=[ts for _scene_id, ts in chunk_plan],
            fps=fps,
            total_frames=total_frames,
        )
        if len(frame_indices) != len(chunk_plan):
            frame_indices = np.array(
                [
                    min(max(int(round(ts * fps)), 0), max(0, total_frames - 1))
                    for _scene_id, ts in chunk_plan
                ],
                dtype=int,
            )
        frames = vr.get_batch(frame_indices).asnumpy()
        resized_frames = [_resize_frame(frame, short_side=short_side) for frame in frames]
        if not resized_frames:
            continue

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are building one chunk of an offline video skeleton memory for a later QA agent.\n"
                    "Caption each scene using only the provided keyframes. Do not answer any user question.\n"
                    "Return concise, factual visual observations. Use the provided scene_id values exactly.\n\n"
                    f"Video duration: {skeleton.get('duration_s')}s\n"
                    "Scene ranges in this chunk:\n"
                    + "\n".join(
                        f"- {scene_id}: {scene_ranges.get(scene_id)}"
                        for scene_id in chunk_ids
                    )
                    + "\n\n"
                ),
            }
        ]
        h, w, c = resized_frames[0].shape
        group_size = 8
        for group_start in range(0, len(resized_frames), group_size):
            group_frames = resized_frames[group_start : group_start + group_size]
            group_plan = chunk_plan[group_start : group_start + group_size]
            while len(group_frames) < group_size:
                group_frames.append(np.zeros((h, w, c), dtype=np.uint8))
                group_plan.append(("PAD", 0.0))
            grid = (
                np.stack(group_frames, axis=0)
                .reshape(2, 4, h, w, c)
                .transpose(0, 2, 1, 3, 4)
                .reshape(2 * h, 4 * w, c)
            )
            cell_lines = []
            for idx, (scene_id, ts) in enumerate(group_plan):
                if scene_id == "PAD":
                    continue
                row, col = divmod(idx, 4)
                cell_lines.append(
                    f"cell[{row},{col}] scene_id={scene_id}, timestamp={ts:.1f}s"
                )
            content.append({"type": "text", "text": "Contact sheet cells:\n" + "\n".join(cell_lines)})
            content.append({"type": "image_url", "image_url": {"url": _array_to_data_url(grid)}})
        content.append(
            {
                "type": "text",
                "text": (
                    "Return ONLY valid JSON with this schema:\n"
                    "{\n"
                    "  \"skeleton_summary\": \"one concise visual summary for this chunk\",\n"
                    "  \"scenes\": [\n"
                    "    {\n"
                    "      \"scene_id\": \"S000\",\n"
                    "      \"scene_caption\": \"what happens in this scene\",\n"
                    "      \"keyframe_observations\": [\n"
                    "        {\"timestamp_s\": 1.0, \"description\": \"what is visible\"}\n"
                    "      ],\n"
                    "      \"objects\": [\"object\"],\n"
                    "      \"actions\": [\"action\"],\n"
                    "      \"possible_events\": [\"event\"]\n"
                    "    }\n"
                    "  ]\n"
                    "}\n"
                ),
            }
        )

        raw = None
        parsed: dict[str, Any] = {}
        for _attempt in range(attempts):
            raw, backend = observe_content(
                config,
                content=content,
                tool_name="skeleton",
                tool_mode="caption",
                output_dir=output_dir,
                return_json=False,
            )
            parsed = extract_json_object(raw) or {}
            if parsed:
                break
        if not parsed:
            errors.append(
                f"{chunk_ids[0]}-{chunk_ids[-1]}: "
                + (
                    "output could not be parsed"
                    if raw
                    else "model response is empty"
                )
            )
            continue

        if parsed.get("skeleton_summary"):
            summaries.append(str(parsed.get("skeleton_summary")))
        by_id = {
            scene.get("scene_id"): scene
            for scene in parsed.get("scenes") or []
            if isinstance(scene, dict)
        }
        for scene in chunk_scenes:
            update = by_id.get(scene.get("scene_id")) or {}
            changed = False
            for key in ("scene_caption", "keyframe_observations", "objects", "actions", "possible_events"):
                if update.get(key) is not None:
                    scene[key] = update.get(key)
                    changed = True
            if changed and (scene.get("scene_caption") or scene.get("keyframe_observations")):
                captioned_scene_count += 1
        caption_frame_count += len(chunk_plan)

    if captioned_scene_count == 0:
        skeleton["caption_error"] = "; ".join(errors) or "skeleton caption failed"
        skeleton["caption_backend"] = backend
        save_skeleton(config, video_path=video_path, skeleton=skeleton)
        return skeleton, False

    skeleton["skeleton_summary"] = " ".join(summaries)
    skeleton.pop("caption_error", None)
    skeleton.pop("caption_raw_preview", None)
    skeleton["caption_backend"] = backend
    skeleton["caption_frame_count"] = caption_frame_count
    if errors:
        skeleton["caption_warnings"] = errors
    else:
        skeleton.pop("caption_warnings", None)
    save_skeleton(config, video_path=video_path, skeleton=skeleton)
    return skeleton, True


def skeleton_to_prompt_text(skeleton: dict[str, Any], *, max_scenes: int | None = None) -> str:
    scenes = list(skeleton.get("scenes") or [])
    if max_scenes:
        scenes = scenes[:max_scenes]
    lines = [
        f"Skeleton version: {skeleton.get('version')}",
        f"Video duration: {skeleton.get('duration_s')}s",
        f"Skeleton summary: {skeleton.get('skeleton_summary') or ''}",
        "Scenes:",
    ]
    for scene in scenes:
        observations = scene.get("keyframe_observations") or []
        obs_text = "; ".join(
            f"{item.get('timestamp_s')}s: {item.get('description')}"
            for item in observations[:4]
            if isinstance(item, dict)
        )
        lines.append(
            "\n".join(
                [
                    f"- {scene.get('scene_id')} {scene.get('t_range')}",
                    f"  anchors={scene.get('anchor_timestamps')} motion_peaks={scene.get('motion_peaks')}",
                    f"  caption={scene.get('scene_caption')}",
                    f"  objects={scene.get('objects')} actions={scene.get('actions')} possible_events={scene.get('possible_events')}",
                    f"  keyframes={obs_text}",
                ]
            )
        )
    return "\n".join(lines)
