import base64
from io import BytesIO

import numpy as np
from PIL import Image

from videoseek.codec import timestamps_to_frame_indices
from videoseek.core.p130_runtime import normalize_p131_order_payload
from videoseek.observer import observe_content
from videoseek.skeleton import (
    get_scene_context_for_window,
    scene_aware_timestamps,
    scene_context_to_prompt_text,
)
from videoseek.tools.v10_format import format_v10_observation, snap_timestamp_observations
from videoseek.utils import (
    convert_to_free_form_text_representation,
    extract_json_object,
)
from config import general_config


skim_tool = {
    "type": "function",
    "function": {
        "name": "skim",
        "description": "To localize moments related to the query, quickly scan of a long segment (> {skim_num_frames}s) by sampling {skim_num_frames} frames from the video segment (start_time - end_time).".format(skim_num_frames=general_config["frame_sampling_factor"] * general_config["skim_base"]),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to skim the video. The query should be a concise question that can be answered by the video.",
                },
                "start_time": {
                    "type": "number",
                    "description": "The start time of the video to skim.",
                },
                "end_time": {
                    "type": "number",
                    "description": "The end time of the video to skim.",
                },
                "mode": {
                    "type": "string",
                    "description": "Internal seek mode: normal, option_coverage, timeline, count_occurrence, or alternative_window.",
                },
            },
            "required": ["query", "start_time", "end_time", "mode"],
            "additionalProperties": False,
        },
    },
}


def _p131_order_active(config: dict, parameters: dict) -> bool:
    return bool(
        config.get("p131_minimal_global_repairs_enabled", False)
        and parameters.get("p131_order_comparative") is True
    )


def _p131_event_catalog(parameters: dict) -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    raw_catalog = parameters.get("event_catalog") or []
    values = (
        [
            {"event_id": event_id, "description": description}
            for event_id, description in raw_catalog.items()
        ]
        if isinstance(raw_catalog, dict)
        else raw_catalog
    )
    for index, value in enumerate(values, start=1):
        if isinstance(value, dict):
            event_id = str(value.get("event_id") or index).strip()
            description = str(value.get("description") or "").strip()
        else:
            event_id = str(index)
            description = str(value or "").strip()
        if not event_id.isdigit() or not description:
            continue
        catalog.append(
            {"event_id": str(int(event_id)), "description": description}
        )
    return catalog


def _p131_limit_timestamps(values: list[float], limit: int) -> list[float]:
    unique = sorted({round(float(value), 3) for value in values})
    if len(unique) <= limit:
        return unique
    positions = np.linspace(0, len(unique) - 1, limit).round().astype(int)
    return [unique[int(position)] for position in sorted(set(positions.tolist()))]


def execute_skim(config: dict, parameters: dict) -> str:
    """
    Execute the skim tool.
    """
    query = parameters["query"]
    p131_order = _p131_order_active(config, parameters)
    p131_catalog = _p131_event_catalog(parameters) if p131_order else []
    raw_start_time = float(parameters["start_time"])
    raw_end_time = float(parameters["end_time"])
    mode = parameters.get("mode", "normal")
    vr = parameters["vr"]
    video_path = parameters.get("video_path", "")
    duration = float(parameters.get("duration") or round(len(vr) / vr.get_avg_fps(), 1))
    start_time = max(0.0, min(raw_start_time, duration))
    end_time = max(start_time, min(raw_end_time, duration))
    base_num_frames = general_config["frame_sampling_factor"] * general_config["skim_base"]
    num_frames = (
        int(config.get("p131_order_skim_num_frames") or 16)
        if p131_order
        else base_num_frames
    )
    window_s = max(0.0, end_time - start_time)
    long_window_s = float(config.get("skim_long_window_s") or 120.0)
    if window_s >= long_window_s and not p131_order:
        long_budget = int(config.get("skim_long_num_frames") or 48)
        target_step_s = float(config.get("skim_long_target_step_s") or 6.0)
        proportional_budget = int(np.ceil(window_s / max(1.0, target_step_s)))
        num_frames = max(base_num_frames, min(long_budget, proportional_budget))
    subtitles = parameters["subtitles"]
    subtitles_str = convert_to_free_form_text_representation(
        [
            subtitle
            for subtitle in subtitles
            if float(subtitle["start_time"]) <= end_time
            and float(subtitle["end_time"]) >= start_time
        ],
        content_type="subtitle",
    )
    scene_context, _scene_cache_path = get_scene_context_for_window(
        config,
        video_path=video_path,
        duration_s=duration,
        start_time=start_time,
        end_time=end_time,
    )
    scene_id = str((scene_context or {}).get("scene_id") or "skim")
    window_id = f"{scene_id}_skim_window" if scene_id != "skim" else "skim_window"
    scene_context_text = scene_context_to_prompt_text(scene_context)

    # Mixed sampling: uniform + I-frame + codec bit-cost peaks. This keeps the
    # agent-visible tool simple while improving evidence discovery inside the
    # selected window.
    total_frames = len(vr)
    start_frame = min(int(start_time * vr.get_avg_fps()), total_frames - 1)
    end_frame = min(max(start_frame + 1, int(end_time * vr.get_avg_fps())), total_frames - 1)
    sampled_timestamps = scene_aware_timestamps(
        video_path=video_path,
        duration_s=duration,
        start_time=start_time,
        end_time=end_time,
        num_frames=num_frames,
        scene=scene_context,
    )
    if p131_order:
        requested_timestamps: list[float] = []
        for value in parameters.get("timestamps") or []:
            try:
                timestamp = float(value)
            except (TypeError, ValueError):
                continue
            if start_time <= timestamp <= end_time:
                requested_timestamps.append(timestamp)
        requested_timestamps = _p131_limit_timestamps(
            requested_timestamps,
            num_frames,
        )
        if len(requested_timestamps) < num_frames:
            for timestamp in sampled_timestamps:
                normalized = round(float(timestamp), 3)
                if normalized not in requested_timestamps:
                    requested_timestamps.append(normalized)
                if len(requested_timestamps) >= num_frames:
                    break
        sampled_timestamps = sorted(requested_timestamps)
    frame_indices = timestamps_to_frame_indices(
        timestamps=sampled_timestamps,
        fps=vr.get_avg_fps(),
        total_frames=total_frames,
    )
    if len(frame_indices) < min(num_frames, max(1, end_frame - start_frame)):
        fallback_indices = np.linspace(
            start_frame,
            max(start_frame, end_frame - 1),
            num_frames,
        ).astype(int)
        if p131_order:
            preserved = [int(value) for value in frame_indices.tolist()]
            for value in fallback_indices.tolist():
                if int(value) not in preserved:
                    preserved.append(int(value))
                if len(preserved) >= num_frames:
                    break
            frame_indices = np.asarray(sorted(preserved), dtype=int)
        else:
            frame_indices = fallback_indices
    else:
        frame_indices = frame_indices[:num_frames]
    cur_timestamps = np.array(
        [round(frame_indice / vr.get_avg_fps(), 1) for frame_indice in frame_indices],
        dtype=np.float32,
    )
    cur_timestamps_list = [round(float(item), 1) for item in cur_timestamps.tolist()]
    cur_timestamps_str = ", ".join(f"{item:.1f}s" for item in cur_timestamps_list)
    frames = vr.get_batch(frame_indices).asnumpy()
    actual_num_frames = len(frames)

    # resize the shorter side to 256
    _, height, width, _ = frames.shape
    short_side = min(height, width)
    scale = 256 / short_side
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    resized_frames = [
        np.array(
            Image.fromarray(frame).resize(
                (target_width, target_height),
                Image.BICUBIC,
            )
        )
        for frame in frames
    ]
    frames = np.stack(resized_frames, axis=0).reshape(
        actual_num_frames, target_height, target_width, 3
    )

    content = [{"type": "text", "text": f"Video segment ({start_time:.1f}s - {end_time:.1f}s):\n"}]
    for frame, timestamp in zip(frames, cur_timestamps):
        img = Image.fromarray(frame)
        output_buffer = BytesIO()
        img.save(output_buffer, format="jpeg")
        byte_data = output_buffer.getvalue()
        base64_image = base64.b64encode(byte_data).decode("utf-8")
        content.append({
            "type": "text",
            "text": f"{timestamp:.1f}s",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        })
    content.append({
        "type": "text",
        "text": (
            f"Video Subtitles:\n{subtitles_str}\n\n"
            + (
                "Skeleton scene context for this skim window:\n"
                f"{scene_context_text}\n\n"
                if scene_context_text
                else ""
            )
            +
            f"Question/query:\n{query}\n\n"
            f"Seek mode: {mode}\n\n"
            "You are the skim observer in a video QA agent. Do not answer the final question.\n"
            "Describe what happens in this segment, using the provided timestamps exactly.\n"
            "If skeleton scene context is provided, preserve that scene_id in your JSON output.\n"
            f"Allowed timestamps: [{cur_timestamps_str}]. Do not invent other timestamp values.\n"
            "Use at most 6 timestamp_observations; choose only the most evidence-bearing frames.\n"
            "Highlight evidence-bearing moments and suggest one short focus window if a detail should be verified with real frames.\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            f"  \"window_id\": \"{window_id}\",\n"
            f"  \"scene_id\": \"{scene_id}\",\n"
            f"  \"t_range\": [{float(start_time):.1f}, {float(end_time):.1f}],\n"
            "  \"timestamp_observations\": [\n"
            f"    {{\"timestamp_s\": {cur_timestamps_list[0]:.1f}, \"scene_id\": \"{scene_id}\", \"description\": \"what is visible\", \"event_tags\": [\"action\"], \"needs_focus\": \"\"}}\n"
            "  ],\n"
            "  \"observed_event\": \"short event summary\",\n"
            "  \"contains_evidence\": true,\n"
            "  \"relevance\": 0.0,\n"
            "  \"suggest_focus_window\": [12.0, 16.0],\n"
            "  \"scene_summaries\": [\n"
            f"    {{\"scene_id\": \"{scene_id}\", \"window_id\": \"{window_id}\", \"t_range\": [12.0, 20.0], \"summary\": \"what happens\", \"possible_evidence\": true, \"suggest_focus_windows\": [[12.0, 16.0]], \"missing_detail\": \"what focus should verify\"}}\n"
            "  ],\n"
            "  \"missing_detail\": \"\"\n"
            "}\n"
        ),
    })

    if p131_order:
        catalog_text = "\n".join(
            f"- E{row['event_id']}: {row['description']}"
            for row in p131_catalog
        )
        content[-1]["text"] = (
            f"Video Subtitles:\n{subtitles_str}\n\n"
            + (
                "Skeleton scene context for this skim window:\n"
                f"{scene_context_text}\n\n"
                if scene_context_text
                else ""
            )
            + "P131 comparative Order scan. Do not answer the multiple-choice question.\n"
            + f"Event catalog:\n{catalog_text}\n\n"
            + f"Allowed timestamps: [{cur_timestamps_str}]. Do not invent timestamps.\n"
            + "Inspect this entire continuous chapter comparatively. For each sampled "
            + "frame that directly or ambiguously bears on a catalog event, return one "
            + "timestamp_observation with event_id, event_match, and observed_fact. "
            + "Use event_match=direct when the distinctive observable core uniquely "
            + "matches one catalog entry. Narrative purpose or intent clauses such as "
            + "'to see a competition' need not be visually provable; do not downgrade "
            + "an otherwise unique visible core for that reason. Nearby scenes may "
            + "disambiguate identity, but context alone is not direct proof. Never infer "
            + "chronology from catalog numbering. Use only event IDs in this catalog.\n"
            + "Return ONLY valid JSON with this schema:\n"
            + "{\n"
            + f'  "window_id": "{window_id}",\n'
            + f'  "scene_id": "{scene_id}",\n'
            + f'  "t_range": [{float(start_time):.1f}, {float(end_time):.1f}],\n'
            + '  "timestamp_observations": [{"timestamp_s": '
            + f"{cur_timestamps_list[0]:.1f}"
            + f', "scene_id": "{scene_id}", "event_id": "1", '
            + '"event_match": "direct|ambiguous|context_only|different_event|not_visible", '
            + '"observed_fact": "pixel-grounded visible fact", "description": "same visible fact"}],\n'
            + '  "observed_event": "local comparative scan summary",\n'
            + '  "contains_evidence": true,\n'
            + '  "relevance": 0.0,\n'
            + '  "missing_event_ids": ["2"],\n'
            + '  "ambiguous_event_ids": ["3"],\n'
            + '  "missing_detail": "which catalog cores still need local verification"\n'
            + "}\n"
        )

    observer_config = config
    if p131_order or len(frames) > int(config.get("local_qwen_max_images") or 24):
        observer_config = dict(config)
        observer_config["observer_backend"] = "api"
        if p131_order:
            observer_config["local_qwen_tools"] = ""

    raw, observer_backend = observe_content(
        observer_config,
        content=content,
        tool_name="skim",
        tool_mode=mode,
        output_dir=parameters.get("output_dir"),
    )

    if raw is None:
        return "Skim observation failed: model response is empty."
    parsed = extract_json_object(raw) or {}
    if parsed:
        if not parsed.get("scene_id") or parsed.get("scene_id") == "skim":
            parsed["scene_id"] = scene_id
        if not parsed.get("window_id") or parsed.get("window_id") == "skim_window":
            parsed["window_id"] = window_id
        parsed.setdefault("t_range", [float(start_time), float(end_time)])
        for item in parsed.get("timestamp_observations") or []:
            if isinstance(item, dict) and (not item.get("scene_id") or item.get("scene_id") == "skim"):
                item["scene_id"] = scene_id
        for item in parsed.get("scene_summaries") or []:
            if isinstance(item, dict):
                if not item.get("scene_id") or item.get("scene_id") == "skim":
                    item["scene_id"] = scene_id
                if not item.get("window_id") or item.get("window_id") == "skim_window":
                    item["window_id"] = window_id
        if scene_context:
            parsed.setdefault("matched_skeleton_scene", scene_id)
        if p131_order:
            # Validate raw observer fields before the generic V10 helper can snap
            # an arbitrary timestamp, or replace a missing one with the first
            # sampled frame.  A demoted row remains useful as a recovery lead.
            normalize_p131_order_payload(
                parsed,
                catalog=p131_catalog,
                allowed_timestamps=cur_timestamps_list,
            )
            parsed["parse_ok"] = True
        snap_timestamp_observations(parsed, allowed_timestamps=cur_timestamps_list)
    parsed.setdefault("observer_backend", observer_backend)
    return format_v10_observation(parsed, fallback_text=raw)
