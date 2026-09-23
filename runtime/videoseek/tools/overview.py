import json
import math
import base64
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from videoseek.codec import (
    codec_aware_timestamps,
    keyframe_only_timestamps,
    probe_keyframes,
    timestamps_to_frame_indices,
)
from videoseek.observer import observe_content
from videoseek.skeleton import (
    ensure_skeleton_captions,
    load_or_build_skeleton,
    skeleton_to_prompt_text,
)
from videoseek.tools.v10_format import format_v10_observation
from videoseek.utils import (
    convert_to_free_form_text_representation,
    extract_json_object,
)
from config import general_config


overview_tool = {
    "type": "function",
    "function": {
        "name": "overview",
        "description": "To get a structured video summary for the entire video.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    }
}


def _annotate_frame_timestamp(frame: np.ndarray, timestamp_s: float) -> np.ndarray:
    """Burn the global timestamp into a contact-sheet cell.

    Keeping the label inside the image avoids relying on the observer to map an
    external timestamp matrix back to eight visually small cells.
    """
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font_size = max(16, int(round(min(image.size) * 0.07)))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    label = f"T={float(timestamp_s):.1f}s"
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    pad = max(3, font_size // 5)
    box = (0, 0, right - left + 2 * pad, bottom - top + 2 * pad)
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((pad - left, pad - top), label, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def _normalize_overview_times(parsed: dict, *, duration: float) -> dict:
    """Normalize seconds at the observer boundary; never invent invalid locations."""
    errors = []

    def seconds(value):
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("expected seconds")
        if isinstance(value, str):
            value = value.strip().removesuffix("s").strip()
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= duration:
            raise ValueError("seconds outside video")
        return value

    def span(value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("expected two endpoints")
        start, end = map(seconds, value)
        if start > end:
            raise ValueError("reversed interval")
        return [start, end]

    for index, row in enumerate(parsed.get("timestamp_observations") or []):
        if not isinstance(row, dict):
            continue
        key = "timestamp_s" if "timestamp_s" in row else "timestamp"
        try:
            row[key] = seconds(row.get(key))
        except (TypeError, ValueError):
            errors.append({"field": f"timestamp_observations[{index}].{key}", "value": row.get(key)})
            # The existing omission handler can now safely reject this row.
            row[key] = None
    for index, row in enumerate(parsed.get("scene_summaries") or []):
        if not isinstance(row, dict):
            continue
        if "t_range" in row:
            try:
                row["t_range"] = span(row["t_range"])
            except (TypeError, ValueError):
                errors.append({"field": f"scene_summaries[{index}].t_range", "value": row.pop("t_range")})
                row["possible_evidence"] = False
        if "suggest_focus_windows" in row:
            windows = []
            source = row["suggest_focus_windows"]
            if not isinstance(source, (list, tuple)):
                errors.append({"field": f"scene_summaries[{index}].suggest_focus_windows", "value": source})
                source = []
            for window in source:
                try:
                    windows.append(span(window))
                except (TypeError, ValueError):
                    errors.append({"field": f"scene_summaries[{index}].suggest_focus_windows", "value": window})
            row["suggest_focus_windows"] = windows
    if errors:
        parsed["overview_time_errors"] = errors
    return parsed


def _normalize_timestamp_observations(
    parsed: dict,
    *,
    allowed_timestamps: list[float],
    fill_omitted: bool,
) -> dict:
    allowed = [round(float(value), 1) for value in allowed_timestamps]
    by_timestamp: dict[float, dict] = {}
    for item in parsed.get("timestamp_observations") or []:
        if not isinstance(item, dict):
            continue
        try:
            raw_timestamp = float(item.get("timestamp_s", item.get("timestamp")))
        except (TypeError, ValueError):
            continue
        if not allowed:
            continue
        timestamp = min(allowed, key=lambda candidate: abs(candidate - raw_timestamp))
        candidate = dict(item)
        candidate["timestamp_s"] = timestamp
        candidate.setdefault("scene_id", "overview")
        candidate.setdefault("event_tags", [])
        candidate.setdefault("needs_focus", "")
        description = str(candidate.get("description") or candidate.get("desc") or "").strip()
        candidate["description"] = description
        existing = by_timestamp.get(timestamp)
        if existing is None or len(description) > len(str(existing.get("description") or "")):
            by_timestamp[timestamp] = candidate

    omitted = [timestamp for timestamp in allowed if timestamp not in by_timestamp]
    if fill_omitted:
        for timestamp in omitted:
            by_timestamp[timestamp] = {
                "timestamp_s": timestamp,
                "scene_id": "overview",
                "description": "Overview observer omitted this sampled frame.",
                "event_tags": ["overview_omitted_frame"],
                "needs_focus": "Inspect this timestamp only if its surrounding scene becomes relevant.",
            }

    parsed["timestamp_observations"] = [
        by_timestamp[timestamp]
        for timestamp in allowed
        if timestamp in by_timestamp
    ]
    parsed["overview_expected_timestamp_count"] = len(allowed)
    parsed["overview_observed_timestamp_count"] = len(allowed) - len(omitted)
    parsed["overview_omitted_timestamps"] = omitted
    parsed["overview_timestamp_coverage"] = round(
        (len(allowed) - len(omitted)) / max(1, len(allowed)),
        4,
    )
    return parsed


def execute_overview(config: dict, parameters: dict) -> str:
    if bool(config.get("dual_path_overview_enabled")) and not bool(
        config.get("use_skeleton_overview")
    ):
        from videoseek.tools.dual_path_overview import execute_dual_path_overview

        return execute_dual_path_overview(
            config,
            parameters,
            remote_runner=_execute_standard_overview,
        )
    return _execute_standard_overview(config, parameters)


def _execute_standard_overview(config: dict, parameters: dict) -> str:
    """
    Execute the overview tool.
    """
    if bool(config.get("use_skeleton_overview")):
        return _execute_skeleton_overview(config, parameters)

    vr = parameters['vr']
    duration = round(len(vr) / vr.get_avg_fps(), 1)
    video_path = parameters.get("video_path", "")
    question = parameters.get("question", "")
    subtitles = parameters['subtitles']
    subtitles_str = convert_to_free_form_text_representation(subtitles, content_type='subtitle')

    num_frames = int(
        config.get("overview_num_frames")
        or (
            int(config.get("frame_sampling_factor") or general_config["frame_sampling_factor"])
            * int(config.get("overview_base") or general_config["overview_base"])
        )
    )
    if num_frames % 8 != 0:
        # round num_frames to the nearest multiple of 8
        num_frames = int(np.ceil(num_frames / 8.) * 8)
    contact_sheet_group_size = int(config.get("overview_contact_sheet_group_size") or 4)
    if contact_sheet_group_size not in {4, 8}:
        contact_sheet_group_size = 4
    sheet_rows, sheet_cols = (2, 2) if contact_sheet_group_size == 4 else (2, 4)

    requested_sampling_mode = str(
        config.get("overview_sampling_mode") or "hybrid"
    ).strip().lower()
    if requested_sampling_mode not in {"hybrid", "iframes"}:
        requested_sampling_mode = "hybrid"

    # The ablation keeps the frame budget fixed while changing only the source
    # timestamps. If a video has too few parseable I-frames to form one sheet,
    # retain the baseline hybrid fallback and expose that in the observation.
    total_frames = len(vr)
    available_keyframes = [
        timestamp
        for timestamp in probe_keyframes(video_path)
        if 0.0 <= timestamp <= duration
    ]
    effective_sampling_mode = requested_sampling_mode
    if requested_sampling_mode == "iframes":
        sampled_timestamps = keyframe_only_timestamps(
            video_path=video_path,
            start_time=0.0,
            end_time=duration,
            num_frames=num_frames,
        )
        if len(sampled_timestamps) < contact_sheet_group_size:
            effective_sampling_mode = "hybrid_fallback_no_iframes"
            sampled_timestamps = codec_aware_timestamps(
                video_path=video_path,
                duration_s=duration,
                start_time=0.0,
                end_time=duration,
                num_frames=num_frames,
            )
    else:
        sampled_timestamps = codec_aware_timestamps(
            video_path=video_path,
            duration_s=duration,
            start_time=0.0,
            end_time=duration,
            num_frames=num_frames,
        )
    frame_indices = timestamps_to_frame_indices(
        timestamps=sampled_timestamps,
        fps=vr.get_avg_fps(),
        total_frames=total_frames,
    )
    if requested_sampling_mode != "iframes" and len(frame_indices) < num_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    else:
        frame_indices = frame_indices[:num_frames]
    usable = (len(frame_indices) // contact_sheet_group_size) * contact_sheet_group_size
    frame_indices = frame_indices[:usable]
    num_frames = len(frame_indices)
    cur_timestamps = np.array([round(frame_indice / vr.get_avg_fps(), 1) for frame_indice in frame_indices], dtype=np.float32)
    allowed_timestamps = [round(float(value), 1) for value in cur_timestamps.tolist()]
    cur_timestamps_str = ', '.join([f"{t:.1f}s" for t in allowed_timestamps])
    frames = vr.get_batch(frame_indices).asnumpy()

    # Keep enough detail for each cell after composing a 2x4 contact sheet.
    _, height, width, _ = frames.shape
    short_side = min(height, width)

    overview_short_side = max(192, int(config.get("overview_frame_short_side") or 320))
    scale = overview_short_side / short_side
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))

    resized_frames = []
    embed_timestamps = bool(config.get("overview_embed_timestamp_labels", True))
    for frame, timestamp in zip(frames, cur_timestamps):
        resized = np.array(
            Image.fromarray(frame).resize(
                (target_width, target_height),
                Image.BICUBIC,
            )
        )
        if embed_timestamps:
            resized = _annotate_frame_timestamp(resized, float(timestamp))
        resized_frames.append(resized)
    frames = np.stack(resized_frames, axis=0)  # (T, H, W, C)

    # Smaller 2x2 sheets preserve scene structure and make cell/timestamp
    # binding easier. The legacy 2x4 layout remains available by config.
    group_size = contact_sheet_group_size
    num_frames, h, w, c = frames.shape
    cur_timestamps = cur_timestamps.reshape(-1, sheet_rows, sheet_cols)

    num_groups = num_frames // group_size
    frames = frames.reshape(num_groups, sheet_rows, sheet_cols, h, w, c)
    frames = frames.transpose(0, 1, 3, 2, 4, 5).reshape(
        num_groups,
        sheet_rows * h,
        sheet_cols * w,
        c,
    )

    # OpenAI chat-completions multimodal format: content is a list of typed parts.
    sampling_note = (
        "The video frames are I-frame-only codec samples distributed across the full video."
        if effective_sampling_mode == "iframes"
        else "The video frames are hybrid codec-aware samples."
    )
    content = [{"type": "text", "text": f"The video segment is located at 0.0s - {duration:.1f}s:\n{sampling_note}\n"}]
    for frame, timestamp in zip(frames, cur_timestamps):
        timestamp_rows = [
            "[" + ", ".join(f"{t:.1f}s" for t in row) + "]"
            for row in timestamp
        ]
        timestamp_str = "[" + ",\n".join(timestamp_rows) + "]"
        img = Image.fromarray(frame)
        output_buffer = BytesIO()
        img.save(output_buffer, format="jpeg")
        byte_data = output_buffer.getvalue()
        base64_image = base64.b64encode(byte_data).decode("utf-8")
        content.append({"type": "text", "text": ( f"[{timestamp[0, 0]:.1f}s - {timestamp[-1, -1]:.1f}s]:\nTimestamp Matrix:\n{timestamp_str}\n")})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    content.append(
        {
            "type": "text",
            "text": (
                "Video Subtitles:\n"
                f"{subtitles_str}\n\n"
                f"Question:\n{question}\n\n"
                "You are the overview observer in a video QA agent. Do not answer the question.\n"
                "Describe the global timeline and mark scene/window regions that may need later real-frame skim or focus.\n"
                f"Use numeric seconds without unit suffixes for all JSON time fields, including t_range and suggest_focus_windows. Frame timestamps must match these values: {allowed_timestamps}.\n"
                + (
                    f"The timestamp_observations array must contain exactly {num_frames} rows, one for every "
                    f"provided timestamp in chronological order. Give each frame a factual caption of up to "
                    f"{max(1, int(config.get('overview_frame_caption_words') or 50))} words, even when it appears irrelevant. "
                    "Mention the visible setting or background structure, salient people count, and query-relevant objects/actions when visible; do not only narrate the foreground action. "
                    "Do not merge or omit timestamps.\n"
                    if bool(config.get("overview_require_all_timestamps"))
                    else ""
                )
                + "For scene_summaries, possible_evidence means possible evidence for the given Question, not merely an interesting event.\n"
                + "Return ONLY valid JSON with this schema:\n"
                "{\n"
                "  \"global_summary\": \"one concise whole-video summary\",\n"
                "  \"timestamp_observations\": [\n"
                "    {\"timestamp_s\": 1.0, \"scene_id\": \"overview\", \"description\": \"what is visible\", \"event_tags\": [\"enter\"], \"needs_focus\": \"\"}\n"
                "  ],\n"
                "  \"scene_summaries\": [\n"
                "    {\"scene_id\": \"overview\", \"t_range\": [0.0, 10.0], \"summary\": \"what happens\", \"possible_evidence\": true, \"suggest_focus_windows\": [[3.0, 8.0]], \"missing_detail\": \"what real frames should verify\"}\n"
                "  ],\n"
                "  \"open_gaps\": []\n"
                "}\n"
            ),
        }
    )
    raw, observer_backend = observe_content(
        config,
        content=content,
        tool_name="overview",
        tool_mode=None,
        output_dir=parameters.get("output_dir"),
        return_json=True,
    )

    if raw is None:
        return "Overview observation failed: model response is empty."
    if raw == "":
        return "Overview observation failed: model response is empty."
    parsed = extract_json_object(raw) or {}
    if not parsed and raw:
        parsed = {
            "global_summary": "Overview model returned unstructured text.",
            "timestamp_observations": [],
            "scene_summaries": [],
            "open_gaps": ["overview output could not be parsed as JSON"],
        }
    _normalize_overview_times(parsed, duration=duration)
    if bool(config.get("overview_require_all_timestamps")):
        _normalize_timestamp_observations(
            parsed,
            allowed_timestamps=allowed_timestamps,
            fill_omitted=bool(config.get("overview_fill_omitted_timestamps", True)),
        )
    parsed.setdefault("observer_backend", observer_backend)
    parsed["overview_sampling_mode_requested"] = requested_sampling_mode
    parsed["overview_sampling_mode_effective"] = effective_sampling_mode
    parsed["overview_requested_frame_count"] = int(
        config.get("overview_num_frames")
        or int(config.get("frame_sampling_factor") or general_config["frame_sampling_factor"])
        * int(config.get("overview_base") or general_config["overview_base"])
    )
    parsed["overview_actual_frame_count"] = num_frames
    parsed["overview_available_iframe_count"] = len(available_keyframes)
    parsed["overview_sampled_timestamps"] = allowed_timestamps
    return format_v10_observation(parsed, fallback_text=raw)


def _execute_skeleton_overview(config: dict, parameters: dict) -> str:
    vr = parameters["vr"]
    duration = float(parameters.get("duration") or round(len(vr) / vr.get_avg_fps(), 1))
    video_path = parameters.get("video_path", "")
    question = parameters.get("question", "")
    subtitles = parameters["subtitles"]
    subtitles_str = convert_to_free_form_text_representation(
        subtitles,
        content_type="subtitle",
    )

    skeleton, cache_path = load_or_build_skeleton(
        config,
        video_path=video_path,
        duration_s=duration,
    )
    skeleton, caption_created = ensure_skeleton_captions(
        config,
        skeleton=skeleton,
        video_path=video_path,
        vr=vr,
        output_dir=parameters.get("output_dir"),
    )

    skeleton_text = skeleton_to_prompt_text(
        skeleton,
        max_scenes=int(config.get("skeleton_overview_max_scenes") or 30),
    )
    content = [
        {
            "type": "text",
            "text": (
                "You are the task-aware overview observer in a video QA agent.\n"
                "You receive an offline codec skeleton memory instead of raw full-video frames.\n"
                "Do not answer the final question. Use the skeleton to build a concise global map, "
                "choose candidate scenes/windows, and identify what should be skimmed or focused next.\n\n"
                "Rules:\n"
                "- Use existing scene_id values from the skeleton.\n"
                "- Use scene t_range, anchor timestamps, and motion_peaks to propose bounded windows.\n"
                "- Avoid suggesting a full-video skim unless no scene-level candidate exists.\n"
                "- If an event is missing, mark it as an open_gap and suggest a small set of candidate scenes or windows.\n"
                "- Treat skeleton captions as routing memory, not final proof.\n\n"
                f"Question:\n{question}\n\n"
                f"Video Subtitles:\n{subtitles_str}\n\n"
                f"Offline Skeleton Memory:\n{skeleton_text}\n\n"
                "Return ONLY valid JSON with this schema:\n"
                "{\n"
                "  \"global_summary\": \"one concise task-aware whole-video overview\",\n"
                "  \"timestamp_observations\": [\n"
                "    {\"timestamp_s\": 1.0, \"scene_id\": \"S000\", \"description\": \"what the skeleton indicates\", \"event_tags\": [\"event\"], \"needs_focus\": \"\"}\n"
                "  ],\n"
                "  \"scene_summaries\": [\n"
                "    {\"scene_id\": \"S000\", \"t_range\": [0.0, 10.0], \"summary\": \"why this scene may matter\", \"possible_evidence\": true, \"suggest_focus_windows\": [[3.0, 8.0]], \"missing_detail\": \"what real frames should verify\"}\n"
                "  ],\n"
                "  \"open_gaps\": [\n"
                "    {\"gap\": \"missing event or option coverage\", \"suggested_window\": [12.0, 20.0]}\n"
                "  ]\n"
                "}\n"
            ),
        }
    ]
    raw, observer_backend = observe_content(
        config,
        content=content,
        tool_name="overview",
        tool_mode="skeleton_text",
        output_dir=parameters.get("output_dir"),
        return_json=True,
    )
    if not raw:
        return "Skeleton overview observation failed: model response is empty."
    parsed = extract_json_object(raw) or {}
    if not parsed:
        parsed = {
            "global_summary": "Skeleton overview returned unstructured text.",
            "timestamp_observations": [],
            "scene_summaries": [],
            "open_gaps": [
                {
                    "gap": "skeleton overview output could not be parsed",
                    "suggested_window": None,
                }
            ],
        }
    parsed.setdefault("observer_backend", observer_backend)
    parsed.setdefault("skeleton_overview", True)
    parsed.setdefault("skeleton_caption_created", caption_created)
    parsed.setdefault("skeleton_cache_path", str(cache_path))
    return format_v10_observation(parsed, fallback_text=raw)
