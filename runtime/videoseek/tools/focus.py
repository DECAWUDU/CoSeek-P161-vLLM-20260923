import base64
import re
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from videoseek.codec import timestamps_to_frame_indices
from videoseek.observer import observe_content
from videoseek.skeleton import (
    get_scene_context_for_window,
    scene_aware_timestamps,
    scene_context_to_prompt_text,
)
from videoseek.tools.v10_format import format_v10_observation, snap_timestamp_observations
from .v10_format import local_receipt_contract, normalize_local_receipt

from videoseek.tools.temporal_sampling import (
    merge_mandatory_timestamps,
    supplement_frame_indices,
)
from videoseek.tools.spatial_grounding import prepare_grounded_frames
from videoseek.utils import (
    convert_to_free_form_text_representation,
    extract_json_object,
)
from config import general_config


focus_tool = {
    "type": "function",
    "function": {
        "name": "focus",
        "description": "To verify fine visual details, dense inspection of a short clip (start_time - end_time,≤ {focus_num_frames}s, at 1 FPS).".format(focus_num_frames=general_config["frame_sampling_factor"] * general_config["focus_base"]),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to focus the video. The query should be a concise question that can be answered by the video.",
                },
                "start_time": {
                    "type": "number",
                    "description": "The start time of the video to focus.",
                },
                "end_time": {
                    "type": "number",
                    "description": "The end time of the video to focus.",
                },
                "mode": {
                    "type": "string",
                    "description": "Internal focus mode: use detail_verify for static object/color/identity details, temporal_strip for actions/motion/collision/kick/fall/before-after changes, option_verify for checking a specific option, or wider_context for surrounding context.",
                },
            },
            "required": ["query", "start_time", "end_time", "mode"],
            "additionalProperties": False,
        },
    },
}


def _question_scope_hint(question: str) -> str:
    """Infer only the evidence scope, never the answer or next action."""
    text = re.sub(r"\s+", " ", str(question or "")).lower()
    if re.search(
        r"\b(throughout|entire|whole|overall|"
        r"in (?:this|the)(?: [a-z][a-z-]*){0,2} video|"
        r"across the video|total count|how many times|number of occurrences)\b",
        text,
    ):
        return "global_video"
    if re.search(
        r"\b(before|after|when|while|during|first|second time|at the moment)\b",
        text,
    ):
        return "event_instance"
    return "local_window"


def _normalize_scope_coverage(
    *,
    question_scope: str,
    reported_coverage: str,
    reported_reason: str,
    start_time: float,
    end_time: float,
    duration: float,
) -> tuple[str, str]:
    coverage = str(reported_coverage or "unknown").strip().lower()
    if coverage not in {"sufficient", "partial", "insufficient"}:
        coverage = "unknown"
    inspected_duration = max(0.0, float(end_time) - float(start_time))
    if (
        question_scope == "global_video"
        and duration > 0
        and inspected_duration / duration < 0.8
    ):
        coverage = "partial"
        local_reason = (
            f"Local window {start_time:.1f}-{end_time:.1f}s covers only part of the "
            f"{duration:.1f}s video; local visual detail may be clear, but other "
            "Candidate Pool locations remain outside this observation."
        )
        observer_reason = str(reported_reason or "").strip()
        return coverage, (
            local_reason
            + (f" Observer note: {observer_reason}" if observer_reason else "")
        )
    return coverage, str(reported_reason or "").strip()


def _direct_packet_anchor_positions(frames: np.ndarray, count: int) -> list[int]:
    """Keep a stable center view plus visually changing moments."""
    frame_count = len(frames)
    count = max(0, min(int(count), frame_count))
    if count == 0:
        return []
    center = (frame_count - 1) // 2
    selected = [center]
    motion_rows: list[tuple[float, int]] = []
    for position in range(1, frame_count):
        previous = frames[position - 1][::8, ::8].astype(np.int16)
        current = frames[position][::8, ::8].astype(np.int16)
        motion_rows.append((float(np.mean(np.abs(current - previous))), position))
    minimum_gap = max(1, frame_count // max(2, count * 2))
    for _score, position in sorted(motion_rows, reverse=True):
        if all(abs(position - existing) >= minimum_gap for existing in selected):
            selected.append(position)
        if len(selected) >= count:
            break
    if len(selected) < count:
        fallback = np.linspace(0, frame_count - 1, count + 2)[1:-1]
        for value in fallback:
            position = int(round(float(value)))
            if position not in selected:
                selected.append(position)
            if len(selected) >= count:
                break
    return sorted(selected[:count])


def _direct_temporal_sheet_data_url(
    rows: list[tuple[float, np.ndarray]],
    *,
    strip_id: str,
) -> str:
    if not rows:
        raise ValueError("direct temporal sheet requires at least one frame")
    columns = min(2, len(rows))
    sheet_rows = (len(rows) + columns - 1) // columns
    cell_width = 448
    cell_height = 252
    sheet = Image.new(
        "RGB",
        (columns * cell_width, sheet_rows * cell_height),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (timestamp, frame) in enumerate(rows):
        image = Image.fromarray(frame).convert("RGB")
        image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        cell_left = (index % columns) * cell_width
        cell_top = (index // columns) * cell_height
        x = cell_left + (cell_width - image.width) // 2
        y = cell_top + (cell_height - image.height) // 2
        sheet.paste(image, (x, y))
        label = f"{strip_id} | {timestamp:.1f}s"
        label_x = cell_left + 8
        label_y = cell_top + 8
        box = draw.textbbox((label_x, label_y), label)
        draw.rectangle(
            (box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3),
            fill=(0, 0, 0),
        )
        draw.text((label_x, label_y), label, fill=(255, 255, 255))
    output = BytesIO()
    sheet.save(output, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode(
        "utf-8"
    )


def _pil_from_image_url(item: dict) -> Image.Image | None:
    try:
        url = str(item["image_url"]["url"])
        encoded = url.split(",", 1)[1]
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    except (KeyError, IndexError, TypeError, ValueError, OSError):
        return None


def _direct_detail_sheet_data_url(
    rows: list[tuple[float, Image.Image, str]],
    *,
    tile_max_side: int,
) -> str:
    if not rows:
        raise ValueError("direct detail sheet requires at least one image")
    tile_side = max(128, int(tile_max_side))
    label_height = 34
    prepared: list[tuple[float, Image.Image, str]] = []
    for timestamp, image, label in rows:
        resized = image.convert("RGB").copy()
        resized.thumbnail((tile_side, tile_side), Image.Resampling.LANCZOS)
        prepared.append((timestamp, resized, label))
    sheet = Image.new(
        "RGB",
        (tile_side * len(prepared), tile_side + label_height),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (timestamp, image, label) in enumerate(prepared):
        left = index * tile_side
        x = left + (tile_side - image.width) // 2
        y = label_height + (tile_side - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text(
            (left + 8, 9),
            f"detail | {timestamp:.1f}s | {label}",
            fill=(255, 255, 255),
        )
    output = BytesIO()
    sheet.save(output, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode(
        "utf-8"
    )


def execute_focus(config: dict, parameters: dict) -> str:
    """
    Execute the focus tool.
    """
    query = parameters["query"]
    raw_start_time = float(parameters["start_time"])
    raw_end_time = float(parameters["end_time"])
    mode = parameters.get("mode", "detail_verify")
    max_num_frames = int(config.get("frame_sampling_factor", general_config["frame_sampling_factor"])) * int(config.get("focus_base", general_config["focus_base"]))
    vr = parameters["vr"]
    video_path = parameters.get("video_path", "")
    duration = float(parameters.get("duration") or round(len(vr) / vr.get_avg_fps(), 1))
    start_time = max(0.0, min(raw_start_time, duration))
    end_time = max(start_time, min(raw_end_time, duration))
    context_pad_s = float(config.get("focus_context_pad_s") or 0.0)
    if context_pad_s > 0:
        start_time = max(0.0, start_time - context_pad_s)
        end_time = min(duration, end_time + context_pad_s)
    subtitles = parameters["subtitles"]
    question = parameters.get("question") or ""
    question_scope_hint = _question_scope_hint(question)
    force_option_evidence = bool(parameters.get("force_option_evidence"))
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
    scene_id = str((scene_context or {}).get("scene_id") or "focus")
    window_id = f"{scene_id}_focus_window" if scene_id != "focus" else "focus_window"
    scene_context_text = scene_context_to_prompt_text(scene_context)

    # Short-window real-frame inspection. Mix uniform sampling with codec peaks
    # so collisions, pickup, give, fall, and other transitions are less likely
    # to land between sampled frames.
    total_frames = len(vr)
    start_frame = min(int(start_time * vr.get_avg_fps()), total_frames - 1)
    end_frame = min(max(start_frame + 1, int(end_time * vr.get_avg_fps())), total_frames - 1)
    num_frames = min(max(1, int(end_frame - start_frame)), max_num_frames)
    base_timestamps = scene_aware_timestamps(
        video_path=video_path,
        duration_s=duration,
        start_time=start_time,
        end_time=end_time,
        num_frames=num_frames,
        scene=scene_context,
    )
    requested_mandatory_timestamps = [
        float(value)
        for value in (parameters.get("mandatory_timestamps") or [])
        if start_time <= float(value) <= end_time
    ]
    sampled_timestamps = merge_mandatory_timestamps(
        base_timestamps,
        requested_mandatory_timestamps,
        start_time=start_time,
        end_time=end_time,
        budget=num_frames,
    )
    frame_indices = timestamps_to_frame_indices(
        timestamps=sampled_timestamps,
        fps=vr.get_avg_fps(),
        total_frames=total_frames,
    )
    if len(frame_indices) < num_frames:
        fallback_indices = np.linspace(
            start_frame,
            max(start_frame, end_frame - 1),
            num_frames,
        ).astype(int)
        frame_indices = np.asarray(
            supplement_frame_indices(
                frame_indices.tolist(),
                fallback_indices.tolist(),
                budget=num_frames,
            ),
            dtype=int,
        )
    else:
        frame_indices = np.asarray(frame_indices[:num_frames], dtype=int)
    cur_timestamps = np.array(
        [round(frame_indice / vr.get_avg_fps(), 1) for frame_indice in frame_indices],
        dtype=np.float32,
    )
    cur_timestamps_list = [round(float(item), 1) for item in cur_timestamps.tolist()]
    cur_timestamps_str = ", ".join(f"{item:.1f}s" for item in cur_timestamps_list)
    frames = vr.get_batch(frame_indices).asnumpy()

    direct_packet_enabled = bool(
        force_option_evidence
        and config.get("grounded_direct_verify_packet_enabled", False)
    )
    direct_anchor_count = max(
        1,
        int(config.get("grounded_direct_verify_packet_anchor_count") or 2),
    )
    direct_anchor_positions = (
        _direct_packet_anchor_positions(frames, direct_anchor_count)
        if direct_packet_enabled
        else []
    )
    grounded_replacements: dict[int, list[dict]] = {}
    grounding_audit: dict | None = None
    if force_option_evidence and config.get("grounded_frame_verify_enabled", False):
        grounding_config = dict(config)
        mandatory_anchor_indices: set[int] = set()
        if direct_packet_enabled:
            grounding_config["grounded_frame_verify_max_anchors"] = max(
                direct_anchor_count,
                int(config.get("grounded_frame_verify_max_anchors") or 1),
            )
            grounding_config["grounded_verify_packet_enabled"] = True
            mandatory_anchor_indices = {
                int(frame_indices[position]) for position in direct_anchor_positions
            }
        grounded_replacements, grounding_audit = prepare_grounded_frames(
            grounding_config,
            frames=frames,
            frame_indices=[int(value) for value in frame_indices.tolist()],
            timestamps=[float(value) for value in cur_timestamps.tolist()],
            mandatory_anchor_indices=mandatory_anchor_indices,
            candidate_anchor_indices={},
            query=query,
            question=question,
            output_dir=parameters.get("output_dir"),
        )

    content = [{"type": "text", "text": f"Video clip ({start_time:.1f}s - {end_time:.1f}s):\n"}]
    direct_packet_audit: dict | None = None
    if direct_packet_enabled:
        group_size = max(
            1,
            int(config.get("grounded_direct_verify_packet_group_size") or 4),
        )
        temporal_sheet_count = 0
        for start in range(0, len(frames), group_size):
            rows = [
                (float(cur_timestamps[position]), frames[position])
                for position in range(start, min(start + group_size, len(frames)))
            ]
            strip_id = f"T{temporal_sheet_count + 1:02d}"
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"{strip_id} ordered temporal sheet with {len(rows)} source "
                        "frames; read left-to-right, top-to-bottom."
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _direct_temporal_sheet_data_url(
                            rows,
                            strip_id=strip_id,
                        ),
                        "detail": "low",
                    },
                }
            )
            temporal_sheet_count += 1

        grounding_records = {
            int(item.get("position")): item
            for item in (grounding_audit or {}).get("records") or []
            if isinstance(item, dict) and isinstance(item.get("position"), int)
        }
        detail_rows: list[tuple[float, Image.Image, str]] = []
        crop_count = 0
        full_anchor_count = 0
        for position in direct_anchor_positions:
            replacement_images = [
                item
                for item in (grounded_replacements.get(position) or [])
                if item.get("type") == "image_url"
            ]
            detail_image = (
                _pil_from_image_url(replacement_images[-1])
                if replacement_images
                else None
            )
            record = grounding_records.get(position) or {}
            packet_mode = str(record.get("packet_mode") or "")
            if detail_image is not None and packet_mode == "crop_only":
                label = "crop proposal"
                crop_count += 1
            else:
                detail_image = Image.fromarray(frames[position]).convert("RGB")
                label = "full anchor"
                full_anchor_count += 1
            detail_rows.append(
                (float(cur_timestamps[position]), detail_image, label)
            )
        if detail_rows:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "High-resolution detail anchors. Validate each crop against "
                        "the ordered temporal sheets before using it as evidence."
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _direct_detail_sheet_data_url(
                            detail_rows,
                            tile_max_side=int(
                                config.get(
                                    "grounded_direct_verify_packet_detail_max_side"
                                )
                                or 960
                            ),
                        ),
                        "detail": "high",
                    },
                }
            )
        direct_packet_audit = {
            "enabled": True,
            "source_frame_count": len(frames),
            "group_size": group_size,
            "temporal_sheet_count": temporal_sheet_count,
            "anchor_count": len(detail_rows),
            "anchor_timestamps_s": [
                round(float(timestamp), 1) for timestamp, _image, _label in detail_rows
            ],
            "detail_crop_count": crop_count,
            "detail_full_anchor_count": full_anchor_count,
            "api_image_count": temporal_sheet_count + (1 if detail_rows else 0),
        }
    else:
        for position, (frame, timestamp) in enumerate(zip(frames, cur_timestamps)):
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="jpeg")
            byte_data = output_buffer.getvalue()
            base64_image = base64.b64encode(byte_data).decode("utf-8")
            content.append({"type": "text", "text": f"{timestamp:.1f}s"})
            replacement = grounded_replacements.get(position)
            if replacement:
                content.extend(replacement)
            else:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    option_evidence_instruction = ""
    option_evidence_schema = ""
    evidence_need_instruction = ""
    evidence_need_schema = ""
    if force_option_evidence:
        option_evidence_instruction = (
            "Original multiple-choice question and choices:\n"
            f"{question}\n\n"
            f"Question scope inferred from the original question: {question_scope_hint}. Preserve this exact scope even if the local verification query is narrower.\n"
            "Because this is frame_verify, bind the visual evidence to the answer choices explicitly.\n"
            "First identify the target object/person/event named by the question; if multiple visual objects could match, state the ambiguity in missing_detail.\n"
            "Separate the visible fact from target identity: a salient object in this window is not automatically the object, person, event, or relation asked about.\n"
            "Set target_match to matched only when the inspected frames establish that identity or relation; otherwise use partial, ambiguous, or not_visible.\n"
            "Independently set target_event_match: direct only when the requested action/event/relation itself is visible; context_only when the target or setting is visible without the requested event; different_event when another event is shown; ambiguous when occurrence cannot be decided; not_visible when neither the event nor usable context appears. A visible location or participant alone is not a direct event match.\n"
            "Unless the question explicitly requires simultaneity or one shot, assess participation over the temporally continuous event across adjacent shots; do not require every participant to be co-visible in one frame.\n"
            "Also classify question_scope as local_window, event_instance, or global_video. Set scope_coverage to sufficient only when this inspected window is enough for that stated scope; a verified local instance is normally partial evidence for a global_video claim.\n"
            "This target_match field is an evidence annotation, not a final-answer decision.\n"
            "Fill option_evidence for every visible/relevant option listed in the original question using exact option letters.\n"
            "status must be one of: support, weak_support, contradict, unresolved.\n"
            "Use support only when the verified visual evidence directly favors that option. "
            "Use weak_support when the evidence is suggestive but incomplete or detail_sufficient is false. "
            "Use contradict when the verified evidence argues against the option. "
            "Use unresolved when this window does not decide the option.\n"
            "Do not merge semantically close choices such as happy/excited; choose the best matching option and explain the distinction in reason.\n\n"
        )
        option_evidence_schema = (
            "  \"option_evidence\": {\n"
            "    \"A\": {\"status\": \"unresolved\", \"reason\": \"visual reason for option A\"},\n"
            "    \"B\": {\"status\": \"unresolved\", \"reason\": \"visual reason for option B\"},\n"
            "    \"C\": {\"status\": \"unresolved\", \"reason\": \"visual reason for option C\"},\n"
            "    \"D\": {\"status\": \"unresolved\", \"reason\": \"visual reason for option D\"}\n"
            "  },\n"
        )
        if config.get("packet_adaptive_evidence_need_enabled", False):
            evidence_need_instruction = (
                "Classify the principal remaining evidence need independently of the answer. "
                "Use sufficient when this window resolves the verification objective; "
                "spatial_detail only when the target and moment are bound but a small local "
                "attribute is unreadable; temporal_coverage when more moments are required; "
                "semantic_binding when the visible entity/event is not yet tied to the target; "
                "or conflict when observations disagree. For spatial_detail, provide one shown "
                "timestamp and a neutral detail_query. Do not use spatial_detail for an unseen "
                "target or missing video-wide coverage.\n\n"
            )
            evidence_need_schema = (
                '  "evidence_need": "sufficient|spatial_detail|temporal_coverage|semantic_binding|conflict",\n'
                '  "detail_target": {"timestamp_s": null},\n'
                '  "detail_query": "neutral local-detail question or empty string",\n'
            )

    content.append(
        {
            "type": "text",
            "text": (
                "Video Subtitles:\n"
                f"{subtitles_str}\n\n"
                + (
                    "Skeleton scene context for this focus window:\n"
                    f"{scene_context_text}\n\n"
                    if scene_context_text
                    else ""
                )
                +
                option_evidence_instruction
                + evidence_need_instruction
                +
                f"Question/query:\n{query}\n\n"
                f"Focus mode: {mode}\n\n"
                "You are the focus observer in a video QA agent. Inspect this short real-frame clip.\n"
                "Do not produce the final answer. Describe visible fine details, temporal change, object/person interaction, and what remains unclear.\n"
                + (
                    "All sampled timestamps are preserved in ordered temporal sheets. A separate high-resolution detail sheet contains only crop proposals or full-frame anchors. Read each temporal sheet left-to-right and top-to-bottom, use its timestamp labels, and validate detail identity against the temporal context. A crop proposal is not evidence by itself.\n"
                    if direct_packet_enabled
                    else (
                        "Some timestamp entries contain a low-resolution global image plus an untrusted crop proposed by a lightweight grounder. First validate that the crop contains the question-relevant target using the global image. Ignore the crop when it is incomplete or points to the wrong target. The grounder's proposal is not evidence by itself.\n"
                        if grounded_replacements
                        else ""
                    )
                )
                +
                "If skeleton scene context is provided, preserve the provided scene_id and window_id exactly.\n"
                f"Allowed timestamps: [{cur_timestamps_str}]. Do not invent other timestamp values.\n"
                "Use at most 6 timestamp_observations; choose representative evidence frames instead of describing every frame.\n"
                "Return ONLY valid JSON with this schema:\n"
                "{\n"
                f"  \"window_id\": \"{window_id}\",\n"
                f"  \"scene_id\": \"{scene_id}\",\n"
                f"  \"t_range\": [{float(start_time):.1f}, {float(end_time):.1f}],\n"
                "  \"timestamp_observations\": [\n"
                f"    {{\"timestamp_s\": {cur_timestamps_list[0]:.1f}, \"scene_id\": \"{scene_id}\", \"description\": \"what is visible\", \"event_tags\": [\"state_change\"], \"needs_focus\": \"\"}}\n"
                "  ],\n"
                "  \"overall_summary\": \"short local event summary\",\n"
                "  \"local_motion\": \"what changes over time\",\n"
                "  \"interaction\": \"person-object or person-person interaction if visible\",\n"
                "  \"state_change\": \"before-current-after state change\",\n"
                "  \"target_entity_or_event\": \"the object, person, event, or relation asked about\",\n"
                "  \"target_match\": \"matched|partial|ambiguous|not_visible\",\n"
                "  \"target_event_match\": \"direct|context_only|different_event|ambiguous|not_visible\",\n"
                "  \"target_binding_reason\": \"why the visible fact does or does not refer to the question target\",\n"
                "  \"question_scope\": \"local_window|event_instance|global_video\",\n"
                "  \"scope_coverage\": \"sufficient|partial|insufficient\",\n"
                "  \"scope_coverage_reason\": \"what portion of the question scope this window establishes\",\n"
                "  \"observed_fact\": \"neutral visible fact without selecting an answer\",\n"
                "  \"supports_options\": [],\n"
                "  \"contradicts_options\": [],\n"
                f"{option_evidence_schema}"
                f"{evidence_need_schema}"
                "  \"detail_sufficient\": false,\n"
                "  \"missing_detail\": \"what still needs evidence\",\n"
                "  \"suggest_focus_window\": null\n"
                "}\n"
            )
        }
    )

    local_receipt = bool(parameters.get("_local_receipt_request"))
    if local_receipt:
        receipt_cid = str(parameters.get("candidate_id") or parameters.get("window_id") or f"FV_{start_time:.1f}_{end_time:.1f}")
        receipt_candidates = [{"candidate_id": receipt_cid, "t_range": [start_time, end_time]}]
        receipt_allowed = {receipt_cid: cur_timestamps_list}
        # Require receipts only for existing explicit anchors, not every shown frame.
        explicit_anchors = {round(float(t), 1) for t in requested_mandatory_timestamps}
        receipt_anchors = {receipt_cid: (direct_packet_audit or {}).get("anchor_timestamps_s")
                           or [t for t in cur_timestamps_list if t in explicit_anchors]}
        context = f"Original question:\n{question}\nVerification objective:\n{query}\nCandidate {receipt_cid}: [{start_time:.1f}, {end_time:.1f}] seconds."
        context += f"\nVideo subtitles:\n{subtitles_str}\nScene context:\n{scene_context_text}"
        content[0]["text"], content[-1]["text"] = local_receipt_contract(context, receipt_allowed, receipt_anchors)
    observer_config = config
    if len(frames) > int(config.get("local_qwen_max_images") or 24):
        observer_config = dict(config)
        observer_config["observer_backend"] = "api"

    raw, observer_backend = observe_content(
        observer_config,
        content=content,
        tool_name="focus",
        tool_mode=mode,
        output_dir=parameters.get("output_dir"),
    )

    if raw is None and not local_receipt:
        return "Focus observation failed: model response is empty."
    parsed = extract_json_object(raw or "") or {}
    if local_receipt:
        payload = normalize_local_receipt(parsed, candidates=receipt_candidates, allowed=receipt_allowed,
                                          anchors=receipt_anchors, backend=observer_backend)
        payload.update(num_frames=len(frames), sampled_timestamps=cur_timestamps_list,
                       direct_packet_audit=direct_packet_audit, grounding_audit=grounding_audit,
                       scene_id=scene_id, window_id=receipt_cid)
        return format_v10_observation(payload)
    if parsed:
        if grounding_audit:
            parsed["grounding_audit"] = grounding_audit
        target_match = str(parsed.get("target_match") or "unknown").strip().lower()
        if target_match not in {"matched", "partial", "ambiguous", "not_visible"}:
            target_match = "unknown"
        parsed["target_match"] = target_match
        target_event_match = str(
            parsed.get("target_event_match") or "unknown"
        ).strip().lower()
        if target_event_match not in {
            "direct",
            "context_only",
            "different_event",
            "ambiguous",
            "not_visible",
        }:
            target_event_match = "unknown"
        parsed["target_event_match"] = target_event_match
        parsed["target_entity_or_event"] = str(
            parsed.get("target_entity_or_event") or ""
        ).strip()
        parsed["target_binding_reason"] = str(
            parsed.get("target_binding_reason") or ""
        ).strip()
        parsed["question_scope"] = question_scope_hint
        scope_coverage, scope_reason = _normalize_scope_coverage(
            question_scope=question_scope_hint,
            reported_coverage=parsed.get("scope_coverage") or "unknown",
            reported_reason=parsed.get("scope_coverage_reason") or "",
            start_time=start_time,
            end_time=end_time,
            duration=duration,
        )
        parsed["scope_coverage"] = scope_coverage
        parsed["scope_coverage_reason"] = scope_reason
        parsed["observed_fact"] = str(
            parsed.get("observed_fact")
            or parsed.get("overall_summary")
            or ""
        ).strip()
        if not parsed.get("scene_id") or parsed.get("scene_id") == "focus":
            parsed["scene_id"] = scene_id
        if not parsed.get("window_id") or parsed.get("window_id") == "focus_window":
            parsed["window_id"] = window_id
        parsed.setdefault("t_range", [float(start_time), float(end_time)])
        for item in parsed.get("timestamp_observations") or []:
            if isinstance(item, dict) and (not item.get("scene_id") or item.get("scene_id") == "focus"):
                item["scene_id"] = scene_id
        if not parsed.get("scene_summaries"):
            summary = (
                parsed.get("overall_summary")
                or parsed.get("interaction")
                or parsed.get("local_motion")
                or ""
            )
            if summary:
                parsed["scene_summaries"] = [
                    {
                        "scene_id": scene_id,
                        "window_id": window_id,
                        "t_range": parsed.get("t_range") or [float(start_time), float(end_time)],
                        "summary": str(summary)[:220],
                        "possible_evidence": bool(parsed.get("detail_sufficient", False)),
                        "suggest_focus_windows": (
                            [parsed["suggest_focus_window"]]
                            if isinstance(parsed.get("suggest_focus_window"), list)
                            else []
                        ),
                        "missing_detail": str(parsed.get("missing_detail") or ""),
                    }
                ]
        if scene_context:
            parsed.setdefault("matched_skeleton_scene", scene_id)
        snap_timestamp_observations(parsed, allowed_timestamps=cur_timestamps_list)
    parsed["sampling_audit"] = {
        "requested_mandatory_timestamps": [
            round(float(value), 3) for value in requested_mandatory_timestamps
        ],
        "sampled_timestamps": cur_timestamps_list,
        "mandatory_timestamp_count": len(requested_mandatory_timestamps),
        "frame_budget": num_frames,
    }
    if direct_packet_audit:
        parsed["direct_packet_audit"] = direct_packet_audit
        parsed["visual_layout"] = "direct_verify_packet"
        parsed["num_frames"] = len(frames)
    parsed.setdefault("observer_backend", observer_backend)
    return format_v10_observation(parsed, fallback_text=raw)
