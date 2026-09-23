import base64
import math
import re
import time
from io import BytesIO

import numpy as np
from PIL import Image

from videoseek.codec import timestamps_to_frame_indices
from videoseek.core.minimal_global_fsm import should_use_minimal_global_fsm
from videoseek.observer import observe_content
from videoseek.skeleton import (
    get_scene_context_for_window,
    scene_aware_timestamps,
    scene_context_to_prompt_text,
)
from videoseek.tools.v10_format import format_v10_observation, snap_timestamp_observations, local_search_instruction
from videoseek.tools.temporal_sampling import (
    merge_mandatory_timestamps,
    supplement_frame_indices,
)
from videoseek.utils import convert_to_free_form_text_representation, extract_json_object
from config import general_config


focus_qwen_tool = {
    "type": "function",
    "function": {
        "name": "focus_qwen",
        "description": (
            "Use local Qwen to cheaply localize likely evidence frames inside a short clip. "
            "It is only a routing/localization tool; final visual decisions must use frame_verify. "
            "When several disjoint short candidates are already known, pass them together in windows "
            "so they are inspected in one agent action."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise visual detail to localize in this short clip.",
                },
                "start_time": {
                    "type": "number",
                    "description": "The start time of the video clip to inspect.",
                },
                "end_time": {
                    "type": "number",
                    "description": "The end time of the video clip to inspect.",
                },
                "mode": {
                    "type": "string",
                    "description": "detail_verify, option_verify, or wider_context.",
                },
                "windows": {
                    "type": "array",
                    "description": (
                        "Optional disjoint absolute/global [start, end] candidate windows. "
                        "start_time and end_time must enclose them."
                    ),
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "minItems": 1,
                    "maxItems": 4,
                },
            },
            "required": ["query", "start_time", "end_time", "mode"],
            "additionalProperties": False,
        },
    },
}


def _frame_to_resized_array(frame: np.ndarray, *, short_side: int) -> np.ndarray:
    image = Image.fromarray(frame)
    width, height = image.size
    shortest = max(1, min(width, height))
    scale = float(short_side) / float(shortest)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.array(image.resize(target, Image.BICUBIC))


def focus_qwen_frame_budget(config: dict, *, window_s: float) -> int:
    """Return the focus frame count cap for fixed-budget or FPS sampling."""
    max_frames = max(1, int(config.get("focus_qwen_max_frames") or 8))
    if not bool(config.get("focus_qwen_temporal_density_enabled")):
        return max_frames
    target_fps = max(0.05, float(config.get("focus_qwen_target_fps") or 1.0))
    return min(max_frames, max(1, int(math.ceil(max(0.0, window_s) * target_fps))))


def _array_to_data_url(image_array: np.ndarray) -> str:
    image = Image.fromarray(image_array)
    output = BytesIO()
    image.save(output, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("utf-8")


def _make_2x2_sheet(frames: list[np.ndarray]) -> np.ndarray:
    h, w, c = frames[0].shape
    padded = list(frames)
    while len(padded) < 4:
        padded.append(np.zeros((h, w, c), dtype=np.uint8))
    return (
        np.stack(padded[:4], axis=0)
        .reshape(2, 2, h, w, c)
        .transpose(0, 2, 1, 3, 4)
        .reshape(2 * h, 2 * w, c)
    )


def _fallback_payload(
    *,
    raw: str | None,
    start_time: float,
    end_time: float,
    scene_id: str,
    window_id: str,
    observer_backend: str,
    observer_wall_s: float,
    num_frames: int,
) -> dict:
    text = (raw or "").strip()
    return {
        "window_id": window_id,
        "scene_id": scene_id,
        "t_range": [round(float(start_time), 1), round(float(end_time), 1)],
        "timestamp_observations": [],
        "overall_summary": text[:280],
        "supports_options": [],
        "contradicts_options": [],
        "detail_sufficient": False,
        "missing_detail": "local Qwen output was not valid compact JSON",
        "observer_backend": observer_backend,
        "observer_wall_s": round(observer_wall_s, 3),
        "num_frames": num_frames,
        "parse_ok": False,
    }


def _as_bool(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _compact_visual_query(query: str, *, max_chars: int = 420) -> str:
    """Keep local Qwen focused on visual localization, not option reasoning."""
    text = str(query or "").replace("\r", "\n")
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith(("question and choices", "please directly answer")):
            continue
        if re.match(r"^\(?[a-d]\)?\s*[\).:：-]", line, flags=re.IGNORECASE):
            continue
        if "answer with" in lower and "option" in lower:
            continue
        kept.append(line)
    compact = " ".join(kept) if kept else re.sub(r"\s+", " ", text).strip()
    compact = re.sub(r"(?i)^continue evidence search before answering\.?\s*", "", compact)
    compact = re.sub(r"\b(A|B|C|D)\s*:\s*[^.;]+[.;]?", "", compact)
    compact = re.sub(r"\s+", " ", compact).strip()
    if len(compact) > max_chars:
        compact = compact[: max_chars - 3].rstrip(" ,.;:") + "..."
    return compact or "localize the visually relevant evidence"


def _is_template_polluted_caption(caption: str) -> bool:
    text = " ".join(str(caption or "").strip().lower().split())
    if not text:
        return True
    compact = re.sub(r"\s+", "", text)
    if compact in {"low|medium", "medium|high", "low|medium|high"}:
        return True
    if {"low", "medium"}.issubset(set(compact.split("|"))) and len(compact.split("|")) <= 3:
        return True
    bad_exact = {
        "actual visible content",
        "low|medium|high",
        "<=5 words",
        "<=7 words",
        "specific detail frame_verify should check",
    }
    if text in bad_exact:
        return True
    if "|" in text and all(token in text for token in ("low", "medium", "high")):
        return True
    if text.startswith("<=") and "word" in text:
        return True
    return False


def _valid_global_window(value, *, start_time: float, end_time: float):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start = float(value[0])
        end = float(value[1])
    except Exception:
        return None
    if end <= start:
        return None
    eps = 1e-3
    if start < start_time - eps or end > end_time + eps:
        return None
    return [round(start, 1), round(end, 1)]


def _clamp_window_around_center(
    start_time: float,
    end_time: float,
    *,
    duration: float,
    max_window_s: float,
) -> tuple[float, float]:
    """Shrink a short focus request so context padding will not exceed tool limits."""
    start_time = max(0.0, min(float(start_time), float(duration)))
    end_time = max(start_time, min(float(end_time), float(duration)))
    window_s = end_time - start_time
    if max_window_s <= 0 or window_s <= max_window_s:
        return start_time, end_time

    center = (start_time + end_time) / 2.0
    half = max_window_s / 2.0
    new_start = center - half
    new_end = center + half
    if new_start < 0.0:
        new_end = min(float(duration), new_end - new_start)
        new_start = 0.0
    if new_end > float(duration):
        overflow = new_end - float(duration)
        new_start = max(0.0, new_start - overflow)
        new_end = float(duration)
    if new_end - new_start > max_window_s:
        new_end = min(float(duration), new_start + max_window_s)
    return new_start, new_end


def _normalize_frame_rows(
    parsed: dict,
    *,
    allowed_timestamps: list[float],
    scene_id: str,
    caption_char_limit: int = 120,
) -> list[dict]:
    raw_rows = (
        parsed.get("frames")
        or parsed.get("frame_captions")
        or parsed.get("frame_tags")
        or []
    )
    if not isinstance(raw_rows, list):
        return []
    allowed = [round(float(item), 1) for item in allowed_timestamps]
    rows: list[dict] = []
    seen: set[str] = set()
    for idx, row in enumerate(raw_rows):
        frame_id = f"F{idx + 1:02d}"
        timestamp = allowed[min(idx, len(allowed) - 1)] if allowed else 0.0
        caption = ""
        relevance = "low"
        target_match = "unknown"
        event_match = "unknown"
        if isinstance(row, dict):
            fxx_keys = [key for key in row.keys() if re.match(r"^F\d{2}$", str(key))]
            if fxx_keys and not row.get("frame_id"):
                frame_id = str(fxx_keys[0])
                try:
                    timestamp = float(row.get(fxx_keys[0], timestamp))
                except Exception:
                    pass
            else:
                frame_id = str(row.get("frame_id") or row.get("id") or frame_id)
            try:
                timestamp = float(row.get("timestamp_s", row.get("timestamp", timestamp)))
            except Exception:
                pass
            caption = str(
                row.get("caption")
                or row.get("tag")
                or row.get("description")
                or row.get("describe visible frame")
                or row.get("visible frame")
                or row.get("visible_content")
                or row.get("content")
                or ""
            )
            relevance = str(row.get("query_relevance") or row.get("relevance") or row.get("confidence") or relevance)
            target_match = str(row.get("target_match") or target_match)
            event_match = str(row.get("event_match") or event_match)
        elif isinstance(row, (list, tuple)):
            if len(row) >= 1 and row[0] is not None:
                frame_id = str(row[0])
            if len(row) >= 2:
                try:
                    timestamp = float(row[1])
                except Exception:
                    pass
            if len(row) >= 3 and row[2] is not None:
                caption = str(row[2])
            if len(row) >= 4 and row[3] is not None:
                relevance = str(row[3])
        else:
            continue
        if not caption.strip():
            continue
        if _is_template_polluted_caption(caption):
            continue
        if allowed:
            timestamp = min(allowed, key=lambda candidate: abs(candidate - float(timestamp)))
        if frame_id in seen:
            continue
        seen.add(frame_id)
        relevance = relevance.strip().lower()
        if relevance not in {"low", "medium", "high", "unknown"}:
            relevance = "unknown"
        target_match = target_match.strip().lower()
        if target_match not in {"matched", "possible", "not_matched", "unknown"}:
            target_match = "unknown"
        event_match = event_match.strip().lower()
        if event_match not in {"direct", "context_only", "not_matched", "unknown"}:
            event_match = "unknown"
        rows.append(
            {
                "frame_id": frame_id,
                "timestamp_s": round(float(timestamp), 1),
                **({"qwen_omitted": True} if isinstance(row, dict) and row.get("qwen_omitted") else {}),
                "caption": caption.strip()[:caption_char_limit],
                "query_relevance": relevance,
                "target_match": target_match,
                "event_match": event_match,
                "scene_id": scene_id,
            }
        )
    return rows


def _fill_missing_frame_rows(
    rows: list[dict],
    *,
    allowed_timestamps: list[float],
    scene_id: str,
    reason: str = "qwen omitted frame",
) -> list[dict]:
    existing = {str(row.get("frame_id")) for row in rows}
    by_id = {str(row.get("frame_id")): row for row in rows}
    filled: list[dict] = []
    for idx, ts in enumerate(allowed_timestamps):
        frame_id = f"F{idx + 1:02d}"
        if frame_id in existing:
            filled.append(by_id[frame_id])
            continue
        filled.append(
            {
                "frame_id": frame_id,
                "timestamp_s": round(float(ts), 1),
                "caption": reason,
                "query_relevance": "unknown",
                "target_match": "unknown",
                "event_match": "unknown",
                "scene_id": scene_id,
                "qwen_omitted": True,
            }
        )
    return filled


def _parse_caption_only_rows(
    raw: str | None,
    *,
    allowed_timestamps: list[float],
    scene_id: str,
    caption_char_limit: int = 120,
) -> list[dict]:
    """Parse plain Qwen caption lines into normalized frame rows.

    Expected format is intentionally simple:
      F01: visible content | relevance=low|medium|high
    """
    text = str(raw or "")
    allowed = [round(float(item), 1) for item in allowed_timestamps]
    rows: list[dict] = []
    seen: set[str] = set()
    line_re = re.compile(
        r"^\s*(?:[-*]\s*)?(?P<frame>F\d{1,2})\s*(?:\([^)]*\))?\s*(?:[:：|\\-–—])\s*(?P<body>.+?)\s*$",
        flags=re.IGNORECASE,
    )
    for line in text.splitlines():
        clean = line.strip().strip("`")
        if not clean:
            continue
        match = line_re.match(clean)
        if not match:
            continue
        frame_id = match.group("frame").upper()
        if len(frame_id) == 2:
            frame_id = f"F0{frame_id[-1]}"
        try:
            idx = max(0, int(frame_id[1:]) - 1)
        except Exception:
            continue
        if idx >= len(allowed) or frame_id in seen:
            continue

        body = re.sub(r"^\s*caption\s*[:：]\s*", "", match.group("body").strip(), flags=re.IGNORECASE)
        if _is_template_polluted_caption(body):
            continue

        relevance_field = re.search(
            r"\b(?:relevance|confidence)\s*[:=]\s*(low|medium|high)\b",
            body, flags=re.IGNORECASE,
        )
        relevance = relevance_field.group(1).lower() if relevance_field else "unknown"

        # Ordinary punctuation belongs to the observation. Only an explicit
        # metadata key starts the tag suffix (including legacy ; and / forms).
        caption = re.split(
            r"\s*[|/;]\s*(?=(?:relevance|confidence|target(?:_match)?|event(?:_match)?)\s*[:=])",
            body, maxsplit=1, flags=re.IGNORECASE,
        )[0]
        caption = re.sub(r"\s+(?:relevance|confidence)\s*[:=]\s*(?:low|medium|high)\b.*$",
                         "", caption, flags=re.IGNORECASE)
        caption = caption.strip(" |,.;")
        target_match_match = re.search(
            r"\btarget(?:_match)?\s*[:=]\s*(matched|possible|not_matched|unknown)\b",
            body,
            flags=re.IGNORECASE,
        )
        event_match_match = re.search(
            r"\bevent(?:_match)?\s*[:=]\s*(direct|context_only|not_matched|unknown)\b",
            body,
            flags=re.IGNORECASE,
        )
        target_match = (
            target_match_match.group(1).lower() if target_match_match else "unknown"
        )
        event_match = (
            event_match_match.group(1).lower() if event_match_match else "unknown"
        )
        if not caption or _is_template_polluted_caption(caption):
            continue

        seen.add(frame_id)
        rows.append(
            {
                "frame_id": frame_id,
                "timestamp_s": allowed[idx],
                "caption": caption[:caption_char_limit],
                "query_relevance": relevance,
                "target_match": target_match,
                "event_match": event_match,
                "scene_id": scene_id,
            }
        )
    return rows


def _caption_summary_from_rows(rows: list[dict], *, max_chars: int = 180) -> str:
    captions: list[str] = []
    for row in rows:
        caption = str(row.get("caption") or "").strip()
        if not caption:
            continue
        if str(row.get("query_relevance") or "").lower() in {"medium", "high"}:
            captions.append(caption)
    if not captions:
        captions = [str(row.get("caption") or "").strip() for row in rows if row.get("caption")]
    deduped: list[str] = []
    seen: set[str] = set()
    for caption in captions:
        key = caption.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(caption)
        if len(deduped) >= 4:
            break
    return "; ".join(deduped)[:max_chars]


def _frame_rows_to_timestamp_observations(rows: list[dict], *, scene_id: str, window_id: str) -> list[dict]:
    observations: list[dict] = []
    for row in rows:
        event_tags = ["qwen_focus_frame"]
        if row.get("qwen_omitted"):
            event_tags.append("qwen_omitted_frame")
        observations.append(
            {
                "timestamp_s": row["timestamp_s"],
                **{k: row[k] for k in ("frame_index", "source_timestamp_s") if k in row},
                "scene_id": scene_id,
                "window_id": window_id,
                "description": row["caption"],
                "event_tags": event_tags,
                "confidence": row["query_relevance"],
                "visual_confidence": "unknown",
                "target_match": row.get("target_match") or "unknown",
                "event_match": row.get("event_match") or "unknown",
                "needs_focus": "",
                "frame_ids": [row["frame_id"]],
            }
        )
    return observations


def _execute_focus_qwen_single(config: dict, parameters: dict) -> str:
    query = parameters["query"]
    qwen_query = _compact_visual_query(
        query,
        max_chars=int(config.get("focus_qwen_query_max_chars") or 420),
    )
    raw_start_time = float(parameters["start_time"])
    raw_end_time = float(parameters["end_time"])
    mode = parameters.get("mode", "detail_verify")
    observer_mode = (
        "local_observation" if parameters.get("local_search_question")
        and not should_use_minimal_global_fsm(parameters["local_search_question"]) else mode
    )
    vr = parameters["vr"]
    video_path = parameters.get("video_path", "")
    duration = float(parameters.get("duration") or round(len(vr) / vr.get_avg_fps(), 1))

    start_time = max(0.0, min(raw_start_time, duration))
    end_time = max(start_time, min(raw_end_time, duration))
    requested_start_time = start_time
    requested_end_time = end_time
    max_window_s = float(config.get("focus_qwen_max_window_s") or 20.0)
    context_pad_s = float(config.get("focus_qwen_context_pad_s") or 2.0)
    clamp_info: dict | None = None
    if (
        context_pad_s > 0
        and _as_bool(config.get("focus_qwen_clamp_before_padding"), default=True)
    ):
        requested_window_s = end_time - start_time
        max_prepad_window_s = max(1.0, max_window_s - 2.0 * context_pad_s)
        # Only fix near-limit focus windows that become invalid after padding.
        # Genuinely long windows should still be routed to skim_qwen.
        if max_prepad_window_s < requested_window_s <= max_window_s:
            clamped_start, clamped_end = _clamp_window_around_center(
                start_time,
                end_time,
                duration=duration,
                max_window_s=max_prepad_window_s,
            )
            if (round(clamped_start, 3), round(clamped_end, 3)) != (
                round(start_time, 3),
                round(end_time, 3),
            ):
                clamp_info = {
                    "requested_t_range": [
                        round(float(requested_start_time), 1),
                        round(float(requested_end_time), 1),
                    ],
                    "clamped_t_range_before_padding": [
                        round(float(clamped_start), 1),
                        round(float(clamped_end), 1),
                    ],
                    "max_prepad_window_s": round(float(max_prepad_window_s), 1),
                    "context_pad_s": round(float(context_pad_s), 1),
                }
                start_time, end_time = clamped_start, clamped_end
    if context_pad_s > 0:
        start_time = max(0.0, start_time - context_pad_s)
        end_time = min(duration, end_time + context_pad_s)

    if end_time - start_time > max_window_s:
        payload = _fallback_payload(
            raw=(
                f"focus_qwen window is too long ({end_time - start_time:.1f}s). "
                f"Use skim_qwen for windows longer than {max_window_s:.1f}s."
            ),
            start_time=start_time,
            end_time=end_time,
            scene_id="focus_qwen",
            window_id="focus_qwen_window",
            observer_backend="local_qwen_not_called",
            observer_wall_s=0.0,
            num_frames=0,
        )
        payload["missing_detail"] = (
            f"focus_qwen is short-window only; route this {end_time - start_time:.1f}s "
            "window to skim_qwen first."
        )
        if clamp_info:
            payload["focus_qwen_window_clamp"] = clamp_info
        return format_v10_observation(payload, fallback_text=payload["overall_summary"])

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
    scene_id = str((scene_context or {}).get("scene_id") or "focus_qwen")
    window_id = f"{scene_id}_focus_qwen_window" if scene_id != "focus_qwen" else "focus_qwen_window"
    scene_context_text = scene_context_to_prompt_text(scene_context)

    total_frames = len(vr)
    fps = vr.get_avg_fps()
    start_frame = min(int(start_time * fps), total_frames - 1)
    end_frame = min(max(start_frame + 1, int(end_time * fps)), total_frames - 1)
    max_num_frames = focus_qwen_frame_budget(
        config,
        window_s=end_time - start_time,
    )
    num_frames = min(max(1, int(end_frame - start_frame)), max_num_frames)
    sample_timestamps = scene_aware_timestamps(
            video_path=video_path,
            duration_s=duration,
            start_time=start_time,
            end_time=end_time,
            num_frames=num_frames,
            scene=scene_context,
        )
    sample_timestamps = merge_mandatory_timestamps(
        sample_timestamps,
        parameters.get("mandatory_timestamps"),
        start_time=start_time,
        end_time=end_time,
        budget=num_frames,
    )
    frame_indices = timestamps_to_frame_indices(
        timestamps=sample_timestamps,
        fps=fps,
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
        frame_indices = frame_indices[:num_frames]

    cur_timestamps = [round(float(idx) / fps, 1) for idx in frame_indices]
    frames = vr.get_batch(frame_indices).asnumpy()
    short_side = int(config.get("focus_qwen_short_side") or 320)
    resized_frames = [_frame_to_resized_array(frame, short_side=short_side) for frame in frames]
    frame_caption_words = max(1, int(config.get("focus_qwen_frame_caption_words") or 7))
    summary_words = max(1, int(config.get("focus_qwen_summary_words") or 16))
    caption_char_limit = max(120, frame_caption_words * 12)
    caption_only_output = _as_bool(config.get("qwen_caption_only_output"), default=False) or _as_bool(
        config.get("focus_qwen_caption_only_output"),
        default=False,
    )

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "You are focus_qwen, a local visual observer for a video QA agent.\n"
                "Your job is cheap short-window localization for frame_verify, not final answering.\n"
                "The images are individual video frames in chronological order. Each image "
                "has an Fxx id and an absolute/global video timestamp.\n"
                + (
                    "Caption each inspected frame only. Do not output JSON, markdown, bullets, "
                    "answer options, support labels, or final decisions.\n"
                    if caption_only_output
                    else (
                        "Caption each inspected frame briefly, then choose candidate frame ids and "
                        "one absolute/global frame_verify window if useful. If the target "
                        "object/action is absent but the frames clearly show a different "
                        "object/action, say that clearly as routing information. Do not treat a "
                        "different object as support for the target.\n"
                    )
                )
                +
                f"Describe each inspected frame in about {frame_caption_words} words when visual detail is useful; "
                "shorter is fine for blank or irrelevant frames.\n"
                + (
                    "Return plain text lines only.\n\n"
                    if caption_only_output
                    else "Be concise. Return only valid compact JSON.\n\n"
                )
                +
                f"Clip range: {start_time:.1f}s - {end_time:.1f}s\n"
                f"Mode: {observer_mode}\n"
                + (local_search_instruction(parameters["local_search_question"], query)
                   if parameters.get("local_search_question") else f"Visual target: {qwen_query}\n")
                +
                f"Subtitles in this clip:\n{subtitles_str}\n\n"
                + (
                    "Skeleton scene context for this focus window:\n"
                    f"{scene_context_text}\n\n"
                    if scene_context_text
                    else ""
                )
            ),
        }
    ]

    cell_lines: list[str] = []
    cell_counter = 1
    for frame, ts in zip(resized_frames, cur_timestamps):
        cell_id = f"F{cell_counter:02d}"
        line = f"{cell_id}: timestamp_s={ts:.1f}"
        cell_lines.append(line)
        cell_counter += 1
        content.append({"type": "text", "text": "Frame:\n" + line})
        content.append({"type": "image_url", "image_url": {"url": _array_to_data_url(frame)}})

    allowed_timestamps_str = ", ".join(f"{ts:.1f}" for ts in cur_timestamps)
    if caption_only_output:
        final_instruction = (
            "Valid frame ids and absolute/global timestamps:\n"
            + "\n".join(cell_lines)
            + "\n\n"
            f"There are exactly {len(cur_timestamps)} frames. Return exactly {len(cur_timestamps)} plain-text lines, "
            "one per Fxx frame, in order.\n"
            "Line format: F01: visible caption <= "
            f"{frame_caption_words} words | relevance=low|medium|high | "
            "target_match=matched|possible|not_matched|unknown | "
            "event_match=direct|context_only|not_matched|unknown\n"
            "Use only visible content. If the frame is irrelevant, still describe what is visible and set relevance=low.\n"
            "Relevance policy: high only when the complete visual target and its distinguishing relation/action are directly visible; "
            "medium when only some target entities or scene context are visible; low when absent or unclear.\n"
            "target_match describes whether the queried entity is visibly present. event_match=direct only when the queried action/state is visibly occurring; context_only is related context without the event.\n"
            "Do not output JSON. Do not output markdown. Do not copy the format placeholder as content."
        )
    else:
        final_instruction = (
            "Valid frame ids and absolute/global timestamps:\n"
            + "\n".join(cell_lines)
            + "\n\n"
            f"Allowed absolute/global timestamps: [{allowed_timestamps_str}]. Use only these timestamps.\n"
            f"Any suggest_frame_verify_window must be absolute/global seconds inside [{start_time:.1f}, {end_time:.1f}], or null.\n"
            "If skeleton scene context is provided, preserve the provided scene_id and window_id exactly.\n"
            "Return ONLY valid JSON with this exact compact schema. "
            "The frames array must contain one row for every listed Fxx id, in order:\n"
            "{\n"
            f"  \"window_id\": \"{window_id}\",\n"
            f"  \"scene_id\": \"{scene_id}\",\n"
            f"  \"t_range\": [{start_time:.1f}, {end_time:.1f}],\n"
            "  \"frames\": [\n"
            f"    [\"F01\", {cur_timestamps[0]:.1f}, \"describe visible frame\", \"low\"]\n"
            "  ],\n"
            f"  \"summary\": \"<={summary_words} words\",\n"
            "  \"best_frames\": [\"F01\", \"F02\"],\n"
            "  \"verify_window\": null,\n"
            "  \"verify_query\": \"specific detail frame_verify should check\",\n"
            "  \"detail_sufficient\": false,\n"
            "  \"missing\": \"what frame_verify still needs to check\"\n"
            "}\n\n"
            f"Hard limits: exactly {len(cur_timestamps)} frames rows, one per input Fxx; "
            f"frame caption <= {frame_caption_words} words; "
            f"summary <= {summary_words} words; best_frames at most 4 items; "
            "set detail_sufficient=false; do not output answer options, support labels, or option evidence; "
            "no markdown; no extra keys except the schema; do not copy schema placeholder text."
        )
    content.append({"type": "text", "text": final_instruction})

    observer_config = dict(config)
    observer_config["observer_backend"] = "local_qwen"
    observer_config["local_qwen_tools"] = "focus_qwen"
    observer_config["local_qwen_max_new_tokens"] = int(
        config.get("focus_qwen_max_new_tokens")
        or config.get("local_qwen_max_new_tokens")
        or 256
    )
    observer_config["local_qwen_max_images"] = max(
        int(config.get("local_qwen_max_images") or 24),
        max(1, len(resized_frames)),
    )
    observer_config["local_qwen_fallback_to_api"] = bool(
        config.get("focus_qwen_fallback_to_api", False)
    )

    t0 = time.time()
    try:
        raw, observer_backend = observe_content(
            observer_config,
            content=content,
            tool_name="focus_qwen",
            tool_mode=mode,
            output_dir=parameters.get("output_dir"),
        )
    except Exception as exc:
        observer_wall = time.time() - t0
        payload = _fallback_payload(
            raw=f"{type(exc).__name__}: {exc}",
            start_time=start_time,
            end_time=end_time,
            scene_id=scene_id,
            window_id=window_id,
            observer_backend="local_qwen_error",
            observer_wall_s=observer_wall,
            num_frames=len(resized_frames),
        )
        if clamp_info:
            payload["focus_qwen_window_clamp"] = clamp_info
        return format_v10_observation(payload, fallback_text=str(exc))

    observer_wall = time.time() - t0
    if caption_only_output:
        caption_rows = _parse_caption_only_rows(
            raw,
            allowed_timestamps=cur_timestamps,
            scene_id=scene_id,
            caption_char_limit=caption_char_limit,
        )
        if caption_rows:
            caption_rows = _fill_missing_frame_rows(
                caption_rows,
                allowed_timestamps=cur_timestamps,
                scene_id=scene_id,
            )
            summary = _caption_summary_from_rows(caption_rows, max_chars=180)
            parsed = {
                "window_id": window_id,
                "scene_id": scene_id,
                "t_range": [round(float(start_time), 1), round(float(end_time), 1)],
                "frames": caption_rows,
                "summary": summary,
                "overall_summary": summary,
                "best_frames": [
                    row["frame_id"]
                    for row in caption_rows
                    if row.get("query_relevance") in {"medium", "high"} and not row.get("qwen_omitted")
                ][:4],
                "candidate_frames": [
                    row["frame_id"]
                    for row in caption_rows
                    if row.get("query_relevance") in {"medium", "high"} and not row.get("qwen_omitted")
                ][:4],
                "verify_window": None,
                "suggest_frame_verify_window": None,
                "verify_query": qwen_query,
                "suggest_frame_verify_query": qwen_query,
                "detail_sufficient": False,
                "missing": "",
                "missing_detail": "",
                "caption_only_output": True,
            }
        else:
            parsed = {}
    else:
        parsed = extract_json_object(raw) or {}
    if not parsed:
        payload = _fallback_payload(
            raw=raw,
            start_time=start_time,
            end_time=end_time,
            scene_id=scene_id,
            window_id=window_id,
            observer_backend=observer_backend,
            observer_wall_s=observer_wall,
            num_frames=len(resized_frames),
        )
        if clamp_info:
            payload["focus_qwen_window_clamp"] = clamp_info
        payload["qwen_omitted_frame_ids"] = [f"F{idx + 1:02d}" for idx in range(len(cur_timestamps))]
        if _as_bool(config.get("qwen_fill_missing_frame_captions"), default=False):
            frame_rows = _fill_missing_frame_rows(
                [],
                allowed_timestamps=cur_timestamps,
                scene_id=scene_id,
                reason="qwen invalid json",
            )
            payload["frame_captions"] = frame_rows
            payload["timestamp_observations"] = _frame_rows_to_timestamp_observations(
                frame_rows,
                scene_id=scene_id,
                window_id=window_id,
            )
        return format_v10_observation(payload, fallback_text=raw)

    if not parsed.get("scene_id") or parsed.get("scene_id") == "focus_qwen":
        parsed["scene_id"] = scene_id
    if not parsed.get("window_id") or parsed.get("window_id") == "focus_qwen_window":
        parsed["window_id"] = window_id
    parsed["t_range"] = [round(float(start_time), 1), round(float(end_time), 1)]
    if clamp_info:
        parsed["focus_qwen_window_clamp"] = clamp_info
    if not parsed.get("overall_summary"):
        parsed["overall_summary"] = parsed.get("summary") or parsed.get("observed_event") or ""
    if not parsed.get("candidate_frames"):
        parsed["candidate_frames"] = parsed.get("best_frames") or []
    if not parsed.get("suggest_frame_verify_query"):
        parsed["suggest_frame_verify_query"] = parsed.get("verify_query") or ""
    if not parsed.get("missing_detail"):
        parsed["missing_detail"] = parsed.get("missing") or parsed.get("missing_detail") or ""
    if "suggest_frame_verify_window" not in parsed:
        parsed["suggest_frame_verify_window"] = parsed.get("verify_window")
    suggested_window = parsed.get("suggest_frame_verify_window")
    valid_suggested_window = _valid_global_window(
        suggested_window,
        start_time=start_time,
        end_time=end_time,
    )
    if suggested_window is not None and valid_suggested_window is None:
        parsed["invalid_suggest_frame_verify_window"] = suggested_window
    parsed["suggest_frame_verify_window"] = valid_suggested_window
    frame_rows = _normalize_frame_rows(
        parsed,
        allowed_timestamps=cur_timestamps,
        scene_id=scene_id,
        caption_char_limit=caption_char_limit,
    )
    for row in frame_rows:
        source_position = int(row["frame_id"][1:]) - 1
        if 0 <= source_position < len(frame_indices):
            row["frame_index"] = int(frame_indices[source_position])
            row["source_timestamp_s"] = float(frame_indices[source_position]) / fps
    before_ids = {row["frame_id"] for row in frame_rows if not row.get("qwen_omitted")}
    expected_ids = [f"F{idx + 1:02d}" for idx in range(len(cur_timestamps))]
    omitted_ids = [frame_id for frame_id in expected_ids if frame_id not in before_ids]
    if caption_only_output or _as_bool(config.get("qwen_fill_missing_frame_captions"), default=False):
        before_ids = {row["frame_id"] for row in frame_rows if not row.get("qwen_omitted")}
        frame_rows = _fill_missing_frame_rows(
            frame_rows,
            allowed_timestamps=cur_timestamps,
            scene_id=scene_id,
        )
        omitted_ids = [row["frame_id"] for row in frame_rows if row["frame_id"] not in before_ids]
    if frame_rows:
        parsed["frames"] = [
            [
                row["frame_id"],
                row["timestamp_s"],
                row["caption"],
                row["query_relevance"],
            ]
            for row in frame_rows
        ]
        parsed["frame_captions"] = frame_rows
        parsed["timestamp_observations"] = _frame_rows_to_timestamp_observations(
            frame_rows,
            scene_id=scene_id,
            window_id=window_id,
        )
    # Code-owned request association survives aggregation, including context frames.
    # This is routing metadata only; the observer never writes it.
    for row in parsed.get("timestamp_observations") or []:
        row["focus_request_window"] = [requested_start_time, requested_end_time]
    if omitted_ids:
        parsed["qwen_omitted_frame_ids"] = omitted_ids
    parsed["observed_frame_count"] = len(frame_rows) - len(omitted_ids)
    parsed["observer_status"] = "partial" if omitted_ids else "complete"
    has_qwen_candidate = bool(parsed.get("candidate_frames")) or any(
        row.get("query_relevance") in {"medium", "high"} for row in frame_rows
    )
    parsed["qwen_reported_detail_sufficient"] = bool(parsed.get("detail_sufficient", False))
    parsed["qwen_supports_options"] = parsed.get("supports_options") or []
    parsed["qwen_contradicts_options"] = parsed.get("contradicts_options") or []
    parsed["detail_sufficient"] = False
    parsed["supports_options"] = []
    parsed["contradicts_options"] = []
    for item in parsed.get("timestamp_observations") or []:
        if isinstance(item, dict) and (not item.get("scene_id") or item.get("scene_id") == "focus_qwen"):
            item["scene_id"] = scene_id
    snap_timestamp_observations(parsed, allowed_timestamps=cur_timestamps)
    if not parsed.get("scene_summaries"):
        summary = parsed.get("overall_summary") or ""
        if summary:
            parsed["scene_summaries"] = [
                {
                    "scene_id": scene_id,
                    "window_id": window_id,
                    "t_range": parsed.get("t_range") or [float(start_time), float(end_time)],
                    "summary": str(summary)[:180],
                    "possible_evidence": bool(has_qwen_candidate),
                    "suggest_focus_windows": [],
                    "missing_detail": str(parsed.get("missing_detail") or ""),
                }
            ]
    if scene_context:
        parsed.setdefault("matched_skeleton_scene", scene_id)
    parsed["observer_backend"] = observer_backend
    parsed["observer_wall_s"] = round(observer_wall, 3)
    parsed["num_frames"] = len(resized_frames)
    parsed["parse_ok"] = True
    return format_v10_observation(parsed, fallback_text=raw)


def _normalize_focus_windows(
    value: object,
    *,
    duration: float,
    max_windows: int,
) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    normalized: list[list[float]] = []
    seen: set[tuple[float, float]] = set()
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            start = max(0.0, min(float(item[0]), duration))
            end = max(0.0, min(float(item[1]), duration))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        key = (round(start, 3), round(end, 3))
        if key in seen:
            continue
        seen.add(key)
        normalized.append([start, end])
        if len(normalized) >= max(1, max_windows):
            break
    return normalized


def execute_focus_qwen(config: dict, parameters: dict) -> str:
    """Inspect disjoint short candidates in one planner action when requested.

    Each child window still goes through the existing per-frame caption path.
    The wrapper only aggregates observations; it does not rank candidates or
    turn Qwen routing captions into answer evidence.
    """
    if not config.get("focus_qwen_multi_window_enabled", False):
        return _execute_focus_qwen_single(config, parameters)

    vr = parameters.get("vr")
    if vr is None:
        return _execute_focus_qwen_single(config, parameters)
    duration = float(parameters.get("duration") or round(len(vr) / vr.get_avg_fps(), 1))
    windows = _normalize_focus_windows(
        parameters.get("windows"),
        duration=duration,
        max_windows=int(config.get("focus_qwen_multi_window_max_windows") or 4),
    )
    budgets = parameters.get("_frame_budgets")
    if budgets is not None and (len(budgets) != len(windows) or
                               any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in budgets)):
        raise ValueError("Invalid code-owned Fine frame allocation")
    if len(windows) <= 1:
        if budgets:
            config = dict(config, focus_qwen_max_frames=budgets[0])
        single_parameters = dict(parameters)
        single_parameters.pop("windows", None)
        if windows:
            single_parameters["start_time"], single_parameters["end_time"] = windows[0]
        return _execute_focus_qwen_single(config, single_parameters)

    child_payloads: list[dict] = []
    for index, (start_time, end_time) in enumerate(windows, start=1):
        child_parameters = dict(parameters)
        child_parameters.pop("windows", None)
        child_parameters["start_time"] = start_time
        child_parameters["end_time"] = end_time
        child_config = dict(config, focus_qwen_max_frames=budgets[index - 1]) if budgets else config
        output = _execute_focus_qwen_single(child_config, child_parameters)
        payload = extract_json_object(output)
        if not isinstance(payload, dict):
            payload = _fallback_payload(
                raw=output,
                start_time=start_time,
                end_time=end_time,
                scene_id="focus_qwen",
                window_id=f"focus_qwen_multi_W{index:02d}",
                observer_backend="local_qwen_parse_error",
                observer_wall_s=0.0,
                num_frames=0,
            )
            payload["parse_ok"] = False
        child_window_id = f"{payload.get('window_id') or 'focus_qwen_window'}_W{index:02d}"
        payload["window_id"] = child_window_id
        for item in payload.get("timestamp_observations") or []:
            if isinstance(item, dict):
                item["window_id"] = child_window_id
        for item in payload.get("scene_summaries") or []:
            if isinstance(item, dict):
                item["window_id"] = child_window_id
        child_payloads.append(payload)

    timestamp_observations = [
        item
        for payload in child_payloads
        for item in (payload.get("timestamp_observations") or [])
        if isinstance(item, dict)
    ]
    scene_summaries = [
        item
        for payload in child_payloads
        for item in (payload.get("scene_summaries") or [])
        if isinstance(item, dict)
    ]
    window_observations = []
    summary_parts = []
    for payload in child_payloads:
        span = payload.get("t_range") or []
        summary = str(payload.get("overall_summary") or payload.get("summary") or "").strip()
        if summary:
            summary_parts.append(f"{span}: {summary}")
        window_observations.append(
            {
                "scene_id": payload.get("scene_id") or "",
                "window_id": payload.get("window_id") or "",
                "t_range": span,
                "summary": summary,
                "possible_evidence": payload.get("possible_evidence"),
                "parse_ok": payload.get("parse_ok", False),
                "observer_wall_s": payload.get("observer_wall_s") or 0.0,
                "num_frames": payload.get("num_frames") or 0,
            }
        )

    scene_ids = {str(payload.get("scene_id") or "") for payload in child_payloads}
    scene_ids.discard("")
    aggregate = {
        "multi_window_focus": True,
        "window_id": "focus_qwen_multi_window",
        "scene_id": next(iter(scene_ids)) if len(scene_ids) == 1 else "multi_scene",
        "t_range": [round(min(window[0] for window in windows), 1), round(max(window[1] for window in windows), 1)],
        "requested_windows": [[round(start, 1), round(end, 1)] for start, end in windows],
        "window_observations": window_observations,
        "independent_candidate_windows": True,
        "timestamp_observations": timestamp_observations,
        "scene_summaries": scene_summaries,
        "overall_summary": " | ".join(summary_parts)[:1000],
        "possible_evidence": any(
            payload.get("possible_evidence") is True for payload in child_payloads
        ),
        "supports_options": [],
        "contradicts_options": [],
        "detail_sufficient": False,
        "missing_detail": "",
        "observer_backend": "local_qwen",
        "observer_wall_s": round(
            sum(float(payload.get("observer_wall_s") or 0.0) for payload in child_payloads),
            3,
        ),
        "num_frames": sum(int(payload.get("num_frames") or 0) for payload in child_payloads),
        "observer_batch_count": len(child_payloads),
        "observer_status": "partial" if any(p.get("observer_status") == "partial" or p.get("parse_ok") is False for p in child_payloads) else "complete",
        "observed_frame_count": sum(p.get("observed_frame_count", p.get("num_frames", 0)) for p in child_payloads),
        "parse_ok": all(payload.get("parse_ok") is True for payload in child_payloads),
    }
    return format_v10_observation(aggregate, fallback_text=aggregate["overall_summary"])
