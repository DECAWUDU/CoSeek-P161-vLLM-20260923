import base64
import math
import re
import time
from io import BytesIO

import numpy as np
from PIL import Image

from config import general_config
from videoseek.codec import timestamps_to_frame_indices
from videoseek.core.memory import extract_v10_payload
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


skim_qwen_tool = {
    "type": "function",
    "function": {
        "name": "skim_qwen",
        "description": (
            "Use local Qwen as a cheap coarse visual index for a candidate window. "
            "It roughly says what the window contains and suggests one short focus window. "
            "It is only a routing/localization tool; final visual decisions must use frame_verify."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise question describing what evidence to localize in this window.",
                },
                "start_time": {
                    "type": "number",
                    "description": "The start time of the candidate window.",
                },
                "end_time": {
                    "type": "number",
                    "description": "The end time of the candidate window.",
                },
                "mode": {
                    "type": "string",
                    "description": "normal, option_coverage, or alternative_window.",
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
        "observed_event": text[:220],
        "scene_tags": [],
        "possible_evidence": False,
        "relevance": 0.0,
        "suggest_focus_window": None,
        "scene_summaries": [],
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


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        return [item.strip() for item in re.split(r"[,;/|]", raw) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _compact_visual_query(query: str, *, max_chars: int = 520) -> str:
    """Keep local Qwen prompts short and visual; option reasoning belongs to API verify."""
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


def _skim_frame_budget(
    config: dict,
    *,
    window_s: float,
    query_aware_probe: bool,
    recovery_skim: bool = False,
) -> int:
    if _as_bool(config.get("skim_qwen_temporal_density_enabled"), default=False):
        target_fps = _temporal_density_target_fps(config, window_s=window_s)
        requested = max(1, int(math.ceil(max(0.0, window_s) * target_fps)))
        max_frames = max(0, int(config.get("skim_qwen_density_max_frames") or 0))
        return min(requested, max_frames) if max_frames > 0 else requested
    if recovery_skim:
        return max(
            1,
            int(
                config.get("skim_qwen_recovery_max_frames")
                or config.get("skim_qwen_long_window_frames")
                or config.get("skim_qwen_max_frames")
                or 16
            ),
        )
    if query_aware_probe:
        return int(
            config.get("query_aware_scene_probe_max_frames")
            or config.get("skim_qwen_max_frames")
            or 8
        )
    if not _as_bool(config.get("skim_qwen_dynamic_frame_budget"), default=True):
        return int(config.get("skim_qwen_max_frames") or 8)
    if window_s <= 30.0:
        target = int(config.get("skim_qwen_short_window_frames") or 8)
    elif window_s <= 60.0:
        target = int(config.get("skim_qwen_mid_window_frames") or 12)
    else:
        target = int(config.get("skim_qwen_long_window_frames") or 16)
    max_frames = int(config.get("skim_qwen_max_frames") or target)
    return max(1, min(target, max_frames))


def _density_timestamps(
    *,
    start_time: float,
    end_time: float,
    target_fps: float,
    frame_count: int | None = None,
) -> list[float]:
    """Return deterministic global timestamps at the requested temporal rate."""
    target_fps = max(0.05, float(target_fps))
    if frame_count is None:
        count = max(1, int(math.ceil(max(0.0, end_time - start_time) * target_fps)))
        step_s = 1.0 / target_fps
        return [
            round(min(end_time, start_time + idx * step_s), 6)
            for idx in range(count)
        ]
    count = max(1, int(frame_count))
    if count == 1:
        return [round((start_time + end_time) / 2.0, 6)]
    step_s = max(0.0, end_time - start_time) / (count - 1)
    return [
        round(min(end_time, start_time + idx * step_s), 6)
        for idx in range(count)
    ]


def _temporal_density_target_fps(config: dict, *, window_s: float) -> float:
    """Choose a bounded observation rate from the logical skim duration."""
    if not _as_bool(config.get("skim_qwen_adaptive_density_enabled"), default=False):
        return max(0.05, float(config.get("skim_qwen_target_fps") or 1.0))
    short_limit_s = max(1.0, float(config.get("skim_qwen_density_short_limit_s") or 15.0))
    mid_limit_s = max(
        short_limit_s,
        float(config.get("skim_qwen_density_mid_limit_s") or 30.0),
    )
    if window_s <= short_limit_s:
        value = config.get("skim_qwen_density_short_fps") or 1.0
    elif window_s <= mid_limit_s:
        value = config.get("skim_qwen_density_mid_fps") or 0.75
    else:
        value = config.get("skim_qwen_density_long_fps") or 0.5
    return max(0.05, float(value))


def _temporal_density_band(config: dict, *, window_s: float) -> str:
    if not _as_bool(config.get("skim_qwen_adaptive_density_enabled"), default=False):
        return "fixed"
    short_limit_s = max(1.0, float(config.get("skim_qwen_density_short_limit_s") or 15.0))
    mid_limit_s = max(
        short_limit_s,
        float(config.get("skim_qwen_density_mid_limit_s") or 30.0),
    )
    if window_s <= short_limit_s:
        return "short"
    if window_s <= mid_limit_s:
        return "mid"
    return "long"


def _sample_skim_frame_indices(parameters, *, start_time, end_time, num_frames,
                               density_enabled, target_fps, video_path, duration,
                               scene_context):
    """Choose pixels once for a logical window, independently of inference batches."""
    vr = parameters["vr"]
    fps, total_frames = vr.get_avg_fps(), len(vr)
    start_frame = min(int(start_time * fps), total_frames - 1)
    end_frame = min(max(start_frame + 1, int(end_time * fps)), total_frames - 1)
    sample_timestamps = (
        _density_timestamps(
            start_time=start_time,
            end_time=end_time,
            target_fps=target_fps,
            frame_count=num_frames,
        )
        if density_enabled
        else scene_aware_timestamps(
            video_path=video_path,
            duration_s=duration,
            start_time=start_time,
            end_time=end_time,
            num_frames=num_frames,
            scene=scene_context,
        )
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

    return frame_indices


def _execute_density_batched_skim_qwen(
    config: dict,
    parameters: dict,
    *,
    start_time: float,
    end_time: float,
    scene_id: str,
    window_id: str,
    query: str,
) -> str:
    """Run one logical skim as bounded local-Qwen micro-batches."""
    window_s = max(0.0, end_time - start_time)
    target_fps = _temporal_density_target_fps(config, window_s=window_s)
    batch_frames = max(1, int(config.get("skim_qwen_density_batch_frames") or 8))
    total_budget = _skim_frame_budget(
        config,
        window_s=window_s,
        query_aware_probe=bool(parameters.get("query_aware_probe", False)),
        recovery_skim=_as_bool(parameters.get("recovery_skim"), default=False),
    )
    vr = parameters["vr"]
    fps = vr.get_avg_fps()
    total_budget = min(total_budget, max(1, int((end_time - start_time) * fps)))
    selected = _sample_skim_frame_indices(parameters, start_time=start_time, end_time=end_time,
        num_frames=total_budget, density_enabled=True, target_fps=target_fps,
        video_path=str(parameters.get("video_path") or ""),
        duration=float(parameters.get("duration") or len(vr) / fps), scene_context=None)
    batch_count = max(1, int(math.ceil(len(selected) / batch_frames)))
    payloads: list[dict] = []
    child_outputs: list[str] = []

    for batch in np.array_split(selected, batch_count):
        child_parameters = dict(parameters)
        child_parameters["start_time"] = float(batch[0]) / fps
        child_parameters["end_time"] = min(end_time, (float(batch[-1]) + 1) / fps)
        child_parameters["density_micro_batch"] = True
        child_parameters["density_frame_budget"] = len(batch)
        child_parameters["_sampled_frame_indices"] = batch.tolist()
        child_output = execute_skim_qwen(config, child_parameters)
        child_outputs.append(child_output)
        child_payload = extract_v10_payload(child_output)
        if isinstance(child_payload, dict):
            payloads.append(child_payload)
            backend = str(child_payload.get("observer_backend") or "").lower()
            if (
                _as_bool(
                    config.get("skim_qwen_fail_fast_on_backend_error"),
                    default=True,
                )
                and "error" in backend
            ):
                # A persistent worker that cannot load or execute the model will
                # fail every remaining micro-batch in the same way. Preserve the
                # failure as tool health instead of repeatedly retrying and then
                # presenting an empty scan as negative visual evidence.
                break

    if not payloads:
        return child_outputs[-1] if child_outputs else format_v10_observation(
            _fallback_payload(
                raw="density skim produced no observer payload",
                start_time=start_time,
                end_time=end_time,
                scene_id=scene_id,
                window_id=window_id,
                observer_backend="local_qwen_error",
                observer_wall_s=0.0,
                num_frames=0,
            )
        )

    rows_by_frame: dict[tuple, dict] = {}
    suggested_windows: list[list[float]] = []
    missing_details: list[str] = []
    observer_backends: list[str] = []
    observer_wall_s = 0.0
    relevance = 0.0
    possible_evidence = False
    parse_ok = len(payloads) == len(child_outputs)
    batch_diagnostics: list[dict] = []

    for payload in payloads:
        observer_wall_s += float(payload.get("observer_wall_s") or 0.0)
        observer_backends.append(str(payload.get("observer_backend") or ""))
        parse_ok = parse_ok and payload.get("parse_ok") is not False
        possible_evidence = possible_evidence or bool(payload.get("possible_evidence"))
        try:
            relevance = max(relevance, float(payload.get("relevance") or 0.0))
        except Exception:
            pass
        missing = str(payload.get("missing_detail") or "").strip()
        if missing and missing not in missing_details:
            missing_details.append(missing)
        batch_diagnostics.append(
            {
                "t_range": payload.get("t_range"),
                "num_frames": payload.get("num_frames"),
                "qwen_omitted_frame_ids": payload.get("qwen_omitted_frame_ids") or [],
                "parse_ok": payload.get("parse_ok") is not False,
            }
        )
        for window in payload.get("suggest_focus_windows") or []:
            valid = _valid_global_window(window, start_time=start_time, end_time=end_time)
            if valid is None:
                continue
            if any(_window_overlap_ratio(valid, old) >= 0.85 for old in suggested_windows):
                continue
            suggested_windows.append(valid)
        frame_rows = payload.get("frame_captions") or payload.get("frames") or []
        if not frame_rows:
            frame_rows = [
                {
                    "frame_id": str((item.get("frame_ids") or [""])[0]),
                    "timestamp_s": item.get("timestamp_s"),
                    **{k: item[k] for k in ("target_match", "event_match", "frame_index", "source_timestamp_s") if k in item},
                    "caption": item.get("description") or "",
                    "query_relevance": item.get("confidence") or "low",
                    "scene_id": item.get("scene_id") or scene_id,
                    "qwen_omitted": "qwen_omitted_frame" in (item.get("event_tags") or []),
                }
                for item in (payload.get("timestamp_observations") or [])
                if isinstance(item, dict)
            ]
        for row in frame_rows:
            if not isinstance(row, dict):
                continue
            try:
                ts = round(float(row.get("timestamp_s")), 3)
            except Exception:
                continue
            key = ("frame", int(row["frame_index"])) if "frame_index" in row else ("time", ts)
            rows_by_frame.setdefault(key, dict(row))

    frame_rows = sorted(rows_by_frame.values(), key=lambda row: float(row.get("source_timestamp_s", row["timestamp_s"])))
    for idx, row in enumerate(frame_rows, start=1):
        row["frame_id"] = f"F{idx:03d}" if len(frame_rows) >= 100 else f"F{idx:02d}"
        row["scene_id"] = scene_id

    max_windows = max(1, int(config.get("skim_qwen_max_candidate_windows") or 4))
    suggested_windows = _rank_candidate_windows(
        suggested_windows,
        rows=frame_rows,
        query=query,
        max_windows=max_windows,
    )
    summary = _caption_summary_from_rows(frame_rows, max_chars=240)
    backend_values = {item for item in observer_backends if item}
    backend_error_count = sum("error" in item.lower() for item in observer_backends)
    if backend_error_count == len(observer_backends) and observer_backends:
        observer_backend = "local_qwen_error"
        observer_status = "unavailable"
    elif backend_error_count:
        observer_backend = "local_qwen_partial"
        observer_status = "partial"
    else:
        observer_backend = (
            backend_values.pop() if len(backend_values) == 1 else "local_qwen_batched"
        )
        observer_status = "partial" if any(p.get("observer_status") == "partial" for p in payloads) else "complete"
    payload = {
        "window_id": window_id,
        "scene_id": scene_id,
        "t_range": [round(start_time, 1), round(end_time, 1)],
        "timestamp_observations": _frame_rows_to_timestamp_observations(
            frame_rows,
            scene_id=scene_id,
            window_id=window_id,
        ),
        "window_summary": summary,
        "observed_event": summary,
        "possible_evidence": possible_evidence,
        "relevance": relevance,
        "suggest_focus_window": suggested_windows[0] if suggested_windows else None,
        "suggest_focus_windows": suggested_windows,
        "suggest_frame_verify_query": query,
        "missing_detail": "; ".join(missing_details[:3]),
        "caption_only_output": True,
        "observer_backend": observer_backend,
        "observer_status": observer_status,
        "observer_error_batch_count": backend_error_count,
        "observer_wall_s": round(observer_wall_s, 3),
        "num_frames": len(frame_rows),
        "observer_batch_count": len(child_outputs),
        "observer_planned_batch_count": batch_count,
        "sampling_strategy": "adaptive_temporal_density",
        "sampling_density_band": _temporal_density_band(config, window_s=window_s),
        "target_fps": target_fps,
        "requested_frame_count": max(1, int(math.ceil(window_s * target_fps))),
        "frame_budget": total_budget,
        "frame_budget_capped": total_budget < max(1, int(math.ceil(window_s * target_fps))),
        "micro_batch_frame_limit": batch_frames,
        "qwen_batch_diagnostics": batch_diagnostics,
        "effective_fps": round(len(frame_rows) / max(1e-6, window_s), 4),
        "sampled_timestamps": [round(float(row["timestamp_s"]), 3) for row in frame_rows],
        "compact_render": True,
        "parse_ok": parse_ok,
    }
    payload["scene_summaries"] = _make_scene_summary_from_parsed(
        payload,
        start_time=start_time,
        end_time=end_time,
        scene_id=scene_id,
        window_id=window_id,
    )
    return format_v10_observation(payload)


def _normalize_frame_rows(
    parsed: dict,
    *,
    allowed_timestamps: list[float],
    scene_id: str,
    caption_char_limit: int = 96,
    detail_char_limit: int = 160,
) -> list[dict]:
    raw_rows = (
        parsed.get("frames")
        or parsed.get("frame_tags")
        or parsed.get("frame_captions")
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
        matched_requirements: list[str] = []
        missing_requirements: list[str] = []
        evidence_detail = ""
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
            matched_requirements = _as_str_list(row.get("matched_requirements") or row.get("matched"))
            missing_requirements = _as_str_list(row.get("missing_requirements") or row.get("missing"))
            evidence_detail = str(row.get("evidence_detail") or row.get("detail") or "").strip()
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
            if len(row) >= 5:
                matched_requirements = _as_str_list(row[4])
            if len(row) >= 6:
                missing_requirements = _as_str_list(row[5])
            if len(row) >= 7 and row[6] is not None:
                evidence_detail = str(row[6]).strip()
        else:
            continue
        caption = caption.strip()
        if not caption:
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
                "caption": caption[:caption_char_limit],
                "query_relevance": relevance,
                "target_match": target_match,
                "event_match": event_match,
                "matched_requirements": matched_requirements[:8],
                "missing_requirements": missing_requirements[:8],
                "evidence_detail": evidence_detail[:detail_char_limit],
                "scene_id": scene_id,
            }
        )
    return rows


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
                "matched_requirements": [],
                "missing_requirements": [],
                "evidence_detail": "",
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
    caption_char_limit: int = 96,
) -> list[dict]:
    """Parse plain Qwen caption lines into skim frame rows."""
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
                "matched_requirements": [],
                "missing_requirements": [],
                "evidence_detail": "",
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


def _query_keywords(query: str) -> set[str]:
    stopwords = {
        "about",
        "after",
        "among",
        "answer",
        "before",
        "being",
        "check",
        "color",
        "does",
        "find",
        "frame",
        "from",
        "identify",
        "into",
        "look",
        "note",
        "object",
        "option",
        "scene",
        "should",
        "specific",
        "table",
        "that",
        "their",
        "there",
        "this",
        "video",
        "what",
        "where",
        "which",
        "white",
        "yellow",
        "blue",
        "green",
        "with",
    }
    words = re.findall(r"[a-zA-Z0-9]+", str(query or "").lower())
    return {word for word in words if len(word) >= 4 and word not in stopwords}


def _question_stem(question: str) -> str:
    text = str(question or "").strip()
    if not text:
        return ""
    text = re.split(r"(?im)^\s*(?:choices?|options?)\s*[:：]", text, maxsplit=1)[0]
    text = re.split(r"(?im)^\s*[A-Z]\s*[\).:：]", text, maxsplit=1)[0]
    text = re.sub(r"(?i)^\s*question\s*[:：]\s*", "", text).strip()
    return text


def _normalize_requirement_phrase(text: str) -> str:
    text = re.sub(
        r"(?is)\b(what|which|who|where|when|why|how)\b.*?\b(is|are|was|were)\b",
        "",
        str(text or ""),
    )
    text = re.sub(r"(?is)\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ?.,")


def _clean_target_requirement_phrase(text: str) -> str:
    if not text:
        return ""
    match = re.search(
        r"\b(?P<entity>woman|man|girl|boy|person|people|men|children|child)\b\s+"
        r"(?:who|that)\s+(?P<event>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return _normalize_requirement_phrase(f"{match.group('entity')} {match.group('event')}")
    match = re.search(
        r"\b(?P<entity>woman|man|girl|boy|person|people|men|children|child)\b(?P<tail>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return _normalize_requirement_phrase(f"{match.group('entity')} {match.group('tail')}")
    return _normalize_requirement_phrase(text)


def _clean_context_requirement_phrase(text: str) -> str:
    text = re.split(r"\b(at|in|near|on)\s+the\s+beginning\b", str(text or ""), maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r"\bat\s+the\s+beginning\b", "", text, flags=re.IGNORECASE)
    return _normalize_requirement_phrase(text)


def _extract_attribute_requirement(stem: str) -> str:
    if re.search(r"\bdress\b", stem, flags=re.IGNORECASE):
        return "dress/outfit color visible"
    if re.search(r"\boutfit|clothing|clothes|wear", stem, flags=re.IGNORECASE):
        return "outfit/clothing color visible"
    if re.search(r"\bcolor|colour\b", stem, flags=re.IGNORECASE):
        return "target color visible"
    return ""


def _query_requirements_for_skim(question: str, query: str) -> list[dict]:
    stem = _question_stem(question) or str(query or "")
    combined = f"{stem} {query or ''}"
    specs: list[str] = []
    relation_match = re.search(
        r"\b(while|when|as|during)\b",
        stem,
        flags=re.IGNORECASE,
    )
    if relation_match:
        before = stem[: relation_match.start()].strip(" ?.,")
        after = stem[relation_match.end() :].strip(" ?.,")
        target = _clean_target_requirement_phrase(before)
        context = _clean_context_requirement_phrase(after)
        if target:
            specs.append(target)
        if context:
            specs.append(context)
    else:
        target = _clean_target_requirement_phrase(stem)
        if target and len(_query_keywords(target)) <= 8:
            specs.append(target)

    attribute = _extract_attribute_requirement(combined)
    if attribute:
        specs.append(attribute)

    deduped: list[dict] = []
    seen: set[str] = set()
    for text in specs:
        text = re.sub(r"\s+", " ", text).strip(" ?.,")
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"id": f"R{len(deduped) + 1}", "text": text})
    return deduped[:5]


def _row_matches_query(row: dict, query_keywords: set[str]) -> bool:
    caption = str(row.get("caption") or "").lower()
    if not caption:
        return False
    if row.get("matched_requirements"):
        return True
    if str(row.get("query_relevance") or "").lower() in {"medium", "high"}:
        return True
    return any(keyword in caption for keyword in query_keywords)


def _windows_from_frame_rows(
    rows: list[dict],
    *,
    query: str,
    start_time: float,
    end_time: float,
    window_s: float,
    max_windows: int,
) -> list[list[float]]:
    if not rows or max_windows <= 0:
        return []
    query_keywords = _query_keywords(query)
    half = max(1.0, float(window_s) / 2.0)
    scored: list[tuple[tuple[int, int, float], list[float]]] = []
    for row in rows:
        if not _row_matches_query(row, query_keywords):
            continue
        ts = float(row["timestamp_s"])
        start = max(float(start_time), ts - half)
        end = min(float(end_time), ts + half)
        if end <= start:
            continue
        relevance = str(row.get("query_relevance") or "").lower()
        relevance_score = {"high": 3, "medium": 2, "low": 1}.get(relevance, 0)
        caption = str(row.get("caption") or "").lower()
        keyword_hits = sum(1 for keyword in query_keywords if keyword in caption)
        score = (relevance_score, keyword_hits, -abs(ts - ((start_time + end_time) / 2.0)))
        scored.append((score, [round(start, 1), round(end, 1)]))

    windows: list[list[float]] = []
    for _score, window in sorted(scored, key=lambda item: item[0], reverse=True):
        if any(_window_overlap_ratio(window, old) >= 0.65 for old in windows):
            continue
        windows.append(window)
        if len(windows) >= max_windows:
            break
    return _rank_candidate_windows(
        windows,
        rows=rows,
        query=query,
        max_windows=max_windows,
    )


def _rank_candidate_windows(
    windows: list[list[float]],
    *,
    rows: list[dict],
    query: str,
    max_windows: int,
) -> list[list[float]]:
    if not windows:
        return []
    query_keywords = _query_keywords(query)
    visual_terms = {
        "bag",
        "bottle",
        "container",
        "cup",
        "door",
        "face",
        "food",
        "glass",
        "hand",
        "item",
        "object",
        "person",
        "table",
        "turtle",
        "lobster",
    }

    def row_score(row: dict) -> float:
        caption = str(row.get("caption") or "").lower()
        relevance = str(row.get("query_relevance") or "").lower()
        score = {"high": 3.0, "medium": 2.0, "low": 0.5}.get(relevance, 0.0)
        score += 0.8 * sum(1 for term in query_keywords if term in caption)
        score += 0.35 * sum(1 for term in visual_terms if term in caption)
        score += 1.25 * len(_as_str_list(row.get("matched_requirements")))
        return score

    center_ts = (
        sum(float(row.get("timestamp_s", 0.0)) for row in rows) / max(1, len(rows))
        if rows
        else 0.0
    )

    def window_score(window: list[float]) -> tuple[float, int, float, float]:
        covered = [
            row
            for row in rows
            if float(window[0]) - 1e-3 <= float(row.get("timestamp_s", -1)) <= float(window[1]) + 1e-3
        ]
        near_margin_s = 4.0
        near_boundary = [
            row
            for row in rows
            if row not in covered
            and (
                abs(float(row.get("timestamp_s", -1)) - float(window[0])) <= near_margin_s
                or abs(float(row.get("timestamp_s", -1)) - float(window[1])) <= near_margin_s
            )
        ]
        score = sum(row_score(row) for row in covered)
        score += 0.6 * sum(row_score(row) for row in near_boundary)
        if covered:
            score += 0.25 * len(covered)
        if near_boundary:
            score += 0.1 * len(near_boundary)
        covered_requirements: set[str] = set()
        for row in covered + near_boundary:
            covered_requirements.update(_as_str_list(row.get("matched_requirements")))
        if covered_requirements:
            score += 2.0 * len(covered_requirements)
        duration = max(0.0, float(window[1]) - float(window[0]))
        midpoint = (float(window[0]) + float(window[1])) / 2.0
        # Prefer windows with more useful covered frames; tie-breaker keeps
        # shorter windows first, then windows near the sampled candidate center.
        return (
            score,
            len(covered) + len(near_boundary),
            -duration,
            -abs(midpoint - center_ts),
        )

    ranked: list[list[float]] = []
    for window in sorted(windows, key=window_score, reverse=True):
        if any(_window_overlap_ratio(window, old) >= 0.85 for old in ranked):
            continue
        ranked.append(window)
        if len(ranked) >= max_windows:
            break
    return ranked


def _windows_from_requirement_rows(
    rows: list[dict],
    *,
    start_time: float,
    end_time: float,
    window_s: float,
    max_windows: int,
    requirements: list[dict],
) -> list[list[float]]:
    if not rows or not requirements or max_windows <= 0:
        return []
    requirement_ids = {str(item.get("id")) for item in requirements if item.get("id")}
    useful_rows = [
        row
        for row in rows
        if requirement_ids & set(_as_str_list(row.get("matched_requirements")))
    ]
    if not useful_rows:
        return []

    def make_window(anchor_start: float, anchor_end: float) -> list[float] | None:
        anchor_start = float(anchor_start)
        anchor_end = float(anchor_end)
        if anchor_end < anchor_start:
            anchor_start, anchor_end = anchor_end, anchor_start
        pad = min(5.0, max(2.0, float(window_s) * 0.18))
        start = max(float(start_time), anchor_start - pad)
        end = min(float(end_time), anchor_end + pad)
        if end - start > window_s:
            midpoint = (anchor_start + anchor_end) / 2.0
            half = float(window_s) / 2.0
            start = max(float(start_time), midpoint - half)
            end = min(float(end_time), midpoint + half)
            if end - start < window_s:
                if start <= start_time:
                    end = min(float(end_time), start + window_s)
                elif end >= end_time:
                    start = max(float(start_time), end - window_s)
        if end <= start:
            return None
        return [round(start, 1), round(end, 1)]

    raw_windows: list[list[float]] = []
    for row in useful_rows:
        ts = float(row.get("timestamp_s", start_time))
        window = make_window(ts, ts)
        if window:
            raw_windows.append(window)
    for i, left in enumerate(useful_rows):
        left_ids = set(_as_str_list(left.get("matched_requirements"))) & requirement_ids
        left_ts = float(left.get("timestamp_s", start_time))
        for right in useful_rows[i + 1 :]:
            right_ids = set(_as_str_list(right.get("matched_requirements"))) & requirement_ids
            union_ids = left_ids | right_ids
            if len(union_ids) < 2:
                continue
            right_ts = float(right.get("timestamp_s", start_time))
            if abs(right_ts - left_ts) > float(window_s):
                continue
            window = make_window(min(left_ts, right_ts), max(left_ts, right_ts))
            if window:
                raw_windows.append(window)

    def score_window(window: list[float]) -> tuple[float, float, float, float]:
        covered = [
            row
            for row in rows
            if float(window[0]) - 2.0 <= float(row.get("timestamp_s", -1)) <= float(window[1]) + 2.0
        ]
        matched_ids: set[str] = set()
        relevance = 0.0
        for row in covered:
            matched_ids.update(set(_as_str_list(row.get("matched_requirements"))) & requirement_ids)
            relevance += {"high": 1.0, "medium": 0.6, "low": 0.2}.get(
                str(row.get("query_relevance") or "").lower(),
                0.0,
            )
        duration = max(0.0, window[1] - window[0])
        all_covered = 1.0 if len(matched_ids) >= len(requirement_ids) else 0.0
        return (len(matched_ids), all_covered, relevance, -duration)

    ranked: list[list[float]] = []
    for window in sorted(raw_windows, key=score_window, reverse=True):
        if any(_window_overlap_ratio(window, old) >= 0.85 for old in ranked):
            continue
        ranked.append(window)
        if len(ranked) >= max_windows:
            break
    return ranked


def _window_overlap_ratio(left: list[float], right: list[float]) -> float:
    overlap = min(left[1], right[1]) - max(left[0], right[0])
    if overlap <= 0:
        return 0.0
    denom = max(1e-6, min(left[1] - left[0], right[1] - right[0]))
    return overlap / denom


def _valid_global_windows(value, *, start_time: float, end_time: float) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    windows: list[list[float]] = []
    for item in value:
        raw_window = item
        if isinstance(item, dict):
            raw_window = (
                item.get("t_range")
                or item.get("window")
                or item.get("range")
                or item.get("time_range")
            )
        window = _valid_global_window(raw_window, start_time=start_time, end_time=end_time)
        if window is None:
            continue
        if any(_window_overlap_ratio(window, old) >= 0.85 for old in windows):
            continue
        windows.append(window)
    return windows


def _clamp_window_duration(
    window: list[float],
    *,
    start_time: float,
    end_time: float,
    max_window_s: float,
) -> list[float]:
    if max_window_s <= 0 or window[1] - window[0] <= max_window_s:
        return [round(float(window[0]), 1), round(float(window[1]), 1)]
    midpoint = (float(window[0]) + float(window[1])) / 2.0
    half = float(max_window_s) / 2.0
    start = max(float(start_time), midpoint - half)
    end = min(float(end_time), midpoint + half)
    if end - start < max_window_s:
        if start <= start_time:
            end = min(float(end_time), start + max_window_s)
        elif end >= end_time:
            start = max(float(start_time), end - max_window_s)
    return [round(start, 1), round(end, 1)]


def _frame_rows_to_timestamp_observations(rows: list[dict], *, scene_id: str, window_id: str) -> list[dict]:
    observations: list[dict] = []
    for row in rows:
        matched_requirements = _as_str_list(row.get("matched_requirements"))
        missing_requirements = _as_str_list(row.get("missing_requirements"))
        event_tags = ["qwen_skim_frame"] + [f"matched_{item}" for item in matched_requirements]
        if row.get("qwen_omitted"):
            event_tags.append("qwen_omitted_frame")
        observation = {
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
            "frame_ids": [row["frame_id"]],
        }
        if matched_requirements:
            observation["matched_requirements"] = matched_requirements
        if missing_requirements:
            observation["missing_requirements"] = missing_requirements
            observation["needs_focus"] = "missing " + ", ".join(missing_requirements[:4])
        if row.get("evidence_detail"):
            observation["evidence_detail"] = row.get("evidence_detail")
        observations.append(observation)
    return observations


def _make_scene_summary_from_parsed(
    parsed: dict,
    *,
    start_time: float,
    end_time: float,
    scene_id: str,
    window_id: str,
) -> list[dict]:
    suggest_windows = _valid_global_windows(
        parsed.get("suggest_focus_windows"),
        start_time=start_time,
        end_time=end_time,
    )
    focus_window = _valid_global_window(
        parsed.get("suggest_focus_window"),
        start_time=start_time,
        end_time=end_time,
    )
    if focus_window:
        suggest_windows = [focus_window] + [
            item for item in suggest_windows if _window_overlap_ratio(item, focus_window) < 0.85
        ]
    if suggest_windows:
        parsed["suggest_focus_window"] = suggest_windows[0]
    else:
        parsed["suggest_focus_window"] = None
    parsed["suggest_focus_windows"] = suggest_windows
    t_range = [round(float(start_time), 1), round(float(end_time), 1)]

    return [
        {
            "scene_id": scene_id,
            "window_id": window_id,
            "t_range": t_range,
            "summary": str(parsed.get("observed_event") or "")[:180],
            "possible_evidence": bool(parsed.get("possible_evidence", False)),
            "suggest_focus_windows": suggest_windows,
            "missing_detail": str(parsed.get("missing_detail") or ""),
        }
    ]


def execute_skim_qwen(config: dict, parameters: dict) -> str:
    query = parameters["query"]
    qwen_query = _compact_visual_query(
        query,
        max_chars=int(config.get("skim_qwen_query_max_chars") or 520),
    )
    start_time = float(parameters["start_time"])
    end_time = float(parameters["end_time"])
    mode = parameters.get("mode", "normal")
    query_aware_probe = bool(parameters.get("query_aware_probe", False))
    recovery_skim = _as_bool(parameters.get("recovery_skim"), default=False) or str(mode).lower() in {
        "recovery",
        "requirement_recovery",
    }
    vr = parameters["vr"]
    video_path = parameters.get("video_path", "")
    question = parameters.get("question", "")
    duration = float(parameters.get("duration") or round(len(vr) / vr.get_avg_fps(), 1))

    start_time = max(0.0, min(start_time, duration))
    end_time = max(start_time, min(end_time, duration))
    if end_time <= start_time:
        end_time = min(duration, start_time + 1.0)

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
    scene_id = str((scene_context or {}).get("scene_id") or "skim_qwen")
    window_id = f"{scene_id}_skim_qwen_window" if scene_id != "skim_qwen" else "skim_qwen_window"
    scene_context_text = scene_context_to_prompt_text(scene_context)

    density_enabled = _as_bool(
        config.get("skim_qwen_temporal_density_enabled"),
        default=False,
    )
    window_s = max(0.0, end_time - start_time)
    target_fps = _temporal_density_target_fps(config, window_s=window_s)
    density_frame_count = _skim_frame_budget(
        config,
        window_s=window_s,
        query_aware_probe=query_aware_probe,
        recovery_skim=recovery_skim,
    )
    density_batch_frames = max(
        1,
        int(config.get("skim_qwen_density_batch_frames") or 8),
    )
    if (
        density_enabled
        and not _as_bool(parameters.get("density_micro_batch"), default=False)
        and density_frame_count > density_batch_frames
    ):
        return _execute_density_batched_skim_qwen(
            config,
            parameters,
            start_time=start_time,
            end_time=end_time,
            scene_id=scene_id,
            window_id=window_id,
            query=qwen_query,
        )

    total_frames = len(vr)
    fps = vr.get_avg_fps()
    start_frame = min(int(start_time * fps), total_frames - 1)
    end_frame = min(max(start_frame + 1, int(end_time * fps)), total_frames - 1)
    max_num_frames = max(
        1,
        int(
            parameters.get("density_frame_budget")
            or _skim_frame_budget(
                config,
                window_s=window_s,
                query_aware_probe=query_aware_probe,
                recovery_skim=recovery_skim,
            )
        ),
    )
    num_frames = min(max(1, int(end_frame - start_frame)), max_num_frames)
    fixed_frames = parameters.get("_sampled_frame_indices")
    frame_indices = (np.asarray(fixed_frames, dtype=int) if fixed_frames is not None else
        _sample_skim_frame_indices(parameters, start_time=start_time, end_time=end_time,
            num_frames=num_frames, density_enabled=density_enabled, target_fps=target_fps,
            video_path=video_path, duration=duration, scene_context=scene_context))

    cur_timestamps = [round(float(idx) / fps, 1) for idx in frame_indices]
    frames = vr.get_batch(frame_indices).asnumpy()
    if query_aware_probe:
        short_side = int(
            config.get("query_aware_scene_probe_short_side")
            or config.get("skim_qwen_short_side")
            or 288
        )
    else:
        short_side = int(config.get("skim_qwen_short_side") or 288)
    resized_frames = [_frame_to_resized_array(frame, short_side=short_side) for frame in frames]
    use_single_images = bool(
        config.get(
            "skim_qwen_single_images",
            query_aware_probe and bool(config.get("query_aware_scene_probe_single_images", True)),
        )
    )
    image_style_text = (
        "The images are individual frames from one candidate scene. "
        "Each image is labeled by an Fxx id and timestamp.\n"
        if use_single_images
        else (
            "The images are 2x2 contact sheets sampled across one candidate window. "
            "Cell order per sheet is F01=top-left, F02=top-right, F03=bottom-left, F04=bottom-right.\n"
        )
    )
    frame_caption_words = max(1, int(config.get("skim_qwen_frame_caption_words") or 7))
    relevant_detail_words = max(0, int(config.get("skim_qwen_relevant_detail_words") or 0))
    window_summary_words = max(1, int(config.get("skim_qwen_window_summary_words") or 12))
    caption_char_limit = max(96, frame_caption_words * 12)
    detail_char_limit = max(160, relevant_detail_words * 12) if relevant_detail_words else 160
    caption_only_output = _as_bool(config.get("qwen_caption_only_output"), default=False) or _as_bool(
        config.get("skim_qwen_caption_only_output"),
        default=False,
    )
    detail_schema_suffix = ""
    detail_instruction = ""
    if relevant_detail_words > 0:
        detail_schema_suffix = (
            ", [], [], \"extra query-relevant visual detail or empty\""
        )
        detail_instruction = (
            f"For high-relevance frames, add evidence_detail <= {relevant_detail_words} words "
            "describing why the frame matters; keep it empty for low-relevance frames.\n"
        )
    include_requirement_prompt = _as_bool(
        config.get("skim_qwen_include_requirements_in_prompt"),
        default=False,
    )
    query_requirements = (
        _query_requirements_for_skim(question, qwen_query)
        if include_requirement_prompt
        else []
    )
    query_requirement_text = ""
    if query_requirements and include_requirement_prompt:
        query_requirement_text = (
            "Query requirements to check per frame. Mark matched_requirements with these ids only "
            "when visibly supported in that frame; put missing ids when the frame is relevant but the "
            "requirement is not visible:\n"
            + "\n".join(f"{item['id']}: {item['text']}" for item in query_requirements)
            + "\n"
        )

    max_window_s = float(config.get("skim_qwen_recommended_max_window_s") or 90.0)
    long_window_note = ""
    if end_time - start_time > max_window_s:
        long_window_note = (
            f"This window is longer than the recommended {max_window_s:.0f}s. "
            "Treat the result as very coarse and suggest a narrower focus window if possible.\n"
        )

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "You are skim_qwen, a local visual index tool for a video QA agent.\n"
                "Your task is coarse localization for frame_verify, not final answering and not strong verification.\n"
                f"{image_style_text}"
                "Return a very short tag for each sampled frame/cell so the planner has a timestamped index. "
                "Frame tags must describe visible facts only. Do not copy words from the query unless that "
                "object/action is plainly visible in that sampled frame. If hands, objects, or the target "
                "action are too small or occluded, write an unclear/low tag.\n"
                "Do not decide final colors, counts, action labels, answer options, support labels, or option evidence. Instead, find frames "
                "where the relevant surface/person/object/action may be visible enough for a later frame_verify.\n"
                f"Describe each sampled frame in about {frame_caption_words} words when visual detail is useful; "
                "shorter is fine for blank or irrelevant frames.\n"
                f"{detail_instruction}"
                "Localize candidate evidence for the query. If the sampled frames show a different object/action "
                "than the target, state that mismatch explicitly in observed_event or missing_detail; "
                "possible_evidence=true means this is a useful candidate window for later verification, not a final answer.\n"
                + (
                    "Return plain text lines only. Do not output JSON, markdown, answer options, or support labels.\n\n"
                    if caption_only_output
                    else "Be concise and only return compact JSON.\n\n"
                )
                +
                f"Window range: {start_time:.1f}s - {end_time:.1f}s\n"
                f"Mode: {mode}\n"
                + (local_search_instruction(parameters["local_search_question"], query)
                   if parameters.get("local_search_question") else f"Visual target: {qwen_query}\n")
                +
                f"{query_requirement_text}"
                f"{long_window_note}"
                f"Subtitles in this window:\n{subtitles_str}\n\n"
                + (
                    "Skeleton scene context for this skim window:\n"
                    f"{scene_context_text}\n\n"
                    if scene_context_text
                    else ""
                )
            ),
        }
    ]

    cell_lines: list[str] = []
    cell_counter = 1
    image_count = 0
    if use_single_images:
        for frame, ts in zip(resized_frames, cur_timestamps):
            cell_id = f"F{cell_counter:02d}"
            line = f"{cell_id}: timestamp_s={ts:.1f}"
            cell_lines.append(line)
            cell_counter += 1
            content.append({"type": "text", "text": "Frame:\n" + line})
            content.append({"type": "image_url", "image_url": {"url": _array_to_data_url(frame)}})
            image_count += 1
    else:
        for group_start in range(0, len(resized_frames), 4):
            group_frames = resized_frames[group_start : group_start + 4]
            group_timestamps = cur_timestamps[group_start : group_start + 4]
            sheet = _make_2x2_sheet(group_frames)
            local_lines = []
            for ts in group_timestamps:
                cell_id = f"F{cell_counter:02d}"
                local_lines.append(f"{cell_id}: timestamp_s={ts:.1f}")
                cell_lines.append(f"{cell_id}: timestamp_s={ts:.1f}")
                cell_counter += 1
            content.append({"type": "text", "text": "Contact sheet cells:\n" + "\n".join(local_lines)})
            content.append({"type": "image_url", "image_url": {"url": _array_to_data_url(sheet)}})
            image_count += 1

    allowed_timestamps_str = ", ".join(f"{ts:.1f}" for ts in cur_timestamps)
    if caption_only_output:
        final_instruction = (
            "Valid cells:\n"
            + "\n".join(cell_lines)
            + "\n\n"
            f"There are exactly {len(cur_timestamps)} cells. Return exactly {len(cur_timestamps)} plain-text lines, "
            "one per Fxx cell, in order.\n"
            "Line format: F01: visible caption <= "
            f"{frame_caption_words} words | relevance=low|medium|high | "
            "target_match=matched|possible|not_matched|unknown | "
            "event_match=direct|context_only|not_matched|unknown\n"
            "Use only visible content. If the cell is irrelevant, still describe what is visible and set relevance=low.\n"
            "Relevance policy: high only when the complete visual target and its distinguishing relation/action are directly visible; "
            "medium when only some target entities or scene context are visible; low when absent or unclear.\n"
            "target_match describes whether the queried entity is visibly present. event_match=direct only when the queried action/state is visibly occurring; context_only is related context without the event.\n"
            "Do not output JSON. Do not output markdown. Do not copy the format placeholder as content."
        )
    else:
        final_instruction = (
            "Valid cells:\n"
            + "\n".join(cell_lines)
            + "\n\n"
            f"Allowed timestamps: [{allowed_timestamps_str}]. Use only these timestamps.\n"
            "If skeleton scene context is provided, preserve the provided scene_id and window_id exactly.\n"
            "Return ONLY valid JSON with this exact compact schema. "
            "The frames array must contain one row for every listed Fxx id, in order:\n"
            "{\n"
            f"  \"window_id\": \"{window_id}\",\n"
            f"  \"scene_id\": \"{scene_id}\",\n"
            f"  \"t_range\": [{start_time:.1f}, {end_time:.1f}],\n"
            "  \"frames\": [\n"
            f"    [\"F01\", {cur_timestamps[0]:.1f}, \"describe visible frame\", \"low\"{detail_schema_suffix}]\n"
            "  ],\n"
            f"  \"window_summary\": \"<={window_summary_words} words\",\n"
            "  \"possible_evidence\": false,\n"
            "  \"relevance\": 0.0,\n"
            "  \"suggest_focus_window\": null,\n"
            "  \"suggest_focus_windows\": [],\n"
            "  \"suggest_frame_verify_query\": \"what frame_verify should verify\",\n"
            "  \"missing\": \"\"\n"
            "}\n\n"
            f"Hard limits: exactly {len(cur_timestamps)} frames rows, one per valid Fxx cell; "
            f"frame caption <= {frame_caption_words} words; "
            f"window_summary <= {window_summary_words} words; "
            + (
                f"evidence_detail <= {relevant_detail_words} words when included; "
                if relevant_detail_words > 0
                else ""
            )
            +
            "suggest_focus_window must be [start_s,end_s] or null, never a cell id; "
            "suggest_focus_windows must be a list of absolute/global second windows inside the input range; "
            "use short windows around candidate frames where frame_verify should inspect next; "
            "possible_evidence=true when a useful candidate frame/window is visible, even if details still require frame_verify; "
            "do not repeat the same target phrase for every frame unless every frame clearly shows it; "
            "do not output answer options, support labels, or option evidence; "
            "no markdown; do not copy schema placeholder text."
        )
    content.append({"type": "text", "text": final_instruction})

    observer_config = dict(config)
    observer_config["observer_backend"] = "local_qwen"
    observer_config["local_qwen_tools"] = "skim_qwen"
    if recovery_skim:
        observer_config["local_qwen_max_new_tokens"] = int(
            config.get("skim_qwen_recovery_max_new_tokens")
            or config.get("skim_qwen_max_new_tokens")
            or config.get("local_qwen_max_new_tokens")
            or 768
        )
    elif query_aware_probe:
        observer_config["local_qwen_max_new_tokens"] = int(
            config.get("query_aware_scene_probe_max_new_tokens")
            or config.get("skim_qwen_max_new_tokens")
            or config.get("local_qwen_max_new_tokens")
            or 512
        )
    else:
        observer_config["local_qwen_max_new_tokens"] = int(
            config.get("skim_qwen_max_new_tokens")
            or config.get("local_qwen_max_new_tokens")
            or 256
        )
    observer_config["local_qwen_max_images"] = max(
        int(config.get("local_qwen_max_images") or 24),
        max(1, int(image_count)),
    )
    observer_config["local_qwen_fallback_to_api"] = bool(
        config.get("skim_qwen_fallback_to_api", False)
    )

    t0 = time.time()
    try:
        raw, observer_backend = observe_content(
            observer_config,
            content=content,
            tool_name="skim_qwen",
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
                "window_summary": summary,
                "observed_event": summary,
                "possible_evidence": any(
                    row.get("query_relevance") in {"medium", "high"} and not row.get("qwen_omitted")
                    for row in caption_rows
                ),
                "relevance": 0.7
                if any(row.get("query_relevance") == "high" for row in caption_rows)
                else 0.4
                if any(row.get("query_relevance") == "medium" for row in caption_rows)
                else 0.1,
                "suggest_focus_window": None,
                "suggest_focus_windows": [],
                "suggest_frame_verify_query": qwen_query,
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

    if not parsed.get("scene_id") or parsed.get("scene_id") == "skim_qwen":
        parsed["scene_id"] = scene_id
    if not parsed.get("window_id") or parsed.get("window_id") == "skim_qwen_window":
        parsed["window_id"] = window_id
    parsed.setdefault("t_range", [float(start_time), float(end_time)])
    if not parsed.get("observed_event"):
        parsed["observed_event"] = (
            parsed.get("window_summary")
            or parsed.get("summary")
            or parsed.get("observed_event")
            or ""
        )
    if not parsed.get("missing_detail"):
        parsed["missing_detail"] = parsed.get("missing") or parsed.get("missing_detail") or ""
    if query_requirements:
        parsed["query_requirements"] = query_requirements
    candidate_window_value = (
        config.get("skim_qwen_recovery_candidate_window_s")
        if recovery_skim
        else config.get("skim_qwen_candidate_window_s")
    )
    candidate_window_s = float(candidate_window_value or 18.0)
    max_suggest_value = (
        config.get("skim_qwen_recovery_max_candidate_windows")
        if recovery_skim
        else config.get("skim_qwen_max_candidate_windows")
    )
    max_suggest_windows = int(max_suggest_value or 4)
    valid_suggested_windows = _valid_global_windows(
        parsed.get("best_joint_windows"),
        start_time=start_time,
        end_time=end_time,
    )
    valid_suggested_windows = [
        _clamp_window_duration(
            window,
            start_time=start_time,
            end_time=end_time,
            max_window_s=candidate_window_s,
        )
        for window in valid_suggested_windows
    ]
    raw_suggested_windows = _valid_global_windows(
        parsed.get("suggest_focus_windows"),
        start_time=start_time,
        end_time=end_time,
    )
    raw_suggested_windows = [
        _clamp_window_duration(
            window,
            start_time=start_time,
            end_time=end_time,
            max_window_s=candidate_window_s,
        )
        for window in raw_suggested_windows
    ]
    for window in raw_suggested_windows:
        if any(_window_overlap_ratio(window, old) >= 0.85 for old in valid_suggested_windows):
            continue
        valid_suggested_windows.append(window)
    suggested_window = parsed.get("suggest_focus_window")
    valid_suggested_window = _valid_global_window(
        suggested_window,
        start_time=start_time,
        end_time=end_time,
    )
    if valid_suggested_window is not None:
        valid_suggested_window = _clamp_window_duration(
            valid_suggested_window,
            start_time=start_time,
            end_time=end_time,
            max_window_s=candidate_window_s,
        )
    if suggested_window is not None and valid_suggested_window is None:
        parsed["invalid_suggest_focus_window"] = suggested_window
    if valid_suggested_window is not None:
        valid_suggested_windows = [valid_suggested_window] + [
            item
            for item in valid_suggested_windows
            if _window_overlap_ratio(item, valid_suggested_window) < 0.85
        ]
    frame_rows = _normalize_frame_rows(
        parsed,
        allowed_timestamps=cur_timestamps,
        scene_id=scene_id,
        caption_char_limit=caption_char_limit,
        detail_char_limit=detail_char_limit,
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
    if _as_bool(config.get("skim_qwen_auto_focus_windows"), default=True):
        requirement_windows = _windows_from_requirement_rows(
            frame_rows,
            start_time=start_time,
            end_time=end_time,
            window_s=candidate_window_s,
            max_windows=max_suggest_windows,
            requirements=query_requirements,
        )
        for window in requirement_windows:
            if any(_window_overlap_ratio(window, old) >= 0.85 for old in valid_suggested_windows):
                continue
            valid_suggested_windows.append(window)
        auto_windows = _windows_from_frame_rows(
            frame_rows,
            query=query,
            start_time=start_time,
            end_time=end_time,
            window_s=candidate_window_s,
            max_windows=max_suggest_windows,
        )
        for window in auto_windows:
            if any(_window_overlap_ratio(window, old) >= 0.85 for old in valid_suggested_windows):
                continue
            valid_suggested_windows.append(window)
    valid_suggested_windows = valid_suggested_windows[:max(1, max_suggest_windows)]
    parsed["suggest_focus_windows"] = valid_suggested_windows
    parsed["suggest_focus_window"] = valid_suggested_windows[0] if valid_suggested_windows else None
    if valid_suggested_windows and parsed.get("possible_evidence") is not True:
        parsed["possible_evidence"] = True
    if valid_suggested_windows:
        valid_suggested_windows = _rank_candidate_windows(
            valid_suggested_windows,
            rows=frame_rows,
            query=query,
            max_windows=max(1, max_suggest_windows),
        )
        parsed["suggest_focus_windows"] = valid_suggested_windows
        parsed["suggest_focus_window"] = valid_suggested_windows[0]
    if frame_rows:
        parsed["frames"] = [
            {
                "frame_id": row["frame_id"],
                "timestamp_s": row["timestamp_s"],
                "caption": row["caption"],
                "query_relevance": row["query_relevance"],
                "matched_requirements": row.get("matched_requirements") or [],
                "missing_requirements": row.get("missing_requirements") or [],
                "evidence_detail": row.get("evidence_detail") or "",
            }
            for row in frame_rows
        ]
        parsed["frame_captions"] = frame_rows
        parsed["timestamp_observations"] = _frame_rows_to_timestamp_observations(
            frame_rows,
            scene_id=scene_id,
            window_id=window_id,
        )
    if omitted_ids:
        parsed["qwen_omitted_frame_ids"] = omitted_ids
    parsed["observed_frame_count"] = len(frame_rows) - len(omitted_ids)
    parsed["observer_status"] = "partial" if omitted_ids else "complete"
    for item in parsed.get("timestamp_observations") or []:
        if isinstance(item, dict) and (not item.get("scene_id") or item.get("scene_id") == "skim_qwen"):
            item["scene_id"] = scene_id
    snap_timestamp_observations(parsed, allowed_timestamps=cur_timestamps)
    parsed["scene_summaries"] = _make_scene_summary_from_parsed(
        parsed,
        start_time=start_time,
        end_time=end_time,
        scene_id=scene_id,
        window_id=window_id,
    )
    if scene_context:
        parsed.setdefault("matched_skeleton_scene", scene_id)
    parsed["observer_backend"] = observer_backend
    parsed["observer_wall_s"] = round(observer_wall, 3)
    parsed["num_frames"] = len(resized_frames)
    if density_enabled:
        parsed["sampling_strategy"] = "adaptive_temporal_density"
        parsed["sampling_density_band"] = _temporal_density_band(
            config,
            window_s=window_s,
        )
        parsed["target_fps"] = target_fps
        parsed["requested_frame_count"] = max(
            1,
            int(math.ceil(window_s * target_fps)),
        )
        parsed["frame_budget"] = max_num_frames
        parsed["frame_budget_capped"] = max_num_frames < parsed["requested_frame_count"]
        parsed["micro_batch_frame_limit"] = density_batch_frames
        parsed["effective_fps"] = round(len(cur_timestamps) / max(1e-6, window_s), 4)
        parsed["sampled_timestamps"] = cur_timestamps
        parsed["observer_batch_count"] = 1
        parsed["compact_render"] = True
        parsed.pop("frames", None)
        parsed.pop("frame_captions", None)
    parsed["parse_ok"] = True
    return format_v10_observation(parsed, fallback_text=raw)
