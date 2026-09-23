from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from videoseek.observer import call_local_qwen_parts


def _question_stem(question: str) -> str:
    lines: list[str] = []
    for raw_line in str(question or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^\(?[A-Z]\)?[\s.:)]", line, re.I):
            break
        if line.lower().startswith("please directly answer"):
            break
        lines.append(line.removeprefix("Question:").strip())
    return " ".join(lines)[:600]


def build_answer_blind_grounding_prompt(
    *, query: str, question: str, high_recall: bool = False
) -> str:
    proposal_policy = (
        "Return a tight box for the most plausible visible referent matching the "
        "observable entity nouns and spatial relation in the request. Exact identity, "
        "attribute, action, or answer may remain uncertain; do not reject a plausible "
        "region merely because those details cannot yet be confirmed. Use VISIBLE no "
        "only when no plausible visual referent is present.\n"
        if high_recall
        else "Do not guess when the relevant target is absent.\n"
    )
    return (
        "You are a spatial proposal generator for one video frame.\n"
        "Locate only the visible subject, object, or interacting participants whose "
        "unknown property, category, action, or relation must be inspected.\n"
        "Any proposed colors, categories, action labels, numbers, option letters, or "
        "conclusions in the request are unknown candidate values. Ignore them as "
        "location cues and do not decide the answer.\n"
        "For an interaction, use one union box covering the subject, object, and "
        "interaction area. "
        + proposal_policy
        + "\n"
        f"Original question stem: {_question_stem(question)}\n"
        f"Verification request: {str(query or '').strip()[:1000]}\n\n"
        "Return exactly four plain-text lines:\n"
        "VISIBLE yes or no\n"
        "BOX x1 y1 x2 y2\n"
        "LABEL target\n"
        "CONFIDENCE an integer from 0 to 100\n"
        "Coordinates must be integers normalized to 0..1000. If absent, write BOX NONE."
    )


def parse_grounding_box(raw: str) -> dict[str, Any]:
    lines = [line.strip() for line in str(raw or "").splitlines() if line.strip()]
    compact = " ".join(" ".join(lines).split())
    visible_match = re.search(r"VISIBLE\s*[:=]?\s*(yes|no)", compact, re.I)
    visible = None if not visible_match else visible_match.group(1).lower() == "yes"
    # Qwen occasionally preserves the requested four-line order but drops the
    # field names. Accept that bounded variant without searching arbitrary text.
    if visible is None and lines and re.fullmatch(r"yes|no", lines[0], re.I):
        visible = lines[0].lower() == "yes"
    box_none = bool(
        re.search(r"(?:BOX|BBOX(?:_0_1000)?)\s*[:=]?\s*NONE", compact, re.I)
    )
    match = re.search(
        r"(?:BOX|BBOX(?:_0_1000)?)\s*[:=]?\s*[\[<(]?\s*"
        r"(-?\d+(?:\.\d+)?)\s*[, ]+\s*(-?\d+(?:\.\d+)?)\s*[, ]+\s*"
        r"(-?\d+(?:\.\d+)?)\s*[, ]+\s*(-?\d+(?:\.\d+)?)",
        compact,
        re.I,
    )
    positional_match = None
    if not match and len(lines) >= 2:
        if re.fullmatch(r"(?:BOX\s*)?NONE", lines[1], re.I):
            box_none = True
        else:
            positional_match = re.fullmatch(
                r"[\[<(]?\s*(-?\d+(?:\.\d+)?)\s*[, ]+\s*"
                r"(-?\d+(?:\.\d+)?)\s*[, ]+\s*(-?\d+(?:\.\d+)?)\s*[, ]+\s*"
                r"(-?\d+(?:\.\d+)?)\s*[\])>]?$",
                lines[1],
            )
    coordinate_match = match or positional_match
    raw_box = (
        None
        if box_none or not coordinate_match
        else [float(coordinate_match.group(i)) for i in range(1, 5)]
    )
    box = None
    clamped = False
    if raw_box:
        box = [max(0.0, min(1000.0, value)) for value in raw_box]
        clamped = box != raw_box
        if box[2] <= box[0] or box[3] <= box[1]:
            box = None
    valid = bool(visible is True and box is not None)
    return {
        "visible": visible,
        "box": box if valid else None,
        "box_valid": valid,
        "coordinates_clamped": clamped,
        "parse_ok": visible is not None and (box_none or coordinate_match is not None),
    }


def _jpeg_bytes(image: Image.Image, *, quality: int = 92) -> bytes:
    output = BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def _data_url(image_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")


def _resize_short_side(image: Image.Image, short_side: int) -> Image.Image:
    width, height = image.size
    current = min(width, height)
    if short_side <= 0 or current <= short_side:
        return image.copy()
    scale = short_side / float(current)
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )


def _resize_long_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    current = max(width, height)
    if max_side <= 0 or current <= max_side:
        return image.copy()
    scale = max_side / float(current)
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )


def _normalized_to_pixels(
    box: list[float],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = size
    return (
        max(0, min(width - 1, round(box[0] * width / 1000.0))),
        max(0, min(height - 1, round(box[1] * height / 1000.0))),
        max(1, min(width, round(box[2] * width / 1000.0))),
        max(1, min(height, round(box[3] * height / 1000.0))),
    )


def _expand_box(
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    *,
    margin: float,
    min_side_ratio: float,
) -> tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    box_width = max(1.0, (x2 - x1) * (1.0 + 2.0 * max(0.0, margin)))
    box_height = max(1.0, (y2 - y1) * (1.0 + 2.0 * max(0.0, margin)))
    box_width = max(box_width, width * max(0.0, min_side_ratio))
    box_height = max(box_height, height * max(0.0, min_side_ratio))
    max_aspect = 4.0
    if box_width / box_height > max_aspect:
        box_height = box_width / max_aspect
    elif box_height / box_width > max_aspect:
        box_width = box_height / max_aspect
    left = max(0, math.floor(cx - box_width / 2.0))
    top = max(0, math.floor(cy - box_height / 2.0))
    right = min(width, math.ceil(cx + box_width / 2.0))
    bottom = min(height, math.ceil(cy + box_height / 2.0))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _evenly_limit(values: list[int], limit: int) -> list[int]:
    values = sorted(set(values))
    if len(values) <= limit:
        return values
    indexes = np.linspace(0, len(values) - 1, limit).round().astype(int)
    return [values[int(index)] for index in sorted(set(indexes.tolist()))]


def select_grounding_positions(
    *,
    frame_indices: list[int],
    mandatory_anchor_indices: set[int],
    candidate_anchor_indices: dict[int, list[str]],
    max_anchors: int,
    candidate_balanced: bool = False,
    max_per_candidate: int | None = None,
) -> list[int]:
    if not frame_indices or max_anchors <= 0:
        return []
    index_to_position = {
        int(frame_index): position for position, frame_index in enumerate(frame_indices)
    }
    priority = [
        index_to_position[index]
        for index in sorted(mandatory_anchor_indices)
        if index in index_to_position
    ]
    if candidate_balanced:
        positions_by_candidate: dict[str, list[int]] = {}
        for frame_index in sorted(candidate_anchor_indices):
            if frame_index not in index_to_position:
                continue
            position = index_to_position[frame_index]
            for candidate_id in candidate_anchor_indices[frame_index]:
                positions_by_candidate.setdefault(str(candidate_id), []).append(position)
        if max_per_candidate is not None:
            per_candidate_limit = max(1, int(max_per_candidate))
            positions_by_candidate = {
                candidate_id: _evenly_limit(
                    sorted(set(rows)),
                    per_candidate_limit,
                )
                for candidate_id, rows in positions_by_candidate.items()
            }
        for candidate_id in sorted(positions_by_candidate):
            rows = sorted(set(positions_by_candidate[candidate_id]))
            if rows:
                priority.append(rows[len(rows) // 2])
        balanced = list(dict.fromkeys(priority))
        if len(balanced) >= max_anchors:
            return sorted(balanced[:max_anchors])
        extras_by_candidate: dict[str, list[int]] = {}
        for candidate_id in sorted(positions_by_candidate):
            rows = sorted(set(positions_by_candidate[candidate_id]))
            center = rows[len(rows) // 2] if rows else -1
            extras_by_candidate[candidate_id] = sorted(
                (position for position in rows if position not in balanced),
                key=lambda position: (abs(position - center), position),
            )
        # Allocate additional anchors round-robin so one candidate cannot use
        # the whole grounding budget before the others receive temporal backup.
        while len(balanced) < max_anchors:
            added = False
            for candidate_id in sorted(extras_by_candidate):
                rows = extras_by_candidate[candidate_id]
                if not rows:
                    continue
                balanced.append(rows.pop(0))
                added = True
                if len(balanced) >= max_anchors:
                    break
            if not added:
                break
        if balanced:
            return sorted(set(balanced))
    else:
        priority.extend(
            index_to_position[index]
            for index in sorted(candidate_anchor_indices)
            if index in index_to_position
        )
    if not priority:
        priority = [len(frame_indices) // 2]
    return _evenly_limit(priority, max_anchors)


def _cache_root(output_dir: str | None, config: dict) -> Path:
    root = Path(
        config.get("grounded_frame_verify_cache_dir")
        or output_dir
        or config.get("output_dir")
        or "./output"
    )
    cache = root / "_grounded_frame_verify"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def prepare_grounded_frames(
    config: dict,
    *,
    frames: np.ndarray,
    frame_indices: list[int],
    timestamps: list[float],
    mandatory_anchor_indices: set[int],
    candidate_anchor_indices: dict[int, list[str]],
    query: str,
    question: str,
    output_dir: str | None,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    max_anchors = max(0, int(config.get("grounded_frame_verify_max_anchors") or 1))
    positions = select_grounding_positions(
        frame_indices=frame_indices,
        mandatory_anchor_indices=mandatory_anchor_indices,
        candidate_anchor_indices=candidate_anchor_indices,
        max_anchors=max_anchors,
        candidate_balanced=bool(
            config.get("grounded_candidate_binding_enabled", False)
        ),
        max_per_candidate=(
            int(config.get("grounded_verify_packet_anchors_per_candidate") or 2)
            if config.get("grounded_verify_packet_enabled", False)
            else None
        ),
    )
    crop_threshold = float(config.get("grounded_frame_verify_crop_area_threshold") or 0.5)
    margin = float(config.get("grounded_frame_verify_padding_ratio") or 0.18)
    min_side_ratio = float(config.get("grounded_frame_verify_min_crop_side_ratio") or 0.12)
    context_short_side = int(config.get("grounded_frame_verify_context_short_side") or 320)
    packet_enabled = bool(config.get("grounded_verify_packet_enabled", False))
    crop_max_side = int(config.get("grounded_verify_packet_crop_max_side") or 768)
    max_new_tokens = int(config.get("grounded_frame_verify_max_new_tokens") or 96)
    cache_enabled = bool(config.get("grounded_frame_verify_cache_enabled", True))
    cache_root = _cache_root(output_dir, config)
    high_recall = bool(config.get("grounded_high_recall_proposal_enabled", False))
    prompt = build_answer_blind_grounding_prompt(
        query=query,
        question=question,
        high_recall=high_recall,
    )
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]

    replacements: dict[int, list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for position in positions:
        timestamp = float(timestamps[position])
        image = Image.fromarray(frames[position]).convert("RGB")
        full_bytes = _jpeg_bytes(image, quality=92)
        digest = hashlib.sha1(full_bytes + prompt.encode("utf-8")).hexdigest()
        image_path = cache_root / f"{digest}.jpg"
        result_path = cache_root / f"{digest}.json"
        if not image_path.exists():
            image_path.write_bytes(full_bytes)

        cached = _load_json(result_path) if cache_enabled else None
        grounding_wall_s = 0.0
        error = ""
        cache_hit = cached is not None
        if cached is None:
            started = time.perf_counter()
            try:
                raw = call_local_qwen_parts(
                    config,
                    parts=[
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": prompt},
                    ],
                    output_dir=output_dir,
                    max_new_tokens=max_new_tokens,
                )
                (cache_root / f"{digest}.raw.txt").write_text(raw, encoding="utf-8")
                parsed = parse_grounding_box(raw)
            except Exception as exc:
                raw = ""
                parsed = {
                    "visible": None,
                    "box": None,
                    "box_valid": False,
                    "coordinates_clamped": False,
                    "parse_ok": False,
                }
                error = f"{type(exc).__name__}: {exc}"[:500]
            grounding_wall_s = time.perf_counter() - started
            cached = {
                **parsed,
                "prompt_hash": prompt_hash,
                "raw_sha1": hashlib.sha1(raw.encode("utf-8")).hexdigest() if raw else "",
            }
            if cache_enabled and not error and parsed.get("parse_ok"):
                result_path.write_text(
                    json.dumps(cached, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        parsed = dict(cached)
        packet_mode = "full_fallback"
        crop_area_ratio = None
        expanded_box = None
        if parsed.get("box_valid") and isinstance(parsed.get("box"), list):
            raw_box = _normalized_to_pixels(parsed["box"], image.size)
            expanded_box = _expand_box(
                raw_box,
                image.size,
                margin=margin,
                min_side_ratio=min_side_ratio,
            )
            crop_area = (expanded_box[2] - expanded_box[0]) * (expanded_box[3] - expanded_box[1])
            crop_area_ratio = crop_area / float(image.width * image.height)
            if crop_area_ratio <= crop_threshold:
                context = _resize_short_side(image, context_short_side)
                crop = _resize_long_side(image.crop(expanded_box), crop_max_side)
                crop_item = {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url(_jpeg_bytes(crop, quality=95)),
                        "detail": "high",
                    },
                }
                if packet_enabled:
                    replacements[position] = [
                        {
                            "type": "text",
                            "text": (
                                "Untrusted high-resolution crop from the anchor timestamp; "
                                "validate it against this candidate's temporal strip:"
                            ),
                        },
                        crop_item,
                    ]
                    packet_mode = "crop_only"
                else:
                    replacements[position] = [
                        {"type": "text", "text": "Low-resolution full-frame context for this timestamp:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(_jpeg_bytes(context, quality=86)), "detail": "low"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Untrusted high-resolution crop proposal from the same timestamp; "
                                "validate its identity against the global context:"
                            ),
                        },
                        crop_item,
                    ]
                    packet_mode = "context_crop"
                crop_path = cache_root / f"{digest}.crop.jpg"
                if not crop_path.exists():
                    crop.save(crop_path, format="JPEG", quality=95)

        fallback_reason = ""
        if packet_mode == "full_fallback":
            fallback_reason = error or (
                "crop_too_large" if parsed.get("box_valid") else "target_not_grounded"
            )
        records.append(
            {
                "timestamp_s": round(timestamp, 3),
                "frame_index": int(frame_indices[position]),
                "position": int(position),
                "candidate_ids": list(
                    candidate_anchor_indices.get(int(frame_indices[position]), [])
                ),
                "selected": True,
                "packet_mode": packet_mode,
                "box_valid": bool(parsed.get("box_valid")),
                "parse_ok": bool(parsed.get("parse_ok")),
                "coordinates_clamped": bool(parsed.get("coordinates_clamped")),
                "bbox_norm_0_1000": parsed.get("box"),
                "expanded_box_px": list(expanded_box) if expanded_box else None,
                "crop_area_ratio": crop_area_ratio,
                "cache_hit": cache_hit,
                "grounding_wall_s": round(grounding_wall_s, 4),
                "fallback_reason": fallback_reason,
            }
        )

    audit = {
        "enabled": True,
        "protocol": (
            "answer_blind_high_recall_bbox_v2"
            if high_recall
            else "answer_blind_bbox_only_v1"
        ),
        "verify_packet_enabled": packet_enabled,
        "prompt_hash": prompt_hash,
        "requested_max_anchors": max_anchors,
        "selected_anchor_count": len(positions),
        "context_crop_count": sum(row["packet_mode"] == "context_crop" for row in records),
        "crop_only_count": sum(row["packet_mode"] == "crop_only" for row in records),
        "crop_count": sum(
            row["packet_mode"] in {"context_crop", "crop_only"} for row in records
        ),
        "full_fallback_count": sum(row["packet_mode"] == "full_fallback" for row in records),
        "grounding_wall_s": round(sum(row["grounding_wall_s"] for row in records), 4),
        "records": records,
    }
    with (cache_root / "grounding_trace.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    return replacements, audit
