from __future__ import annotations

import base64
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from videoseek.codec import (
    codec_aware_timestamps,
    keyframe_only_timestamps,
    probe_interframe_packet_sizes,
    timestamps_to_frame_indices,
)
from videoseek.core.memory import extract_v10_payload
from videoseek.observer import observe_content
from videoseek.skeleton import load_or_build_skeleton
from videoseek.tools.v10_format import format_v10_observation


RemoteOverviewRunner = Callable[[dict, dict], str]

_CAPTION_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?F(?P<index>\d{1,3})\s*[:：|\-]\s*(?P<caption>.+?)\s*$",
    re.IGNORECASE,
)
_QUERY_STOP_WORDS = {
    "about",
    "after",
    "again",
    "and",
    "among",
    "before",
    "being",
    "between",
    "could",
    "does",
    "during",
    "from",
    "her",
    "hers",
    "him",
    "his",
    "have",
    "how",
    "its",
    "into",
    "the",
    "many",
    "most",
    "much",
    "person",
    "people",
    "question",
    "should",
    "shown",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "video",
    "what",
    "when",
    "where",
    "which",
    "who",
    "while",
    "with",
    "would",
}
_NON_DISCRIMINATIVE_CHOICE_TERMS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
}
_NON_DISCRIMINATIVE_QUERY_TERMS = {
    "action",
    "activity",
    "animal",
    "appear",
    "are",
    "being",
    "color",
    "event",
    "happen",
    "have",
    "is",
    "item",
    "keep",
    "kept",
    "kind",
    "object",
    "scene",
    "thing",
    "time",
    "total",
    "type",
    "unusual",
    "visible",
    "was",
    "were",
}
_NON_DISCRIMINATIVE_PROTOTYPE_TERMS = {
    "appear",
    "are",
    "confirm",
    "group",
    "indoor",
    "look",
    "move",
    "open",
    "perform",
    "rather",
    "scen",
    "show",
    "similar",
    "specific",
    "verify",
    "visible",
}


def planner_visible_overview_output(output: str) -> str:
    """Hide reserve-only local candidates from the immediate planner turn.

    The full payload is merged into observation memory before this function is
    used. The planner should initially see the same remote overview as V76;
    local candidates are exposed later only through the coverage-reserve view.
    """

    payload = extract_v10_payload(output)
    if not payload:
        return output
    dual_path = payload.get("dual_path_overview") or {}
    if dual_path.get("integration_mode") != "reserve":
        return output
    visible = deepcopy(payload)
    visible.pop("dual_path_candidates", None)
    visible.pop("dual_path_overview", None)
    visible.pop("internal_qwen_calls", None)
    return format_v10_observation(visible, fallback_text=output)


def _overview_frame_budget(config: dict[str, Any]) -> int:
    budget = int(
        config.get("overview_num_frames")
        or int(config.get("frame_sampling_factor") or 4)
        * int(config.get("overview_base") or 16)
    )
    return max(4, int(math.ceil(budget / 8.0) * 8))


def _local_frame_budget(config: dict[str, Any], *, duration: float) -> int:
    explicit = int(config.get("dual_path_local_frames") or 0)
    if explicit > 0:
        return explicit
    minimum = max(8, int(config.get("dual_path_local_min_frames") or 32))
    maximum = max(minimum, int(config.get("dual_path_local_max_frames") or 64))
    target_interval = max(
        1.0,
        float(config.get("dual_path_local_target_interval_s") or 12.0),
    )
    adaptive = int(math.ceil(max(0.0, float(duration)) / target_interval))
    adaptive = int(math.ceil(max(minimum, adaptive) / 8.0) * 8)
    return min(maximum, adaptive)


def _remote_overview_timestamps(
    config: dict[str, Any], *, video_path: str, duration: float
) -> list[float]:
    budget = _overview_frame_budget(config)
    mode = str(config.get("overview_sampling_mode") or "hybrid").lower()
    if mode == "iframes":
        timestamps = keyframe_only_timestamps(
            video_path=video_path,
            start_time=0.0,
            end_time=duration,
            num_frames=budget,
        )
        group_size = int(config.get("overview_contact_sheet_group_size") or 4)
        if len(timestamps) >= group_size:
            return [round(float(item), 3) for item in timestamps[:budget]]
    return [
        round(float(item), 3)
        for item in codec_aware_timestamps(
            video_path=video_path,
            duration_s=duration,
            start_time=0.0,
            end_time=duration,
            num_frames=budget,
        )
    ]


def _scene_id_for_timestamp(skeleton: dict[str, Any], timestamp: float) -> str:
    scenes = [item for item in skeleton.get("scenes") or [] if isinstance(item, dict)]
    for scene in scenes:
        span = scene.get("t_range") or []
        if len(span) != 2:
            continue
        try:
            if float(span[0]) <= timestamp <= float(span[1]):
                return str(scene.get("scene_id") or "overview")
        except (TypeError, ValueError):
            continue
    return "overview"


def _candidate(
    timestamp: float,
    *,
    source: str,
    score: float,
    skeleton: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp_s": round(float(timestamp), 3),
        "source": source,
        "score": round(float(score), 4),
        "scene_id": _scene_id_for_timestamp(skeleton, float(timestamp)),
    }


def _far_enough(
    timestamp: float,
    *,
    remote_timestamps: list[float],
    selected: list[dict[str, Any]],
    remote_gap_s: float,
    local_gap_s: float,
) -> bool:
    if any(abs(timestamp - old) < remote_gap_s for old in remote_timestamps):
        return False
    return all(
        abs(timestamp - float(item["timestamp_s"])) >= local_gap_s
        for item in selected
    )


def _diverse_take(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    selected: list[dict[str, Any]],
    remote_timestamps: list[float],
    remote_gap_s: float,
    local_gap_s: float,
) -> None:
    if limit <= 0:
        return
    queues: dict[str, list[dict[str, Any]]] = {}
    for item in sorted(
        candidates,
        key=lambda value: (-float(value.get("score") or 0.0), float(value["timestamp_s"])),
    ):
        queues.setdefault(str(item.get("scene_id") or "overview"), []).append(item)

    scene_order = sorted(
        queues,
        key=lambda scene_id: -float(queues[scene_id][0].get("score") or 0.0),
    )
    added = 0
    while scene_order and added < limit:
        next_order: list[str] = []
        for scene_id in scene_order:
            queue = queues[scene_id]
            while queue:
                item = queue.pop(0)
                timestamp = float(item["timestamp_s"])
                if _far_enough(
                    timestamp,
                    remote_timestamps=remote_timestamps,
                    selected=selected,
                    remote_gap_s=remote_gap_s,
                    local_gap_s=local_gap_s,
                ):
                    selected.append(item)
                    added += 1
                    break
            if queue:
                next_order.append(scene_id)
            if added >= limit:
                break
        scene_order = next_order


def select_supplementary_frames(
    *,
    remote_timestamps: list[float],
    packet_samples: list[tuple[float, int]],
    skeleton: dict[str, Any],
    duration: float,
    limit: int,
    remote_gap_s: float = 0.5,
    local_gap_s: float = 0.45,
) -> list[dict[str, Any]]:
    """Select complementary P/B dynamics, scene boundaries, and coverage gaps."""

    limit = max(0, int(limit))
    if limit == 0 or duration <= 0:
        return []
    remote = sorted(
        set(max(0.0, min(float(value), duration)) for value in remote_timestamps)
    )

    # Raw global packet-size ranking tends to concentrate on one highly dynamic
    # shot. Select one local maximum per temporal bin first, then rank those
    # maxima by their within-video packet cost.
    valid_packets = [
        (float(timestamp), int(size))
        for timestamp, size in packet_samples
        if 0.0 <= float(timestamp) <= duration and int(size) > 0
    ]
    bin_count = max(8, min(max(8, limit * 2), int(math.ceil(duration / 10.0))))
    bin_width = duration / max(1, bin_count)
    binned_peaks: list[tuple[float, int]] = []
    for bin_index in range(bin_count):
        start = bin_index * bin_width
        end = duration if bin_index + 1 == bin_count else (bin_index + 1) * bin_width
        values = [
            item
            for item in valid_packets
            if start <= item[0] <= end
        ]
        if values:
            binned_peaks.append(max(values, key=lambda item: item[1]))
    max_packet_size = max((size for _, size in binned_peaks), default=1)
    packet_candidates = [
        _candidate(
            timestamp,
            source="interframe_bitcost_peak",
            score=0.62 + 0.38 * size / max(1, max_packet_size),
            skeleton=skeleton,
        )
        for timestamp, size in binned_peaks
    ]

    boundary_candidates: list[dict[str, Any]] = []
    scenes = [item for item in skeleton.get("scenes") or [] if isinstance(item, dict)]
    for scene in scenes[1:]:
        span = scene.get("t_range") or []
        if len(span) != 2:
            continue
        try:
            boundary = float(span[0])
        except (TypeError, ValueError):
            continue
        for offset, score in ((-0.6, 0.82), (0.6, 0.84)):
            timestamp = boundary + offset
            if 0.0 <= timestamp <= duration:
                boundary_candidates.append(
                    _candidate(
                        timestamp,
                        source="scene_boundary",
                        score=score,
                        skeleton=skeleton,
                    )
                )

    coverage_candidates: list[dict[str, Any]] = []
    anchors = sorted(set([0.0, *remote, duration]))
    mean_gap = duration / max(1, len(remote) - 1)
    for start, end in zip(anchors, anchors[1:]):
        gap = end - start
        if gap <= max(1.0, remote_gap_s * 2.0):
            continue
        fractions = [0.5]
        if gap > max(12.0, mean_gap * 1.5):
            fractions = [1.0 / 3.0, 2.0 / 3.0]
        for fraction in fractions:
            coverage_candidates.append(
                _candidate(
                    start + gap * fraction,
                    source="coverage_gap",
                    score=min(0.94, 0.66 + gap / max(20.0, duration)),
                    skeleton=skeleton,
                )
            )

    selected: list[dict[str, Any]] = []
    packet_quota = int(round(limit * 0.6))
    boundary_quota = int(round(limit * 0.2))
    coverage_quota = max(0, limit - packet_quota - boundary_quota)
    for candidates, quota in (
        (packet_candidates, packet_quota),
        (boundary_candidates, boundary_quota),
        (coverage_candidates, coverage_quota),
    ):
        _diverse_take(
            candidates,
            limit=quota,
            selected=selected,
            remote_timestamps=remote,
            remote_gap_s=remote_gap_s,
            local_gap_s=local_gap_s,
        )

    if len(selected) < limit:
        fallback = packet_candidates + boundary_candidates + coverage_candidates
        fallback.extend(
            _candidate(
                timestamp,
                source="uniform_rescue",
                score=0.55,
                skeleton=skeleton,
            )
            for timestamp in np.linspace(0.0, duration, limit * 3 + 2)[1:-1]
        )
        _diverse_take(
            fallback,
            limit=limit - len(selected),
            selected=selected,
            remote_timestamps=remote,
            remote_gap_s=remote_gap_s,
            local_gap_s=local_gap_s,
        )

    return sorted(selected[:limit], key=lambda item: float(item["timestamp_s"]))


def _resize_frame(frame: np.ndarray, *, short_side: int) -> np.ndarray:
    image = Image.fromarray(frame)
    width, height = image.size
    scale = max(1, short_side) / max(1, min(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(image.resize(size, Image.Resampling.BICUBIC))


def _frame_data_url(frame: np.ndarray) -> str:
    output = BytesIO()
    Image.fromarray(frame).save(output, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _decode_supplementary_frames(
    parameters: dict[str, Any],
    specs: list[dict[str, Any]],
    *,
    short_side: int,
) -> list[dict[str, Any]]:
    vr = parameters["vr"]
    fps = float(vr.get_avg_fps())
    indices = timestamps_to_frame_indices(
        timestamps=[float(item["timestamp_s"]) for item in specs],
        fps=fps,
        total_frames=len(vr),
    )
    if not len(indices):
        return []
    arrays = vr.get_batch(indices).asnumpy()
    decoded: list[dict[str, Any]] = []
    for index, frame in zip(indices.tolist(), arrays):
        timestamp = round(float(index) / max(1e-6, fps), 3)
        source = min(
            specs,
            key=lambda item: abs(float(item["timestamp_s"]) - timestamp),
        )
        decoded.append(
            {
                **source,
                "timestamp_s": timestamp,
                "frame": _resize_frame(frame, short_side=short_side),
            }
        )
    return decoded


def _question_only(text: str) -> str:
    """Remove multiple-choice options from the local routing query."""

    lines: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if re.match(r"^\([A-Za-z]\)\s+", line):
            break
        if line.lower().startswith("please directly answer"):
            break
        if line:
            lines.append(line)
    return " ".join(lines).strip()


def _choice_terms(text: str) -> set[str]:
    """Extract answer-choice concepts for local retrieval, not captioning."""

    choices = [
        match.group("choice")
        for raw in str(text or "").splitlines()
        if (
            match := re.match(
                r"^\s*\([A-Za-z]\)\s+(?P<choice>.+?)\s*$",
                raw,
            )
        )
    ]
    return (
        _content_terms(" ".join(choices))
        - _NON_DISCRIMINATIVE_CHOICE_TERMS
    )


def _local_batch_caption(
    config: dict[str, Any],
    parameters: dict[str, Any],
    batch: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, float]:
    question = _question_only(str(parameters.get("question") or ""))
    words = max(5, int(config.get("dual_path_local_caption_words") or 15))
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are a local visual routing observer. Do not answer the question and do not discuss answer choices. "
                "Caption every frame independently using only visible facts. Emphasize visible entities, actions, "
                "state changes, counts, colors, and interactions that could help locate the question's evidence, "
                "but still describe a frame when the target is absent.\n\n"
                f"Question:\n{question}\n\n"
            ),
        }
    ]
    timestamps: list[float] = []
    for index, item in enumerate(batch, start=1):
        timestamp = round(float(item["timestamp_s"]), 1)
        timestamps.append(timestamp)
        content.append(
            {
                "type": "text",
                "text": f"Frame F{index:02d}, global timestamp={timestamp:.1f}s:",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _frame_data_url(item["frame"])},
            }
        )
    content.append(
        {
            "type": "text",
            "text": (
                f"Return exactly {len(batch)} plain-text lines in chronological order.\n"
                f"Format: F01: factual visible caption of at most {words} words\n"
                "Do not output JSON, markdown, confidence, relevance labels, option letters, or a final answer."
            ),
        }
    )

    observer_config = dict(config)
    observer_config["observer_backend"] = "local_qwen"
    observer_config["local_qwen_tools"] = "local_overview_qwen"
    observer_config["local_qwen_fallback_to_api"] = False
    observer_config["local_qwen_max_images"] = max(
        len(batch), int(config.get("local_qwen_max_images") or 24)
    )
    observer_config["local_qwen_max_new_tokens"] = int(
        config.get("dual_path_local_max_new_tokens") or 512
    )
    observer_config["local_qwen_timeout_s"] = int(
        config.get("dual_path_local_batch_timeout_s") or 120
    )
    started = time.perf_counter()
    raw, _backend = observe_content(
        observer_config,
        content=content,
        tool_name="local_overview_qwen",
        tool_mode="supplementary_overview",
        output_dir=parameters.get("output_dir"),
    )
    wall = time.perf_counter() - started
    rows = _parse_plain_caption_rows(
        raw,
        timestamps=timestamps,
        caption_char_limit=max(96, words * 12),
    )
    return rows, str(raw or ""), wall


def _parse_plain_caption_rows(
    raw: Any,
    *,
    timestamps: list[float],
    caption_char_limit: int,
) -> list[dict[str, Any]]:
    """Parse caption-only Qwen output without asking the model for JSON."""

    parsed: dict[int, str] = {}
    for line in str(raw or "").splitlines():
        match = _CAPTION_LINE_RE.match(line)
        if not match:
            continue
        index = int(match.group("index"))
        if not 1 <= index <= len(timestamps) or index in parsed:
            continue
        caption = match.group("caption").strip().strip("`").strip()
        if caption:
            parsed[index] = caption[:caption_char_limit]
    return [
        {
            "frame_id": f"F{index:02d}",
            "timestamp_s": round(float(timestamps[index - 1]), 3),
            "caption": parsed[index],
        }
        for index in sorted(parsed)
    ]


def _content_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", str(text or "").lower()):
        if len(token) < 3 or token in _QUERY_STOP_WORDS:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("es") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        terms.add(token)
    return terms


def _canonical_retrieval_terms(text: str) -> set[str]:
    """Normalize lightweight caption terms without adding a model call."""

    return {
        term[:-1] if len(term) > 4 and term.endswith("e") else term
        for term in _content_terms(text)
    }


def _compact_text(text: Any, limit: int = 180) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 3)].rstrip() + "..."


def _semantic_terms_from_remote(
    remote_payload: dict[str, Any],
    *,
    remote_scene: dict[str, Any] | None = None,
) -> set[str]:
    texts: list[str] = []
    if remote_scene is not None:
        texts.append(str(remote_scene.get("summary") or ""))
    else:
        texts.extend(
            str(item.get("summary") or item.get("observed_event") or "")
            for item in remote_payload.get("scene_summaries") or []
            if isinstance(item, dict)
        )
        texts.extend(
            str(item.get("description") or item.get("desc") or "")
            for item in remote_payload.get("timestamp_observations") or []
            if isinstance(item, dict)
        )
    return _content_terms(" ".join(texts))


def _local_candidate_score(
    row: dict[str, Any],
    spec: dict[str, Any],
    *,
    query_terms: set[str],
    choice_terms: set[str],
) -> tuple[float, list[str], list[str]]:
    caption_terms = _content_terms(str(row.get("caption") or ""))
    matched_query = sorted(query_terms & caption_terms)
    matched_choices = sorted(choice_terms & caption_terms)
    query_relevance = len(matched_query) / max(1, len(query_terms))
    codec_prior = float(spec.get("score") or 0.0)
    specificity = min(1.0, len(caption_terms) / 10.0)
    if choice_terms:
        # One concrete option concept is already a strong routing clue.
        # Dividing by the number of choices would suppress sparse matches.
        choice_relevance = min(1.0, float(len(matched_choices)))
        score = (
            0.25 * query_relevance
            + 0.35 * choice_relevance
            + 0.25 * codec_prior
            + 0.15 * specificity
        )
    else:
        # Preserve V94 exactly for count and other questions whose options do
        # not provide usable textual concepts.
        score = (
            0.5 * query_relevance
            + 0.35 * codec_prior
            + 0.15 * specificity
        )
    return (
        round(min(1.0, score), 4),
        matched_query,
        matched_choices,
    )


def build_local_candidates(
    frame_rows: list[dict[str, Any]],
    frame_specs: list[dict[str, Any]],
    *,
    duration: float,
    top_k: int,
    window_s: float,
    question: str = "",
    choice_aware: bool = True,
) -> list[dict[str, Any]]:
    query_terms = _content_terms(_question_only(question))
    choice_terms = _choice_terms(question) if choice_aware else set()
    raw: list[dict[str, Any]] = []
    half = max(1.0, float(window_s) / 2.0)
    for row in frame_rows:
        timestamp = float(row["timestamp_s"])
        spec = min(
            frame_specs,
            key=lambda item: abs(float(item["timestamp_s"]) - timestamp),
        )
        priority, matched_terms, matched_choice_terms = _local_candidate_score(
            row,
            spec,
            query_terms=query_terms,
            choice_terms=choice_terms,
        )
        start = max(0.0, timestamp - half)
        end = min(duration, timestamp + half)
        raw.append(
            {
                "scene_id": spec.get("scene_id") or "overview",
                "t_range": [round(start, 3), round(end, 3)],
                "summary": str(row.get("caption") or "").strip(),
                "priority": priority,
                "anchors": [round(timestamp, 3)],
                "candidate_type": spec.get("source") or "supplementary",
                "matched_query_terms": matched_terms,
                "matched_choice_terms": matched_choice_terms,
                "evidence_level": "routing",
                "observer_backend": "local_qwen",
                "needs_verify": True,
            }
        )

    selected: list[dict[str, Any]] = []
    remaining = list(raw)
    covered_choice_terms: set[str] = set()
    while remaining and len(selected) < max(1, int(top_k)):
        # Greedy concept diversity prevents repeated high-motion frames of one
        # concept from crowding out a different observed answer concept.
        item = max(
            remaining,
            key=lambda value: (
                float(value["priority"])
                + (
                    0.25
                    if set(value.get("matched_choice_terms") or [])
                    - covered_choice_terms
                    else 0.0
                ),
                float(value["priority"]),
                -float(value["t_range"][0]),
            ),
        )
        remaining.remove(item)
        duplicate = None
        for old in selected:
            overlap = max(
                0.0,
                min(item["t_range"][1], old["t_range"][1])
                - max(item["t_range"][0], old["t_range"][0]),
            )
            shorter = max(
                1e-6,
                min(
                    item["t_range"][1] - item["t_range"][0],
                    old["t_range"][1] - old["t_range"][0],
                ),
            )
            if overlap / shorter >= 0.5:
                duplicate = old
                break
        if duplicate is not None:
            duplicate["anchors"] = sorted(set(duplicate["anchors"] + item["anchors"]))
            duplicate["scene_ids"] = sorted(
                set(
                    duplicate.get("scene_ids", [duplicate["scene_id"]])
                    + [item["scene_id"]]
                )
            )
            if item["summary"] not in duplicate["summary"]:
                duplicate["summary"] = f"{duplicate['summary']}; {item['summary']}"[:320]
            duplicate["priority"] = max(duplicate["priority"], item["priority"])
            duplicate["matched_query_terms"] = sorted(
                set(duplicate.get("matched_query_terms") or [])
                | set(item.get("matched_query_terms") or [])
            )
            duplicate["matched_choice_terms"] = sorted(
                set(duplicate.get("matched_choice_terms") or [])
                | set(item.get("matched_choice_terms") or [])
            )
            covered_choice_terms.update(
                duplicate.get("matched_choice_terms") or []
            )
            continue
        item["scene_ids"] = [item["scene_id"]]
        selected.append(item)
        covered_choice_terms.update(item.get("matched_choice_terms") or [])
    for index, item in enumerate(selected, start=1):
        item["candidate_id"] = f"OVL{index:03d}"
        item["recommended_verify_window"] = item["t_range"]
    return selected


def run_local_supplementary_scan(
    config: dict[str, Any],
    parameters: dict[str, Any],
    frames: list[dict[str, Any]],
    *,
    remote_done: threading.Event,
) -> dict[str, Any]:
    started = time.perf_counter()
    batch_size = max(1, int(config.get("dual_path_local_batch_size") or 8))
    deadline_s = max(1.0, float(config.get("dual_path_local_deadline_s") or 100.0))
    minimum_processed = min(
        len(frames),
        max(0, int(config.get("dual_path_local_min_processed_frames") or 0)),
    )
    frame_rows: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    frames_attempted = 0
    stop_reason = "budget_complete"
    for batch_index in range(0, len(frames), batch_size):
        if (
            batch_index > 0
            and remote_done.is_set()
            and frames_attempted >= minimum_processed
        ):
            stop_reason = "remote_completed"
            break
        if batch_index > 0 and time.perf_counter() - started >= deadline_s:
            stop_reason = "deadline_reached"
            break
        batch = frames[batch_index : batch_index + batch_size]
        frames_attempted += len(batch)
        try:
            rows, raw, wall = _local_batch_caption(config, parameters, batch)
        except Exception as exc:
            stop_reason = f"local_error:{type(exc).__name__}"
            raw_batches.append(
                {
                    "batch_index": batch_index // batch_size,
                    "timestamps": [item["timestamp_s"] for item in batch],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break
        for local_index, row in enumerate(rows):
            global_index = batch_index + local_index + 1
            row["frame_id"] = f"OVF{global_index:03d}"
            spec = min(
                batch,
                key=lambda item: abs(float(item["timestamp_s"]) - float(row["timestamp_s"])),
            )
            row["scene_id"] = spec.get("scene_id") or "overview"
            row["sampling_source"] = spec.get("source")
        frame_rows.extend(rows)
        raw_batches.append(
            {
                "batch_index": batch_index // batch_size,
                "timestamps": [item["timestamp_s"] for item in batch],
                "wall_s": round(wall, 3),
                "parsed_rows": len(rows),
                "raw": raw,
            }
        )

    visible_top_k = max(1, int(config.get("dual_path_local_top_k") or 5))
    configured_pool_k = int(config.get("dual_path_local_candidate_pool_k") or 0)
    candidate_pool_k = max(
        visible_top_k,
        configured_pool_k if configured_pool_k > 0 else len(frame_rows),
    )
    candidate_pool = build_local_candidates(
        frame_rows,
        frames,
        duration=float(parameters.get("duration") or 0.0),
        top_k=candidate_pool_k,
        window_s=float(config.get("dual_path_candidate_window_s") or 12.0),
        question=str(parameters.get("question") or ""),
        choice_aware=bool(
            config.get("dual_path_choice_aware_retrieval_enabled", True)
        ),
    )
    return {
        "source": "local_supplementary_qwen",
        "coverage": {
            "frames_requested": len(frames),
            "minimum_processed_frames": minimum_processed,
            "frames_attempted": frames_attempted,
            "frames_processed": len(frame_rows),
            "batches_completed": sum(1 for item in raw_batches if "wall_s" in item),
            "deadline_s": deadline_s,
            "stop_reason": stop_reason,
        },
        "frame_rows": frame_rows,
        "candidates": candidate_pool[:visible_top_k],
        "candidate_pool": candidate_pool,
        "raw_batches": raw_batches,
        "wall_s": round(time.perf_counter() - started, 3),
    }


def _window_overlap(left: list[float], right: list[float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _windows_intersect(left: list[float], right: list[float]) -> bool:
    """Treat timestamp points and closed temporal windows consistently."""

    return max(float(left[0]), float(right[0])) <= min(
        float(left[1]), float(right[1])
    ) + 1e-6


def _gap_statistics(timestamps: list[float], *, duration: float) -> dict[str, float]:
    anchors = sorted(
        set(
            [0.0, max(0.0, float(duration))]
            + [max(0.0, min(float(value), duration)) for value in timestamps]
        )
    )
    gaps = [max(0.0, end - start) for start, end in zip(anchors, anchors[1:])]
    if not gaps:
        gaps = [max(0.0, float(duration))]
    return {
        "max_gap_s": round(max(gaps), 3),
        "p90_gap_s": round(float(np.percentile(gaps, 90)), 3),
        "mean_gap_s": round(float(np.mean(gaps)), 3),
    }


def _covered_scene_ids(
    skeleton: dict[str, Any], timestamps: list[float]
) -> set[str]:
    return {
        _scene_id_for_timestamp(skeleton, float(timestamp))
        for timestamp in timestamps
    }


def _window_overlap_ratio(left: list[float], right: list[float]) -> float:
    overlap = _window_overlap(left, right)
    shorter = min(
        max(1e-6, float(left[1]) - float(left[0])),
        max(1e-6, float(right[1]) - float(right[0])),
    )
    return overlap / shorter


def _temporally_spread_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        (row for row in rows if row.get("timestamp_s") is not None),
        key=lambda row: float(row["timestamp_s"]),
    )
    if limit <= 0 or not ordered:
        return []
    if len(ordered) <= limit:
        return ordered
    indexes = sorted(
        {
            round(index * (len(ordered) - 1) / max(1, limit - 1))
            for index in range(limit)
        }
    )
    return [ordered[index] for index in indexes]


def _has_discriminative_query_match(candidate: dict[str, Any]) -> bool:
    return bool(
        set(candidate.get("matched_query_terms") or [])
        - _NON_DISCRIMINATIVE_QUERY_TERMS
    )


def select_local_candidates_for_fusion(
    candidates: list[dict[str, Any]],
    *,
    selective: bool,
    limit: int,
    query_target_reserve: bool = False,
) -> list[dict[str, Any]]:
    def discriminative_query_terms(item: dict[str, Any]) -> set[str]:
        return (
            set(item.get("matched_query_terms") or [])
            - _NON_DISCRIMINATIVE_QUERY_TERMS
        )

    def prototype_terms(item: dict[str, Any]) -> set[str]:
        return set(item.get("matched_remote_prototype_terms") or [])

    ranked = sorted(
        candidates,
        key=lambda item: (
            -int(bool(discriminative_query_terms(item))),
            -len(discriminative_query_terms(item)),
            -int(bool(prototype_terms(item))),
            -float(item.get("remote_prototype_temporal_novelty") or 0.0),
            -len(prototype_terms(item)),
            -float(item.get("priority") or 0.0),
            -float(
                item.get("information_gain_score")
                if selective
                else item.get("priority")
                or 0.0
            ),
            float((item.get("t_range") or [0.0])[0]),
        ),
    )
    selected = ranked[: max(0, int(limit))]
    if not query_target_reserve or not selected:
        return selected

    term_counts: dict[str, int] = {}
    for item in selected:
        for term in discriminative_query_terms(item):
            term_counts[term] = term_counts.get(term, 0) + 1
    covered_terms = set(term_counts)
    outside = [item for item in ranked if item not in selected]
    target_candidate = max(
        outside,
        key=lambda item: (
            len(discriminative_query_terms(item) - covered_terms),
            len(discriminative_query_terms(item)),
            float(item.get("priority") or 0.0),
        ),
        default=None,
    )
    if target_candidate is None:
        return selected
    uncovered_terms = discriminative_query_terms(target_candidate) - covered_terms
    if not uncovered_terms:
        return selected

    replacement_index = min(
        range(len(selected)),
        key=lambda index: (
            sum(
                term_counts.get(term, 0) == 1
                for term in discriminative_query_terms(selected[index])
            ),
            -index,
        ),
    )
    replacement_unique_terms = {
        term
        for term in discriminative_query_terms(selected[replacement_index])
        if term_counts.get(term, 0) == 1
    }
    if len(uncovered_terms) <= len(replacement_unique_terms):
        return selected
    selected[replacement_index] = target_candidate
    return selected


def _remote_routing_windows(
    remote_summaries: list[dict[str, Any]],
    *,
    max_scene_window_s: float,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(remote_summaries, start=1):
        if not isinstance(scene, dict) or not bool(scene.get("possible_evidence")):
            continue
        scene_id = str(
            scene.get("window_id")
            or scene.get("scene_id")
            or f"remote_scene_{scene_index}"
        )
        focus_windows = scene.get("suggest_focus_windows") or []
        for focus_index, value in enumerate(focus_windows, start=1):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            windows.append(
                {
                    "id": f"{scene_id}:focus{focus_index}",
                    "window": [float(value[0]), float(value[1])],
                    "scene": scene,
                }
            )
        span = scene.get("t_range") or []
        if (
            not focus_windows
            and len(span) == 2
            and float(span[1]) - float(span[0]) <= max_scene_window_s
        ):
            windows.append(
                {
                    "id": scene_id,
                    "window": [float(span[0]), float(span[1])],
                    "scene": scene,
                }
            )
    return windows


def build_coverage_comparison(
    *,
    remote_timestamps: list[float],
    local_specs: list[dict[str, Any]],
    remote_payload: dict[str, Any],
    local_payload: dict[str, Any],
    skeleton: dict[str, Any],
    duration: float,
) -> dict[str, Any]:
    """Measure sampling complementarity without assigning answer correctness."""

    local_timestamps = [
        float(item["timestamp_s"])
        for item in local_specs
        if item.get("timestamp_s") is not None
    ]
    union_timestamps = sorted(set(remote_timestamps + local_timestamps))
    remote_scenes = _covered_scene_ids(skeleton, remote_timestamps)
    local_scenes = _covered_scene_ids(skeleton, local_timestamps)
    all_scenes = {
        str(item.get("scene_id") or "overview")
        for item in skeleton.get("scenes") or []
        if isinstance(item, dict)
    }

    remote_possible_ranges: list[list[float]] = []
    remote_focus_windows: list[list[float]] = []
    for item in remote_payload.get("scene_summaries") or []:
        if not isinstance(item, dict) or not bool(item.get("possible_evidence")):
            continue
        span = item.get("t_range") or []
        if len(span) == 2:
            remote_possible_ranges.append([float(span[0]), float(span[1])])
        for window in item.get("suggest_focus_windows") or []:
            if isinstance(window, (list, tuple)) and len(window) == 2:
                remote_focus_windows.append([float(window[0]), float(window[1])])

    local_candidates = [
        item
        for item in local_payload.get("candidates") or []
        if isinstance(item, dict) and len(item.get("t_range") or []) == 2
    ]
    outside_remote_focus = 0
    outside_remote_possible_scene = 0
    for item in local_candidates:
        span = [float(value) for value in item["t_range"]]
        if not any(_windows_intersect(span, window) for window in remote_focus_windows):
            outside_remote_focus += 1
        if not any(
            _windows_intersect(span, window) for window in remote_possible_ranges
        ):
            outside_remote_possible_scene += 1

    remote_gaps = _gap_statistics(remote_timestamps, duration=duration)
    union_gaps = _gap_statistics(union_timestamps, duration=duration)
    max_gap_reduction = remote_gaps["max_gap_s"] - union_gaps["max_gap_s"]
    p90_gap_reduction = remote_gaps["p90_gap_s"] - union_gaps["p90_gap_s"]
    return {
        "duration_s": round(float(duration), 3),
        "remote_frame_count": len(set(remote_timestamps)),
        "local_frame_count": len(set(local_timestamps)),
        "union_frame_count": len(union_timestamps),
        "remote_gap_statistics": remote_gaps,
        "union_gap_statistics": union_gaps,
        "max_gap_reduction_s": round(max_gap_reduction, 3),
        "max_gap_reduction_ratio": round(
            max_gap_reduction / max(1e-6, remote_gaps["max_gap_s"]), 4
        ),
        "p90_gap_reduction_s": round(p90_gap_reduction, 3),
        "p90_gap_reduction_ratio": round(
            p90_gap_reduction / max(1e-6, remote_gaps["p90_gap_s"]), 4
        ),
        "skeleton_scene_count": len(all_scenes),
        "remote_scene_count": len(remote_scenes),
        "local_scene_count": len(local_scenes),
        "union_scene_count": len(remote_scenes | local_scenes),
        "local_only_scene_count": len(local_scenes - remote_scenes),
        "remote_possible_scene_count": len(remote_possible_ranges),
        "remote_focus_window_count": len(remote_focus_windows),
        "local_candidate_count": len(local_candidates),
        "local_candidate_outside_remote_focus_count": outside_remote_focus,
        "local_candidate_outside_remote_possible_scene_count": (
            outside_remote_possible_scene
        ),
    }


def merge_overview_payloads(
    remote_payload: dict[str, Any],
    local_payload: dict[str, Any],
    *,
    skeleton: dict[str, Any],
    timing: dict[str, Any],
    coverage_comparison: dict[str, Any] | None = None,
    integration_mode: str = "fused",
    fused_top_k: int = 3,
    planner_caption_limit: int = 24,
    remote_dedup_overlap: float = 0.35,
    candidate_window_s: float = 12.0,
    query_target_reserve_enabled: bool = False,
    semantic_prototype_ranking_enabled: bool = False,
) -> dict[str, Any]:
    merged = deepcopy(remote_payload)
    selective = integration_mode == "selective"
    if integration_mode == "reserve":
        merged["dual_path_candidates"] = deepcopy(
            local_payload.get("candidates") or []
        )
        merged["dual_path_overview"] = {
            "mode": "latency_hiding_local_cloud",
            "integration_mode": "reserve",
            "local_coverage": local_payload.get("coverage") or {},
            "local_candidate_count": len(local_payload.get("candidates") or []),
            "coverage_comparison": coverage_comparison or {},
            "timing": timing,
        }
        merged["internal_qwen_calls"] = int(
            (local_payload.get("coverage") or {}).get("batches_completed") or 0
        )
        merged.setdefault("observer_backend", "api")
        return merged

    remote_summaries = [
        item for item in merged.get("scene_summaries") or [] if isinstance(item, dict)
    ]
    for item in merged.get("timestamp_observations") or []:
        if not isinstance(item, dict):
            continue
        try:
            timestamp = float(item.get("timestamp_s"))
        except (TypeError, ValueError):
            continue
        item["model_scene_id"] = item.get("scene_id")
        item["scene_id"] = _scene_id_for_timestamp(skeleton, timestamp)
        item.setdefault("observer_backend", "api")
        item.setdefault("evidence_level", "routing")

    for item in remote_summaries:
        span = item.get("t_range") or []
        if len(span) != 2:
            continue
        try:
            center = (float(span[0]) + float(span[1])) / 2.0
        except (TypeError, ValueError):
            continue
        item["model_scene_id"] = item.get("scene_id")
        item["scene_id"] = _scene_id_for_timestamp(skeleton, center)
        item.setdefault("observer_backend", "api")
        item.setdefault("evidence_level", "routing")

    remote_windows = _remote_routing_windows(
        remote_summaries,
        max_scene_window_s=max(20.0, float(candidate_window_s) * 2.0),
    )
    local_candidates = deepcopy(
        (
            local_payload.get("candidate_pool")
            if semantic_prototype_ranking_enabled
            else local_payload.get("candidates")
        )
        or []
    )
    novel_candidates: list[dict[str, Any]] = []
    deduplicated_candidates: list[dict[str, Any]] = []
    threshold = max(0.0, min(1.0, float(remote_dedup_overlap)))
    global_remote_terms = _semantic_terms_from_remote(remote_payload)
    remote_prototype_terms: set[str] = set()
    if semantic_prototype_ranking_enabled:
        prototype_counts: dict[str, int] = {}
        for item in remote_summaries:
            if not bool(item.get("possible_evidence")):
                continue
            for term in _canonical_retrieval_terms(item.get("summary") or ""):
                prototype_counts[term] = prototype_counts.get(term, 0) + 1
        # A repeated term across independent positive scenes is a stable
        # semantic prototype. Single-scene details remain too noisy and are
        # handled by direct query-term matching instead.
        remote_prototype_terms = {
            term for term, count in prototype_counts.items() if count >= 2
        } - _NON_DISCRIMINATIVE_PROTOTYPE_TERMS
    for candidate in local_candidates:
        span = candidate.get("t_range") or []
        if len(span) != 2:
            continue
        local_window = [float(span[0]), float(span[1])]
        overlaps = sorted(
            (
                (
                    _window_overlap_ratio(local_window, item["window"]),
                    item,
                )
                for item in remote_windows
            ),
            key=lambda value: value[0],
            reverse=True,
        )
        best_ratio, best_remote = overlaps[0] if overlaps else (0.0, None)
        candidate["remote_overlap_ratio"] = round(float(best_ratio), 4)
        candidate["remote_overlap_id"] = best_remote["id"] if best_remote else ""
        if semantic_prototype_ranking_enabled:
            matched_prototype_terms = sorted(
                (
                    _canonical_retrieval_terms(candidate.get("summary") or "")
                    & remote_prototype_terms
                )
                - _NON_DISCRIMINATIVE_PROTOTYPE_TERMS
            )
            candidate["matched_remote_prototype_terms"] = matched_prototype_terms
            if remote_windows:
                temporal_distance = min(
                    (
                        0.0
                        if _windows_intersect(local_window, item["window"])
                        else min(
                            abs(local_window[0] - float(item["window"][1])),
                            abs(float(item["window"][0]) - local_window[1]),
                        )
                    )
                    for item in remote_windows
                )
                candidate["remote_prototype_temporal_novelty"] = round(
                    min(
                        1.0,
                        max(0.0, temporal_distance)
                        / max(1.0, float(candidate_window_s)),
                    ),
                    4,
                )
            else:
                candidate["remote_prototype_temporal_novelty"] = 1.0
        caption_terms = _content_terms(candidate.get("summary") or "")
        comparison_terms = (
            _semantic_terms_from_remote(
                remote_payload,
                remote_scene=best_remote["scene"],
            )
            if best_remote is not None
            else global_remote_terms
        )
        novel_terms = sorted(caption_terms - comparison_terms)
        semantic_novelty = len(novel_terms) / max(1, len(caption_terms))
        temporal_novelty = max(0.0, 1.0 - float(best_ratio))
        information_gain = (
            0.5 * float(candidate.get("priority") or 0.0)
            + 0.3 * semantic_novelty
            + 0.2 * temporal_novelty
        )
        candidate["semantic_novelty"] = round(semantic_novelty, 4)
        candidate["semantic_novel_terms"] = novel_terms[:8]
        candidate["information_gain_score"] = round(
            min(1.0, information_gain), 4
        )
        semantic_enrichment = (
            selective
            and semantic_novelty >= 0.35
            and bool(novel_terms)
        )
        if (
            best_remote is not None
            and best_ratio >= threshold
            and not semantic_enrichment
        ):
            candidate["fusion_status"] = "deduplicated_with_remote"
            deduplicated_candidates.append(candidate)
            remote_scene = best_remote["scene"]
            remote_scene.setdefault("supplemental_local_anchors", [])
            for anchor in candidate.get("anchors") or []:
                value = round(float(anchor), 1)
                if value not in remote_scene["supplemental_local_anchors"]:
                    remote_scene["supplemental_local_anchors"].append(value)
            continue
        candidate["fusion_status"] = (
            "semantic_enrichment"
            if best_remote is not None and best_ratio >= threshold
            else "novel_local_routing"
        )
        novel_candidates.append(candidate)

    novel_candidates = select_local_candidates_for_fusion(
        novel_candidates,
        selective=selective,
        limit=fused_top_k,
        query_target_reserve=query_target_reserve_enabled,
    )
    local_scene_summaries: list[dict[str, Any]] = []
    local_timestamp_rows: list[dict[str, Any]] = []
    rows_by_timestamp = {
        round(float(row.get("timestamp_s") or -1.0), 1): row
        for row in local_payload.get("frame_rows") or []
    }
    for candidate in novel_candidates:
        local_scene_summaries.append(
            {
                "scene_id": candidate.get("scene_id"),
                "window_id": candidate.get("candidate_id"),
                "t_range": candidate.get("t_range"),
                "summary": _compact_text(candidate.get("summary")),
                "possible_evidence": True,
                "suggest_focus_windows": [candidate.get("recommended_verify_window")],
                "missing_detail": "Local codec/Qwen routing clue; verify before answering.",
                "observer_backend": "local_qwen",
                "evidence_level": "routing_only",
                "candidate_type": candidate.get("candidate_type"),
                "priority": candidate.get("priority"),
                "information_gain_score": candidate.get(
                    "information_gain_score"
                ),
                "matched_query_terms": candidate.get("matched_query_terms") or [],
                "matched_choice_terms": candidate.get("matched_choice_terms")
                or [],
                "matched_remote_prototype_terms": candidate.get(
                    "matched_remote_prototype_terms"
                )
                or [],
                "remote_prototype_temporal_novelty": candidate.get(
                    "remote_prototype_temporal_novelty"
                ),
                "semantic_novel_terms": candidate.get(
                    "semantic_novel_terms"
                )
                or [],
                "fusion_status": candidate.get("fusion_status"),
                "provenance": "local_interframe_dynamics",
            }
        )
        anchors = candidate.get("anchors") or []
        if not anchors:
            continue
        timestamp = round(float(anchors[0]), 1)
        row = rows_by_timestamp.get(timestamp) or {}
        local_timestamp_rows.append(
            {
                "timestamp_s": timestamp,
                "scene_id": candidate.get("scene_id"),
                "window_id": candidate.get("candidate_id"),
                "description": _compact_text(
                    row.get("caption") or candidate.get("summary")
                ),
                "event_tags": [
                    "dual_path_local_candidate",
                    candidate.get("candidate_type"),
                ],
                "confidence": "routing_only",
                "frame_ids": [row.get("frame_id")] if row.get("frame_id") else [],
                "needs_focus": "Verify this local routing candidate before answering.",
                "observer_backend": "local_qwen",
                "evidence_level": "routing_only",
                "provenance": "local_interframe_dynamics",
            }
        )

    planner_caption_rows = [
        {
            "timestamp_s": round(float(row["timestamp_s"]), 1),
            "scene_id": row.get("scene_id"),
            "window_id": "dual_path_local_timeline",
            "description": _compact_text(row.get("caption")),
            "event_tags": [
                "dual_path_local_caption",
                row.get("sampling_source"),
            ],
            "confidence": "routing_only",
            "frame_ids": [row.get("frame_id")] if row.get("frame_id") else [],
            "needs_focus": "Local factual caption; verify if selected.",
            "observer_backend": "local_qwen",
            "evidence_level": "routing_only",
            "provenance": "local_interframe_dynamics",
        }
        for row in _temporally_spread_rows(
            list(local_payload.get("frame_rows") or []),
            limit=max(0, int(planner_caption_limit)),
        )
    ]
    planner_timestamps = {
        round(float(row["timestamp_s"]), 1) for row in planner_caption_rows
    }
    for row in local_timestamp_rows:
        timestamp = round(float(row["timestamp_s"]), 1)
        if timestamp not in planner_timestamps:
            planner_caption_rows.append(row)
            planner_timestamps.add(timestamp)
    planner_caption_rows.sort(key=lambda row: float(row["timestamp_s"]))

    if integration_mode in {"append", "fused", "selective"}:
        combined_timestamp_rows = list(
            merged.get("timestamp_observations") or []
        ) + planner_caption_rows
        combined_timestamp_rows.sort(
            key=lambda row: float(row.get("timestamp_s") or 0.0)
        )
        merged["timestamp_observations"] = combined_timestamp_rows
        merged["scene_summaries"] = remote_summaries + local_scene_summaries
    if selective:
        merged.pop("dual_path_candidates", None)
    else:
        merged["dual_path_candidates"] = novel_candidates
    merged["unified_timestamped_evidence_map"] = {
        "remote_candidate_count": len(remote_windows),
        "local_candidate_count": len(local_candidates),
        "local_novel_candidate_count": len(novel_candidates),
        "local_deduplicated_candidate_count": len(deduplicated_candidates),
        "local_planner_caption_count": len(planner_caption_rows),
        "local_candidates_are_routing_only": True,
    }
    if selective:
        local_coverage = local_payload.get("coverage") or {}
        merged["dual_path_overview"] = {
            "mode": "latency_hiding_local_cloud",
            "integration_mode": "selective",
            "local_coverage": {
                "frames_processed": local_coverage.get("frames_processed"),
                "batches_completed": local_coverage.get("batches_completed"),
                "stop_reason": local_coverage.get("stop_reason"),
            },
            "local_candidate_count": len(local_candidates),
            "local_novel_candidate_count": len(novel_candidates),
            "local_deduplicated_candidate_count": len(deduplicated_candidates),
            "local_planner_caption_count": len(planner_caption_rows),
            "timing": {
                "exposed_local_wait_s": timing.get("exposed_local_wait_s", 0.0),
                "latency_hiding_ratio": timing.get("latency_hiding_ratio", 0.0),
            },
        }
    else:
        merged["dual_path_overview"] = {
            "mode": "latency_hiding_local_cloud",
            "integration_mode": integration_mode,
            "local_coverage": local_payload.get("coverage") or {},
            "local_candidate_count": len(local_candidates),
            "local_novel_candidate_count": len(novel_candidates),
            "local_deduplicated_candidate_count": len(deduplicated_candidates),
            "local_planner_caption_count": len(planner_caption_rows),
            "coverage_comparison": coverage_comparison or {},
            "timing": timing,
        }
    merged["internal_qwen_calls"] = int(
        (local_payload.get("coverage") or {}).get("batches_completed") or 0
    )
    merged.setdefault("observer_backend", "api")
    return merged


def _write_trace(output_dir: str | None, payload: dict[str, Any]) -> str:
    if not output_dir:
        return ""
    trace_dir = Path(output_dir) / "dual_path_overview_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"dual_path_{time.time_ns()}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def execute_dual_path_overview(
    config: dict[str, Any],
    parameters: dict[str, Any],
    *,
    remote_runner: RemoteOverviewRunner,
) -> str:
    """Run baseline remote Overview and a complementary local scan concurrently."""

    started = time.perf_counter()
    vr = parameters["vr"]
    duration = float(parameters.get("duration") or round(len(vr) / vr.get_avg_fps(), 1))
    video_path = str(parameters.get("video_path") or "")
    skeleton, _cache_path = load_or_build_skeleton(
        config,
        video_path=video_path,
        duration_s=duration,
    )
    remote_timestamps = _remote_overview_timestamps(
        config,
        video_path=video_path,
        duration=duration,
    )
    specs = select_supplementary_frames(
        remote_timestamps=remote_timestamps,
        packet_samples=list(probe_interframe_packet_sizes(video_path)),
        skeleton=skeleton,
        duration=duration,
        limit=_local_frame_budget(config, duration=duration),
        remote_gap_s=float(config.get("dual_path_remote_exclusion_s") or 0.5),
        local_gap_s=float(config.get("dual_path_local_min_gap_s") or 0.45),
    )
    frames = _decode_supplementary_frames(
        parameters,
        specs,
        short_side=int(config.get("dual_path_local_frame_short_side") or 384),
    )
    preprocess_wall = time.perf_counter() - started

    remote_done = threading.Event()
    parallel_started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dual-overview")
    local_future = executor.submit(
        run_local_supplementary_scan,
        config,
        {**parameters, "duration": duration},
        frames,
        remote_done=remote_done,
    )
    try:
        # Keep API dispatch on the caller thread for its budget/timeout context.
        remote_raw = remote_runner(config, parameters)
        remote_finished = time.perf_counter()
        remote_done.set()
        local_payload = local_future.result()
        local_finished = time.perf_counter()
    finally:
        remote_done.set()
        executor.shutdown(wait=True, cancel_futures=True)

    remote_wall = remote_finished - parallel_started
    local_wall = float(local_payload.get("wall_s") or 0.0)
    timing = {
        "preprocess_wall_s": round(preprocess_wall, 3),
        "remote_wall_s": round(remote_wall, 3),
        "local_wall_s": round(local_wall, 3),
        "parallel_wall_s": round(local_finished - parallel_started, 3),
        "hidden_local_wall_s": round(min(remote_wall, local_wall), 3),
        "exposed_local_wait_s": round(max(0.0, local_finished - remote_finished), 3),
        "latency_hiding_ratio": round(
            min(1.0, remote_wall / max(1e-6, local_wall)), 4
        ),
    }
    remote_payload = extract_v10_payload(remote_raw) or {}
    coverage_comparison = build_coverage_comparison(
        remote_timestamps=remote_timestamps,
        local_specs=frames,
        remote_payload=remote_payload,
        local_payload=local_payload,
        skeleton=skeleton,
        duration=duration,
    )
    trace = {
        "tool": "dual_path_overview",
        "shadow_only": bool(config.get("dual_path_overview_shadow_only", True)),
        "remote_timestamps": remote_timestamps,
        "supplementary_frame_specs": [
            {key: value for key, value in item.items() if key != "frame"}
            for item in frames
        ],
        "remote_payload": remote_payload,
        "local_payload": local_payload,
        "coverage_comparison": coverage_comparison,
        "timing": timing,
    }
    trace_path = ""
    if bool(config.get("dual_path_overview_trace_enabled", True)):
        trace_path = _write_trace(parameters.get("output_dir"), trace)

    if bool(config.get("dual_path_overview_shadow_only", True)):
        return remote_raw
    if not remote_payload:
        return remote_raw
    merged = merge_overview_payloads(
        remote_payload,
        local_payload,
        skeleton=skeleton,
        timing={**timing, "trace_path": trace_path},
        coverage_comparison=coverage_comparison,
        integration_mode=str(config.get("dual_path_integration_mode") or "fused"),
        fused_top_k=int(config.get("dual_path_fused_top_k") or 3),
        planner_caption_limit=int(
            24
            if config.get("dual_path_planner_caption_limit") is None
            else config.get("dual_path_planner_caption_limit")
        ),
        remote_dedup_overlap=float(
            config.get("dual_path_remote_dedup_overlap") or 0.35
        ),
        candidate_window_s=float(
            config.get("dual_path_candidate_window_s") or 12.0
        ),
        query_target_reserve_enabled=bool(
            config.get("dual_path_query_target_reserve_enabled", False)
        ),
        semantic_prototype_ranking_enabled=bool(
            config.get("dual_path_semantic_prototype_ranking_enabled", False)
        ),
    )
    return format_v10_observation(merged, fallback_text=remote_raw)
