import json
import hashlib
import re
from copy import deepcopy
from typing import Any

from .evidence_state import (
    ensure_structured_evidence_state,
    format_structured_evidence_for_answer,
    format_structured_evidence_for_prompt,
    init_structured_evidence_state,
    update_structured_evidence_state,
)
from .candidate_frontier import (
    format_candidate_frontier_for_prompt,
    init_candidate_frontier,
    refresh_candidate_frontier,
)
from .evidence_episode_frontier import (
    format_evidence_episode_frontier_for_answer,
    format_evidence_episode_frontier_for_prompt,
    refresh_evidence_episode_frontier,
)


_JSON_BLOCK_RE = re.compile(
    r"V10_OBSERVATION_JSON\s*:\s*```json\s*(\{.*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)


def init_observation_memory() -> dict[str, Any]:
    return {
        "version": "coseek_p21_temporal_evidence_memory",
        "timestamped_observations": [],
        "scene_memory": [],
        "open_gaps": [],
        "tool_observations": [],
        "structured_evidence": init_structured_evidence_state(),
        "compact_investigation_state": {},
        "event_coverage": [],
        "temporal_evidence_ledger": {},
        "verified_observation_reuse_stats": {
            "reuse_count": 0,
            "reused_cache_ids": [],
        },
        "verification_facts": [],
        "observation_conflicts": [],
        "overview_candidates": [],
        "candidate_pool": [],
        "scene_coverage": [],
        "candidate_frontier": init_candidate_frontier(),
    }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _question_stem(question: str | None) -> str:
    text = question or ""
    return re.split(r"\n\s*\([A-Z]\)\s+", text, maxsplit=1)[0].strip()


def _extract_choices(question: str | None) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for match in re.finditer(
        r"(?m)^\s*\(([A-Z])\)\s*(.+?)\s*$",
        question or "",
    ):
        choices.append((match.group(1).upper(), match.group(2).strip()))
    return choices


_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "into", "onto",
    "what", "when", "where", "which", "who", "why", "how", "does", "did",
    "was", "were", "are", "after", "before", "then", "there", "video",
    "scene", "action", "person", "people", "man", "woman", "girl", "boy",
    "some", "they", "them", "their", "shown", "total", "count", "order",
    "following", "option", "represents", "actual", "occurrences",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten",
}


_WEAK_EVIDENCE_RE = re.compile(
    r"\b(possible|potential|possibly|could|may|might|unclear|ambiguous|"
    r"verify|confirm|if any|elsewhere|no clear|not clear|not visible)\b",
    re.IGNORECASE,
)


def _keywords(text: str | None) -> set[str]:
    normalized = (text or "").lower()
    normalized = re.sub(r"\bwheel\s*chairs?\b", "wheelchair", normalized)
    # Hyphenated negations such as "non-dog livestock" still carry the
    # discriminating entity term and must remain retrievable by option text.
    normalized = normalized.replace("-", " ")
    words = {
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", normalized)
    }
    return {word for word in words if word not in _STOPWORDS}


def _text_for_memory_item(item: dict[str, Any]) -> str:
    parts = [
        item.get("description"),
        item.get("summary"),
        item.get("fact"),
        item.get("event"),
        item.get("missing_detail"),
        item.get("needs_focus"),
        " ".join(str(tag) for tag in item.get("event_tags") or []),
        " ".join(str(x) for x in item.get("supports_options") or []),
        " ".join(str(x) for x in item.get("contradicts_options") or []),
    ]
    return " ".join(str(part) for part in parts if part)


def _terms_related(left: str, right: str) -> bool:
    if left == right:
        return True
    # Generic compound/plural matching: stairs <-> staircase, bottles <-> bottle.
    def roots(term: str) -> set[str]:
        values = {term}
        if len(term) > 4 and term.endswith("s"):
            values.add(term[:-1])
        if len(term) > 6 and term.endswith("ing"):
            values.add(term[:-3])
        if len(term) > 5 and term.endswith("ed"):
            values.add(term[:-2])
        return values

    return any(
        a == b or (min(len(a), len(b)) >= 5 and (a in b or b in a))
        for a in roots(left)
        for b in roots(right)
    )


def _query_overlap_score(text: str | None, question: str | None) -> int:
    # Include option text as well as the question stem: an overview scene may
    # mention a competing option even when the stem uses only a generic noun.
    query_terms = _keywords(question)
    text_terms = _keywords(text)
    if not query_terms or not text_terms:
        return 0
    return sum(
        1
        for query_term in query_terms
        if any(_terms_related(query_term, text_term) for text_term in text_terms)
    )


def _select_query_relevant_items(
    items: list[dict[str, Any]],
    *,
    question: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    ranked = sorted(
        enumerate(items),
        key=lambda row: (
            _query_overlap_score(_text_for_memory_item(row[1]), question),
            str(row[1].get("source_tool") or "") != "overview",
            row[0],
        ),
        reverse=True,
    )[:limit]
    selected_indices = {index for index, _item in ranked}
    return [item for index, item in enumerate(items) if index in selected_indices]


def _positive_text_for_memory_item(item: dict[str, Any]) -> str:
    parts = [
        item.get("description"),
        item.get("summary"),
        " ".join(str(tag) for tag in item.get("event_tags") or []),
        " ".join(str(x) for x in item.get("supports_options") or []),
    ]
    return " ".join(str(part) for part in parts if part)


def _routing_positive_text(item: dict[str, Any]) -> str:
    parts = [str(item.get("summary") or "")]
    missing = str(item.get("missing_detail") or "").strip()
    if missing and not re.search(
        r"(?i)^\s*(?:n/?a\b|none\b|not\b)|\bnone\s+relevant\b|\bnot\b.{0,40}\brelated\b",
        missing,
    ):
        parts.append(missing)
    return " ".join(parts)


def _has_explicit_negative_routing_cue(item: dict[str, Any]) -> bool:
    missing = re.sub(
        r"\s+", " ", str(item.get("missing_detail") or "")
    ).strip().lower()
    if not missing:
        return False
    return bool(
        re.match(
            r"^(?:none?\b|no\b|not\b|unrelated\b|irrelevant\b|without\b|"
            r"does\s+not\b|doesn't\b|cannot\b|can't\b)",
            missing,
        )
    )


def _item_time(item: dict[str, Any]) -> float | None:
    ts = _safe_float(item.get("timestamp_s"))
    if ts is not None:
        return ts
    span = item.get("t_range") or item.get("time_range_s")
    if isinstance(span, (list, tuple)) and span:
        return _safe_float(span[0])
    return None


def _item_window(item: dict[str, Any]) -> list[float] | None:
    ts = _safe_float(item.get("timestamp_s"))
    if ts is not None:
        return [round(max(0.0, ts - 1.0), 1), round(ts + 1.0, 1)]
    span = item.get("t_range") or item.get("time_range_s")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        start = _safe_float(span[0])
        end = _safe_float(span[1])
        if start is not None and end is not None and end >= start:
            return [round(start, 1), round(end, 1)]
    return None


def _evidence_items(memory: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in memory.get("timestamped_observations") or []:
        if isinstance(item, dict):
            clone = dict(item)
            clone["_kind"] = "obs"
            items.append(clone)
    for item in memory.get("scene_memory") or []:
        if isinstance(item, dict):
            clone = dict(item)
            clone["_kind"] = "summary"
            items.append(clone)
    return items


def _match_score(text: str, phrase: str) -> int:
    phrase_terms = _keywords(phrase)
    if not phrase_terms:
        return 0
    text_terms = _keywords(text)
    return len(phrase_terms & text_terms)


def _is_negative_match(text: str, phrase: str) -> bool:
    terms = _keywords(phrase)
    if not terms:
        return False
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]*", (text or "").lower())
    negators = {
        "no", "not", "none", "without", "absent", "missing", "unlikely",
        "neither", "nor",
    }
    for idx, token in enumerate(tokens):
        if token not in terms:
            continue
        context = tokens[max(0, idx - 6) : idx]
        if any(word in negators for word in context):
            return True
        context_text = " ".join(context)
        if "no evidence" in context_text or "not visible" in context_text:
            return True
    return False


def _is_positive_candidate(item: dict[str, Any], phrase: str) -> bool:
    positive_text = _positive_text_for_memory_item(item)
    if _match_score(positive_text, phrase) <= 0:
        return False
    if _is_negative_match(_text_for_memory_item(item), phrase):
        return False
    if item.get("_kind") == "summary" and item.get("possible_evidence") is False:
        return False
    return True


def _is_uncertain_evidence(item: dict[str, Any]) -> bool:
    text = _positive_text_for_memory_item(item)
    return bool(_WEAK_EVIDENCE_RE.search(text))


def _extract_order_events(question: str | None) -> list[tuple[str, str]]:
    stem = _question_stem(question)
    events: list[tuple[str, str]] = []
    for match in re.finditer(
        r"\((\d+)\)\s*(.+?)(?=(?:\(\d+\)|$))",
        stem,
        flags=re.DOTALL,
    ):
        text = re.sub(r"\s+", " ", match.group(2)).strip(" ;,.")
        if text:
            events.append((match.group(1), text))
    if events:
        return events

    seen: set[str] = set()
    for _label, choice in _extract_choices(question):
        if "-->" not in choice:
            continue
        for chunk in choice.split("-->"):
            text = re.sub(r"\s+", " ", chunk).strip(" ;,.")
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                events.append((str(len(events) + 1), text))
    return events


def _extract_count_target(question: str | None) -> str:
    stem = _question_stem(question)
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", stem)
    for left, right in quoted:
        text = (left or right).strip()
        if text:
            return text
    match = re.search(
        r"(?is)(?:featuring|of|for)\s+(?:the\s+)?(.+?)(?:\s+appear|\s+occurs|\s+in total|$)",
        stem,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip(" ?.")
    return ""


def _is_count_question(question: str | None) -> bool:
    stem = _question_stem(question).lower()
    return any(pattern in stem for pattern in ("how many", "total count", "count of occurrences"))


def _is_order_question(question: str | None) -> bool:
    stem = _question_stem(question).lower()
    return "order" in stem or "chronological" in stem or "-->" in (question or "")


def _extract_object_state_target(question: str | None) -> str:
    stem = _question_stem(question)
    patterns = [
        r"(?is)where\s+was\s+(?:the\s+)?(.+?)\s+before\s+i\s+picked\s+it\s+up",
        r"(?is)where\s+was\s+(?:the\s+)?(.+?)\s+before\s+.*?picked\s+it\s+up",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip(" ?.")
    return ""


def _format_item_ref(item: dict[str, Any]) -> str:
    scene = item.get("scene_id") or "?"
    window = _item_window(item)
    when = window if window is not None else item.get("timestamp_s")
    text = _text_for_memory_item(item).strip()
    return f"{scene} {when}: {text[:120]}"


def _build_event_timeline(memory: dict[str, Any], question: str | None, *, max_hits: int = 3) -> list[str]:
    events = _extract_order_events(question)
    if not events:
        return []
    items = _evidence_items(memory)
    lines = ["Task-aware event timeline candidates:"]
    for event_id, event_text in events:
        scored = []
        for item in items:
            if not _is_positive_candidate(item, event_text):
                continue
            score = _match_score(_positive_text_for_memory_item(item), event_text)
            uncertain = _is_uncertain_evidence(item)
            scored.append((score, _item_time(item) if _item_time(item) is not None else 1e9, uncertain, item))
        scored.sort(key=lambda row: (row[2], row[1], -row[0]))
        if not scored:
            lines.append(f"- event {event_id} `{event_text}`: not covered yet")
            continue
        first_time = scored[0][1]
        first_text = f"{first_time:.1f}s" if first_time != 1e9 else "unknown time"
        lines.append(
            f"- event {event_id} `{event_text}`: first_{'uncertain' if scored[0][2] else 'confirmed'}={first_text}; "
            f"matches={len(scored)}"
        )
        for _score, _time, uncertain, item in scored[:max_hits]:
            status = "uncertain" if uncertain else "confirmed"
            lines.append(f"  {status} evidence: {_format_item_ref(item)}")
    lines.append(
        "Timeline note: if a required event is not covered or only negatively mentioned, seek that event before final ordering."
    )
    return lines


def _build_occurrence_groups(memory: dict[str, Any], question: str | None) -> list[str]:
    if not _is_count_question(question):
        return []
    target = _extract_count_target(question)
    if not target:
        return []
    if not _keywords(target):
        return []
    matches: list[tuple[float, float, dict[str, Any]]] = []
    for item in _evidence_items(memory):
        if not _is_positive_candidate(item, target):
            continue
        window = _item_window(item)
        if window is None:
            continue
        if item.get("_kind") == "summary" and (window[1] - window[0]) > 45.0:
            continue
        matches.append((window[0], window[1], item))
    matches.sort(key=lambda row: row[0])
    if not matches:
        return [f"Task-aware occurrence groups for `{target}`: not covered yet"]

    groups: list[dict[str, Any]] = []
    merge_gap_s = 8.0
    for start, end, item in matches:
        if not groups or start - groups[-1]["end"] > merge_gap_s:
            groups.append({"start": start, "end": end, "items": [item]})
        else:
            groups[-1]["end"] = max(groups[-1]["end"], end)
            groups[-1]["items"].append(item)

    lines = [f"Task-aware occurrence groups for `{target}`:"]
    for idx, group in enumerate(groups, start=1):
        example = _format_item_ref(group["items"][0])
        confirmed = any(not _is_uncertain_evidence(item) for item in group["items"])
        status = "confirmed" if confirmed else "uncertain"
        lines.append(
            f"- occurrence_{idx}: status={status} [{group['start']:.1f}, {group['end']:.1f}] "
            f"evidence_count={len(group['items'])}; example={example}"
        )
    lines.append(
        "Counting note: count confirmed occurrence groups, not frames; uncertain groups require more evidence before they are counted."
    )
    return lines


def _build_object_state(memory: dict[str, Any], question: str | None) -> list[str]:
    target = _extract_object_state_target(question)
    if not target:
        return []
    choices = _extract_choices(question)
    location_terms = [(label, text, _keywords(text)) for label, text in choices]
    items = _evidence_items(memory)
    lines = [f"Task-aware object-state candidates for `{target}`:"]
    target_terms = _keywords(target)
    for item in sorted(items, key=lambda it: _item_time(it) or 1e9):
        text = _positive_text_for_memory_item(item)
        text_terms = _keywords(text)
        if target_terms and not (target_terms & text_terms):
            if not any(terms & text_terms for _label, _choice, terms in location_terms):
                continue
        option_hits = [
            label
            for label, _choice, terms in location_terms
            if terms and terms & text_terms
        ]
        option_text = f" option_hits={option_hits}" if option_hits else ""
        lines.append(f"- {_format_item_ref(item)}{option_text}")
        if len(lines) >= 8:
            break
    if len(lines) == 1:
        lines.append("- not covered yet")
    return lines


def format_task_trajectory_memory(
    memory: dict[str, Any] | None,
    question: str | None,
    *,
    max_lines: int = 28,
    include_occurrence_groups: bool = False,
    include_timeline_view: bool = True,
    include_object_state_view: bool = True,
) -> str:
    if not memory or not question:
        return ""
    sections: list[str] = []
    if include_timeline_view and _is_order_question(question):
        sections.extend(_build_event_timeline(memory, question))
    # Occurrence grouping is useful for offline diagnosis, but the current
    # prompt-level version was unstable on count questions, so keep it opt-in.
    if include_occurrence_groups and _is_count_question(question):
        sections.extend(_build_occurrence_groups(memory, question))
    if include_object_state_view:
        sections.extend(_build_object_state(memory, question))
    if not sections:
        return ""
    return "\n".join(sections[:max_lines])


def extract_v10_payload(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None

    # Best-effort fallback for tool outputs that are only JSON.
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except Exception:
            return None
    return None


def _memory_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:05d}"


def _short_text(value: Any, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip(" ,.;:") + "..."


def _safe_window(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start = _safe_float(value[0])
    end = _safe_float(value[1])
    if start is None or end is None or end < start:
        return None
    return [round(start, 1), round(end, 1)]


def _params_window(parameters: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(parameters, dict):
        return None
    start = _safe_float(parameters.get("start_time"))
    end = _safe_float(parameters.get("end_time"))
    if start is None or end is None or end < start:
        return None
    return [round(start, 1), round(end, 1)]


def _window_label(window: Any) -> str:
    span = _safe_window(window)
    if span is None:
        return "unknown"
    return f"[{span[0]:.1f}, {span[1]:.1f}]"


def _evidence_level_for_tool(
    tool_name: str,
    detail_sufficient: Any = None,
    decision_sufficient: Any = None,
) -> str:
    if tool_name in {"overview", "skim_qwen"}:
        return "routing"
    if tool_name in {"focus_qwen", "localize_qwen"}:
        return "candidate"
    if tool_name in {"frame_verify", "focus"} and (
        detail_sufficient is True or decision_sufficient is True
    ):
        return "verified"
    if tool_name in {"frame_verify", "focus", "skim"}:
        return "candidate"
    return "candidate"


def _dedupe_append(
    items: list[dict[str, Any]],
    item: dict[str, Any],
    seen: set[tuple[Any, ...]],
    key: tuple[Any, ...],
) -> None:
    if key in seen:
        return
    seen.add(key)
    items.append(item)


def _evidence_item_by_id(memory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    state = ensure_structured_evidence_state(memory)
    out: dict[str, dict[str, Any]] = {}
    for item in state.get("evidence_items") or []:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("evidence_id") or "")
        if eid:
            out[eid] = item
    return out


_EVENT_TERM_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "into", "onto",
    "what", "when", "where", "which", "who", "why", "how", "does", "did",
    "was", "were", "are", "is", "after", "before", "then", "there", "video",
    "shown", "show", "actual", "following", "event", "events", "option",
}


_EVENT_TERM_CANONICAL = {
    "guy": "man",
    "guys": "man",
    "men": "man",
    "lady": "woman",
    "female": "woman",
    "girls": "girl",
    "sits": "sit",
    "sitting": "sit",
    "seated": "sit",
    "talks": "talk",
    "talking": "talk",
    "speaks": "talk",
    "speaking": "talk",
    "surfing": "surf",
    "surfer": "surf",
    "surfs": "surf",
    "spins": "spin",
    "spinning": "spin",
    "skates": "skate",
    "skating": "skate",
    "skater": "skate",
    "credits": "credit",
    "indoors": "indoor",
    "inside": "indoor",
}


def _event_terms(text: Any) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(text or "").lower())
    terms: set[str] = set()
    for token in tokens:
        if token in _EVENT_TERM_STOPWORDS:
            continue
        canonical = _EVENT_TERM_CANONICAL.get(token)
        if canonical:
            terms.add(canonical)
            continue
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        terms.add(_EVENT_TERM_CANONICAL.get(token, token))
    if "interview" in terms:
        terms.update({"talk", "sit", "indoor"})
    return terms


def _question_event_facets(question: str | None, *, max_items: int) -> list[dict[str, Any]]:
    text = str(question or "")
    stem = re.split(r"(?m)^\s*\([A-Z]\)\s+", text, maxsplit=1)[0]
    numbered = re.findall(
        r"\(\d+\)\s*(.+?)(?=(?:\s*;\s*)?\(\d+\)|\.?\s*$)",
        stem,
        flags=re.DOTALL,
    )
    labels = [re.sub(r"\s+", " ", item).strip(" ;,.") for item in numbered]

    if len(labels) < 2:
        labels = []
        for _letter, choice in _extract_choices(text):
            if not re.search(r"-->|->|→", choice):
                continue
            labels.extend(
                re.sub(r"\s+", " ", item).strip(" ;,.")
                for item in re.split(r"\s*(?:-->|->|→)\s*", choice)
                if item.strip()
            )

    facets: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for label in labels:
        terms = _event_terms(label)
        if not terms:
            continue
        key = tuple(sorted(terms))
        if key in seen:
            continue
        seen.add(key)
        facets.append(
            {
                "event_id": f"EVT{len(facets) + 1:02d}",
                "event": label[:160],
                "terms": sorted(terms),
            }
        )
        if len(facets) >= max(2, int(max_items)):
            break
    return facets if len(facets) >= 2 else []


def _event_evidence_text(item: dict[str, Any]) -> str:
    # ``observed_fact`` can summarize a whole multi-window verifier call and
    # mention several unrelated events. Prefer the timestamp-local description
    # so one frame cannot satisfy every event facet in the question.
    description = str(item.get("description") or "").strip()
    if description:
        return description
    return " ".join(
        str(value or "")
        for value in (
            item.get("target_entity_or_event"),
            " ".join(str(tag) for tag in item.get("event_tags") or []),
            item.get("observed_fact"),
        )
        if value
    )


def _event_evidence_is_negated(text: str, facet_terms: set[str]) -> bool:
    lowered = str(text or "").lower()
    if not re.search(r"\b(?:no|not|without|never|neither)\b", lowered):
        return False
    for term in facet_terms:
        variants = {term}
        if term == "credit":
            variants.add("credits")
        elif term == "skate":
            variants.update({"skater", "skates", "skating"})
        pattern = "|".join(re.escape(value) for value in sorted(variants))
        if re.search(
            rf"\b(?:no|not|without|never|neither)\b[^.;]{{0,72}}\b(?:{pattern})\b",
            lowered,
        ):
            return True
    return False


def _evidence_timestamp(item: dict[str, Any]) -> float | None:
    # A broad window start is not the event timestamp. Only use observations
    # that already carry an explicit frame/anchor time.
    return _safe_float(item.get("timestamp_s"))


def build_event_coverage(
    memory: dict[str, Any] | None,
    *,
    question: str | None,
    max_items: int = 8,
) -> list[dict[str, Any]]:
    """Project existing evidence into one compact record per explicit event.

    The strongest evidence and the earliest matching anchor are intentionally
    separate. Order questions need the first observed occurrence, while the
    strongest verifier evidence may refer to a later repeat of the event.
    """
    if not memory:
        return []
    facets = _question_event_facets(question, max_items=max_items)
    if not facets:
        memory["event_coverage"] = []
        return []

    state = ensure_structured_evidence_state(memory)
    evidence_items = [
        item
        for item in state.get("evidence_items") or []
        if isinstance(item, dict) and str(item.get("description") or "").strip()
    ]
    level_rank = {"verified": 4, "candidate": 3, "routing": 2, "uncertain": 1}
    coverage: list[dict[str, Any]] = []
    for facet in facets:
        facet_terms = set(facet["terms"])
        minimum_match = 1 if len(facet_terms) <= 2 else 2
        ranked: list[tuple[tuple[Any, ...], dict[str, Any], set[str]]] = []
        for item in evidence_items:
            evidence_text = _event_evidence_text(item)
            if _evidence_timestamp(item) is None:
                continue
            if _event_evidence_is_negated(evidence_text, facet_terms):
                continue
            evidence_terms = _event_terms(evidence_text)
            matched = facet_terms & evidence_terms
            if len(matched) < minimum_match:
                continue
            level = str(item.get("evidence_level") or "uncertain")
            precision = len(matched) / max(1, len(evidence_terms))
            score = (
                level_rank.get(level, 0),
                len(matched),
                precision,
                item.get("detail_sufficient") is True,
                _evidence_timestamp(item) is not None,
            )
            ranked.append((score, item, matched))

        if not ranked:
            coverage.append(
                {
                    **facet,
                    "status": "unobserved",
                    "timestamp_s": None,
                    "t_range": None,
                    "evidence_ref": None,
                    "source": None,
                    "fact": "",
                    "matched_terms": [],
                    "earliest_status": "unobserved",
                    "earliest_timestamp_s": None,
                    "earliest_evidence_ref": None,
                    "earliest_fact": "",
                }
            )
            continue

        _score, best, matched = max(ranked, key=lambda row: row[0])
        _earliest_score, earliest, earliest_matched = min(
            ranked,
            key=lambda row: (
                _evidence_timestamp(row[1]),
                -len(row[2]),
                -level_rank.get(str(row[1].get("evidence_level") or "uncertain"), 0),
            ),
        )
        coverage.append(
            {
                **facet,
                "status": str(best.get("evidence_level") or "uncertain"),
                "timestamp_s": _evidence_timestamp(best),
                "t_range": _safe_window(best.get("t_range")),
                "evidence_ref": best.get("evidence_id"),
                "source": f"{best.get('source_tool') or 'tool'}/{best.get('backend') or '?'}",
                "fact": _short_text(best.get("description"), 180),
                "matched_terms": sorted(matched),
                "earliest_status": str(earliest.get("evidence_level") or "uncertain"),
                "earliest_timestamp_s": _evidence_timestamp(earliest),
                "earliest_evidence_ref": earliest.get("evidence_id"),
                "earliest_source": (
                    f"{earliest.get('source_tool') or 'tool'}/{earliest.get('backend') or '?'}"
                ),
                "earliest_fact": _short_text(earliest.get("description"), 160),
                "earliest_matched_terms": sorted(earliest_matched),
            }
        )

    memory["event_coverage"] = deepcopy(coverage)
    return coverage


def format_event_coverage_for_prompt(
    memory: dict[str, Any] | None,
    *,
    question: str | None,
    max_items: int = 8,
) -> str:
    coverage = build_event_coverage(memory, question=question, max_items=max_items)
    if not coverage:
        return ""
    lines = [
        "Compact Event Coverage (read-only projection of existing evidence):",
        "For each explicit event, earliest is the first matching timestamp anchor and strongest is the best-supported local evidence. This view does not force or block an answer.",
    ]
    for item in coverage:
        timestamp = item.get("timestamp_s")
        when = f"{float(timestamp):.1f}s" if isinstance(timestamp, (int, float)) else _window_label(item.get("t_range"))
        ref = item.get("evidence_ref") or "none"
        source = item.get("source") or "none"
        fact = f" => {item.get('fact')}" if item.get("fact") else ""
        earliest_timestamp = item.get("earliest_timestamp_s")
        earliest_when = (
            f"{float(earliest_timestamp):.1f}s"
            if isinstance(earliest_timestamp, (int, float))
            else "unknown"
        )
        earliest_ref = item.get("earliest_evidence_ref") or "none"
        earliest_status = item.get("earliest_status") or "unobserved"
        lines.append(
            f"- {item.get('event_id')} {item.get('event')}: "
            f"earliest={earliest_status}@{earliest_when} ref={earliest_ref}; "
            f"strongest={item.get('status')}@{when} ref={ref} source={source}{fact}"
        )
    return "\n".join(lines)


_CONTINUITY_EVIDENCE_RE = re.compile(
    r"\b(same (?:continuous )?(?:event|occurrence|scene|sequence)|"
    r"continuation|continues?|uninterrupted|identical setup|same event_group)\b",
    re.IGNORECASE,
)

_NEGATED_EVENT_EVIDENCE_RE = re.compile(
    r"(?:\b(?:no|none|zero|without)\b[^.;]{0,90}"
    r"\b(?:action|activity|contact|event|interact\w*|occurrence|grappl\w*|"
    r"shov\w*|tussl\w*)\b|"
    r"\b(?:action|activity|event|interact\w*|occurrence)\b[^.;]{0,45}"
    r"\b(?:absent|does not occur|is not shown|not present|not visible)\b)",
    re.IGNORECASE,
)


def _event_span_overlap_ratio(left: Any, right: Any) -> float:
    left_span = _safe_window(left)
    right_span = _safe_window(right)
    if left_span is None or right_span is None:
        return 0.0
    overlap = _overlap_duration(left_span, right_span)
    shorter = min(
        max(0.001, left_span[1] - left_span[0]),
        max(0.001, right_span[1] - right_span[0]),
    )
    return overlap / shorter


def _global_event_group_id(verification_id: str, local_group_id: Any) -> str:
    local = str(local_group_id or "").strip()
    if not local:
        return ""
    prefix = f"{verification_id}:"
    return local if local.startswith(prefix) else prefix + local


def _verification_fact_polarity(row: dict[str, Any]) -> str:
    event_match = str(row.get("event_match") or "unknown").strip().lower()
    target_match = str(row.get("target_match") or "unknown").strip().lower()
    if event_match == "direct":
        return "direct"
    if event_match in {"context_only", "different_event", "not_visible"}:
        return "not_direct"
    if target_match in {"mismatch", "not_visible"}:
        return "not_direct"
    return "uncertain"


def _record_verification_facts(
    memory: dict[str, Any],
    *,
    verification_id: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    """Audit contradictory observations over the same pixels without routing."""
    facts = memory.setdefault("verification_facts", [])
    conflicts = memory.setdefault("observation_conflicts", [])
    conflict_ids: list[str] = []
    for row in rows:
        span = _safe_window(row.get("t_range"))
        if span is None:
            continue
        normalized = {
            "fact_id": _memory_id("vf", len(facts) + 1),
            "verification_id": verification_id,
            "candidate_id": str(row.get("candidate_id") or ""),
            "t_range": span,
            "best_timestamp_s": _safe_float(row.get("best_timestamp_s")),
            "target_match": str(row.get("target_match") or "unknown"),
            "event_match": str(row.get("event_match") or "unknown"),
            "observed_fact": _short_text(row.get("observed_fact"), 260),
            "event_group_id": str(row.get("event_group_id") or ""),
        }
        polarity = _verification_fact_polarity(normalized)
        for previous in facts:
            if previous.get("verification_id") == verification_id:
                continue
            previous_polarity = _verification_fact_polarity(previous)
            if {polarity, previous_polarity} != {"direct", "not_direct"}:
                continue
            if _event_span_overlap_ratio(span, previous.get("t_range")) < 0.5:
                continue
            conflict_id = _memory_id("conflict", len(conflicts) + 1)
            conflicts.append(
                {
                    "conflict_id": conflict_id,
                    "conflict_type": "overlapping_direct_vs_non_direct",
                    "t_range": [
                        round(max(span[0], previous["t_range"][0]), 3),
                        round(min(span[1], previous["t_range"][1]), 3),
                    ],
                    "fact_refs": [previous.get("fact_id"), normalized["fact_id"]],
                    "verification_ids": [
                        previous.get("verification_id"),
                        verification_id,
                    ],
                    "facts": [
                        previous.get("observed_fact") or "",
                        normalized.get("observed_fact") or "",
                    ],
                }
            )
            conflict_ids.append(conflict_id)
        facts.append(normalized)
    return conflict_ids


def _ledger_fact_terms(value: Any) -> set[str]:
    return _keywords(str(value or ""))


def _ledger_rows_describe_continuity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            left.get("fact"),
            left.get("binding_reason"),
            right.get("fact"),
            right.get("binding_reason"),
        )
    )
    return bool(_CONTINUITY_EVIDENCE_RE.search(text))


def _ledger_rows_share_episode(left: dict[str, Any], right: dict[str, Any]) -> bool:
    overlap_ratio = _event_span_overlap_ratio(left.get("t_range"), right.get("t_range"))
    if overlap_ratio >= 0.6:
        return True

    left_span = _safe_window(left.get("t_range"))
    right_span = _safe_window(right.get("t_range"))
    if left_span is None or right_span is None:
        return False
    gap = max(0.0, max(left_span[0], right_span[0]) - min(left_span[1], right_span[1]))
    if gap <= 8.0 and _ledger_rows_describe_continuity(left, right):
        return True
    if gap <= 1.0:
        left_terms = _ledger_fact_terms(left.get("fact"))
        right_terms = _ledger_fact_terms(right.get("fact"))
        if left_terms and right_terms and left_terms & right_terms:
            return True

    same_call_group = bool(
        left.get("verification_id")
        and left.get("verification_id") == right.get("verification_id")
        and left.get("event_group_id")
        and left.get("event_group_id") == right.get("event_group_id")
    )
    return same_call_group and _ledger_rows_describe_continuity(left, right)


def _merge_ledger_event(event: dict[str, Any], row: dict[str, Any]) -> None:
    spans = [
        span
        for span in (
            _safe_window(event.get("t_range")),
            _safe_window(row.get("t_range")),
        )
        if span is not None
    ]
    if spans:
        event["t_range"] = [
            round(min(span[0] for span in spans), 3),
            round(max(span[1] for span in spans), 3),
        ]
    anchor = _safe_float(row.get("best_timestamp_s"))
    anchors = event.setdefault("anchor_timestamps", [])
    if anchor is not None and round(anchor, 3) not in anchors:
        anchors.append(round(anchor, 3))
        anchors.sort()
    for field in ("source_refs", "verification_ids", "candidate_ids", "supports_options", "contradicts_options"):
        values = row.get(field) or []
        target = event.setdefault(field, [])
        for value in values:
            if value and value not in target:
                target.append(value)
    facts = event.setdefault("facts", [])
    fact = _short_text(row.get("fact"), 260)
    if fact and fact not in facts:
        facts.append(fact)
    event["observation_count"] = int(event.get("observation_count") or 0) + 1


def _ledger_row_category(row: dict[str, Any], option_letters="ABCD") -> str:
    """Separate verified occurrence evidence from verified local absence.

    ``target_match`` answers whether the named entity/location is bound, while
    ``event_match`` answers whether the requested event actually occurs. Older
    verifier payloads may not contain the latter, so direct option support and
    explicit local-absence language provide a conservative compatibility path.
    """
    target_match = str(row.get("target_match") or "unknown").strip().lower()
    event_match = str(row.get("event_match") or "unknown").strip().lower()
    if event_match == "direct":
        return "event"
    if event_match in {
        "context_only",
        "different_event",
        "not_visible",
        "ambiguous",
    }:
        return "boundary"
    if target_match in {"mismatch", "not_visible", "ambiguous", "partial"}:
        return "boundary"
    if target_match != "matched":
        return "boundary"
    if row.get("supports_options"):
        return "event"
    contradicts = set(row.get("contradicts_options") or [])
    if set(option_letters).issubset(contradicts):
        return "boundary"
    text = " ".join(
        str(row.get(field) or "")
        for field in ("fact", "binding_reason")
    )
    if _NEGATED_EVENT_EVIDENCE_RE.search(text):
        return "boundary"
    return "event"


def _ledger_decision_signal(event: dict[str, Any], option_letters="ABCD") -> str:
    supports = set(event.get("supports_options") or [])
    contradicts = set(event.get("contradicts_options") or [])
    if len(supports) == 1 and (set(option_letters) - supports) <= contradicts:
        return "option_discriminative"
    if supports:
        return "option_support"
    return "visual_fact_only"


def _merge_ledger_boundary(boundary: dict[str, Any], row: dict[str, Any]) -> None:
    spans = [
        span
        for span in (
            _safe_window(boundary.get("t_range")),
            _safe_window(row.get("t_range")),
        )
        if span is not None
    ]
    if spans:
        boundary["t_range"] = [
            round(min(span[0] for span in spans), 3),
            round(max(span[1] for span in spans), 3),
        ]
    for field in ("source_refs", "verification_ids"):
        target = boundary.setdefault(field, [])
        for value in row.get(field) or []:
            if value and value not in target:
                target.append(value)
    facts = boundary.setdefault("facts", [])
    fact = _short_text(row.get("fact"), 220)
    if fact and fact not in facts:
        facts.append(fact)
    boundary["observation_count"] = int(boundary.get("observation_count") or 0) + 1


def _ledger_boundary_status(row: dict[str, Any]) -> str:
    event_match = str(row.get("event_match") or "unknown").strip().lower()
    if event_match not in {"", "unknown"}:
        return event_match
    target_match = str(row.get("target_match") or "unknown").strip().lower()
    text = " ".join(
        str(row.get(field) or "")
        for field in ("fact", "binding_reason")
    )
    if target_match == "matched" and _NEGATED_EVENT_EVIDENCE_RE.search(text):
        return "event_absent"
    if target_match == "partial":
        return "context_only"
    return target_match


def _ledger_contextual_candidate(row: dict[str, Any]) -> bool:
    """Keep relevant context distinct from both events and hard negatives.

    A verifier can see the requested activity's setup, participants, or
    materials while missing the exact action at the sampled timestamps.  Such
    evidence should not inflate the verified event count, but discarding it as
    a boundary makes the observed coverage look more certain than it is.
    """
    event_match = str(row.get("event_match") or "unknown").strip().lower()
    target_match = str(row.get("target_match") or "unknown").strip().lower()
    if event_match == "context_only":
        return target_match in {"matched", "partial", "unknown"}
    return event_match == "ambiguous" and target_match in {"matched", "partial"}


def build_temporal_evidence_ledger(
    memory: dict[str, Any] | None,
    *,
    max_events: int = 16,
    max_boundaries: int = 8,
) -> dict[str, Any]:
    """Deduplicate verified event spans without deciding the answer.

    The ledger is a read-only projection over existing frame-verifier output.
    It preserves earlier positive evidence when a later observation is vague,
    and keeps temporally separate matches separate unless the verifier explicitly
    describes continuity. It never invokes a tool or blocks an answer.
    """
    if not memory:
        return {}

    option_letters = [c["letter"] for c in (memory.get("structured_evidence") or {}).get("choices", [])] or "ABCD"
    positive_rows: list[dict[str, Any]] = []
    contextual_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    tool_observations = [
        item
        for item in memory.get("tool_observations") or []
        if isinstance(item, dict)
    ]
    candidate_event_context: dict[tuple[str, str], str] = {}
    for observation in tool_observations:
        verification_id = str(observation.get("verification_id") or "")
        if not verification_id:
            continue
        for assessment in observation.get("candidate_assessments") or []:
            if not isinstance(assessment, dict):
                continue
            candidate_id = str(assessment.get("candidate_id") or "")
            if candidate_id:
                candidate_event_context[(verification_id, candidate_id)] = str(
                    assessment.get("event_group_id") or ""
                )
    for item in memory.get("candidate_binding_memory") or []:
        if not isinstance(item, dict) or item.get("source_tool") != "frame_verify":
            continue
        span = _safe_window(item.get("t_range"))
        if span is None:
            continue
        target_match = str(item.get("target_match") or "unknown").strip().lower()
        event_match = str(item.get("event_match") or "unknown").strip().lower()
        row = {
            "t_range": span,
            "best_timestamp_s": item.get("best_timestamp_s"),
            "fact": item.get("observed_fact") or item.get("routing_summary") or "",
            "binding_reason": item.get("target_binding_reason") or "",
            "source_refs": [item.get("id")],
            "verification_ids": [item.get("verification_id")],
            "candidate_ids": [item.get("candidate_id")],
            "verification_id": item.get("verification_id"),
            "event_group_id": item.get("event_group_id")
            or candidate_event_context.get(
                (
                    str(item.get("verification_id") or ""),
                    str(item.get("candidate_id") or ""),
                )
            )
            or "",
            "supports_options": item.get("supports_options") or [],
            "contradicts_options": item.get("contradicts_options") or [],
            "target_match": target_match,
            "event_match": event_match,
        }
        category = _ledger_row_category(row, option_letters)
        if category == "event":
            positive_rows.append(row)
        elif _ledger_contextual_candidate(row):
            contextual_rows.append(row)
        else:
            boundary_rows.append(row)

    binding_verifications = {
        str(item.get("verification_id") or "")
        for item in memory.get("candidate_binding_memory") or []
        if isinstance(item, dict)
        and item.get("verification_id")
        and _safe_window(item.get("t_range")) is not None
    }
    ranged_binding_candidate_ids = {
        str(item.get("candidate_id") or "")
        for item in memory.get("candidate_binding_memory") or []
        if isinstance(item, dict)
        and item.get("candidate_id")
        and _safe_window(item.get("t_range")) is not None
    }
    for item in tool_observations:
        if item.get("tool") != "frame_verify":
            continue
        verification_id = str(item.get("verification_id") or "")
        assessment_ids = {
            str(assessment.get("candidate_id") or "")
            for assessment in item.get("candidate_assessments") or []
            if isinstance(assessment, dict) and assessment.get("candidate_id")
        }
        parameters = (
            item.get("parameters")
            if isinstance(item.get("parameters"), dict)
            else {}
        )
        if parameters.get("candidate_windows") and not assessment_ids:
            # P124: a multi-window call range is transport coverage, not an
            # event span. Without per-candidate binding there is no stable
            # occurrence that the ledger may safely materialize.
            continue
        if verification_id in binding_verifications or (
            assessment_ids and assessment_ids <= ranged_binding_candidate_ids
        ):
            continue
        span = _safe_window(item.get("t_range"))
        if span is None:
            span = _safe_window(
                [parameters.get("start_time"), parameters.get("end_time")]
            )
        if span is None:
            continue
        target_match = str(item.get("target_match") or "unknown").strip().lower()
        event_match = str(item.get("target_event_match") or "unknown").strip().lower()
        row = {
            "t_range": span,
            "best_timestamp_s": None,
            "fact": item.get("observed_fact") or item.get("summary") or "",
            "binding_reason": item.get("target_binding_reason") or "",
            "source_refs": [verification_id],
            "verification_ids": [verification_id],
            "candidate_ids": [],
            "verification_id": verification_id,
            "event_group_id": "",
            "supports_options": item.get("supports_options") or [],
            "contradicts_options": item.get("contradicts_options") or [],
            "target_match": target_match,
            "event_match": event_match,
        }
        category = _ledger_row_category(row, option_letters)
        if category == "event":
            positive_rows.append(row)
        elif _ledger_contextual_candidate(row):
            contextual_rows.append(row)
        else:
            boundary_rows.append(row)

    events: list[dict[str, Any]] = []
    for row in sorted(positive_rows, key=lambda item: item["t_range"]):
        matching = next(
            (event for event in events if _ledger_rows_share_episode(event, row)),
            None,
        )
        if matching is None:
            matching = {
                "event_id": f"TE{len(events) + 1:03d}",
                "t_range": list(row["t_range"]),
                "anchor_timestamps": [],
                "facts": [],
                "source_refs": [],
                "verification_ids": [],
                "candidate_ids": [],
                "supports_options": [],
                "contradicts_options": [],
                "observation_count": 0,
                "verification_id": row.get("verification_id"),
                "event_group_id": row.get("event_group_id"),
                "fact": row.get("fact") or "",
                "binding_reason": row.get("binding_reason") or "",
            }
            events.append(matching)
        _merge_ledger_event(matching, row)

    for event in events:
        event["verification_count"] = len(event.get("verification_ids") or [])
        event["fact"] = " | ".join(event.pop("facts", [])[:3])
        event["decision_signal"] = _ledger_decision_signal(event, option_letters)

    boundary_clusters: list[dict[str, Any]] = []
    for row in sorted(boundary_rows, key=lambda item: item["t_range"]):
        status = _ledger_boundary_status(row)
        matching = next(
            (
                item
                for item in boundary_clusters
                if item.get("status") == status
                and _event_span_overlap_ratio(item.get("t_range"), row.get("t_range"))
                >= 0.6
            ),
            None,
        )
        if matching is None:
            matching = {
                "t_range": list(row["t_range"]),
                "status": status,
                "facts": [],
                "source_refs": [],
                "verification_ids": [],
                "observation_count": 0,
            }
            boundary_clusters.append(matching)
        _merge_ledger_boundary(matching, row)

    boundaries: list[dict[str, Any]] = []
    for boundary in boundary_clusters:
        span = boundary["t_range"]
        overlapping = [
            event["event_id"]
            for event in events
            if _event_span_overlap_ratio(event.get("t_range"), span) >= 0.5
        ]
        boundaries.append(
            {
                "t_range": span,
                "status": boundary.get("status"),
                "fact": " | ".join(boundary.pop("facts", [])[:2]),
                "overlaps_verified_events": overlapping,
                "source_refs": boundary.get("source_refs") or [],
                "verification_count": len(boundary.get("verification_ids") or []),
            }
        )

    contextual_clusters: list[dict[str, Any]] = []
    for row in sorted(contextual_rows, key=lambda item: item["t_range"]):
        matching = next(
            (
                item
                for item in contextual_clusters
                if _event_span_overlap_ratio(item.get("t_range"), row.get("t_range"))
                >= 0.6
            ),
            None,
        )
        if matching is None:
            matching = {
                "context_id": f"TC{len(contextual_clusters) + 1:03d}",
                "t_range": list(row["t_range"]),
                "facts": [],
                "source_refs": [],
                "verification_ids": [],
                "observation_count": 0,
            }
            contextual_clusters.append(matching)
        _merge_ledger_boundary(matching, row)

    contextual_candidates: list[dict[str, Any]] = []
    for item in contextual_clusters:
        span = item["t_range"]
        overlapping = [
            event["event_id"]
            for event in events
            if (
                (_safe_window(event.get("t_range")) is not None)
                and max(
                    _safe_window(event.get("t_range"))[0],
                    span[0],
                )
                <= min(
                    _safe_window(event.get("t_range"))[1],
                    span[1],
                )
            )
        ]
        contextual_candidates.append(
            {
                "context_id": item.get("context_id"),
                "t_range": span,
                "fact": " | ".join(item.pop("facts", [])[:2]),
                "overlaps_verified_events": overlapping,
                "source_refs": item.get("source_refs") or [],
                "verification_count": len(item.get("verification_ids") or []),
            }
        )

    distinct_contextual_count = sum(
        1
        for item in contextual_candidates
        if not item.get("overlaps_verified_events")
    )

    reuse_stats = memory.get("verified_observation_reuse_stats") or {}
    ledger = {
        "version": "p21_temporal_evidence_ledger_v2",
        "stable_event_span_count": len(events),
        "plausible_observed_span_range": [
            len(events),
            len(events) + distinct_contextual_count,
        ],
        "events": events[: max(1, int(max_events))],
        "contextual_candidates": contextual_candidates[: max(1, int(max_boundaries))],
        "boundary_observations": (
            boundaries[-max(1, int(max_boundaries)) :]
            if int(max_boundaries) > 0
            else []
        ),
        "verified_observation_reuse_count": int(reuse_stats.get("reuse_count") or 0),
    }
    memory["temporal_evidence_ledger"] = deepcopy(ledger)
    return ledger


def format_temporal_evidence_ledger_for_prompt(
    memory: dict[str, Any] | None,
    *,
    max_events: int = 16,
    max_boundaries: int = 8,
) -> str:
    ledger = build_temporal_evidence_ledger(
        memory,
        max_events=max_events,
        max_boundaries=max_boundaries,
    )
    events = ledger.get("events") or []
    contextual_candidates = ledger.get("contextual_candidates") or []
    boundaries = ledger.get("boundary_observations") or []
    reuse_count = int(ledger.get("verified_observation_reuse_count") or 0)
    if not events and not contextual_candidates and not boundaries and reuse_count == 0:
        return ""
    lines = [
        "Temporal Evidence Ledger (stable read-only projection):",
        "Each TE row is one deduplicated verified event span. The row count is observed coverage, not automatically the answer; uninspected events may still exist.",
        f"Observed span range={ledger.get('plausible_observed_span_range') or [len(events), len(events)]}: lower bound counts direct verified events; upper bound also includes distinct context-only candidates and is not a final count.",
    ]
    for event in events:
        lines.append(
            f"- {event.get('event_id')} {_window_label(event.get('t_range'))} "
            f"anchors={event.get('anchor_timestamps') or []} "
            f"verifications={event.get('verification_count') or 0}: "
            f"{_short_text(event.get('fact'), 260)} "
            f"decision_signal={event.get('decision_signal') or 'visual_fact_only'} "
            f"support={event.get('supports_options') or []} "
            f"contradict={event.get('contradicts_options') or []}"
        )
    if contextual_candidates:
        lines.append(
            "Contextual candidates (relevant setup/participants/materials, but the exact event is not verified):"
        )
        for item in contextual_candidates:
            lines.append(
                f"- {item.get('context_id')} {_window_label(item.get('t_range'))} "
                f"verifications={item.get('verification_count') or 0} "
                f"overlaps={item.get('overlaps_verified_events') or []}: "
                f"{_short_text(item.get('fact'), 180)}"
            )
    if boundaries:
        lines.append("Local non-match/conflict boundaries (do not erase verified TE rows):")
        for item in boundaries:
            lines.append(
                f"- {_window_label(item.get('t_range'))} status={item.get('status')} "
                f"verifications={item.get('verification_count') or 0} "
                f"overlaps={item.get('overlaps_verified_events') or []}: "
                f"{_short_text(item.get('fact'), 180)}"
            )
    if reuse_count:
        lines.append(
            f"Verified observation cache: {reuse_count} repeated visual request(s) reused; "
            "those calls added no new pixels or evidence."
        )
    return "\n".join(lines)


def _verified_refs(refs: list[Any], by_id: dict[str, dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for ref in refs or []:
        eid = str(ref)
        item = by_id.get(eid)
        if item and item.get("evidence_level") == "verified":
            out.append(eid)
    return out


def _parameter_windows(parameters: dict[str, Any] | None) -> list[list[float]]:
    if not isinstance(parameters, dict):
        return []
    windows: list[list[float]] = []
    for value in parameters.get("windows") or []:
        span = _safe_window(value)
        if span is not None and span not in windows:
            windows.append(span)
    if windows:
        return windows
    span = _params_window(parameters)
    return [span] if span is not None else []


def _overlap_duration(left: Any, right: Any) -> float:
    a = _safe_window(left)
    b = _safe_window(right)
    if a is None or b is None:
        return 0.0
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _merge_intervals(windows: list[list[float]]) -> list[list[float]]:
    valid = sorted(
        (span for span in (_safe_window(item) for item in windows) if span is not None),
        key=lambda span: (span[0], span[1]),
    )
    merged: list[list[float]] = []
    for start, end in valid:
        if not merged or start > merged[-1][1] + 0.1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [[round(start, 1), round(end, 1)] for start, end in merged]


def _window_coverage(target: Any, inspected: list[dict[str, Any]]) -> tuple[list[list[float]], float]:
    span = _safe_window(target)
    if span is None or span[1] <= span[0]:
        return [], 0.0
    clipped: list[list[float]] = []
    for item in inspected:
        observed = _safe_window(item.get("t_range"))
        if observed is None or _overlap_duration(span, observed) <= 0:
            continue
        clipped.append([max(span[0], observed[0]), min(span[1], observed[1])])
    merged = _merge_intervals(clipped)
    covered = sum(max(0.0, end - start) for start, end in merged)
    return merged, round(min(1.0, covered / max(0.1, span[1] - span[0])), 3)


def _representative_timestamp_cues(
    cues: list[dict[str, Any]],
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Keep a compact mix of query cues and temporally distinct scene facts."""
    if limit <= 0:
        return []
    unique: list[dict[str, Any]] = []
    seen: set[tuple[float, str]] = set()
    for cue in sorted(cues, key=lambda item: float(item.get("timestamp_s") or 0.0)):
        key = (
            round(float(cue.get("timestamp_s") or 0.0), 1),
            str(cue.get("description") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(cue)
    if len(unique) <= limit:
        return unique

    ranked = sorted(
        unique,
        key=lambda cue: (
            int(cue.get("query_relevance_score") or 0),
            bool(cue.get("needs_focus")),
            len(_keywords(str(cue.get("description") or ""))),
        ),
        reverse=True,
    )
    selected = ranked[: max(1, limit // 2)]
    remaining = [cue for cue in unique if cue not in selected]
    while remaining and len(selected) < limit:
        cue = max(
            remaining,
            key=lambda item: min(
                abs(
                    float(item.get("timestamp_s") or 0.0)
                    - float(chosen.get("timestamp_s") or 0.0)
                )
                for chosen in selected
            ),
        )
        selected.append(cue)
        remaining.remove(cue)
    return sorted(selected, key=lambda cue: float(cue.get("timestamp_s") or 0.0))


def _window_overlap_ratio(left: Any, right: Any) -> float:
    a = _safe_window(left)
    b = _safe_window(right)
    if a is None or b is None:
        return 0.0
    shortest = min(max(0.1, a[1] - a[0]), max(0.1, b[1] - b[0]))
    return _overlap_duration(a, b) / shortest


def _target_reason_information_score(reason: Any) -> tuple[int, int]:
    text = str(reason or "").strip()
    generic_instruction = bool(
        re.search(r"(?i)\buse\s+(?:frame_verify|focus|skim)|final visual decision", text)
    )
    return (0 if generic_instruction else 1, len(_keywords(text)))


def _compact_search_target_frontier(
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove redundant local targets while preserving distinct search alternatives."""
    compact: list[dict[str, Any]] = []
    routing_scenes: set[str] = set()
    for source in targets:
        item = deepcopy(source)
        target_type = str(item.get("target") or "")
        if target_type == "inspect_overview_candidate":
            scene_id = str(item.get("scene_id") or "")
            if scene_id and scene_id in routing_scenes:
                continue
            if scene_id:
                routing_scenes.add(scene_id)
            compact.append(item)
            continue

        if target_type == "verify_candidate_window":
            duplicate: dict[str, Any] | None = None
            for existing in compact:
                if existing.get("target") != "verify_candidate_window":
                    continue
                if str(existing.get("scene_id") or "") != str(item.get("scene_id") or ""):
                    continue
                if _window_overlap_ratio(existing.get("t_range"), item.get("t_range")) >= 0.65:
                    duplicate = existing
                    break
            if duplicate is None:
                compact.append(item)
                continue
            old_span = _safe_window(duplicate.get("t_range"))
            new_span = _safe_window(item.get("t_range"))
            if old_span is not None and new_span is not None:
                duplicate["t_range"] = [
                    round(min(old_span[0], new_span[0]), 1),
                    round(max(old_span[1], new_span[1]), 1),
                ]
            if _target_reason_information_score(item.get("reason")) > _target_reason_information_score(
                duplicate.get("reason")
            ):
                duplicate["reason"] = item.get("reason")
            continue

        if target_type != "resolve_missing_context":
            compact.append(item)
            continue

        duplicate: dict[str, Any] | None = None
        for existing in compact:
            if existing.get("target") != "resolve_missing_context":
                continue
            if _window_overlap_ratio(existing.get("t_range"), item.get("t_range")) >= 0.65:
                duplicate = existing
                break
        if duplicate is None:
            compact.append(item)
            continue

        old_span = _safe_window(duplicate.get("t_range"))
        new_span = _safe_window(item.get("t_range"))
        if old_span is not None and new_span is not None:
            duplicate["t_range"] = [
                round(min(old_span[0], new_span[0]), 1),
                round(max(old_span[1], new_span[1]), 1),
            ]
        if _target_reason_information_score(item.get("reason")) > _target_reason_information_score(
            duplicate.get("reason")
        ):
            duplicate["reason"] = item.get("reason")
    return compact


def _build_scene_coverage(
    overview_candidates: list[dict[str, Any]],
    inspected_windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for candidate in overview_candidates:
        span = _safe_window(candidate.get("t_range"))
        if span is None:
            continue
        relevant = [
            item
            for item in inspected_windows
            if _overlap_duration(span, item.get("t_range")) > 0
        ]
        intervals, ratio = _window_coverage(span, relevant)
        coverage.append(
            {
                "scene_id": candidate.get("scene_id") or "",
                "t_range": span,
                "summary": _short_text(candidate.get("summary"), 170),
                "missing_detail": _short_text(candidate.get("missing_detail"), 150),
                "routing_possible_evidence": candidate.get("possible_evidence"),
                "covered_intervals": intervals,
                "coverage_ratio": ratio,
                "actual_scene_ids": sorted(
                    {str(item.get("scene_id") or "") for item in relevant if item.get("scene_id")}
                ),
                "tools": sorted({str(item.get("tool") or "") for item in relevant if item.get("tool")}),
                "evidence_found": any(
                    item.get("possible_evidence") is True or item.get("detail_sufficient") is True
                    for item in relevant
                ),
                "negative_observations": sum(item.get("possible_evidence") is False for item in relevant),
                "inspection_count": len(relevant),
            }
        )
    return coverage


def _stable_candidate_id(source: str, scene_id: str, span: list[float]) -> str:
    raw = f"{source}|{scene_id}|{span[0]:.1f}|{span[1]:.1f}"
    return "C" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:7].upper()


def _candidate_status(
    span: list[float],
    inspected_windows: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    status_rank = {
        "unvisited": 0,
        "skimmed": 1,
        "focused": 2,
        "verified_insufficient": 3,
        "verified": 4,
    }
    status = "unvisited"
    observations: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for item in inspected_windows:
        observed = _safe_window(item.get("t_range"))
        if observed is None or _overlap_duration(span, observed) <= 0:
            continue
        tool = str(item.get("tool") or "")
        next_status = "unvisited"
        if tool == "skim_qwen":
            next_status = "skimmed"
        elif tool in {"focus_qwen", "localize_qwen"}:
            next_status = "focused"
        elif tool in {"frame_verify", "focus"}:
            next_status = (
                "verified"
                if item.get("detail_sufficient") is True
                else "verified_insufficient"
            )
        if status_rank[next_status] > status_rank[status]:
            status = next_status
        summary = _short_text(item.get("summary"), 150)
        if summary:
            observations.append(
                {
                    "tool": tool,
                    "t_range": observed,
                    "summary": summary,
                    "detail_sufficient": item.get("detail_sufficient"),
                }
            )
        target_match = str(item.get("target_match") or "").strip().lower()
        if target_match and target_match != "unknown":
            bindings.append(
                {
                    "tool": tool,
                    "t_range": observed,
                    "target": _short_text(item.get("target_entity_or_event"), 100),
                    "target_match": target_match,
                    "reason": _short_text(item.get("target_binding_reason"), 160),
                    "question_scope": item.get("question_scope") or "unknown",
                    "scope_coverage": item.get("scope_coverage") or "unknown",
                    "scope_coverage_reason": _short_text(
                        item.get("scope_coverage_reason"), 160
                    ),
                }
            )
    return status, observations[-3:], bindings[-3:]


def _build_persistent_candidate_pool(
    memory: dict[str, Any],
    *,
    question: str | None,
    inspected_windows: list[dict[str, Any]],
    max_candidates: int,
    timestamp_radius_s: float,
    candidate_evidence_projection: bool = False,
) -> list[dict[str, Any]]:
    """Build a compact, stable pool from overview scenes and timestamp cues.

    The pool is a navigation representation only. It does not create actions,
    suppress candidates, or block answers.
    """
    candidates: list[dict[str, Any]] = []
    scene_candidates: list[dict[str, Any]] = []
    for item in memory.get("scene_memory") or []:
        if not isinstance(item, dict) or str(item.get("source_tool") or "") != "overview":
            continue
        span = _safe_window(item.get("t_range"))
        if span is None:
            continue
        scene_id = str(item.get("scene_id") or "overview")
        summary = _short_text(item.get("summary"), 220)
        missing = _short_text(item.get("missing_detail"), 160)
        query_score = _query_overlap_score(f"{summary} {missing}", question)
        candidate = {
            "candidate_id": _stable_candidate_id("scene", scene_id, span),
            "source": "overview_scene",
            "scene_id": scene_id,
            "t_range": span,
            "summary": summary,
            "possible_evidence": item.get("possible_evidence"),
            "suggest_focus_windows": [
                window
                for window in (
                    _safe_window(value)
                    for value in item.get("suggest_focus_windows") or []
                )
                if window is not None
            ],
            "missing_detail": missing,
            "query_relevance_score": query_score,
            "anchor_timestamps": [],
            "timestamp_cues": [],
            "routing_negative_cue": _has_explicit_negative_routing_cue(item),
        }
        candidates.append(candidate)
        scene_candidates.append(candidate)

    overview_observations = sorted(
        [
            item
            for item in memory.get("timestamped_observations") or []
            if isinstance(item, dict)
            and str(item.get("source_tool") or "") == "overview"
            and _safe_float(item.get("timestamp_s")) is not None
        ],
        key=lambda item: float(item.get("timestamp_s") or 0.0),
    )
    max_timestamp = max(
        [float(item.get("timestamp_s") or 0.0) for item in overview_observations]
        or [0.0]
    )
    radius = max(2.0, float(timestamp_radius_s))
    for item in overview_observations:
        description = str(item.get("description") or "").strip()
        if not description or "observer omitted this sampled frame" in description.lower():
            continue
        tags = [str(value) for value in item.get("event_tags") or []]
        needs_focus = str(item.get("needs_focus") or "").strip()
        cue_text = " ".join([description, " ".join(tags), needs_focus])
        query_score = _query_overlap_score(cue_text, question)
        explicit_focus = bool(needs_focus) and needs_focus.lower() not in {
            "no", "none", "false", "n/a",
        }
        timestamp = round(float(item.get("timestamp_s") or 0.0), 1)
        cue_window = [
            round(max(0.0, timestamp - radius), 1),
            round(min(max_timestamp, timestamp + radius), 1),
        ]
        if cue_window[1] <= cue_window[0]:
            cue_window[1] = round(cue_window[0] + 0.1, 1)
        scene_id = str(item.get("scene_id") or "overview")
        matching = [
            candidate
            for candidate in scene_candidates
            if candidate.get("scene_id") == scene_id
            and candidate["t_range"][0] <= timestamp <= candidate["t_range"][1]
        ]
        if not matching:
            matching = [
                candidate
                for candidate in scene_candidates
                if candidate["t_range"][0] <= timestamp <= candidate["t_range"][1]
            ]
        keep_scene_context = bool(
            candidate_evidence_projection
            and matching
            and any(
                int(candidate.get("query_relevance_score") or 0) > 0
                or candidate.get("possible_evidence") is True
                for candidate in matching
            )
        )
        if query_score <= 0 and not explicit_focus and not keep_scene_context:
            continue
        if matching:
            target = max(
                matching,
                key=lambda candidate: (
                    candidate.get("scene_id") == scene_id,
                    -(candidate["t_range"][1] - candidate["t_range"][0]),
                ),
            )
            if timestamp not in target["anchor_timestamps"]:
                target["anchor_timestamps"].append(timestamp)
            target["timestamp_cues"].append(
                {
                    "timestamp_s": timestamp,
                    "description": _short_text(description, 170),
                    "query_relevance_score": query_score,
                    "needs_focus": _short_text(needs_focus, 100),
                }
            )
            if (query_score > 0 or explicit_focus) and cue_window not in target["suggest_focus_windows"]:
                target["suggest_focus_windows"].append(cue_window)
            target["query_relevance_score"] = max(
                int(target.get("query_relevance_score") or 0), query_score
            )
            if target.get("source") == "overview_scene":
                target["source"] = "overview_scene+timestamp"
            continue

        candidates.append(
            {
                "candidate_id": _stable_candidate_id("timestamp", scene_id, cue_window),
                "source": "overview_timestamp",
                "scene_id": scene_id,
                "t_range": cue_window,
                "summary": _short_text(description, 220),
                "possible_evidence": True if explicit_focus else None,
                "suggest_focus_windows": [cue_window],
                "missing_detail": _short_text(needs_focus, 160),
                "query_relevance_score": query_score,
                "anchor_timestamps": [timestamp],
                "timestamp_cues": [
                    {
                        "timestamp_s": timestamp,
                        "description": _short_text(description, 170),
                        "query_relevance_score": query_score,
                        "needs_focus": _short_text(needs_focus, 100),
                    }
                ],
                "routing_negative_cue": False,
            }
        )

    for candidate in candidates:
        status, observations, bindings = _candidate_status(
            candidate["t_range"], inspected_windows
        )
        candidate["status"] = status
        candidate["observations"] = observations
        candidate["target_bindings"] = bindings
        candidate["anchor_timestamps"] = sorted(candidate.get("anchor_timestamps") or [])[:8]
        if candidate_evidence_projection:
            candidate["timestamp_cues"] = _representative_timestamp_cues(
                candidate.get("timestamp_cues") or [],
                limit=4,
            )
        else:
            candidate["timestamp_cues"] = sorted(
                candidate.get("timestamp_cues") or [],
                key=lambda cue: (
                    int(cue.get("query_relevance_score") or 0),
                    bool(cue.get("needs_focus")),
                ),
                reverse=True,
            )[:4]
        candidate["suggest_focus_windows"] = candidate.get("suggest_focus_windows", [])[:6]

    if candidate_evidence_projection:
        candidates.sort(
            key=lambda candidate: (
                candidate.get("possible_evidence") is True,
                not bool(candidate.get("routing_negative_cue")),
                int(candidate.get("query_relevance_score") or 0),
                bool(candidate.get("timestamp_cues")),
                candidate.get("status") == "unvisited",
                -float(candidate["t_range"][0]),
            ),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda candidate: (
                int(candidate.get("query_relevance_score") or 0),
                candidate.get("possible_evidence") is True,
                bool(candidate.get("timestamp_cues")),
                candidate.get("status") == "unvisited",
                -float(candidate["t_range"][0]),
            ),
            reverse=True,
        )
    selected = candidates[: max(1, int(max_candidates))]
    memory["candidate_pool"] = deepcopy(selected)
    return selected


def build_compact_investigation_state(
    memory: dict[str, Any] | None,
    *,
    question: str | None = None,
    max_windows: int = 8,
    max_facts: int = 10,
    max_events: int = 8,
    max_gaps: int = 8,
    max_targets: int = 8,
    scope_aware_evidence_memory: bool = False,
    separate_routing_candidates: bool = False,
    query_relevant_retention: bool = False,
    persistent_candidate_pool: bool = False,
    persistent_candidate_pool_max_items: int = 20,
    persistent_candidate_pool_timestamp_radius_s: float = 8.0,
    candidate_evidence_projection: bool = False,
) -> dict[str, Any]:
    """Project raw trajectory memory into a short planner-facing state.

    This is intentionally only a memory organization layer. It does not block
    answers, create tool calls, or change routing decisions.
    """
    if not memory:
        return {}

    state = ensure_structured_evidence_state(memory)
    conflict_memory_enabled = bool(
        (memory.get("runtime_config") or {}).get(
            "coseek1_observer_conflict_memory_enabled", False
        )
    )
    choices = state.get("choices") or [
        {"letter": label, "text": text} for label, text in _extract_choices(question)
    ]
    by_id = _evidence_item_by_id(memory)

    overview_candidates: list[dict[str, Any]] = []
    if separate_routing_candidates:
        seen_candidates: set[tuple[Any, ...]] = set()
        for item in memory.get("scene_memory") or []:
            if not isinstance(item, dict) or str(item.get("source_tool") or "") != "overview":
                continue
            span = _safe_window(item.get("t_range"))
            key = (item.get("scene_id"), tuple(span or []))
            _dedupe_append(
                overview_candidates,
                {
                    "scene_id": item.get("scene_id") or "",
                    "t_range": span,
                    "summary": _short_text(item.get("summary"), 170),
                    "possible_evidence": item.get("possible_evidence"),
                    "suggest_focus_windows": item.get("suggest_focus_windows") or [],
                    "missing_detail": _short_text(item.get("missing_detail"), 150),
                    "status": "routing_only",
                },
                seen_candidates,
                key,
            )

    inspected_windows: list[dict[str, Any]] = []
    seen_windows: set[tuple[Any, ...]] = set()
    for item in memory.get("tool_observations") or []:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "")
        if tool == "answer" or (separate_routing_candidates and tool == "overview"):
            continue
        parameters = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
        windows = _parameter_windows(parameters) or [_safe_window(item.get("t_range"))]
        for window in windows:
            if window is None:
                continue
            key = (tool, tuple(window), item.get("scene_id"), item.get("window_id"))
            _dedupe_append(
                inspected_windows,
                {
                    "tool": tool,
                    "backend": item.get("observer_backend"),
                    "scene_id": item.get("scene_id") or "",
                    "window_id": item.get("window_id") or "",
                    "t_range": window,
                    "evidence_level": _evidence_level_for_tool(
                        tool,
                        item.get("detail_sufficient"),
                        item.get("decision_sufficient"),
                    ),
                    "summary": _short_text(item.get("summary") or item.get("missing_detail")),
                    "possible_evidence": item.get("possible_evidence"),
                    "detail_sufficient": item.get("detail_sufficient"),
                    "decision_sufficient": item.get("decision_sufficient"),
                    "target_entity_or_event": item.get("target_entity_or_event") or "",
                    "target_match": item.get("target_match") or "unknown",
                    "target_event_match": item.get("target_event_match") or "unknown",
                    "target_binding_reason": item.get("target_binding_reason") or "",
                    "question_scope": item.get("question_scope") or "unknown",
                    "scope_coverage": item.get("scope_coverage") or "unknown",
                    "scope_coverage_reason": item.get("scope_coverage_reason") or "",
                    "observed_fact": item.get("observed_fact") or "",
                    "evidence_need": item.get("evidence_need") or "",
                },
                seen_windows,
                key,
            )

    for item in memory.get("scene_memory") or []:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("source_tool") or "scene")
        if separate_routing_candidates and tool == "overview":
            continue
        window = _safe_window(item.get("t_range"))
        key = (tool, tuple(window or []), item.get("scene_id"), item.get("window_id"))
        _dedupe_append(
            inspected_windows,
            {
                "tool": tool,
                "backend": item.get("observer_backend"),
                "scene_id": item.get("scene_id") or "",
                "window_id": item.get("window_id") or "",
                "t_range": window,
                "evidence_level": _evidence_level_for_tool(
                    tool,
                    item.get("detail_sufficient"),
                    item.get("decision_sufficient"),
                ),
                "summary": _short_text(item.get("summary") or item.get("missing_detail")),
                "possible_evidence": item.get("possible_evidence"),
                "detail_sufficient": item.get("detail_sufficient"),
                "decision_sufficient": item.get("decision_sufficient"),
                "target_entity_or_event": item.get("target_entity_or_event") or "",
                "target_match": item.get("target_match") or "unknown",
                "target_event_match": item.get("target_event_match") or "unknown",
                "target_binding_reason": item.get("target_binding_reason") or "",
                "question_scope": item.get("question_scope") or "unknown",
                "scope_coverage": item.get("scope_coverage") or "unknown",
                "scope_coverage_reason": item.get("scope_coverage_reason") or "",
                "observed_fact": item.get("observed_fact") or "",
                "evidence_need": item.get("evidence_need") or "",
            },
            seen_windows,
            key,
        )

    if separate_routing_candidates and persistent_candidate_pool:
        overview_candidates = _build_persistent_candidate_pool(
            memory,
            question=question,
            inspected_windows=inspected_windows,
            max_candidates=persistent_candidate_pool_max_items,
            timestamp_radius_s=persistent_candidate_pool_timestamp_radius_s,
            candidate_evidence_projection=candidate_evidence_projection,
        )

    scene_coverage = (
        _build_scene_coverage(overview_candidates, inspected_windows)
        if separate_routing_candidates
        else []
    )
    memory["overview_candidates"] = overview_candidates
    memory["scene_coverage"] = scene_coverage

    visual_facts: list[dict[str, Any]] = []
    for item in sorted(
        [it for it in memory.get("timestamped_observations") or [] if isinstance(it, dict)],
        key=lambda it: _safe_float(it.get("timestamp_s")) or 0.0,
    ):
        desc = _short_text(item.get("description"), 150)
        if not desc:
            continue
        visual_facts.append(
            {
                "time": round(_safe_float(item.get("timestamp_s")) or 0.0, 1),
                "source": item.get("source_tool") or "tool",
                "backend": item.get("observer_backend"),
                "scene_id": item.get("scene_id") or "",
                "candidate_id": item.get("candidate_id") or "",
                "target_match": item.get("target_match") or "unknown",
                "event_match": item.get("event_match") or "unknown",
                "fact": desc,
                "needs_focus": _short_text(item.get("needs_focus"), 120),
                "evidence_scope": item.get("evidence_scope") or "unknown",
            }
        )
    if query_relevant_retention:
        visual_facts = _select_query_relevant_items(
            visual_facts,
            question=question,
            limit=max_facts,
        )

    inferred_events: list[dict[str, Any]] = []
    for item in memory.get("scene_memory") or []:
        if not isinstance(item, dict):
            continue
        if separate_routing_candidates and str(item.get("source_tool") or "") == "overview":
            continue
        summary = _short_text(item.get("summary"), 170)
        if not summary:
            continue
        inferred_events.append(
            {
                "source": item.get("source_tool") or "tool",
                "backend": item.get("observer_backend"),
                "scene_id": item.get("scene_id") or "",
                "window_id": item.get("window_id") or "",
                "t_range": _safe_window(item.get("t_range")),
                "event": summary,
                "possible_evidence": item.get("possible_evidence"),
                "detail_sufficient": item.get("detail_sufficient"),
                "suggest_focus_windows": item.get("suggest_focus_windows") or [],
                "evidence_scope": item.get("evidence_scope") or "unknown",
            }
        )

    target_bindings = [
        {
            "tool": item.get("tool"),
            "scene_id": item.get("scene_id") or "",
            "t_range": _safe_window(item.get("t_range")),
            "target": _short_text(item.get("target_entity_or_event"), 100),
            "target_match": str(item.get("target_match") or "unknown"),
            "target_event_match": str(item.get("target_event_match") or "unknown"),
            "reason": _short_text(item.get("target_binding_reason"), 170),
            "question_scope": item.get("question_scope") or "unknown",
            "scope_coverage": item.get("scope_coverage") or "unknown",
            "scope_coverage_reason": _short_text(
                item.get("scope_coverage_reason"), 170
            ),
            "observed_fact": _short_text(item.get("observed_fact"), 170),
            "evidence_need": str(item.get("evidence_need") or ""),
        }
        for item in memory.get("tool_observations") or []
        if isinstance(item, dict)
        and str(item.get("target_match") or "").strip()
        and str(item.get("target_match") or "").strip().lower() != "unknown"
    ][-6:]

    option_hypotheses: list[dict[str, Any]] = []
    verified_answer_candidate: dict[str, Any] | None = None
    for rec in state.get("option_support") or []:
        if not isinstance(rec, dict):
            continue
        support = [str(x) for x in (rec.get("supports") or [])]
        weak_support = [str(x) for x in (rec.get("weak_supports") or [])]
        contradict = [str(x) for x in (rec.get("contradicts") or [])]
        local_contradict = [str(x) for x in (rec.get("local_contradicts") or [])]
        unresolved = [str(x) for x in (rec.get("unresolved") or [])]
        verified_support = _verified_refs(support, by_id)
        verified_contradict = _verified_refs(
            contradict + local_contradict,
            by_id,
        )
        evidence_conflict = bool(verified_support and verified_contradict)
        verified_support_scopes = {
            eid: str((by_id.get(eid) or {}).get("evidence_scope") or "unknown")
            for eid in verified_support
        }
        row = {
            "option": rec.get("option"),
            "choice": _short_text(rec.get("choice"), 90),
            "verified_support": verified_support,
            "verified_contradict": verified_contradict,
            "evidence_conflict": evidence_conflict,
            "support": support,
            "weak_support": weak_support,
            "contradict": contradict,
            "local_contradict": local_contradict,
            "unresolved": unresolved,
            "verified_support_scopes": verified_support_scopes,
        }
        option_hypotheses.append(row)
        if (
            verified_support
            and verified_answer_candidate is None
            and (not conflict_memory_enabled or not evidence_conflict)
        ):
            verified_answer_candidate = row

    observer_conflicts = [
        {
            "option": row.get("option"),
            "choice": row.get("choice"),
            "support_refs": row.get("verified_support") or [],
            "contradict_refs": row.get("verified_contradict") or [],
            "conflict_type": "verified_vs_verified",
        }
        for row in option_hypotheses
        if conflict_memory_enabled and row.get("evidence_conflict")
    ]
    if conflict_memory_enabled:
        option_choices = {
            str(row.get("option") or "").upper(): row.get("choice") or ""
            for row in option_hypotheses
        }
        signal_refs: dict[str, dict[str, list[tuple[str, str]]]] = {}
        for item in state.get("evidence_items") or []:
            if not isinstance(item, dict):
                continue
            level = str(item.get("evidence_level") or "")
            if level not in {"candidate", "verified"}:
                continue
            evidence_id = str(item.get("evidence_id") or "")
            if not evidence_id:
                continue
            for key, direction in (
                ("supports_options", "support"),
                ("contradicts_options", "contradict"),
            ):
                for label in item.get(key) or []:
                    letter = str(label or "").strip().upper()
                    if letter not in option_choices:
                        continue
                    rows = signal_refs.setdefault(
                        letter, {"support": [], "contradict": []}
                    )
                    ref = (evidence_id, level)
                    if ref not in rows[direction]:
                        rows[direction].append(ref)
        existing_options = {str(item.get("option") or "") for item in observer_conflicts}
        for letter, signals in signal_refs.items():
            support_refs = signals["support"]
            contradict_refs = signals["contradict"]
            if not support_refs or not contradict_refs or letter in existing_options:
                continue
            levels = {level for _eid, level in support_refs + contradict_refs}
            if "verified" not in levels:
                continue
            observer_conflicts.append(
                {
                    "option": letter,
                    "choice": option_choices.get(letter, ""),
                    "support_refs": [eid for eid, _level in support_refs][-4:],
                    "contradict_refs": [eid for eid, _level in contradict_refs][-4:],
                    "support_levels": [level for _eid, level in support_refs][-4:],
                    "contradict_levels": [level for _eid, level in contradict_refs][-4:],
                    "conflict_type": "candidate_or_verified_observer_disagreement",
                }
            )

    covered_options = {
        str(row.get("option") or "").upper()
        for row in option_hypotheses
        if (
            row.get("support")
            or row.get("weak_support")
            or row.get("contradict")
            or (not scope_aware_evidence_memory and row.get("local_contradict"))
        )
    }
    uncovered_options = [
        str(choice.get("letter") or "").upper()
        for choice in choices
        if isinstance(choice, dict)
        and str(choice.get("letter") or "").upper()
        and str(choice.get("letter") or "").upper() not in covered_options
    ]

    verified_windows = [
        item
        for item in inspected_windows
        if item.get("evidence_level") == "verified"
    ]
    missing_context: list[dict[str, Any]] = []
    seen_gaps: set[tuple[Any, ...]] = set()
    for item in memory.get("open_gaps") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "open") == "resolved":
            continue
        gap = _short_text(item.get("gap") or item.get("missing_detail"), 170)
        if not gap or not _meaningful_gap(gap):
            continue
        key = (gap, str(item.get("suggested_window")))
        _dedupe_append(
            missing_context,
            {
                "source": item.get("source_tool") or "tool",
                "gap": gap,
                "suggested_window": _safe_window(item.get("suggested_window")),
            },
            seen_gaps,
            key,
        )

    for item in state.get("uncertainties") or []:
        if not isinstance(item, dict):
            continue
        evidence = by_id.get(str(item.get("evidence_id") or "")) or {}
        if separate_routing_candidates and evidence.get("source_tool") == "overview":
            continue
        if scope_aware_evidence_memory and (
            evidence.get("detail_sufficient") is True
            or evidence.get("decision_sufficient") is True
        ):
            continue
        # A routing/candidate uncertainty is superseded once stronger visual
        # evidence has inspected the same temporal scope. Keep partially
        # covered gaps so a narrow verification cannot erase a wider need.
        evidence_level = str(evidence.get("evidence_level") or "")
        if evidence_level in {"routing", "candidate"}:
            uncertainty_ts = _safe_float(
                evidence.get("timestamp_s")
                if evidence.get("timestamp_s") is not None
                else item.get("timestamp_s")
            )
            if uncertainty_ts is not None and any(
                (span := _safe_window(window.get("t_range"))) is not None
                and span[0] <= uncertainty_ts <= span[1]
                for window in verified_windows
            ):
                continue
            uncertainty_window = _safe_window(
                evidence.get("t_range") or item.get("t_range")
            )
            if uncertainty_window is not None:
                _covered, verified_ratio = _window_coverage(
                    uncertainty_window,
                    verified_windows,
                )
                if verified_ratio >= 0.8:
                    continue
        gap = _short_text(item.get("uncertainty"), 170)
        if not gap or not _meaningful_gap(gap):
            continue
        key = (gap, str(item.get("t_range")), item.get("scene_id"))
        _dedupe_append(
            missing_context,
            {
                "source": item.get("evidence_id") or "evidence",
                "scene_id": item.get("scene_id") or "",
                "gap": gap,
                "suggested_window": _safe_window(item.get("t_range")),
                "timestamp_s": item.get("timestamp_s"),
            },
            seen_gaps,
            key,
        )

    next_search_targets: list[dict[str, Any]] = []
    seen_targets: set[tuple[Any, ...]] = set()
    for item in inferred_events:
        if item.get("possible_evidence") is False:
            continue
        windows = item.get("suggest_focus_windows") or []
        if not windows and item.get("possible_evidence") and item.get("detail_sufficient") is not True:
            windows = [item.get("t_range")]
        for window in windows:
            span = _safe_window(window)
            if span is None:
                continue
            _covered, verified_ratio = _window_coverage(span, verified_windows)
            if verified_ratio >= 0.8:
                continue
            key = ("window", tuple(span), item.get("scene_id"))
            _dedupe_append(
                next_search_targets,
                {
                    "target": "verify_candidate_window",
                    "scene_id": item.get("scene_id") or "",
                    "t_range": span,
                    "reason": _short_text(item.get("event"), 140),
                },
                seen_targets,
                key,
            )
    for gap in missing_context:
        span = _safe_window(gap.get("suggested_window"))
        if span is None:
            continue
        key = ("gap", tuple(span), gap.get("gap"))
        _dedupe_append(
            next_search_targets,
            {
                "target": "resolve_missing_context",
                "t_range": span,
                "reason": _short_text(gap.get("gap"), 140),
            },
            seen_targets,
            key,
        )

    if separate_routing_candidates:
        facts_by_scene: dict[str, list[str]] = {}
        for fact in memory.get("timestamped_observations") or []:
            if not isinstance(fact, dict):
                continue
            scene_id = str(fact.get("scene_id") or "")
            description = str(fact.get("description") or "").strip()
            if scene_id and description:
                facts_by_scene.setdefault(scene_id, []).append(description)
        routing_targets: list[dict[str, Any]] = []
        for candidate in overview_candidates:
            scene_id = str(candidate.get("scene_id") or "")
            candidate_text = " ".join(
                [_routing_positive_text(candidate)]
                + facts_by_scene.get(scene_id, [])
            )
            relevance_score = _query_overlap_score(candidate_text, question)
            query_relevant = relevance_score > 0
            if candidate.get("possible_evidence") is False and not query_relevant:
                continue
            windows = candidate.get("suggest_focus_windows") or [candidate.get("t_range")]
            for window in windows:
                span = _safe_window(window)
                if span is None:
                    continue
                _covered, ratio = _window_coverage(span, inspected_windows)
                if ratio >= 0.8:
                    continue
                routing_targets.append(
                    {
                        "target": "inspect_overview_candidate",
                        "scene_id": candidate.get("scene_id") or "",
                        "t_range": span,
                        "coverage_ratio": ratio,
                        "possible_evidence": candidate.get("possible_evidence"),
                        "routing_negative_cue": candidate.get("routing_negative_cue", False),
                        "query_relevance_score": relevance_score,
                        "reason": _short_text(
                            candidate.get("missing_detail") or candidate.get("summary"),
                            140,
                        ),
                    }
                )
        routing_targets.sort(
            key=lambda item: (
                item.get("possible_evidence") is True,
                not bool(item.get("routing_negative_cue")),
                int(item.get("query_relevance_score") or 0),
                -float(item.get("coverage_ratio") or 0.0),
            ),
            reverse=True,
        )
        for target in routing_targets:
            key = (
                "overview_candidate",
                tuple(target.get("t_range") or []),
                target.get("scene_id"),
            )
            _dedupe_append(
                next_search_targets,
                target,
                seen_targets,
                key,
            )

    if candidate_evidence_projection:
        next_search_targets = _compact_search_target_frontier(next_search_targets)

    answer_status: dict[str, Any]
    if verified_answer_candidate:
        support_scopes = verified_answer_candidate.get("verified_support_scopes") or {}
        candidate_option = str(verified_answer_candidate.get("option") or "")
        candidate_conflicts = [
            item
            for item in observer_conflicts
            if str(item.get("option") or "") == candidate_option
        ]
        local_only = bool(support_scopes) and all(
            scope in {"local_timestamp", "local_window"}
            for scope in support_scopes.values()
        )
        answer_status = {
            "status": (
                "conflicted_verified_candidate_available"
                if candidate_conflicts
                else (
                    "local_verified_candidate_available"
                    if scope_aware_evidence_memory and local_only
                    else "verified_candidate_available"
                )
            ),
            "option": verified_answer_candidate.get("option"),
            "choice": verified_answer_candidate.get("choice"),
            "support_refs": verified_answer_candidate.get("verified_support"),
            "support_scopes": support_scopes,
            "observer_conflicts": candidate_conflicts,
            "uncovered_options": uncovered_options,
        }
    else:
        weak_rows = [
            {
                "option": row.get("option"),
                "choice": row.get("choice"),
                "weak_support": row.get("weak_support") or row.get("support"),
                "contradict": row.get("contradict"),
            }
            for row in option_hypotheses
            if row.get("support") or row.get("weak_support") or row.get("contradict")
        ][:4]
        answer_status = {
            "status": "no_verified_answer_yet",
            "weak_hypotheses": weak_rows,
            "uncovered_options": uncovered_options,
        }

    if answer_status.get("status") == "local_verified_candidate_available":
        answer_status["uninspected_alternatives"] = [
            {
                "scene_id": target.get("scene_id") or "",
                "t_range": target.get("t_range"),
                "reason": target.get("reason") or "",
                "coverage_ratio": target.get("coverage_ratio"),
                "routing_possible_evidence": target.get("possible_evidence"),
            }
            for target in next_search_targets
            if target.get("target") == "inspect_overview_candidate"
        ][:6]
        answer_status["search_coverage"] = {
            "overview_scene_count": len(overview_candidates),
            "materially_inspected_scene_count": sum(
                float(item.get("coverage_ratio") or 0.0) >= 0.5
                for item in scene_coverage
            ),
        }

    if verified_answer_candidate and answer_status.get("status") == "verified_candidate_available":
        next_search_targets = []

    compact = {
        "version": "coseek1_compact_investigation_state_v3_candidate_pool",
        "overview_candidates": overview_candidates,
        "inspected_windows": inspected_windows[-max_windows:],
        "scene_coverage": scene_coverage,
        "visual_facts": (
            visual_facts if query_relevant_retention else visual_facts[-max_facts:]
        ),
        "inferred_events": inferred_events[-max_events:],
        "target_bindings": target_bindings,
        "option_hypotheses": option_hypotheses,
        "observer_conflicts": observer_conflicts,
        "missing_context": missing_context[-max_gaps:],
        "answer_status": answer_status,
        "next_search_targets": next_search_targets[:max_targets],
    }
    memory["compact_investigation_state"] = compact
    return compact


def format_compact_investigation_state_for_prompt(
    memory: dict[str, Any] | None,
    *,
    question: str | None = None,
    max_windows: int = 8,
    max_facts: int = 10,
    max_events: int = 8,
    max_gaps: int = 8,
    max_targets: int = 8,
    scope_aware_evidence_memory: bool = False,
    separate_routing_candidates: bool = False,
    query_relevant_retention: bool = False,
    persistent_candidate_pool: bool = False,
    persistent_candidate_pool_max_items: int = 20,
    persistent_candidate_pool_timestamp_radius_s: float = 8.0,
    candidate_evidence_projection: bool = False,
) -> str:
    compact = build_compact_investigation_state(
        memory,
        question=question,
        max_windows=max_windows,
        max_facts=max_facts,
        max_events=max_events,
        max_gaps=max_gaps,
        max_targets=max_targets,
        scope_aware_evidence_memory=scope_aware_evidence_memory,
        separate_routing_candidates=separate_routing_candidates,
        query_relevant_retention=query_relevant_retention,
        persistent_candidate_pool=persistent_candidate_pool,
        persistent_candidate_pool_max_items=persistent_candidate_pool_max_items,
        persistent_candidate_pool_timestamp_radius_s=persistent_candidate_pool_timestamp_radius_s,
        candidate_evidence_projection=candidate_evidence_projection,
    )
    if not compact:
        return ""

    lines: list[str] = [
        "Compact Investigation State:",
        "This is a navigation summary of trajectory memory, not an answer gate.",
    ]

    overview_candidates = compact.get("overview_candidates") or []
    if overview_candidates:
        lines.append(
            "Persistent Candidate Pool (overview routing map; candidate identity and status are memory, not final evidence):"
            if persistent_candidate_pool
            else "Overview routing candidates (inferred map; not inspected evidence):"
        )
        for candidate_index, item in enumerate(overview_candidates):
            focus = item.get("suggest_focus_windows") or []
            focus_text = f" focus={focus[:3]}" if focus else ""
            missing = f" missing={item.get('missing_detail')}" if item.get("missing_detail") else ""
            anchors = item.get("anchor_timestamps") or []
            anchor_text = f" anchors={anchors}" if anchors else ""
            bindings = item.get("target_bindings") or []
            binding_text = f" target_bindings={bindings}" if bindings else ""
            cue_rows = item.get("timestamp_cues") or []
            cue_text = ""
            if candidate_evidence_projection and candidate_index < 6 and cue_rows:
                rendered_cues = [
                    f"{float(cue.get('timestamp_s') or 0.0):.1f}s "
                    f"{_short_text(cue.get('description'), 72)}"
                    for cue in cue_rows
                ]
                cue_text = f" scene_facts={rendered_cues}"
            lines.append(
                f"- {item.get('candidate_id') or '?'} source={item.get('source') or 'overview_scene'} "
                f"scene={item.get('scene_id') or '?'} {_window_label(item.get('t_range'))}: "
                f"{item.get('summary') or ''} possible={item.get('possible_evidence')}"
                f" relevance={item.get('query_relevance_score', 0)} status={item.get('status', 'routing_only')}"
                f"{anchor_text}{focus_text}{missing}{cue_text}{binding_text}"
            )

    inspected = compact.get("inspected_windows") or []
    if inspected:
        lines.append("Inspected windows:")
        for item in inspected:
            lines.append(
                f"- {item.get('tool')}/{item.get('backend') or '?'} "
                f"{item.get('scene_id') or '?'} {_window_label(item.get('t_range'))}: "
                f"{item.get('summary') or ''} level={item.get('evidence_level')} "
                f"possible={item.get('possible_evidence')} sufficient={item.get('detail_sufficient')} "
                f"need={item.get('evidence_need') or 'unknown'}"
            )

    coverage = compact.get("scene_coverage") or []
    if coverage:
        lines.append("Scene coverage from real-frame tools:")
        for item in coverage:
            lines.append(
                f"- {item.get('scene_id') or '?'} {_window_label(item.get('t_range'))}: "
                f"coverage={float(item.get('coverage_ratio') or 0.0):.0%} "
                f"intervals={item.get('covered_intervals') or []} "
                f"evidence_found={item.get('evidence_found')} "
                f"negative_observations={item.get('negative_observations') or 0} "
                f"routing_possible={item.get('routing_possible_evidence')} "
                f"summary={item.get('summary') or ''} "
                f"missing={item.get('missing_detail') or ''}"
            )

    facts = compact.get("visual_facts") or []
    if facts:
        lines.append("Timestamped visual facts:")
        for item in facts:
            need = f" need={item.get('needs_focus')}" if item.get("needs_focus") else ""
            candidate = (
                f" candidate={item.get('candidate_id')}"
                if item.get("candidate_id")
                else ""
            )
            binding = (
                f" target_match={item.get('target_match') or 'unknown'}"
                f" event_match={item.get('event_match') or 'unknown'}"
                if item.get("candidate_id")
                else ""
            )
            lines.append(
                f"- {item.get('time'):.1f}s {item.get('source')}/{item.get('backend') or '?'} "
                f"{item.get('scene_id') or '?'}{candidate}: {item.get('fact')} "
                f"{binding} "
                f"scope={item.get('evidence_scope') or 'unknown'}{need}"
            )

    events = compact.get("inferred_events") or []
    if events:
        lines.append("Inferred scene/event candidates:")
        for item in events:
            focus = item.get("suggest_focus_windows") or []
            focus_text = f" focus={focus[:3]}" if focus else ""
            lines.append(
                f"- {item.get('source')}/{item.get('backend') or '?'} "
                f"{item.get('scene_id') or '?'} {_window_label(item.get('t_range'))}: "
                f"{item.get('event')} possible={item.get('possible_evidence')} "
                f"sufficient={item.get('detail_sufficient')} "
                f"scope={item.get('evidence_scope') or 'unknown'}{focus_text}"
            )

    bindings = compact.get("target_bindings") or []
    if bindings:
        lines.append("Question-target binding from strong observations:")
        for item in bindings:
            lines.append(
                f"- {item.get('tool')}/{item.get('scene_id') or '?'} "
                f"{_window_label(item.get('t_range'))}: target={item.get('target') or '?'} "
                f"match={item.get('target_match')} fact={item.get('observed_fact') or ''} "
                f"event_match={item.get('target_event_match') or 'unknown'} "
                f"question_scope={item.get('question_scope') or 'unknown'} "
                f"scope_coverage={item.get('scope_coverage') or 'unknown'} "
                f"need={item.get('evidence_need') or 'unknown'} "
                f"reason={item.get('reason') or ''} "
                f"scope_reason={item.get('scope_coverage_reason') or ''}"
            )

    options = compact.get("option_hypotheses") or []
    if options:
        lines.append("Option hypotheses:")
        for row in options:
            lines.append(
                f"- {row.get('option')} {row.get('choice')}: "
                f"verified={row.get('verified_support') or []} "
                f"verified_contradict={row.get('verified_contradict') or []} "
                f"conflict={bool(row.get('evidence_conflict'))} "
                f"support={row.get('support') or []} weak={row.get('weak_support') or []} "
                f"contradict={row.get('contradict') or []} "
                f"local_contradict={row.get('local_contradict') or []} "
                f"unresolved={row.get('unresolved') or []}"
            )

    conflicts = compact.get("observer_conflicts") or []
    if conflicts:
        lines.append(
            "Observer conflicts containing verified evidence (retain both sides):"
        )
        for item in conflicts:
            lines.append(
                f"- option={item.get('option')} {item.get('choice') or ''}: "
                f"support={item.get('support_refs') or []} "
                f"contradict={item.get('contradict_refs') or []}"
            )

    answer_status = compact.get("answer_status") or {}
    if answer_status:
        uncovered = answer_status.get("uncovered_options") or []
        uncovered_text = f"; uncovered_options={uncovered}" if uncovered else ""
        if answer_status.get("status") == "conflicted_verified_candidate_available":
            lines.append(
                "Current answer status: conflicted_candidate="
                f"{answer_status.get('option')} support={answer_status.get('support_refs') or []} "
                f"conflicts={answer_status.get('observer_conflicts') or []}. "
                "The trajectory contains competing visual observations; this state reports "
                "the disagreement and does not select a resolution."
            )
        elif answer_status.get("status") == "local_verified_candidate_available":
            alternatives = answer_status.get("uninspected_alternatives") or []
            lines.append(
                "Current answer status: provisional_local_candidate="
                f"{answer_status.get('option')} support={answer_status.get('support_refs') or []} "
                f"scopes={answer_status.get('support_scopes') or {}}; "
                f"uncovered_options={uncovered}; "
                f"search_coverage={answer_status.get('search_coverage') or {}}. "
                "This verifies the candidate only inside the inspected window; uncovered options "
                "have not been contradicted elsewhere."
            )
            if alternatives:
                lines.append(f"Uninspected competing routing candidates: {alternatives}")
        elif answer_status.get("status") == "verified_candidate_available":
            lines.append(
                "Current answer status: "
                f"verified_candidate={answer_status.get('option')} "
                f"support={answer_status.get('support_refs') or []}"
                f" scopes={answer_status.get('support_scopes') or {}}"
                f"{uncovered_text}"
            )
        else:
            weak = answer_status.get("weak_hypotheses") or []
            lines.append(
                "Current answer status: no verified answer yet"
                + (f"; weak_hypotheses={weak}" if weak else "")
                + uncovered_text
            )

    gaps = compact.get("missing_context") or []
    if gaps:
        lines.append("Missing context / open gaps:")
        for item in gaps:
            when = item.get("timestamp_s")
            when_text = f" {when:.1f}s" if isinstance(when, (int, float)) else ""
            lines.append(
                f"- {item.get('source')}{when_text} {_window_label(item.get('suggested_window'))}: "
                f"{item.get('gap')}"
            )

    targets = compact.get("next_search_targets") or []
    if targets:
        lines.append("Natural next search targets:")
        for item in targets:
            option = f" option={item.get('option')}" if item.get("option") else ""
            lines.append(
                f"- {item.get('target')}{option} {_window_label(item.get('t_range'))}: "
                f"{item.get('reason')}"
                + (
                    f" coverage={float(item.get('coverage_ratio') or 0.0):.0%}"
                    if item.get("coverage_ratio") is not None
                    else ""
                )
            )

    return "\n".join(lines)


def _meaningful_gap(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" .")
    if not text:
        return False
    lowered = text.lower()
    if lowered in {"none", "null", "n/a", "no missing detail", "nothing missing"}:
        return False
    if re.match(r"^(none|null|n/a)\s*[,;:]", lowered):
        return False
    return True


def _append_open_gap(
    memory: dict[str, Any],
    *,
    source_tool: str,
    gap: Any,
    suggested_window: Any = None,
    scene_id: Any = None,
    persistent: bool = False,
) -> None:
    if not _meaningful_gap(gap):
        return
    gaps = memory.setdefault("open_gaps", [])
    span = _safe_window(suggested_window)
    text = _short_text(gap, 240)
    key = (source_tool, str(scene_id or ""), tuple(span or []), text.lower())
    for item in gaps:
        if not isinstance(item, dict):
            continue
        existing = (
            str(item.get("source_tool") or ""),
            str(item.get("scene_id") or ""),
            tuple(_safe_window(item.get("suggested_window")) or []),
            _short_text(item.get("gap"), 240).lower(),
        )
        if existing == key:
            return
    record = {
        "source_tool": source_tool,
        "scene_id": scene_id or "",
        "gap": text,
        "suggested_window": span,
    }
    if persistent:
        record.update(
            {
                "gap_id": f"G{len(gaps) + 1:05d}",
                "status": "open",
                "resolved_by": None,
            }
        )
    gaps.append(record)


def _resolve_overlapping_gaps(
    memory: dict[str, Any],
    *,
    tool_name: str,
    payload: dict[str, Any],
    semantic_resolution: bool = False,
) -> None:
    decision_sufficient = payload.get("decision_sufficiency_validated") is True
    if tool_name == "overview" or (
        payload.get("detail_sufficient") is not True and not decision_sufficient
    ):
        return
    if semantic_resolution:
        evidence_need = str(
            payload.get("decision_evidence_need")
            if decision_sufficient
            else payload.get("evidence_need") or "sufficient"
        ).strip().lower()
        if evidence_need != "sufficient":
            return
        target_match = str(payload.get("target_match") or "unknown").strip().lower()
        if target_match not in {"matched", "partial"}:
            return
        if str(payload.get("scope_coverage") or "unknown").strip().lower() in {
            "partial",
            "insufficient",
        }:
            return
    observed = _safe_window(payload.get("t_range"))
    if observed is None:
        return
    for item in memory.get("open_gaps") or []:
        if not isinstance(item, dict) or str(item.get("status") or "open") != "open":
            continue
        target = _safe_window(item.get("suggested_window"))
        if target is None or _overlap_duration(observed, target) <= 0:
            continue
        item["status"] = "resolved"
        item["resolved_by"] = tool_name
        item["resolved_window"] = observed


def merge_tool_observation(
    memory: dict[str, Any],
    *,
    tool_name: str,
    parameters: dict[str, Any] | None,
    output: str,
    use_structured_evidence_state: bool = False,
    question_context: str | None = None,
    include_evidence_scope: bool = True,
    scope_aware_evidence_memory: bool = False,
    persistent_open_gaps: bool = False,
    semantic_gap_resolution: bool = False,
    preserve_aggregate_verifier_decision: bool = False,
) -> dict[str, Any]:
    ensure_structured_evidence_state(memory)
    payload = extract_v10_payload(output)
    if not isinstance(payload, dict):
        memory.setdefault("tool_observations", []).append(
            {
                "tool": tool_name,
                "parameters": deepcopy(parameters or {}),
                "summary": output[:800] if output else "",
                "parsed": False,
            }
        )
        return memory

    candidate_windows: dict[str, dict[str, Any]] = {}
    for candidate in (parameters or {}).get("candidate_windows") or []:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if candidate_id:
            candidate_windows[candidate_id] = candidate
    candidate_assessments = [
        deepcopy(item)
        for item in (payload.get("candidate_assessments") or [])
        if isinstance(item, dict) and str(item.get("candidate_id") or "").strip()
    ]
    assessment_by_id = {
        str(item["candidate_id"]): item for item in candidate_assessments
    }
    verification_id = f"verify_{len(memory.get('tool_observations') or []) + 1:05d}"
    if payload.get("receipt_schema") == "local_visual_receipt_v1":
        payload["verification_id"] = verification_id
    tool_integrity_enabled = bool(
        (memory.get("runtime_config") or {}).get(
            "coseek1_tool_integrity_repair_enabled", False
        )
    )
    if tool_integrity_enabled:
        for assessment in candidate_assessments:
            local_group_id = str(assessment.get("event_group_id") or "").strip()
            if local_group_id:
                assessment["local_event_group_id"] = local_group_id
                assessment["event_group_id"] = _global_event_group_id(
                    verification_id, local_group_id
                )

    tool_obs = {
        "tool": tool_name,
        "parameters": deepcopy(parameters or {}),
        "summary": payload.get("global_summary")
        or payload.get("overall_summary")
        or payload.get("observed_event")
        or "",
        "parsed": True,
        "scene_id": payload.get("scene_id"),
        "window_id": payload.get("window_id"),
        "t_range": payload.get("t_range"),
        "observer_backend": payload.get("observer_backend"),
        "possible_evidence": payload.get("possible_evidence")
        if "possible_evidence" in payload
        else payload.get("contains_evidence"),
        "relevance": payload.get("relevance"),
        "detail_sufficient": payload.get("detail_sufficient"),
        "visual_detail_sufficient": payload.get("visual_detail_sufficient"),
        "decision_sufficient": payload.get("decision_sufficiency_validated") is True,
        "decision_evidence_need": payload.get("decision_evidence_need") or "",
        "residual_visual_detail": payload.get("residual_visual_detail") or "",
        "supports_options": payload.get("supports_options") or [],
        "contradicts_options": payload.get("contradicts_options") or [],
        "missing_detail": payload.get("missing_detail") or "",
        "target_entity_or_event": payload.get("target_entity_or_event") or "",
        "target_match": payload.get("target_match") or "unknown",
        "target_event_match": payload.get("target_event_match") or "unknown",
        "target_binding_reason": payload.get("target_binding_reason") or "",
        "question_scope": payload.get("question_scope") or "unknown",
        "scope_coverage": payload.get("scope_coverage") or "unknown",
        "scope_coverage_reason": payload.get("scope_coverage_reason") or "",
        "observed_fact": payload.get("observed_fact") or "",
        "evidence_need": payload.get("evidence_need") or "",
        "evidence_needs": payload.get("evidence_needs") or [],
        "observer_conflict_options": payload.get("observer_conflict_options") or [],
        "candidate_assessments": candidate_assessments,
        "candidate_binding_complete": payload.get("candidate_binding_complete"),
        "candidate_search_status": payload.get("candidate_search_status") or "",
        "verification_id": verification_id,
        "observation_reused": payload.get("observation_reused") is True,
        "no_new_visual_information": payload.get("no_new_visual_information") is True,
        "reuse_cache_id": payload.get("reuse_cache_id") or "",
        "reuse_count": int(payload.get("reuse_count") or 0),
    }
    if payload.get("receipt_schema") == "local_visual_receipt_v1":
        tool_obs["receipt_schema"] = payload["receipt_schema"]
        tool_obs["receipt_parse_ok"] = payload.get("parse_ok")
    memory.setdefault("tool_observations", []).append(tool_obs)

    # A cache hit is an audit event, not a new observation. Keep the latest
    # tool record visible to the planner but do not duplicate timestamp facts,
    # candidate bindings, gaps, or structured evidence.
    if payload.get("observation_reused") is True:
        stats = memory.setdefault(
            "verified_observation_reuse_stats",
            {"reuse_count": 0, "reused_cache_ids": []},
        )
        stats["reuse_count"] = int(stats.get("reuse_count") or 0) + 1
        cache_id = str(payload.get("reuse_cache_id") or "").strip()
        cache_ids = stats.setdefault("reused_cache_ids", [])
        if cache_id and cache_id not in cache_ids:
            cache_ids.append(cache_id)
        return memory

    if candidate_assessments:
        binding_memory = memory.setdefault("candidate_binding_memory", [])
        new_binding_records: list[dict[str, Any]] = []
        for assessment in candidate_assessments:
            candidate_id = str(assessment.get("candidate_id") or "")
            source_candidate = candidate_windows.get(candidate_id) or {}
            local_event_group_id = (
                assessment.get("local_event_group_id")
                or assessment.get("event_group_id")
                or source_candidate.get("event_group_id")
                or ""
            )
            if str(local_event_group_id).startswith(f"{verification_id}:"):
                local_event_group_id = str(local_event_group_id).split(":", 1)[1]
            stored_event_group_id = (
                _global_event_group_id(verification_id, local_event_group_id)
                if tool_integrity_enabled
                else str(local_event_group_id)
            )
            binding_record = {
                "id": _memory_id("bind", len(binding_memory) + 1),
                "source_tool": tool_name,
                "candidate_id": candidate_id,
                "verification_id": verification_id,
                "event_group_id": stored_event_group_id,
                "rank": source_candidate.get("rank"),
                "t_range": assessment.get("event_span")
                or source_candidate.get("t_range")
                or source_candidate.get("recommended_verify_window"),
                "event_span_source": (
                    "candidate_assessment"
                    if assessment.get("event_span")
                    else "candidate_verify_window"
                ),
                "routing_summary": source_candidate.get("summary")
                or source_candidate.get("fine_summary")
                or "",
                "target_match": assessment.get("target_match") or "unknown",
                "event_match": assessment.get("event_match") or "unknown",
                "observed_fact": assessment.get("observed_fact") or "",
                "target_binding_reason": assessment.get("target_binding_reason") or "",
                "best_timestamp_s": assessment.get("best_timestamp_s"),
                "supports_options": assessment.get("supports_options") or [],
                "contradicts_options": assessment.get("contradicts_options") or [],
                "option_set_conflict": assessment.get("option_set_conflict") is True,
                "observer_backend": payload.get("observer_backend"),
            }
            if payload.get("receipt_schema") == "local_visual_receipt_v1":
                for key in ("missing_detail", "anchor_observations", "target_event_id", "t_range"):
                    binding_record[key] = deepcopy(assessment.get(key))
                binding_record["receipt_schema"] = payload["receipt_schema"]
            if tool_integrity_enabled:
                binding_record["local_event_group_id"] = str(local_event_group_id)
            binding_memory.append(binding_record)
            new_binding_records.append(binding_record)
    else:
        new_binding_records = []

    if tool_integrity_enabled and tool_name == "frame_verify":
        verification_rows = list(new_binding_records)
        if not verification_rows:
            parameters = parameters or {}
            verification_rows = [
                {
                    "candidate_id": "",
                    "t_range": payload.get("t_range")
                    or [parameters.get("start_time"), parameters.get("end_time")],
                    "best_timestamp_s": None,
                    "target_match": payload.get("target_match") or "unknown",
                    "event_match": payload.get("target_event_match") or "unknown",
                    "observed_fact": payload.get("observed_fact")
                    or payload.get("overall_summary")
                    or "",
                    "event_group_id": "",
                }
            ]
        conflict_ids = _record_verification_facts(
            memory,
            verification_id=verification_id,
            rows=verification_rows,
        )
        if conflict_ids:
            tool_obs["observation_conflict_ids"] = conflict_ids

    timestamped = memory.setdefault("timestamped_observations", [])
    for item in payload.get("timestamp_observations") or []:
        if not isinstance(item, dict):
            continue
        ts_value = item.get("timestamp_s")
        if ts_value is None:
            ts_value = item.get("timestamp")
        ts = _safe_float(ts_value)
        if ts is None:
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        candidate_assessment = assessment_by_id.get(candidate_id) or {}
        rec = {
            "id": _memory_id("obs", len(timestamped) + 1),
            "source_tool": tool_name,
            "timestamp_s": ts,
            "scene_id": item.get("scene_id") or payload.get("scene_id"),
            "window_id": item.get("window_id") or payload.get("window_id"),
            "time_range_s": item.get("time_range_s") or payload.get("t_range"),
            "description": item.get("description") or item.get("desc") or "",
            "event_tags": item.get("event_tags") or [],
            "confidence": item.get("confidence") or payload.get("confidence") or "weak",
            "observer_backend": item.get("observer_backend") or payload.get("observer_backend"),
            "evidence_refs": item.get("evidence_refs") or item.get("frame_ids") or [],
            "needs_focus": item.get("needs_focus") or payload.get("missing_detail") or "",
            "candidate_id": candidate_id,
            "target_match": item.get("target_match")
            or candidate_assessment.get("target_match")
            or payload.get("target_match")
            or "unknown",
            "event_match": item.get("event_match")
            or candidate_assessment.get("event_match")
            or payload.get("target_event_match")
            or "unknown",
            "target_binding_reason": item.get("target_binding_reason")
            or candidate_assessment.get("target_binding_reason")
            or payload.get("target_binding_reason")
            or "",
            "supports_options": item.get("supports_options")
            or candidate_assessment.get("supports_options")
            or [],
            "contradicts_options": item.get("contradicts_options")
            or candidate_assessment.get("contradicts_options")
            or [],
            "option_set_conflict": item.get("option_set_conflict") is True
            or candidate_assessment.get("option_set_conflict") is True,
            "evidence_need": payload.get("evidence_need") or "",
            "evidence_needs": payload.get("evidence_needs") or [],
            "decision_sufficient": payload.get("decision_sufficiency_validated") is True,
            "decision_evidence_need": payload.get("decision_evidence_need") or "",
        }
        if include_evidence_scope:
            rec["evidence_scope"] = "local_timestamp"
        timestamped.append(rec)

    scene_memory = memory.setdefault("scene_memory", [])
    for item in payload.get("scene_summaries") or []:
        if not isinstance(item, dict):
            continue
        scene_record = {
                "source_tool": tool_name,
                "scene_id": item.get("scene_id"),
                "window_id": item.get("window_id"),
                "t_range": item.get("t_range") or item.get("time_range_s"),
                "summary": item.get("summary") or item.get("observed_event") or "",
                "possible_evidence": item.get("possible_evidence"),
                "observer_backend": payload.get("observer_backend"),
                "suggest_focus_windows": item.get("suggest_focus_windows")
                or item.get("suggest_focus_window")
                or [],
                "missing_detail": item.get("missing_detail") or "",
                "detail_sufficient": payload.get("detail_sufficient"),
                "decision_sufficient": payload.get("decision_sufficiency_validated") is True,
                "decision_evidence_need": payload.get("decision_evidence_need") or "",
                "supports_options": payload.get("supports_options") or [],
                "contradicts_options": payload.get("contradicts_options") or [],
                "relevance": payload.get("relevance"),
                "target_entity_or_event": payload.get("target_entity_or_event") or "",
                "target_match": payload.get("target_match") or "unknown",
                "target_binding_reason": payload.get("target_binding_reason") or "",
                "question_scope": payload.get("question_scope") or "unknown",
                "scope_coverage": payload.get("scope_coverage") or "unknown",
                "scope_coverage_reason": payload.get("scope_coverage_reason") or "",
                "observed_fact": payload.get("observed_fact") or "",
                "evidence_need": payload.get("evidence_need") or "",
                "evidence_needs": payload.get("evidence_needs") or [],
            }
        if include_evidence_scope:
            scene_record["evidence_scope"] = "local_window"
        scene_memory.append(scene_record)

    # Overview gaps belong to routing candidates, not the global unresolved
    # evidence list. A sufficient observation also cannot create a new gap.
    semantically_sufficient = (
        payload.get("decision_sufficiency_validated") is True
        or (
            payload.get("detail_sufficient") is True
            and str(payload.get("evidence_need") or "sufficient").strip().lower()
            == "sufficient"
        )
    )
    record_open_gaps = not (
        (
            semantically_sufficient
            if semantic_gap_resolution
            else (
                payload.get("detail_sufficient") is True
                or payload.get("decision_sufficiency_validated") is True
            )
        )
        or (scope_aware_evidence_memory and tool_name == "overview")
    )
    if record_open_gaps:
        missing = payload.get("missing_detail")
        _append_open_gap(
            memory,
            source_tool=tool_name,
            gap=missing,
            suggested_window=payload.get("suggest_focus_window") or payload.get("t_range"),
            scene_id=payload.get("scene_id"),
            persistent=persistent_open_gaps,
        )
        for scene in payload.get("scene_summaries") or []:
            if not isinstance(scene, dict):
                continue
            focus_windows = scene.get("suggest_focus_windows") or scene.get("suggest_focus_window") or []
            suggested = focus_windows[0] if focus_windows and isinstance(focus_windows[0], (list, tuple)) else focus_windows
            _append_open_gap(
                memory,
                source_tool=tool_name,
                gap=scene.get("missing_detail"),
                suggested_window=suggested or scene.get("t_range"),
                scene_id=scene.get("scene_id"),
                persistent=persistent_open_gaps,
            )
        for item in payload.get("open_gaps") or []:
            if isinstance(item, dict):
                _append_open_gap(
                    memory,
                    source_tool=str(item.get("source_tool") or tool_name),
                    gap=item.get("gap") or item.get("missing_detail"),
                    suggested_window=item.get("suggested_window"),
                    scene_id=item.get("scene_id"),
                    persistent=persistent_open_gaps,
                )
            elif item:
                _append_open_gap(
                    memory,
                    source_tool=tool_name,
                    gap=item,
                    persistent=persistent_open_gaps,
                )

    if persistent_open_gaps:
        _resolve_overlapping_gaps(
            memory,
            tool_name=tool_name,
            payload=payload,
            semantic_resolution=semantic_gap_resolution,
        )

    if use_structured_evidence_state:
        update_structured_evidence_state(
            memory,
            tool_name=tool_name,
            parameters=parameters,
            payload=payload,
            question_context=question_context,
            include_evidence_scope=include_evidence_scope,
            scope_aware_evidence=scope_aware_evidence_memory,
            preserve_aggregate_verifier_decision=preserve_aggregate_verifier_decision,
        )

    return memory


def format_memory_for_prompt(
    memory: dict[str, Any] | None,
    *,
    question: str | None = None,
    max_observations: int = 18,
    max_scenes: int = 10,
    max_gaps: int = 8,
    include_compact_planner_state: bool = False,
    include_structured_evidence: bool = True,
    max_structured_evidence_items: int = 10,
    include_compact_event_coverage: bool = False,
    max_event_coverage_items: int = 8,
    include_timeline_view: bool = True,
    include_object_state_view: bool = True,
    include_candidate_frontier: bool = False,
    max_candidate_frontier_items: int = 16,
    candidate_frontier_include_timestamp_cues: bool = True,
    candidate_frontier_timestamp_cue_radius_s: float = 6.0,
    candidate_frontier_include_evidence_scope: bool = True,
    candidate_frontier_include_temporal_boundaries: bool = True,
    candidate_frontier_temporal_boundary_max_gap_s: float = 20.0,
    candidate_frontier_include_investigation_state: bool = False,
    candidate_frontier_max_investigation_candidates: int = 12,
    include_evidence_episode_frontier: bool = False,
    max_evidence_episode_items: int = 18,
    evidence_episode_anchor_radius_s: float = 6.0,
    evidence_episode_focus_window_s: float = 20.0,
    scope_aware_evidence_memory: bool = False,
    separate_routing_candidates: bool = False,
    query_relevant_retention: bool = False,
    persistent_candidate_pool: bool = False,
    persistent_candidate_pool_max_items: int = 20,
    persistent_candidate_pool_timestamp_radius_s: float = 8.0,
    candidate_evidence_projection: bool = False,
    include_temporal_evidence_ledger: bool = False,
    max_temporal_evidence_events: int = 16,
    max_temporal_evidence_boundaries: int = 8,
) -> str:
    if not memory:
        return "(empty)"

    lines: list[str] = []
    if include_temporal_evidence_ledger:
        ledger_text = format_temporal_evidence_ledger_for_prompt(
            memory,
            max_events=max_temporal_evidence_events,
            max_boundaries=max_temporal_evidence_boundaries,
        )
        if ledger_text:
            lines.append(ledger_text)
    if include_compact_event_coverage:
        event_coverage_text = format_event_coverage_for_prompt(
            memory,
            question=question,
            max_items=max_event_coverage_items,
        )
        if event_coverage_text:
            lines.append(event_coverage_text)
    if include_evidence_episode_frontier:
        refresh_evidence_episode_frontier(
            memory,
            question=question,
            max_episodes=max_evidence_episode_items + 48,
            anchor_radius_s=evidence_episode_anchor_radius_s,
            focus_window_s=evidence_episode_focus_window_s,
        )
        episode_text = format_evidence_episode_frontier_for_prompt(
            memory,
            max_episodes=max_evidence_episode_items,
        )
        if episode_text:
            lines.append(episode_text)
    if include_candidate_frontier:
        refresh_candidate_frontier(
            memory,
            question=question,
            max_candidates=max_candidate_frontier_items + 8,
            include_timestamp_cues=candidate_frontier_include_timestamp_cues,
            timestamp_cue_radius_s=candidate_frontier_timestamp_cue_radius_s,
            include_evidence_scope=candidate_frontier_include_evidence_scope,
            include_temporal_boundaries=candidate_frontier_include_temporal_boundaries,
            temporal_boundary_max_gap_s=candidate_frontier_temporal_boundary_max_gap_s,
        )
        frontier_text = format_candidate_frontier_for_prompt(
            memory,
            max_candidates=max_candidate_frontier_items,
            include_investigation_state=candidate_frontier_include_investigation_state,
            max_investigation_candidates=candidate_frontier_max_investigation_candidates,
        )
        if frontier_text:
            lines.append(frontier_text)
    if include_compact_planner_state:
        compact_text = format_compact_investigation_state_for_prompt(
            memory,
            question=question,
            max_windows=max_scenes,
            max_facts=max_observations,
            max_gaps=max_gaps,
            max_targets=max_gaps,
            scope_aware_evidence_memory=scope_aware_evidence_memory,
            separate_routing_candidates=separate_routing_candidates,
            query_relevant_retention=query_relevant_retention,
            persistent_candidate_pool=persistent_candidate_pool,
            persistent_candidate_pool_max_items=persistent_candidate_pool_max_items,
            persistent_candidate_pool_timestamp_radius_s=persistent_candidate_pool_timestamp_radius_s,
            candidate_evidence_projection=candidate_evidence_projection,
        )
        if compact_text:
            lines.append(compact_text)
    task_memory = format_task_trajectory_memory(
        memory,
        question,
        include_timeline_view=include_timeline_view,
        include_object_state_view=include_object_state_view,
    )
    if task_memory:
        lines.append(task_memory)
    if include_structured_evidence:
        structured_text = format_structured_evidence_for_prompt(
            memory,
            max_items=max_structured_evidence_items,
        )
        if structured_text:
            lines.append(structured_text)

    candidate_bindings = [
        item
        for item in (memory.get("candidate_binding_memory") or [])
        if isinstance(item, dict)
    ][-max_scenes:]
    if candidate_bindings:
        lines.append("Candidate target bindings:")
        for item in candidate_bindings:
            lines.append(
                f"- {item.get('candidate_id')} {item.get('t_range')}: "
                f"target_match={item.get('target_match') or 'unknown'} "
                f"event_match={item.get('event_match') or 'unknown'} "
                f"fact={str(item.get('observed_fact') or '')[:180]} "
                f"binding={str(item.get('target_binding_reason') or '')[:140]} "
                f"support={item.get('supports_options') or []} "
                f"contradict={item.get('contradicts_options') or []} "
                f"option_conflict={item.get('option_set_conflict') is True}"
            )

    observations = sorted(
        memory.get("timestamped_observations") or [],
        key=lambda item: _safe_float(item.get("timestamp_s")) or 0.0,
    )
    if observations:
        lines.append("Observed timeline:")
        shown_observations = (
            _select_query_relevant_items(
                observations,
                question=question,
                limit=max_observations,
            )
            if query_relevant_retention
            else observations[-max_observations:]
        )
        for item in shown_observations:
            ts = _safe_float(item.get("timestamp_s")) or 0.0
            source = item.get("source_tool") or "tool"
            scene = item.get("scene_id") or "?"
            candidate_id = str(item.get("candidate_id") or "").strip()
            desc = str(item.get("description") or "").strip()
            need = str(item.get("needs_focus") or "").strip()
            suffix = f" need_focus={need}" if need else ""
            scope = item.get("evidence_scope")
            scope_text = f" scope={scope}" if scope else ""
            candidate_text = f" candidate={candidate_id}" if candidate_id else ""
            target_text = (
                f" target_match={item.get('target_match')}"
                if candidate_id and item.get("target_match")
                else ""
            )
            event_text = (
                f" event_match={item.get('event_match')}"
                if candidate_id and item.get("event_match")
                else ""
            )
            lines.append(
                f"- {source} {scene} {ts:.1f}s{candidate_text}: "
                f"{desc}{suffix}{target_text}{event_text}"
            )
            if scope_text:
                lines[-1] += scope_text

    all_scenes = list(memory.get("scene_memory") or [])
    scenes = (
        _select_query_relevant_items(
            all_scenes,
            question=question,
            limit=max_scenes,
        )
        if query_relevant_retention
        else all_scenes[-max_scenes:]
    )
    if scenes:
        lines.append("Scene/window memory:")
        for item in scenes:
            scene = item.get("scene_id") or "?"
            window = item.get("window_id")
            span = item.get("t_range")
            target = f"{scene}/{window}" if window else scene
            coverage = item.get("requirement_coverage") or {}
            coverage_text = ""
            if isinstance(coverage, dict) and coverage:
                coverage_text = (
                    f" req_matched={coverage.get('matched_requirements') or []}"
                    f" req_missing={coverage.get('missing_requirements') or []}"
                    f" relation_verified={coverage.get('relation_verified')}"
                )
            lines.append(
                f"- {target} {span}: {item.get('summary')} "
                f"scope={item.get('evidence_scope') or 'unknown'} "
                f"possible={item.get('possible_evidence')} detail_sufficient={item.get('detail_sufficient')} "
                f"support={item.get('supports_options') or []} contradict={item.get('contradicts_options') or []} "
                f"focus={item.get('suggest_focus_windows')} missing={item.get('missing_detail')}"
                f"{coverage_text}"
            )

    gaps = list(memory.get("open_gaps") or [])[-max_gaps:]
    if gaps:
        lines.append("Open gaps:")
        for item in gaps:
            lines.append(
                f"- {item.get('gap')} suggested_window={item.get('suggested_window')}"
            )

    return "\n".join(lines) if lines else "(empty)"


def format_evidence_for_answer(
    memory: dict[str, Any] | None,
    *,
    question: str | None = None,
    max_scenes: int = 8,
    max_observations_per_scene: int = 8,
    max_summaries_per_scene: int = 4,
    include_compact_planner_state: bool = False,
    include_structured_evidence: bool = True,
    max_structured_evidence_items: int = 16,
    include_timeline_view: bool = True,
    include_object_state_view: bool = True,
    scope_aware_evidence_memory: bool = False,
    separate_routing_candidates: bool = False,
    query_relevant_retention: bool = False,
    include_temporal_evidence_ledger: bool = False,
    max_temporal_evidence_events: int = 16,
) -> str:
    """Compact evidence digest for final answer prompts.

    This is a memory organization layer, not a decision rule. It groups
    observations by scene so evidence found earlier is not hidden by recency.
    """
    if not memory:
        return "(empty)"

    observations = [
        item
        for item in memory.get("timestamped_observations") or []
        if isinstance(item, dict)
    ]
    scenes = [
        item
        for item in memory.get("scene_memory") or []
        if isinstance(item, dict)
    ]

    scene_ids: list[str] = []
    for item in scenes + observations:
        scene_id = str(item.get("scene_id") or "global")
        if scene_id not in scene_ids:
            scene_ids.append(scene_id)

    def scene_score(scene_id: str) -> tuple[int, float]:
        scene_obs = [item for item in observations if str(item.get("scene_id") or "global") == scene_id]
        scene_summaries = [item for item in scenes if str(item.get("scene_id") or "global") == scene_id]
        latest_ts = max((_safe_float(item.get("timestamp_s")) or 0.0 for item in scene_obs), default=0.0)
        evidence_count = len(scene_obs) + len(scene_summaries)
        return evidence_count, latest_ts

    ranked_scene_ids = sorted(scene_ids, key=scene_score, reverse=True)[:max_scenes]
    lines: list[str] = [
        "Evidence digest grouped by scene/window:",
        "Use this digest to compare all collected evidence, not only the most recent observation.",
    ]
    if include_temporal_evidence_ledger:
        ledger_text = format_temporal_evidence_ledger_for_prompt(
            memory,
            max_events=max_temporal_evidence_events,
            max_boundaries=6,
        )
        if ledger_text:
            lines.append(ledger_text)
    episode_state = memory.get("evidence_episode_frontier")
    if isinstance(episode_state, dict) and episode_state.get("episodes"):
        episode_text = format_evidence_episode_frontier_for_answer(
            memory,
            max_episodes=max_scenes + 4,
        )
        if episode_text:
            lines.append(episode_text)
    if include_compact_planner_state:
        compact_text = format_compact_investigation_state_for_prompt(
            memory,
            question=question,
            max_windows=max_scenes,
            max_facts=max_observations_per_scene * 2,
            max_events=max_summaries_per_scene * 2,
            max_gaps=8,
            max_targets=8,
            scope_aware_evidence_memory=scope_aware_evidence_memory,
            separate_routing_candidates=separate_routing_candidates,
            query_relevant_retention=query_relevant_retention,
        )
        if compact_text:
            lines.append(compact_text)
    task_memory = format_task_trajectory_memory(
        memory,
        question,
        max_lines=36,
        include_timeline_view=include_timeline_view,
        include_object_state_view=include_object_state_view,
    )
    if task_memory:
        lines.append(task_memory)
    if include_structured_evidence:
        structured_text = format_structured_evidence_for_answer(
            memory,
            max_items=max_structured_evidence_items,
        )
        if structured_text:
            lines.append(structured_text)

    for scene_id in ranked_scene_ids:
        scene_summaries = [
            item
            for item in scenes
            if str(item.get("scene_id") or "global") == scene_id
        ][-max_summaries_per_scene:]
        scene_obs = sorted(
            [
                item
                for item in observations
                if str(item.get("scene_id") or "global") == scene_id
            ],
            key=lambda item: _safe_float(item.get("timestamp_s")) or 0.0,
        )[-max_observations_per_scene:]
        if not scene_summaries and not scene_obs:
            continue

        lines.append(f"Scene {scene_id}:")
        for item in scene_summaries:
            source = item.get("source_tool") or "tool"
            backend = item.get("observer_backend")
            backend_text = f"/{backend}" if backend else ""
            window = item.get("window_id") or "window"
            span = item.get("t_range")
            summary = str(item.get("summary") or "").strip()
            possible = item.get("possible_evidence")
            focus = item.get("suggest_focus_windows") or []
            missing = str(item.get("missing_detail") or "").strip()
            detail_sufficient = item.get("detail_sufficient")
            supports = item.get("supports_options") or []
            contradicts = item.get("contradicts_options") or []
            coverage = item.get("requirement_coverage") or {}
            coverage_text = ""
            if isinstance(coverage, dict) and coverage:
                coverage_text = (
                    f" req_matched={coverage.get('matched_requirements') or []}"
                    f" req_missing={coverage.get('missing_requirements') or []}"
                    f" relation_verified={coverage.get('relation_verified')}"
                )
            lines.append(
                f"- summary {source}{backend_text} {window} {span}: "
                f"{summary} scope={item.get('evidence_scope') or 'unknown'} "
                f"possible_evidence={possible} detail_sufficient={detail_sufficient} "
                f"support={supports} contradict={contradicts} focus={focus} missing={missing}"
                f"{coverage_text}"
            )
        for item in scene_obs:
            ts = _safe_float(item.get("timestamp_s")) or 0.0
            source = item.get("source_tool") or "tool"
            backend = item.get("observer_backend")
            backend_text = f"/{backend}" if backend else ""
            window = item.get("window_id") or "window"
            desc = str(item.get("description") or "").strip()
            tags = item.get("event_tags") or []
            need = str(item.get("needs_focus") or "").strip()
            need_text = f" need_focus={need}" if need else ""
            lines.append(
                f"- obs {source}{backend_text} {window} {ts:.1f}s: "
                f"{desc} scope={item.get('evidence_scope') or 'unknown'} tags={tags}{need_text}"
            )

    if len(lines) <= 2:
        return "(empty)"
    return "\n".join(lines)
