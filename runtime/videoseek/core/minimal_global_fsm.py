"""Pure deterministic core for P130 Count/Order reasoning.

The module deliberately has no dependency on the agent, memory projections, or
LLM adapters.  Callers pass local verifier receipts (or the temporal ledger),
and receive a canonical event table plus a reducer state.  Local option claims
are never copied into that table.

Two boundaries are explicit:

* A requested/search window is not a coverage certificate.  Count enumeration
  is complete only when the caller passes ``enumeration_complete=True``.
* A forced terminal prediction is never relabelled as a validated decision.

All public functions return new JSON-serialisable values and do not mutate
their inputs.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "p130_minimal_global_fsm_v1"
RECOVERY_SIGNATURE_VERSION = "p130_recovery_signature_v1"

_COUNT_QUESTION_RE = re.compile(
    r"(?:\bhow\s+many\b|\btotal\s+(?:count|number)\b|"
    r"\bcount\s+of\s+(?:occurrences?|times?|instances?|events?|scenes?)\b|"
    r"\bnumber\s+of\s+(?:occurrences?|times?|instances?|events?|scenes?)\b|"
    r"\bhow\s+often\b|多少次|几次|总数|总共(?:出现|发生))",
    re.IGNORECASE,
)
_ORDER_QUESTION_RE = re.compile(
    r"(?:\bchronological\s+order\b|\bcorrect\s+(?:chronological\s+)?order\b|"
    r"\bactual\s+order\b|\barrange\b[^?.\n]{0,100}\bevents?\b|"
    r"先后顺序|时间顺序|按.*顺序)",
    re.IGNORECASE,
)
_CHOICE_MARKER_RE = re.compile(
    r"(?im)(?:^|\n)[ \t]*[\(\[]?([A-H])[\)\].:]\s*"
)
_NUMBERED_EVENT_MARKER_RE = re.compile(r"\((\d+)\)\s*")
_ARROW_RE = re.compile(r"\s*(?:--?>|→|⇒)\s*")
_FINAL_INSTRUCTION_RE = re.compile(
    r"(?im)\n\s*(?:please\s+(?:directly\s+)?answer|"
    r"if\s+the\s+question\s+is|answer\s+(?:only\s+)?with|"
    r"respond\s+(?:only\s+)?with).*$",
    re.DOTALL,
)
_ANSWER_RE = re.compile(r"(?i)(?:^|\b)(?:option\s*)?\(?([A-H])\)?(?:\b|$)")
_GLOBAL_COUNT_SCOPE_RE = re.compile(
    r"(?:\b(?:times?|occurrences?|instances?)\b|\bhow\s+often\b|"
    r"\bappear(?:s|ed|ing)?\b[^?.\n]{0,80}\bin\s+total\b|"
    r"\bthroughout\s+(?:this|the)\s+video\b[^?.\n]{0,120}"
    r"\b(?:total\s+count|count\s+of|how\s+many)\b|"
    r"(?:出现|发生).{0,20}(?:多少次|几次|总数|总共))",
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    "zero": 0,
    "none": 0,
    "never": 0,
    "one": 1,
    "once": 1,
    "two": 2,
    "twice": 2,
    "three": 3,
    "thrice": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

_NON_DIRECT_MATCHES = {
    "ambiguous",
    "context",
    "context_only",
    "different_event",
    "mismatch",
    "not_visible",
    "negative",
    "partial",
    "unknown",
}
_DIRECT_BINDING_CONFIDENCE = {"direct", "exact", "high", "verified"}
_P131_ORDER_SAME_EVENT_MAX_ANCHOR_GAP_S = 18.0


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _window(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    start, end = _safe_float(value[0]), _safe_float(value[1])
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return [round(start, 3), round(end, 3)]


def _normalise_id(value: Any) -> str:
    text = str(value or "").strip()
    numbered = re.fullmatch(r"(?:event\s*)?\(?([0-9]+)\)?", text, re.IGNORECASE)
    if numbered:
        return numbered.group(1)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _answer_letter(value: Any) -> str:
    """Parse only the complete option token, never a letter inside prose."""
    text = str(value or "").strip()
    match = re.fullmatch(r"\(?([A-H])\)?[.]?", text, re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _count_value(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    digit = re.search(r"(?<![\d.])-?\d+(?![\d.])", text)
    if digit:
        number = int(digit.group(0))
        return number if number >= 0 else None
    for token in re.findall(r"[a-z]+|[零一二两三四五六七八九十]+", text):
        if token in _NUMBER_WORDS:
            return _NUMBER_WORDS[token]
    return None


def _order_sequence(value: Any) -> list[str]:
    text = str(value or "").strip()
    pieces = _ARROW_RE.split(text)
    if len(pieces) < 2:
        pieces = re.split(r"\s*[,;]\s*", text)
    sequence = [_normalise_id(piece) for piece in pieces]
    sequence = [piece for piece in sequence if piece]
    return sequence if len(sequence) >= 2 else []


def detect_global_mode(question: str | None) -> str:
    """Return ``count``, ``order``, or ``other`` for supported global modes."""

    text = str(question or "")
    if _COUNT_QUESTION_RE.search(text):
        return "count"
    if _ORDER_QUESTION_RE.search(text):
        return "order"
    return "other"


def _choice_rows(question: str) -> tuple[str, list[dict[str, Any]]]:
    matches = list(_CHOICE_MARKER_RE.finditer(question))
    if not matches:
        return question.strip(), []
    stem = question[: matches[0].start()].strip()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        letter = match.group(1).upper()
        if letter in seen:
            continue
        seen.add(letter)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        text = question[match.end() : end].strip()
        text = _FINAL_INSTRUCTION_RE.sub("", "\n" + text).strip()
        rows.append({"option": letter, "text": text})
    return stem, rows


def _numbered_events(stem: str) -> list[dict[str, str]]:
    markers = list(_NUMBERED_EVENT_MARKER_RE.finditer(stem))
    rows: list[dict[str, str]] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(stem)
        description = stem[marker.end() : end].strip(" ;\n\t")
        if description:
            rows.append({"event_id": marker.group(1), "description": description})
    return rows


def _named_option_events(options: list[dict]) -> list[dict[str, str]]:
    """Recover only a shared permutation catalog; never infer synonymous events."""
    sequences = [_ARROW_RE.split(row["text"].strip()) for row in options]
    if len(sequences) < 2:
        return []
    catalogs = [{_normalise_id(text) for text in sequence} for sequence in sequences]
    if any(len(sequence) < 2 or len(catalog) != len(sequence)
           or catalog != catalogs[0] for sequence, catalog in zip(sequences, catalogs)):
        return []
    descriptions: dict[str, str] = {}
    for sequence in sequences:
        for text in sequence:
            text = text.strip()
            key = _normalise_id(text)
            if not key or key.isdigit():
                return []
            # Punctuation collisions are not evidence of event identity.
            if key in descriptions and descriptions[key].casefold() != text.casefold():
                return []
            descriptions[key] = min(descriptions.get(key, text), text)
    return [{"event_id": str(i + 1), "description": descriptions[key]}
            for i, key in enumerate(sorted(descriptions))]


def parse_global_question(question: str | None) -> dict[str, Any]:
    """Parse the mode, choices, numeric counts, and numbered order events.

    The final answer instruction is removed from the last choice.  Parse errors
    are reported instead of guessed so the caller can retain P128 behaviour.
    """

    text = str(question or "")
    mode = detect_global_mode(text)
    stem, options = _choice_rows(text)
    required_events = _numbered_events(stem) if mode == "order" else []
    named_events = _named_option_events(options) if mode != "count" and not required_events else []
    if named_events:
        mode = "order"
        required_events = named_events
    named_ids = {_normalise_id(row["description"]): row["event_id"] for row in named_events}
    parse_errors: list[str] = []
    enriched_options: list[dict[str, Any]] = []
    for row in options:
        enriched = deepcopy(row)
        if mode == "count":
            enriched["numeric_value"] = _count_value(row["text"])
            if enriched["numeric_value"] is None:
                parse_errors.append(f"count_value_missing:{row['option']}")
        elif mode == "order":
            enriched["sequence"] = _order_sequence(row["text"])
            if named_ids:
                enriched["sequence"] = [named_ids[key] for key in enriched["sequence"]]
            if not enriched["sequence"]:
                parse_errors.append(f"order_sequence_missing:{row['option']}")
        enriched_options.append(enriched)
    if mode in {"count", "order"} and not enriched_options:
        parse_errors.append("choices_missing")
    if mode == "order" and not required_events:
        parse_errors.append("required_events_missing")
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "question": text,
        "stem": stem,
        "options": enriched_options,
        "required_events": required_events,
        "required_event_ids": [row["event_id"] for row in required_events],
        "parse_errors": parse_errors,
    }


def should_use_minimal_global_fsm(question: str | None) -> bool:
    """Return true only for P130's narrow, fully parseable global contract.

    Generic object/person cardinality questions such as ``How many people ...``
    remain on the frozen P128 path.  P130 Count is reserved for repeated event
    occurrences across the video, and P130 Order requires an explicit numbered
    event catalog or a shared named-event permutation catalog in the choices.
    """

    parsed = parse_global_question(question)
    if parsed.get("parse_errors"):
        return False
    mode = parsed.get("mode")
    if mode == "count":
        numeric = [
            row.get("numeric_value")
            for row in parsed.get("options") or []
            if isinstance(row.get("numeric_value"), int)
        ]
        return len(numeric) >= 2 and bool(_GLOBAL_COUNT_SCOPE_RE.search(str(question or "")))
    if mode == "order":
        required = set(parsed.get("required_event_ids") or [])
        if len(required) < 2:
            return False
        return all(
            required.issubset(set(row.get("sequence") or []))
            for row in parsed.get("options") or []
        )
    return False

















def _overlap_ratio(left: Any, right: Any) -> float:
    a, b = _window(left), _window(right)
    if a is None or b is None:
        return 0.0
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    shorter = max(0.001, min(a[1] - a[0], b[1] - b[0]))
    return overlap / shorter


def _compatible_target_ids(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    a = set(left.get("target_event_ids") or [])
    b = set(right.get("target_event_ids") or [])
    return not a or not b or a == b


def _span_gap(left: Any, right: Any) -> float:
    a, b = _window(left), _window(right)
    if a is None or b is None:
        return float("inf")
    if a[1] < b[0]:
        return b[0] - a[1]
    if b[1] < a[0]:
        return a[0] - b[1]
    return 0.0


def _closest_anchor_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a = [
        value
        for raw in left.get("anchor_timestamps") or []
        if (value := _safe_float(raw)) is not None
    ]
    b = [
        value
        for raw in right.get("anchor_timestamps") or []
        if (value := _safe_float(raw)) is not None
    ]
    if not a or not b:
        return float("inf")
    return min(abs(x - y) for x in a for y in b)


def _combined_anchor_span(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    values = [
        value
        for source in (left, right)
        for raw in source.get("anchor_timestamps") or []
        if (value := _safe_float(raw)) is not None
    ]
    return max(values) - min(values) if values else float("inf")


def _same_event(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left.get("match") != "direct" or right.get("match") != "direct":
        return False
    left_ids, right_ids = left.get("target_event_ids") or [], right.get("target_event_ids") or []
    overlap = _overlap_ratio(left.get("t_range"), right.get("t_range"))
    if left_ids or right_ids:
        return len(left_ids) == len(right_ids) == 1 and left_ids == right_ids and overlap > 0
    return overlap >= .5 or (
        _span_gap(left.get("t_range"), right.get("t_range")) <= 1.
        and _closest_anchor_distance(left, right) <= 3.
    )


def _merge_event(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(left))
    spans = [span for span in (_window(left.get("t_range")), _window(right.get("t_range"))) if span]
    if spans:
        merged["t_range"] = [min(span[0] for span in spans), max(span[1] for span in spans)]
    for key in (
        "receipt_ids",
        "frame_ids",
        "anchor_timestamps",
        "target_event_ids",
        "source_action_ids",
    ):
        values: list[Any] = []
        for value in list(left.get(key) or []) + list(right.get(key) or []):
            if value not in values:
                values.append(deepcopy(value))
        merged[key] = sorted(values) if key != "receipt_ids" else values
    facts = [str(value).strip() for value in (left.get("fact"), right.get("fact")) if str(value or "").strip()]
    merged["fact"] = " | ".join(dict.fromkeys(facts))
    return merged


def _event_sort_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    anchors = event.get("anchor_timestamps") or []
    anchor = _safe_float(anchors[0]) if anchors else None
    span = _window(event.get("t_range"))
    if anchor is None and span:
        anchor = (span[0] + span[1]) / 2.0
    return (
        float("inf") if anchor is None else anchor,
        str(event.get("event_key") or ""),
    )


def build_event_table(receipts: Any) -> list[dict[str, Any]]:
    """Create the canonical event table without copying local option claims.

    Inputs are canonical observations, already checked at the tool boundary.
    Historical output formats are not interpreted by this reducer.
    """

    table: list[dict[str, Any]] = []
    for raw in receipts:
        row = deepcopy(dict(raw))
        match_index = next(
            (index for index, prior in enumerate(table) if _same_event(prior, row)), None
        )
        if match_index is None:
            table.append(row)
        else:
            table[match_index] = _merge_event(table[match_index], row)
    return sorted(table, key=_event_sort_key)


def event_table_version(event_table: Any) -> str:
    """Return a stable evidence version for no-progress/recovery checks."""

    table = build_event_table(event_table)
    payload = [
        {
            "event_key": row.get("event_key"),
            "receipt_ids": sorted(row.get("receipt_ids") or []),
            "t_range": row.get("t_range"),
            "anchor_timestamps": row.get("anchor_timestamps") or [],
            "match": row.get("match"),
            "target_event_ids": sorted(row.get("target_event_ids") or []),
            "binding_confidence": row.get("binding_confidence"),
            "fact": row.get("fact"),
            "source_action_ids": sorted(row.get("source_action_ids") or []),
            "p131_order_run_ids": sorted(row.get("p131_order_run_ids") or []),
        }
        for row in table
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parsed(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    return parse_global_question(value)


def _base_state(parsed: Mapping[str, Any], table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": str(parsed.get("mode") or "other"),
        "options": deepcopy(list(parsed.get("options") or [])),
        "event_table": deepcopy(list(table)),
        "evidence_version": event_table_version(table),
        "parse_errors": deepcopy(list(parsed.get("parse_errors") or [])),
        "viable_options": [str(row.get("option") or "") for row in parsed.get("options") or []],
        "eliminated_options": [],
        "decision_sufficient": False,
        "validated_option": None,
    }


def reduce_count(
    question: str | Mapping[str, Any],
    receipts_or_table: Any,
    *,
    enumeration_complete: bool = False,
) -> dict[str, Any]:
    """Reduce direct non-overlapping occurrences to deterministic Count state.

    ``enumeration_complete`` is an explicit certificate supplied by the caller;
    no requested window, duration, or nominal coverage field can imply it.
    """

    parsed = _parsed(question)
    table = build_event_table(receipts_or_table)
    state = _base_state(parsed, table)
    direct_events = [row for row in table if row.get("match") == "direct"]
    lower_bound = len(direct_events)
    viable: list[str] = []
    eliminated: list[str] = []
    numeric_viable: list[tuple[str, int]] = []
    for row in parsed.get("options") or []:
        option = str(row.get("option") or "")
        value = row.get("numeric_value")
        if not isinstance(value, int):
            viable.append(option)
        elif value < lower_bound or (enumeration_complete and value != lower_bound):
            eliminated.append(option)
        else:
            viable.append(option)
            numeric_viable.append((option, value))

    # Under the multiple-choice contract, uncertainty maps to the same offered
    # answer only if every remaining offered numeric value is identical.
    viable_values = {value for _option, value in numeric_viable}
    non_numeric_viable = [
        option
        for option in viable
        if option not in {numeric_option for numeric_option, _value in numeric_viable}
    ]
    maps_same = (
        not enumeration_complete
        and not non_numeric_viable
        and len(viable) == 1
        and len(viable_values) == 1
    )
    sufficient = (
        not state["parse_errors"]
        and len(viable) == 1
        and (bool(enumeration_complete) or maps_same)
    )
    next_values = sorted(value for value in viable_values if value > lower_bound)
    state.update(
        {
            "mode": "count",
            "direct_events": deepcopy(direct_events),
            "observed_count_lower_bound": lower_bound,
            "enumeration_complete": bool(enumeration_complete),
            "possible_count_upper_bound": lower_bound if enumeration_complete else None,
            "viable_options": viable,
            "eliminated_options": eliminated,
            "remaining_uncertainty_maps_same_option": maps_same,
            "decision_sufficient": sufficient,
            "validated_option": viable[0] if sufficient else None,
            "recovery_need": None
            if sufficient
            else {
                "kind": "residual_count_search",
                "exclude_spans": [row.get("t_range") for row in direct_events if row.get("t_range")],
                "next_offered_count": next_values[0] if next_values else None,
            },
        }
    )
    return state


def _binding_anchor(event: Mapping[str, Any]) -> float | None:
    anchors = event.get("anchor_timestamps") or []
    for value in anchors:
        anchor = _safe_float(value)
        if anchor is not None:
            return anchor
    span = _window(event.get("t_range"))
    return (span[0] + span[1]) / 2.0 if span else None


def _direct_order_binding(event: Mapping[str, Any], required: set[str]) -> str | None:
    ids = event.get("target_event_ids") or []
    return ids[0] if event.get("match") == "direct" and len(ids) == 1 and ids[0] in required else None


def _precedes(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Missing temporal extent cannot create an ordering assertion."""
    left_span, right_span = _window(left.get("t_range")), _window(right.get("t_range"))
    return bool(left_span and right_span and left_span[1] <= right_span[0])


def reduce_order(
    question: str | Mapping[str, Any],
    receipts_or_table: Any,
) -> dict[str, Any]:
    """Reduce explicit local event-ID bindings to chronological constraints."""

    parsed = _parsed(question)
    table = build_event_table(receipts_or_table)
    state = _base_state(parsed, table)
    required_ids = [str(value) for value in parsed.get("required_event_ids") or []]
    required = set(required_ids)
    candidates: dict[str, list[dict[str, Any]]] = {event_id: [] for event_id in required_ids}
    ambiguous_receipts: list[dict[str, Any]] = []
    for event in table:
        event_id = _direct_order_binding(event, required)
        if event_id is None:
            if event.get("target_event_ids"):
                ambiguous_receipts.append(deepcopy(event))
            continue
        if _binding_anchor(event) is not None:
            candidates[event_id].append(deepcopy(event))

    bound: dict[str, dict[str, Any]] = {}
    binding_conflicts: list[dict[str, Any]] = []
    for event_id in required_ids:
        rows = candidates.get(event_id) or []
        if len(rows) == 1:
            bound[event_id] = rows[0]
        elif len(rows) > 1:
            binding_conflicts.append(
                {
                    "event_id": event_id,
                    "reason": "multiple_nonoverlapping_direct_bindings",
                    "event_keys": [row.get("event_key") for row in rows],
                }
            )

    ordered = sorted(
        (
            {
                "event_id": event_id,
                "timestamp_s": _binding_anchor(event),
                "event": deepcopy(event),
            }
            for event_id, event in bound.items()
        ),
        key=lambda row: (row["timestamp_s"], row["event_id"]),
    )
    precedence_edges: list[list[str]] = []
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if _precedes(left["event"], right["event"]):
                precedence_edges.append([left["event_id"], right["event_id"]])

    viable: list[str] = []
    eliminated: list[str] = []
    option_sequences: dict[str, list[str]] = {}
    for row in parsed.get("options") or []:
        option = str(row.get("option") or "")
        sequence = [str(value) for value in row.get("sequence") or []]
        option_sequences[option] = sequence
        positions = {event_id: index for index, event_id in enumerate(sequence)}
        parse_ok = bool(sequence) and all(event_id in positions for event_id in required_ids)
        consistent = parse_ok and all(
            positions.get(left, -1) < positions.get(right, -1)
            for left, right in precedence_edges
        )
        if consistent:
            viable.append(option)
        else:
            eliminated.append(option)

    unbound = [event_id for event_id in required_ids if event_id not in bound]
    coverage_closed = bool(required_ids) and not unbound and not binding_conflicts
    # Order is an exact global claim.  Partial precedence can eliminate options,
    # but it cannot validate the survivor until every requested event has one
    # unambiguous direct binding.
    sufficient = (
        not state["parse_errors"]
        and coverage_closed
        and len(viable) == 1
        and bool(precedence_edges)
    )
    state.update(
        {
            "mode": "order",
            "required_event_ids": required_ids,
            "bound_event_ids": [event_id for event_id in required_ids if event_id in bound],
            "unbound_event_ids": unbound,
            "event_bindings": {
                event_id: deepcopy(bound[event_id]) for event_id in required_ids if event_id in bound
            },
            "binding_conflicts": binding_conflicts,
            "ambiguous_receipts": ambiguous_receipts,
            "ordered_events": ordered,
            "precedence_edges": precedence_edges,
            "option_sequences": option_sequences,
            "viable_options": viable,
            "eliminated_options": eliminated,
            "coverage_closed": coverage_closed,
            "decision_sufficient": sufficient,
            "validated_option": viable[0] if sufficient else None,
            "recovery_need": None
            if sufficient
            else {
                "kind": "bind_order_events",
                "target_event_ids": unbound,
                "binding_conflict_event_ids": [row["event_id"] for row in binding_conflicts],
            },
        }
    )
    return state


def reduce_global(
    question: str | Mapping[str, Any],
    receipts_or_table: Any,
    *,
    enumeration_complete: bool = False,
) -> dict[str, Any]:
    """Dispatch to the supported reducer and leave other questions untouched."""

    parsed = _parsed(question)
    mode = str(parsed.get("mode") or "other")
    if mode == "count":
        return reduce_count(parsed, receipts_or_table, enumeration_complete=enumeration_complete)
    if mode == "order":
        return reduce_order(parsed, receipts_or_table)
    table = build_event_table(receipts_or_table)
    state = _base_state(parsed, table)
    state.update(
        {
            "decision_sufficient": False,
            "validated_option": None,
            "delegated_to_p128": True,
        }
    )
    return state


def make_recovery_signature(
    mode: str,
    *,
    target_event_ids: Iterable[Any] | None = None,
    windows: Iterable[Any] | None = None,
    evidence_version: Any = 0,
) -> str:
    """Build a query-wording-independent signature for no-progress detection."""

    ids = sorted(
        {
            event_id
            for value in (target_event_ids or [])
            if (event_id := _normalise_id(value))
        }
    )
    normalised_windows = sorted(
        {
            tuple(window)
            for value in (windows or [])
            if (window := _window(value)) is not None
        }
    )
    payload = {
        "version": RECOVERY_SIGNATURE_VERSION,
        "mode": str(mode or "other").strip().lower(),
        "target_event_ids": ids,
        "windows": [list(window) for window in normalised_windows],
        "evidence_version": str(evidence_version),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{RECOVERY_SIGNATURE_VERSION}:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def recovery_allowed(signature: str, prior_recoveries: Iterable[Any] | None) -> bool:
    """Return false when the same semantic recovery already ran at this version."""

    seen: set[str] = set()
    for row in prior_recoveries or []:
        if isinstance(row, Mapping):
            value = row.get("signature") or row.get("recovery_signature")
        else:
            value = row
        if value is not None:
            seen.add(str(value))
    return str(signature) not in seen


def validate_answer(answer: Any, state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a validated decision only when the reducer proved one option."""

    source = deepcopy(dict(state or {}))
    proposed = _answer_letter(answer)
    validated = _answer_letter(source.get("validated_option"))
    sufficient = source.get("decision_sufficient") is True and bool(validated)
    if not sufficient:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "proposed_option": proposed or None,
            "selected_option": None,
            "validated": False,
            "forced": False,
            "reason": "decision_insufficient",
            "viable_options": deepcopy(list(source.get("viable_options") or [])),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "proposed_option": proposed or None,
        "selected_option": validated,
        "validated": True,
        "forced": False,
        "overrode_proposal": bool(proposed and proposed != validated),
        "reason": "deterministic_reducer",
        "viable_options": deepcopy(list(source.get("viable_options") or [])),
    }


def force_answer(
    answer: Any,
    state: Mapping[str, Any] | None,
    *,
    reason: str,
) -> dict[str, Any]:
    """Label a terminal best-effort answer without claiming validation.

    The proposed answer is accepted only when it is viable.  If it is not and
    exactly one viable option remains, that option is selected.  Multiple
    alternatives are returned to the caller instead of choosing one by order.
    """

    source = deepcopy(dict(state or {}))
    proposed = _answer_letter(answer)
    viable = [
        letter
        for value in source.get("viable_options") or []
        if (letter := _answer_letter(value))
    ]
    if proposed and proposed in viable:
        selected = proposed
    elif len(viable) == 1:
        selected = viable[0]
    else:
        selected = ""
    if not selected:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "forced_unresolved",
            "proposed_option": proposed or None,
            "selected_option": None,
            "validated": False,
            "forced": False,
            "reason": str(reason),
            "viable_options": viable,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "forced",
        "proposed_option": proposed or None,
        "selected_option": selected,
        "validated": False,
        "forced": True,
        "reason": str(reason),
        "viable_options": viable,
    }


__all__ = [
    "RECOVERY_SIGNATURE_VERSION",
    "SCHEMA_VERSION",
    "build_event_table",
    "detect_global_mode",
    "event_table_version",
    "force_answer",
    "make_recovery_signature",
    "parse_global_question",
    "recovery_allowed",
    "reduce_count",
    "reduce_global",
    "reduce_order",
    "should_use_minimal_global_fsm",
    "validate_answer",
]
