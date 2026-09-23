"""Pure P105 Overview receipt, admissibility, and action identity helpers.

The functions in this module do not select an action, invoke a tool, or merge
visual evidence.  They only maintain the lazily-created execution audit state
authorized by the P105 protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from typing import Any


STATE_KEY = "overview_execution_state"
STATE_SCHEMA_VERSION = "p105_overview_admissibility_v1"
ACTION_SIGNATURE_VERSION = "p105_action_signature_v1"
BLOCKED_FEEDBACK_PREFIX = "P105_OVERVIEW_ACTION_BLOCKED:\n"

_V10_JSON_BLOCK_RE = re.compile(
    r"V10_OBSERVATION_JSON\s*:\s*```json\s*(\{.*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)
_UNORDERED_PARAMETER_LISTS = {
    "candidate_ids",
    "candidate_windows",
    "event_ids",
    "requested_candidate_ids",
    "search_windows",
    "verified_candidate_ids",
    "verified_windows",
    "windows",
}
_RUNTIME_ONLY_PARAMETER_KEYS = {
    "duration",
    "memory",
    "output_dir",
    "question",
    "subtitles",
    "video_path",
    "vr",
}


def _new_state() -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "attempt_count": 0,
        "valid_attempt_count": 0,
        "failed_attempt_count": 0,
        "valid_receipt": None,
        "attempts": [],
        "blocked_overview_count": 0,
        "blocked_overviews": [],
    }


def _execution_state(memory: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(memory, dict):
        raise TypeError("memory must be a dict")
    state = memory.get(STATE_KEY)
    if state is None:
        state = _new_state()
        memory[STATE_KEY] = state
    elif not isinstance(state, dict):
        raise TypeError(f"memory[{STATE_KEY!r}] must be a dict")
    return state


def has_valid_overview_receipt(memory: dict[str, Any] | None) -> bool:
    """Return whether the first-valid immutable Overview receipt exists.

    This read-only query deliberately does not create execution state.
    """

    if not isinstance(memory, dict):
        return False
    state = memory.get(STATE_KEY)
    if not isinstance(state, dict):
        return False
    receipt = state.get("valid_receipt")
    return isinstance(receipt, dict) and receipt.get("status") == "valid"


def _parse_v10_payload(output: str | None) -> dict[str, Any] | None:
    if not isinstance(output, str) or not output:
        return None
    match = _V10_JSON_BLOCK_RE.search(output)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _count_value(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _valid_time_range(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    start = _finite_float(value[0])
    end = _finite_float(value[1])
    return start is not None and end is not None and end > start


def _classify_overview_output(
    output: str | None,
    *,
    require_all_timestamps: bool,
) -> dict[str, Any]:
    payload = _parse_v10_payload(output)
    if payload is None:
        return {
            "status": "failed",
            "failure_reasons": ["missing_or_unparseable_v10"],
            "v10_parse_ok": False,
            "global_summary_nonempty": False,
            "timestamp_count": 0,
            "valid_scene_summary_count": 0,
            "expected_timestamp_count": None,
            "observed_timestamp_count": None,
            "timestamp_coverage": None,
            "payload_sha256": None,
        }

    summary_ok = bool(str(payload.get("global_summary") or "").strip())
    timestamp_rows = [
        row
        for row in (payload.get("timestamp_observations") or [])
        if isinstance(row, dict)
    ]
    valid_scene_rows = [
        row
        for row in (payload.get("scene_summaries") or [])
        if isinstance(row, dict)
        and bool(str(row.get("summary") or "").strip())
        and _valid_time_range(row.get("t_range"))
    ]
    expected = _count_value(payload.get("overview_expected_timestamp_count"))
    observed = _count_value(payload.get("overview_observed_timestamp_count"))
    coverage = _finite_float(payload.get("overview_timestamp_coverage"))

    reasons: list[str] = []
    if not summary_ok:
        reasons.append("empty_global_summary")
    if not timestamp_rows:
        reasons.append("missing_timestamp_observation")
    if not valid_scene_rows:
        reasons.append("missing_valid_scene_summary")

    if require_all_timestamps:
        if expected is None or expected <= 0:
            reasons.append("invalid_expected_timestamp_count")
        if expected is not None and (observed is None or observed < expected):
            reasons.append("incomplete_observed_timestamp_count")
        if expected is not None and len(timestamp_rows) < expected:
            reasons.append("incomplete_rendered_timestamp_count")
        if coverage is None or coverage < 0.999:
            reasons.append("incomplete_timestamp_coverage")
    else:
        if observed is None or observed <= 0:
            reasons.append("nonpositive_observed_timestamp_count")
        if coverage is None or coverage <= 0.0:
            reasons.append("nonpositive_timestamp_coverage")

    return {
        "status": "failed" if reasons else "valid",
        "failure_reasons": reasons,
        "v10_parse_ok": True,
        "global_summary_nonempty": summary_ok,
        "timestamp_count": len(timestamp_rows),
        "valid_scene_summary_count": len(valid_scene_rows),
        "expected_timestamp_count": expected,
        "observed_timestamp_count": observed,
        "timestamp_coverage": coverage,
        "payload_sha256": _payload_hash(payload),
    }


def _step_value(step: Any) -> int | str:
    if isinstance(step, bool):
        return str(step).lower()
    try:
        return int(step)
    except (TypeError, ValueError, OverflowError):
        return re.sub(r"\s+", " ", str(step or "unknown")).strip() or "unknown"


def record_overview_attempt(
    memory: dict[str, Any],
    output: str | None,
    require_all_timestamps: bool,
    step: Any,
) -> dict[str, Any]:
    """Classify and append one real Overview execution attempt.

    Invalid attempts never create or clear a receipt.  The first valid receipt
    is copied once and is not overwritten by subsequent calls.
    """

    state = _execution_state(memory)
    classification = _classify_overview_output(
        output,
        require_all_timestamps=bool(require_all_timestamps),
    )
    attempt_number = int(state.get("attempt_count") or 0) + 1
    attempt = {
        "attempt_id": f"OA{attempt_number:04d}",
        "step": _step_value(step),
        "require_all_timestamps": bool(require_all_timestamps),
        **classification,
        "receipt_created": False,
    }

    state["attempt_count"] = attempt_number
    if attempt["status"] == "valid":
        state["valid_attempt_count"] = int(state.get("valid_attempt_count") or 0) + 1
        if not has_valid_overview_receipt(memory):
            receipt = {
                key: deepcopy(attempt[key])
                for key in (
                    "attempt_id",
                    "step",
                    "require_all_timestamps",
                    "status",
                    "v10_parse_ok",
                    "global_summary_nonempty",
                    "timestamp_count",
                    "valid_scene_summary_count",
                    "expected_timestamp_count",
                    "observed_timestamp_count",
                    "timestamp_coverage",
                    "payload_sha256",
                )
            }
            state["valid_receipt"] = receipt
            attempt["receipt_created"] = True
    else:
        state["failed_attempt_count"] = int(state.get("failed_attempt_count") or 0) + 1

    state.setdefault("attempts", []).append(attempt)
    state["last_attempt"] = deepcopy(attempt)
    return attempt


def record_blocked_overview(
    memory: dict[str, Any],
    step: Any,
    action_signature: str,
) -> dict[str, Any]:
    """Record a post-receipt Overview proposal blocked before tool execution."""

    if not has_valid_overview_receipt(memory):
        raise ValueError("cannot block Overview before a valid receipt exists")
    state = _execution_state(memory)
    block_number = int(state.get("blocked_overview_count") or 0) + 1
    signature = str(action_signature or "").strip()
    if not signature:
        signature = canonical_action_signature("overview", {})
    receipt = state["valid_receipt"]
    event = {
        "block_id": f"OB{block_number:04d}",
        "step": _step_value(step),
        "action_signature": signature,
        "valid_receipt_attempt_id": receipt.get("attempt_id"),
        "visual_observer_called": False,
        "merged_as_visual_evidence": False,
    }
    state["blocked_overview_count"] = block_number
    state.setdefault("blocked_overviews", []).append(event)
    state["last_blocked_overview"] = deepcopy(event)
    return event


def format_blocked_overview_feedback(memory: dict[str, Any]) -> str:
    """Return deterministic non-V10 feedback for an executor-blocked Overview."""

    if not has_valid_overview_receipt(memory):
        raise ValueError("blocked Overview feedback requires a valid receipt")
    state = memory[STATE_KEY]
    receipt = state["valid_receipt"]
    blocked = state.get("last_blocked_overview") or {}
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "tool": "overview",
        "status": "rejected",
        "executed": False,
        "visual_observer_called": False,
        "merged_as_visual_evidence": False,
        "reason": "valid_overview_already_completed",
        "valid_receipt": {
            "attempt_id": receipt.get("attempt_id"),
            "step": receipt.get("step"),
            "payload_sha256": receipt.get("payload_sha256"),
        },
        "blocked_proposal": {
            "block_id": blocked.get("block_id"),
            "step": blocked.get("step"),
            "action_signature": blocked.get("action_signature"),
        },
        "next_action": "choose_another_currently_allowed_tool",
    }
    return BLOCKED_FEEDBACK_PREFIX + json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_number(value: int | float) -> int | float | dict[str, str]:
    number = float(value)
    if not math.isfinite(number):
        return {"invalid_number": str(value)}
    if number.is_integer():
        return int(number)
    return round(number, 9)


def _canonical_value(value: Any, *, parent_key: str = "") -> Any:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value).strip()
        return value
    if isinstance(value, (int, float)):
        return _canonical_number(value)
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item, parent_key=str(key))
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
            if str(key) not in _RUNTIME_ONLY_PARAMETER_KEYS
        }
    if isinstance(value, (list, tuple, set)):
        # Only the outer collection is unordered. Preserve endpoint order and
        # duplicate invalid values inside each interval/candidate so malformed
        # inputs cannot collapse to the same legacy ``None:None`` identity.
        child_key = "" if parent_key in _UNORDERED_PARAMETER_LISTS else parent_key
        normalized = [_canonical_value(item, parent_key=child_key) for item in value]
        if parent_key in _UNORDERED_PARAMETER_LISTS or isinstance(value, set):
            by_json = {
                json.dumps(item, sort_keys=True, separators=(",", ":"), default=str): item
                for item in normalized
            }
            return [by_json[key] for key in sorted(by_json)]
        return normalized
    text = re.sub(r"\s+", " ", str(value)).strip()
    return {
        "invalid_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": text,
    }


def canonical_action_signature(tool_name: Any, parameters: Any) -> str:
    """Return a deterministic identity for one complete logical action.

    Known window collections are treated as unordered sets, while all retained
    semantic parameters remain in the identity.  Invalid values are represented
    explicitly instead of collapsing to a ``tool:None:None`` sentinel.
    """

    tool = re.sub(r"\s+", " ", str(tool_name or "")).strip().lower() or "<missing>"
    if isinstance(parameters, dict):
        retained = {
            str(key): value
            for key, value in parameters.items()
            if str(key) not in _RUNTIME_ONLY_PARAMETER_KEYS
        }
        normalized_parameters = _canonical_value(retained)
    else:
        normalized_parameters = {
            "invalid_parameters": _canonical_value(parameters),
            "parameter_type": f"{type(parameters).__module__}.{type(parameters).__qualname__}",
        }
    canonical = json.dumps(
        {
            "schema_version": ACTION_SIGNATURE_VERSION,
            "tool": tool,
            "parameters": normalized_parameters,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{ACTION_SIGNATURE_VERSION}:{digest}"


__all__ = [
    "BLOCKED_FEEDBACK_PREFIX",
    "STATE_KEY",
    "canonical_action_signature",
    "format_blocked_overview_feedback",
    "has_valid_overview_receipt",
    "record_blocked_overview",
    "record_overview_attempt",
]
