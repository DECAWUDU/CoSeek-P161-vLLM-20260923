#!/usr/bin/env python3
"""Run the preregistered P99 fixed-state Planner A/B without visual tools."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

from litellm import completion


P99_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = P99_ROOT / "offline" / "p99_fixed_state_manifest.json"
DEFAULT_OUTPUT = P99_ROOT / "phase2" / "p99_fixed_state_calls.jsonl"
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]+")
WRITE_LOCK = threading.Lock()


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _redact(value: Any) -> str:
    return SECRET_RE.sub("sk-REDACTED", str(value))


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _response_content(response: Any) -> str:
    choices = _field(response, "choices", []) or []
    if not choices:
        return ""
    message = _field(choices[0], "message", {})
    content = _field(message, "content", "")
    return "" if content is None else str(content)


def _usage(response: Any) -> dict[str, int]:
    usage = _field(response, "usage", {})
    return {
        "prompt_tokens": int(_field(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(_field(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(_field(usage, "total_tokens", 0) or 0),
    }


def _load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            call_id = str(record.get("call_id") or "")
            if not call_id or call_id in completed:
                raise RuntimeError(f"invalid/duplicate call_id at line {line_no}: {call_id!r}")
            completed.add(call_id)
    return completed


def _append_record(path: Path, record: dict[str, Any]) -> None:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


def _call_id(state_id: str, repeat: int, arm: str) -> str:
    return f"{state_id}_R{repeat}_{arm}"


def _run_call(
    *,
    state: dict[str, Any],
    state_index: int,
    repeat: int,
    arm: str,
    api_base: str,
    api_key: str,
    output: Path,
    model: str,
) -> dict[str, Any]:
    call_id = _call_id(str(state["state_id"]), repeat, arm)
    messages = state[f"{arm}_messages"]
    started = time.time()
    record: dict[str, Any] = {
        "call_id": call_id,
        "state_id": state["state_id"],
        "state_index": state_index,
        "repeat": repeat,
        "arm": arm,
        "category": state["category"],
        "qid": state["qid"],
        "question_type": state["question_type"],
        "historical_tail": bool(state["tail"]),
        "historical_step": int(state["step"]),
        "message_sha256": _sha256_json(messages),
        "api_base": api_base,
        "model_requested": model,
        "seed": 42,
        "temperature": 1.0,
        "reasoning_effort": "medium",
        "max_completion_tokens": 32768,
        "started_at_epoch": started,
    }
    try:
        response = completion(
            model=model,
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            max_completion_tokens=32768,
            seed=42,
            temperature=1.0,
            reasoning_effort="medium",
            timeout=900,
            num_retries=0,
        )
        content = _response_content(response)
        record.update(
            {
                "status": "ok",
                "response_id": _field(response, "id", None),
                "model_returned": _field(response, "model", None),
                "content": content,
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "usage": _usage(response),
            }
        )
    except Exception as exc:  # recorded once; protocol forbids a result-driven retry
        record.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": _redact(exc),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )
    record["wall_seconds"] = round(time.time() - started, 3)
    record["finished_at_epoch"] = time.time()
    _append_record(output, record)
    status = record["status"]
    usage = record["usage"]
    print(
        f"[{call_id}] {status} prompt={usage['prompt_tokens']} "
        f"completion={usage['completion_tokens']} wall={record['wall_seconds']}s",
        flush=True,
    )
    return record


def _arm_order(state_index: int, repeat: int) -> tuple[str, str]:
    return ("control", "treatment") if (state_index + repeat - 1) % 2 == 0 else ("treatment", "control")


def _run_pair(
    *,
    state: dict[str, Any],
    state_index: int,
    repeat: int,
    completed: set[str],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for arm in _arm_order(state_index, repeat):
        call_id = _call_id(str(state["state_id"]), repeat, arm)
        if call_id in completed:
            continue
        records.append(
            _run_call(
                state=state,
                state_index=state_index,
                repeat=repeat,
                arm=arm,
                **kwargs,
            )
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-base", default="https://quanzil.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    states = manifest.get("states") or []
    if len(states) != 24 or int(manifest.get("arm_repeats", 0)) != 3:
        raise RuntimeError("frozen manifest must contain 24 states and 3 arm repeats")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing API credential in {args.api_key_env}")
    if not 1 <= args.workers <= 4:
        raise RuntimeError("protocol permits 1..4 pair workers")

    completed = _load_completed(args.output)
    all_ids = {
        _call_id(str(state["state_id"]), repeat, arm)
        for state in states
        for repeat in range(1, 4)
        for arm in ("control", "treatment")
    }
    unknown = completed - all_ids
    if unknown:
        raise RuntimeError(f"output contains non-protocol call IDs: {sorted(unknown)}")
    print(f"P99 Phase 2: {len(completed)}/144 calls already recorded; route={args.api_base}", flush=True)

    common = {
        "api_base": args.api_base,
        "api_key": api_key,
        "output": args.output,
        "model": args.model,
    }

    # Route admission is the first preregistered call, not an extra probe.
    admission_id = _call_id(str(states[0]["state_id"]), 1, "control")
    if admission_id not in completed:
        admission = _run_call(
            state=states[0], state_index=0, repeat=1, arm="control", **common
        )
        completed.add(admission_id)
        if admission["status"] != "ok":
            print("Route admission failed; stopping Phase 2 without launching concurrent calls.", flush=True)
            return 2

    tasks = [
        (idx, state, repeat)
        for idx, state in enumerate(states)
        for repeat in range(1, 4)
        if any(
            _call_id(str(state["state_id"]), repeat, arm) not in completed
            for arm in ("control", "treatment")
        )
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _run_pair,
                state=state,
                state_index=idx,
                repeat=repeat,
                completed=completed,
                **common,
            )
            for idx, state, repeat in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    final_completed = _load_completed(args.output)
    print(f"P99 Phase 2 finished with {len(final_completed)}/144 unique records.", flush=True)
    return 0 if len(final_completed) == 144 else 3


if __name__ == "__main__":
    raise SystemExit(main())
