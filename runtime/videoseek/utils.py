import os
import random
import re
import time
import json
import threading
from contextvars import ContextVar, Token
from typing import Callable
from litellm import completion


class ApiRequestBudgetExceeded(RuntimeError):
    """Raised before dispatch when an active investigation budget has stopped."""


class ApiRequestBudgetGuard:
    """Callable budget guard that remains lightweight across config deepcopy."""

    def __init__(self, callback: Callable[[], bool]):
        self._callback = callback
        self._lock = threading.Lock()

    def __call__(self) -> bool:
        with self._lock:
            return bool(self._callback())

    def __deepcopy__(self, memo):
        # frame_verify derives request configs with deepcopy.  The guard must
        # continue observing the original agent's live budget state.
        return self


_api_request_budget_guard: ContextVar[Callable[[], bool] | None] = ContextVar(
    "coseek_api_request_budget_guard", default=None
)


def install_api_request_budget_guard(guard: Callable[[], bool]) -> Token:
    """Install a task-local guard for every remote LLM request in one agent run."""

    return _api_request_budget_guard.set(guard)


def reset_api_request_budget_guard(token: Token) -> None:
    _api_request_budget_guard.reset(token)


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 8,
    max_delay: float = 60,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Raise exceptions for any errors not specified
            except Exception as e:
                if isinstance(e, ApiRequestBudgetExceeded):
                    raise
                if (
                    "rate limit" in str(e).lower()
                    or "timed out" in str(e)
                    or "Too Many Requests" in str(e)
                    or "Forbidden for url" in str(e)
                    or "the maximum usage" in str(e).lower()
                    or "server had an error" in str(e).lower()
                    or "badgateway" in str(e).lower()
                    or "bad gateway" in str(e).lower()
                    or "upstream service temporarily unavailable" in str(e).lower()
                    or "service unavailable" in str(e).lower()
                    or "gateway timeout" in str(e).lower()
                    or "status code: 502" in str(e).lower()
                    or "status code: 503" in str(e).lower()
                    or "has no attribute 'upper'" in str(e).lower()
                    or "internal" in str(e).lower()
                ):
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        print("Max retries reached. Exiting.")
                        return None

                    # Increment the delay
                    delay = min(
                        max_delay,
                        delay * exponential_base * (1 + jitter * random.random()),
                    )
                    print(f"Retrying in {delay} seconds for {str(e)}...")
                    # Sleep for the delay
                    time.sleep(delay)
                else:
                    print(str(e))
                    return None

    return wrapper


@retry_with_exponential_backoff
def call_llm_api(
    model_name: str,
    messages: list,
    api_base: str,
    api_key: str = None,
    api_version: str = None,
    max_tokens: int = 32768,
    reasoning_effort: str = "medium",
    seed: int = 42,
    temperature: float = 1.0,
    tools: list = None,
    tool_choice: str = None,
    return_json: bool = False,
) -> dict:
    guard = _api_request_budget_guard.get()
    if guard is not None and not guard():
        raise ApiRequestBudgetExceeded(
            "adaptive investigation token budget stopped this remote API request"
        )
    response = completion(
        model=model_name,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        api_version=api_version,
        max_completion_tokens=max_tokens,
        seed=seed,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        response_format={"type": "json_object"} if return_json else None,
        timeout=900,
    )
    usage = getattr(response, "usage", None)
    usage_log = os.environ.get("COSEEK_USAGE_LOG")
    if usage_log and usage is not None:
        image_count = 0
        text_chars = 0
        for message in messages or []:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                text_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image_url":
                        image_count += 1
                    elif item.get("type") == "text":
                        text_chars += len(str(item.get("text") or ""))
        record = {
            "model": model_name,
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            "message_count": len(messages or []),
            "text_chars": text_chars,
            "image_count": image_count,
            "has_images": image_count > 0,
        }
        with open(usage_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return response


def role_model_config(config: dict, *, role: str) -> dict:
    if role != "observer":
        return {
            "model_name": config["model_name"],
            "api_base": config["api_base"],
            "api_key": config["api_key"],
            "api_version": config["api_version"],
            "reasoning_effort": config["reasoning_effort"],
        }
    return {
        "model_name": config.get("observer_model_name") or config["model_name"],
        "api_base": config.get("observer_api_base") or config["api_base"],
        "api_key": config.get("observer_api_key") or config["api_key"],
        "api_version": config.get("observer_api_version") or config["api_version"],
        "reasoning_effort": config.get("observer_reasoning_effort") or "low",
    }


def load_subtitles(subtitle_path: str):
    """Parse SRT file and return list of {start_time, end_time, subtitle} dicts."""
    if subtitle_path is None or not os.path.exists(subtitle_path):
        return []
    with open(subtitle_path, "r", encoding="utf-8") as f:
        content = f.read()

    result = []
    # SRT format: index, HH:MM:SS,mmm --> HH:MM:SS,mmm, then text lines
    pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    blocks = re.split(r"\n\n+", content.strip())

    def to_seconds(h, m, s, ms):
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    for block in blocks:
        match = pattern.search(block)
        if match:
            start = to_seconds(*match.groups()[:4])
            end = to_seconds(*match.groups()[4:8])
            text = block[match.end() :].strip().replace("\n", " ")
            result.append(
                {
                    "start_time": round(start, 1),
                    "end_time": round(end, 1),
                    "subtitle": text,
                }
            )
    return result


def convert_to_free_form_text_representation(
    history: list[dict], content_type: str = "caption"
) -> str:
    """
    This function will form the textual representation for the entire video to be used for QA.
    It gives a good structured representation of the entire video.
    JSON types of representations are good for outputs, but free-form/ semi-structured should be better for input.
    """
    if content_type == "subtitle" and history and history[0].get("source_marked"):
        return format_dataset_subtitles(history)
    free_form_text_representation = ""
    if len(history) == 0:
        return f"No {content_type} found."
    for i in history:
        if i[content_type] is None:
            continue
        x = ""
        start_time, end_time = i["start_time"], i["end_time"]
        x += f"**Timestamp**: {start_time}s - {end_time}s\n"
        x += f"**{content_type.capitalize()}**: {i[content_type]}\n"

        free_form_text_representation += f"{x}\n"

    return free_form_text_representation


def extract_json_object(text: str | None) -> dict | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except Exception:
        pass

    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(stripped[start : idx + 1])
                except Exception:
                    return None
    return None


def select_subtitles_for_windows(subtitles, windows):
    """Union of true interval overlaps, preserving source order and IDs."""
    return [row for row in subtitles if any(
        float(row["start_time"]) <= float(b) and float(row["end_time"]) >= float(a)
        for a, b in windows)]


def format_dataset_subtitles(rows):
    """Lossless compact rendering; this is external text, not pixel evidence."""
    if not rows: return "No dataset subtitle overlaps the requested window(s)."
    header = ("DATASET_SUBTITLES_BEGIN\n"
        "External transcript, not screen OCR or visual proof. Use times to locate related scenes. "
        "Distinguish what is said from what is visible; do not put transcript-only claims in visual observations. "
        "When using a subtitle in reasoning, cite its S-id.\n")
    lines = [f"[{r['subtitle_id']}] {r['start_time']:.1f}-{r['end_time']:.1f}s | {r['subtitle']}" for r in rows]
    return header + "\n".join(lines) + "\nDATASET_SUBTITLES_END"


def append_window_subtitles(content, parameters, windows):
    rows = parameters.get("subtitles") or []
    if rows and rows[0].get("source_marked"):
        selected = select_subtitles_for_windows(rows, windows)
        content.append({"type": "text", "text": format_dataset_subtitles(selected)})
    return content


SUBTITLE_WINDOW_CONTRACT = (
    "SUBTITLE ACCESS: The full transcript remains available locally. In your existing planner "
    "JSON you may add subtitle_windows: [[start_s,end_s], ...] and subtitle_refs: [\"S0001\", ...] "
    "for verbatim text needed in later decisions. State relevant subtitle locations even when "
    "your next action is overview. Search/verification windows and cited [S-ids] also retain "
    "nearby transcript context. To revisit another portion, include its absolute window in a "
    "subsequent proposal; [0, video_duration] requests the full transcript. These fields only "
    "select text, never execute a visual tool or verify a claim. Normal tool/action schema is unchanged."
)


def planner_subtitle_view(subtitles, proposals, *, windowed=False, margin_s=8.0):
    """Read-only projection from existing planner history; no second evidence store.

    Keep the union of referenced windows/IDs, including older unresolved leads.
    No rank, Top-K, text truncation, or inference of timestamps from prose.
    Missing/invalid scope falls back to the full transcript, with an explicit label.
    """
    import math
    windows, refs, invalid = [], set(), False
    by_id = {r['subtitle_id']: r for r in subtitles}

    def add_window(value):
        nonlocal invalid
        if isinstance(value, dict):
            value = [value.get('start', value.get('start_time')),
                     value.get('end', value.get('end_time'))]
        try:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError('window needs two endpoints')
            if any(isinstance(v, bool) for v in value): raise ValueError('boolean time')
            a, b = map(float, value)
            if not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= b):
                raise ValueError('invalid time')
            windows.append([max(0., a-margin_s), b+margin_s])
        except (TypeError, ValueError):
            invalid = True

    for proposal in proposals:
        if not isinstance(proposal, dict): continue
        explicit = proposal.get('subtitle_windows', [])
        if not isinstance(explicit, list): invalid = True
        else:
            for value in explicit: add_window(value)
        requested = proposal.get('subtitle_refs', [])
        if not isinstance(requested, list): invalid = True
        else:
            for ref in requested:
                if not isinstance(ref, str) or ref not in by_id: invalid = True
                else: refs.add(ref)
        # Explicit bracketed citations in existing rationale/state are references,
        # never candidate ranks, frame numbers, or free-text timestamp guesses.
        refs.update(re.findall(r'\[(S[0-9]{4,})\]', json.dumps(proposal, ensure_ascii=False)))
        if refs - by_id.keys(): invalid = True
        action = proposal.get('action') or {}
        if not isinstance(action, dict): continue
        tool = action.get('tool', action.get('function_name'))
        if tool not in {'localize_qwen','frame_verify','skim_qwen','focus_qwen'}: continue
        params = action.get('parameters') or {}
        if not isinstance(params, dict): continue
        for value in params.get('search_windows', []) if isinstance(params.get('search_windows', []), list) else []:
            add_window(value)
        if 'start_time' in params and 'end_time' in params:
            add_window([params['start_time'],params['end_time']])
    for ref in sorted(refs & by_id.keys()):
        row = by_id[ref]; add_window([row['start_time'],row['end_time']])
    selected = select_subtitles_for_windows(subtitles, windows) if windows else []
    mode = 'selected'
    if not windowed or invalid or not selected:
        selected = list(subtitles)
        mode = 'full' if not windowed else ('full_invalid_scope' if invalid else 'full_no_scope')
    text = format_dataset_subtitles(selected)
    if mode == 'selected':
        text += (f"\nTranscript view: {len(selected)}/{len(subtitles)} records; union of prior "
                 f"references and requested windows, with {margin_s:g}s context per side. "
                 "Omitted transcript remains locally available; omission is not negative evidence.")
    else:
        text += f"\nTranscript view: full ({len(subtitles)} records; {mode})."
    return text, dict(mode=mode, selected_ids=[x['subtitle_id'] for x in selected],
                      retained_windows=windows, invalid_scope=invalid)
