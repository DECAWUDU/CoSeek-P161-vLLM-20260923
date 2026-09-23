from copy import deepcopy
import hashlib, json, os, signal, sys, time
from pathlib import Path

TRANSPORT_BUDGET = {
    "provider": "openai_compatible_direct_sdk",
    "max_http_requests": 100,
    "http_timeout_s": 400,
    "max_suffix_walltime_s": 10800,
    "sdk_max_retries": 0,
    "outer_retries": 2,
    "max_episode_retries": 2,
    "retryable_http_statuses": [408, 429, 500, 502, 503, 504],
    "retryable_http_status_classes": ["5xx"],
    "retry_delay_schedule_s": [5, 15],
    "follow_redirects": False,
    "trust_env": False,
    "persistent_http_client": False,
}

class ReplayContractError(BaseException):
    """Fatal: must not be swallowed by the ordinary tool fail-open handler."""

class ExperimentTransportFailure(BaseException):
    """Stop the whole experimental suffix, including inside tool handlers."""


class AttemptTimeout(ExperimentTransportFailure):
    """The current physical HTTP attempt exceeded its deadline."""


class SuffixTimeout(ExperimentTransportFailure):
    """The complete cell exceeded its wall-time budget."""

def sanitized_error_details(exc, api_key):
    """Keep diagnostic fields without credentials or encoded image payloads."""
    import re
    def clean(value, limit=1200):
        if not isinstance(value, (str, int, float, bool)):
            return None
        text = str(value)
        if api_key:
            text = text.replace(str(api_key), "<redacted>")
        text = re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted-key>", text)
        text = re.sub(r"(?i)Bearer\s+[^\s\"',;]+", "Bearer <redacted>", text)
        text = re.sub(r"data:image/[^\s\"']+", "<redacted-image>", text)
        text = re.sub(r"[A-Za-z0-9+/=_-]{80,}", "<redacted-long-value>", text)
        return text[:limit]
    try:
        body = getattr(exc, "body", None)
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            body = body["error"]
        detail = {k: clean(body[k]) for k in ("message", "type", "code", "param")
                  if isinstance(body, dict) and k in body}
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        allowed = ("x-request-id", "request-id", "retry-after", "retry-after-ms",
                   "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
                   "x-ratelimit-reset-requests", "x-ratelimit-limit-tokens",
                   "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens")
        safe_headers = {k: clean(headers[k], 256) for k in allowed if k in headers}
        return {"error_details": detail, "response_headers": safe_headers,
                "request_id": clean(getattr(exc, "request_id", None), 256)}
    except Exception:
        return {"diagnostic_capture_failed": True}

class BoundedTransport:
    """Process-local adapter; production utils and retries remain unchanged."""
    def __init__(self, replay, output_dir):
        self.replay, self.output_dir = replay, output_dir
        self.started = None
        self.attempts = self.succeeded = self.failed = 0
        self.semantic_calls = self.retries = 0
        self.unknown_usage = 0
        self.active_attempt = None
        self.stopped = False
        self.original_call = None
        self.adapter = self.call
        self.installed_bindings = []
        self.original_handler = None
        self.client = None
        self.http_client = None
        self.client_route = None
        self.wire_request = None

    def _journal(self, row):
        # Never journal credentials, request contents, or raw SDK exceptions.
        with (self.output_dir / "transport_attempts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _remaining(self):
        if self.started is None:
            raise ExperimentTransportFailure("Transport used before suffix start")
        return TRANSPORT_BUDGET["max_suffix_walltime_s"] - (time.monotonic() - self.started)

    def _alarm(self, signum, frame):
        if self.active_attempt:
            raise AttemptTimeout("HTTP deadline exceeded")
        raise SuffixTimeout("Suffix walltime exceeded")

    def start_suffix(self):
        if self.started is not None:
            raise ExperimentTransportFailure("Suffix transport budget was already started")
        self.started = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, TRANSPORT_BUDGET["max_suffix_walltime_s"])

    def install(self):
        import videoseek.utils as utils
        if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
            raise ExperimentTransportFailure("Existing process alarm conflicts with suffix budget")
        self.original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._alarm)
        self.original_call = utils.call_llm_api
        for name, module in list(sys.modules.items()):
            if name.startswith("videoseek.") and getattr(module, "call_llm_api", None) is self.original_call:
                setattr(module, "call_llm_api", self.adapter)
                self.installed_bindings.append(name)
        if "videoseek.utils" not in self.installed_bindings:
            raise ExperimentTransportFailure("Cannot bind the shared API adapter")

    def close(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        if self.original_handler is not None:
            signal.signal(signal.SIGALRM, self.original_handler)
        # Include late imports that obtained the patched utils binding.
        for name, module in list(sys.modules.items()):
            if name.startswith("videoseek.") and getattr(module, "call_llm_api", None) is self.adapter:
                setattr(module, "call_llm_api", self.original_call)
        if self.client is not None:
            self.client.close()
        self.client = self.http_client = None

    def summary(self):
        return {
            "attempted_requests": self.attempts,
            "successful_requests": self.succeeded,
            "failed_requests": self.failed,
            "unfinished_requests": self.attempts - self.succeeded - self.failed,
            "usage_unknown_requests": self.unknown_usage,
            "semantic_calls": self.semantic_calls,
            "retry_attempts": self.retries,
            "cost_comparable": self.unknown_usage == 0 and self.failed == 0,
            "elapsed_suffix_s": round(time.monotonic() - self.started, 3) if self.started is not None else 0.0,
            "budget": deepcopy(TRANSPORT_BUDGET),
            "patched_bindings": sorted(self.installed_bindings),
        }

    def _audit_wire_request(self, request):
        # Capture the serialized HTTP body, not pre-SDK kwargs. Never log content/headers.
        body = request.content
        data = json.loads(body)
        self.wire_request = {
            "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body),
            "keys": sorted(data), "model": data.get("model"),
            "max_completion_tokens": data.get("max_completion_tokens"),
            "has_max_tokens": "max_tokens" in data,
        }
        self._journal({"event": "wire_request", "attempt": self.active_attempt,
                       "semantic_call": self.semantic_calls, **self.wire_request})

    def _provider_parameter_mismatch(self, exc):
        body = getattr(exc, "body", None)
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            body = body["error"]
        wire = self.wire_request or {}
        cap = wire.get("max_completion_tokens")
        return (getattr(exc, "status_code", None) == 400
                and isinstance(body, dict)
                and body.get("code") == "unsupported_parameter"
                and body.get("param") == "max_tokens"
                and wire.get("has_max_tokens") is False
                and isinstance(cap, int) and not isinstance(cap, bool) and cap > 0)

    def _is_retryable(self, exc):
        status = getattr(exc, "status_code", None)
        if self._provider_parameter_mismatch(exc):
            return True
        if isinstance(exc, AttemptTimeout):
            return True
        if isinstance(status, int) and (
            status in TRANSPORT_BUDGET["retryable_http_statuses"] or 500 <= status <= 599
        ):
            return True
        return type(exc).__name__ in {
            "APITimeoutError", "APIConnectionError", "ReadTimeout", "ConnectTimeout",
            "TimeoutError", "ConnectError", "RemoteProtocolError", "ReadError", "WriteError"
        }

    def _get_client(self, *, api_base, api_key):
        import httpx
        from openai import OpenAI
        route = (api_base.rstrip("/"), api_key)
        if self.client is not None:
            if route != self.client_route:
                raise ReplayContractError("API route or credential changed within one cell")
            self.client.close()
            self.client = self.http_client = None
        if self.client is None:
            timeout = httpx.Timeout(float(TRANSPORT_BUDGET["http_timeout_s"]))
            self.http_client = httpx.Client(
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
                event_hooks={"request": [self._audit_wire_request]},
            )
            self.client = OpenAI(
                base_url=api_base,
                api_key=api_key,
                max_retries=0,
                timeout=timeout,
                http_client=self.http_client,
            )
            self.client_route = route
        elif route != self.client_route:
            raise ReplayContractError("API route or credential changed within one cell")
        return self.client

    def call(self, model_name, messages, api_base, api_key=None, api_version=None,
             max_tokens=32768, reasoning_effort="medium", seed=42, temperature=1.0,
             tools=None, tool_choice=None, return_json=False):
        if self.replay.active:
            raise ReplayContractError("API adapter invoked during frozen replay")
        if api_version:
            raise ExperimentTransportFailure("This experiment requires an OpenAI-compatible endpoint without api_version")
        remaining = self._remaining()
        if remaining <= 0:
            raise SuffixTimeout("Suffix walltime exhausted before API dispatch")
        from litellm import ModelResponse
        image_count = text_chars = 0
        for message in messages or []:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                text_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        image_count += int(item.get("type") == "image_url")
                        if item.get("type") == "text":
                            text_chars += len(str(item.get("text") or ""))
        metadata = {"model": model_name, "message_count": len(messages or []),
                    "text_chars": text_chars, "image_count": image_count,
                    "has_images": image_count > 0}
        self.semantic_calls += 1
        semantic_call = self.semantic_calls
        from vllm_adapter import build_request
        request = build_request(model_name,messages,max_tokens,seed,temperature,tools,tool_choice,return_json)
        max_physical_attempts = 1 + int(TRANSPORT_BUDGET["outer_retries"])
        for physical_index in range(max_physical_attempts):
            remaining = self._remaining()
            if remaining <= 0:
                raise SuffixTimeout("Suffix walltime exhausted before API dispatch")
            if self.attempts >= TRANSPORT_BUDGET["max_http_requests"]:
                raise ExperimentTransportFailure("HTTP request budget exhausted before API dispatch")
            timeout = min(float(TRANSPORT_BUDGET["http_timeout_s"]), remaining)
            self.attempts += 1
            attempt = self.attempts
            self.active_attempt = attempt
            self.wire_request = None
            dispatch_start = time.monotonic()
            self._journal({
                "event": "attempt_started",
                "attempt": attempt,
                "semantic_call": semantic_call,
                "physical_index": physical_index + 1,
                "timeout_s": timeout,
                "elapsed_suffix_s": round(dispatch_start - self.started, 3),
                **metadata,
            })
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                # SDK retries stay disabled; this adapter owns all retries
                # and enforces a cumulative two-retry episode budget.
                client = self._get_client(api_base=api_base, api_key=api_key)
                raw = client.chat.completions.create(**request, timeout=timeout)
                raw_payload = raw.model_dump()
                from vllm_adapter import validate_response
                validate_response(raw_payload)
                raw_usage = raw_payload.get("usage")
                token_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
                usage_available = isinstance(raw_usage, dict) and all(
                    isinstance(raw_usage.get(key), int)
                    and not isinstance(raw_usage.get(key), bool)
                    and raw_usage[key] >= 0
                    for key in token_keys
                )
                if not usage_available:
                    self.unknown_usage += 1
                    raise ExperimentTransportFailure(
                        "Raw API response lacks usage; cost comparison unavailable"
                    )
                response = ModelResponse(**raw_payload)
                record = {
                    **metadata,
                    "semantic_call": semantic_call,
                    "physical_attempt": attempt,
                    "usage_available": True,
                    **{key: raw_usage[key] for key in token_keys},
                }
                with Path(os.environ["COSEEK_USAGE_LOG"]).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.succeeded += 1
                self._journal({
                    "event": "attempt_finished",
                    "attempt": attempt,
                    "semantic_call": semantic_call,
                    "physical_index": physical_index + 1,
                    "status": "succeeded",
                    "elapsed_s": round(time.monotonic() - dispatch_start, 3),
                    "usage": record,
                })
                return response
            except BaseException as exc:
                signal.setitimer(signal.ITIMER_REAL, 0)
                self.failed += 1
                http_status = getattr(exc, "status_code", None)
                retryable = self._is_retryable(exc)
                self._journal({
                    "event": "attempt_finished",
                    "attempt": attempt,
                    "semantic_call": semantic_call,
                    "physical_index": physical_index + 1,
                    "status": "failed",
                    "elapsed_s": round(time.monotonic() - dispatch_start, 3),
                    "error_type": type(exc).__name__,
                    "http_status": int(http_status) if isinstance(http_status, int) else None,
                    "retryable": retryable,
                    "provider_parameter_mismatch": self._provider_parameter_mismatch(exc),
                    **sanitized_error_details(exc, api_key),
                })
                can_retry = (retryable and physical_index + 1 < max_physical_attempts
                             and self.retries < TRANSPORT_BUDGET["max_episode_retries"])
                if can_retry:
                    self.retries += 1
                    delay = min(
                        float(TRANSPORT_BUDGET["retry_delay_schedule_s"][physical_index]),
                        max(0.0, self._remaining()),
                    )
                    self._journal({
                        "event": "retry_scheduled",
                        "semantic_call": semantic_call,
                        "after_attempt": attempt,
                        "episode_retry": self.retries,
                        "delay_s": delay,
                    })
                    self.active_attempt = None
                    remaining = self._remaining()
                    signal.setitimer(signal.ITIMER_REAL, max(0.001, remaining))
                    time.sleep(delay)
                    continue
                self.stopped = True
                raise ExperimentTransportFailure(
                    f"Semantic call {semantic_call} failed after {physical_index + 1} "
                    f"physical attempt(s); last error={type(exc).__name__}"
                ) from None
            finally:
                self.active_attempt = None
                remaining = self._remaining()
                signal.setitimer(
                    signal.ITIMER_REAL,
                    0 if self.stopped else max(0.001, remaining),
                )
        raise ExperimentTransportFailure("Retry loop exited without a response")
