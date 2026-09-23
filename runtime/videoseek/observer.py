from __future__ import annotations

import base64
import atexit
import hashlib
import json
import os
import select
import subprocess
import tempfile
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .local_qwen import get_local_qwen_client
from .utils import ApiRequestBudgetExceeded, call_llm_api, role_model_config


_PERSISTENT_QWEN_WORKERS: dict[tuple[Any, ...], "_PersistentQwenWorker"] = {}


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [chunk.strip() for chunk in str(value).split(",") if chunk.strip()]


def _mode_allowed(config: dict, *, tool_name: str, tool_mode: str | None) -> bool:
    mode_key = f"local_qwen_{tool_name}_modes"
    allowed_raw = config.get(mode_key)
    if not allowed_raw:
        return True
    allowed = set(_as_list(allowed_raw))
    return (tool_mode or "") in allowed


def select_observer_backend(
    config: dict,
    *,
    tool_name: str,
    tool_mode: str | None = None,
) -> str:
    backend = (config.get("observer_backend") or "api").strip().lower()
    if backend in {"openai", "remote"}:
        return "api"
    if backend == "local_qwen":
        return "local_qwen"
    if backend == "hybrid":
        local_tools = set(_as_list(config.get("local_qwen_tools") or "focus"))
        if tool_name in local_tools and _mode_allowed(
            config,
            tool_name=tool_name,
            tool_mode=tool_mode,
        ):
            return "local_qwen"
        return "api"
    return "api"


def _image_cache_dir(config: dict, *, tool_name: str, output_dir: str | None) -> Path:
    root = Path(
        output_dir
        or config.get("observer_image_cache_dir")
        or config.get("output_dir")
        or "./output"
    )
    path = root / "_local_qwen_inputs" / tool_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _decode_data_url(url: str) -> bytes:
    if "," not in url:
        raise ValueError("Only data:image/*;base64 URLs are supported for local Qwen")
    return base64.b64decode(url.split(",", 1)[1])


def _resize_image_bytes_for_local_qwen(image_bytes: bytes, *, max_side: int) -> bytes:
    if max_side <= 0:
        return image_bytes
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        width, height = img.size
        longest = max(width, height)
        if longest <= max_side:
            return image_bytes
        scale = max_side / float(longest)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        output = BytesIO()
        resized.save(output, format="JPEG", quality=90, optimize=True)
        return output.getvalue()


def _content_to_local_qwen_parts(
    content: list[dict],
    *,
    config: dict,
    tool_name: str,
    output_dir: str | None,
) -> list[dict[str, str]]:
    cache_dir = _image_cache_dir(config, tool_name=tool_name, output_dir=output_dir)
    parts: list[dict[str, str]] = [
        {
            "type": "text",
            "text": (
                "You are the local Qwen visual observer in a video QA agent. "
                "The attached images are video frames in chronological order. "
                "Text immediately before each image gives its timestamp or grid range. "
                "Compare actual visual content inside the frames, including object "
                "position, color, identity, and state. Return only the JSON schema "
                "requested by the task prompt; do not answer the final multiple-choice question.\n\n"
            ),
        }
    ]
    image_idx = 0
    for item in content:
        if item.get("type") == "text":
            parts.append({"type": "text", "text": str(item.get("text", ""))})
            continue
        if item.get("type") != "image_url":
            continue

        image_idx += 1
        url = ((item.get("image_url") or {}).get("url") or "").strip()
        if not url.startswith("data:image"):
            raise ValueError(f"Unsupported local Qwen image URL: {url[:64]}")
        image_bytes = _decode_data_url(url)
        image_bytes = _resize_image_bytes_for_local_qwen(
            image_bytes,
            max_side=int(config.get("local_qwen_max_image_side") or 768),
        )
        digest = hashlib.sha1(image_bytes).hexdigest()[:16]
        image_path = cache_dir / f"{tool_name}_{image_idx:03d}_{digest}.jpg"
        if not image_path.exists():
            image_path.write_bytes(image_bytes)
        parts.append({"type": "text", "text": f"\n[Image {image_idx:03d}]\n"})
        parts.append({"type": "image", "image": str(image_path)})
    return parts


def _call_api_observer(
    config: dict,
    *,
    content: list[dict],
    return_json: bool = False,
) -> str | None:
    request_guard = config.get("_adaptive_api_request_guard")
    if callable(request_guard) and not request_guard():
        raise ApiRequestBudgetExceeded(
            "adaptive investigation token budget stopped this observer API request"
        )
    observer_config = role_model_config(config, role="observer")
    response = call_llm_api(
        messages=[{"role": "user", "content": content}],
        model_name=observer_config["model_name"],
        api_base=observer_config["api_base"],
        api_key=observer_config["api_key"],
        api_version=observer_config["api_version"],
        max_tokens=config["max_tokens"],
        reasoning_effort=observer_config["reasoning_effort"],
        seed=config["seed"],
        temperature=config["temperature"],
        return_json=return_json,
    )
    if response is None:
        return None
    return response.choices[0].message.content


def _call_local_qwen_observer(
    config: dict,
    *,
    content: list[dict],
    tool_name: str,
    output_dir: str | None,
) -> str:
    parts = _content_to_local_qwen_parts(
        content,
        config=config,
        tool_name=tool_name,
        output_dir=output_dir,
    )
    max_images = int(config.get("local_qwen_max_images") or 24)
    image_count = sum(1 for item in parts if item.get("type") == "image")
    if image_count > max_images:
        raise RuntimeError(
            f"local Qwen image count {image_count} exceeds local_qwen_max_images={max_images}"
        )
    local_python = config.get("local_qwen_python")
    if local_python:
        if _as_bool(config.get("local_qwen_persistent_worker"), default=False):
            return _call_local_qwen_persistent(config, parts=parts, output_dir=output_dir)
        return _call_local_qwen_subprocess(config, parts=parts)

    client = get_local_qwen_client(config)
    return client.create_from_parts(
        parts,
        max_output_tokens=int(config.get("local_qwen_max_new_tokens") or 768),
    )


def call_local_qwen_parts(
    config: dict,
    *,
    parts: list[dict[str, str]],
    output_dir: str | None = None,
    max_new_tokens: int | None = None,
) -> str:
    """Call the shared local-Qwen worker without observer prompt wrapping.

    Grounding uses a small plain-text contract rather than the caption observer
    schema. This function deliberately has no remote fallback; its caller must
    retain the original frame when the local proposal fails.
    """
    local_config = dict(config)
    if max_new_tokens is not None:
        local_config["local_qwen_max_new_tokens"] = int(max_new_tokens)
    local_python = local_config.get("local_qwen_python")
    if local_python:
        if _as_bool(local_config.get("local_qwen_persistent_worker"), default=False):
            return _call_local_qwen_persistent(
                local_config,
                parts=parts,
                output_dir=output_dir,
            )
        return _call_local_qwen_subprocess(local_config, parts=parts)

    client = get_local_qwen_client(local_config)
    return client.create_from_parts(
        parts,
        max_output_tokens=int(
            max_new_tokens
            or local_config.get("local_qwen_max_new_tokens")
            or 768
        ),
    )


def _call_local_qwen_subprocess(config: dict, *, parts: list[dict[str, str]]) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    worker = repo_root / "scripts" / "local_qwen_observer_worker.py"
    request = {
        "model_path": config.get("local_qwen_model_path"),
        "torch_dtype": config.get("local_qwen_torch_dtype") or "bfloat16",
        "device_map": config.get("local_qwen_device_map") or "auto",
        "max_memory": config.get("local_qwen_max_memory") or None,
        "cuda_visible_devices": config.get("local_qwen_cuda_visible_devices") or None,
        "max_new_tokens": int(config.get("local_qwen_max_new_tokens") or 768),
        "no_cpu_offload": bool(config.get("local_qwen_no_cpu_offload", True)),
        "parts": parts,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(request, handle, ensure_ascii=False)
        request_path = Path(handle.name)
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(repo_root)
            if not env.get("PYTHONPATH")
            else f"{repo_root}:{env['PYTHONPATH']}"
        )
        timeout_s = int(config.get("local_qwen_timeout_s") or 900)
        proc = subprocess.run(
            [str(config["local_qwen_python"]), str(worker), str(request_path)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "local Qwen worker failed: "
                f"returncode={proc.returncode}\nSTDERR:\n{proc.stderr[-4000:]}\n"
                f"STDOUT:\n{proc.stdout[-1000:]}"
            )
        stdout = proc.stdout.strip()
        # Progress bars and warnings can precede the final JSON line.
        last_line = stdout.splitlines()[-1] if stdout else ""
        return json.loads(last_line)["raw"]
    finally:
        request_path.unlink(missing_ok=True)


class _PersistentQwenWorker:
    def __init__(self, config: dict, *, output_dir: str | None) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.worker = self.repo_root / "scripts" / "local_qwen_persistent_worker.py"
        self.python = str(config["local_qwen_python"])
        self.timeout_s = int(config.get("local_qwen_timeout_s") or 900)
        self.max_new_tokens = int(config.get("local_qwen_max_new_tokens") or 768)
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None
        self.stderr_handle = None
        self.config_path: Path | None = None
        self.worker_config = {
            "model_path": config.get("local_qwen_model_path"),
            "torch_dtype": config.get("local_qwen_torch_dtype") or "bfloat16",
            "device_map": config.get("local_qwen_device_map") or "auto",
            "max_memory": config.get("local_qwen_max_memory") or None,
            "cuda_visible_devices": config.get("local_qwen_cuda_visible_devices") or None,
            "max_new_tokens": self.max_new_tokens,
            "no_cpu_offload": bool(config.get("local_qwen_no_cpu_offload", True)),
        }
        log_root = Path(output_dir or config.get("output_dir") or "./output")
        log_root.mkdir(parents=True, exist_ok=True)
        self.log_path = log_root / "_local_qwen_persistent_worker.stderr.log"

    def ensure_started(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        ) as handle:
            json.dump(self.worker_config, handle, ensure_ascii=False)
            self.config_path = Path(handle.name)

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(self.repo_root)
            if not env.get("PYTHONPATH")
            else f"{self.repo_root}:{env['PYTHONPATH']}"
        )
        self.stderr_handle = self.log_path.open("a", encoding="utf-8")
        self.stderr_handle.write(
            f"\n===== start local qwen persistent worker {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
        )
        self.stderr_handle.flush()
        self.proc = subprocess.Popen(
            [self.python, str(self.worker), str(self.config_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            bufsize=1,
            env=env,
        )
        ready = self._read_json_line(timeout_s=self.timeout_s)
        if ready.get("status") != "ready":
            self.stop(kill=True)
            raise RuntimeError(f"local Qwen persistent worker did not become ready: {ready}")
        if self.config_path:
            self.config_path.unlink(missing_ok=True)
            self.config_path = None

    def request(self, parts: list[dict[str, str]], *, max_new_tokens: int | None = None) -> str:
        with self.lock:
            self.ensure_started()
            assert self.proc is not None and self.proc.stdin is not None
            request_id = uuid.uuid4().hex
            payload = {
                "id": request_id,
                "parts": parts,
                "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            }
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            response = self._read_json_line(timeout_s=self.timeout_s)
            if response.get("id") != request_id:
                raise RuntimeError(f"local Qwen persistent worker returned mismatched response: {response}")
            if response.get("error"):
                raise RuntimeError(
                    "local Qwen persistent worker request failed: "
                    f"{response.get('error')}\n{response.get('traceback', '')[-4000:]}"
                )
            return str(response.get("raw") or "")

    def _read_json_line(self, *, timeout_s: int) -> dict[str, Any]:
        assert self.proc is not None and self.proc.stdout is not None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"local Qwen persistent worker exited with code {self.proc.returncode}; "
                    f"stderr log: {self.log_path}"
                )
            remaining = max(0.1, deadline - time.time())
            ready, _, _ = select.select([self.proc.stdout], [], [], min(1.0, remaining))
            if not ready:
                continue
            line = self.proc.stdout.readline()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                if self.stderr_handle:
                    self.stderr_handle.write(f"[ignored stdout] {line}")
                    self.stderr_handle.flush()
                continue
        self.stop(kill=True)
        raise TimeoutError(f"local Qwen persistent worker timed out after {timeout_s}s")

    def stop(self, *, kill: bool = False) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            if kill:
                proc.kill()
            else:
                try:
                    if proc.stdin is not None:
                        proc.stdin.write(json.dumps({"cmd": "shutdown", "id": "shutdown"}) + "\n")
                        proc.stdin.flush()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        if self.config_path:
            self.config_path.unlink(missing_ok=True)
            self.config_path = None
        if self.stderr_handle:
            self.stderr_handle.close()
            self.stderr_handle = None
        self.proc = None


def _persistent_worker_key(config: dict) -> tuple[Any, ...]:
    return (
        str(config.get("local_qwen_python") or ""),
        str(Path(config.get("local_qwen_model_path") or "").expanduser()),
        str(config.get("local_qwen_cuda_visible_devices") or ""),
        str(config.get("local_qwen_device_map") or "auto"),
        str(config.get("local_qwen_max_memory") or ""),
        str(config.get("local_qwen_torch_dtype") or "bfloat16"),
        bool(config.get("local_qwen_no_cpu_offload", True)),
    )


def _call_local_qwen_persistent(
    config: dict,
    *,
    parts: list[dict[str, str]],
    output_dir: str | None,
) -> str:
    resident_socket = os.environ.get("COSEEK_QWEN_SOCKET")
    if resident_socket:
        from .resident_qwen import request
        return request(resident_socket, config, parts, output_dir=output_dir)
    key = _persistent_worker_key(config)
    worker = _PERSISTENT_QWEN_WORKERS.get(key)
    if worker is None:
        worker = _PersistentQwenWorker(config, output_dir=output_dir)
        _PERSISTENT_QWEN_WORKERS[key] = worker
    return worker.request(
        parts,
        max_new_tokens=int(config.get("local_qwen_max_new_tokens") or 768),
    )


def _shutdown_persistent_workers() -> None:
    for worker in list(_PERSISTENT_QWEN_WORKERS.values()):
        worker.stop()
    _PERSISTENT_QWEN_WORKERS.clear()


atexit.register(_shutdown_persistent_workers)


def observe_content(
    config: dict,
    *,
    content: list[dict],
    tool_name: str,
    tool_mode: str | None = None,
    output_dir: str | None = None,
    return_json: bool = False,
) -> tuple[str | None, str]:
    """Route visual observation to API, local Qwen, or hybrid policy."""

    backend = select_observer_backend(config, tool_name=tool_name, tool_mode=tool_mode)
    if backend == "local_qwen":
        try:
            raw = _call_local_qwen_observer(
                config,
                content=content,
                tool_name=tool_name,
                output_dir=output_dir,
            )
            if output_dir:
                path = Path(output_dir) / "local_observer_responses.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"tool": tool_name, "mode": tool_mode,
                        "input_texts": [r["text"] for r in content if r.get("type") == "text"],
                        "image_count": sum(r.get("type") == "image_url" for r in content),
                        "raw_response": raw}, ensure_ascii=False) + "\n")
            return raw, "local_qwen"
        except Exception as exc:
            if not _as_bool(config.get("local_qwen_fallback_to_api"), default=True):
                raise
            raw = _call_api_observer(config, content=content, return_json=return_json)
            if raw:
                raw = (
                    "LOCAL_QWEN_FALLBACK_TO_API:\n"
                    f"{type(exc).__name__}: {exc}\n\n"
                    f"{raw}"
                )
            return raw, "api_fallback_after_local_qwen_error"

    return _call_api_observer(config, content=content, return_json=return_json), "api"
