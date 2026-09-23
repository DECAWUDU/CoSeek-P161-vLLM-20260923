#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from videoseek.local_qwen import LocalQwenVLClient  # noqa: E402


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: local_qwen_persistent_worker.py CONFIG_JSON", file=sys.stderr)
        return 2

    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    client = LocalQwenVLClient(
        config["model_path"],
        torch_dtype=config.get("torch_dtype") or "bfloat16",
        device_map=config.get("device_map") or "auto",
        max_memory=config.get("max_memory") or None,
        cuda_visible_devices=config.get("cuda_visible_devices") or None,
        max_new_tokens=int(config.get("max_new_tokens") or 768),
        no_cpu_offload=bool(config.get("no_cpu_offload", True)),
    )
    _emit({"status": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = request.get("id")
            if request.get("cmd") == "shutdown":
                _emit({"id": request_id, "status": "shutdown"})
                return 0
            raw = client.create_from_parts(
                request["parts"],
                max_output_tokens=int(
                    request.get("max_new_tokens")
                    or config.get("max_new_tokens")
                    or 768
                ),
            )
            _emit({"id": request_id, "raw": raw})
        except Exception as exc:
            _emit(
                {
                    "id": locals().get("request", {}).get("id"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
