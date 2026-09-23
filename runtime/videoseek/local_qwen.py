from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def parse_max_memory(value: str | dict | None) -> dict[int | str, str] | None:
    if not value:
        return None
    if isinstance(value, dict):
        parsed: dict[int | str, str] = {}
        for key, item in value.items():
            try:
                parsed[int(key)] = str(item)
            except (TypeError, ValueError):
                parsed[str(key)] = str(item)
        return parsed

    parsed: dict[int | str, str] = {}
    for chunk in str(value).split(","):
        if not chunk.strip():
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid max memory chunk: {chunk!r}")
        key, mem = chunk.split(":", 1)
        key = key.strip()
        mem = mem.strip()
        try:
            parsed[int(key)] = mem
        except ValueError:
            parsed[key] = mem
    return parsed or None


class LocalQwenVLClient:
    """Local Qwen vision observer with an OpenAI-tool-adapter-facing API."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_memory: str | dict | None = None,
        cuda_visible_devices: str | None = None,
        max_new_tokens: int = 768,
        no_cpu_offload: bool = True,
    ) -> None:
        if cuda_visible_devices:
            # Must be set before torch initializes CUDA in this process.
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cuda_visible_devices))

        import torch
        from transformers import AutoConfig, AutoProcessor

        dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
        }
        parsed_max_memory = parse_max_memory(max_memory)
        if parsed_max_memory:
            load_kwargs["max_memory"] = parsed_max_memory

        self.model_path = Path(model_path)
        model_config = AutoConfig.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        self.model_type = str(getattr(model_config, "model_type", ""))
        if self.model_type == "qwen3_5":
            from transformers import Qwen3_5ForConditionalGeneration as model_class
        elif self.model_type == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration as model_class
        elif self.model_type == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration as model_class
        else:
            from transformers import AutoModelForImageTextToText as model_class

        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        self.model = model_class.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            **load_kwargs,
        )
        device_map_actual = getattr(self.model, "hf_device_map", {}) or {}
        offloaded = {
            name: device
            for name, device in device_map_actual.items()
            if str(device).lower() in {"cpu", "disk"}
        }
        if no_cpu_offload and offloaded:
            raise RuntimeError(
                f"Qwen loaded with CPU/disk offload, refusing to continue: {offloaded}"
            )
        self.max_new_tokens = max_new_tokens

    def create_from_parts(
        self,
        parts: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        import torch
        from PIL import Image

        content: list[dict[str, str]] = []
        images = []
        for part in parts:
            if part.get("type") == "image":
                image_path = str(part["image"])
                content.append({"type": "image", "image": image_path})
                images.append(Image.open(image_path).convert("RGB"))
            else:
                content.append({"type": "text", "text": part.get("text", "")})

        messages = [{"role": "user", "content": content}]
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.model_type == "qwen3_5":
            template_kwargs["enable_thinking"] = False
        text = self.processor.apply_chat_template(messages, **template_kwargs)
        inputs = self.processor(
            text=[text],
            images=images or None,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_output_tokens or self.max_new_tokens,
                do_sample=False,
            )
        return self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


_LOCAL_QWEN_CLIENTS: dict[tuple[Any, ...], LocalQwenVLClient] = {}


def get_local_qwen_client(config: dict) -> LocalQwenVLClient:
    model_path = config.get("local_qwen_model_path")
    if not model_path:
        raise ValueError("local_qwen_model_path is required when local Qwen is enabled")

    key = (
        str(Path(model_path).expanduser()),
        config.get("local_qwen_torch_dtype") or "bfloat16",
        config.get("local_qwen_device_map") or "auto",
        str(config.get("local_qwen_max_memory") or ""),
        str(config.get("local_qwen_cuda_visible_devices") or ""),
        int(config.get("local_qwen_max_new_tokens") or 768),
        bool(config.get("local_qwen_no_cpu_offload", True)),
    )
    if key not in _LOCAL_QWEN_CLIENTS:
        _LOCAL_QWEN_CLIENTS[key] = LocalQwenVLClient(
            model_path=key[0],
            torch_dtype=key[1],
            device_map=key[2],
            max_memory=key[3] or None,
            cuda_visible_devices=key[4] or None,
            max_new_tokens=key[5],
            no_cpu_offload=key[6],
        )
    return _LOCAL_QWEN_CLIENTS[key]
