from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from inference.dataio.samples import TaskSample
from inference.utils.text import strip_modality_tags
from inference.utils.torch_helpers import apply_dtype_kw, load_auto_processor

from .base import BaseModelWrapper, ModelCapabilities


def _infer_device(model: torch.nn.Module) -> torch.device:
    """
    Best-effort device detection that works with both single-device and device_map=auto loads.
    Falls back to CPU when the model has no parameters (should not happen in practice).
    """
    if hasattr(model, "device"):
        try:
            return torch.device(model.device)
        except Exception:
            pass
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _truncate_video_payload(
    payload: Any,
    max_frames: Optional[int],
) -> Any:
    """Respect a manual max frame request without touching the default preprocessing otherwise."""
    if max_frames is None:
        return payload
    if isinstance(payload, torch.Tensor):
        if payload.ndim >= 4 and payload.shape[0] > max_frames:
            return payload[:max_frames]
        if payload.ndim == 5 and payload.shape[1] > max_frames:
            return payload[:, :max_frames]
        return payload
    if isinstance(payload, list) and len(payload) > max_frames:
        return payload[:max_frames]
    return payload


class MiMoVLWrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=True, supports_video=True)

    def __init__(self, *args: Any, video_max_frames: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.video_max_frames = video_max_frames
        self.processor = None
        self.tokenizer = None
        self.model = None

    def _load(self) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration

        load_kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        load_kwargs.setdefault("device_map", "auto")
        cache_dir = load_kwargs.get("cache_dir")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **load_kwargs,
        ).eval()
        self.processor = load_auto_processor(
            self.model_id,
            trust_remote_code=False,
            cache_dir=cache_dir,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)

    def _build_messages(self, sample: TaskSample) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_hint:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_hint.strip()}],
                }
            )

        user_content: List[Dict[str, Any]] = []
        media_path = Path(sample.fake_path).resolve()
        if sample.modality == "image":
            user_content.append({"type": "image", "image": f"file://{media_path}"})
        else:
            user_content.append({"type": "video", "video": f"file://{media_path}"})

        prompt = strip_modality_tags(sample.prompt or "")
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            prompt = f"{prefix}\n\n{prompt}" if prompt else prefix
        prompt = prompt.strip()
        if not prompt:
            prompt = "Please describe the provided content."
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})
        return messages

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        assert self.processor is not None
        assert self.model is not None

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "MiMoVLWrapper requires the optional package `qwen-vl-utils` for vision preprocessing. "
                "Install it (e.g., `pip install qwen-vl-utils`) before evaluating MiMo-VL models."
            ) from exc

        messages = self._build_messages(sample)
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        if video_inputs is not None:
            video_inputs = [_truncate_video_payload(v, self.video_max_frames) for v in video_inputs]

        model_inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        device = _infer_device(self.model)
        model_inputs = model_inputs.to(device=device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
        }
        if self.generation_overrides:
            gen_kwargs.update(self.generation_overrides)

        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **gen_kwargs)

        input_ids = model_inputs["input_ids"]
        start = input_ids.shape[-1]
        continuation = output_ids[:, start:]

        tokenizer = self.tokenizer or getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Processor does not expose a tokenizer for decoding MiMo-VL outputs.")

        response = tokenizer.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()


__all__ = ["MiMoVLWrapper"]
