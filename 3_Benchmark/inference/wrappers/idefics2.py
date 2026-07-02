from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from inference.dataio.samples import TaskSample
from inference.utils.media import load_video_frames_raw
from inference.utils.text import strip_modality_tags
from inference.utils.torch_helpers import apply_dtype_kw, load_auto_processor

from .base import BaseModelWrapper, ModelCapabilities


def _load_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


class Idefics2Wrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=True, supports_video=True)

    def __init__(self, *args, video_max_frames: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_video_frames = 8
        self.video_max_frames = video_max_frames if video_max_frames is not None else self.default_video_frames
        self.processor = None
        self.model = None

    def _load(self) -> None:
        from transformers import Idefics2ForConditionalGeneration

        cache_dir = self._load_kwargs.get("cache_dir")
        self.processor = load_auto_processor(
            self.model_id,
            trust_remote_code=False,
            cache_dir=cache_dir,
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            try:
                tokenizer.padding_side = "left"
                if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                pass

        load_kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        self.model = Idefics2ForConditionalGeneration.from_pretrained(
            self.model_id,
            **load_kwargs,
        ).eval()

    def _prepare_images(self, sample: TaskSample) -> List[Any]:
        if sample.modality == "image":
            return [_load_image(sample.fake_path)]

        kwargs: Dict[str, Any] = {}
        if self.video_max_frames is not None:
            kwargs["num_segments"] = self.video_max_frames
            kwargs["max_frames"] = self.video_max_frames
        frames = load_video_frames_raw(Path(sample.fake_path), **kwargs)
        return frames

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        assert self.model is not None and self.processor is not None

        user_text = strip_modality_tags(sample.prompt).strip()
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            user_text = f"{prefix}\n\n{user_text}" if user_text else prefix
        if not user_text:
            user_text = "Please describe the provided content."

        images = self._prepare_images(sample)
        user_content: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
        user_content.append({"type": "text", "text": user_text})

        messages: List[Dict[str, Any]] = []
        if self.system_hint:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_hint}]})
        messages.append({"role": "user", "content": user_content})

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        processor_kwargs: Dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = images
        inputs = self.processor(**processor_kwargs)
        inputs = {
            k: v.to(self.model.device if hasattr(self.model, "device") else torch.device("cpu"))
            if isinstance(v, torch.Tensor)
            else v
            for k, v in inputs.items()
        }

        gen_args: Dict[str, Any] = dict(max_new_tokens=self.max_new_tokens, do_sample=False, temperature=0.0)
        if "min_new_tokens" not in (self.generation_overrides or {}):
            gen_args["min_new_tokens"] = 1
        gen_args.update(self.generation_overrides or {})

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_args)

        sequences = output.sequences if hasattr(output, "sequences") else output
        start = inputs["input_ids"].shape[-1]
        gen_only = sequences[:, start:]
        text = self.processor.batch_decode(
            gen_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()


__all__ = ["Idefics2Wrapper"]
