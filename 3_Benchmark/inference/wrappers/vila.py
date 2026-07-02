from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from inference.dataio.samples import TaskSample
from inference.utils.media import load_video_frames_raw
from inference.utils.torch_helpers import apply_dtype_kw, load_auto_processor

from .base import BaseModelWrapper, ModelCapabilities


class VILAWrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=True, supports_video=True)

    def __init__(self, *args, video_max_frames: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_max_frames = video_max_frames

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM

        cache_dir = self._load_kwargs.get("cache_dir")
        self.processor = load_auto_processor(self.model_id, trust_remote_code=True, cache_dir=cache_dir)
        kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, **kwargs
        ).eval()

    def _compose_messages(self, user_text: str) -> List[Dict[str, List[Dict[str, str]]]]:
        messages: List[Dict[str, List[Dict[str, str]]]] = []
        if self.system_hint:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_hint}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        return messages

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        from PIL import Image

        user_text = f"{sample.prompt}\n"
        if self.user_prefix:
            user_text = self.user_prefix + "\n\n" + user_text
        messages = self._compose_messages(user_text)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        with torch.inference_mode():
            if sample.modality == "image":
                image = Image.open(sample.fake_path).convert("RGB")
                inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            else:
                try:
                    inputs = self.processor(text=prompt, videos=str(sample.fake_path), return_tensors="pt")
                except Exception:
                    frames = load_video_frames_raw(Path(sample.fake_path), num_segments=self.video_max_frames)
                    inputs = self.processor(text=prompt, videos=[frames], return_tensors="pt")

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            gen_args = dict(max_new_tokens=self.max_new_tokens, do_sample=False, temperature=0.0)
            gen_args.update(self.generation_overrides or {})
            output = self.model.generate(**inputs, **gen_args)
            text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        return text.strip()
