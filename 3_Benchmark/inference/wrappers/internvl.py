from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch

from inference.dataio.samples import TaskSample
from inference.utils.media import load_image_to_pixel_values, load_video_to_pixel_values
from inference.utils.torch_helpers import apply_dtype_kw

from .base import BaseModelWrapper, ModelCapabilities


class InternVLWrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=True, supports_video=True)

    def __init__(self, *args, video_max_frames: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_max_frames = video_max_frames

    def _load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        cache_dir = self._load_kwargs.get("cache_dir")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False, cache_dir=cache_dir
        )
        kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)

        # 🔧 避免重複 key
        kwargs.pop("attn_implementation", None)

        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            attn_implementation="eager",  # 明確指定
            **kwargs,
        ).eval()


    def _model_device(self):
        return getattr(self.model, "device", None) or next(self.model.parameters()).device

    def _chat(
        self,
        pixel_values: torch.Tensor,
        question: str,
        num_patches_list: Optional[List[int]] = None,
    ) -> str:
        gen_cfg = dict(max_new_tokens=self.max_new_tokens, do_sample=False, temperature=0.0)
        gen_cfg.update(self.generation_overrides or {})
        with torch.inference_mode():
            try:
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=gen_cfg,
                    num_patches_list=num_patches_list,
                    history=None,
                )
            except TypeError:
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    num_patches_list=num_patches_list,
                    history=None,
                    **gen_cfg,
                )
        if isinstance(response, (tuple, list)):
            response = response[0]
        return str(response).strip()

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        device = self._model_device()
        dtype = self._torch_dtype

        base_text = f"{sample.prompt}\n"
        if self.user_prefix:
            base_text = self.user_prefix + "\n\n" + base_text
        if self.system_hint:
            base_text = self.system_hint + "\n\n" + base_text

        if sample.modality == "image":
            pixel_values = load_image_to_pixel_values(Path(sample.fake_path), input_size=448, max_num=12)
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            question = "<image>\n" + base_text
            return self._chat(pixel_values, question, num_patches_list=None)

        pixel_values, num_patches_list, _ = load_video_to_pixel_values(
            Path(sample.fake_path),
            input_size=448,
            max_num=1,
            num_segments=self.video_max_frames,
            max_frames=self.video_max_frames,
        )
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
        question = video_prefix + base_text
        return self._chat(pixel_values, question, num_patches_list=num_patches_list)
