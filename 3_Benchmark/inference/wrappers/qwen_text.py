from __future__ import annotations

from typing import Any, Dict, List

import torch

from inference.dataio.samples import TaskSample
from inference.utils.text import strip_modality_tags
from inference.utils.torch_helpers import apply_dtype_kw

from .base import BaseModelWrapper, ModelCapabilities


def _infer_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        try:
            return torch.device(model.device)
        except Exception:
            pass
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


class Qwen3TextWrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=False, supports_video=False, supports_audio=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Qwen3's model card recommends sampling (not greedy) and provides separate
        # best-practice params for thinking/non-thinking modes.
        super().__init__(*args, **kwargs)
        self.enable_thinking: bool = False
        self.tokenizer = None
        self.model = None

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).eval()

        cache_dir = self._load_kwargs.get("cache_dir")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True, cache_dir=cache_dir)
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache_dir)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _build_prompt(self, sample: TaskSample) -> tuple[str, bool]:
        body = strip_modality_tags(sample.prompt or "").strip()
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            body = f"{prefix}\n\n{body}" if body else prefix
        if not body:
            body = "Please follow the instructions."

        messages: List[Dict[str, str]] = []
        if self.system_hint:
            messages.append({"role": "system", "content": self.system_hint})
        messages.append({"role": "user", "content": body})

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return (
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.enable_thinking,
                    ),
                    True,
                )
            except Exception:
                pass

        if self.system_hint:
            return f"{self.system_hint}\n\n{body}", False
        return body, False

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        assert self.model is not None and self.tokenizer is not None

        prompt, used_template = self._build_prompt(sample)
        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=not used_template,
        )

        device = _infer_device(self.model)
        model_inputs = {k: v.to(device=device) for k, v in model_inputs.items()}

        gen_args: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            # Qwen3 best practices (HF model card):
            # - DO NOT use greedy decoding; use sampling.
            "do_sample": True,
            "temperature": 0.7 if not self.enable_thinking else 0.6,
            "top_p": 0.8 if not self.enable_thinking else 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if "min_new_tokens" not in (self.generation_overrides or {}):
            # Avoid immediate <|im_end|> / empty generations.
            gen_args["min_new_tokens"] = 16
        if self.generation_overrides:
            gen_args.update(self.generation_overrides)

        with torch.inference_mode():
            output = self.model.generate(**model_inputs, **gen_args)

        input_len = int(model_inputs["input_ids"].shape[-1])
        # `generate` returns a tensor shaped [batch, seq] for CausalLMs.
        output_tensor = output[0] if isinstance(output, (list, tuple)) else output
        if isinstance(output_tensor, torch.Tensor) and output_tensor.ndim == 2:
            output_tensor = output_tensor[0]
        new_tokens = output_tensor[input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text if self.preserve_whitespace else text.strip()


__all__ = ["Qwen3TextWrapper"]
