from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import torch

from inference.dataio.samples import TaskSample
from inference.utils.media import load_video_frames_raw
from inference.utils.text import strip_modality_tags
from inference.utils.torch_helpers import apply_dtype_kw

from .base import BaseModelWrapper, ModelCapabilities

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".3gp",
}
MODALITY_NORMALIZER = {
    "img": "image",
    "image": "image",
    "vid": "video",
    "video": "video",
}


class LlavaOneVisionWrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=True, supports_video=True)

    def __init__(self, *args, video_max_frames: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_max_frames = max(1, video_max_frames)
        self.processor = None
        self.model = None
        self.num_frames = self.video_max_frames

    def _load(self) -> None:
        from transformers import AutoProcessor

        try:
            from transformers import LlavaOnevisionForConditionalGeneration as LlavaModelCls
        except ImportError:
            # Older Transformers versions do not expose the dedicated Llava class.
            # Fall back to the generic vision-to-seq auto loader.
            try:
                from transformers import AutoModelForVision2Seq as LlavaModelCls
            except ImportError as exc:  # noqa: PT009
                raise ImportError(
                    "Neither LlavaOnevisionForConditionalGeneration nor AutoModelForVision2Seq "
                    "is available in the installed Transformers package."
                ) from exc
            warnings.warn(
                "Falling back to AutoModelForVision2Seq for LLaVA-OneVision; consider upgrading transformers.",
                RuntimeWarning,
                stacklevel=2,
            )

        cache_dir = self._load_kwargs.get("cache_dir")
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True, cache_dir=cache_dir)
        try:
            if hasattr(self.processor, "tokenizer"):
                self.processor.tokenizer.padding_side = "left"
        except Exception:
            pass
        kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        self.model = LlavaModelCls.from_pretrained(self.model_id, **kwargs).eval()
        self.num_frames = self.video_max_frames

    def _resolve_media_kind(self, sample: TaskSample) -> str:
        """Infer whether the current sample should be treated as an image or video."""
        suffix = Path(sample.fake_path).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        declared = MODALITY_NORMALIZER.get((sample.modality or "").lower())
        if declared in {"image", "video"}:
            return declared
        raise ValueError(f"Unsupported modality '{sample.modality}' for LLaVA-OneVision.")

    def _build_messages(self, media_kind: str, user_text: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if media_kind == "image":
            content.append({"type": "image"})
        else:
            content.append({"type": "video"})
        content.append({"type": "text", "text": user_text})

        messages: List[Dict[str, Any]] = []
        if self.system_hint:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_hint}]})
        messages.append({"role": "user", "content": content})
        return messages

    def _load_image(self, sample_path: Path):
        from PIL import Image

        with Image.open(sample_path) as img:
            return img.convert("RGB")

    def _prepare_processor_inputs(
        self,
        prompt: str,
        media_kind: str,
        sample_path: Path,
    ) -> Dict[str, Any]:
        processor_kwargs: Dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
        if media_kind == "image":
            processor_kwargs["images"] = self._load_image(sample_path)
        else:
            frames = load_video_frames_raw(
                sample_path,
                num_segments=self.video_max_frames,
                max_frames=self.video_max_frames,
            )
            self.num_frames = min(len(frames), self.video_max_frames)
            processor_kwargs["videos"] = [frames]
            processor_kwargs["num_frames"] = self.num_frames
        return self.processor(**processor_kwargs)

    def _sanitize_processor_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = dict(payload)
        for key in ["batch_num_images", "batch_num_frames", "num_images", "num_frames"]:
            cleaned.pop(key, None)
        kwargs = cleaned.get("kwargs")
        if isinstance(kwargs, dict):
            kwargs.pop("batch_num_images", None)
            kwargs.pop("batch_num_frames", None)
        return cleaned

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        assert self.processor is not None and self.model is not None

        media_kind = self._resolve_media_kind(sample)
        user_text = strip_modality_tags(sample.prompt).strip()
        if self.user_prefix:
            user_text = f"{self.user_prefix}\n\n{user_text}"

        messages = self._build_messages(media_kind, user_text)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        with torch.inference_mode():
            proc_inputs = self._prepare_processor_inputs(prompt, media_kind, Path(sample.fake_path))
            proc_inputs = self._sanitize_processor_payload(proc_inputs)
            inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in proc_inputs.items()}

            gen_args: Dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "temperature": 0.0,
            }
            if "min_new_tokens" not in (self.generation_overrides or {}):
                gen_args["min_new_tokens"] = 1
            gen_args.update(self.generation_overrides or {})
            output = self.model.generate(**inputs, **gen_args)

        sequences = output.sequences if hasattr(output, "sequences") else output
        start = inputs["input_ids"].shape[-1]
        gen_only = sequences[:, start:]
        text = self.processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text if self.preserve_whitespace else text.strip()


class Llava15Wrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=True, supports_video=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = None
        self.model = None

    def _load(self) -> None:
        from transformers import LlavaProcessor

        try:
            from transformers import LlavaForConditionalGeneration as LlavaModelCls
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as LlavaModelCls
            except ImportError as exc:  # noqa: PT009
                raise ImportError(
                    "Neither LlavaForConditionalGeneration nor AutoModelForVision2Seq is available in the installed Transformers package."
                ) from exc
            warnings.warn(
                "Falling back to AutoModelForVision2Seq for LLaVA-1.5; consider upgrading transformers.",
                RuntimeWarning,
                stacklevel=2,
            )

        cache_dir = self._load_kwargs.get("cache_dir")
        self.processor = LlavaProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)

        try:
            if hasattr(self.processor, "tokenizer"):
                self.processor.tokenizer.padding_side = "left"
        except Exception:
            pass

        kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        self.model = LlavaModelCls.from_pretrained(self.model_id, **kwargs).eval()

    def _build_messages(self, user_text: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_hint:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_hint}]})
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]})
        return messages

    def _build_prompt(self, user_text: str) -> str:
        assert self.processor is not None
        parts: List[str] = []
        if self.system_hint:
            parts.append(self.system_hint.strip())
        parts.append(f"USER: <image>\n{user_text}\nASSISTANT:")
        return "\n\n".join(part for part in parts if part)

    def _load_image(self, sample_path: Path):
        from PIL import Image

        with Image.open(sample_path) as img:
            return img.convert("RGB")

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        assert self.processor is not None and self.model is not None

        if sample.modality != "image":
            raise ValueError(f"Unsupported modality '{sample.modality}' for LLaVA-1.5.")

        user_text = strip_modality_tags(sample.prompt).strip()
        if self.user_prefix:
            user_text = f"{self.user_prefix}\n\n{user_text}" if user_text else self.user_prefix
        if not user_text:
            user_text = "Please answer the question about the image."

        prompt = self._build_prompt(user_text)

        with torch.inference_mode():
            proc_inputs = self.processor(
                text=prompt,
                images=self._load_image(Path(sample.fake_path)),
                return_tensors="pt",
            )
            inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in proc_inputs.items()}

            gen_args: Dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }

            if "min_new_tokens" not in (self.generation_overrides or {}):
                gen_args["min_new_tokens"] = 1
            gen_args.update(self.generation_overrides or {})

            output = self.model.generate(**inputs, **gen_args)

        sequences = output.sequences if hasattr(output, "sequences") else output
        start = inputs["input_ids"].shape[-1]
        gen_only = sequences[:, start:]
        text = self.processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text if self.preserve_whitespace else text.strip()
