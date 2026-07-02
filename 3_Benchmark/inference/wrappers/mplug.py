from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from inference.dataio.samples import TaskSample
from inference.utils.media import load_video_frames_raw
from inference.utils.text import strip_modality_tags
from inference.utils.torch_helpers import apply_dtype_kw

from .base import BaseModelWrapper, ModelCapabilities


_LOCAL_MODEL_ENV = "MPLUG_OWL3_MODEL_DIR"


def _expand_candidates(base: Path, model_id: str) -> List[Path]:
    cleaned = model_id.strip("/\\")
    candidates: List[Path] = []
    if cleaned:
        candidates.append(base / cleaned)
    candidates.append(base / Path(model_id).name)
    return candidates


def _find_local_model_root(model_id: str, cache_dir: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []

    env_dir = os.environ.get(_LOCAL_MODEL_ENV)
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    direct_path = Path(model_id)
    if direct_path.is_dir():
        candidates.append(direct_path)

    if cache_dir:
        candidates.extend(_expand_candidates(Path(cache_dir), model_id))

    fallback_base = Path(__file__).resolve().parents[2] / "models" / "mPLUG"
    candidates.extend(_expand_candidates(fallback_base, model_id))

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir() and (resolved / "config.json").is_file():
            return resolved
    return None


def _infer_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        try:
            return torch.device(model.device)
        except Exception:
            pass
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


class MplugOwl3Wrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=True, supports_video=True)

    def __init__(self, *args: Any, video_max_frames: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.video_max_frames = video_max_frames
        self.processor = None
        self.tokenizer = None

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._ensure_cache_compat()

        cache_dir = self._load_kwargs.get("cache_dir")
        local_root = _find_local_model_root(self.model_id, cache_dir)
        model_source = str(local_root) if local_root else self.model_id

        if local_root:
            os.environ.setdefault(_LOCAL_MODEL_ENV, str(local_root))

        load_kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        load_kwargs.setdefault("trust_remote_code", True)
        self.model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs).eval()

        tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)

        init_processor = getattr(self.model, "init_processor", None)
        if not callable(init_processor):
            raise RuntimeError("Loaded mPLUG-Owl3 model does not expose init_processor(tokenizer).")
        self.processor = init_processor(self.tokenizer)

    @staticmethod
    def _ensure_cache_compat() -> None:
        try:
            from transformers.cache_utils import Cache  # type: ignore
        except Exception:
            return

        if hasattr(Cache, "get_max_length"):
            return

        def _fallback_max_length(self):  # type: ignore[no-untyped-def]
            return getattr(self, "max_cache_length", None)

        try:
            setattr(Cache, "get_max_length", _fallback_max_length)  # type: ignore[arg-type]
        except Exception:
            pass

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system_hint:
            messages.append({"role": "system", "content": self.system_hint})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": ""})
        return messages

    def _prepare_prompt(self, sample: TaskSample) -> str:
        body = strip_modality_tags(sample.prompt or "").strip()
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            body = f"{prefix}\n\n{body}" if body else prefix
        if not body:
            body = "Please describe the provided content."

        if sample.modality == "image":
            tag = "<|image|>"
        else:
            tag = "<|video|>"
        return f"{tag}\n{body}" if body else f"{tag}\n"

    def _prepare_media(self, sample: TaskSample) -> Dict[str, Any]:
        if sample.modality == "image":
            image = Image.open(sample.fake_path).convert("RGB")
            return {"images": [image], "videos": None}

        frames = load_video_frames_raw(
            Path(sample.fake_path),
            max_frames=self.video_max_frames,
        )
        return {"images": None, "videos": [frames]}

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        prompt = self._prepare_prompt(sample)
        messages = self._build_messages(prompt)
        media_payload = self._prepare_media(sample)

        model_inputs = self.processor(
            messages,
            images=media_payload["images"],
            videos=media_payload["videos"],
            return_tensors="pt",
        )

        device = _infer_device(self.model)
        model_inputs = model_inputs.to(device=device)

        inputs: Dict[str, Any] = dict(model_inputs)
        inputs["tokenizer"] = self.tokenizer
        inputs.setdefault("decode_text", True)

        gen_args: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "num_beams": 1,
        }
        if "min_new_tokens" not in (self.generation_overrides or {}):
            gen_args["min_new_tokens"] = 1
        if self.generation_overrides:
            gen_args.update(self.generation_overrides)
        inputs.update(gen_args)

        with torch.inference_mode():
            result = self.model.generate(**inputs)

        if isinstance(result, (list, tuple)):
            answer = result[0] if result else ""
        else:
            answer = result
        text = str(answer)
        return text if self.preserve_whitespace else text.strip()


__all__ = ["MplugOwl3Wrapper"]
