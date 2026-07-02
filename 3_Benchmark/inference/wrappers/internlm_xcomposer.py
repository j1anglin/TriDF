from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from inference.dataio.samples import TaskSample
from inference.utils.text import strip_modality_tags
from inference.utils.torch_helpers import apply_dtype_kw

from .base import BaseModelWrapper, ModelCapabilities


_LOCAL_MODEL_ENV = "INTERNLM_XCOMPOSER_MODEL_DIR"


def _expand_candidates(base: Path, model_id: str) -> List[Path]:
    """Return possible local paths for a given model id within base."""
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

    fallback_base = Path(__file__).resolve().parents[2] / "models_local"
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


class InternLMXComposer2D5Wrapper(BaseModelWrapper):
    """Wrapper for `internlm/internlm-xcomposer2d5-7b`."""

    capabilities = ModelCapabilities(supports_image=True, supports_video=True)

    def __init__(self, *args: Any, hd_num: int = 24, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hd_num = hd_num

    def _load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        cache_dir = self._load_kwargs.get("cache_dir")
        local_root = _find_local_model_root(self.model_id, cache_dir)
        model_source = str(local_root) if local_root else self.model_id

        if local_root:
            font_path = local_root / "SimHei.ttf"
            if font_path.is_file():
                os.environ.setdefault("INTERNLM_XCOMPOSER_FONT", str(font_path))
            vision_dir = local_root / "internlm-xcomposer2d5-clip"
            if vision_dir.is_dir():
                os.environ.setdefault("INTERNLM_XCOMPOSER_VISION_TOWER", str(vision_dir))
            os.environ.setdefault(_LOCAL_MODEL_ENV, str(local_root))

        load_kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        load_kwargs.setdefault("trust_remote_code", True)
        self.model = AutoModel.from_pretrained(model_source, **load_kwargs).eval()

        tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
        try:
            self.model.tokenizer = self.tokenizer  # some utilities expect this hook
        except Exception:  # pragma: no cover - best effort
            pass

    def _prepare_query(self, sample: TaskSample) -> str:
        body = strip_modality_tags(sample.prompt or "").strip()
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            body = f"{prefix}\n\n{body}" if body else prefix
        if not body:
            body = "Please analyze the provided content."
        return body

    def _build_media_list(self, sample: TaskSample) -> List[str]:
        # InternLM-XComposer expects a list even for single inputs.
        return [sample.fake_path]

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()

        query = self._prepare_query(sample)
        media_inputs = self._build_media_list(sample)

        call_kwargs: Dict[str, Any] = {
            "tokenizer": self.tokenizer,
            "query": query,
            "image": media_inputs,
            "history": [],
            "hd_num": self.hd_num,
        }

        gen_args: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "temperature": 0.0,
        }
        if self.system_hint:
            gen_args["meta_instruction"] = self.system_hint.strip()
        if "use_meta" not in (self.generation_overrides or {}):
            gen_args["use_meta"] = True

        if self.generation_overrides:
            gen_args.update(self.generation_overrides)

        response, _ = self.model.chat(**call_kwargs, **gen_args)
        return str(response).strip()


__all__ = ["InternLMXComposer2D5Wrapper"]
