from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Type

import torch
from huggingface_hub import snapshot_download
from transformers import GenerationConfig

from inference.dataio.samples import TaskSample
from inference.utils.text import strip_modality_tags

from .base import BaseModelWrapper, ModelCapabilities


def _expand(path: Optional[str | os.PathLike[str]]) -> Optional[Path]:
    if not path:
        return None
    return Path(path).expanduser().resolve()


class AudioFlamingoWrapper(BaseModelWrapper):
    """Wrapper for NVIDIA Audio Flamingo 3 (audio→text)."""

    capabilities = ModelCapabilities(supports_image=False, supports_video=False, supports_audio=True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model = None
        self._sound_factory = None
        self._model_dir: Optional[Path] = None
        self._think_mode = os.environ.get("AUDIO_FLAMINGO_THINK_MODE", "").lower() in {"1", "true", "yes"}

    # ----- helpers ---------------------------------------------------------

    def _resolve_snapshot(self) -> Path:
        load_kw = dict(self._load_kwargs or {})
        cache_dir = _expand(load_kw.pop("cache_dir", None))
        local_files_only = bool(load_kw.pop("local_files_only", False))
        revision = load_kw.pop("revision", None)
        token = load_kw.pop("token", None)

        local_override = _expand(self.model_id)
        if local_override and local_override.is_dir():
            return local_override

        snapshot_kwargs: Dict[str, Any] = {
            "repo_id": self.model_id,
            "repo_type": "model",
        }
        if cache_dir:
            snapshot_kwargs["cache_dir"] = str(cache_dir)
        if local_files_only:
            snapshot_kwargs["local_files_only"] = True
        if revision:
            snapshot_kwargs["revision"] = revision
        if token is not None:
            snapshot_kwargs["token"] = token

        try:
            resolved = snapshot_download(**snapshot_kwargs)
        except Exception as exc:  # noqa: BLE001
            hint = "Set --cache-dir to the download location or run huggingface-cli download first."
            raise RuntimeError(f"Failed to obtain Audio Flamingo snapshot: {exc}\n{hint}") from exc
        return Path(resolved)

    def _prepare_sys_path(self, root: Path) -> None:
        path_str = str(root)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    def _resolve_sound_factory(self, llava_mod: Any) -> Type[Any]:
        sound_cls = getattr(llava_mod, "Sound", None)
        if sound_cls is not None:
            return sound_cls

        class _CompatSound:
            def __init__(self, path: str) -> None:
                self.path = str(path)

            def __repr__(self) -> str:
                return f"Sound(path={self.path!r})"

        setattr(llava_mod, "Sound", _CompatSound)
        return _CompatSound

    # ----- load ------------------------------------------------------------

    def _load(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Audio Flamingo requires a CUDA-enabled GPU for inference.")

        self._model_dir = self._resolve_snapshot()
        self._prepare_sys_path(self._model_dir)

        try:
            import llava  # type: ignore
            from llava import conversation as clib  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Failed to import Audio Flamingo modules. Ensure dependencies from NVIDIA/audio-flamingo "
                "(e.g., transformers>=4.44, torchaudio, gradio, peft) are installed."
            ) from exc

        load_kw = dict(self._load_kwargs or {})
        device_map = load_kw.pop("device_map", "auto")
        load_kw.pop("cache_dir", None)
        load_kw.pop("local_files_only", None)
        load_kw.pop("revision", None)
        load_kw.pop("token", None)

        model: Any
        load_fn = getattr(llava, "load", None)

        if callable(load_fn):
            model_obj = load_fn(
                str(self._model_dir),
                device_map=device_map,
                **load_kw,
            )
            if isinstance(model_obj, tuple):
                model = next((item for item in model_obj if hasattr(item, "generate_content")), model_obj[1])
            else:
                model = model_obj
        else:
            fallback_model = None
            entry_spec = None
            try:  # pragma: no cover - dynamic import guard
                import importlib.util as _importlib_util

                entry_spec = _importlib_util.find_spec("llava.entry")
            except Exception:
                entry_spec = None

            if entry_spec is not None:
                try:
                    from llava.entry import load as entry_load  # type: ignore

                    fallback_model = entry_load(
                        str(self._model_dir),
                        device_map=device_map,
                        **load_kw,
                    )
                except Exception:
                    fallback_model = None

            if fallback_model is None:
                try:
                    from llava.mm_utils import get_model_name_from_path  # type: ignore
                    from llava.model.builder import load_pretrained_model  # type: ignore
                except Exception as exc:  # pragma: no cover - dependency guard
                    raise ImportError(
                        "Audio Flamingo requires a llava installation that exposes `llava.load`, "
                        "`llava.entry.load`, or builder utilities. Install the NVIDIA/audio-flamingo "
                        "llava fork (see README requirements)."
                    ) from exc

                model_name = get_model_name_from_path(str(self._model_dir))
                builder_kwargs = dict(load_kw)
                flash_flag = builder_kwargs.pop("flash_attn", None)
                attn_impl = builder_kwargs.pop("attn_implementation", None)
                use_flash_attn = bool(flash_flag) if flash_flag is not None else attn_impl == "flash_attention_2"
                try:
                    signature = inspect.signature(load_pretrained_model)
                except (TypeError, ValueError):
                    signature = None
                has_var_kw = False
                if signature is not None:
                    has_var_kw = any(
                        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
                    )

                def _allow_param(param_name: str) -> bool:
                    if signature is None or has_var_kw:
                        return True
                    return param_name in signature.parameters

                call_kwargs = {k: v for k, v in builder_kwargs.items() if _allow_param(k)}

                if _allow_param("use_flash_attn"):
                    call_kwargs["use_flash_attn"] = use_flash_attn
                else:
                    if flash_flag is not None and _allow_param("flash_attn"):
                        call_kwargs["flash_attn"] = flash_flag
                    if attn_impl is not None and _allow_param("attn_implementation"):
                        call_kwargs["attn_implementation"] = attn_impl
                tokenizer, model_candidate, *_ = load_pretrained_model(
                    str(self._model_dir),
                    None,
                    model_name,
                    device_map=device_map,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    **call_kwargs,
                )
                fallback_model = model_candidate

            if fallback_model is None:
                raise ImportError(
                    "Failed to construct Audio Flamingo model via available llava loaders. "
                    "Ensure the NVIDIA/audio-flamingo llava fork is installed."
                )

            if isinstance(fallback_model, tuple):
                model = next((item for item in fallback_model if hasattr(item, "generate_content")), fallback_model[1])
            else:
                model = fallback_model

        model = model.to("cuda")

        if self._think_mode:
            stage_dir = self._model_dir / "stage35"
            if not stage_dir.is_dir():
                raise FileNotFoundError(
                    "AUDIO_FLAMINGO_THINK_MODE=1 but stage35 weights are missing. Download the full checkpoint "
                    "or unset the environment variable."
                )
            non_lora_path = stage_dir / "non_lora_trainables.bin"
            if non_lora_path.is_file():
                non_lora = torch.load(non_lora_path, map_location="cpu")
                patched = {k[6:] if k.startswith("model.") else k: v for k, v in non_lora.items()}
                model.load_state_dict(patched, strict=False)
            try:
                from peft import PeftModel  # type: ignore
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "Think mode requires the `peft` package. Install it (e.g., pip install peft)."
                ) from exc
            model = PeftModel.from_pretrained(
                model,
                str(stage_dir),
                device_map="auto",
                torch_dtype=torch.float16,
        )

        clib.default_conversation = clib.conv_templates["auto"].copy()
        self._model = model
        self._sound_factory = self._resolve_sound_factory(llava)

    # ----- generation ------------------------------------------------------

    def _build_generation_config(self) -> GenerationConfig:
        base_cfg = getattr(self._model, "default_generation_config", None)
        if isinstance(base_cfg, GenerationConfig):
            gen_cfg = base_cfg.clone()
        else:
            gen_cfg = GenerationConfig.from_model_config(self._model.config)

        gen_cfg.max_new_tokens = self.max_new_tokens
        overrides = dict(self.generation_overrides or {})
        if "min_new_tokens" in overrides:
            gen_cfg.min_new_tokens = overrides["min_new_tokens"]
        if "do_sample" in overrides:
            gen_cfg.do_sample = bool(overrides["do_sample"])
        else:
            gen_cfg.do_sample = False
        if "temperature" in overrides:
            gen_cfg.temperature = overrides["temperature"]
        if "top_p" in overrides:
            gen_cfg.top_p = overrides["top_p"]
        if "top_k" in overrides:
            gen_cfg.top_k = overrides["top_k"]
        if "repetition_penalty" in overrides:
            gen_cfg.repetition_penalty = overrides["repetition_penalty"]
        if "num_beams" in overrides:
            gen_cfg.num_beams = overrides["num_beams"]
        return gen_cfg

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        if sample.modality != "audio":
            raise ValueError(f"Audio Flamingo only supports audio modality. Received: {sample.modality}")
        assert self._model is not None and self._sound_factory is not None

        prompt = strip_modality_tags(sample.prompt or "").strip()
        pieces: Iterable[str] = []
        if self.system_hint:
            pieces = (*pieces, self.system_hint.strip())
        if self.user_prefix:
            pieces = (*pieces, self.user_prefix.strip())
        if prompt:
            pieces = (*pieces, prompt)
        merged_prompt = "\n\n".join(pieces)
        if not merged_prompt:
            merged_prompt = "Please describe the audio."

        audio_path = Path(sample.fake_path).resolve()
        media = self._sound_factory(str(audio_path))
        gen_cfg = self._build_generation_config()

        try:
            response = self._model.generate_content([media, merged_prompt], generation_config=gen_cfg)
        except Exception as exc:  # noqa: BLE001
            return f"[ERROR] {exc.__class__.__name__}: {exc}"
        return str(response).strip()


__all__ = ["AudioFlamingoWrapper"]
