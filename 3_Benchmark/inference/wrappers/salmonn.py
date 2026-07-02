from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch

from inference.dataio.samples import TaskSample
from inference.utils.text import strip_modality_tags

from .base import BaseModelWrapper, ModelCapabilities


def _expand(path: Optional[str | os.PathLike[str]]) -> Optional[Path]:
    if not path:
        return None
    return Path(path).expanduser().resolve()


def _first_existing(paths: Iterable[Path], *, require_file: bool = False) -> Optional[Path]:
    for candidate in paths:
        if not candidate:
            continue
        try:
            if require_file and candidate.is_file():
                return candidate
            if not require_file and candidate.is_dir():
                return candidate
        except OSError:
            continue
    return None


class SalmonnWrapper(BaseModelWrapper):
    """
    Wrapper for tsinghua-ee/SALMONN (audio-only large model).

    SALMONN relies on several external checkpoints (Whisper Large-v2, BEATs AS2M,
    Vicuna 13B, and the SALMONN LoRA). The wrapper expects these assets to be
    placed on disk and tries to locate them automatically. You can explicitly
    point to the resources via the following environment variables:
      - SALMONN_CKPT_PATH
      - SALMONN_WHISPER_PATH
      - SALMONN_BEATS_PATH
      - SALMONN_VICUNA_PATH

    If automatic discovery fails, set the variables above to the appropriate
    directories/files.
    """

    capabilities = ModelCapabilities(supports_image=False, supports_video=False, supports_audio=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_dir: Optional[Path] = None
        self._cache_dir: Optional[Path] = None
        self.device: Optional[torch.device] = None
        self.model = None

    # ----- path helpers --------------------------------------------------

    def _resolve_model_dir(self) -> Path:
        raw = Path(str(self.model_id)).expanduser()
        if raw.is_dir():
            return raw

        load_cache = None
        if self._load_kwargs:
            load_cache = _expand(self._load_kwargs.get("cache_dir"))
        env_cache = _expand(os.environ.get("HF_HOME")) or _expand(os.environ.get("HUGGINGFACE_HUB_CACHE"))

        candidates = []
        if load_cache:
            candidates.extend([
                load_cache / str(self.model_id),
                load_cache / str(self.model_id).split("/")[-1],
            ])
        if env_cache:
            candidates.extend([
                env_cache / str(self.model_id),
                env_cache / str(self.model_id).split("/")[-1],
            ])
        found = _first_existing(candidates)
        if found:
            return found
        raise FileNotFoundError(
            f"Unable to locate SALMONN repository for '{self.model_id}'. "
            "Download it locally (e.g. huggingface-cli download tsinghua-ee/SALMONN --local-dir <path>) "
            "or pass a direct path."
        )

    def _resolve_ckpt_path(self, model_dir: Path) -> Path:
        env_override = _expand(os.environ.get("SALMONN_CKPT_PATH"))
        if env_override:
            if env_override.is_file():
                return env_override
            raise FileNotFoundError(f"SALMONN_CKPT_PATH points to a missing file: {env_override}")

        default = model_dir / "salmonn_v1.pth"
        if default.is_file():
            return default
        raise FileNotFoundError(
            f"Cannot find SALMONN checkpoint. Expected '{default}'. "
            "Set SALMONN_CKPT_PATH to the downloaded salmonn_v1.pth file."
        )

    def _resolve_whisper_path(self, model_dir: Path) -> Path:
        env_override = _expand(os.environ.get("SALMONN_WHISPER_PATH") or os.environ.get("WHISPER_PATH"))
        if env_override:
            if env_override.is_dir():
                return env_override
            raise FileNotFoundError(f"SALMONN_WHISPER_PATH points to a missing directory: {env_override}")

        candidates = [
            model_dir / "whisper-large-v2",
            model_dir / "whisper",
        ]
        if self._cache_dir:
            candidates.extend([
                self._cache_dir / "openai" / "whisper-large-v2",
                self._cache_dir / "whisper-large-v2",
            ])
        found = _first_existing(candidates)
        if found:
            return found
        raise FileNotFoundError(
            "Cannot locate Whisper Large-v2 assets required by SALMONN. "
            "Set SALMONN_WHISPER_PATH to the directory containing Whisper (e.g., openai/whisper-large-v2)."
        )

    def _resolve_beats_path(self, model_dir: Path) -> Path:
        env_override = _expand(os.environ.get("SALMONN_BEATS_PATH") or os.environ.get("BEATS_PATH"))
        if env_override:
            if env_override.is_file():
                return env_override
            raise FileNotFoundError(f"SALMONN_BEATS_PATH points to a missing file: {env_override}")

        candidates = [
            model_dir / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            model_dir / "beats" / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        ]
        if self._cache_dir:
            candidates.append(self._cache_dir / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        # Last resort: scan immediate subdirectories for .pt
        possible = list(model_dir.glob("**/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"))
        if self._cache_dir and not possible:
            possible = list(self._cache_dir.glob("**/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"))
        for cand in possible:
            if cand.is_file():
                return cand

        raise FileNotFoundError(
            "Cannot locate BEATs checkpoint 'BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'. "
            "Download it and set SALMONN_BEATS_PATH to the file."
        )

    def _resolve_vicuna_path(self, model_dir: Path) -> Path:
        env_override = _expand(os.environ.get("SALMONN_VICUNA_PATH") or os.environ.get("VICUNA_PATH"))
        if env_override:
            if env_override.is_dir():
                return env_override
            raise FileNotFoundError(f"SALMONN_VICUNA_PATH points to a missing directory: {env_override}")

        candidates = [
            model_dir / "vicuna-13b-v1.1",
            model_dir / "vicuna",
        ]
        if self._cache_dir:
            candidates.extend([
                self._cache_dir / "lmsys" / "vicuna-13b-v1.1",
                self._cache_dir / "vicuna-13b-v1.1",
            ])
        found = _first_existing(candidates)
        if found:
            return found

        if self._cache_dir:
            vicuna_dir = next(self._cache_dir.glob("**/vicuna-13b-v1.1"), None)
            if vicuna_dir and vicuna_dir.is_dir():
                return vicuna_dir

        raise FileNotFoundError(
            "Cannot locate Vicuna 13B v1.1 weights required by SALMONN. "
            "Set SALMONN_VICUNA_PATH to the Vicuna directory."
        )

    # ----- load & inference ----------------------------------------------

    def _load(self) -> None:
        self._model_dir = self._resolve_model_dir()
        if self._load_kwargs:
            self._cache_dir = _expand(self._load_kwargs.get("cache_dir"))

        if str(self._model_dir) not in sys.path:
            sys.path.insert(0, str(self._model_dir))

        try:
            from model import SALMONN  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Failed to import SALMONN from the local repository. "
                "Ensure tsinghua-ee/SALMONN is downloaded completely."
            ) from exc

        ckpt_path = self._resolve_ckpt_path(self._model_dir)
        whisper_path = self._resolve_whisper_path(self._model_dir)
        beats_path = self._resolve_beats_path(self._model_dir)
        vicuna_path = self._resolve_vicuna_path(self._model_dir)

        if not torch.cuda.is_available():
            raise RuntimeError("SALMONN currently requires a CUDA-enabled GPU for inference.")

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        device_index = 0
        if visible and visible[0].strip():
            try:
                device_index = int(visible[0])
            except ValueError:
                device_index = 0
        self.device = torch.device(f"cuda:{device_index}")

        self.model = SALMONN(
            ckpt=str(ckpt_path),
            whisper_path=str(whisper_path),
            beats_path=str(beats_path),
            vicuna_path=str(vicuna_path),
            lora_alpha=32,
            low_resource=False,
        )
        self.model.to(self.device)
        self.model.eval()

        # Guard against problematic defaults in generation config (e.g., negative max_length)
        try:
            gen_cfg = getattr(self.model.llama_model, "generation_config", None)
            if gen_cfg is not None and getattr(gen_cfg, "max_length", None) is not None:
                if isinstance(gen_cfg.max_length, int) and gen_cfg.max_length < 0:
                    gen_cfg.max_length = None
        except Exception:
            pass

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        if sample.modality != "audio":
            raise ValueError(f"SALMONN only supports audio modality. Received: {sample.modality}")
        if not self.model or not self.device:
            raise RuntimeError("SALMONN model is not initialized.")

        # Build the prompt by prepending system hint and an anti-echo directive
        # so models do not parrot the template. Keep this lightweight and
        # only active when a system hint is provided (e.g., detection tasks).
        base_prompt = strip_modality_tags(sample.prompt or "").strip()

        prefix_parts = []
        if self.system_hint:
            prefix_parts.append(self.system_hint.strip())
            # Anti-echo guard specifically for formatted detection prompts.
            prefix_parts.append(
                (
                "\nYou are a forensic media authenticity inspector.\nRole:\n- Decide whether each provided sample is authentic or manipulated.\n- Explain concrete, observable artifacts found in the sample.\n\nHard Constraints:\n- Follow the required output format exactly.\n- The first line of your response must be a single line: either \"Likely Authentic.\" or \"Likely Manipulated.\"\n- Use precise, neutral, technical language.\n"
                )
            )
        if self.user_prefix:
            prefix_parts.append(self.user_prefix.strip())

        if base_prompt:
            prefix_parts.append(base_prompt)
        prompt = "\n\n".join(p for p in prefix_parts if p)
        if not prompt:
            prompt = "Please describe the audio."

        raw_outputs = self.model.generate(
            sample.fake_path,
            prompt=prompt,
            device=str(self.device),
            max_new_tokens=self.max_new_tokens,
        )
        if not raw_outputs:
            return ""
        text = str(raw_outputs[0]).strip()
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[-1].strip()
        return text


__all__ = ["SalmonnWrapper"]
