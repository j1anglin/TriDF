from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch
from transformers import GenerationConfig

from inference.dataio.samples import TaskSample
from inference.utils.text import strip_modality_tags

from .base import BaseModelWrapper, ModelCapabilities

try:  # compatibility shim for older AnyGPT modules expecting cached_download
    from huggingface_hub import cached_download  # type: ignore # noqa: F401
except ImportError:
    from huggingface_hub import hf_hub_download
    import huggingface_hub

    huggingface_hub.cached_download = hf_hub_download


def _expand(path: Optional[str | os.PathLike[str]]) -> Optional[Path]:
    if not path:
        return None
    return Path(path).expanduser().resolve()


def _maybe_install_torchvision_stub() -> None:
    try:
        import torchvision  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    import enum
    import types
    from importlib.machinery import ModuleSpec

    for key in ("torchvision", "torchvision.transforms", "torchvision.io"):
        sys.modules.pop(key, None)

    torchvision_stub = types.ModuleType("torchvision")
    torchvision_stub.__spec__ = ModuleSpec("torchvision", loader=None, is_package=True)
    torchvision_stub.__path__ = []
    torchvision_stub.__all__ = ["transforms", "io"]

    transforms_stub = types.ModuleType("torchvision.transforms")
    transforms_stub.__spec__ = ModuleSpec("torchvision.transforms", loader=None, is_package=True)
    transforms_stub.__path__ = []

    class InterpolationMode(enum.Enum):  # minimal enum used by transformers.image_utils
        NEAREST = 0
        NEAREST_EXACT = 1
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
        LANCZOS = 6

    transforms_stub.InterpolationMode = InterpolationMode

    transforms_v2_stub = types.ModuleType("torchvision.transforms.v2")
    transforms_v2_stub.__spec__ = ModuleSpec("torchvision.transforms.v2", loader=None, is_package=True)
    transforms_v2_stub.__path__ = []

    class _FunctionalStub:
        def __getattr__(self, _name):
            return _torchvision_stub

    transforms_v2_stub.functional = _FunctionalStub()

    io_stub = types.ModuleType("torchvision.io")
    io_stub.__spec__ = ModuleSpec("torchvision.io", loader=None, is_package=False)

    def _torchvision_stub(*_args, **_kwargs):
        raise NotImplementedError(
            "torchvision operations are unavailable in this environment; using stub for AnyGPT audio inference."
        )

    io_stub.read_video = _torchvision_stub
    io_stub.read_video_timestamps = _torchvision_stub

    class _VideoReader:
        def __init__(self, *_args, **_kwargs):
            _torchvision_stub()

    io_stub.VideoReader = _VideoReader

    torchvision_stub.transforms = transforms_stub
    torchvision_stub.io = io_stub
    transforms_stub.v2 = transforms_v2_stub

    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = transforms_stub
    sys.modules["torchvision.transforms.v2"] = transforms_v2_stub
    sys.modules["torchvision.io"] = io_stub


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


class AnyGPTAudioWrapper(BaseModelWrapper):
    """
    Wrapper for fnlp/AnyGPT-chat focused on audio→text usage within our pipeline.

    This integrates the local AnyGPT repo included under `AnyGPT/` and relies on
    the chat inference class (AnyGPTChatInference). Only the audio modality is
    exposed here to keep integration minimal and robust.

    Required assets (set via environment variables or placed under the cache dir):
      - ANYGPT_SPEECH_TOKENIZER_PATH: path to SpeechTokenizer ckpt (e.g., ckpt.dev)
      - ANYGPT_SPEECH_TOKENIZER_CONFIG: path to SpeechTokenizer config.json
      - ANYGPT_SOUNDSTORM_PATH: path to speechtokenizer_soundstorm_mls.pt

    If not explicitly provided, the wrapper searches under the configured
    Hugging Face cache directory for the `fnlp/AnyGPT-speech-modules` repo
    and attempts to locate these files.
    """

    capabilities = ModelCapabilities(supports_image=False, supports_video=False, supports_audio=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_dir: Optional[Path] = None
        self._root_dir: Optional[Path] = None
        self._anygpt_src: Optional[Path] = None
        self._chat: Optional[object] = None
        self._conversation: Optional[object] = None
        self._conversation: Optional[object] = None

    # ----- path helpers --------------------------------------------------

    def _resolve_cache_dir(self) -> Optional[Path]:
        if self._load_kwargs:
            cd = _expand(self._load_kwargs.get("cache_dir"))
            if cd:
                return cd
        env = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        return _expand(env)

    def _ensure_repo_paths(self) -> None:
        def _add_paths(repo_root: Path) -> bool:
            anygpt_root = repo_root / "AnyGPT"
            anygpt_src = anygpt_root / "anygpt" / "src"
            if not anygpt_root.exists():
                return False
            if str(anygpt_root) not in sys.path:
                sys.path.insert(0, str(anygpt_root))
            if anygpt_src.exists():
                if str(anygpt_src) not in sys.path:
                    sys.path.insert(0, str(anygpt_src))
                infer_dir = anygpt_src / "infer"
                if infer_dir.exists() and str(infer_dir) not in sys.path:
                    sys.path.insert(0, str(infer_dir))
            self._root_dir = repo_root
            self._anygpt_src = anygpt_src if anygpt_src.exists() else None
            return True

        candidates = []
        env_root = _expand(os.environ.get("ANYGPT_ROOT"))
        if env_root:
            candidates.append(env_root)
        candidates.extend([
            Path("/workspace"),
            Path(__file__).resolve().parents[2],
        ])

        for root in candidates:
            if _add_paths(root):
                return

        raise ImportError(
            "Unable to locate the AnyGPT checkout. Expected under /workspace/AnyGPT "
            "or AnyGPT_ROOT."
        )

    def _resolve_speech_modules(self, cache_dir: Optional[Path]) -> tuple[Path, Path, Path]:
        # Env overrides
        ckpt = _expand(os.environ.get("ANYGPT_SPEECH_TOKENIZER_PATH"))
        cfg = _expand(os.environ.get("ANYGPT_SPEECH_TOKENIZER_CONFIG"))
        sstorm = _expand(os.environ.get("ANYGPT_SOUNDSTORM_PATH"))

        if ckpt and cfg and sstorm:
            return ckpt, cfg, sstorm

        # Heuristic search under cache dir for AnyGPT-speech-modules
        candidates = []
        if cache_dir:
            base = cache_dir / "fnlp" / "AnyGPT-speech-modules"
            alt = cache_dir / "AnyGPT-speech-modules"
            for d in (base, alt):
                if d.is_dir():
                    candidates.append(d)

        # Also try a local models folder under repo
        if self._root_dir:
            models_root = self._root_dir / "models"
            for d in (
                models_root / "fnlp" / "AnyGPT-speech-modules",
                models_root / "AnyGPT-speech-modules",
                models_root / "soundstorm",
                models_root / "speechtokenizer",
            ):
                if d.exists():
                    candidates.append(d)

        ckpt_path: Optional[Path] = None
        cfg_path: Optional[Path] = None
        sstorm_path: Optional[Path] = None

        for base in candidates:
            if ckpt_path is None:
                maybe = list(base.glob("**/ckpt.dev"))
                if maybe:
                    ckpt_path = maybe[0]
            if cfg_path is None:
                maybe = list(base.glob("**/config.json"))
                if maybe:
                    cfg_path = maybe[0]
            if sstorm_path is None:
                maybe = list(base.glob("**/speechtokenizer_soundstorm_mls.pt"))
                if maybe:
                    sstorm_path = maybe[0]
            if ckpt_path and cfg_path and sstorm_path:
                break

        if not (ckpt_path and cfg_path and sstorm_path):
            raise FileNotFoundError(
                "Missing AnyGPT speech modules. Provide env vars ANYGPT_SPEECH_TOKENIZER_PATH, "
                "ANYGPT_SPEECH_TOKENIZER_CONFIG, ANYGPT_SOUNDSTORM_PATH, or download "
                "fnlp/AnyGPT-speech-modules into your cache_dir or models/."
            )
        return ckpt_path, cfg_path, sstorm_path

    # ----- load & inference ----------------------------------------------

    def _load(self) -> None:
        self._ensure_repo_paths()
        self._cache_dir = self._resolve_cache_dir()

        try:
            _maybe_install_torchvision_stub()
            # Import after sys.path is adjusted
            from infer.cli_infer_chat_model import AnyGPTChatInference  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                f"Error: {exc}"
            ) from exc


        # Resolve speech modules (ckpt + config + soundstorm)
        ckpt_path, cfg_path, sstorm_path = self._resolve_speech_modules(self._cache_dir)

        # Build output directory under /tmp to avoid polluting the repo
        out_dir = Path(os.environ.get("ANYGPT_OUTPUT_DIR", "/tmp/anygpt_chat_infer")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_config_symlink(out_dir)

        # image_tokenizer not needed for audio-only flows; pass None
        image_tokenizer_path = None

        # Let transformers fetch the chat weights by HF id or local if provided
        model_name_or_path = str(self.model_id)

        # Instantiate the chat inference object
        try:
            self._chat = AnyGPTChatInference(
                model_name_or_path=model_name_or_path,
                image_tokenizer_path=image_tokenizer_path,
                output_dir=str(out_dir),
                speech_tokenizer_path=str(ckpt_path),
                speech_tokenizer_config=str(cfg_path),
                soundstorm_path=str(sstorm_path),
            )
            try:
                from infer.cli_infer_chat_model import conversation as chat_conversation  # type: ignore
            except Exception:
                chat_conversation = None
            self._conversation = chat_conversation
        except ImportError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to initialize AnyGPT chat inference: {exc.__class__.__name__}: {exc}"
            ) from exc

    def _ensure_config_symlink(self, out_dir: Path) -> None:
        if not self._root_dir:
            return
        config_dir = self._root_dir / "AnyGPT" / "config"
        if not config_dir.is_dir():
            return
        target_dir = out_dir / "config"
        if target_dir.exists():
            if target_dir.is_symlink() or target_dir.is_dir():
                return
            try:
                target_dir.unlink()
            except OSError:
                return
        try:
            os.symlink(config_dir, target_dir, target_is_directory=True)
        except OSError:
            pass

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        if sample.modality != "audio":
            raise ValueError(f"AnyGPTAudioWrapper only supports audio modality. Received: {sample.modality}")
        if not self._chat:
            raise RuntimeError("AnyGPT chat inference is not initialized.")

        # Prepare instruction text
        base_prompt = strip_modality_tags(sample.prompt or "").strip()
        if self.user_prefix:
            base_prompt = f"{self.user_prefix.strip()}\n\n{base_prompt}" if base_prompt else self.user_prefix.strip()
        if self.system_hint:
            base_prompt = f"{self.system_hint.strip()}\n\n{base_prompt}" if base_prompt else self.system_hint.strip()
        if not base_prompt:
            base_prompt = "Please describe the audio."

        speech_path = Path(sample.fake_path).resolve()
        try:
            from infer.cli_infer_chat_model import conversation

            if self._conversation is not None:
                conversation = self._conversation
            if conversation is not None:
                conversation.messages = []

            instruction_text = (
                f"{base_prompt}\n\n"
            #    "Analyse the audio evidence and respond in plain text only. "
            #    "Do not restate the instructions or output modality tokens."
            )

            prompt_seq = self._chat.preprocess(
                "interleaved",
                instruction_text,
                image_files=[],
                speech_files=[str(speech_path)],
                music_files=[],
            )

            tokenizer = self._chat.tokenizer
            device = self._chat.device
            inputs = tokenizer(prompt_seq, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            config_dir = (
                (self._root_dir / "AnyGPT" / "config") if self._root_dir else Path(__file__).resolve().parents[2] / "AnyGPT" / "config"
            )
            text_config = config_dir / "text_generate_config.json"
            if not text_config.exists():
                text_config = config_dir / "generate_config.json"
            config_dict = json.loads(text_config.read_text(encoding="utf-8"))
            gen_config = GenerationConfig(**config_dict)

            eos_ids = [tokenizer.eos_token_id]
            for tok in ("<eosp>", "<eoim>", "<eomu>", "<eoau>"):
                tok_id = tokenizer.convert_tokens_to_ids(tok)
                if tok_id is not None and tok_id >= 0:
                    eos_ids.append(tok_id)
            eos_ids = [eid for eid in eos_ids if eid is not None]
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

            generated = self._chat.model.generate(
                **inputs,
                generation_config=gen_config,
                max_new_tokens=config_dict.get("max_new_tokens", 512),
                eos_token_id=list(dict.fromkeys(eos_ids)),
                pad_token_id=pad_id,
            )
            full_tokens = generated[0]
            answer = tokenizer.decode(full_tokens, skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001
            return f"[ERROR] {exc.__class__.__name__}: {exc}"

        answer = str(answer)
        answer = answer.replace("[MMGPT]", "")
        answer = answer.replace("<eom>", "").replace("<eos>", "")
        answer = re.sub(r"<[A-Za-z0-9🗣️🎶👀]+>", " ", answer)
        answer = answer.replace("<sosp>", "").replace("<eosp>", "")
        answer = answer.replace("<soim>", "").replace("<eoim>", "")
        answer = answer.replace("<somu>", "").replace("<eomu>", "")
        answer = answer.replace("<soau>", "").replace("<eoau>", "")
        matches = list(re.finditer(r"Likely (?:Authentic|Manipulated)\.", answer))
        if matches:
            answer = answer[matches[-1].start():]
        answer = re.sub(r"\s+", " ", answer).strip()
        if 'Likely Authentic."' in answer or "Likely Manipulated.'" in answer:
            answer = (
                "Likely Manipulated.\n"
                "Artifact Findings\n"
                "- Analysis Failure: The AnyGPT chat model repeated the instructions instead of providing findings."
            )
        elif not answer.startswith("Likely Authentic.") and not answer.startswith("Likely Manipulated."):
            answer = (
                "Likely Manipulated.\n"
                "Artifact Findings\n"
                "- Analysis Failure: The AnyGPT chat model did not return a usable response."
            )
        elif "Artifact Findings" not in answer:
            answer += "\nArtifact Findings\n- Analysis Failure: The AnyGPT chat model did not return detailed findings."
        return answer


__all__ = ["AnyGPTAudioWrapper"]
