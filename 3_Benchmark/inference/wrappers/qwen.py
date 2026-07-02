from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
import numpy as np
import torch

from inference.dataio.samples import TaskSample
from inference.utils.text import strip_modality_tags
from inference.utils.torch_helpers import apply_dtype_kw, load_auto_processor

from .base import BaseModelWrapper, ModelCapabilities

def _resample_waveform(
    waveform: torch.Tensor,
    source_sr: int,
    target_sr: int,
) -> torch.Tensor:
    if source_sr == target_sr:
        return waveform
    if waveform.numel() == 0:
        return waveform

    try:
        import torchaudio.functional as F  # type: ignore

        resampled = F.resample(waveform.unsqueeze(0), source_sr, target_sr)
        return resampled.squeeze(0)
    except Exception as torchaudio_exc:  # noqa: BLE001
        try:
            import librosa  # type: ignore
        except ImportError:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Resampling audio inputs requires either `torchaudio` or `librosa`. "
                "Install one of them to evaluate Qwen2 Audio models."
            ) from torchaudio_exc

        resampled_np = librosa.resample(
            waveform.detach().cpu().numpy(),
            orig_sr=source_sr,
            target_sr=target_sr,
        )
        return torch.from_numpy(resampled_np.astype(np.float32))


def _load_audio_waveform(path: Path, target_sr: int) -> np.ndarray:
    try:
        import soundfile as sf  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Qwen2AudioWrapper requires the optional package `soundfile`. "
            "Install it (e.g., `pip install soundfile`) to enable audio evaluation."
        ) from exc

    data, sample_rate = sf.read(str(path), always_2d=False)
    if isinstance(data, tuple):
        data, sample_rate = data  # type: ignore[assignment]

    waveform = torch.from_numpy(np.asarray(data, dtype=np.float32))
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=-1)
    waveform = waveform.contiguous()
    waveform = _resample_waveform(waveform, int(sample_rate), int(target_sr))
    return waveform.cpu().numpy()


def _infer_device(model: torch.nn.Module) -> torch.device:
    """
    Best-effort device detection compatible with both single-device and device_map=auto loads.
    Falls back to CPU if no parameters are available.
    """
    if hasattr(model, "device"):
        try:
            return torch.device(model.device)
        except Exception:
            pass
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _truncate_video_payload(payload: Any, max_frames: Optional[int]) -> Any:
    """Respect manual max frame limits without altering processor defaults otherwise."""
    if max_frames is None:
        return payload
    if isinstance(payload, torch.Tensor):
        if payload.ndim >= 4 and payload.shape[0] > max_frames:
            return payload[:max_frames]
        if payload.ndim == 5 and payload.shape[1] > max_frames:
            return payload[:, :max_frames]
        return payload
    if isinstance(payload, list) and len(payload) > max_frames:
        return payload[:max_frames]
    return payload


class Qwen3OmniWrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=True, supports_video=True, supports_audio=True)

    def __init__(self, *args, video_max_frames: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_max_frames = video_max_frames
        self.processor = None
        self.model = None

    def _load(self) -> None:
        from inference.utils.transformers_compat import patch_qwen3_omni_moe_talker_code_predictor_config

        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        patch_qwen3_omni_moe_talker_code_predictor_config()

        cache_dir = self._load_kwargs.get("cache_dir")
        kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(self.model_id, **kwargs).eval()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)

        try:
            self.model.disable_talker()
        except Exception:
            pass

        # --- pad_token_id fix ---
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            pad_id = tok.pad_token_id or tok.eos_token_id
        else:
            pad_id = getattr(self.model.config, "pad_token_id", None) or getattr(self.model.config, "eos_token_id", None)
        self.model.config.pad_token_id = pad_id
        self.model.generation_config.pad_token_id = pad_id
        self.pad_token_id = pad_id
        self.num_frames = self.video_max_frames
        self.use_audio_in_video = False

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()

        content: List[Dict[str, Any]] = []
        if sample.modality == "image":
            content.append({"type": "image", "image": sample.fake_path})
        elif sample.modality == "video":
            content.append({"type": "video", "video": sample.fake_path})
        elif sample.modality == "audio":
            content.append({"type": "audio", "audio": sample.fake_path})
        else:
            raise ValueError(f"Unsupported modality for Qwen Omni wrapper: {sample.modality}")

        user_text = f"{sample.prompt}\n"
        if self.user_prefix:
            user_text = self.user_prefix + "\n\n" + user_text
        content.append({"type": "text", "text": user_text})

        messages: List[Dict[str, Any]] = []
        if self.system_hint:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_hint}]})
        messages.append({"role": "user", "content": content})

        conversations = [messages]

        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        try:
            from qwen_omni_utils import process_mm_info
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Qwen3OmniWrapper requires the optional package `qwen_omni_utils` to pack multimedia inputs. "
                "Install it (e.g., `pip install qwen-omni-utils`) before evaluating Qwen Omni models."
            ) from exc

        audios, images, videos = process_mm_info(
            conversations, use_audio_in_video=self.use_audio_in_video
        )

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        gen_args = dict(
            return_audio=False,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=self.use_audio_in_video,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.pad_token_id, 
        )
        gen_args.update(self.generation_overrides or {})
        raw_output = self.model.generate(**inputs, **gen_args)
        if isinstance(raw_output, (tuple, list)):
            text_ids = raw_output[0]
        else:
            text_ids = raw_output

        if isinstance(text_ids, str):
            # Some transformer builds may already return decoded text
            return text_ids.strip()

        sequences = None
        if hasattr(text_ids, "sequences"):
            sequences = text_ids.sequences
        elif isinstance(text_ids, torch.Tensor):
            sequences = text_ids
        else:
            raise TypeError(f"Unsupported generate() output type: {type(text_ids).__name__}")

        start = inputs["input_ids"].shape[1]
        response = self.processor.batch_decode(
            sequences[:, start:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()


class Qwen3VLWrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=True, supports_video=True)

    def __init__(
        self, 
        *args: Any, 
        video_max_frames: int = 16, 
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.video_max_frames = video_max_frames
        self.processor = None
        self.model = None

    def _load(self) -> None:
        from transformers import AutoConfig
        
        try:
            from transformers import AutoModelForImageTextToText as FallbackModel
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as FallbackModel
            except ImportError:
                from transformers import AutoModelForCausalLM as FallbackModel

        load_kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        load_kwargs.setdefault("device_map", "auto")
        cache_dir = load_kwargs.get("cache_dir")

        load_kwargs["attn_implementation"] = "sdpa"

        config_kwargs: Dict[str, Any] = {}
        if cache_dir:
            config_kwargs["cache_dir"] = cache_dir
            
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True, **config_kwargs)

        model_cls = None
        try:
            from transformers import Qwen3VLForConditionalGeneration
            model_cls = Qwen3VLForConditionalGeneration
        except ImportError:
            pass
            
        if model_cls is None:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                model_cls = Qwen2_5_VLForConditionalGeneration
            except ImportError:
                pass

        if model_cls is not None:
            self.model = model_cls.from_pretrained(self.model_id, trust_remote_code=True, **load_kwargs).eval()
        else:
            print(f"[INFO] Using {FallbackModel.__name__} as fallback loader...")
            self.model = FallbackModel.from_pretrained(self.model_id, trust_remote_code=True, **load_kwargs).eval()
            
        self.processor = load_auto_processor(self.model_id, trust_remote_code=True, cache_dir=cache_dir)

    def _build_messages(self, sample: TaskSample) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_hint:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": self.system_hint.strip()}]}
            )

        user_content: List[Dict[str, Any]] = []
        media_path = Path(sample.fake_path).resolve()
        uri = f"file://{media_path}"
        if sample.modality == "image":
            user_content.append({"type": "image", "image": uri})
        elif sample.modality == "video":
            user_content.append({"type": "video", "video": uri})
        else:
            raise ValueError(f"Qwen3VLWrapper only supports image/video modalities, received: {sample.modality}")

        prompt = strip_modality_tags(sample.prompt or "")
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            prompt = f"{prefix}\n\n{prompt}" if prompt else prefix
        prompt = prompt.strip()
        if not prompt:
            prompt = "Please describe the provided content."
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})
        return messages

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        assert self.processor is not None
        assert self.model is not None

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Qwen3VLWrapper requires the optional package `qwen-vl-utils` for vision preprocessing. "
                "Install it (e.g., `pip install qwen-vl-utils`) before evaluating Qwen3-VL models."
            ) from exc

        messages = self._build_messages(sample)
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        if video_inputs is not None:
            video_inputs = [_truncate_video_payload(v, self.video_max_frames) for v in video_inputs]

        model_inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        device = _infer_device(self.model)
        model_inputs = model_inputs.to(device=device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
        }

        if self.generation_overrides:
            gen_kwargs.update(self.generation_overrides)

        input_ids = model_inputs["input_ids"]

        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **gen_kwargs)

        start = input_ids.shape[-1]

        continuation = output_ids[:, start:]
            
        response = self.processor.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        del model_inputs
        del output_ids
        del continuation
        torch.cuda.empty_cache()

        return response.strip()

class Qwen3TextWrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=False, supports_video=False, supports_audio=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = None
        self.tokenizer = None

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        load_kwargs.setdefault("device_map", "auto")
        cache_dir = load_kwargs.get("cache_dir")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **load_kwargs,
        ).eval()
        tokenizer_kwargs: Dict[str, Any] = {}
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if hasattr(self.model, "config"):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _build_prompt(self, prompt: str) -> str:
        messages: List[Dict[str, Any]] = []
        if self.system_hint:
            messages.append({"role": "system", "content": self.system_hint.strip()})
        messages.append({"role": "user", "content": prompt})

        tokenizer = self.tokenizer
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        if self.system_hint:
            return f"{self.system_hint.strip()}\n\n{prompt}"
        return prompt

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        assert self.model is not None
        assert self.tokenizer is not None

        if sample.modality not in {"text", "", None}:
            raise ValueError(f"Qwen3TextWrapper only supports text modality, received: {sample.modality}")

        prompt = strip_modality_tags(sample.prompt or "").strip()
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            prompt = f"{prefix}\n\n{prompt}" if prompt else prefix
        if not prompt:
            prompt = "Please answer the user query."

        full_prompt = self._build_prompt(prompt)
        model_inputs = self.tokenizer(full_prompt, return_tensors="pt")
        device = _infer_device(self.model)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.generation_overrides:
            gen_kwargs.update(self.generation_overrides)

        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **gen_kwargs)

        input_ids = model_inputs["input_ids"]
        start = input_ids.shape[-1]
        continuation = output_ids[:, start:]
        response = self.tokenizer.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()


class Qwen2AudioWrapper(BaseModelWrapper):
    capabilities = ModelCapabilities(supports_image=False, supports_video=False, supports_audio=True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = None
        self.processor = None

    def _load(self) -> None:
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        load_kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        load_kwargs.setdefault("device_map", "auto")
        cache_dir = load_kwargs.get("cache_dir")

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_id,
            **load_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
        )

    def _resolve_device(self) -> torch.device:
        assert self.model is not None
        if hasattr(self.model, "device"):
            try:
                return torch.device(self.model.device)
            except Exception:
                pass
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()
        if sample.modality != "audio":
            raise ValueError(f"Qwen2AudioWrapper only supports audio modality. Received: {sample.modality}")
        assert self.model is not None
        assert self.processor is not None

        prompt = strip_modality_tags(sample.prompt or "").strip()
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            prompt = f"{prefix}\n\n{prompt}" if prompt else prefix
        if not prompt:
            prompt = "Please describe the audio."

        media_path = Path(sample.fake_path).resolve()
        messages: List[Dict[str, Any]] = []
        if self.system_hint:
            messages.append({"role": "system", "content": self.system_hint.strip()})
        user_content: List[Dict[str, Any]] = [
            {"type": "audio", "audio_url": f"file://{media_path}"},
            {"type": "text", "text": prompt},
        ]
        messages.append({"role": "user", "content": user_content})

        chat_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        feature_extractor = getattr(self.processor, "feature_extractor", None)
        target_sr = int(getattr(feature_extractor, "sampling_rate", 16000))
        audio_array = _load_audio_waveform(media_path, target_sr)

        inputs = self.processor(
            text=[chat_prompt],
            audio=[audio_array],
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )

        model_device = self._resolve_device()
        prepared_inputs: Dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                tensor = value.to(model_device)
                if tensor.is_floating_point() and self.model is not None:
                    model_dtype = getattr(self.model, "dtype", None)
                    if model_dtype is not None and tensor.dtype != model_dtype:
                        tensor = tensor.to(model_dtype)
                prepared_inputs[key] = tensor
            else:
                prepared_inputs[key] = value

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
        }
        if self.generation_overrides:
            gen_kwargs.update(self.generation_overrides)

        with torch.inference_mode():
            output_ids = self.model.generate(**prepared_inputs, **gen_kwargs)

        input_ids = prepared_inputs["input_ids"]
        start = input_ids.shape[-1]
        generated = output_ids[:, start:]
        response = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()


__all__ = ["Qwen3OmniWrapper", "Qwen3VLWrapper", "Qwen3TextWrapper", "Qwen2AudioWrapper"]
