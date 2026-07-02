from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type

from .base import BaseModelWrapper
from .gemini import GeminiWrapper
from .internvl import InternVLWrapper
from .idefics2 import Idefics2Wrapper
from .llava import Llava15Wrapper, LlavaOneVisionWrapper
from .phi import Phi35VisionWrapper, Phi4MultimodalWrapper
from .minicpm import MiniCPMV26Wrapper
from .mimo import MiMoVLWrapper
from .mplug import MplugOwl3Wrapper
from .openai import OpenAIGPTBatchWrapper
from .claude import ClaudeBatchWrapper
from .qwen import Qwen3OmniWrapper, Qwen3VLWrapper, Qwen3TextWrapper, Qwen2AudioWrapper
from .vila import VILAWrapper
from .internlm_xcomposer import InternLMXComposer2D5Wrapper
from .salmonn import SalmonnWrapper
from .anygpt import AnyGPTAudioWrapper
from .audio_flamingo import AudioFlamingoWrapper


@dataclass
class WrapperRegistry:
    exact: Dict[str, Type[BaseModelWrapper]] = field(default_factory=dict)
    aliases: Dict[str, Type[BaseModelWrapper]] = field(default_factory=dict)
    keyword_hints: List[Tuple[str, Type[BaseModelWrapper]]] = field(default_factory=list)

    def register(
        self,
        identifier: str,
        cls: Type[BaseModelWrapper],
        *,
        aliases: Sequence[str] = (),
        keywords: Sequence[str] = (),
    ) -> None:
        self.exact[identifier] = cls
        for alias in aliases:
            self.aliases[alias.lower()] = cls
        hinted = set(k.lower() for k in keywords)
        hinted.add(identifier.split("/")[-1].lower())
        for hint in hinted:
            self.keyword_hints.append((hint, cls))

    def match(self, model_id: str) -> Optional[Type[BaseModelWrapper]]:
        if model_id in self.exact:
            return self.exact[model_id]

        base = os.path.basename(model_id.rstrip("/"))
        if base in self.exact:
            return self.exact[base]

        lower_base = base.lower()
        if lower_base in self.aliases:
            return self.aliases[lower_base]

        for hint, cls in self.keyword_hints:
            if hint in lower_base or lower_base in hint:
                return cls
        return None

    def available(self) -> Iterable[str]:
        return self.exact.keys()


REGISTRY = WrapperRegistry()

REGISTRY.register(
    "llava-hf/llava-1.5-7b-hf",
    Llava15Wrapper,
    aliases=["llava-1.5-7b-hf", "llava-v1.5-7b", "llava-v1.5-7b-hf", "liuhaotian/llava-v1.5-7b"],
    keywords=["llava-1.5", "v1.5"],
)
REGISTRY.register(
    "llava-hf/llava-1.5-13b-hf",
    Llava15Wrapper,
    aliases=["llava-1.5-13b-hf", "llava-v1.5-13b", "llava-v1.5-13b-hf", "liuhaotian/llava-v1.5-13b"],
    keywords=["llava-1.5", "v1.5"],
)
REGISTRY.register(
    "llava-hf/llava-onevision-qwen2-72b-ov-hf",
    LlavaOneVisionWrapper,
    aliases=["llava-onevision-qwen2-72b-ov-hf"],
    keywords=["llava", "onevision"],
)
REGISTRY.register(
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    LlavaOneVisionWrapper,
    aliases=["llava-onevision-qwen2-7b-ov-hf"],
    keywords=["llava", "onevision", "7b"],
)
REGISTRY.register(
    "OpenGVLab/InternVL2-40B",
    InternVLWrapper,
    aliases=["InternVL2-40B", "InternVL2_5-38B", "InternVL2_5-78B"],
    keywords=["internvl"],
)
REGISTRY.register(
    "OpenGVLab/InternVL2_5-38B",
    InternVLWrapper,
    keywords=["internvl"],
)
REGISTRY.register(
    "OpenGVLab/InternVL2_5-78B",
    InternVLWrapper,
    keywords=["internvl"],
)
REGISTRY.register(
    "OpenGVLab/InternVL2_5-8B",
    InternVLWrapper,
    aliases=["InternVL2_5-8B"],
    keywords=["internvl"],
)
REGISTRY.register(
    "OpenGVLab/InternVL2_5-26B",
    InternVLWrapper,
    aliases=["InternVL2_5-26B"],
    keywords=["internvl"],
)
REGISTRY.register(
    "OpenGVLab/InternVL3_5-8B",
    InternVLWrapper,
    aliases=["InternVL3_5-8B"],
    keywords=["internvl", "internvl3", "internvl3.5"],
)
REGISTRY.register(
    "OpenGVLab/InternVL3_5-38B",
    InternVLWrapper,
    aliases=["InternVL3_5-38B"],
    keywords=["internvl", "internvl3", "internvl3.5"],
)
REGISTRY.register(
    "Efficient-Large-Model/VILA1.5-40b",
    VILAWrapper,
    aliases=["VILA1.5-40b"],
    keywords=["vila"],
)
REGISTRY.register(
    "openbmb/MiniCPM-V-2_6",
    MiniCPMV26Wrapper,
    aliases=["MiniCPM-V-2_6"],
    keywords=["minicpm", "cpm"],
)
REGISTRY.register(
    "internlm/internlm-xcomposer2d5-7b",
    InternLMXComposer2D5Wrapper,
    aliases=["internlm-xcomposer2d5-7b"],
    keywords=["internlm", "xcomposer", "xcomposer2.5"],
)
REGISTRY.register(
    "microsoft/Phi-3.5-vision-instruct",
    Phi35VisionWrapper,
    aliases=["Phi-3.5-vision-instruct"],
    keywords=["phi3.5", "phi-3.5", "phi"],
)
REGISTRY.register(
    "microsoft/Phi-4-multimodal-instruct",
    Phi4MultimodalWrapper,
    aliases=["Phi-4-multimodal-instruct"],
    keywords=["phi4", "phi-4", "phi"],
)
REGISTRY.register(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    Qwen3OmniWrapper,
    aliases=["Qwen3-Omni-30B-A3B-Instruct"],
    keywords=["qwen3-omni", "qwen-omni"],
)
REGISTRY.register(
    "Qwen/Qwen3-8B",
    Qwen3TextWrapper,
    aliases=["Qwen3-8B", "Qwen3-8B-Instruct", "Qwen/Qwen3-8B-Instruct"],
    keywords=["qwen3-8b", "qwen3", "qwen-8b"],
)
REGISTRY.register(
    "Qwen/Qwen3-VL-8B-Instruct",
    Qwen3VLWrapper,
    aliases=["Qwen3-VL-8B-Instruct"],
    keywords=["qwen3-vl", "qwen3vl", "qwen-vl"],
)
REGISTRY.register(
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    Qwen3VLWrapper,
    aliases=["Qwen3-VL-30B-A3B-Instruct"],
    keywords=["qwen3-vl", "qwen3vl", "qwen-vl", "30b", "a3b"],
)
REGISTRY.register(
    "Qwen/Qwen2-Audio-7B-Instruct",
    Qwen2AudioWrapper,
    aliases=["Qwen2-Audio-7B-Instruct"],
    keywords=["qwen2-audio", "qwen2audio", "audio"],
)
REGISTRY.register(
    "gemini-2.5-flash",
    GeminiWrapper,
    aliases=["gemini-2.5-flash-latest"],
    keywords=["gemini", "flash"],
)
REGISTRY.register(
    "gemini-2.5-flash-lite",
    GeminiWrapper,
    aliases=["gemini-2.5-flash-lite-latest"],
    keywords=["gemini", "flash-lite", "lite"],
)
REGISTRY.register(
    "gemini-2.5-pro",
    GeminiWrapper,
    keywords=["gemini", "pro"],
)
REGISTRY.register(
    "claude-sonnet-4-5",
    ClaudeBatchWrapper,
    aliases=[
        "claude-sonnet-4-5-latest",
        "claude-sonnet-4-5-20250929",
    ],
    keywords=["claude", "sonnet", "4.5", "anthropic"],
)
REGISTRY.register(
    "mPLUG/mPLUG-Owl3-7B-240728",
    MplugOwl3Wrapper,
    aliases=["mPLUG-Owl3-7B-240728"],
    keywords=["mplug", "owl3", "owl-3"],
)
REGISTRY.register(
    "HuggingFaceM4/idefics2-8b",
    Idefics2Wrapper,
    aliases=["idefics2-8b"],
    keywords=["idefics2", "idefics"],
)
REGISTRY.register(
    "XiaomiMiMo/MiMo-VL-7B-SFT",
    MiMoVLWrapper,
    aliases=["MiMo-VL-7B-SFT"],
    keywords=["mimo", "vl-7b", "qwen2.5"],
)
REGISTRY.register(
    "tsinghua-ee/SALMONN",
    SalmonnWrapper,
    aliases=["SALMONN"],
    keywords=["salmonn"],
)
REGISTRY.register(
    "fnlp/AnyGPT-chat",
    AnyGPTAudioWrapper,
    aliases=["AnyGPT-chat"],
    keywords=["anygpt", "audio"],
)
REGISTRY.register(
    "nvidia/audio-flamingo-3",
    AudioFlamingoWrapper,
    aliases=["audio-flamingo-3"],
    keywords=["flamingo", "audio"],
)
REGISTRY.register(
    "gpt-5.0",
    OpenAIGPTBatchWrapper,
    keywords=["gpt5", "openai", "gpt-5"],
)
REGISTRY.register(
    "gpt-5.0-mini",
    OpenAIGPTBatchWrapper,
    keywords=["gpt5", "mini", "openai"],
)
REGISTRY.register(
    "gpt-5-mini",
    OpenAIGPTBatchWrapper,
    keywords=["gpt5", "mini", "openai"],
)
REGISTRY.register(
    "gpt-5.0-pro",
    OpenAIGPTBatchWrapper,
    keywords=["gpt5", "pro", "openai"],
)


def match_wrapper_cls(model_id: str) -> Optional[Type[BaseModelWrapper]]:
    return REGISTRY.match(model_id)


def available_wrappers() -> Iterable[str]:
    return REGISTRY.available()


def build_model_wrapper(
    model_id: str,
    *,
    max_new_tokens: int = 512,
    torch_dtype = None,
    load_kwargs: Optional[Dict[str, object]] = None,
    video_max_frames: Optional[int] = None,
) -> BaseModelWrapper:
    cls = match_wrapper_cls(model_id)
    if cls is None:
        available = ', '.join(sorted(available_wrappers()))
        raise KeyError(f"Model {model_id} is not registered. Available: {available}")
    kwargs: Dict[str, object] = dict(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch_dtype,
        load_kwargs=load_kwargs,
    )
    # 只有明確指定 video_max_frames 時才傳遞，否則使用 wrapper 的預設值
    if video_max_frames is not None:
        import inspect
        sig = inspect.signature(cls.__init__)
        if 'video_max_frames' in sig.parameters:
            kwargs['video_max_frames'] = video_max_frames
    return cls(**kwargs)
