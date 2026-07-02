from __future__ import annotations

import base64
import mimetypes
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from inference.dataio.samples import TaskSample
from inference.utils.media import load_video_frames_raw, require_video_support
from inference.utils.text import strip_modality_tags

from .base import BaseModelWrapper, ModelCapabilities


class OpenAIGPTBatchWrapper(BaseModelWrapper):
    """Builds Responses API batch payloads for GPT-5 family models."""

    capabilities = ModelCapabilities(supports_image=True, supports_video=True, supports_audio=True)
    max_image_side: int = 1568
    max_image_pixels: int = 1_150_000
    max_image_bytes: int = 5 * 1024 * 1024
    direct_image_margin_bytes: int = 256 * 1024
    video_frame_max_side: int = 1024
    video_frame_max_pixels: int = 900_000

    def __init__(self, *args: Any, video_frame_count: int = 16, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._api_key: Optional[str] = None
        self.video_frame_count = max(1, video_frame_count)

    def _load(self) -> None:
        if self._api_key:
            return
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY_GPT5")
            or os.getenv("OPENAI_API_KEY_BATCH")
        )
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY (or OPENAI_API_KEY_GPT5 / OPENAI_API_KEY_BATCH) must be set for GPT batch inference"
            )
        self._api_key = api_key

    def prepare_batch_request(self, sample: TaskSample) -> Dict[str, Any]:
        self.ensure_loaded()

        input_messages: List[Dict[str, Any]] = []
        system_hint = (self.system_hint or "").strip()
        if system_hint:
            input_messages.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_hint}],
                }
            )

        user_text = strip_modality_tags(sample.prompt or "")
        if self.user_prefix:
            user_text = f"{self.user_prefix.strip()}\n\n{user_text}".strip()

        user_content: List[Dict[str, Any]] = []
        media_parts = self._load_media_parts(sample)
        if media_parts:
            user_content.extend(media_parts)
        if user_text:
            user_content.append({"type": "input_text", "text": user_text})

        if not user_content:
            raise ValueError(f"Sample {sample.sample_id} did not produce any user content")

        input_messages.append({"role": "user", "content": user_content})

        model_id = (self.model_id or "").strip()
        if model_id in {"gpt-5.0", "gpt5"}:
            model_id = "gpt-5"
        elif model_id.startswith("gpt-5.0-"):
            model_id = model_id.replace("gpt-5.0-", "gpt-5-", 1)

        body: Dict[str, Any] = {
            "model": model_id or "gpt-5",
            "input": input_messages,
            "max_output_tokens": self.max_new_tokens,
        }
        if self.generation_overrides:
            body.update(self.generation_overrides)

        return {
            "custom_id": str(sample.sample_id),
            "method": "POST",       
            "url": "/v1/responses",
            "body": body,
        }

    def _load_media_parts(self, sample: TaskSample) -> List[Dict[str, Any]]:
        modality = (sample.modality or "").lower()
        if modality not in {"image", "video", "audio"}:
            return []
        if modality == "video":
            return self._encode_video(sample)

        media_path = Path(sample.fake_path)
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")
        if modality == "audio":
            data = base64.b64encode(media_path.read_bytes()).decode("utf-8")
            mime = mimetypes.guess_type(str(media_path))[0] or "audio/wav"
            data_url = self._to_data_url(data, mime)
            return [{"type": "input_audio", "audio_url": data_url}]

        with Image.open(media_path) as pil_image:
            detected_format = pil_image.format
            image = pil_image.convert("RGB")
        mime_guess = mimetypes.guess_type(str(media_path))[0]
        detected_mime = self._mime_from_format(detected_format) or mime_guess
        raw_bytes: Optional[bytes] = None
        try:
            raw_bytes = media_path.read_bytes()
        except Exception:
            raw_bytes = None
        data_url: Optional[str] = None
        if raw_bytes is not None and self._can_send_raw_image(len(raw_bytes)) and detected_mime is not None:
            encoded_raw = base64.b64encode(raw_bytes).decode("utf-8")
            if len(encoded_raw) <= self.max_image_bytes:
                data_url = self._to_data_url(encoded_raw, detected_mime)
            else:
                data_url = self._image_to_data_url(image)
        else:
            data_url = self._image_to_data_url(image)
        return [{"type": "input_image", "image_url": data_url}]

    def _encode_video(self, sample: TaskSample) -> List[Dict[str, Any]]:
        require_video_support()
        frames = load_video_frames_raw(
            Path(sample.fake_path),
            num_segments=self.video_frame_count,
            max_frames=self.video_frame_count,
        )
        encoded: List[Dict[str, Any]] = []
        for frame in frames:
            prepared = self._constrain_video_frame(frame)
            data_url = self._image_to_data_url(prepared)
            encoded.append({"type": "input_image", "image_url": data_url})
        if not encoded:
            raise ValueError(f"Failed to sample frames from video: {sample.fake_path}")
        return encoded

    @staticmethod
    def _to_data_url(b64_data: str, mime: str) -> str:
        return f"data:{mime};base64,{b64_data}"

    def _image_to_data_url(self, image: Image.Image) -> str:
        working = self._constrain_image(image.copy())
        buffer = BytesIO()
        qualities = (95, 90, 80, 70, 60)

        def encode_current() -> int:
            for quality in qualities:
                buffer.seek(0)
                buffer.truncate()
                working.save(buffer, format="JPEG", quality=quality, optimize=True)
                if buffer.tell() <= self.max_image_bytes or quality == qualities[-1]:
                    break
            return buffer.tell()

        current_size = encode_current()
        while current_size > self.max_image_bytes and min(working.size) > 1:
            scale = min(
                0.95,
                (self.max_image_bytes / max(current_size, 1)) ** 0.5,
            )
            if scale >= 1.0:
                scale = 0.9
            new_size = (
                max(1, int(working.size[0] * scale)),
                max(1, int(working.size[1] * scale)),
            )
            if new_size == working.size:
                if working.size[0] > 1 and working.size[1] > 1:
                    new_size = (working.size[0] - 1, working.size[1] - 1)
                else:
                    break
            working = working.resize(new_size, Image.LANCZOS)
            working = self._constrain_image(working)
            current_size = encode_current()

        data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return self._to_data_url(data, "image/jpeg")

    def _constrain_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = 1.0
        max_side = max(width, height)
        if max_side > self.max_image_side:
            scale = min(scale, self.max_image_side / max_side)
        pixels = width * height
        if pixels > self.max_image_pixels:
            scale = min(scale, (self.max_image_pixels / pixels) ** 0.5)
        if scale < 1.0:
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(new_size, Image.LANCZOS)
        return image

    def _constrain_video_frame(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = 1.0
        max_side = max(width, height)
        if max_side > self.video_frame_max_side:
            scale = min(scale, self.video_frame_max_side / max_side)
        pixels = width * height
        if pixels > self.video_frame_max_pixels:
            scale = min(scale, (self.video_frame_max_pixels / pixels) ** 0.5)
        if scale < 1.0:
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(new_size, Image.LANCZOS)
        return image

    def _can_send_raw_image(self, byte_len: int) -> bool:
        if byte_len > self.max_image_bytes:
            return False
        soft_limit = max(1, self.max_image_bytes - self.direct_image_margin_bytes)
        return byte_len <= soft_limit

    @staticmethod
    def _mime_from_format(image_format: Optional[str]) -> Optional[str]:
        if not image_format:
            return None
        fmt = image_format.lower()
        if fmt in {"jpeg", "jpg"}:
            return "image/jpeg"
        if fmt == "png":
            return "image/png"
        if fmt == "webp":
            return "image/webp"
        if fmt == "bmp":
            return "image/bmp"
        if fmt == "gif":
            return "image/gif"
        return None
