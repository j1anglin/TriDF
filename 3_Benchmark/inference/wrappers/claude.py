from __future__ import annotations

import base64
import mimetypes
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from inference.dataio.samples import TaskSample
from inference.utils.media import load_video_frames_raw, require_video_support
from inference.utils.text import strip_modality_tags

from .base import BaseModelWrapper, ModelCapabilities


class ClaudeBatchWrapper(BaseModelWrapper):
    """Prepares request payloads for the Claude Messages Batch API."""

    capabilities = ModelCapabilities(supports_image=True, supports_video=True)

    max_image_side: int = 1568
    max_image_pixels: int = 1_150_000
    max_image_bytes: int = 5 * 1024 * 1024
    direct_image_margin_bytes: int = 256 * 1024  # keep ~250 KB headroom before sending raw bytes
    video_frame_max_side: int = 1024
    video_frame_max_pixels: int = 900_000

    def __init__(self, *args, video_frame_count: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._api_key: Optional[str] = None
        self.video_frame_count = max(1, video_frame_count)

    def _load(self) -> None:
        if self._api_key:
            return
        api_key = (
            os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("CLAUDE_API_KEY")
            or os.getenv("CLAUDE_OPUS_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY (or CLAUDE_API_KEY / CLAUDE_OPUS_API_KEY) must be set for Claude batch inference"
            )
        self._api_key = api_key

    def prepare_batch_request(self, sample: TaskSample) -> Dict[str, object]:
        self.ensure_loaded()

        user_text = strip_modality_tags(sample.prompt or "")
        if self.user_prefix:
            user_text = f"{self.user_prefix.strip()}\n\n{user_text}".strip()

        user_parts: List[Dict[str, object]] = []
        modality = (sample.modality or "").lower()
        if modality in {"image", "img"}:
            user_parts.append(self._encode_image(sample))
        elif modality == "video":
            user_parts.extend(self._encode_video_frames(sample))
        elif modality == "audio":
            raise ValueError("Claude batch currently does not support modality 'audio'.")

        if user_text:
            user_parts.append({"type": "text", "text": user_text})
        elif not user_parts:
            # Claude requires at least some content in the user message.
            user_parts.append({"type": "text", "text": " "})

        messages = [
            {
                "role": "user",
                "content": user_parts,
            }
        ]

        params: Dict[str, object] = {
            "model": self.model_id,
            "max_tokens": self.max_new_tokens,
            "messages": messages,
        }
        if self.system_hint:
            params["system"] = self.system_hint
        if self.generation_overrides:
            params.update(self.generation_overrides)

        return {
            "custom_id": str(sample.sample_id),
            "params": params,
        }

    def _encode_image(self, sample: TaskSample) -> Dict[str, object]:
        media_path = Path(sample.fake_path)
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")
        with Image.open(media_path) as pil_image:
            detected_format = pil_image.format
            image = pil_image.convert("RGB")
        data: Optional[str] = None
        mime, _ = mimetypes.guess_type(str(media_path))
        detected_mime = self._mime_from_format(detected_format) or mime
        raw_bytes: Optional[bytes] = None
        try:
            raw_bytes = media_path.read_bytes()
        except Exception:
            raw_bytes = None
        media_type = "image/jpeg"
        if raw_bytes is not None and self._can_send_raw_image(len(raw_bytes)):
            encoded_raw = base64.b64encode(raw_bytes).decode("utf-8")
            if len(encoded_raw) <= self.max_image_bytes:
                data = encoded_raw
                media_type = detected_mime or "application/octet-stream"
            else:
                data = self._encode_image_bytes(image)
                media_type = "image/jpeg"
        else:
            data = self._encode_image_bytes(image)
            media_type = "image/jpeg"
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }

    def _encode_video_frames(self, sample: TaskSample) -> List[Dict[str, object]]:
        require_video_support()
        frames = load_video_frames_raw(
            Path(sample.fake_path),
            num_segments=self.video_frame_count,
            max_frames=self.video_frame_count,
        )
        encoded: List[Dict[str, object]] = []
        for frame in frames:
            prepared = self._constrain_video_frame(frame)
            data = self._encode_image_bytes(prepared)
            encoded.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": data,
                    },
                }
            )
        if not encoded:
            raise ValueError(f"Failed to sample frames from video: {sample.fake_path}")
        return encoded

    def _encode_image_bytes(self, image: Image.Image) -> str:
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

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

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

    def _can_send_raw_image(self, byte_len: int) -> bool:
        if byte_len > self.max_image_bytes:
            return False
        soft_limit = max(1, self.max_image_bytes - self.direct_image_margin_bytes)
        return byte_len <= soft_limit

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

    @property
    def api_key(self) -> str:
        self.ensure_loaded()
        assert self._api_key is not None
        return self._api_key
