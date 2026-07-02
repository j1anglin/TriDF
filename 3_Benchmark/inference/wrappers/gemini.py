from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai_legacy  # type: ignore
except ImportError:
    genai_legacy = None

try:  # pragma: no cover - optional dependency
    from google import genai as genai_modern  # type: ignore
except ImportError:
    genai_modern = None

from inference.dataio.samples import TaskSample
from inference.utils.gemini_pricing import estimate_cost_usd, usage_to_dict
from inference.utils.text import strip_modality_tags

from .base import BaseModelWrapper, ModelCapabilities


class GeminiWrapper(BaseModelWrapper):
    # Gemini 2.5 Flash supports audio alongside images/videos; advertise this so
    # inference runners do not filter out audio samples by mistake.
    capabilities = ModelCapabilities(supports_image=True, supports_video=True, supports_audio=True)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_usage: Optional[Dict[str, Any]] = None
        self._genai: Optional[Any] = None
        self._genai_mode: Optional[str] = None
        self._api_key: Optional[str] = None
        cache_env = os.getenv("GEMINI_PROMPT_CACHE", "1").strip().lower()
        self.prompt_cache_enabled = cache_env not in ("0", "false", "no", "off")
        try:
            ttl_value = int(os.getenv("GEMINI_PROMPT_CACHE_TTL", str(6 * 60 * 60)))
        except ValueError:
            ttl_value = 6 * 60 * 60
        self.prompt_cache_ttl = max(ttl_value, 60)
        self._prompt_cache_registry: Dict[str, Dict[str, Any]] = {}
        self.prompt_cache_charges: List[Dict[str, Any]] = []

    def _load(self) -> None:
        if self._genai:
            return
        if genai_legacy is None and genai_modern is None:
            raise ImportError(
                "Google Generative AI is not installed. Please install it with "
                "`pip install google-generativeai Pillow` or `pip install google-genai Pillow`"
            )
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
        )
        if not api_key:
            raise ValueError("GEMINI_API_KEY, GOOGLE_API_KEY, or GOOGLE_GENERATIVE_AI_API_KEY must be set")

        if genai_legacy is not None:
            genai_legacy.configure(api_key=api_key)
            self._genai = genai_legacy
            self._genai_mode = "legacy"
        elif genai_modern is not None:
            self._genai = genai_modern.Client(api_key=api_key)
            self._genai_mode = "modern"
        else:  # pragma: no cover - defensive, should not happen
            raise RuntimeError("No compatible Gemini client available.")

        self._api_key = api_key
        # No model initialization here, it will be done per request in batch mode

    def generate(self, sample: TaskSample) -> str:
        # This method will no longer be used for batch processing.
        # It can be kept for single-sample debugging if needed, but we'll bypass it.
        raise NotImplementedError("GeminiWrapper now operates in batch mode. Use a batch runner.")

    def prepare_batch_request(self, sample: TaskSample) -> Dict[str, Any]:
        """Prepares a single request dictionary for the Gemini Batch REST API.

        Key details for Batch REST compatibility:
        - Use `system_instruction` (not a `role: system` turn in contents).
        - `contents` should include a single user turn with optional media and text.
        - Keep `generation_config` in snake_case; the batch runner will map as needed.
        - `model` stays here for compatibility; the batch runner will strip it from body.
        """
        self.ensure_loaded()
        assert self._genai is not None

        extra_meta = getattr(sample, "extra_data", {}) or {}
        cache_template = extra_meta.get("prompt_cache_template")
        cache_ttl = extra_meta.get("prompt_cache_ttl")
        ttl_hint: Optional[int]
        if cache_ttl is None:
            ttl_hint = None
        else:
            try:
                ttl_hint = int(cache_ttl)
            except (TypeError, ValueError):
                ttl_hint = None
        analysis_override = extra_meta.get("prompt_cache_analysis_text")
        if not analysis_override:
            analysis_override = (sample.media_meta or {}).get("analysis_text")
        cached_content_name: Optional[str] = None
        cached_analysis_text: Optional[str] = None
        if cache_template and analysis_override and self.prompt_cache_enabled:
            cached_content_name = self._ensure_prompt_cache(str(cache_template), ttl_hint)
            if cached_content_name:
                cleaned = str(analysis_override).strip()
                cached_analysis_text = f"# **1. Analysis Text**\n\n{cleaned}\n\n# **Begin Evaluation**"

        # 1) Upload media (image/video) to Google's file store and build a file_data part
        media_part, uploaded_file = self._prepare_media(sample)

        # 2) Build system_instruction separately (preferred by REST)
        system_hint = (self.system_hint or "").strip()
        system_instruction: Optional[Dict[str, Any]] = None
        if system_hint:
            system_instruction = {"parts": [{"text": system_hint}]}

        # 3) Build user contents (parts: [optional file_data, text])
        user_text = strip_modality_tags(sample.prompt or "")
        if self.user_prefix:
            user_text = f"{self.user_prefix.strip()}\n\n{user_text}".strip()

        if cached_analysis_text and cached_content_name:
            contents = [{"role": "user", "parts": [{"text": cached_analysis_text}]}]
        else:
            user_parts: List[Dict[str, Any]] = []
            if media_part:
                user_parts.append(media_part)
            # Even if prompt is empty, we keep an empty text part minimal to satisfy schema
            user_parts.append({"text": user_text})
            contents = [{"role": "user", "parts": user_parts}]

        # 4) Generation config (snake_case; batch runner will pass-through)
        generation_config: Dict[str, Any] = {
            "max_output_tokens": self.max_new_tokens,
            "temperature": 0.0,
            **self.generation_overrides,
        }

        # 5) Assemble request for Batch
        request: Dict[str, Any] = {
            "model": self.model_id,
            "contents": contents,
            "generation_config": generation_config,
        }
        if system_instruction:
            request["system_instruction"] = system_instruction
        if cached_content_name:
            request["cached_content"] = cached_content_name

        # Persist uploaded file handle for later cleanup
        if not hasattr(sample, "extra_data"):
            sample.extra_data = {}
        sample.extra_data["uploaded_file"] = uploaded_file
        return request

    def _prepare_media(self, sample: TaskSample) -> Tuple[Optional[Dict[str, Any]], Optional[Any]]:
        """Uploads media and returns the part for the request, and the file object for cleanup.

        For the REST Batch API, media in contents should be referenced using:
        {"file_data": {"mime_type": <mime>, "file_uri": <uri>}}
        """
        # ====================================================================
        # [ 修正點 ]
        # 允許 'audio' modality 進入上傳邏輯。
        #
        # [ 原始程式碼 ]
        # if sample.modality not in ["image", "video"]:
        #
        # [ 修正後的程式碼 ]
        if sample.modality not in ["image", "video", "audio"]:
        # ====================================================================
            return None, None

        media_path = Path(sample.fake_path)
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        print(f"Uploading {sample.modality}: {media_path.name}...")
        file_path = str(media_path)
        if self._genai_mode == "legacy":
            uploaded_file = self._genai.upload_file(path=file_path)
            fetch_file = lambda name: self._genai.get_file(name=name)
            delete_file = lambda name: self._genai.delete_file(name=name)
        elif self._genai_mode == "modern":
            uploaded_file = self._genai.files.upload(file=file_path)
            fetch_file = lambda name: self._genai.files.get(name=name)
            delete_file = lambda name: self._genai.files.delete(name=name)
        else:
            raise RuntimeError("Gemini client is not initialized.")

        def _extract_state(file_obj: Any) -> str:
            state = getattr(file_obj, "state", None)
            if isinstance(state, str):
                return state.upper()
            name = getattr(state, "name", None)
            if isinstance(name, str):
                return name.upper()
            return ""

        def _extract_name(file_obj: Any) -> str:
            for attr in ("name", "id", "file_id", "fileId"):
                value = getattr(file_obj, attr, None)
                if isinstance(value, str) and value:
                    return value
            raise AttributeError("Uploaded file is missing a name identifier.")

        file_name = _extract_name(uploaded_file)

        # Wait for the file to be ACTIVE
        timeout = 300.0
        start_time = time.time()
        while _extract_state(uploaded_file) == "PROCESSING":
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for file {file_name} to become active.")
            time.sleep(5)
            uploaded_file = fetch_file(file_name)

        final_state = _extract_state(uploaded_file)
        if final_state != "ACTIVE":
            raise RuntimeError(f"File {file_name} failed processing with state: {final_state}")

        print(f"Upload complete: {file_name}")

        def _extract_attr(file_obj: Any, candidates: Tuple[str, ...]) -> Optional[str]:
            for attr in candidates:
                value = getattr(file_obj, attr, None)
                if isinstance(value, str) and value:
                    return value
            return None

        mime_type = _extract_attr(uploaded_file, ("mime_type", "mimeType", "mime"))
        if not mime_type:
            raise AttributeError("Uploaded file is missing mime_type information.")

        file_uri = _extract_attr(uploaded_file, ("uri", "file_uri", "fileUri"))
        if not file_uri:
            file_uri = file_name

        # Build the part for contents
        media_part = {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}

        # Store helpers for cleanup
        if not hasattr(sample, "extra_data"):
            sample.extra_data = {}
        sample.extra_data["uploaded_file"] = uploaded_file
        sample.extra_data["uploaded_file_name"] = file_name
        sample.extra_data["delete_uploaded_file"] = delete_file

        return media_part, uploaded_file

    def _ensure_prompt_cache(self, template_text: str, ttl_seconds: Optional[int]) -> Optional[str]:
        if not template_text or not self.prompt_cache_enabled:
            return None
        if not self._api_key:
            return None
        ttl = max(int(ttl_seconds) if ttl_seconds else self.prompt_cache_ttl, 60)
        cache_key = hashlib.sha256(f"{self.model_id}::{template_text}".encode("utf-8")).hexdigest()
        entry = self._prompt_cache_registry.get(cache_key)
        now = time.time()
        if entry and entry.get("expires_at", 0) > now + 5:
            return entry.get("name")

        try:
            cache_info = self._create_prompt_cache(template_text, ttl, cache_key)
        except Exception as exc:
            print(f"[WARN] Failed to create Gemini prompt cache (key={cache_key[:8]}): {exc}")
            self._prompt_cache_registry[cache_key] = {"name": None, "expires_at": now + 300}
            return None

        cache_name = cache_info["name"]
        expires_at = cache_info["expires_at"]
        self._prompt_cache_registry[cache_key] = {"name": cache_name, "expires_at": expires_at}
        self.prompt_cache_charges.append(cache_info)
        return cache_name

    def _create_prompt_cache(
        self, template_text: str, ttl_seconds: int, cache_key: str
    ) -> Dict[str, Any]:
        assert self._api_key, "API key required for prompt caching."
        model_path = self.model_id if self.model_id.startswith("models/") else f"models/{self.model_id}"
        payload: Dict[str, Any] = {
            "model": model_path,
            "ttl": f"{ttl_seconds}s",
            "display_name": f"prompt-cache-{cache_key[:8]}",
            "contents": [{"role": "user", "parts": [{"text": template_text}]}],
        }
        if self.system_hint:
            payload["system_instruction"] = {"parts": [{"text": self.system_hint}]}

        url = "https://generativelanguage.googleapis.com/v1beta/cachedContents"
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail: Any
            try:
                detail = response.json()
            except Exception:  # noqa: BLE001
                detail = response.text
            raise RuntimeError(f"cachedContents create failed: {detail}") from exc

        data: Dict[str, Any] = response.json()
        cache_name = data.get("name")
        if not cache_name:
            raise RuntimeError(f"cachedContents response missing name: {data}")
        expire_time = data.get("expireTime") or data.get("expire_time")
        expires_at = self._parse_expire_time(expire_time, ttl_seconds)
        usage_metadata = usage_to_dict(data.get("usageMetadata") or data.get("usage_metadata"))
        cost_metadata = estimate_cost_usd(self.model_id, "text", usage_metadata)
        if cost_metadata:
            print(
                f"[INFO] Created Gemini prompt cache {cache_name} "
                f"(ttl={ttl_seconds}s, cost ${cost_metadata['usd']:.6f}, input {cost_metadata['input_tokens']} tok)"
            )
        else:
            print(f"[INFO] Created Gemini prompt cache {cache_name} (ttl={ttl_seconds}s)")
        return {
            "name": cache_name,
            "expires_at": expires_at,
            "usage_metadata": usage_metadata,
            "cost_metadata": cost_metadata,
        }

    @staticmethod
    def _parse_expire_time(timestamp: Optional[str], fallback_ttl: int) -> float:
        if timestamp:
            try:
                cleaned = timestamp.replace("Z", "+00:00")
                dt = datetime.fromisoformat(cleaned)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:
                pass
        return time.time() + fallback_ttl

    def cleanup_batch_files(self, samples: List[TaskSample]):
        """Deletes all uploaded files after a batch job."""
        print("\n--- Cleaning up uploaded Gemini files ---")
        deleted_count = 0
        for sample in samples:
            if hasattr(sample, "extra_data"):
                file_name = sample.extra_data.get("uploaded_file_name")
                delete_file = sample.extra_data.get("delete_uploaded_file")
                if file_name and callable(delete_file):
                    try:
                        print(f"Deleting {file_name}...")
                        delete_file(file_name)
                        deleted_count += 1
                    except Exception as e:
                        print(f"[WARNING] Failed to delete file {file_name}: {e}")
        print(f"Successfully deleted {deleted_count} files.")
