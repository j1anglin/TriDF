from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from inference.dataio.samples import TaskSample
from inference.utils.media import load_video_frames_raw
from inference.utils.torch_helpers import apply_dtype_kw, load_auto_processor

from .base import BaseModelWrapper, ModelCapabilities


# ---------- image helpers (strict: no parameter mutation, no resize) ----------

def _open_image_rgb(path: Union[str, Path]) -> Image.Image:
    """Load an image from disk into a fully-materialized RGB PIL Image."""
    p = str(path)
    try:
        with Image.open(p) as img:
            try:
                if getattr(img, "is_animated", False):
                    img.seek(0)
            except Exception:
                pass
            img.load()
            img = img.convert("RGB")
            return img.copy()
    except UnidentifiedImageError as exc:  # noqa: BLE001
        raise ValueError(f"Cannot identify image file: {p}") from exc


def _to_rgb_pil(obj: Union[Image.Image, np.ndarray]) -> Image.Image:
    """Convert a numpy array or PIL Image into a fully-loaded RGB PIL Image (no resize)."""
    if isinstance(obj, Image.Image):
        try:
            obj.load()
        except Exception:
            pass
        return obj.convert("RGB").copy()

    if isinstance(obj, np.ndarray):
        arr = obj
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3:
            channels = arr.shape[-1]
            if channels == 4:
                arr = arr[..., :3]
            elif channels == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif channels < 3:
                raise ValueError(f"Unsupported channel count: {channels}")
            else:
                arr = arr[..., :3]
        else:
            raise ValueError(f"Unsupported ndarray shape: {arr.shape}")
        img = Image.fromarray(arr.astype(np.uint8))
        img.load()
        return img.convert("RGB").copy()

    raise TypeError(f"Unsupported image type: {type(obj)}")


# ---------- (read-only) config helpers ----------

def _norm_img_size_tuple(size_val: Any) -> Optional[Tuple[int, int]]:
    """Normalize various image_size encodings into a (width, height) tuple."""
    if size_val is None:
        return None
    if isinstance(size_val, int):
        return (int(size_val), int(size_val))
    if isinstance(size_val, (list, tuple)) and len(size_val) == 2:
        a, b = int(size_val[0]), int(size_val[1])
        return (b, a)
    if isinstance(size_val, dict):
        if "width" in size_val and "height" in size_val:
            return (int(size_val["width"]), int(size_val["height"]))
        if "shortest_edge" in size_val:
            s = int(size_val["shortest_edge"])
            return (s, s)
    return None


def _read_model_image_size(model) -> Optional[Tuple[int, int]]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    vision_cfg = getattr(cfg, "vision_config", None)
    if vision_cfg is not None:
        norm = _norm_img_size_tuple(getattr(vision_cfg, "image_size", None))
        if norm is not None:
            return norm
    norm = _norm_img_size_tuple(getattr(cfg, "image_size", None))
    if norm is not None:
        return norm
    return None


def _read_processor_image_size(processor) -> Optional[Tuple[int, int]]:
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return None
    return _norm_img_size_tuple(getattr(ip, "size", None))


# ---------- wrapper (benchmark-strict with input-only safe fallback) ----------

class MiniCPMV26Wrapper(BaseModelWrapper):
    """
    Benchmark-Strict 版本：
    - 不修改 tokenizer/processor/model 的任何屬性（不設 padding_side、不 patch pad、不覆寫 size/crop、不改切片）。
    - 不馬賽克、不手動 resize。
    - 若底層因多圖/多影格在預設流程下拋出「Sizes of tensors must match except in dimension 1」，
      則僅本次推論退回「每則訊息僅保留第一張影像」後重試一次，確保 benchmark pipeline 不中斷。
    """

    capabilities = ModelCapabilities(supports_image=True, supports_video=True)

    def __init__(
        self,
        *args,
        video_max_frames: int = 32,
        max_slice_nums: Optional[int] = None,   # 僅在你顯式傳入時才覆寫；預設不動
        use_image_id: Optional[bool] = None,    # 同上
        slice_mode: Optional[bool] = None,      # 同上
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.video_max_frames = video_max_frames
        self._requested_slice_settings = dict(
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            slice_mode=slice_mode,
        )
        self._slice_settings: Dict[str, Optional[Union[int, bool]]] = dict(
            max_slice_nums=None,
            use_image_id=None,
            slice_mode=None,
        )
        self.tokenizer = None
        self.processor = None
        self._declared_wh: Optional[Tuple[int, int]] = None  # read-only reference

    # ----- lifecycle helpers -----

    def _apply_slice_settings(
        self,
        max_slice_nums: Optional[int],
        use_image_id: Optional[bool],
        slice_mode: Optional[bool],
    ) -> None:
        """
        嚴格：預設不改。只有呼叫者明確傳入覆寫值時，才嘗試設置對應屬性。
        """
        if all(v is None for v in (max_slice_nums, use_image_id, slice_mode)):
            # 讀取目前宣告值作紀錄，不動任何設定
            cfg = getattr(self.model, "config", None)
            processor_ip = getattr(self.processor, "image_processor", None)
            slice_cfg = getattr(cfg, "slice_config", None) if cfg is not None else None

            current_max = None
            if slice_cfg is not None and hasattr(slice_cfg, "max_slice_nums"):
                try:
                    current_max = int(getattr(slice_cfg, "max_slice_nums"))
                except Exception:
                    current_max = None
            if current_max is None and processor_ip is not None and hasattr(processor_ip, "max_slice_nums"):
                try:
                    current_max = int(getattr(processor_ip, "max_slice_nums"))
                except Exception:
                    current_max = None

            current_use: Optional[bool] = None
            if cfg is not None and hasattr(cfg, "use_image_id"):
                try:
                    current_use = bool(getattr(cfg, "use_image_id"))
                except Exception:
                    current_use = None
            if current_use is None and processor_ip is not None and hasattr(processor_ip, "use_image_id"):
                try:
                    current_use = bool(getattr(processor_ip, "use_image_id"))
                except Exception:
                    current_use = None

            current_slice_mode: Optional[bool] = None
            if cfg is not None and hasattr(cfg, "slice_mode"):
                try:
                    current_slice_mode = bool(getattr(cfg, "slice_mode"))
                except Exception:
                    current_slice_mode = None
            if current_slice_mode is None and processor_ip is not None and hasattr(processor_ip, "slice_mode"):
                try:
                    current_slice_mode = bool(getattr(processor_ip, "slice_mode"))
                except Exception:
                    current_slice_mode = None

            self._slice_settings = dict(
                max_slice_nums=current_max,
                use_image_id=current_use,
                slice_mode=current_slice_mode,
            )
            return

        # 僅當你真的想覆寫時才改
        processor_ip = getattr(self.processor, "image_processor", None)
        cfg = getattr(self.model, "config", None)
        slice_cfg = getattr(cfg, "slice_config", None) if cfg is not None else None

        if max_slice_nums is not None:
            value = int(max_slice_nums)
            if processor_ip is not None:
                try:
                    setattr(processor_ip, "max_slice_nums", value)
                except Exception:
                    pass
            if slice_cfg is not None and hasattr(slice_cfg, "max_slice_nums"):
                try:
                    setattr(slice_cfg, "max_slice_nums", value)
                except Exception:
                    pass

        if use_image_id is not None:
            flag = bool(use_image_id)
            if processor_ip is not None:
                try:
                    setattr(processor_ip, "use_image_id", flag)
                except Exception:
                    pass
            if cfg is not None:
                try:
                    setattr(cfg, "use_image_id", flag)
                except Exception:
                    pass

        if slice_mode is not None:
            flag = bool(slice_mode)
            if processor_ip is not None:
                try:
                    setattr(processor_ip, "slice_mode", flag)
                except Exception:
                    pass
            if cfg is not None:
                try:
                    setattr(cfg, "slice_mode", flag)
                except Exception:
                    pass

        # 讀回目前值做紀錄
        self._apply_slice_settings(None, None, None)

    def _load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        cache_dir = self._load_kwargs.get("cache_dir")
        load_kwargs = apply_dtype_kw(self._load_kwargs, self._torch_dtype)
        load_kwargs.setdefault("trust_remote_code", True)  # 嚴格：不改其他實作參數
        self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs).eval()

        tokenizer_kwargs: Dict[str, Any] = {}
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
        tokenizer_kwargs.setdefault("trust_remote_code", True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)

        self.processor = load_auto_processor(
            self.model_id,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        # 嚴格：不調整 tokenizer.padding_side、不 patch processor.pad、不覆寫 processor.image_processor.size/crop_size
        self._declared_wh = _read_model_image_size(self.model) or _read_processor_image_size(self.processor)

        # 初始化切片設定（只讀，不改）
        self._apply_slice_settings(None, None, None)

        # 若使用者真的有要求覆寫，才改
        if any(value is not None for value in self._requested_slice_settings.values()):
            self._apply_slice_settings(
                self._requested_slice_settings.get("max_slice_nums"),
                self._requested_slice_settings.get("use_image_id"),
                self._requested_slice_settings.get("slice_mode"),
            )

    # ----- prompt preparation (strict; typed items; no resize) -----

    def _build_user_content(self, sample: TaskSample) -> List[Any]:
        prompt_body = (sample.prompt or "").strip()
        if self.user_prefix:
            prefix = self.user_prefix.strip()
            prompt_body = f"{prefix}\n\n{prompt_body}" if prompt_body else prefix
        user_text = f"{prompt_body}\n" if prompt_body else ""

        if sample.modality == "image":
            image = _open_image_rgb(sample.fake_path)
            # 使用 typed 結構，符合多模態 chat 的預期
            return [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]

        # video：擷取多影格 → 逐格轉成 PIL（不 resize），typed 結構
        frames_raw = load_video_frames_raw(
            Path(sample.fake_path),
            num_segments=self.video_max_frames,
            max_frames=self.video_max_frames,
        )
        frames: List[Image.Image] = []
        for frame in list(frames_raw)[: self.video_max_frames]:
            pil = _to_rgb_pil(frame)
            frames.append(pil)

        if not frames:
            return [{"type": "text", "text": user_text}]

        content: List[Any] = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": user_text})
        return content

    # ----- generation -----

    def _chat_kwargs_base(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return dict(
            image=None,
            msgs=messages,
            tokenizer=self.tokenizer,
            processor=self.processor,
            system_prompt=self.system_hint or "",
            sampling=False,
            max_new_tokens=self.max_new_tokens,
            stream=False,
        )

    @staticmethod
    def _force_single_image_in_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        僅保留第一張影像（每則 user 訊息），其餘影像丟棄；文字保留。
        不改任何模型/處理器設定。
        """
        out: List[Dict[str, Any]] = []
        for m in messages:
            if m.get("role") != "user":
                out.append(m)
                continue
            content = m.get("content", [])
            if not isinstance(content, list):
                out.append(m)
                continue
            new_content = []
            kept = False
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    if not kept:
                        new_content.append(item)
                        kept = True
                    # 其餘影像忽略
                else:
                    new_content.append(item)
            out.append({"role": "user", "content": new_content})
        return out

    def _chat_with_settings(self, base_kwargs: Dict[str, Any], overrides: Dict[str, Any]) -> str:
        """
        嚴格：不改參數。若遇到特定維度錯誤，只在「輸入端」退回單圖再試一次。
        """
        kw = dict(base_kwargs)
        # 僅在你有明確覆寫時，把切片相關設定帶進去；否則依模型預設
        slice_max = self._slice_settings.get("max_slice_nums")
        if slice_max is not None and self._requested_slice_settings.get("max_slice_nums") is not None:
            kw["max_slice_nums"] = slice_max
        slice_use = self._slice_settings.get("use_image_id")
        if slice_use is not None and self._requested_slice_settings.get("use_image_id") is not None:
            kw["use_image_id"] = slice_use

        kw.update(overrides)

        tried_single = False
        while True:
            try:
                response = self.model.chat(**kw)
                if isinstance(response, (list, tuple)):
                    response = response[0]
                return str(response).strip()
            except RuntimeError as exc:
                msg = str(exc)
                # 僅針對你回報的維度不合錯誤，做一次「輸入端單圖」重試
                if ("Sizes of tensors must match except in dimension 1" in msg) and (not tried_single):
                    tried_single = True
                    # 將 msgs 轉為單圖版本
                    msgs = kw.get("msgs", [])
                    kw["msgs"] = self._force_single_image_in_messages(msgs)
                    continue
                # 其它錯誤：原樣拋出，讓 benchmark 如實反映
                raise

    def generate(self, sample: TaskSample) -> str:
        self.ensure_loaded()

        messages = [{"role": "user", "content": self._build_user_content(sample)}]
        base_kwargs = self._chat_kwargs_base(messages)

        overrides = dict(self.generation_overrides or {})
        overrides.setdefault("num_beams", 1)

        return self._chat_with_settings(base_kwargs, overrides)
