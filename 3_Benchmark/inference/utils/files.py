from __future__ import annotations

import contextlib
import hashlib
import json
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .media import HAS_DECORD, HAS_IMAGEIO, load_video_frames_raw


def file_sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def verify_image(path: Path) -> Dict[str, Any]:
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(path) as im:
        width, height = im.size
        fmt = im.format
    return {"width": width, "height": height, "format": fmt}


def verify_video(path: Path) -> Dict[str, Any]:
    try:
        if HAS_DECORD:
            from decord import VideoReader, cpu as decord_cpu  # type: ignore

            vr = VideoReader(str(path), ctx=decord_cpu(0), num_threads=1)
            return {"decoded_frames": len(vr)}
        if HAS_IMAGEIO:
            import imageio.v3 as iio  # type: ignore

            meta: Dict[str, Any] = {}
            try:
                meta = iio.immeta(str(path))
            except Exception:
                pass
            nframes = meta.get("nframes")
            if isinstance(nframes, int) and nframes > 0:
                return {"decoded_frames": int(nframes)}
            try:
                iio.imread(str(path), index=0)
                return {"decoded_frames": 1}
            except Exception as exc:
                return {"decoded_frames": 0, "error": str(exc)}
        return {"decoded_frames": 0, "error": "no decoder available"}
    except Exception as exc:
        return {"decoded_frames": 0, "error": str(exc)}


def verify_audio(path: Path) -> Dict[str, Any]:
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(path))
        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_sec": info.duration,
            "format": info.format,
        }
    except Exception:
        pass

    suffix = path.suffix.lower()
    if suffix in {".wav", ".wave"}:
        try:
            with contextlib.closing(wave.open(str(path), "rb")) as wf:
                sample_rate = wf.getframerate()
                frames = wf.getnframes()
                channels = wf.getnchannels()
                width = wf.getsampwidth()
                duration = frames / sample_rate if sample_rate else 0.0
                return {
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "frames": frames,
                    "duration_sec": duration,
                    "sample_width": width,
                    "format": "wav",
                }
        except wave.Error as exc:
            return {"error": f"wav decode error: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    return {"error": "unsupported audio format"}


def save_image_preview(src: Path, dst: Path, max_w: int = 256) -> None:
    from PIL import Image

    with Image.open(src) as im:
        im = im.convert("RGB")
        width, height = im.size
        if width > max_w:
            height = int(height * max_w / width)
            im = im.resize((max_w, height))
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="JPEG", quality=90)


def save_video_contact_sheet(src: Path, dst: Path, cols: int = 4, rows: int = 2) -> None:
    from PIL import Image

    if not HAS_DECORD and not HAS_IMAGEIO:
        return
    try:
        frames = load_video_frames_raw(src, num_segments=cols * rows)
    except Exception:
        return
    if not frames:
        return
    thumb_height = 120
    frames = [f.resize((int(f.width * thumb_height / f.height), thumb_height)) for f in frames]
    cell_w = max(f.width for f in frames)
    canvas = Image.new("RGB", (cols * cell_w, rows * thumb_height), (30, 30, 30))
    for idx, frame in enumerate(frames[: cols * rows]):
        r, c = divmod(idx, cols)
        canvas.paste(frame, (c * cell_w, r * thumb_height))
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, format="JPEG", quality=90)


def preview_paths(preview_root: Path, task: str, sample_id: str) -> Tuple[Path, Path]:
    dst = preview_root / task / f"{sample_id}.jpg"
    meta = dst.with_suffix(".meta.json")
    return dst, meta


def preview_is_fresh(dst: Path, meta: Path, sha256: str, relative_path: str) -> bool:
    try:
        if not (dst.exists() and meta.exists()):
            return False
        with meta.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload.get("sha256") == sha256 and payload.get("src") == relative_path
    except Exception:
        return False


def write_preview_meta(meta: Path, *, sha256: str, relative_path: str, modality: str, label: str, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "sha256": sha256,
        "src": relative_path,
        "modality": modality,
        "label": label,
        "ts": time.time(),
    }
    if extra:
        payload.update(extra)
    meta.parent.mkdir(parents=True, exist_ok=True)
    with meta.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
