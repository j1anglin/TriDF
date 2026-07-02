from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from inference.utils.files import (
    file_sha256,
    preview_is_fresh,
    preview_paths,
    save_image_preview,
    save_video_contact_sheet,
    verify_image,
    verify_video,
    verify_audio,
    write_preview_meta,
)
from inference.utils.jsonio import write_json


@dataclass
class TaskSample:
    task: str
    sample_id: str
    modality: str
    prompt: str
    fake_path: str
    relative_fake_path: str
    label: str = "fake"
    file_sha256: Optional[str] = None
    media_meta: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["fake_path"] = str(payload["fake_path"])
        payload["relative_fake_path"] = str(payload["relative_fake_path"])
        return payload


@dataclass
class ModelResponse:
    model_id: str
    sample: TaskSample
    response: str
    latency_ms: float
    fallback_count: int = 0
    final_seed: int = 0
    system_hint: Optional[str] = None
    usage_metadata: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "sample": self.sample.to_json(),
            "response": self.response,
            "timestamp": time.time(),
            "fallback_count": self.fallback_count,
            "final_seed": self.final_seed,
            "system_hint": self.system_hint,
            "usage_metadata": self.usage_metadata,
        }


MEDIA_COLUMN_ALIASES = {
    "image": {
        "fake": {"fake_image", "fake_img", "image_path", "img_path"},
        "real": {"real_image", "real_img", "real_image_path"},
    },
    "video": {
        "fake": {"fake_mp4", "fake_video", "video_path", "mp4_path"},
        "real": {"real_video", "real_mp4", "real_video_path"},
    },
    "audio": {
        "fake": {"fake_audio", "audio_path", "fake_wav", "fake_sound"},
        "real": {"real_audio", "audio_real_path", "real_wav", "real_sound"},
    },
}

COLLECT_FILE_CANDIDATES = ("fake_collect.csv", "collect.csv")


def _infer_modality_from_task_name(task_name: str) -> Optional[str]:
    name = task_name.lower()
    if name.startswith("vid_") or name.startswith("video_") or "video" in name or "vid" in name:
        return "video"
    if name.startswith("aud_") or name.startswith("audio_") or "audio" in name or "aud" in name:
        return "audio"
    if name.startswith("img_") or name.startswith("image_") or "image" in name or "img" in name:
        return "image"
    return None


def discover_task_dirs(benchmark_root: Path, *, skip_prefixes: Sequence[str] = ()) -> List[Path]:
    task_dirs: List[Path] = []
    for child in sorted(benchmark_root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.name.startswith(prefix) for prefix in skip_prefixes):
            continue
        if any((child / name).exists() for name in COLLECT_FILE_CANDIDATES):
            task_dirs.append(child)
    return task_dirs


def _match_media_column(
    lower_map: Dict[str, str], headers: Sequence[str], modality: str, label: str
) -> Optional[str]:
    label = label.lower()
    for alias in MEDIA_COLUMN_ALIASES.get(modality, {}).get(label, set()):
        if alias in lower_map:
            return lower_map[alias]

    for header in headers:
        lower = header.lower()
        if label not in lower:
            continue
        if modality == "image" and ("img" in lower or "image" in lower):
            return header
        if modality == "video" and ("mp4" in lower or "video" in lower):
            return header
        if modality == "audio" and ("audio" in lower or "wav" in lower or "flac" in lower or "mp3" in lower):
            return header
    return None


def resolve_media_column(
    headers: Sequence[str],
    preferred_labels: Sequence[str] = ("fake", "real"),
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    lower_map = {h.lower(): h for h in headers}
    modalities = ("image", "video", "audio")
    for label in preferred_labels:
        for modality in modalities:
            column = _match_media_column(lower_map, headers, modality, label)
            if column:
                return modality, column, label
    return None, None, None


def _load_task_samples_from_csv(
    task_dir: Path,
    csv_path: Path,
    *,
    label: str,
    max_samples: Optional[int],
    verify_media: bool,
    prompts: Dict[str, str],
) -> List[TaskSample]:
    if max_samples is not None and max_samples <= 0:
        return []

    samples: List[TaskSample] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        search_labels: List[str] = []
        for candidate in (label, "fake", "real"):
            if candidate not in search_labels:
                search_labels.append(candidate)
        modality, media_column, _ = resolve_media_column(headers, tuple(search_labels))
        if not media_column or modality not in prompts:
            lower_map = {h.lower(): h for h in headers}
            data_column = None
            if label.lower() == "real" and "real_data" in lower_map:
                data_column = lower_map["real_data"]
            elif "fake_data" in lower_map:
                data_column = lower_map["fake_data"]
            elif "real_data" in lower_map:
                data_column = lower_map["real_data"]
            if data_column:
                inferred = _infer_modality_from_task_name(task_dir.name)
                if inferred:
                    modality = inferred
                    media_column = data_column
        if modality not in prompts:
            print(f"[WARN] Skipping task {task_dir.name}: unsupported modality inferred from columns {headers}")
            return samples
        if not media_column:
            print(f"[WARN] Skipping task {task_dir.name}: unable to identify media column for label '{label}'")
            return samples

        base_prompt = prompts[modality]
        for row_idx, row in enumerate(reader):
            if max_samples is not None and len(samples) >= max_samples:
                break
            media_rel = (row.get(media_column) or "").strip()
            if not media_rel:
                continue
            media_abs = (task_dir / media_rel).resolve()
            if not media_abs.exists():
                print(f"[WARN] Missing media file: {media_abs}")
                continue

            try:
                sha = file_sha256(media_abs)
                if verify_media:
                    if modality == "image":
                        meta = verify_image(media_abs)
                    elif modality == "video":
                        meta = verify_video(media_abs)
                    else:
                        meta = verify_audio(media_abs)
                else:
                    meta = None
            except Exception as exc:
                print(f"[WARN] Media verification failed for {media_abs}: {exc}")
                sha = None
                meta = None

            raw_sample_id = str(row.get("idx") or row_idx)
            sample_id = raw_sample_id if label == "fake" else f"{label}-{raw_sample_id}"
            samples.append(
                TaskSample(
                    task=task_dir.name,
                    sample_id=sample_id,
                    modality=modality,
                    prompt=base_prompt,
                    fake_path=str(media_abs),
                    relative_fake_path=f"{task_dir.name}/{media_rel}",
                    label=label,
                    file_sha256=sha,
                    media_meta=meta,
                )
            )
    return samples


def load_task_samples(
    task_dir: Path,
    *,
    prompts: Dict[str, str],
    max_samples: Optional[int] = None,
    verify_media: bool = True,
    include_real: bool = True,
    collect_name: str = "fake_collect.csv",
    real_collect_name: Optional[str] = "real_collect.csv",
) -> List[TaskSample]:
    collect_path = task_dir / collect_name
    if not collect_path.exists():
        if collect_name == "fake_collect.csv":
            fallback_path = task_dir / "collect.csv"
            if fallback_path.exists():
                collect_path = fallback_path
            else:
                raise FileNotFoundError(f"Missing {collect_name} in {task_dir}")
        elif collect_name == "collect.csv":
            raise FileNotFoundError(f"Missing {collect_name} in {task_dir}")
        else:
            print(f"[WARN] Missing {collect_name} in {task_dir}. Skipping task.")
            return []

    samples: List[TaskSample] = []
    remaining = max_samples

    fake_samples = _load_task_samples_from_csv(
        task_dir,
        collect_path,
        label="fake",
        max_samples=remaining,
        verify_media=verify_media,
        prompts=prompts,
    )
    samples.extend(fake_samples)
    if remaining is not None:
        remaining = max(0, remaining - len(fake_samples))

    if include_real and real_collect_name:
        real_path = task_dir / real_collect_name
        if not real_path.exists():
            print(f"[WARN] Expected real samples CSV not found: {real_path}")
        elif remaining is None or remaining > 0:
            real_samples = _load_task_samples_from_csv(
                task_dir,
                real_path,
                label="real",
                max_samples=remaining,
                verify_media=verify_media,
                prompts=prompts,
            )
            samples.extend(real_samples)
            if remaining is not None:
                remaining = max(0, remaining - len(real_samples))

    return samples


def load_all_samples(
    task_dirs: Iterable[Path],
    *,
    prompts: Dict[str, str],
    max_samples: Optional[int] = None,
    verify_media: bool = True,
    include_real: bool = True,
    collect_name: str = "fake_collect.csv",
    real_collect_name: Optional[str] = "real_collect.csv",
) -> Dict[str, List[TaskSample]]:
    tasks: Dict[str, List[TaskSample]] = {}
    for task_dir in task_dirs:
        samples = load_task_samples(
            task_dir,
            prompts=prompts,
            max_samples=max_samples,
            verify_media=verify_media,
            include_real=include_real,
            collect_name=collect_name,
            real_collect_name=real_collect_name,
        )
        if samples:
            tasks[task_dir.name] = samples
    return tasks


def persist_task_manifests(tasks: Dict[str, List[TaskSample]], output_root: Path) -> None:
    for task_name, samples in tasks.items():
        payload = [sample.to_json() for sample in samples]
        write_json(payload, output_root / "tasks" / f"{task_name}.json")


def materialize_previews(
    tasks: Dict[str, List[TaskSample]],
    preview_root: Path,
    *,
    policy: str = "auto",
) -> None:
    policy = (policy or "auto").lower()
    if policy not in {"auto", "force", "skip"}:
        raise ValueError(f"Unsupported preview policy: {policy}")
    if policy == "skip":
        return

    for task, samples in tasks.items():
        for sample in samples:
            if not sample.file_sha256:
                continue
            try:
                dst, meta = preview_paths(preview_root, task, sample.sample_id)
                if sample.modality == "audio":
                    extra = sample.media_meta or {}
                    write_preview_meta(
                        meta,
                        sha256=sample.file_sha256,
                        relative_path=sample.relative_fake_path,
                        modality=sample.modality,
                        label=sample.label,
                        extra=extra,
                    )
                    continue
                if policy == "auto" and preview_is_fresh(dst, meta, sample.file_sha256, sample.relative_fake_path):
                    continue
                if sample.modality == "image":
                    save_image_preview(Path(sample.fake_path), dst)
                    width = sample.media_meta.get("width") if sample.media_meta else None
                    height = sample.media_meta.get("height") if sample.media_meta else None
                    extra = {"width": width, "height": height}
                elif sample.modality == "video":
                    try:
                        save_video_contact_sheet(Path(sample.fake_path), dst)
                        extra = None
                    except Exception as exc:
                        print(f"[WARN] Preview failed for video {sample.fake_path}: {exc}")
                        extra = {"error": str(exc)}
                else:
                    extra = {"error": f"Unsupported modality for preview: {sample.modality}"}
                write_preview_meta(
                    meta,
                    sha256=sample.file_sha256,
                    relative_path=sample.relative_fake_path,
                    modality=sample.modality,
                    label=sample.label,
                    extra=extra,
                )
            except Exception as exc:
                print(f"[WARN] Failed to create/verify preview for {getattr(sample, 'fake_path', '?')}: {exc}")
