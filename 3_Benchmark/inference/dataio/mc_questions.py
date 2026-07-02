from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from inference.dataio.samples import TaskSample
from inference.utils.files import file_sha256, verify_audio, verify_image, verify_video
from inference.utils.text import strip_modality_tags

MODALITY_MAP = {
    "img": "image",
    "image": "image",
    "vid": "video",
    "video": "video",
    "aud": "audio",
    "audio": "audio",
}


def normalize_modalities(modality_args: Iterable[str]) -> Set[str]:
    normalized: Set[str] = set()
    for raw in modality_args or []:
        key = str(raw).lower()
        if key in {"image", "video", "audio"}:
            normalized.add(key)
            continue
        canonical = MODALITY_MAP.get(key)
        if canonical:
            normalized.add(canonical)
    return normalized


def collect_question_files(root: Path, allowed_modalities: Iterable[str]) -> List[Path]:
    if not root.exists():
        print(f"[WARN] Questions directory not found: {root}", file=sys.stderr)
        return []

    normalized = normalize_modalities(allowed_modalities)
    prefixes: Set[str] = set()
    if not normalized or "image" in normalized:
        prefixes.add("img")
    if not normalized or "video" in normalized:
        prefixes.add("vid")
    if not normalized or "audio" in normalized:
        prefixes.add("aud")
    include_combined = {"img", "vid"}.issubset(prefixes)

    files: List[Path] = []
    combined_files: List[Path] = []
    for path in sorted(root.glob("*.json")):
        stem = path.stem.lower()
        if stem.startswith("all_questions_combined"):
            if include_combined:
                combined_files.append(path)
            continue

        prefix = stem.split("_", 1)[0]
        if prefix not in {"img", "vid", "aud"}:
            continue
        if prefix in prefixes:
            files.append(path)

    return files if files else combined_files


def load_question_items(
    json_path: Path,
    allowed_modalities: Iterable[str],
    data_root: Path,
    tasks_filter: Optional[Sequence[str]],
    max_samples: Optional[int],
) -> List[Tuple[TaskSample, Dict[str, object]]]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    allowed = normalize_modalities(allowed_modalities)
    if not allowed:
        return []

    task_whitelist = set(tasks_filter) if tasks_filter else None
    results: List[Tuple[TaskSample, Dict[str, object]]] = []
    seen = 0

    for idx, entry in enumerate(payload):
        task_name = entry.get("task")
        if not task_name:
            continue
        if task_whitelist and task_name not in task_whitelist:
            continue

        raw_modality = (entry.get("modality") or "").lower()
        modality = MODALITY_MAP.get(raw_modality, raw_modality)
        if modality not in allowed:
            continue

        raw_sample_path = entry.get("sample_path")
        if not raw_sample_path:
            continue
        raw_path = Path(raw_sample_path)
        if raw_path.is_absolute():
            abs_path = raw_path
            try:
                sample_rel = raw_path.relative_to(data_root)
            except ValueError:
                sample_rel = raw_path
        else:
            first_component = raw_path.parts[0] if raw_path.parts else ""
            if first_component == task_name:
                sample_rel = raw_path
            else:
                sample_rel = Path(task_name) / raw_path
            abs_path = (data_root / sample_rel).resolve()
        if not abs_path.exists():
            print(f"[WARN] Missing media file for {json_path.name}: {sample_rel}", file=sys.stderr)
            continue

        prompt = strip_modality_tags(entry.get("question") or "")

        try:
            media_hash = file_sha256(abs_path)
        except Exception as exc:  # noqa: BLE001
            media_hash = ""
            print(f"[WARN] Failed to hash media {abs_path}: {exc}", file=sys.stderr)

        try:
            if modality == "image":
                media_meta = verify_image(abs_path)
            elif modality == "video":
                media_meta = verify_video(abs_path)
            else:
                media_meta = verify_audio(abs_path)
        except Exception as exc:  # noqa: BLE001
            media_meta = {"error": str(exc)}

        sample = TaskSample(
            task=task_name,
            sample_id=f"{json_path.stem}-{idx}",
            modality=modality,
            prompt=prompt,
            fake_path=str(abs_path),
            relative_fake_path=str(sample_rel),
            file_sha256=media_hash,
            media_meta=media_meta,
        )

        meta = {
            "json_index": idx,
            "raw_question": entry.get("question"),
            "question_type": entry.get("question_type"),
            "artifact_type": entry.get("artifact_type"),
            "raw_modality": raw_modality,
            "raw_sample_path": raw_sample_path,
            "options": entry.get("options"),
        }
        results.append((sample, meta))
        seen += 1
        if max_samples is not None and seen >= max_samples:
            break

    return results


__all__ = [
    "collect_question_files",
    "load_question_items",
    "normalize_modalities",
]
