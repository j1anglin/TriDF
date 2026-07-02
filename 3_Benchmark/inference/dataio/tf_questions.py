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


def _load_json_any(json_path: Path) -> List[object]:
    text = json_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        items: List[object] = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except json.JSONDecodeError:
                items.append(s)
        return items

    if isinstance(obj, dict):
        if obj and all(isinstance(v, list) for v in obj.values()):
            flat: List[object] = []
            for task_name, entries in obj.items():
                if not isinstance(entries, list):
                    continue
                for item in entries:
                    if isinstance(item, dict):
                        if "task" not in item:
                            item = {**item, "task": task_name}
                        flat.append(item)
                    elif isinstance(item, str):
                        s = item.strip()
                        try:
                            maybe = json.loads(s)
                            if isinstance(maybe, dict):
                                if "task" not in maybe:
                                    maybe = {**maybe, "task": task_name}
                                flat.append(maybe)
                            else:
                                flat.append(item)
                        except json.JSONDecodeError:
                            flat.append(item)
                    else:
                        flat.append(item)
            return flat

        for key in ("items", "questions", "data", "samples"):
            val = obj.get(key)
            if isinstance(val, list):
                return val
        return [obj]

    if isinstance(obj, list):
        return obj

    return [obj]


def _coerce_entry_to_dict(entry: object, json_path: Path, idx: int) -> Optional[dict]:
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str):
        s = entry.strip()
        try:
            maybe = json.loads(s)
            if isinstance(maybe, dict):
                return maybe
        except json.JSONDecodeError:
            pass
        print(
            f"[WARN] {json_path.name} line {idx}: string entry not parseable as JSON object; skipping.",
            file=sys.stderr,
        )
        return None
    print(
        f"[WARN] {json_path.name} index {idx}: entry type {type(entry).__name__} not supported; skipping.",
        file=sys.stderr,
    )
    return None


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


def resolve_sample_paths(
    task_name: str,
    raw_sample_path: str,
    *,
    data_root: Path,
    sample_root: Optional[Path],
) -> Tuple[Path, Path]:
    raw = Path(raw_sample_path)
    if raw.is_absolute():
        return raw.resolve(), raw

    if raw.parts and raw.parts[0] == task_name:
        sample_rel = raw
    else:
        sample_rel = Path(task_name) / raw

    base_root = sample_root if sample_root else data_root
    abs_path = (base_root / sample_rel).resolve()
    return abs_path, sample_rel


def load_question_items(
    json_path: Path,
    allowed_modalities: Iterable[str],
    *,
    data_root: Path,
    tasks_filter: Optional[Sequence[str]],
    max_samples: Optional[int],
    sample_root: Optional[Path],
) -> List[Tuple[TaskSample, Dict[str, object]]]:
    payload = _load_json_any(json_path)
    allowed = normalize_modalities(allowed_modalities)
    if not allowed:
        return []

    task_whitelist = set(tasks_filter) if tasks_filter else None
    results: List[Tuple[TaskSample, Dict[str, object]]] = []
    seen = 0

    for idx, raw in enumerate(payload):
        entry = _coerce_entry_to_dict(raw, json_path, idx)
        if not entry:
            continue

        task_name = entry.get("task")
        if not task_name:
            print(f"[WARN] {json_path.name} index {idx}: missing 'task'; skipping.", file=sys.stderr)
            continue
        if task_whitelist and task_name not in task_whitelist:
            continue

        raw_modality = (entry.get("modality") or "").lower()
        modality = MODALITY_MAP.get(raw_modality, raw_modality)
        if modality not in allowed:
            continue

        raw_sample_path = entry.get("sample_path")
        if not raw_sample_path:
            print(f"[WARN] {json_path.name} index {idx}: missing 'sample_path'; skipping.", file=sys.stderr)
            continue

        abs_path, sample_rel = resolve_sample_paths(
            task_name=task_name,
            raw_sample_path=raw_sample_path,
            data_root=data_root,
            sample_root=sample_root,
        )

        if not abs_path.exists():
            print(f"[WARN] Missing media file for {json_path.name}: {sample_rel}", file=sys.stderr)
            continue

        prompt = strip_modality_tags(entry.get("question") or "")

        try:
            media_hash = file_sha256(abs_path)
        except Exception as exc:
            media_hash = ""
            print(f"[WARN] Failed to hash media {abs_path}: {exc}", file=sys.stderr)

        try:
            if modality == "image":
                media_meta = verify_image(abs_path)
            elif modality == "video":
                media_meta = verify_video(abs_path)
            else:
                media_meta = verify_audio(abs_path)
        except Exception as exc:
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
        }
        results.append((sample, meta))
        seen += 1
        if max_samples is not None and seen >= max_samples:
            break

    return results


__all__ = [
    "MODALITY_MAP",
    "collect_question_files",
    "load_question_items",
    "normalize_modalities",
    "resolve_sample_paths",
]
