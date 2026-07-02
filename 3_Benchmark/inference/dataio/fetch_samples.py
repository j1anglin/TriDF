from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from inference.dataio.detection import (
    discover_detection_tasks,
    load_detection_samples,
)
from inference.dataio.mc_questions import (
    collect_question_files as collect_mc_question_files,
    load_question_items as load_mc_question_items,
    normalize_modalities as normalize_mc_modalities,
)
from inference.dataio.perception import (
    discover_perception_tasks,
    load_perception_samples,
)
from inference.dataio.samples import TaskSample
from inference.dataio.tf_questions import (
    collect_question_files as collect_tf_question_files,
    load_question_items as load_tf_question_items,
    normalize_modalities as normalize_tf_modalities,
)


def _flatten_samples(task_map: dict[str, List[TaskSample]]) -> List[TaskSample]:
    samples: List[TaskSample] = []
    for sample_list in task_map.values():
        samples.extend(sample_list)
    return samples


def _limit_samples(samples: List[TaskSample], max_samples: int | None) -> List[TaskSample]:
    if max_samples is None:
        return samples
    return samples[:max_samples]


def _ensure_extra_data(samples: Iterable[TaskSample]) -> None:
    for sample in samples:
        if not hasattr(sample, "extra_data"):
            sample.extra_data = {}


def _parse_question_file_list(question_files: str | None) -> List[str]:
    if not question_files:
        return []
    parts = [entry.strip() for entry in question_files.split(",")]
    return [p for p in parts if p]


def _resolve_question_paths(base_dir: Path, requested: List[str], collector) -> List[Path]:
    if requested:
        paths: List[Path] = []
        for entry in requested:
            candidate = Path(entry)
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            if candidate.exists():
                paths.append(candidate.resolve())
        return paths
    return collector(base_dir, normalize_mc_modalities(["img", "vid", "aud"]))


def fetch_samples_for_task(
    task_name: str,
    max_samples: int | None = None,
    question_files: str | None = None,
    questions_dir: str | None = None,
    data_root: str | None = None,
    collect_name: str | None = None,
    skip_real: bool = False,
    benchmark_root: str = "/workspace",
    tasks_filter: List[str] | None = None,
) -> List[TaskSample]:
    benchmark_path = Path(benchmark_root).resolve()
    data_root_path = Path(data_root or benchmark_root).resolve()

    whitelist = set(tasks_filter) if tasks_filter else None

    task_key = task_name.lower()

    if task_key.startswith("typeb_oeq"):
        task_dirs = discover_detection_tasks(benchmark_path)
        if whitelist is not None:
            task_dirs = [task for task in task_dirs if task.name in whitelist]
            if not task_dirs:
                raise ValueError(f"No TypeB OEQ tasks matched filter: {sorted(whitelist)}")
        include_real = not skip_real
        collect_csv = collect_name or "fake_collect.csv"
        real_csv = "real_collect.csv" if include_real else None
        samples_map = load_detection_samples(
            task_dirs,
            max_samples=max_samples,
            verify_media=False,
            include_real=include_real,
            collect_name=collect_csv,
            real_collect_name=real_csv,
        )
        samples = _limit_samples(_flatten_samples(samples_map), max_samples)
        _ensure_extra_data(samples)
        return samples

    if task_key.startswith("typea_oeq"):
        task_dirs = discover_perception_tasks(benchmark_path)
        if whitelist is not None:
            task_dirs = [task for task in task_dirs if task.name in whitelist]
            if not task_dirs:
                raise ValueError(f"No TypeA OEQ tasks matched filter: {sorted(whitelist)}")
        collect_csv = collect_name or "fake_collect.csv"
        samples_map = load_perception_samples(
            task_dirs,
            max_samples=max_samples,
            verify_media=False,
            collect_name=collect_csv,
        )
        samples = _limit_samples(_flatten_samples(samples_map), max_samples)
        _ensure_extra_data(samples)
        return samples

    if task_key.startswith("perception_mc"):
        q_dir = Path(questions_dir or (benchmark_path / "benchmark_perception_mc")).resolve()
        requested = _parse_question_file_list(question_files)
        modalities = normalize_mc_modalities(["img", "vid", "aud"])

        if requested:
            question_paths: List[Path] = []
            for entry in requested:
                candidate = Path(entry)
                if not candidate.is_absolute():
                    candidate = q_dir / candidate
                if candidate.exists():
                    question_paths.append(candidate.resolve())
        else:
            question_paths = collect_mc_question_files(q_dir, modalities)

        samples: List[TaskSample] = []
        for question_path in question_paths:
            items = load_mc_question_items(
                json_path=question_path,
                allowed_modalities=modalities,
                data_root=data_root_path,
                tasks_filter=tasks_filter,
                max_samples=None,
            )
            for sample, meta in items:
                _ensure_extra_data([sample])
                try:
                    rel_question = question_path.relative_to(q_dir)
                except ValueError:
                    rel_question = question_path.name
                meta_payload = sample.extra_data.setdefault("question_metadata", {})
                meta_payload.update(
                    {
                        "question_file": str(question_path),
                        "question_relpath": str(rel_question),
                        "question_index": meta.get("json_index"),
                        "raw_question": meta.get("raw_question"),
                        "question_type": meta.get("question_type"),
                        "artifact_type": meta.get("artifact_type"),
                        "raw_modality": meta.get("raw_modality"),
                        "raw_sample_path": meta.get("raw_sample_path"),
                        "options": meta.get("options"),
                    }
                )
                samples.append(sample)
                if max_samples is not None and len(samples) >= max_samples:
                    break
            if max_samples is not None and len(samples) >= max_samples:
                break
        _ensure_extra_data(samples)
        return samples

    if task_key.startswith("perception_tf"):
        q_dir = Path(questions_dir or (benchmark_path / "benchmark_perception_tf")).resolve()
        requested = _parse_question_file_list(question_files)
        modalities = normalize_tf_modalities(["img", "vid", "aud"])

        if requested:
            question_paths = []
            for entry in requested:
                candidate = Path(entry)
                if not candidate.is_absolute():
                    candidate = q_dir / candidate
                if candidate.exists():
                    question_paths.append(candidate.resolve())
        else:
            question_paths = collect_tf_question_files(q_dir, modalities)

        samples: List[TaskSample] = []
        for question_path in question_paths:
            items = load_tf_question_items(
                json_path=question_path,
                allowed_modalities=modalities,
                data_root=data_root_path,
                tasks_filter=tasks_filter,
                max_samples=None,
                sample_root=None,
            )
            for sample, meta in items:
                _ensure_extra_data([sample])
                try:
                    rel_question = question_path.relative_to(q_dir)
                except ValueError:
                    rel_question = question_path.name
                meta_payload = sample.extra_data.setdefault("question_metadata", {})
                meta_payload.update(
                    {
                        "question_file": str(question_path),
                        "question_relpath": str(rel_question),
                        "question_index": meta.get("json_index"),
                        "raw_question": meta.get("raw_question"),
                        "question_type": meta.get("question_type"),
                        "artifact_type": meta.get("artifact_type"),
                        "raw_modality": meta.get("raw_modality"),
                        "raw_sample_path": meta.get("raw_sample_path"),
                    }
                )
                samples.append(sample)
                if max_samples is not None and len(samples) >= max_samples:
                    break
            if max_samples is not None and len(samples) >= max_samples:
                break
        _ensure_extra_data(samples)
        return samples

    raise ValueError(f"Unsupported task name: {task_name}")
