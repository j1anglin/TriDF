from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import typeb_oeq

from .samples import (
    ModelResponse,
    TaskSample,
    discover_task_dirs,
    load_all_samples,
    load_task_samples,
    materialize_previews,
    persist_task_manifests,
)

DETECTION_PROMPTS: Dict[str, str] = {
    "image": typeb_oeq.questions[0],
    "video": typeb_oeq.questions[1],
    "audio": typeb_oeq.questions[2],
}
DETECTION_SYSTEM_HINT: str = typeb_oeq.SYSTEM_HINT


def discover_detection_tasks(benchmark_root: Path) -> List[Path]:
    return discover_task_dirs(benchmark_root)


def load_detection_task_samples(
    task_dir: Path,
    *,
    max_samples: Optional[int] = None,
    verify_media: bool = True,
    include_real: bool = True,
    collect_name: str = "fake_collect.csv",
    real_collect_name: Optional[str] = "real_collect.csv",
) -> List[TaskSample]:
    return load_task_samples(
        task_dir,
        prompts=DETECTION_PROMPTS,
        max_samples=max_samples,
        verify_media=verify_media,
        include_real=include_real,
        collect_name=collect_name,
        real_collect_name=real_collect_name,
    )


def load_detection_samples(
    task_dirs: Iterable[Path],
    *,
    max_samples: Optional[int] = None,
    verify_media: bool = True,
    include_real: bool = True,
    collect_name: str = "fake_collect.csv",
    real_collect_name: Optional[str] = "real_collect.csv",
) -> Dict[str, List[TaskSample]]:
    return load_all_samples(
        task_dirs,
        prompts=DETECTION_PROMPTS,
        max_samples=max_samples,
        verify_media=verify_media,
        include_real=include_real,
        collect_name=collect_name,
        real_collect_name=real_collect_name,
    )


__all__ = [
    "ModelResponse",
    "TaskSample",
    "DETECTION_PROMPTS",
    "DETECTION_SYSTEM_HINT",
    "discover_detection_tasks",
    "load_detection_task_samples",
    "load_detection_samples",
    "materialize_previews",
    "persist_task_manifests",
]
