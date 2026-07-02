from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import typea_oeq

from inference.utils.text import strip_modality_tags

from .samples import (
    ModelResponse,
    TaskSample,
    discover_task_dirs,
    load_all_samples,
    load_task_samples,
    materialize_previews,
    persist_task_manifests,
)

PERCEPTION_PROMPTS: Dict[str, str] = {
    "image": strip_modality_tags(typea_oeq.questions[0]),
    "video": strip_modality_tags(typea_oeq.questions[1]),
    "audio": strip_modality_tags(typea_oeq.questions[2]),
}
PERCEPTION_SYSTEM_HINT = getattr(typea_oeq, "SYSTEM_HINT", None)


def discover_perception_tasks(benchmark_root: Path) -> List[Path]:
    return discover_task_dirs(benchmark_root)


def load_perception_task_samples(
    task_dir: Path,
    *,
    max_samples: Optional[int] = None,
    verify_media: bool = True,
    collect_name: str = "fake_collect.csv",
) -> List[TaskSample]:
    return load_task_samples(
        task_dir,
        prompts=PERCEPTION_PROMPTS,
        max_samples=max_samples,
        verify_media=verify_media,
        include_real=False,
        collect_name=collect_name,
        real_collect_name=None,
    )


def load_perception_samples(
    task_dirs: Iterable[Path],
    *,
    max_samples: Optional[int] = None,
    verify_media: bool = True,
    collect_name: str = "fake_collect.csv",
) -> Dict[str, List[TaskSample]]:
    return load_all_samples(
        task_dirs,
        prompts=PERCEPTION_PROMPTS,
        max_samples=max_samples,
        verify_media=verify_media,
        include_real=False,
        collect_name=collect_name,
        real_collect_name=None,
    )


__all__ = [
    "ModelResponse",
    "TaskSample",
    "PERCEPTION_PROMPTS",
    "PERCEPTION_SYSTEM_HINT",
    "discover_perception_tasks",
    "load_perception_task_samples",
    "load_perception_samples",
    "materialize_previews",
    "persist_task_manifests",
]
