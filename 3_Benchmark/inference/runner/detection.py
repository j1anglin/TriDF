from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from inference.dataio.samples import TaskSample

from .common import (
    maybe_localize,
    run_model_inference,
    sanitize_model_id,
)


def run_detection_inference(
    tasks: Dict[str, Sequence[TaskSample]],
    *,
    model_ids: Sequence[str],
    output_root: Path,
    max_new_tokens: int = 512,
    torch_dtype=None,
    load_kwargs: Optional[Dict[str, object]] = None,
    video_max_frames: Optional[int] = None,
    max_retries: int = 2,
    base_seed: int = 42,
    system_hint: Optional[str] = None,
    overwrite_existing: bool = False,
) -> None:
    run_model_inference(
        tasks,
        model_ids=model_ids,
        output_root=output_root,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch_dtype,
        load_kwargs=load_kwargs,
        video_max_frames=video_max_frames,
        max_retries=max_retries,
        base_seed=base_seed,
        system_hint=system_hint,
        overwrite_existing=overwrite_existing,
    )


__all__ = [
    "maybe_localize",
    "run_detection_inference",
    "sanitize_model_id",
]
