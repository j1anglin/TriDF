from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set

import numpy as np
import torch

from inference.dataio.samples import ModelResponse, TaskSample
from inference.utils.jsonio import write_json_atomic
from inference.utils.text import PROMPT_ECHO_RE, canonicalize_text_block, clean_prompt_for_echo
from inference.wrappers.registry import build_model_wrapper

REFUSAL_PATTERNS = [
    r"\bi'?m\s*sorry\b.*\b(can'?t|cannot)\s*(assist|help|comply)\b",
    r"\b(cannot|can't)\s*(assist|help|comply)\b",
    r"\bi\s*can't\s*assist\s*with\s*that\s*request\b",
    r"\bi\s*cannot\s*assist\s*with\s*that\s*request\b",
    r"\bI\s*can(?:not|'t)\s*help\s*with\s*that\b",
    r"\bi'?m\s*sorry\b.*\b(can'?t|cannot)\s*provide\b",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), flags=re.IGNORECASE)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text or ""))


def _looks_like_prompt_echo(sample: TaskSample, answer: str) -> bool:
    if not sample.prompt or not answer:
        return False
    text = answer.strip()
    if not text:
        return False
    match = PROMPT_ECHO_RE.match(text)
    if not match:
        return False

    canon_prompt = clean_prompt_for_echo(sample.prompt)
    if not canon_prompt:
        return False

    canon_user_block = canonicalize_text_block(match.group("user_block"))
    if canon_prompt != canon_user_block:
        return False

    remainder = text[match.end():].strip()
    if not remainder:
        return True

    remainder_compact = re.sub(r"\s+", "", remainder).lower()
    return remainder_compact in {"dk"}


def _needs_seed_retry(model_id: str, sample: TaskSample, answer: str) -> Optional[str]:
    if not (answer or "").strip():
        return "Empty response detected"
    if is_refusal(answer):
        return "Refusal detected"
    if _looks_like_prompt_echo(sample, answer):
        return "Prompt echo detected"
    if "internvl" in model_id.lower() and (answer or "").strip().lower() == "r":
        return "InternVL placeholder response 'r' detected"
    return None


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass


def sanitize_model_id(model_id: str) -> str:
    base = model_id.rstrip("/").split("/")[-1]
    return base.replace(":", "_")


def maybe_localize(model_id: str, cache_dir: str) -> str:
    p1 = os.path.join(cache_dir, model_id)
    p2 = os.path.join(cache_dir, model_id.split("/")[-1])
    return p1 if os.path.isdir(p1) else (p2 if os.path.isdir(p2) else model_id)


def _format_exception_chain(exc: BaseException) -> str:
    segments = []
    seen: Set[int] = set()
    current: Optional[BaseException] = exc
    while current and id(current) not in seen:
        seen.add(id(current))
        segments.append(f"{current.__class__.__name__}: {current}")
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        if current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
            continue
        break
    return " -> ".join(segments)


def run_model_inference(
    tasks: Dict[str, Sequence[TaskSample]],
    *,
    model_ids: Sequence[str],
    output_root: Path,
    max_new_tokens: int = 512,
    torch_dtype: Optional[torch.dtype] = None,
    load_kwargs: Optional[Dict[str, object]] = None,
    video_max_frames: Optional[int] = None,
    max_retries: int = 2,
    base_seed: int = 42,
    system_hint: Optional[str] = None,
    overwrite_existing: bool = False,
) -> None:
    for raw_model_id in model_ids:
        print(f"[INFO] Loading model wrapper: {raw_model_id}")
        wrapper = build_model_wrapper(
            raw_model_id,
            max_new_tokens=max_new_tokens,
            torch_dtype=torch_dtype,
            load_kwargs=load_kwargs,
            video_max_frames=video_max_frames,
        )
        short_model_name = sanitize_model_id(raw_model_id)
        model_output_dir = output_root / "models" / short_model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        for task_name, samples in tasks.items():
            supported = set(wrapper.supported_modalities)
            task_samples = [sample for sample in samples if sample.modality in supported]
            if not task_samples:
                print(
                    f"[INFO] Skipping task: {task_name} (no samples compatible with modalities {sorted(supported)})"
                )
                continue

            print(
                f"[INFO] Running task: {task_name} with {len(task_samples)} / {len(samples)} compatible samples"
            )
            task_results = []
            per_sample_dir = model_output_dir / task_name
            per_sample_dir.mkdir(parents=True, exist_ok=True)

            for sample in task_samples:
                sample_path = per_sample_dir / f"{sample.sample_id}.json"
                if sample_path.exists():
                    if overwrite_existing:
                        print(f"[INFO] Overwriting existing result: {sample_path}")
                    else:
                        try:
                            with sample_path.open("r", encoding="utf-8") as handle:
                                existing_record = json.load(handle)
                            task_results.append(existing_record)
                            print(f"[INFO] Skipping existing result: {sample_path}")
                            continue
                        except Exception as exc:  # noqa: BLE001
                            print(f"[WARN] Failed to read existing result {sample_path}: {exc}. Recomputing...")

                start = time.time()
                attempt = 0
                answer = ""
                while True:
                    wrapper.system_hint = system_hint
                    wrapper.user_prefix = None
                    wrapper.generation_overrides = {}
                    if hasattr(wrapper, "last_usage"):
                        wrapper.last_usage = None
                    _set_all_seeds(base_seed + attempt)

                    try:
                        answer = wrapper.generate(sample)
                    except Exception as exc:  # noqa: BLE001
                        answer = f"[ERROR] {_format_exception_chain(exc)}"

                    retry_reason = _needs_seed_retry(raw_model_id, sample, answer)
                    if retry_reason is None or attempt >= max_retries:
                        break
                    attempt += 1
                    print(f"[INFO] {retry_reason} → retry with new seed (attempt={attempt})")

                latency = (time.time() - start) * 1000.0
                record = ModelResponse(
                    model_id=raw_model_id,
                    sample=sample,
                    response=answer,
                    latency_ms=latency,
                    fallback_count=attempt,
                    final_seed=base_seed + attempt,
                    system_hint=wrapper.system_hint,
                    usage_metadata=getattr(wrapper, "last_usage", None),
                ).to_json()
                if hasattr(wrapper, "last_usage"):
                    wrapper.last_usage = None
                write_json_atomic(record, sample_path)
                print(f"[INFO] Wrote per-sample result: {sample_path}")
                task_results.append(record)

            summary_path = model_output_dir / f"{task_name}.json"
            write_json_atomic(task_results, summary_path)
            print(f"[INFO] Saved task summary: {summary_path}")


__all__ = [
    "maybe_localize",
    "run_model_inference",
    "sanitize_model_id",
    "_set_all_seeds",
]
