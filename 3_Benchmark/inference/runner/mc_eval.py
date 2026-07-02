from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

try:  # pragma: no cover - defensive import
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]

from inference.dataio.samples import ModelResponse, TaskSample
from inference.dataio.mc_questions import load_question_items
from inference.utils.torch_helpers import build_loading_kwargs, resolve_torch_dtype
from inference.wrappers.registry import build_model_wrapper

from .common import _needs_seed_retry, _set_all_seeds, maybe_localize, sanitize_model_id


def instantiate_wrapper(
    model_id: str,
    *,
    max_new_tokens: int,
    torch_dtype: str,
    device_map: str,
    gpus: Optional[str],
    per_gpu_max_memory_gib: Optional[int],
    flash_attn: bool,
    cache_dir: Optional[str],
    offline: bool,
    video_max_frames: int,
    system_hint: Optional[str],
    user_prefix: Optional[str],
    offload: bool,
):
    dtype = resolve_torch_dtype(torch_dtype)
    load_kwargs = build_loading_kwargs(
        device_map=device_map,
        gpus=gpus,
        per_gpu_max_memory_gib=per_gpu_max_memory_gib,
        flash_attn=flash_attn,
        cache_dir=cache_dir,
        offline=offline,
    )
    if offload:
        offload_dir_env = (
            os.environ.get("HF_OFFLOAD_DIR")
            or os.environ.get("TRANSFORMERS_OFFLOAD_DIR")
            or os.environ.get("OFFLOAD_DIR")
        )
        if offload_dir_env:
            offload_dir = Path(offload_dir_env)
        else:
            base = Path(cache_dir) if cache_dir else Path(".")
            offload_dir = base / ".hf_offload"
        try:
            offload_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        load_kwargs.setdefault("device_map", "auto")
        load_kwargs["offload_folder"] = str(offload_dir)
        load_kwargs.setdefault("offload_state_dict", True)
        load_kwargs.setdefault("offload_buffers", True)
        max_memory = load_kwargs.setdefault("max_memory", {})
        max_memory.setdefault("cpu", os.environ.get("HF_OFFLOAD_CPU_LIMIT", "256GiB"))
    wrapper = build_model_wrapper(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        torch_dtype=dtype,
        load_kwargs=load_kwargs,
        video_max_frames=video_max_frames,
    )
    wrapper.system_hint = system_hint
    wrapper.user_prefix = user_prefix
    return wrapper


def build_output_path(
    output_root: Path,
    questions_root: Path,
    json_path: Path,
    model_id: str,
) -> Path:
    rel_json = json_path.relative_to(questions_root)
    model_folder = sanitize_model_id(model_id)
    return (output_root / model_folder / rel_json).with_suffix(".jsonl")


def _existing_output_complete(output_path: Path, expected_indices: Set[str]) -> bool:
    indices: Set[str] = set()
    try:
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] {output_path.name}: invalid JSON; recomputing.")
                    return False
                idx = record.get("question_index")
                if idx is None:
                    print(f"[WARN] {output_path.name}: missing question_index; recomputing.")
                    return False
                indices.add(str(idx))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to read existing output {output_path}: {exc}; recomputing.")
        return False
    missing = expected_indices - indices
    if missing:
        print(f"[WARN] Existing output {output_path.name} missing {len(missing)} questions; recomputing.")
        return False
    return True


def _load_existing_indices(output_path: Path) -> Optional[Set[str]]:
    indices: Set[str] = set()
    try:
        with output_path.open("r+", encoding="utf-8") as handle:
            last_good_pos = handle.tell()
            while True:
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    last_good_pos = handle.tell()
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    print(f"[WARN] {output_path.name}: invalid JSON; truncating trailing data.")
                    handle.seek(last_good_pos)
                    handle.truncate()
                    break
                idx = record.get("question_index")
                if idx is None:
                    print(f"[WARN] {output_path.name}: record missing question_index; truncating trailing data.")
                    handle.seek(last_good_pos)
                    handle.truncate()
                    break
                indices.add(str(idx))
                last_good_pos = handle.tell()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to read existing output {output_path}: {exc}; restarting file.")
        return None
    return indices


def _clear_gpu_cache(processed: int) -> None:
    if torch is None or not torch.cuda.is_available():  # type: ignore[return-value]
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"[INFO] Cleared GPU cache after {processed} samples.")


def evaluate_question_files(
    question_files: Sequence[Path],
    *,
    requested_model_id: str,
    resolved_model_id: str,
    questions_root: Path,
    output_root: Path,
    wrapper,
    allowed_modalities: Iterable[str],
    data_root: Path,
    tasks_filter: Optional[Sequence[str]],
    max_samples: Optional[int],
    overwrite: bool,
    max_retries: int = 2,
    base_seed: int = 42,
    clear_gpu_cache_every: int = 0,
) -> None:
    cache_freq = max(0, int(clear_gpu_cache_every))
    for json_path in question_files:
        items = load_question_items(
            json_path=json_path,
            allowed_modalities=allowed_modalities,
            data_root=data_root,
            tasks_filter=tasks_filter,
            max_samples=max_samples,
        )
        if not items:
            print(f"[INFO] Skipping {json_path.name}: no matching samples.")
            continue

        expected_indices = {str(meta["json_index"]) for _, meta in items}
        output_path = build_output_path(
            output_root=output_root,
            questions_root=questions_root,
            json_path=json_path,
            model_id=requested_model_id,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        seen_indices: Set[str] = set()
        resume_mode = False

        if output_path.exists():
            if overwrite:
                try:
                    output_path.unlink()
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Failed to remove existing output {output_path}: {exc}")
            else:
                if _existing_output_complete(output_path, expected_indices):
                    print(
                        f"[INFO] Skipping {json_path.name}: existing output covers {len(expected_indices)} questions."
                    )
                    continue
                seen_indices = _load_existing_indices(output_path)
                if seen_indices is None:
                    try:
                        output_path.unlink()
                        print(f"[INFO] Recomputing {json_path.name}: existing output unreadable.")
                    except Exception as exc:  # noqa: BLE001
                        print(f"[WARN] Failed to remove existing output {output_path}: {exc}")
                    seen_indices = set()
                else:
                    print(
                        f"[INFO] Resuming {json_path.name}: {len(seen_indices)} / {len(expected_indices)} already processed."
                    )
                resume_mode = True

        total = len(items)
        print(f"[INFO] {json_path.name}: {total} samples to process.")

        mode = "a" if resume_mode else "w"
        with output_path.open(mode, encoding="utf-8") as handle:
            processed = len(seen_indices)
            for index, (sample, meta) in enumerate(items, start=1):
                qidx = str(meta["json_index"])
                if qidx in seen_indices:
                    continue
                attempt = 0
                response_text = ""
                error_msg: Optional[str] = None
                latency_ms = 0.0
                while True:
                    _set_all_seeds(base_seed + attempt)
                    if hasattr(wrapper, "last_usage"):
                        wrapper.last_usage = None
                    start_ts = time.perf_counter()
                    try:
                        response_text = wrapper.generate(sample)
                        error_msg = None
                    except Exception as exc:  # noqa: BLE001
                        response_text = ""
                        error_msg = str(exc)
                        print(
                            f"[ERROR] {json_path.stem}#{meta['json_index']}:{requested_model_id} -> {error_msg}",
                            file=sys.stderr,
                        )
                    latency_ms = (time.perf_counter() - start_ts) * 1000.0

                    retry_reason = _needs_seed_retry(requested_model_id, sample, response_text)
                    if error_msg and attempt < max_retries:
                        retry_reason = retry_reason or "Generation error"

                    if retry_reason and attempt < max_retries:
                        attempt += 1
                        print(f"[INFO] {retry_reason} → retry with new seed (attempt={attempt})")
                        continue
                    break

                record = ModelResponse(
                    model_id=requested_model_id,
                    sample=sample,
                    response=response_text,
                    latency_ms=latency_ms,
                    fallback_count=attempt,
                    final_seed=base_seed + attempt,
                    system_hint=wrapper.system_hint,
                    usage_metadata=getattr(wrapper, "last_usage", None),
                ).to_json()
                if hasattr(wrapper, "last_usage"):
                    wrapper.last_usage = None
                if resolved_model_id != requested_model_id:
                    record["resolved_model_path"] = resolved_model_id
                record.update(
                    {
                        "question_file": str(json_path),
                        "question_index": meta["json_index"],
                        "raw_question": meta["raw_question"],
                        "question_type": meta["question_type"],
                        "artifact_type": meta["artifact_type"],
                        "raw_modality": meta["raw_modality"],
                        "raw_sample_path": meta["raw_sample_path"],
                        "options": meta.get("options"),
                    }
                )
                if error_msg:
                    record["error"] = error_msg
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                processed += 1
                progress = processed / total
                print(
                    f"[INFO] {json_path.stem}: processed {processed}/{total} ({progress * 100:.2f}%)",
                    end="\r",
                )
                if cache_freq and (processed % cache_freq == 0):
                    _clear_gpu_cache(processed)
        print()
        print(f"[INFO] Saved results to {output_path}")


__all__ = [
    "instantiate_wrapper",
    "evaluate_question_files",
]
