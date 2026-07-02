from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

from inference.dataio.samples import TaskSample
from inference.runner.openai_batch_runner import _OpenAIOutputWriter, _estimate_cost_usd
from inference.wrappers.openai import OpenAIGPTBatchWrapper


def _resolve_jsonl_path(output_dir: Path, sample: TaskSample) -> Path:
    meta = getattr(sample, "extra_data", {}) or {}
    meta = meta.get("question_metadata") or {}
    rel = meta.get("question_relpath")
    if rel:
        target = Path(rel)
    else:
        question_file = meta.get("question_file") or f"{sample.task or 'unknown_task'}.json"
        target = Path(question_file).name
        target = Path(target)
    return (output_dir / target).with_suffix(".jsonl")


def _filter_completed(
    samples: List[TaskSample],
    *,
    output_dir: Path,
    writer: _OpenAIOutputWriter,
) -> List[TaskSample]:
    pending_samples: List[TaskSample] = []
    skipped = 0
    jsonl_cache: Dict[Path, Set[str]] = {}

    for sample in samples:
        if writer.category == "default":
            sample_path = output_dir / (sample.task or "unknown_task") / f"{sample.sample_id}.json"
            if sample_path.exists():
                skipped += 1
                continue
        else:
            jsonl_path = _resolve_jsonl_path(output_dir, sample)
            processed_ids = jsonl_cache.get(jsonl_path)
            if processed_ids is None:
                processed_ids = set()
                if jsonl_path.exists():
                    try:
                        with jsonl_path.open("r", encoding="utf-8") as handle:
                            for line in handle:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    record = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                sample_info = record.get("sample")
                                if isinstance(sample_info, dict) and sample_info.get("sample_id") is not None:
                                    processed_ids.add(str(sample_info.get("sample_id")))
                                else:
                                    custom_id = record.get("custom_id")
                                    if custom_id is not None:
                                        processed_ids.add(str(custom_id))
                    except Exception as exc:
                        print(f"[WARN] Failed to read existing JSONL {jsonl_path}: {exc}")
                jsonl_cache[jsonl_path] = processed_ids
            if str(sample.sample_id) in processed_ids:
                skipped += 1
                continue
        pending_samples.append(sample)

    if skipped:
        print(f"[INFO] Skip-completed enabled: {skipped} existing sample(s) ignored.")
    return pending_samples


def run_openai_nobatch_job(
    *,
    wrapper: OpenAIGPTBatchWrapper,
    samples: List[TaskSample],
    output_dir: Path,
    task_name: str,
    skip_completed: bool = False,
) -> None:
    print(f"\n--- Starting OpenAI GPT No-Batch Job for task: {task_name} ---")
    print(f"Found {len(samples)} samples to process.")
    if not samples:
        print("No samples provided. Exiting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = _OpenAIOutputWriter(task_name, output_dir, wrapper)

    if skip_completed:
        samples = _filter_completed(samples, output_dir=output_dir, writer=writer)
        if not samples:
            print("[INFO] No remaining samples after skip check.")
            return

    wrapper.ensure_loaded()

    total_cost_usd = 0.0
    total_input_tokens = 0
    total_cached_input_tokens = 0
    total_output_tokens = 0
    samples_with_cost = 0
    processed_samples = 0

    for sample in samples:
        try:
            response_text = wrapper.generate(sample)
            usage_metadata = getattr(wrapper, "last_usage", None)
            cost_details = _estimate_cost_usd(wrapper.model_id, usage_metadata)
            if cost_details:
                total_cost_usd += cost_details["usd"]
                total_input_tokens += cost_details["input_tokens"]
                total_cached_input_tokens += cost_details.get("cached_tokens", 0)
                total_output_tokens += cost_details["output_tokens"]
                samples_with_cost += 1
                cost_metadata = {**cost_details, "usd": round(cost_details["usd"], 8)}
                cached_note = ""
                if cost_details.get("cached_tokens"):
                    cached_note = f", cached {cost_details['cached_tokens']} tok"
                print(
                    f"[INFO] Sample {sample.sample_id}: cost ${cost_details['usd']:.6f} "
                    f"(input {cost_details['input_tokens']} tok{cached_note}, "
                    f"output {cost_details['output_tokens']} tok)"
                )
            else:
                cost_metadata = None

            writer.record_success(
                sample,
                response_text,
                usage_metadata=usage_metadata,
                cost_metadata=cost_metadata,
            )
            processed_samples += 1
        except Exception as exc:
            writer.record_failure(sample, f"[ERROR] {exc}")
        finally:
            if hasattr(wrapper, "last_usage"):
                wrapper.last_usage = None

    if processed_samples:
        print(f"Successfully processed and saved {processed_samples} results.")
    else:
        print("No samples were successfully processed.")

    if samples_with_cost:
        cached_clause = ""
        if total_cached_input_tokens:
            cached_clause = f", cached {total_cached_input_tokens} tok"
        print(
            "[INFO] OpenAI API cost summary: "
            f"${total_cost_usd:.6f} across {samples_with_cost} samples "
            f"(input {total_input_tokens} tok{cached_clause}, output {total_output_tokens} tok)"
        )
    else:
        print("[INFO] No usage metadata returned; OpenAI API cost summary unavailable.")

    writer.finalize()
