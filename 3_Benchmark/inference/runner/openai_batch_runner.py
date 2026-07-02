from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import requests

from inference.dataio.samples import ModelResponse, TaskSample
from inference.utils.jsonio import write_json_atomic
from inference.wrappers.openai import OpenAIGPTBatchWrapper

OPENAI_BATCH_API = "https://api.openai.com/v1"

TokenCounts = Tuple[int, int, int]

_OPENAI_BATCH_RATES: Dict[str, Dict[str, float]] = {
    "gpt-5.0-pro": {"input": 10.0, "output": 30.0},
    "gpt-5.0-mini": {"input": 0.125, "cached_input": 0.0125, "output": 1.0},
    "gpt-5-mini": {"input": 0.125, "cached_input": 0.0125, "output": 1.0},
    "gpt-5.0": {"input": 5.0, "output": 15.0},
    "gpt-5": {"input": 0.625, "cached_input": 0.063, "output": 5.0},
    "o3": {"input": 2.0, "cached_input": 0.5, "output": 8.0},
}

def _auth_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

def _lookup_pricing(model_id: str) -> Optional[Dict[str, float]]:
    key = model_id.lower()
    for name, rates in _OPENAI_BATCH_RATES.items():
        if name in key:
            return rates
    return None

def _extract_token_counts(usage: Optional[Dict[str, Any]]) -> TokenCounts:
    if not usage:
        return 0, 0, 0
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    return input_tokens, output_tokens, total_tokens

def _estimate_cost_usd(model_id: str, usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rates = _lookup_pricing(model_id)
    if not rates:
        return None
    input_tokens, output_tokens, total_tokens = _extract_token_counts(usage)
    if input_tokens == 0 and output_tokens == 0:
        return None

    cached_tokens = 0
    if usage:
        details = usage.get("input_tokens_details") or usage.get("inputTokensDetails")
        if isinstance(details, dict):
            cached_tokens = int(details.get("cached_tokens") or details.get("cachedTokens") or 0)
            cached_tokens = max(0, min(cached_tokens, input_tokens))

    billed_input_tokens = max(input_tokens - cached_tokens, 0)
    input_rate = rates["input"]
    cached_rate = rates.get("cached_input", input_rate)

    input_cost = (billed_input_tokens / 1_000_000) * input_rate
    cached_cost = (cached_tokens / 1_000_000) * cached_rate if cached_tokens else 0.0
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    return {
        "usd": input_cost + cached_cost + output_cost,
        "input_tokens": input_tokens,
        "billed_input_tokens": billed_input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_rate_per_million": input_rate,
        "cached_input_rate_per_million": cached_rate if cached_tokens else None,
        "output_rate_per_million": rates["output"],
    }

def _upload_batch_file(jsonl_path: Path, api_key: str) -> str:
    with jsonl_path.open("rb") as handle:
        resp = requests.post(
            f"{OPENAI_BATCH_API}/files",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"purpose": "batch"},
            files={"file": (jsonl_path.name, handle, "application/jsonl")},
            timeout=300,
        )
    resp.raise_for_status()
    file_id = resp.json().get("id")
    if not file_id:
        raise RuntimeError(f"OpenAI file upload response missing id: {resp.text}")
    return file_id

def _create_batch(file_id: str, display_name: str, api_key: str, completion_window: str) -> Dict[str, Any]:
    payload = {
        "input_file_id": file_id,
        "endpoint": "/v1/responses",
        "completion_window": completion_window,
        "metadata": {"display_name": display_name},
    }
    resp = requests.post(
        f"{OPENAI_BATCH_API}/batches",
        headers=_auth_headers(api_key),
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()

def _get_batch(batch_id: str, api_key: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{OPENAI_BATCH_API}/batches/{batch_id}",
        headers=_auth_headers(api_key),
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()

def _download_file_lines(file_id: str, api_key: str) -> List[str]:
    resp = requests.get(
        f"{OPENAI_BATCH_API}/files/{file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.text.splitlines()

class _OpenAIOutputWriter:
    def __init__(self, task_name: str, output_dir: Path, wrapper: OpenAIGPTBatchWrapper) -> None:
        self.task_name = task_name
        self.task_key = task_name.lower()
        self.output_dir = Path(output_dir)
        self.wrapper = wrapper
        self.model_id = wrapper.model_id
        self.system_hint = getattr(wrapper, "system_hint", None)
        self.category = self._infer_category()
        self._task_records: Dict[str, List[Dict[str, Any]]] = {}
        self._jsonl_records: Dict[Path, List[Dict[str, Any]]] = {}

    def _infer_category(self) -> str:
        if self.task_key.startswith("perception_mc"):
            return "mc"
        if self.task_key.startswith("perception_tf"):
            return "tf"
        return "default"

    def record_success(
        self,
        sample: TaskSample,
        response_text: str,
        *,
        usage_metadata: Optional[Dict[str, Any]],
        cost_metadata: Optional[Dict[str, Any]],
    ) -> None:
        self._record(sample, response_text=response_text, usage_metadata=usage_metadata, cost_metadata=cost_metadata, error=None)

    def record_failure(self, sample: TaskSample, message: str) -> None:
        self._record(sample, response_text=message, usage_metadata=None, cost_metadata=None, error=message)

    def _record(
        self,
        sample: TaskSample,
        *,
        response_text: str,
        usage_metadata: Optional[Dict[str, Any]],
        cost_metadata: Optional[Dict[str, Any]],
        error: Optional[str],
    ) -> None:
        record = ModelResponse(
            model_id=self.model_id,
            sample=sample,
            response=response_text,
            latency_ms=0.0,
            fallback_count=0,
            final_seed=0,
            system_hint=self.system_hint,
            usage_metadata=usage_metadata,
        ).to_json()
        if cost_metadata:
            record["cost_metadata"] = cost_metadata
        if error and self.category != "default":
            record["error"] = error

        if self.category == "default":
            task_name = sample.task or "unknown_task"
            sample_dir = self.output_dir / task_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_path = sample_dir / f"{sample.sample_id}.json"
            write_json_atomic(record, sample_path)
            self._task_records.setdefault(task_name, []).append(record)
            return

        meta = getattr(sample, "extra_data", {}) or {}
        meta = meta.get("question_metadata") or {}
        record.update(
            {
                "question_file": meta.get("question_file"),
                "question_index": meta.get("question_index"),
                "raw_question": meta.get("raw_question"),
                "question_type": meta.get("question_type"),
                "artifact_type": meta.get("artifact_type"),
                "raw_modality": meta.get("raw_modality"),
                "raw_sample_path": meta.get("raw_sample_path"),
            }
        )
        if self.category == "mc" and meta.get("options") is not None:
            record["options"] = meta["options"]

        rel = meta.get("question_relpath")
        if rel:
            target = Path(rel)
        else:
            question_file = meta.get("question_file") or f"{sample.task or 'unknown_task'}.json"
            target = Path(question_file).name
            target = Path(target)
        jsonl_path = (self.output_dir / target).with_suffix(".jsonl")
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl_records.setdefault(jsonl_path, []).append(record)

    def finalize(self) -> None:
        if self.category == "default":
            for task_name, records in self._task_records.items():
                summary = self.output_dir / f"{task_name}.json"
                write_json_atomic(records, summary)
            return
        # For MC/TF categories, records are streamed directly to disk.


def run_openai_batch_job(
    *,
    wrapper: OpenAIGPTBatchWrapper,
    samples: List[TaskSample],
    output_dir: Path,
    task_name: str,
    batch_size: int = 200,
    poll_interval: int = 30,
    completion_window: str = "24h",
    max_parallel_batches: int = 1,
    skip_completed: bool = False,
) -> None:
    print(f"\n--- Starting OpenAI GPT Batch Job for task: {task_name} ---")
    print(f"Found {len(samples)} samples to process.")
    if not samples:
        print("No samples provided. Exiting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = _OpenAIOutputWriter(task_name, output_dir, wrapper)

    def _resolve_jsonl_path(sample: TaskSample) -> Path:
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

    if skip_completed:
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
                jsonl_path = _resolve_jsonl_path(sample)
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
        samples = pending_samples
    if not samples:
        print("[INFO] No remaining samples after skip check.")
        return

    total_samples = len(samples)
    chunk_size = max(int(batch_size) if batch_size else total_samples, 1)
    total_chunks = (total_samples + chunk_size - 1) // chunk_size
    print(f"[INFO] Submitting OpenAI batch in {total_chunks} chunk(s) (chunk size={chunk_size}).")
    parallel_limit = max(int(max_parallel_batches) if max_parallel_batches else 1, 1)
    if parallel_limit > 1:
        print(f"[INFO] Parallel batch submissions enabled: up to {parallel_limit} operation(s) in flight.")
    else:
        print("[INFO] Parallel batch submissions disabled: running one operation at a time.")

    total_cost_usd = 0.0
    total_input_tokens = 0
    total_cached_input_tokens = 0
    total_output_tokens = 0
    samples_with_cost = 0
    processed_samples = 0
    poll_interval_secs = max(int(poll_interval), 1)

    wrapper.ensure_loaded()
    api_key = getattr(wrapper, "_api_key", None)
    if not api_key:
        raise RuntimeError("OpenAI API key missing on wrapper.")

    active_chunks: List[Dict[str, Any]] = []

    def _prepare_chunk_context(chunk_index: int, chunk_samples: List[TaskSample]) -> Optional[Dict[str, Any]]:
        requests_payload: List[Dict[str, Any]] = []
        valid_samples: List[TaskSample] = []
        custom_ids: List[str] = []
        for sample in chunk_samples:
            try:
                prepared = wrapper.prepare_batch_request(sample)
                custom_ids.append(str(prepared.get("custom_id", sample.sample_id)))
                requests_payload.append(prepared)
                valid_samples.append(sample)
            except Exception as exc:
                print(f"[ERROR] Failed to prepare sample {sample.sample_id}: {exc}")
                writer.record_failure(sample, f"[ERROR] Batch preparation failed: {exc}")

        if not requests_payload:
            print(f"[WARN] Chunk {chunk_index + 1}: no valid requests; skipping.")
            return None

        chunk_label = f"{task_name}_chunk{chunk_index + 1:03d}" if total_chunks > 1 else task_name
        jsonl_path = output_dir / f"{chunk_label}_input.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for entry in requests_payload:
                handle.write(json.dumps(entry) + "\n")
        print(f"[INFO] Chunk {chunk_index + 1}: JSONL written to {jsonl_path}")

        return {
            "chunk_index": chunk_index,
            "chunk_label": chunk_label,
            "chunk_samples": valid_samples,
            "custom_ids": custom_ids,
            "jsonl_path": jsonl_path,
        }

    def _submit_chunk_context(ctx: Dict[str, Any]) -> bool:
        chunk_index = ctx["chunk_index"]
        chunk_samples: List[TaskSample] = ctx["chunk_samples"]
        try:
            file_id = _upload_batch_file(ctx["jsonl_path"], api_key)
            print(f"[INFO] Chunk {chunk_index + 1}: uploaded batch file {file_id}")
            batch_info = _create_batch(file_id, ctx["chunk_label"], api_key, completion_window)
        except Exception as exc:
            print(f"[ERROR] Chunk {chunk_index + 1}: failed to create batch: {exc}")
            for sample in chunk_samples:
                writer.record_failure(sample, f"[ERROR] Batch creation failed: {exc}")
            return False

        batch_id = batch_info.get("id")
        if not batch_id:
            print(f"[ERROR] Chunk {chunk_index + 1}: batch creation response missing id: {batch_info}")
            for sample in chunk_samples:
                writer.record_failure(sample, "[ERROR] Batch creation response missing id")
            return False

        ctx["batch_id"] = batch_id
        ctx["next_poll_time"] = time.time() + poll_interval_secs
        active_chunks.append(ctx)
        print(f"[INFO] Chunk {chunk_index + 1}: batch {batch_id} submitted.")
        return True

    def _process_completed_chunk(ctx: Dict[str, Any], status_payload: Dict[str, Any]) -> None:
        nonlocal processed_samples, total_cost_usd, total_input_tokens, total_cached_input_tokens, total_output_tokens, samples_with_cost
        chunk_index = ctx["chunk_index"]
        valid_samples: List[TaskSample] = ctx["chunk_samples"]
        custom_ids: List[str] = ctx.get("custom_ids", [str(s.sample_id) for s in valid_samples])
        status = status_payload.get("status")
        if status != "completed":
            error_msg = status_payload.get("error") or status_payload.get("errors")
            print(f"[ERROR] Chunk {chunk_index + 1}: batch ended with status {status}: {error_msg}")
        output_file_id = status_payload.get("output_file_id")
        if status != "completed" or not output_file_id:
            err_id = status_payload.get("error_file_id")
            if err_id:
                try:
                    lines = _download_file_lines(err_id, api_key)
                    print("[ERROR] Batch error file (first 10 lines):")
                    for line in lines[:10]:
                        print(line)
                except Exception as exc:
                    print(f"[WARN] Could not download error_file_id={err_id}: {exc}")
            for sample in valid_samples:
                writer.record_failure(sample, f"[ERROR] Batch ended with status {status}")
            return

        try:
            lines = _download_file_lines(output_file_id, api_key)
        except Exception as exc:
            print(f"[ERROR] Chunk {chunk_index + 1}: failed to download output: {exc}")
            for sample in valid_samples:
                writer.record_failure(sample, f"[ERROR] Failed to download batch output: {exc}")
            return

        responses_by_id: Dict[str, Dict[str, Any]] = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = str(record.get("custom_id"))
            responses_by_id[custom_id] = record

        completed_here = 0
        for sample, custom_id in zip(valid_samples, custom_ids):
            record = responses_by_id.get(custom_id)
            if not record:
                writer.record_failure(sample, "[ERROR] Missing response in batch output.")
                continue

            if record.get("error"):
                writer.record_failure(sample, f"[ERROR] {record['error']}")
                continue

            response_payload = record.get("response") or {}
            response_body = response_payload.get("body") if isinstance(response_payload, dict) else None
            if not isinstance(response_body, dict):
                response_body = response_payload if isinstance(response_payload, dict) else {}

            collected_text: List[str] = []
            choices = response_body.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message")
                    if isinstance(message, dict) and message.get("content"):
                        collected_text.append(str(message.get("content")))

            outputs = response_body.get("output")
            if isinstance(outputs, list):
                for item in outputs:
                    if not isinstance(item, dict): continue
                    contents = item.get("content")
                    if isinstance(contents, list):
                        for entry in contents:
                            if isinstance(entry, dict) and entry.get("type") in {"output_text", "text"} and entry.get("text"):
                                collected_text.append(str(entry.get("text")))

            response_text = "\n".join(collected_text).strip()
            if not response_text:
                response_text = str(response_body)

            usage_metadata = response_body.get("usage")
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
            completed_here += 1

        print(f"[INFO] Chunk {chunk_index + 1}: completed ({completed_here} samples processed).")

    next_start = 0
    next_chunk_index = 0

    while next_start < total_samples or active_chunks:
        while next_start < total_samples and len(active_chunks) < parallel_limit:
            end = min(next_start + chunk_size, total_samples)
            chunk_samples = samples[next_start:end]
            ctx = _prepare_chunk_context(next_chunk_index, chunk_samples)
            next_start = end
            next_chunk_index += 1
            if ctx is None:
                continue
            if not _submit_chunk_context(ctx):
                continue

        if not active_chunks:
            continue

        now = time.time()
        next_poll = min(item["next_poll_time"] for item in active_chunks)
        wait_time = max(0.0, next_poll - now)
        if wait_time > 0:
            time.sleep(wait_time)
        now = time.time()

        for item in list(active_chunks):
            if now < item["next_poll_time"]:
                continue
            batch_id = item.get("batch_id")
            try:
                status_payload = _get_batch(batch_id, api_key)
            except Exception as exc:
                print(f"[ERROR] Chunk {item['chunk_index'] + 1}: failed to poll batch {batch_id}: {exc}")
                for sample in item["chunk_samples"]:
                    writer.record_failure(sample, f"[ERROR] Batch polling failed: {exc}")
                active_chunks.remove(item)
                continue

            status = status_payload.get("status")
            print(f"[INFO] Chunk {item['chunk_index'] + 1}: status={status} ({time.ctime()})")
            item["next_poll_time"] = now + poll_interval_secs

            if status in {"completed", "failed", "cancelled", "expired"}:
                _process_completed_chunk(item, status_payload)
                active_chunks.remove(item)

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
