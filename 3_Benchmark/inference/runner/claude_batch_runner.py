from __future__ import annotations

import json
import time
from pathlib import Path
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

from inference.dataio.samples import ModelResponse, TaskSample
from inference.utils.jsonio import write_json_atomic
from inference.wrappers.claude import ClaudeBatchWrapper

CLAUDE_BATCH_ENDPOINT = "https://api.anthropic.com/v1/messages/batches"
CLAUDE_API_VERSION = "2023-06-01"
CLAUDE_BATCH_INPUT_RATE = 1.50  # USD per 1M tokens
CLAUDE_BATCH_OUTPUT_RATE = 7.50  # USD per 1M tokens


class _ClaudeResultsPending(Exception):
    """Raised when a completed batch has not exposed its results file yet."""


class _ClaudeChunkError(RuntimeError):
    """Raised when a Claude batch chunk returns an error response."""


def _normalize_custom_id(raw_id: Any) -> str:
    candidate = str(raw_id)
    if len(candidate) <= 64:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    prefix = candidate[:24]
    return f"{prefix}-{digest[:32]}"


class _ClaudeOutputWriter:
    def __init__(self, task_name: str, output_dir: Path, wrapper: ClaudeBatchWrapper) -> None:
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
        self._record(
            sample,
            response_text=response_text,
            usage_metadata=usage_metadata,
            cost_metadata=cost_metadata,
            error=None,
        )

    def record_failure(self, sample: TaskSample, message: str) -> None:
        self._record(
            sample,
            response_text=message,
            usage_metadata=None,
            cost_metadata=None,
            error=message,
        )

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

        question_meta = getattr(sample, "extra_data", {}) or {}
        question_meta = question_meta.get("question_metadata") or {}
        record.update(
            {
                "question_file": question_meta.get("question_file"),
                "question_index": question_meta.get("question_index"),
                "raw_question": question_meta.get("raw_question"),
                "question_type": question_meta.get("question_type"),
                "artifact_type": question_meta.get("artifact_type"),
                "raw_modality": question_meta.get("raw_modality"),
                "raw_sample_path": question_meta.get("raw_sample_path"),
            }
        )
        if self.category == "mc":
            options = question_meta.get("options")
            if options is not None:
                record["options"] = options

        question_rel = question_meta.get("question_relpath")
        if question_rel:
            rel_path = Path(question_rel)
        else:
            question_file = question_meta.get("question_file") or f"{sample.task or 'unknown_task'}.json"
            rel_path = Path(question_file).name
            rel_path = Path(rel_path)

        target_path = (self.output_dir / rel_path).with_suffix(".jsonl")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl_records.setdefault(target_path, []).append(record)

    def finalize(self) -> None:
        if self.category == "default":
            for task_name, records in self._task_records.items():
                summary_path = self.output_dir / f"{task_name}.json"
                write_json_atomic(records, summary_path)


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": CLAUDE_API_VERSION,
        "content-type": "application/json",
    }


def _create_batch(api_key: str, requests_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
    body = {"requests": requests_payload}
    resp = requests.post(CLAUDE_BATCH_ENDPOINT, headers=_headers(api_key), json=body, timeout=300)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - network failure
        detail = resp.text
        raise RuntimeError(f"Claude batch creation failed: {detail}") from exc
    return resp.json()


def _get_batch(api_key: str, batch_id: str) -> Dict[str, Any]:
    resp = requests.get(f"{CLAUDE_BATCH_ENDPOINT}/{batch_id}", headers=_headers(api_key), timeout=300)
    resp.raise_for_status()
    return resp.json()


def _stream_results_lines(api_key: str, endpoint: str) -> Iterable[Dict[str, Any]]:
    headers = _headers(api_key).copy()
    headers["accept"] = "application/binary"
    resp = requests.get(endpoint, headers=headers, timeout=600, stream=True)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in {404, 409, 425}:
            raise _ClaudeResultsPending from exc
        raise
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            yield json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue


def _download_results(api_key: str, batch_id: str, results_url: Optional[str]) -> List[Dict[str, Any]]:
    endpoints: List[str] = []
    if results_url:
        endpoints.append(results_url)
    endpoints.append(f"{CLAUDE_BATCH_ENDPOINT}/{batch_id}/results")

    pending = False
    for endpoint in endpoints:
        try:
            return list(_stream_results_lines(api_key, endpoint))
        except _ClaudeResultsPending:
            pending = True
        except Exception:
            continue
    if pending:
        raise _ClaudeResultsPending(f"Results not ready for batch {batch_id}")
    raise RuntimeError(f"Failed to download results for batch {batch_id}")


def _extract_text_from_message(message: Dict[str, Any]) -> str:
    content = message.get("content") or []
    pieces: List[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                if block.get("text"):
                    pieces.append(str(block["text"]))
    return "\n".join(pieces).strip()


def _extract_token_counts(usage: Optional[Dict[str, Any]]) -> Tuple[int, int, int]:
    if not usage:
        return 0, 0, 0

    def _get(keys: List[str]) -> int:
        for key in keys:
            if key in usage and usage[key] is not None:
                try:
                    return int(usage[key])
                except (TypeError, ValueError):
                    continue
        return 0

    input_tokens = _get(["input_token_count", "input_tokens", "prompt_tokens"])
    output_tokens = _get(["output_token_count", "output_tokens", "completion_tokens"])
    total_tokens = _get(["total_token_count", "total_tokens"])
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _estimate_cost_usd(usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    input_tokens, output_tokens, total_tokens = _extract_token_counts(usage)
    if input_tokens == 0 and output_tokens == 0:
        return None
    input_cost = (input_tokens / 1_000_000) * CLAUDE_BATCH_INPUT_RATE
    output_cost = (output_tokens / 1_000_000) * CLAUDE_BATCH_OUTPUT_RATE
    total_cost = input_cost + output_cost
    return {
        "usd": total_cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_rate_per_million": CLAUDE_BATCH_INPUT_RATE,
        "output_rate_per_million": CLAUDE_BATCH_OUTPUT_RATE,
    }


def run_claude_batch_job(
    *,
    wrapper: ClaudeBatchWrapper,
    samples: List[TaskSample],
    output_dir: Path,
    task_name: str,
    batch_size: int = 50,
    poll_interval: int = 30,
    max_parallel_batches: int = 1,
    skip_completed: bool = False,
) -> None:
    print(f"\n--- Starting Claude Batch Job for task: {task_name} ---")
    print(f"Found {len(samples)} samples to process.")
    if not samples:
        print("No samples provided. Exiting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = _ClaudeOutputWriter(task_name, output_dir, wrapper)

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
                    except Exception as exc:  # pragma: no cover - filesystem
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
    print(f"[INFO] Submitting Claude batch in {total_chunks} chunk(s) (chunk size={chunk_size}).")
    parallel_limit = max(int(max_parallel_batches) if max_parallel_batches else 1, 1)
    if parallel_limit > 1:
        print(f"[INFO] Parallel batch submissions enabled: up to {parallel_limit} operation(s) in flight.")
    else:
        print("[INFO] Parallel batch submissions disabled: running one operation at a time.")

    processed_samples = 0
    total_cost_usd = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    samples_with_cost = 0
    poll_interval_secs = max(int(poll_interval), 1)
    api_key = wrapper.api_key
    active_chunks: List[Dict[str, Any]] = []

    def _prepare_chunk_context(chunk_index: int, chunk_samples: List[TaskSample]) -> Optional[Dict[str, Any]]:
        valid_samples: List[TaskSample] = []
        requests_payload: List[Dict[str, Any]] = []
        custom_ids: List[str] = []
        for sample in chunk_samples:
            try:
                req = wrapper.prepare_batch_request(sample)
                custom_id = _normalize_custom_id(req.get("custom_id", sample.sample_id))
                req["custom_id"] = custom_id
                requests_payload.append(req)
                valid_samples.append(sample)
                custom_ids.append(custom_id)
            except Exception as exc:
                print(f"[ERROR] Failed to prepare sample {sample.sample_id}: {exc}")
                writer.record_failure(sample, f"[ERROR] Batch preparation failed: {exc}")
        if not valid_samples:
            print(f"[WARN] Chunk {chunk_index + 1}: no valid requests; skipping.")
            return None

        chunk_label = f"{task_name}_chunk{chunk_index + 1:03d}" if total_chunks > 1 else task_name
        return {
            "chunk_index": chunk_index,
            "chunk_label": chunk_label,
            "valid_samples": valid_samples,
            "requests_payload": requests_payload,
            "custom_ids": custom_ids,
        }

    def _submit_chunk_context(ctx: Dict[str, Any]) -> bool:
        chunk_index = ctx["chunk_index"]
        valid_samples = ctx["valid_samples"]
        print(
            f"[INFO] Chunk {chunk_index + 1}: submitting {len(valid_samples)} requests "
            f"({valid_samples[0].sample_id} ... {valid_samples[-1].sample_id})"
        )
        try:
            batch_info = _create_batch(api_key, ctx["requests_payload"])
        except Exception as exc:
            print(f"[ERROR] Chunk {chunk_index + 1}: batch creation failed: {exc}")
            for sample in valid_samples:
                writer.record_failure(sample, f"[ERROR] Batch creation failed: {exc}")
            return False

        batch_id = batch_info.get("id")
        if not batch_id:
            print(f"[ERROR] Chunk {chunk_index + 1}: response missing batch id.")
            for sample in valid_samples:
                writer.record_failure(sample, "[ERROR] Claude batch response missing id")
            return False

        ctx["batch_id"] = batch_id
        ctx["next_poll_time"] = time.time() + poll_interval_secs
        ctx["submitted_at"] = time.time()
        active_chunks.append(ctx)
        print(f"[INFO] Chunk {chunk_index + 1}: batch {batch_id} submitted.")
        return True

    def _process_completed_chunk(ctx: Dict[str, Any], batch_payload: Dict[str, Any]) -> int:
        nonlocal processed_samples, total_cost_usd, total_input_tokens, total_output_tokens, samples_with_cost
        chunk_index = ctx["chunk_index"]
        valid_samples: List[TaskSample] = ctx["valid_samples"]
        custom_ids: List[str] = ctx.get("custom_ids", [str(s.sample_id) for s in valid_samples])
        status = batch_payload.get("processing_status")
        if status != "ended":
            error_msg = batch_payload.get("error") or status
            print(f"[ERROR] Chunk {chunk_index + 1}: batch ended with status {status}: {error_msg}")
            for sample in valid_samples:
                writer.record_failure(sample, f"[ERROR] Batch ended with status {status}")
            raise _ClaudeChunkError(f"Chunk {chunk_index + 1} failed with status {status}")

        results_url = batch_payload.get("results_url")

        try:
            results = _download_results(api_key, ctx["batch_id"], results_url)
        except _ClaudeResultsPending as exc:
            raise
        except Exception as exc:
            print(f"[ERROR] Chunk {chunk_index + 1}: failed to download results: {exc}")
            for sample in valid_samples:
                writer.record_failure(sample, f"[ERROR] Failed to download batch output: {exc}")
            return 0

        responses_by_id = {str(entry.get("custom_id")): entry for entry in results}
        completed_here = 0
        chunk_had_error = False
        for sample, custom_id in zip(valid_samples, custom_ids):
            record = responses_by_id.get(custom_id)
            if not record:
                writer.record_failure(sample, "[ERROR] Missing response in batch output.")
                continue

            result = record.get("result") or {}
            result_type = result.get("type")
            if result_type != "succeeded":
                error_payload = result.get("error") or result
                writer.record_failure(sample, f"[ERROR] Claude batch result error: {error_payload}")
                chunk_had_error = True
                continue

            message = result.get("message") or {}
            text = _extract_text_from_message(message)
            usage_metadata = message.get("usage")
            cost_metadata = None
            cost_details = _estimate_cost_usd(usage_metadata)
            if cost_details:
                total_cost_usd += cost_details["usd"]
                total_input_tokens += cost_details["input_tokens"]
                total_output_tokens += cost_details["output_tokens"]
                samples_with_cost += 1
                cost_metadata = {**cost_details, "usd": round(cost_details["usd"], 8)}
            writer.record_success(
                sample,
                text or "[ERROR] Empty Claude response",
                usage_metadata=usage_metadata,
                cost_metadata=cost_metadata,
            )
            processed_samples += 1
            completed_here += 1

        print(f"[INFO] Chunk {chunk_index + 1}: completed ({completed_here} samples processed).")
        if chunk_had_error:
            raise _ClaudeChunkError(
                f"Chunk {chunk_index + 1} contained response errors; aborting remaining batches."
            )
        return completed_here

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
            _submit_chunk_context(ctx)

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
                batch_payload = _get_batch(api_key, batch_id)
            except Exception as exc:
                print(f"[ERROR] Chunk {item['chunk_index'] + 1}: failed to poll batch {batch_id}: {exc}")
                for sample in item["valid_samples"]:
                    writer.record_failure(sample, f"[ERROR] Batch polling failed: {exc}")
                active_chunks.remove(item)
                continue

            state = batch_payload.get("processing_status")
            print(f"[INFO] Chunk {item['chunk_index'] + 1}: status={state} ({time.ctime()})")
            item["next_poll_time"] = now + poll_interval_secs

            if state in {"ended", "failed", "canceled", "expired"}:
                results_url = batch_payload.get("results_url")
                try:
                    completed_here = _process_completed_chunk(item, batch_payload)
                except _ClaudeResultsPending:
                    print(
                        f"[INFO] Chunk {item['chunk_index'] + 1}: results pending, retrying download on next poll."
                    )
                    continue
                except _ClaudeChunkError as exc:
                    raise
                active_chunks.remove(item)

    if processed_samples:
        print(f"Successfully processed and saved {processed_samples} results.")
    else:
        print("No samples were successfully processed.")

    if samples_with_cost:
        print(
            "[INFO] Claude API cost summary: "
            f"${total_cost_usd:.6f} across {samples_with_cost} samples "
            f"(input {total_input_tokens} tok, output {total_output_tokens} tok)"
        )
    else:
        print("[INFO] Claude API cost summary unavailable.")
    writer.finalize()
