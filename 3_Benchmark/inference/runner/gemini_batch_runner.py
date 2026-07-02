import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from collections import defaultdict

import requests

from inference.dataio.samples import ModelResponse, TaskSample
from inference.utils.jsonio import write_json_atomic
from inference.wrappers.gemini import GeminiWrapper

TokenCounts = Tuple[int, int, int]

# Pricing tables for Gemini batch API (USD per 1M tokens).
_GEMINI_BATCH_RATES = {
    "gemini-2.5-pro": {
        "threshold": 200_000,  # prompts above this use the higher tier
        "input_low": 0.625,
        "input_high": 1.25,
        "output_low": 5.0,
        "output_high": 7.5,
    },
    "gemini-2.5-flash": {
        "input_default": 0.15,  # text / image / video
        "input_audio": 0.50,
        "output": 1.25,
    },
}


# -------------------------
# Helpers: usage & pricing
# -------------------------

def _usage_to_dict(token_usage: Any) -> Optional[Dict[str, Any]]:
    """Best-effort conversion of usage metadata to a serializable dictionary."""
    if token_usage is None:
        return None
    if isinstance(token_usage, dict):
        return token_usage
    if hasattr(token_usage, "to_dict"):
        return token_usage.to_dict()

    data: Dict[str, Any] = {}
    candidate_keys = [
        "input_token_count",
        "output_token_count",
        "total_token_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_content_token_count",
        "thinking_token_count",
    ]
    for key in candidate_keys:
        if hasattr(token_usage, key):
            value = getattr(token_usage, key)
            if value is not None:
                data[key] = value
    return data or None


def _extract_token_counts(usage: Optional[Dict[str, Any]]) -> TokenCounts:
    """Returns (input_tokens, output_tokens, total_tokens) from usage metadata."""
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

    input_tokens = _get(
        [
            "input_token_count",
            "input_tokens",
            "prompt_token_count",
            "inputTokenCount",
            "promptTokenCount",
        ]
    )
    output_tokens = _get(
        [
            "output_token_count",
            "output_tokens",
            "candidates_token_count",
            "outputTokenCount",
            "candidatesTokenCount",
        ]
    )
    total_tokens = _get(["total_token_count", "total_tokens", "totalTokenCount"])
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _select_rates(model_id: str, modality: Optional[str], input_tokens: int) -> Optional[Dict[str, float]]:
    """Returns the applicable input/output price-per-million for a given model."""
    model_key = model_id.lower()
    if model_key.startswith("models/"):
        model_key = model_key[len("models/") :]
    if ":" in model_key:
        model_key = model_key.split(":", 1)[0]
    if model_key.startswith("gemini-2.5-pro"):
        rates = _GEMINI_BATCH_RATES["gemini-2.5-pro"]
        threshold = rates["threshold"]
        high_tier = input_tokens > threshold
        return {
            "input_rate": rates["input_high"] if high_tier else rates["input_low"],
            "output_rate": rates["output_high"] if high_tier else rates["output_low"],
            "tier": "tier2" if high_tier else "tier1",
        }

    if model_key.startswith("gemini-2.5-flash"):
        rates = _GEMINI_BATCH_RATES["gemini-2.5-flash"]
        is_audio = (modality or "").lower() == "audio"
        return {
            "input_rate": rates["input_audio"] if is_audio else rates["input_default"],
            "output_rate": rates["output"],
            "tier": "audio" if is_audio else "default",
        }

    return None


def _estimate_cost_usd(model_id: str, sample: TaskSample, usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Computes the API cost for a single sample, if pricing data is available."""
    input_tokens, output_tokens, total_tokens = _extract_token_counts(usage)
    rates = _select_rates(model_id, getattr(sample, "modality", None), input_tokens)
    if rates is None:
        return None
    if input_tokens == 0 and output_tokens == 0:
        return None

    input_cost = (input_tokens / 1_000_000) * rates["input_rate"]
    output_cost = (output_tokens / 1_000_000) * rates["output_rate"]
    total_cost = input_cost + output_cost

    return {
        "usd": total_cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_rate_per_million": rates["input_rate"],
        "output_rate_per_million": rates["output_rate"],
        "pricing_tier": rates["tier"],
    }


# -------------------------
# Helpers: request shaping
# -------------------------

def _convert_request_for_batch(request: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a prepare_batch_request dict into the HTTP payload format.

    * Keeps snake_case keys expected by the REST examples.
    * Normalizes known generation_config fields to the expected casing.
    """
    req_copy = dict(request)
    req_copy.pop("model", None)

    payload: Dict[str, Any] = {
        "contents": req_copy.get("contents", []),
    }

    # generation_config / generationConfig -> generation_config
    generation_config = request.get("generation_config") or request.get("generationConfig")
    if generation_config:
        mapped_config: Dict[str, Any] = {}
        for key, value in generation_config.items():
            if value is None:
                continue
            if key in ("max_output_tokens", "maxOutputTokens"):
                mapped_config["max_output_tokens"] = value
            elif key in ("top_p", "topP"):
                mapped_config["top_p"] = value
            elif key in ("top_k", "topK"):
                mapped_config["top_k"] = value
            elif key in ("candidate_count", "candidateCount"):
                mapped_config["candidate_count"] = value
            elif key in ("presence_penalty", "presencePenalty"):
                mapped_config["presence_penalty"] = value
            elif key in ("frequency_penalty", "frequencyPenalty"):
                mapped_config["frequency_penalty"] = value
            else:
                mapped_config[key] = value
        payload["generation_config"] = mapped_config

    # safety_settings / safetySettings
    safety = request.get("safety_settings") or request.get("safetySettings")
    if safety:
        payload["safety_settings"] = safety

    # tools
    tools = req_copy.get("tools")
    if tools:
        payload["tools"] = tools

    # tool_config / toolConfig
    tool_config = req_copy.get("tool_config") or req_copy.get("toolConfig")
    if tool_config:
        payload["tool_config"] = tool_config

    # system_instruction / systemInstruction
        system_instruction = req_copy.get("system_instruction") or req_copy.get("systemInstruction")
        if system_instruction:
            payload["system_instruction"] = system_instruction
    return payload


# -------------------------
# HTTP wrappers
# -------------------------

def _batch_generate_content(model_path: str, api_key: str, batch_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Calls the Gemini Batch REST endpoint to enqueue a job.

    Endpoint: POST /v1beta/{model_path}:batchGenerateContent
    Body shape uses snake_case fields per REST examples (display_name, input_config...).
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:batchGenerateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=batch_payload, timeout=300)
    class _GeminiChunkError(RuntimeError):
        """Signals that a Gemini batch chunk contained response errors."""

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json()
        except Exception:  # noqa: BLE001
            detail = response.text
        raise RuntimeError(f"Batch generate HTTP {response.status_code}: {detail}") from exc
    return response.json()


def _get_operation(name: str, api_key: str) -> Dict[str, Any]:
    """Polls the batch job (batches.get).

    name: e.g. "batches/123456" returned from the create call
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/{name}"
    headers = {"x-goog-api-key": api_key}
    response = requests.get(url, headers=headers, timeout=300)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json()
        except Exception:  # noqa: BLE001
            detail = response.text
        raise RuntimeError(f"Get operation HTTP {response.status_code}: {detail}") from exc
    return response.json()


def _get_file_metadata(file_name: str, api_key: str) -> Dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}"
    headers = {"x-goog-api-key": api_key}
    response = requests.get(url, headers=headers, timeout=300)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json()
        except Exception:  # noqa: BLE001
            detail = response.text
        raise RuntimeError(f"Get file HTTP {response.status_code}: {detail}") from exc
    return response.json()


def _download_file_bytes(file_name: str, api_key: str) -> bytes:
    """Downloads file content via the :download method.

    Many result files are accessible at `files/{name}:download?alt=media`.
    """
    # Allow direct HTTPS URIs returned by the API.
    if file_name.startswith("http://") or file_name.startswith("https://"):
        headers = {"x-goog-api-key": api_key}
        resp = requests.get(file_name, headers=headers, timeout=600)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network failure
            detail: Any
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Download file HTTP {resp.status_code}: {detail}") from exc
        return resp.content

    # Prefer the explicit :download endpoint.
    url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}:download"
    headers = {"x-goog-api-key": api_key}
    params = {"alt": "media"}
    resp = requests.get(url, headers=headers, params=params, timeout=600)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        # Fallback: try the file metadata for any direct URI (rarely needed for results)
        try:
            meta = _get_file_metadata(file_name, api_key)
            uri = (
                meta.get("file", {}).get("download_uri")
                or meta.get("file", {}).get("downloadUri")
                or meta.get("file", {}).get("uri")
            )
            if uri:
                alt_resp = requests.get(uri, headers=headers, timeout=600)
                alt_resp.raise_for_status()
                return alt_resp.content
        except Exception:
            pass
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Download file HTTP {resp.status_code}: {detail}") from exc
    return resp.content


def _collect_response_files(obj: Any) -> List[str]:
    """Recursively extracts candidate file names/URIs from a batch response object."""
    results: List[str] = []

    def _maybe_add(value: Any) -> None:
        if isinstance(value, str):
            if value and value not in results:
                results.append(value)
        elif isinstance(value, dict):
            for key in ("name", "file", "file_name", "fileName"):
                inner = value.get(key)
                if isinstance(inner, str) and inner:
                    _maybe_add(inner)
                    break

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_lower = key.lower()
                if key_lower in {"file", "filename", "file_name", "responses_file", "responsesfile"}:
                    _maybe_add(value)
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(obj)
    return results


# -------------------------
# Upload JSONL helper (when switching to file input)
# -------------------------

def _upload_jsonl_to_files_api(lines: List[str], display_name: str, api_key: str) -> str:
    """Uploads a JSONL string list as a Files API object. Returns files/<id> name."""
    init_url = "https://generativelanguage.googleapis.com/upload/v1beta/files"
    payload = {"file": {"display_name": display_name}}

    # Start resumable session
    init_headers = {
        "x-goog-api-key": api_key,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Type": "application/jsonl",
        "Content-Type": "application/json",
    }
    # We don't know final bytes yet, but header isn't strictly required at start.
    init_resp = requests.post(init_url, headers=init_headers, json=payload, timeout=300)
    init_resp.raise_for_status()
    upload_url = init_resp.headers.get("X-Goog-Upload-URL") or init_resp.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("Files API resumable upload URL not returned")

    body = ("\n".join(lines)).encode("utf-8")
    up_headers = {
        "x-goog-api-key": api_key,
        "X-Goog-Upload-Command": "upload, finalize",
        "X-Goog-Upload-Offset": "0",
        "Content-Type": "application/jsonl",
        "Content-Length": str(len(body)),
    }
    up_resp = requests.post(upload_url, headers=up_headers, data=body, timeout=600)
    up_resp.raise_for_status()
    info = up_resp.json()
    # Return canonical file name (e.g., files/abc123)
    file_name = info.get("file", {}).get("name") or info.get("name")
    if not file_name:
        raise RuntimeError(f"Upload succeeded but file name missing: {info}")
    if not str(file_name).startswith("files/"):
        file_name = f"files/{file_name}"
    return str(file_name)


# -------------------------
# Output helpers
# -------------------------

class _GeminiOutputWriter:
    def __init__(self, task_name: str, output_dir: Path, wrapper: GeminiWrapper) -> None:
        self.task_name = task_name
        self.task_key = task_name.lower()
        self.output_dir = Path(output_dir)
        self.wrapper = wrapper
        self.model_id = wrapper.model_id
        self.system_hint = getattr(wrapper, "system_hint", None)
        self.category = self._infer_category()
        self._task_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._jsonl_records: Dict[Path, List[Dict[str, Any]]] = defaultdict(list)

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
            self._task_records[task_name].append(record)
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
        # For MC/TF categories, records are streamed directly to disk.


# -------------------------
# Main runner
# -------------------------

def run_gemini_batch_job(
    wrapper: GeminiWrapper,
    samples: List[TaskSample],
    output_dir: Path,
    task_name: str,
    *,
    batch_size: int = 200,
    poll_interval: int = 30,
    inline_payload_limit_bytes: int = 20 * 1024 * 1024,  # 20 MB guideline from docs
    max_parallel_batches: int = 1,
):
    """
    Prepares, runs, and processes Gemini batch jobs.

    * Uses inline requests until the serialized body would exceed ~20 MB, then
      automatically switches to Files API + file_name mode.
    * Parses both "dest" (guide) and "output" (reference) output shapes.
    * Downloads result files via :download and parses JSONL lines.
    * Supports submitting multiple batch operations in parallel.
    """
    print(f"\n--- Starting Gemini Batch Job for task: {task_name} ---")
    print(f"Found {len(samples)} samples to process.")

    if not samples:
        print("No samples provided. Exiting.")
        return

    writer = _GeminiOutputWriter(task_name, output_dir, wrapper)

    def _resolve_jsonl_path(sample: TaskSample) -> Path:
        question_meta = getattr(sample, "extra_data", {}) or {}
        question_meta = question_meta.get("question_metadata") or {}
        rel = question_meta.get("question_relpath")
        if rel:
            target = Path(rel)
        else:
            question_file = question_meta.get("question_file") or f"{sample.task or 'unknown_task'}.json"
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
    total_samples = len(samples)
    if not samples:
        print("[INFO] No remaining samples after skip check.")
        return

    parallel_limit = max(int(max_parallel_batches) if max_parallel_batches else 1, 1)
    chunk_size = max(int(batch_size) if batch_size else total_samples, 1)
    total_chunks = (total_samples + chunk_size - 1) // chunk_size
    print(f"[INFO] Submitting Gemini batch in {total_chunks} chunk(s) (chunk size={chunk_size}).")
    if parallel_limit > 1:
        print(f"[INFO] Parallel batch submissions enabled: up to {parallel_limit} operation(s) in flight.")
    else:
        print("[INFO] Parallel batch submissions disabled: running one operation at a time.")

    total_cost_usd = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    samples_with_cost = 0
    processed_samples = 0
    prepared_samples: List[TaskSample] = []
    poll_interval_secs = max(int(poll_interval), 1)

    try:
        # Ensure the wrapper has initialized and loaded credentials before continuing.
        wrapper.ensure_loaded()
        api_key = getattr(wrapper, "_api_key", "") or ""
        if not api_key:
            raise RuntimeError("Gemini API key is not loaded on wrapper.")
        model_path = wrapper.model_id if wrapper.model_id.startswith("models/") else f"models/{wrapper.model_id}"

        def _process_completed_operation(
            operation: Dict[str, Any],
            valid_samples: List[TaskSample],
            chunk_index: int,
            chunk_label: str,
        ) -> None:
            nonlocal total_cost_usd, total_input_tokens, total_output_tokens, samples_with_cost, processed_samples
            human_index = chunk_index + 1

            if operation.get("error"):
                error_info = operation["error"]
                print(f"[ERROR] Chunk {human_index}: operation failed: {error_info}")
                for sample in valid_samples:
                    writer.record_failure(sample, f"[ERROR] Batch operation failed: {error_info}")
                raise _GeminiChunkError(f"Chunk {human_index} failed: {error_info}")

            resp_root = operation.get("response") or {}
            batch_obj = resp_root.get("batch") or resp_root.get("generateContentBatch") or resp_root
            dest = batch_obj.get("dest") or {}
            output_info = batch_obj.get("output") or (operation.get("metadata") or {}).get("output") or {}

            inlined: List[Dict[str, Any]] = []
            dest_inlined = dest.get("inlined_responses") or dest.get("inlinedResponses")
            if isinstance(dest_inlined, list):
                inlined = dest_inlined
            else:
                inline_container = (
                    output_info.get("inlined_responses")
                    or output_info.get("inlinedResponses")
                    or {}
                )
                if isinstance(inline_container, dict):
                    maybe_list = inline_container.get("inlined_responses") or inline_container.get("inlinedResponses")
                    if isinstance(maybe_list, list):
                        inlined = maybe_list

            if not inlined:
                batch_results = batch_obj.get("results") or []
                if isinstance(batch_results, list):
                    parsed_results: List[Dict[str, Any]] = []
                    for entry in batch_results:
                        if not isinstance(entry, dict):
                            continue
                        payload = entry.get("response") or entry.get("generateContentResponse")
                        error_payload = entry.get("error")
                        metadata = entry.get("metadata") or {}
                        record: Dict[str, Any] = {}
                        if error_payload:
                            record["error"] = error_payload
                        if payload:
                            record["response"] = payload
                        if metadata:
                            record["metadata"] = metadata
                        if record:
                            parsed_results.append(record)
                    if parsed_results:
                        inlined = parsed_results

            candidate_files: List[str] = []
            direct_file = (
                dest.get("file_name")
                or dest.get("fileName")
                or output_info.get("responses_file")
                or output_info.get("responsesFile")
                or output_info.get("fileName")
            )
            if isinstance(direct_file, str):
                candidate_files.append(direct_file)

            candidate_files.extend(_collect_response_files(dest))
            candidate_files.extend(_collect_response_files(output_info))

            seen_files: set[str] = set()
            deduped_files: List[str] = []
            for file_entry in candidate_files:
                normalized = str(file_entry)
                if not normalized or normalized in seen_files:
                    continue
                seen_files.add(normalized)
                deduped_files.append(normalized)

            if deduped_files and not inlined:
                for file_name in deduped_files:
                    try:
                        content = _download_file_bytes(file_name, api_key)
                    except Exception as exc:
                        print(
                            f"[ERROR] Chunk {human_index}: failed to download/parse responses file {file_name}: {exc}"
                        )
                        continue

                    parsed: List[Dict[str, Any]] = []
                    for line in content.decode("utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if "response" in obj or "error" in obj:
                            parsed.append(obj if "response" in obj else {"error": obj.get("error")})
                        else:
                            parsed.append({"response": obj})
                    if parsed:
                        inlined = parsed
                        break

            if not inlined:
                print("\n[DEBUG] Raw operation response (no results found):")
                try:
                    print(json.dumps(operation, indent=2))
                except Exception as exc:
                    print(f"[DEBUG] Failed to serialize operation object: {exc}")
                    print(f"[DEBUG] Raw operation object: {operation}")
                print("--------------------------------------------------\n")

            if not inlined:
                print(
                    f"[ERROR] Chunk {human_index}: no responses received. "
                    f"Keys seen: dest keys={list(dest.keys()) if isinstance(dest, dict) else dest}, "
                    f"output keys={list(output_info.keys()) if isinstance(output_info, dict) else output_info}"
                )
                for sample in valid_samples:
                    writer.record_failure(sample, "[ERROR] No response received from batch job")
                return

            if len(inlined) != len(valid_samples):
                print(
                    f"[WARN] Chunk {human_index}: mismatch between responses "
                    f"({len(inlined)}) and requests ({len(valid_samples)}); truncating."
                )

            chunk_had_error = False
            for sample, result in zip(valid_samples, inlined):
                metadata = result.get("metadata") or {}
                meta_key = metadata.get("sample_id") or metadata.get("sampleId") or metadata.get("key")
                if meta_key and str(meta_key) != str(sample.sample_id):
                    print(f"[WARN] Sample ordering mismatch: expected {sample.sample_id}, got {meta_key}.")

                response_payload = result.get("response") or {}
                error_payload = result.get("error")

                if error_payload:
                    text = f"[ERROR] Batch request error: {error_payload}"
                    usage_metadata = None
                    cost_details = None
                    chunk_had_error = True
                else:
                    candidates = response_payload.get("candidates") or []
                    text = ""
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and "text" in part]
                        text = "\n".join(tp for tp in text_parts if tp).strip()
                    if not text:
                        text = response_payload.get("text") or f"[ERROR] No text in result: {response_payload}"

                    usage_metadata = response_payload.get("usageMetadata") or response_payload.get("usage_metadata")
                    cost_details = _estimate_cost_usd(wrapper.model_id, sample, usage_metadata)

                if cost_details:
                    cost_usd = cost_details["usd"]
                    total_cost_usd += cost_usd
                    total_input_tokens += cost_details["input_tokens"]
                    total_output_tokens += cost_details["output_tokens"]
                    samples_with_cost += 1
                    print(
                        f"[INFO] Sample {sample.sample_id}: cost ${cost_usd:.6f} "
                        f"(input {cost_details['input_tokens']} tok, output {cost_details['output_tokens']} tok)"
                    )
                    cost_metadata = {**cost_details, "usd": round(cost_usd, 8)}
                else:
                    if not error_payload:
                        print(f"[INFO] Sample {sample.sample_id}: cost unavailable (missing usage data).")
                    cost_metadata = None

                if error_payload:
                    writer.record_failure(sample, text)
                else:
                    writer.record_success(
                        sample,
                        text,
                        usage_metadata=usage_metadata,
                        cost_metadata=cost_metadata,
                    )
                processed_samples += 1

            print(f"[INFO] Chunk {human_index}: completed ({len(valid_samples)} samples processed).")
            if chunk_had_error:
                raise _GeminiChunkError(
                    f"Chunk {human_index} contained response errors; aborting remaining batches."
                )

        def _prepare_chunk_context(chunk_index: int, chunk_samples: List[TaskSample]) -> Optional[Dict[str, Any]]:
            chunk_requests: List[Dict[str, Any]] = []
            valid_samples: List[TaskSample] = []

            for sample in chunk_samples:
                try:
                    req = wrapper.prepare_batch_request(sample)
                    chunk_requests.append(req)
                    valid_samples.append(sample)
                except Exception as exc:
                    print(f"[ERROR] Failed to prepare request for sample {sample.sample_id}: {exc}")
                    writer.record_failure(sample, f"[ERROR] Batch preparation failed: {exc}")

            prepared_samples.extend(valid_samples)

            if not valid_samples:
                print(
                    f"[WARN] Chunk {chunk_index + 1}: all {len(chunk_samples)} samples failed during preparation; skipping."
                )
                return None

            chunk_label = f"{task_name}_chunk{chunk_index + 1:03d}" if total_chunks > 1 else task_name
            inline_reqs = [
                {
                    "request": _convert_request_for_batch(req),
                    "metadata": {"sample_id": str(sample.sample_id), "key": str(sample.sample_id)},
                }
                for sample, req in zip(valid_samples, chunk_requests)
            ]
            inline_body = {
                "batch": {
                    "display_name": chunk_label,
                    "input_config": {"requests": {"requests": inline_reqs}},
                }
            }
            inline_bytes = json.dumps(inline_body).encode("utf-8")
            use_file_mode = len(inline_bytes) > inline_payload_limit_bytes
            if use_file_mode:
                print(
                    f"[INFO] Chunk {chunk_index + 1}: inline payload ~{len(inline_bytes)/1024/1024:.2f} MB exceeds limit; "
                    "switching to file_name mode."
                )

            context: Dict[str, Any] = {
                "chunk_index": chunk_index,
                "chunk_label": chunk_label,
                "valid_samples": valid_samples,
                "use_file_mode": use_file_mode,
                "first_sample_id": valid_samples[0].sample_id,
                "last_sample_id": valid_samples[-1].sample_id,
            }
            if use_file_mode:
                jsonl_lines: List[str] = []
                for sample, req in zip(valid_samples, chunk_requests):
                    line_obj = {
                        "key": str(sample.sample_id),
                        "request": _convert_request_for_batch(req),
                    }
                    jsonl_lines.append(json.dumps(line_obj))
                context["jsonl_lines"] = jsonl_lines
                context["inline_body"] = None
            else:
                context["inline_body"] = inline_body
                context["jsonl_lines"] = None

            return context

        active_chunks: List[Dict[str, Any]] = []

        def _submit_chunk_context(ctx: Dict[str, Any]) -> None:
            valid_samples = ctx["valid_samples"]
            first_id = ctx["first_sample_id"]
            last_id = ctx["last_sample_id"]
            print(
                f"[INFO] Chunk {ctx['chunk_index'] + 1}: submitting {len(valid_samples)} requests "
                f"({first_id} ... {last_id})"
            )

            if ctx["use_file_mode"]:
                try:
                    file_name = _upload_jsonl_to_files_api(ctx["jsonl_lines"], f"{ctx['chunk_label']}-input", api_key)
                except Exception as exc:
                    print(f"[ERROR] Chunk {ctx['chunk_index'] + 1}: failed to upload JSONL file: {exc}")
                    for sample in valid_samples:
                        writer.record_failure(sample, f"[ERROR] Batch request failed: {exc}")
                    return
                batch_body: Dict[str, Any] = {
                    "batch": {
                        "display_name": ctx["chunk_label"],
                        "input_config": {"file_name": file_name},
                    }
                }
                ctx["jsonl_lines"] = None
            else:
                batch_body = ctx["inline_body"]
                ctx["inline_body"] = None

            try:
                operation = _batch_generate_content(model_path, api_key, batch_body)
            except Exception as exc:
                print(f"[ERROR] Chunk {ctx['chunk_index'] + 1}: batch request failed: {exc}")
                for sample in valid_samples:
                    writer.record_failure(sample, f"[ERROR] Batch request failed: {exc}")
                return

            operation_name = operation.get("name")
            if not operation_name:
                print(f"[WARN] Chunk {ctx['chunk_index'] + 1}: operation response missing name {operation}")
            else:
                print(f"[INFO] Chunk {ctx['chunk_index'] + 1}: operation {operation_name} submitted.")

            if not operation.get("done", False) and operation_name:
                active_chunks.append(
                    {
                        "chunk_index": ctx["chunk_index"],
                        "chunk_label": ctx["chunk_label"],
                        "valid_samples": valid_samples,
                        "operation_name": operation_name,
                        "operation": operation,
                        "next_poll_time": time.time() + poll_interval_secs,
                    }
                )
            else:
                _process_completed_operation(operation, valid_samples, ctx["chunk_index"], ctx["chunk_label"])

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

                operation_name = item["operation_name"]
                try:
                    operation = _get_operation(operation_name, api_key) if operation_name else item["operation"]
                except Exception as exc:
                    print(
                        f"[ERROR] Chunk {item['chunk_index'] + 1}: failed to poll operation {operation_name}: {exc}"
                    )
                    for sample in item["valid_samples"]:
                        writer.record_failure(sample, f"[ERROR] Batch polling failed: {exc}")
                    active_chunks.remove(item)
                    continue

                state = (
                    operation.get("response", {})
                    .get("batch", {})
                    .get("state")
                    or operation.get("metadata", {}).get("state")
                )
                if state:
                    print(f"[INFO] Chunk {item['chunk_index'] + 1}: operation state={state} ({time.ctime()})")

                item["operation"] = operation
                item["next_poll_time"] = now + poll_interval_secs

                if operation.get("done", False):
                    _process_completed_operation(
                        operation,
                        item["valid_samples"],
                        item["chunk_index"],
                        item["chunk_label"],
                    )
                    active_chunks.remove(item)

        if processed_samples:
            print(f"Successfully processed and saved {processed_samples} results.")
        else:
            print("No samples were successfully processed.")

        if samples_with_cost:
            print(
                "[INFO] Gemini API cost summary: "
                f"${total_cost_usd:.6f} across {samples_with_cost} samples "
                f"(input {total_input_tokens} tok, output {total_output_tokens} tok)"
            )
        else:
            print("[INFO] No usage metadata returned; Gemini API cost summary unavailable.")

    except Exception as e:
        print(f"\n[FATAL ERROR] An error occurred during the batch job: {e}")
    finally:
        try:
            writer.finalize()
        finally:
            wrapper.cleanup_batch_files(prepared_samples)
