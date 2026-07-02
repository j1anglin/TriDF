import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from inference.dataio.fetch_samples import fetch_samples_for_task
from inference.runner.openai_batch_runner import (
    _OpenAIOutputWriter,
    _download_file_lines,
    _estimate_cost_usd,
    _get_batch,
)
from inference.wrappers.registry import build_model_wrapper


def _parse_batch_ids(log_path: Optional[Path]) -> List[str]:
    if not log_path:
        return []
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"\b(batch_[a-f0-9]+)\b", re.IGNORECASE)
    seen: Set[str] = set()
    ordered: List[str] = []
    for match in pattern.findall(text):
        batch_id = match.strip()
        if batch_id and batch_id not in seen:
            seen.add(batch_id)
            ordered.append(batch_id)
    return ordered


def _extract_response_payload(record: Dict) -> Dict:
    response_payload = record.get("response") or {}
    if not isinstance(response_payload, dict):
        return {}
    body = response_payload.get("body")
    if isinstance(body, dict):
        return body
    return response_payload


def _collect_response_text(output_entries: Sequence[Dict]) -> str:
    if not isinstance(output_entries, Sequence):
        return ""
    collected: List[str] = []
    for entry in output_entries:
        if not isinstance(entry, dict):
            continue
        contents = entry.get("content")
        if not isinstance(contents, list):
            continue
        for block in contents:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in {"output_text", "text"}:
                text = block.get("text")
                if text:
                    collected.append(str(text))
    joined = "\n".join(part for part in collected if part).strip()
    return joined or json.dumps(output_entries, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-fetch OpenAI batch outputs from existing batch IDs.")
    parser.add_argument("--log-file", type=Path, help="Log file containing batch_xxx identifiers.")
    parser.add_argument(
        "--batch-id",
        dest="batch_ids",
        action="append",
        default=[],
        help="Explicit batch_id to fetch (repeatable). Overrides log parsing.",
    )
    parser.add_argument("--model-id", required=True, help="Model identifier (e.g., gpt-5).")
    parser.add_argument("--task-name", required=True, help="Benchmark task name, e.g., typeb_oeq.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for benchmark outputs.")
    parser.add_argument("--benchmark-root", default="/workspace", help="Benchmark root.")
    parser.add_argument("--data-root", default=None, help="Data root for MC/TF tasks.")
    parser.add_argument("--collect-name", default="fake_collect.csv", help="Collection CSV name.")
    parser.add_argument("--skip-real", action="store_true", help="Skip real samples when fetching detection data.")
    parser.add_argument("--skip-audio", action="store_true", help="Discard audio samples after fetching.")
    parser.add_argument("--max-samples", type=int, help="Limit samples per task (mirrors original batch run).")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max new tokens for wrapper context.")
    parser.add_argument("--dry-run", action="store_true", help="Only show batch statuses without writing outputs.")
    parser.add_argument("--watch", action="store_true", help="Poll unfinished batches until completion.")
    parser.add_argument(
        "--watch-interval",
        type=int,
        default=60,
        help="Seconds between polls when --watch is enabled (minimum 5s).",
    )
    args = parser.parse_args()

    explicit_ids = [bid.strip() for bid in args.batch_ids if bid and bid.strip()]
    parsed_ids = _parse_batch_ids(args.log_file) if args.log_file else []

    batch_ids: List[str] = []
    seen_ids: Set[str] = set()
    for candidate in explicit_ids or parsed_ids:
        if candidate and candidate not in seen_ids:
            batch_ids.append(candidate)
            seen_ids.add(candidate)

    if not batch_ids:
        raise SystemExit("No batch IDs found. Provide --batch-id or --log-file with batch entries.")

    wrapper = build_model_wrapper(model_id=args.model_id, max_new_tokens=args.max_new_tokens)
    wrapper.ensure_loaded()
    api_key = getattr(wrapper, "_api_key", None)
    if not api_key:
        raise SystemExit("Unable to determine OpenAI API key from wrapper.")

    samples = fetch_samples_for_task(
        args.task_name,
        max_samples=args.max_samples,
        data_root=args.data_root,
        collect_name=args.collect_name,
        skip_real=args.skip_real,
        benchmark_root=args.benchmark_root,
    )
    if args.skip_audio:
        before = len(samples)
        samples = [s for s in samples if (s.modality or "").lower() != "audio"]
        removed = before - len(samples)
        if removed:
            print(f"[INFO] Skip-audio removed {removed} sample(s).")

    sample_map: Dict[str, object] = {}
    for sample in samples:
        sample_map[str(sample.sample_id)] = sample

    if not sample_map:
        raise SystemExit("No samples available for mapping. Verify benchmark_root and filtering options.")

    writer = _OpenAIOutputWriter(
        task_name=args.task_name,
        output_dir=args.output_dir,
        wrapper=wrapper,
    )
    total_cost_usd = 0.0
    total_input_tokens = 0
    total_cached_input_tokens = 0
    total_output_tokens = 0
    samples_with_cost = 0
    processed_samples = 0
    processed_ids: Set[str] = set()

    for batch_id in batch_ids:
        print(f"[INFO] Fetching batch {batch_id} ...")
        output_file_id: Optional[str] = None
        status_payload: Dict[str, Any] = {}
        while True:
            status_payload = _get_batch(batch_id, api_key)
            status = status_payload.get("status")
            output_file_id = status_payload.get("output_file_id")
            if status == "completed" and output_file_id:
                break

            terminal_states = {"failed", "cancelled", "expired"}
            if status in terminal_states:
                print(f"[WARN] Batch {batch_id} ended with status={status}.")
                output_file_id = None
                break

            if not args.watch:
                print(f"[WARN] Batch {batch_id} not ready yet (status={status}). Skipping.")
                output_file_id = None
                break

            wait_time = max(args.watch_interval, 5)
            print(f"[INFO] Batch {batch_id} status={status}; waiting {wait_time}s before next poll.")
            time.sleep(wait_time)

        if not output_file_id:
            continue

        if args.dry_run:
            print(f"[INFO] Batch {batch_id} ready (output_file_id={output_file_id}).")
            continue

        lines = _download_file_lines(output_file_id, api_key)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping malformed JSON line in batch {batch_id}.")
                continue
            sample_id = str(record.get("custom_id"))
            if not sample_id:
                continue
            sample = sample_map.get(sample_id)
            if not sample:
                print(f"[WARN] Sample {sample_id} not found in current mapping; skipping.")
                continue
            if sample_id in processed_ids:
                print(f"[INFO] Sample {sample_id} already processed; skipping duplicate entry.")
                continue

            if record.get("error"):
                writer.record_failure(sample, f"[ERROR] {record['error']}")
                continue

            response_body = _extract_response_payload(record)
            outputs = response_body.get("output") or []
            response_text = _collect_response_text(outputs)

            usage_metadata = response_body.get("usage")
            cost_details = _estimate_cost_usd(wrapper.model_id, usage_metadata)
            cost_metadata = None
            if cost_details:
                total_cost_usd += cost_details["usd"]
                total_input_tokens += cost_details["input_tokens"]
                total_cached_input_tokens += cost_details.get("cached_tokens", 0)
                total_output_tokens += cost_details["output_tokens"]
                samples_with_cost += 1
                cost_metadata = {**cost_details, "usd": round(cost_details["usd"], 8)}

            writer.record_success(
                sample,
                response_text,
                usage_metadata=usage_metadata,
                cost_metadata=cost_metadata,
            )
            processed_ids.add(sample_id)
            processed_samples += 1

        print(f"[INFO] Batch {batch_id}: processed {processed_samples} cumulative sample(s).")

    if not args.dry_run:
        writer.finalize()
        if processed_samples:
            print(f"[INFO] Saved {processed_samples} sample(s) from {len(batch_ids)} batch(es).")
        else:
            print("[WARN] No samples were processed.")

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
            print("[INFO] No usage metadata available for cost summary.")


if __name__ == "__main__":
    main()
