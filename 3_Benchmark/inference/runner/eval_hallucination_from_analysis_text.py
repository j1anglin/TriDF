from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from inference.dataio.samples import ModelResponse, TaskSample
from inference.runner.claude_batch_runner import run_claude_batch_job
from inference.runner.common import _format_exception_chain, _needs_seed_retry, _set_all_seeds, sanitize_model_id
from inference.utils.torch_helpers import build_loading_kwargs, resolve_torch_dtype
from inference.runner.gemini_batch_runner import run_gemini_batch_job
from inference.runner.openai_batch_runner import run_openai_batch_job
from inference.wrappers.registry import build_model_wrapper
from inference.wrappers.claude import ClaudeBatchWrapper
from inference.wrappers.gemini import GeminiWrapper
from inference.wrappers.openai import OpenAIGPTBatchWrapper
from inference.utils.jsonio import write_json_atomic


def _load_mapping_prompt() -> str:
    try:
        from mapping_prompt import mapping_prompt  # type: ignore

        return mapping_prompt
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "benchmark" / "mapping_prompt.py",
        repo_root / "rebuttal_alignment_exp" / "mapping_prompt.py",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("mapping_prompt", candidate)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "mapping_prompt"):
                return str(getattr(module, "mapping_prompt"))

    raise RuntimeError("Unable to locate mapping_prompt.py for hallucination evaluation")


mapping_prompt = _load_mapping_prompt()

PROMPT_CACHE_SENTINEL = (
    "The detailed Analysis Text for this sample will be provided in the next user message immediately after this "
    "shared instruction block. Do not begin your checklist until you have read that follow-up message."
)
PROMPT_CACHE_TEMPLATE = mapping_prompt.format(RESPONSE=PROMPT_CACHE_SENTINEL)
PROMPT_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours ensures reuse within a run

SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]")


def _sanitize_token(text: str, fallback: str = "sample") -> str:
    token = SAFE_TOKEN_RE.sub("-", text.strip())
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token or fallback


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _is_under_path(path: Path, root: Optional[Path]) -> bool:
    if root is None:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sanitize_task_path(path: Path) -> str:
    parts = [p for p in path.parts if p not in ("", ".")]
    if not parts:
        return "analysis_text"
    cleaned = [_sanitize_token(part) for part in parts]
    cleaned = [part for part in cleaned if part]
    return "/".join(cleaned) if cleaned else "analysis_text"


def _iter_result_records(result_path: Path) -> Iterator[Dict[str, object]]:
    if result_path.suffix == ".jsonl":
        with result_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    else:
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, dict):
                    yield entry
        elif isinstance(payload, dict):
            yield payload


def _extract_analysis_text(record: Dict[str, object], *, analysis_field: str) -> Optional[str]:
    field = (analysis_field or "analysis_text").strip().lower()
    if field == "response":
        response_text = record.get("response")
        if response_text:
            return str(response_text)
        # Fall back to analysis_text if response is missing.
        field = "analysis_text"

    if field in ("analysis_text", "auto"):
        sample = record.get("sample")
        if isinstance(sample, dict):
            direct_text = sample.get("analysis_text")
            if direct_text:
                return str(direct_text)
            media_meta = sample.get("media_meta")
            if isinstance(media_meta, dict):
                meta_text = media_meta.get("analysis_text")
                if meta_text:
                    return str(meta_text)

        direct_text = record.get("analysis_text")
        if direct_text:
            return str(direct_text)

    if field == "auto":
        response_text = record.get("response")
        if response_text:
            return str(response_text)

    return None


def _output_path_for(sample: TaskSample, output_dir: Path) -> Path:
    task_name = sample.task or "unknown_task"
    return output_dir / task_name / f"{sample.sample_id}.json"


def _collect_analysis_samples(
    input_root: Path,
    *,
    modalities: Sequence[str],
    max_samples: Optional[int],
    output_dir: Path,
    skip_existing: bool,
    analysis_field: str,
) -> Tuple[List[TaskSample], int, int]:
    collected: List[TaskSample] = []
    seen_ids = set()
    missing_text = 0
    scanned = 0
    modalities_set = {m.lower() for m in modalities if m}

    def _matches_modalities(rel_parts: Sequence[str]) -> bool:
        if not modalities_set:
            return True
        for part in (p.lower() for p in rel_parts):
            if part in modalities_set:
                return True
            if "image" in modalities_set and part.startswith("img"):
                return True
            if "video" in modalities_set and part.startswith("vid"):
                return True
            if "audio" in modalities_set and part.startswith("aud"):
                return True
        return False

    for result_path in sorted(input_root.rglob("*.json")) + sorted(input_root.rglob("*.jsonl")):
        if _is_under_path(result_path, output_dir):
            continue
        if not result_path.is_file():
            continue
        rel_path = result_path.relative_to(input_root)
        if modalities_set and not _matches_modalities(rel_path.parts):
            continue

        for record_index, record in enumerate(_iter_result_records(result_path)):
            scanned += 1
            analysis_text = _extract_analysis_text(record, analysis_field=analysis_field)
            if not analysis_text:
                missing_text += 1
                continue
            analysis_text = str(analysis_text).strip()
            if not analysis_text:
                missing_text += 1
                continue

            sample_meta = record.get("sample") if isinstance(record, dict) else None
            # raw_sample_id = None
            # if isinstance(sample_meta, dict):
            #     raw_sample_id = sample_meta.get("sample_id")
            # if raw_sample_id is None:
            #     raw_sample_id = rel_path.stem
            #     if record_index:
            #         raw_sample_id = f"{raw_sample_id}-{record_index}"
            # sample_id = _sanitize_token(str(raw_sample_id), fallback=rel_path.stem or "sample")

            # raw_task = None
            # if isinstance(sample_meta, dict):
            #     raw_task = sample_meta.get("task")
            # if raw_task:
            #     task = _sanitize_task_path(Path(str(raw_task)))
            # else:
            #     task = _sanitize_task_path(rel_path.parent)
            
            raw_sample_id = None
            if isinstance(sample_meta, dict):
                raw_sample_id = sample_meta.get("sample_id")
            if raw_sample_id is None:
                raw_sample_id = rel_path.stem
                if record_index:
                    raw_sample_id = f"{raw_sample_id}-{record_index}"

            raw_task = None
            if isinstance(sample_meta, dict):
                raw_task = sample_meta.get("task")
            if raw_task:
                task = _sanitize_task_path(Path(str(raw_task)))
            else:
                task = _sanitize_task_path(rel_path.parent)

            source_sample_id = _sanitize_token(
                str(raw_sample_id),
                fallback=rel_path.stem or "sample",
            )

            # OpenAI Batch custom_id must be unique in the batch.
            # TriDF sample_id is unique only within each task, not globally.
            sample_id = _sanitize_token(
                f"{task}__{source_sample_id}",
                fallback=source_sample_id,
            )

            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)

            media_meta: Dict[str, object] = {}
            if isinstance(sample_meta, dict):
                meta = sample_meta.get("media_meta")
                if isinstance(meta, dict):
                    media_meta.update(meta)
                if sample_meta.get("modality") and "source_modality" not in media_meta:
                    media_meta["source_modality"] = sample_meta.get("modality")
                if sample_meta.get("task") and "source_sample_task" not in media_meta:
                    media_meta["source_sample_task"] = sample_meta.get("task")
                if sample_meta.get("sample_id") and "source_sample_id" not in media_meta:
                    media_meta["source_sample_id"] = sample_meta.get("sample_id")
                if sample_meta.get("label") and "source_label" not in media_meta:
                    media_meta["source_label"] = sample_meta.get("label")

            if record.get("model_id") and "source_model_id" not in media_meta:
                media_meta["source_model_id"] = record.get("model_id")

            if isinstance(sample_meta, dict):
                orig_rel = sample_meta.get("relative_fake_path") or sample_meta.get("relative_real_path") or sample_meta.get("relative_path")
                if orig_rel and "source_relative_path" not in media_meta:
                    media_meta["source_relative_path"] = str(orig_rel)

            media_meta.setdefault("analysis_text", analysis_text)
            media_meta.setdefault("source_record_path", str(result_path))
            media_meta.setdefault("source_relative_path", rel_path.as_posix())

            if "source_sample_id" not in media_meta:
                media_meta["source_sample_id"] = source_sample_id

            prompt_text = mapping_prompt.format(RESPONSE=_escape_braces(analysis_text))
            sample = TaskSample(
                task=task,
                sample_id=sample_id,
                modality="text",
                prompt=prompt_text,
                fake_path=str(result_path),
                relative_fake_path=str(rel_path),
                label="analysis",
                media_meta=media_meta,
            )
            sample.extra_data = {
                "prompt_cache_template": PROMPT_CACHE_TEMPLATE,
                "prompt_cache_ttl": PROMPT_CACHE_TTL_SECONDS,
                "prompt_cache_analysis_text": analysis_text,
            }

            if skip_existing and _output_path_for(sample, output_dir).exists():
                continue

            collected.append(sample)
            if max_samples is not None and len(collected) >= max_samples:
                return collected, missing_text, scanned

    return collected, missing_text, scanned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hallucination artifacts using analysis_text fields from JSON records."
    )
    parser.add_argument("--input-root", type=Path, default=Path("./rebuttal_alignment_exp"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--modalities", nargs="*", default=("image", "video", "audio"))
    parser.add_argument(
        "--analysis-field",
        type=str,
        default="analysis_text",
        choices=("analysis_text", "response", "auto"),
        help="Field to use for mapping input (analysis_text, response, auto=analysis_text then response).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        choices=("gemini", "openai", "claude", "local"),
        help="Backend to use for mapping (gemini, openai, claude, local).",
    )
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--max-parallel-batches", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--per-gpu-max-memory-gib", type=int, default=None)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        help="OpenAI batch completion window (ignored for Gemini/Claude).",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch API usage (Gemini only); run per-sample requests sequentially.",
    )
    return parser.parse_args()


def _run_local_inference(
    samples: Sequence[TaskSample],
    *,
    wrapper,
    model_id: str,
    output_dir: Path,
    max_retries: int = 2,
    base_seed: int = 42,
    skip_existing: bool = False,
) -> None:
    def _local_sample_path(task_name: str, sample_id: str) -> Path:
        safe_task = _sanitize_token(task_name or "eval_hallucination")
        safe_sample = _sanitize_token(sample_id or "sample")
        return output_dir / f"{safe_task}__{safe_sample}.json"

    def _local_summary_path(task_name: str) -> Path:
        safe_task = _sanitize_token(task_name or "eval_hallucination")
        return output_dir / f"summary__{safe_task}.json"

    tasks: Dict[str, List[TaskSample]] = {}
    for sample in samples:
        task_name = sample.task or "eval_hallucination"
        tasks.setdefault(task_name, []).append(sample)

    output_dir.mkdir(parents=True, exist_ok=True)

    for task_name, task_samples in tasks.items():
        task_results: List[Dict[str, object]] = []

        for sample in task_samples:
            sample_path = _local_sample_path(task_name, sample.sample_id)
            if sample_path.exists() and skip_existing:
                try:
                    with sample_path.open("r", encoding="utf-8") as handle:
                        existing_record = json.load(handle)
                    task_results.append(existing_record)
                    print(f"[INFO] Skipping existing result: {sample_path}")
                    continue
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Failed to read existing result {sample_path}: {exc}. Recomputing...")

            attempt = 0
            answer = ""
            start = time.time()
            while True:
                wrapper.system_hint = None
                wrapper.user_prefix = None
                wrapper.generation_overrides = {}
                if hasattr(wrapper, "last_usage"):
                    wrapper.last_usage = None
                _set_all_seeds(base_seed + attempt)

                try:
                    answer = wrapper.generate(sample)
                except Exception as exc:  # noqa: BLE001
                    answer = f"[ERROR] {_format_exception_chain(exc)}"

                retry_reason = _needs_seed_retry(model_id, sample, answer)
                if retry_reason is None or attempt >= max_retries:
                    break
                attempt += 1
                print(f"[INFO] {retry_reason} → retry with new seed (attempt={attempt})")

            latency_ms = (time.time() - start) * 1000.0
            record = ModelResponse(
                model_id=model_id,
                sample=sample,
                response=answer,
                latency_ms=latency_ms,
                fallback_count=attempt,
                final_seed=base_seed + attempt,
                system_hint=wrapper.system_hint,
                usage_metadata=getattr(wrapper, "last_usage", None),
            ).to_json()
            if hasattr(wrapper, "last_usage"):
                wrapper.last_usage = None
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(record, sample_path)
            task_results.append(record)
            print(f"[INFO] Wrote per-sample result: {sample_path}")

        summary_path = _local_summary_path(task_name)
        write_json_atomic(task_results, summary_path)
        print(f"[INFO] Saved task summary: {summary_path}")


def _run_nobatch_inference(
    samples: Sequence[TaskSample],
    *,
    wrapper,
    model_id: str,
    output_dir: Path,
    max_retries: int = 2,
    base_seed: int = 42,
    skip_existing: bool = False,
) -> None:
    tasks: Dict[str, List[TaskSample]] = {}
    for sample in samples:
        task_name = sample.task or "eval_hallucination"
        tasks.setdefault(task_name, []).append(sample)

    output_dir.mkdir(parents=True, exist_ok=True)

    for task_name, task_samples in tasks.items():
        task_results: List[Dict[str, object]] = []

        for sample in task_samples:
            sample_path = _output_path_for(sample, output_dir)
            if sample_path.exists() and skip_existing:
                try:
                    with sample_path.open("r", encoding="utf-8") as handle:
                        existing_record = json.load(handle)
                    task_results.append(existing_record)
                    print(f"[INFO] Skipping existing result: {sample_path}")
                    continue
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Failed to read existing result {sample_path}: {exc}. Recomputing...")

            attempt = 0
            answer = ""
            start = time.time()
            while True:
                wrapper.system_hint = None
                wrapper.user_prefix = None
                wrapper.generation_overrides = {}
                if hasattr(wrapper, "last_usage"):
                    wrapper.last_usage = None
                _set_all_seeds(base_seed + attempt)

                try:
                    answer = wrapper.generate(sample)
                except Exception as exc:  # noqa: BLE001
                    answer = f"[ERROR] {_format_exception_chain(exc)}"

                retry_reason = _needs_seed_retry(model_id, sample, answer)
                if retry_reason is None or attempt >= max_retries:
                    break
                attempt += 1
                print(f"[INFO] {retry_reason} → retry with new seed (attempt={attempt})")

            latency_ms = (time.time() - start) * 1000.0
            record = ModelResponse(
                model_id=model_id,
                sample=sample,
                response=answer,
                latency_ms=latency_ms,
                fallback_count=attempt,
                final_seed=base_seed + attempt,
                system_hint=wrapper.system_hint,
                usage_metadata=getattr(wrapper, "last_usage", None),
            ).to_json()
            if hasattr(wrapper, "last_usage"):
                wrapper.last_usage = None
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(record, sample_path)
            task_results.append(record)
            print(f"[INFO] Wrote per-sample result: {sample_path}")

        summary_path = output_dir / f"{task_name or 'eval_hallucination'}.json"
        write_json_atomic(task_results, summary_path)
        print(f"[INFO] Saved task summary: {summary_path}")


def main() -> None:
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_root / "eval_hallucination"

    args.input_root = args.input_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend = args.backend.lower().strip()
    if not args.model_id:
        if backend == "openai":
            args.model_id = "gpt-5-mini"
        elif backend == "claude":
            args.model_id = "claude-sonnet-4-5"
        elif backend == "local":
            args.model_id = "Qwen/Qwen3-8B"
        else:
            args.model_id = "gemini-2.5-flash"

    if args.api_key:
        if backend == "openai":
            os.environ.setdefault("OPENAI_API_KEY", args.api_key)
            os.environ.setdefault("OPENAI_API_KEY_GPT5", args.api_key)
            os.environ.setdefault("OPENAI_API_KEY_BATCH", args.api_key)
        elif backend == "claude":
            os.environ.setdefault("ANTHROPIC_API_KEY", args.api_key)
            os.environ.setdefault("CLAUDE_API_KEY", args.api_key)
            os.environ.setdefault("CLAUDE_OPUS_API_KEY", args.api_key)
        else:
            os.environ.setdefault("GEMINI_API_KEY", args.api_key)
            os.environ.setdefault("GOOGLE_API_KEY", args.api_key)
            os.environ.setdefault("GOOGLE_GENERATIVE_AI_API_KEY", args.api_key)

    samples, missing_text, scanned = _collect_analysis_samples(
        args.input_root,
        modalities=args.modalities,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        skip_existing=False if backend == "local" else args.skip_existing,
        analysis_field=args.analysis_field,
    )

    print(
        f"[INFO] Prepared {len(samples)} sample(s) from analysis_text; "
        f"scanned {scanned} record(s), skipped {missing_text} missing analysis_text."
    )
    if not samples:
        return

    load_kwargs = None
    torch_dtype = None
    if backend == "local":
        torch_dtype = resolve_torch_dtype(args.torch_dtype)
        load_kwargs = build_loading_kwargs(
            device_map=args.device_map,
            gpus=None,
            per_gpu_max_memory_gib=args.per_gpu_max_memory_gib,
            flash_attn=args.flash_attn,
            cache_dir=args.cache_dir,
            offline=args.offline,
        )

    wrapper = build_model_wrapper(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=torch_dtype,
        load_kwargs=load_kwargs,
    )

    if backend == "local":
        _run_local_inference(
            samples,
            wrapper=wrapper,
            model_id=args.model_id,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
        )
    elif args.no_batch:
        if backend != "gemini":
            raise ValueError("--no-batch is only supported for Gemini backend.")
        if not isinstance(wrapper, GeminiWrapper):
            raise ValueError(f"Backend gemini requires Gemini wrapper, got {type(wrapper).__name__}.")
        _run_nobatch_inference(
            samples,
            wrapper=wrapper,
            model_id=args.model_id,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
        )
    elif backend == "openai":
        if not isinstance(wrapper, OpenAIGPTBatchWrapper):
            raise ValueError(f"Backend openai requires OpenAI wrapper, got {type(wrapper).__name__}.")
        run_openai_batch_job(
            wrapper=wrapper,
            samples=samples,
            output_dir=args.output_dir,
            task_name="eval_hallucination",
            batch_size=args.batch_size,
            poll_interval=args.poll_interval,
            completion_window=args.completion_window,
            max_parallel_batches=args.max_parallel_batches,
            skip_completed=args.skip_existing,
        )
    elif backend == "claude":
        if not isinstance(wrapper, ClaudeBatchWrapper):
            raise ValueError(f"Backend claude requires Claude wrapper, got {type(wrapper).__name__}.")
        run_claude_batch_job(
            wrapper=wrapper,
            samples=samples,
            output_dir=args.output_dir,
            task_name="eval_hallucination",
            batch_size=args.batch_size,
            poll_interval=args.poll_interval,
            max_parallel_batches=args.max_parallel_batches,
            skip_completed=args.skip_existing,
        )
    else:
        if not isinstance(wrapper, GeminiWrapper):
            raise ValueError(f"Backend gemini requires Gemini wrapper, got {type(wrapper).__name__}.")
        run_gemini_batch_job(
            wrapper=wrapper,
            samples=samples,
            output_dir=args.output_dir,
            task_name="eval_hallucination",
            batch_size=args.batch_size,
            poll_interval=args.poll_interval,
            max_parallel_batches=args.max_parallel_batches,
        )


if __name__ == "__main__":
    main()