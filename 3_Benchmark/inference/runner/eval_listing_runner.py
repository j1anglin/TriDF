from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

from inference.dataio.samples import TaskSample
from inference.runner.gemini_batch_runner import run_gemini_batch_job
from inference.wrappers.registry import build_model_wrapper
from listing_prompt import mapping_prompt as listing_prompt

PROMPT_CACHE_SENTINEL = (
    "The detailed Analysis Text for this sample will be provided in the next user message immediately after this "
    "shared instruction block. Do not begin extracting defect names until you have read that follow-up message."
)
PROMPT_CACHE_TEMPLATE = listing_prompt.format(RESPONSE=PROMPT_CACHE_SENTINEL)
PROMPT_CACHE_TTL_SECONDS = 6 * 60 * 60

SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]")


def _sanitize_token(text: str, fallback: str = "sample") -> str:
    token = SAFE_TOKEN_RE.sub("-", text.strip())
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token or fallback


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _has_likely_authentic_prefix(text: str) -> bool:
    cleaned = text.lstrip()
    if cleaned.lower().startswith("<s>"):
        cleaned = cleaned[3:].lstrip()
    prefix = cleaned[:120].lower()
    return "likely authentic" in prefix


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


def _collect_listing_samples(
    runs_root: Path,
    *,
    modalities: Sequence[str],
    max_samples: int | None,
    require_substring: str,
    source_model_filters: Sequence[str] | None,
    output_dir: Path,
    skip_existing: bool,
) -> Tuple[List[TaskSample], List[TaskSample]]:
    collected: List[TaskSample] = []
    auto_misclass: List[TaskSample] = []
    substring = require_substring.lower()
    filter_tokens = [token.strip().lower() for token in (source_model_filters or []) if token.strip()]

    def _matches_filter(model_name: str) -> bool:
        if not filter_tokens:
            return True
        name = model_name.lower()
        for token in filter_tokens:
            if token == name or token in name or name in token:
                return True
        return False

    for modality in modalities:
        modality_dir = runs_root / modality
        if not modality_dir.is_dir():
            continue
        for task_dir in sorted(p for p in modality_dir.iterdir() if p.is_dir()):
            if substring not in task_dir.name.lower():
                continue
            model_root = task_dir / "models"
            if not model_root.is_dir():
                continue
            for model_dir in sorted(p for p in model_root.iterdir() if p.is_dir()):
                if not _matches_filter(model_dir.name):
                    continue
                is_detection_task = task_dir.name.lower().startswith("typeb_oeq")
                requires_fake_label = is_detection_task
                for result_path in sorted(model_dir.glob("*.json*")):
                    result_rel_path = result_path.relative_to(runs_root)
                    dataset_name = result_rel_path.stem
                    sample_task = _sanitize_token(f"{modality}_{task_dir.name}_{dataset_name}")
                    for record in _iter_result_records(result_path):
                        response_text = str(record.get("response") or "").strip()
                        if not response_text:
                            continue
                        sample_meta = record.get("sample")
                        sample_label = ""
                        if isinstance(sample_meta, dict):
                            sample_label = str(sample_meta.get("label") or "")
                        if requires_fake_label and sample_label.lower() != "fake":
                            continue
                        original_sample_id = ""
                        if isinstance(sample_meta, dict):
                            original_sample_id = str(sample_meta.get("sample_id") or "")
                        sample_id = _sanitize_token(f"{model_dir.name}_{original_sample_id}_{result_path.stem}")
                        prompt_text = listing_prompt.format(RESPONSE=_escape_braces(response_text))
                        sample = TaskSample(
                            task=sample_task,
                            sample_id=sample_id,
                            modality="text",
                            prompt=prompt_text,
                            fake_path=str(result_path),
                            relative_fake_path=str(result_rel_path),
                            label="listing",
                            media_meta={
                                "source_model_id": record.get("model_id"),
                                "source_sample_task": (sample_meta or {}).get("task") if isinstance(sample_meta, dict) else None,
                                "source_sample_id": original_sample_id or None,
                                "source_modality": (sample_meta or {}).get("modality") if isinstance(sample_meta, dict) else modality,
                                "source_label": sample_label or None,
                                "analysis_text": response_text,
                            },
                        )
                        eval_rel_candidate = result_rel_path.parent / result_rel_path.stem
                        cleaned_parts = tuple(part for part in eval_rel_candidate.parts if part not in ("", ".", ".."))
                        if cleaned_parts:
                            eval_rel_dir = Path(*cleaned_parts)
                        else:
                            eval_rel_dir = eval_rel_candidate
                        target_filename = str(original_sample_id or result_rel_path.stem).strip() or result_rel_path.stem
                        if not target_filename.endswith(".json"):
                            target_filename = f"{target_filename}.json"
                        if not hasattr(sample, "extra_data") or sample.extra_data is None:
                            sample.extra_data = {}
                        sample.extra_data["eval_rel_dir"] = eval_rel_dir.as_posix()
                        sample.extra_data["eval_file_name"] = Path(target_filename).name
                        sample.extra_data.setdefault("prompt_cache_template", PROMPT_CACHE_TEMPLATE)
                        sample.extra_data.setdefault("prompt_cache_ttl", PROMPT_CACHE_TTL_SECONDS)
                        sample_output = output_dir / eval_rel_dir / sample.extra_data["eval_file_name"]
                        if is_detection_task and _has_likely_authentic_prefix(response_text):
                            if skip_existing and sample_output.exists():
                                continue
                            auto_misclass.append(sample)
                            continue
                        if skip_existing and sample_output.exists():
                            continue
                        collected.append(sample)
                        if max_samples is not None and len(collected) >= max_samples:
                            return collected, auto_misclass
    return collected, auto_misclass


def _write_auto_misclassification_outputs(samples: Sequence[TaskSample], output_root: Path) -> None:
    for sample in samples:
        extra_data = getattr(sample, "extra_data", {}) or {}
        rel_dir = Path(extra_data.get("eval_rel_dir", "."))
        file_name = extra_data.get("eval_file_name")
        if not file_name:
            continue
        target_dir = output_root / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / file_name
        payload = {
            "model_id": "auto-misclassification",
            "sample": sample.to_json(),
            "response": "Misclassification",
            "skip_reason": "Likely Authentic prefix detected; Gemini query skipped.",
        }
        target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract defect listings from model responses.")
    parser.add_argument("--runs-root", type=Path, default=Path("./runs_organized"))
    parser.add_argument("--output-dir", type=Path, default=Path("./runs/scoring/OEQ_score"))
    parser.add_argument("--modalities", nargs="*", default=("image", "video", "audio"))
    parser.add_argument("--task-substring", type=str, default="oe")
    parser.add_argument(
        "--source-models",
        nargs="*",
        default=None,
        help="Optional list of model directory names to include (e.g., gemini-2.5-pro internvl).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose eval_listing outputs already exist.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-id", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--max-parallel-batches", type=int, default=1)
    parser.add_argument("--task-name", type=str, default="eval_listing")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--api-key", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.api_key:
        os.environ.setdefault("GEMINI_API_KEY", args.api_key)
        os.environ.setdefault("GOOGLE_API_KEY", args.api_key)
        os.environ.setdefault("GOOGLE_GENERATIVE_AI_API_KEY", args.api_key)

    samples, auto_misclass_samples = _collect_listing_samples(
        args.runs_root,
        modalities=args.modalities,
        max_samples=args.max_samples,
        require_substring=args.task_substring,
        source_model_filters=args.source_models,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
    )
    if auto_misclass_samples:
        _write_auto_misclassification_outputs(auto_misclass_samples, args.output_dir)
    print(
        f"[INFO] Prepared {len(samples)} text-only analysis sample(s) for defect listing; "
        f"skipped {len(auto_misclass_samples)} Likely Authentic sample(s)."
    )
    if not samples:
        return

    wrapper = build_model_wrapper(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )

    run_gemini_batch_job(
        wrapper=wrapper,
        samples=samples,
        output_dir=args.output_dir,
        task_name=args.task_name,
        batch_size=args.batch_size,
        poll_interval=args.poll_interval,
        max_parallel_batches=args.max_parallel_batches,
    )


if __name__ == "__main__":
    main()
