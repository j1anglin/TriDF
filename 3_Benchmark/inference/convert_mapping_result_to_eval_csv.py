#!/usr/bin/env python3
"""
Convert mapping_result JSON/JSONL outputs into eval_hallucination CSVs.

Expected eval_hallucination layout:
  <output-root>/<modality>/<benchmark>/models/<model>/<task>.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# Add project root to path to allow imports when invoked as a script.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from inference.eval_accuracy import get_prediction
from inference.runner.vote_parser_robustness import ARTIFACTS, parse_artifact_map


METADATA_COLUMNS = ["detection_prediction", "source_label"]


_ID_NUMBER_RE = re.compile(r"\d+")


def _infer_numeric_id(text: str) -> str | None:
    numbers = _ID_NUMBER_RE.findall(text or "")
    if not numbers:
        return None
    return numbers[-1]
def _iter_records(path: Path) -> Iterator[Dict[str, object]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
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
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, dict):
                    yield entry
        elif isinstance(payload, dict):
            yield payload


def _guess_task(path: Path, sample: Optional[Dict[str, object]]) -> str:
    if isinstance(sample, dict):
        task = sample.get("task")
        if task:
            return str(task)
        media_meta = sample.get("media_meta")
        if isinstance(media_meta, dict):
            meta_task = media_meta.get("source_sample_task")
            if meta_task:
                return str(meta_task)
    parent = path.parent.name
    if parent and parent != path.root:
        return parent
    return path.stem or "unknown_task"


def _extract_ids(
    path: Path,
    sample: Optional[Dict[str, object]],
    record_index: int,
    *,
    prefer_source_id: bool,
) -> Tuple[str, str]:
    sample_id = ""
    relative_path = ""
    if isinstance(sample, dict):
        raw_id = sample.get("sample_id")
        if raw_id is not None:
            sample_id = str(raw_id)
        rel_path = sample.get("relative_fake_path")
        if rel_path:
            relative_path = str(rel_path)
        media_meta = sample.get("media_meta")
        if isinstance(media_meta, dict):
            meta_id = media_meta.get("source_sample_id")
            if meta_id:
                meta_id = str(meta_id)
                if prefer_source_id:
                    sample_id = meta_id
                elif not sample_id:
                    sample_id = meta_id
            meta_rel = media_meta.get("source_relative_path")
            if meta_rel:
                meta_rel = str(meta_rel)
                if prefer_source_id:
                    relative_path = meta_rel
                elif not relative_path:
                    relative_path = meta_rel
    if not sample_id:
        sample_id = f"{path.stem}-{record_index}"
    return sample_id, relative_path


def _load_source_relative_path(path: Path) -> Optional[str]:
    for record in _iter_records(path):
        if not isinstance(record, dict):
            continue
        sample = record.get("sample")
        if isinstance(sample, dict):
            rel = sample.get("relative_fake_path") or sample.get("relative_real_path") or sample.get("relative_path")
            if rel:
                return str(rel)
        break
    return None


def _load_source_metadata(path: Path) -> Dict[str, str]:
    metadata = {"detection_prediction": "", "source_label": ""}
    for record in _iter_records(path):
        if not isinstance(record, dict):
            continue

        prediction = get_prediction(record)
        if prediction:
            metadata["detection_prediction"] = prediction

        sample = record.get("sample")
        if isinstance(sample, dict):
            source_label = sample.get("label")
            if source_label:
                metadata["source_label"] = str(source_label)
        break
    return metadata


def _extract_source_metadata(sample: Optional[Dict[str, object]]) -> Dict[str, str]:
    metadata = {"detection_prediction": "", "source_label": ""}
    if not isinstance(sample, dict):
        return metadata

    media_meta = sample.get("media_meta")
    if isinstance(media_meta, dict):
        analysis_text = media_meta.get("analysis_text")
        prediction = get_prediction({"response": analysis_text}) if analysis_text else None
        if prediction:
            metadata["detection_prediction"] = prediction

        source_label = media_meta.get("source_label")
        if source_label:
            metadata["source_label"] = str(source_label)

        source_record_path = media_meta.get("source_record_path")
        if source_record_path:
            source_path = Path(str(source_record_path))
            if source_path.is_file():
                source_metadata = _load_source_metadata(source_path)
                for key, value in source_metadata.items():
                    if value and not metadata[key]:
                        metadata[key] = value

    return metadata


def _candidate_source_paths(input_root: Path, model: str, rel_path: str) -> List[Path]:
    rel = Path(rel_path)
    if rel.is_absolute():
        return [rel]

    candidates: List[Path] = [input_root / rel]

    if input_root.name == model and input_root.parent.name == "mapping_result":
        benchmark_root = input_root.parent.parent
        candidates.append(benchmark_root / model / rel)
        candidates.append(benchmark_root / rel)
    else:
        candidates.append(input_root.parent / model / rel)
        candidates.append(input_root.parent / rel)
        candidates.append(input_root.parent.parent / model / rel)
        candidates.append(input_root.parent.parent / rel)
    return candidates


def _resolve_source_relative_path(
    sample: Optional[Dict[str, object]],
    input_root: Path,
    model: str,
    path_hint: str,
    cache: Dict[Path, Optional[str]],
) -> Optional[str]:
    if not isinstance(sample, dict):
        return None
    media_meta = sample.get("media_meta")
    if isinstance(media_meta, dict):
        source_record_path = media_meta.get("source_record_path")
        if source_record_path:
            source_path = Path(str(source_record_path))
            if source_path.is_file():
                if source_path not in cache:
                    cache[source_path] = _load_source_relative_path(source_path)
                return cache[source_path]
        source_rel = media_meta.get("source_relative_path")
        if source_rel:
            for candidate in _candidate_source_paths(input_root, model, str(source_rel)):
                if candidate.is_file():
                    if candidate not in cache:
                        cache[candidate] = _load_source_relative_path(candidate)
                    return cache[candidate]

    if path_hint and path_hint.lower().endswith((".json", ".jsonl")):
        for candidate in _candidate_source_paths(input_root, model, path_hint):
            if candidate.is_file():
                if candidate not in cache:
                    cache[candidate] = _load_source_relative_path(candidate)
                return cache[candidate]
    return None


def _iter_input_files(input_root: Path, output_root: Path) -> List[Path]:
    files = list(input_root.rglob("*.json")) + list(input_root.rglob("*.jsonl"))
    out = []
    for path in files:
        if output_root in path.parents:
            continue
        if path.name.startswith("summary__"):
            continue
        if path.name.endswith("_input.jsonl"):
            continue
        if path.is_file():
            out.append(path)
    return sorted(out)


def _write_task_csv(
    output_root: Path,
    modality: str,
    benchmark: str,
    model: str,
    task: str,
    rows: Iterable[Dict[str, object]],
) -> Path:
    import csv

    out_dir = output_root / modality / benchmark / "models" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}.csv"
    header = ["question_id", "sample_path"] + METADATA_COLUMNS + list(ARTIFACTS)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert mapping_result JSON/JSONL outputs into eval_hallucination CSVs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing mapping_result JSON/JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("eval_hallucination"),
        help="Root directory to write eval_hallucination CSVs.",
    )
    parser.add_argument(
        "--modality",
        nargs="+",
        default=["image"],
        help="Modality folder name(s) in eval_hallucination (default: image).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="rebuttal_cot_exp",
        help="Benchmark folder name in eval_hallucination.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to store under models/<model>.",
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Restrict to specific task names (can repeat).",
    )
    parser.add_argument(
        "--prefer-source-id",
        action="store_true",
        help="Prefer media_meta.source_sample_id/source_relative_path over sample_id/relative_fake_path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse inputs and report counts without writing CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    task_filters = {t for t in (args.task or []) if t}
    per_task_rows: Dict[str, "OrderedDict[str, Dict[str, object]]"] = defaultdict(OrderedDict)
    duplicates = 0
    total_records = 0
    source_path_cache: Dict[Path, Optional[str]] = {}

    for path in _iter_input_files(input_root, output_root):
        for idx, record in enumerate(_iter_records(path)):
            total_records += 1
            sample = record.get("sample") if isinstance(record, dict) else None
            task = _guess_task(path, sample if isinstance(sample, dict) else None)
            if task_filters and task not in task_filters:
                continue
            sample_id, relative_path = _extract_ids(
                path,
                sample if isinstance(sample, dict) else None,
                idx,
                prefer_source_id=args.prefer_source_id,
            )
            resolved_path = _resolve_source_relative_path(
                sample if isinstance(sample, dict) else None,
                input_root,
                args.model,
                relative_path,
                source_path_cache,
            )
            if resolved_path:
                relative_path = resolved_path
            response = record.get("response") if isinstance(record, dict) else ""
            parsed = parse_artifact_map(str(response or ""))
            source_metadata = _extract_source_metadata(sample if isinstance(sample, dict) else None)
            row = {
                "question_id": str(sample_id),
                "sample_path": str(relative_path),
                **source_metadata,
            }
            for art in ARTIFACTS:
                row[art] = bool(parsed.get(art, False))
            key = str(sample_id) or str(relative_path) or f"{path}:{idx}"
            if key in per_task_rows[task]:
                duplicates += 1
                continue
            per_task_rows[task][key] = row

    if not per_task_rows:
        print("[WARN] No rows produced; check input-root or task filters.", file=sys.stderr)
        return

    modalities = [m for m in args.modality if m]
    if not modalities:
        modalities = ["image"]

    _TASK_PREFIX_TO_MODALITY = {
        "img_": "image",
        "vid_": "video",
        "aud_": "audio",
    }

    for task, rows_by_key in per_task_rows.items():
        rows = rows_by_key.values()
        if args.dry_run:
            print(f"[DRY-RUN] task={task} rows={len(rows_by_key)}")
            continue
        # Determine modality from task name prefix; fall back to --modality list.
        task_modalities = [
            mod
            for prefix, mod in _TASK_PREFIX_TO_MODALITY.items()
            if task.startswith(prefix)
        ]
        if not task_modalities:
            task_modalities = modalities
        for modality in task_modalities:
            out_path = _write_task_csv(
                output_root=output_root,
                modality=modality,
                benchmark=args.benchmark,
                model=args.model,
                task=task,
                rows=rows,
            )
            print(f"[OK] wrote {out_path} ({len(rows_by_key)} rows)")

    if duplicates:
        print(f"[WARN] {duplicates} duplicate sample_id rows skipped.", file=sys.stderr)
    print(f"[INFO] scanned {total_records} record(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
