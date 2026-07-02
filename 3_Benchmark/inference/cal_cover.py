#!/usr/bin/env python3
"""
Compute COVER metrics (macro & micro) for eval_hallucination responses.

- Macro COVER: average per-sample coverage (matched / annotated for that sample).
- Micro COVER: total matched artifacts divided by total annotated artifacts.
Ground truth annotations are sourced from 2_GT_Final/2_GT_Final while responses
come from eval_hallucination/<modality>/<benchmark_task>/models/<model>.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

BASE_COLUMNS = {"question_id", "sample_path"}


def parse_bool_truth(value: Optional[str]) -> bool:
    """Interpret ground-truth cell values."""
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in {"false", "0", "none", "null"}:
        return False
    if text in {"[]", "[ ]"}:
        return False
    return True


def parse_bool_prediction(value: Optional[str]) -> bool:
    """Interpret model prediction values; they should already be True/False."""
    if value is None:
        return False
    return str(value).strip().lower() == "true"


@dataclass(frozen=True)
class GTKey:
    task: str
    is_commercial: bool


class GroundTruthStore:
    """Loads and caches ground-truth CSV rows on demand."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: Dict[GTKey, Optional["GroundTruth"]] = {}

    def get(self, task: str, is_commercial: bool) -> Optional["GroundTruth"]:
        key = GTKey(task=task, is_commercial=is_commercial)
        if key not in self._cache:
            path = self._resolve_path(task, is_commercial)
            if not path:
                print(
                    f"[WARN] No ground-truth CSV found for task={task!r} "
                    f"(is_commercial={is_commercial}).",
                    file=sys.stderr,
                )
                self._cache[key] = None
            else:
                self._cache[key] = self._load_csv(path)
        return self._cache[key]

    def _resolve_path(self, task: str, is_commercial: bool) -> Optional[Path]:
        if is_commercial:
            com_path = self.root / f"{task}_com.csv"
            return com_path if com_path.is_file() else None
        base_path = self.root / f"{task}.csv"
        return base_path if base_path.is_file() else None

    def _load_csv(self, path: Path) -> "GroundTruth":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Ground-truth file {path} lacks headers.")
            artifacts = [c for c in reader.fieldnames if c not in BASE_COLUMNS]
            by_sample: Dict[str, Dict[str, bool]] = {}
            by_qid: Dict[str, Dict[str, bool]] = {}
            for row in reader:
                bool_map = {art: parse_bool_truth(row.get(art)) for art in artifacts}
                sample_path = (row.get("sample_path") or "").strip()
                question_id = (row.get("question_id") or "").strip()
                if sample_path:
                    by_sample[sample_path] = bool_map
                if question_id:
                    by_qid[question_id] = bool_map
        return GroundTruth(path=path, artifacts=artifacts, by_sample=by_sample, by_qid=by_qid)


@dataclass
class GroundTruth:
    path: Path
    artifacts: List[str]
    by_sample: Dict[str, Dict[str, bool]]
    by_qid: Dict[str, Dict[str, bool]]


class CoverCalculator:
    def __init__(
        self,
        eval_root: Path,
        gt_root: Path,
        modalities: Optional[Iterable[str]] = None,
        models: Optional[Iterable[str]] = None,
        benchmark_tasks: Optional[Iterable[str]] = None,
        benchmark_aliases: Optional[Dict[str, str]] = None,
        output_path: Optional[Path] = None,
        verbose: bool = False,
    ) -> None:
        self.eval_root = eval_root
        self.gt_store = GroundTruthStore(gt_root)
        self.modalities = set(modalities) if modalities else None
        self.models = set(models) if models else None
        self.benchmark_tasks = set(benchmark_tasks) if benchmark_tasks else None
        self.benchmark_aliases = dict(benchmark_aliases or {})
        self.output_path = output_path
        self.verbose = verbose
        self.stats = defaultdict(
            lambda: {
                "matches": 0,
                "gt": 0,
                "samples": 0,
                "coverage_sum": 0.0,
                "coverage_count": 0,
            }
        )
        self._missing_rows: set[Tuple[Path, str]] = set()

    def run(self) -> List[Dict[str, object]]:
        if not self.eval_root.is_dir():
            raise FileNotFoundError(f"Eval root {self.eval_root} not found.")

        any_file = False
        for modality, benchmark, model, csv_path in self._iter_eval_csvs():
            any_file = True
            self._process_file(modality, benchmark, model, csv_path)

        if not any_file:
            print("[WARN] No eval CSV files discovered with current filters.", file=sys.stderr)

        summary = self._build_summary()
        if self.output_path:
            self._write_summary(summary)
        return summary

    def _iter_eval_csvs(self):
        for modality_dir in sorted(p for p in self.eval_root.iterdir() if p.is_dir()):
            modality = modality_dir.name
            if self.modalities and modality not in self.modalities:
                continue
            for benchmark_dir in sorted(p for p in modality_dir.iterdir() if p.is_dir()):
                benchmark = benchmark_dir.name
                if not self._benchmark_in_filters(benchmark):
                    continue
                models_dir = benchmark_dir / "models"
                if not models_dir.is_dir():
                    continue
                for model_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
                    model = model_dir.name
                    if self.models and model not in self.models:
                        continue
                    for csv_path in sorted(model_dir.glob("*.csv")):
                        yield modality, benchmark, model, csv_path

    def _process_file(self, modality: str, benchmark: str, model: str, csv_path: Path) -> None:
        task_name = csv_path.stem
        is_commercial = "commercial" in benchmark
        normalized_benchmark = self._normalize_benchmark(benchmark)
        gt_info = self.gt_store.get(task_name, is_commercial)
        if not gt_info:
            return
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                print(f"[WARN] Eval file {csv_path} has no headers; skipping.", file=sys.stderr)
                return
            eval_artifacts = [c for c in reader.fieldnames if c not in BASE_COLUMNS]
            artifacts = [c for c in eval_artifacts if c in gt_info.artifacts]
            if not artifacts:
                print(
                    f"[WARN] No overlapping artifact columns between {csv_path} and {gt_info.path}.",
                    file=sys.stderr,
                )
                return
            file_matches = 0
            file_gt = 0
            file_cov_sum = 0.0
            file_cov_count = 0
            file_samples = 0
            for row in reader:
                sample_path = (row.get("sample_path") or "").strip()
                question_id = str(row.get("question_id") or "").strip()
                gt_row = gt_info.by_sample.get(sample_path) or gt_info.by_qid.get(question_id)
                if not gt_row:
                    key = (csv_path, sample_path or question_id)
                    if key not in self._missing_rows:
                        self._missing_rows.add(key)
                        print(
                            f"[WARN] Missing GT row for sample={sample_path or question_id} "
                            f"in task={task_name}.",
                            file=sys.stderr,
                        )
                    continue
                gt_count = 0
                match_count = 0
                for artifact in artifacts:
                    gt_has = gt_row.get(artifact, False)
                    if gt_has:
                        gt_count += 1
                    if gt_has and parse_bool_prediction(row.get(artifact)):
                        match_count += 1
                file_gt += gt_count
                file_matches += match_count
                file_samples += 1
                key_stats = self.stats[(modality, normalized_benchmark, model)]
                key_stats["gt"] += gt_count
                key_stats["matches"] += match_count
                key_stats["samples"] += 1
                if gt_count > 0:
                    coverage = match_count / gt_count
                    file_cov_sum += coverage
                    file_cov_count += 1
                    key_stats["coverage_sum"] += coverage
                    key_stats["coverage_count"] += 1

            if self.verbose:
                macro_cover = file_cov_sum / file_cov_count if file_cov_count else 0.0
                micro_cover = file_matches / file_gt if file_gt else 0.0
                print(
                    f"[INFO] {csv_path}: macro={macro_cover:.4f} micro={micro_cover:.4f} "
                    f"(matched={file_matches}, gt={file_gt}, samples={file_samples}, "
                    f"sample_cov_count={file_cov_count})",
                    file=sys.stderr,
                )

    def _normalize_benchmark(self, benchmark: str) -> str:
        return self.benchmark_aliases.get(benchmark, benchmark)

    def _benchmark_in_filters(self, benchmark: str) -> bool:
        if not self.benchmark_tasks:
            return True
        if benchmark in self.benchmark_tasks:
            return True
        normalized = self._normalize_benchmark(benchmark)
        return normalized in self.benchmark_tasks

    def _build_summary(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for (modality, benchmark, model), stat in sorted(self.stats.items()):
            gt_total = stat["gt"]
            matched = stat["matches"]
            coverage_count = stat["coverage_count"]
            coverage_sum = stat["coverage_sum"]
            macro_cover = coverage_sum / coverage_count if coverage_count else 0.0
            micro_cover = matched / gt_total if gt_total else 0.0
            rows.append(
                {
                    "modality": modality,
                    "benchmark_task": benchmark,
                    "model": model,
                    "gt_artifacts": gt_total,
                    "matched_artifacts": matched,
                    "macro_cover": macro_cover,
                    "micro_cover": micro_cover,
                    "processed_samples": stat["samples"],
                }
            )
        return rows

    def _write_summary(self, rows: List[Dict[str, object]]) -> None:
        header = [
            "modality",
            "benchmark_task",
            "model",
            "gt_artifacts",
            "matched_artifacts",
            "macro_cover",
            "micro_cover",
            "processed_samples",
        ]
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"[OK] Summary written to {self.output_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate COVER metrics from eval_hallucination CSVs.")
    parser.add_argument("--eval-root", type=Path, default=Path("eval_hallucination"), help="Root of eval_hallucination CSVs.")
    parser.add_argument("--gt-root", type=Path, default=Path("2_GT_Final/2_GT_Final"), help="Root of ground-truth annotation CSVs.")
    parser.add_argument("--modality", action="append", help="Filter by modality (image/audio/video).")
    parser.add_argument("--model", action="append", help="Filter by model name (can repeat).")
    parser.add_argument("--benchmark-task", action="append", help="Filter by benchmark task folder (e.g., typea_oeq).")
    parser.add_argument("--output", type=Path, help="Optional CSV output path for the summary table.")
    parser.add_argument("--verbose", action="store_true", help="Print per-file coverage stats to stderr.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calculator = CoverCalculator(
        eval_root=args.eval_root,
        gt_root=args.gt_root,
        modalities=args.modality,
        models=args.model,
        benchmark_tasks=args.benchmark_task,
        benchmark_aliases=None,
        output_path=args.output,
        verbose=args.verbose,
    )
    summary_rows = calculator.run()
    if not summary_rows:
        print("No COVER results to display.", file=sys.stderr)
        return
    print(
        "modality\tbenchmark_task\tmodel\tgt_artifacts\tmatched_artifacts\tmacro_cover\tmicro_cover\tprocessed_samples"
    )
    for row in summary_rows:
        print(
            f"{row['modality']}\t{row['benchmark_task']}\t{row['model']}\t"
            f"{row['gt_artifacts']}\t{row['matched_artifacts']}\t"
            f"{row['macro_cover']:.4f}\t{row['micro_cover']:.4f}\t"
            f"{row['processed_samples']}"
        )


if __name__ == "__main__":
    main()