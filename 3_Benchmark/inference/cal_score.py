#!/usr/bin/env python3
"""
Compute COVER, CHAIR, Hal, and F-beta metrics strictly aligned with arXiv:2512.10652v2.

Metrics:
- Macro COVER: Average per-sample coverage (Recall of artifacts).
- CHAIR: Fraction of hallucinatory artifacts (1 - Precision), computed only for
         samples with at least one ground-truth artifact.
         Penalty: 1.0 if the mapped artifact list is empty, or if a Type-B
         response classifies a fake sample as Likely Authentic.
- Hal: Percentage of scored samples containing any hallucination (CHAIR > 0),
       computed only for samples with at least one ground-truth artifact.
- F^0.5: Weighted harmonic mean of Precision (1-CHAIR) and Recall (Cover), beta=0.5.

Ground truth annotations are sourced from 2_GT_Final/2_GT_Final.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

BASE_COLUMNS = {"question_id", "sample_path", "detection_prediction", "source_label"}


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


def is_likely_authentic(value: Optional[str]) -> bool:
    text = str(value or "").strip().lower().rstrip(".")
    return text.startswith("likely authentic")


def is_real_source_label(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() == "real"

_REAL_SAMPLE_RE = re.compile(r"(?:^|[\/])real[-_]|^real[-_]", re.IGNORECASE)


def _is_real_sample(sample_path: str, question_id: str) -> bool:
    return bool(_REAL_SAMPLE_RE.search(sample_path) or _REAL_SAMPLE_RE.search(question_id))



class GroundTruthStore:
    """Loads and caches ground-truth CSV rows on demand."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: Dict[str, Optional["GroundTruth"]] = {}

    def get(self, task: str) -> Optional["GroundTruth"]:
        """Return merged GT for *task*, combining regular and _com files."""
        if task not in self._cache:
            base_path = self.root / f"{task}.csv"
            com_path = self.root / f"{task}_com.csv"
            base = self._load_csv(base_path) if base_path.is_file() else None
            com = self._load_csv(com_path) if com_path.is_file() else None
            if base is None and com is None:
                print(
                    f"[WARN] No ground-truth CSV found for task={task!r}.",
                    file=sys.stderr,
                )
                self._cache[task] = None
            elif base is None:
                self._cache[task] = com
            elif com is None:
                self._cache[task] = base
            else:
                self._cache[task] = GroundTruth(
                    path=base.path,
                    artifacts=base.artifacts,
                    by_sample={**base.by_sample, **com.by_sample},
                    by_qid={**base.by_qid, **com.by_qid},
                )
        return self._cache[task]

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
        # Stats accumulation structure
        self.stats = defaultdict(
            lambda: {
                "matches": 0,
                "gt": 0,
                "predictions": 0,
                "samples": 0,
                "coverage_sum": 0.0,
                "coverage_count": 0, # Denominator for Cover (only samples with |Y_art| > 0)
                "chair_sum": 0.0,
                "chair_count": 0,    # Denominator for CHAIR/Hal (only samples with |Y_art| > 0)
                "hal_count": 0,      # Count of samples with Hal=1
                "fbeta_sum": 0.0,
                "fbeta_count": 0,    # Denominator for F-beta (only samples with |Y_art| > 0)
            }
        )
        self._missing_rows: set[Tuple[Path, str]] = set()
        self._skipped_real = 0

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
        normalized_benchmark = self._normalize_benchmark(benchmark)
        gt_info = self.gt_store.get(task_name)
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

            file_stats = {
                "matches": 0, "gt": 0, "predictions": 0, "samples": 0,
                "cov_sum": 0.0, "cov_count": 0, "chair_sum": 0.0, "chair_count": 0
            }

            for row in reader:
                sample_path = (row.get("sample_path") or "").strip()
                question_id = str(row.get("question_id") or "").strip()
                gt_row = gt_info.by_sample.get(sample_path) or gt_info.by_qid.get(question_id)
                
                if not gt_row:
                    if _is_real_sample(sample_path, question_id):
                        self._skipped_real += 1
                        continue
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
                pred_count = 0

                for artifact in artifacts:
                    gt_has = gt_row.get(artifact, False)
                    pred_has = parse_bool_prediction(row.get(artifact))
                    
                    if gt_has:
                        gt_count += 1
                    if pred_has:
                        pred_count += 1
                    if gt_has and pred_has:
                        match_count += 1

                # Update accumulated counts
                key_stats = self.stats[(modality, normalized_benchmark, model)]
                key_stats["gt"] += gt_count
                key_stats["matches"] += match_count
                key_stats["predictions"] += pred_count
                key_stats["samples"] += 1
                
                file_stats["samples"] += 1
                file_stats["gt"] += gt_count
                file_stats["matches"] += match_count
                file_stats["predictions"] += pred_count

                # --- Metric Calculations Per Sample (Aligned with Paper) ---
                
                has_artifact_gt = gt_count > 0
                is_fake_sample = not is_real_source_label(row.get("source_label"))
                uses_detection_headline = normalized_benchmark == "typeb_oeq"
                classified_fake_as_real = (
                    uses_detection_headline
                    and is_fake_sample
                    and is_likely_authentic(row.get("detection_prediction"))
                )
                chair = 0.0

                # 1. Coverage (Recall)
                # Defined only when annotated artifacts exist (|Y_art| > 0).
                coverage = 0.0
                if has_artifact_gt:
                    coverage = match_count / gt_count
                    key_stats["coverage_sum"] += coverage
                    key_stats["coverage_count"] += 1
                    file_stats["cov_sum"] += coverage
                    file_stats["cov_count"] += 1

                # 2. CHAIR (1 - Precision)
                # Applies only when annotated artifacts exist (|Y_art| > 0).
                if has_artifact_gt:
                    if classified_fake_as_real:
                        chair = 1.0
                    elif pred_count == 0:
                        chair = 1.0
                    else:
                        chair = 1.0 - (match_count / pred_count)

                    key_stats["chair_sum"] += chair
                    key_stats["chair_count"] += 1
                    file_stats["chair_sum"] += chair
                    file_stats["chair_count"] += 1

                # 3. Hal (Binary Hallucination Indicator)
                if has_artifact_gt and chair > 0:
                    key_stats["hal_count"] += 1

                # 4. F-Beta (beta=0.5)
                # Weighted harmonic mean. Computed only when Coverage is defined.
                if has_artifact_gt:
                    precision = 1.0 - chair
                    beta_sq = 0.25 # 0.5^2
                    numerator = (1 + beta_sq) * precision * coverage
                    denominator = (beta_sq * precision) + coverage
                    
                    f_beta = 0.0
                    if denominator > 0:
                        f_beta = numerator / denominator
                    
                    key_stats["fbeta_sum"] += f_beta
                    key_stats["fbeta_count"] += 1

            if self.verbose:
                macro_cover = file_stats["cov_sum"] / file_stats["cov_count"] if file_stats["cov_count"] else 0.0
                macro_chair = file_stats["chair_sum"] / file_stats["chair_count"] if file_stats["chair_count"] else 0.0
                print(
                    f"[INFO] {csv_path}: macro_cov={macro_cover:.4f} macro_chair={macro_chair:.4f} "
                    f"(matched={file_stats['matches']}, gt={file_stats['gt']}, samples={file_stats['samples']})",
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
            
            # Coverage (Macro average over Fake samples)
            coverage_count = stat["coverage_count"]
            macro_cover = stat["coverage_sum"] / coverage_count if coverage_count else 0.0
            
            # Micro Cover (Aggregated matched / Aggregated GT)
            micro_cover = matched / gt_total if gt_total else 0.0
            
            # CHAIR (Macro average over samples with |Y_art| > 0)
            chair_count = stat["chair_count"]
            macro_chair = stat["chair_sum"] / chair_count if chair_count else 0.0
            
            # Hal Rate (Percentage of samples with |Y_art| > 0 and hallucination)
            samples_processed = stat["samples"]
            hal_rate = stat["hal_count"] / chair_count if chair_count else 0.0
            
            # F-Beta (Macro average over Fake samples)
            fbeta_count = stat["fbeta_count"]
            macro_fbeta = stat["fbeta_sum"] / fbeta_count if fbeta_count else 0.0

            rows.append(
                {
                    "modality": modality,
                    "benchmark_task": benchmark,
                    "model": model,
                    "gt_artifacts": gt_total,
                    "matched_artifacts": matched,
                    "macro_cover": macro_cover,
                    "micro_cover": micro_cover,
                    "macro_chair": macro_chair,
                    "hal_rate": hal_rate,
                    "macro_f_beta": macro_fbeta,
                    "processed_samples": samples_processed,
                }
            )
        return rows

    def build_combined_summary(self) -> List[Dict[str, object]]:
        """Aggregate stats across all modalities for the same benchmark/model."""
        combined: Dict[Tuple[str, str], Dict[str, float]] = {}
        for (modality, benchmark, model), stat in self.stats.items():
            key = (benchmark, model)
            if key not in combined:
                combined[key] = {k: 0.0 for k in stat.keys()}
            for k, v in stat.items():
                combined[key][k] += v

        rows: List[Dict[str, object]] = []
        for (benchmark, model), stat in sorted(combined.items()):
            gt_total = stat["gt"]
            matched = stat["matches"]

            coverage_count = stat["coverage_count"]
            macro_cover = stat["coverage_sum"] / coverage_count if coverage_count else 0.0

            micro_cover = matched / gt_total if gt_total else 0.0

            chair_count = stat["chair_count"]
            macro_chair = stat["chair_sum"] / chair_count if chair_count else 0.0

            samples_processed = stat["samples"]
            hal_rate = stat["hal_count"] / chair_count if chair_count else 0.0

            fbeta_count = stat["fbeta_count"]
            macro_fbeta = stat["fbeta_sum"] / fbeta_count if fbeta_count else 0.0

            rows.append(
                {
                    "modality": "all",
                    "benchmark_task": benchmark,
                    "model": model,
                    "gt_artifacts": gt_total,
                    "matched_artifacts": matched,
                    "macro_cover": macro_cover,
                    "micro_cover": micro_cover,
                    "macro_chair": macro_chair,
                    "hal_rate": hal_rate,
                    "macro_f_beta": macro_fbeta,
                    "processed_samples": samples_processed,
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
            "macro_chair",
            "hal_rate",
            "macro_f_beta",
            "processed_samples",
        ]
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"[OK] Summary written to {self.output_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate COVER, CHAIR, and F-beta metrics.")
    parser.add_argument("--eval-root", type=Path, default=Path("eval_hallucination"), help="Root of eval_hallucination CSVs.")
    parser.add_argument("--gt-root", type=Path, default=Path("2_GT_Final/2_GT_Final"), help="Root of ground-truth annotation CSVs.")
    parser.add_argument("--modality", action="append", help="Filter by modality (image/audio/video).")
    parser.add_argument("--model", action="append", help="Filter by model name (can repeat).")
    parser.add_argument("--benchmark-task", action="append", help="Filter by benchmark task folder.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    parser.add_argument("--verbose", action="store_true", help="Print per-file coverage stats to stderr.")
    parser.add_argument(
        "--combine-modalities",
        action="store_true",
        help="Add a combined row (modality=all) per benchmark/model across all modalities.",
    )
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
    if args.combine_modalities:
        summary_rows.extend(calculator.build_combined_summary())
    if not summary_rows:
        print("No results to display.", file=sys.stderr)
        return
    
    headers = [
        "modality", "benchmark_task", "model", 
        "gt_artifacts", "matched", 
        "macro_cov", "micro_cov", 
        "chair", "hal", "f_0.5", 
        "samples"
    ]
    print("\t".join(headers))
    for row in summary_rows:
        print(
            f"{row['modality']}\t{row['benchmark_task']}\t{row['model']}\t"
            f"{row['gt_artifacts']}\t{row['matched_artifacts']}\t"
            f"{row['macro_cover']:.4f}\t{row['micro_cover']:.4f}\t"
            f"{row['macro_chair']:.4f}\t{row['hal_rate']:.4f}\t"
            f"{row['macro_f_beta']:.4f}\t"
            f"{row['processed_samples']}"
        )


if __name__ == "__main__":
    main()