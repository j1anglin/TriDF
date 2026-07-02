#!/usr/bin/env python3
"""
Aggregate COVER scores for every modality/model combination.

This script leverages inference.cal_cover.CoverCalculator to iterate through
the eval_hallucination outputs, compute coverage per modality, and store the
results in per-modality CSVs plus an optional combined summary file.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from inference.cal_cover import CoverCalculator

SUMMARY_HEADER = [
    "modality",
    "benchmark_task",
    "model",
    "gt_artifacts",
    "matched_artifacts",
    "macro_cover",
    "micro_cover",
    "processed_samples",
]

BENCHMARK_ALIAS_MAP = {
    "typeb_oeq_commercial": "typeb_oeq",
    "typea_oeq_commercial": "typea_oeq",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and record COVER for every modality/model."
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("eval_hallucination"),
        help="Root directory that contains modality subfolders (default: %(default)s).",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("2_GT_Final/2_GT_Final"),
        help="Directory that holds ground-truth CSVs (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cover_reports"),
        help="Directory where per-modality CSVs will be written (default: %(default)s).",
    )
    parser.add_argument(
        "--modality",
        action="append",
        help="Restrict computation to specific modalities (can repeat).",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Restrict to specific model names (can repeat).",
    )
    parser.add_argument(
        "--benchmark-task",
        action="append",
        help="Restrict to specific benchmark directories (e.g., typeb_oeq).",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="cover_all_modalities.csv",
        help="Filename for the combined summary inside output-dir (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose COVER logging (passes through to CoverCalculator).",
    )
    return parser.parse_args()


def discover_modalities(eval_root: Path, filters: Optional[Iterable[str]]) -> List[str]:
    if filters:
        return sorted({m for m in filters})
    modalities = [
        path.name
        for path in sorted(eval_root.iterdir())
        if path.is_dir()
    ]
    return modalities


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: List[Dict[str, object]], modality: str) -> None:
    if not rows:
        print(f"[WARN] No COVER data produced for modality={modality}.", file=sys.stderr)
        return
    print(f"=== {modality} ===")
    print("\t".join(SUMMARY_HEADER))
    for row in rows:
        print(
            f"{row['modality']}\t{row['benchmark_task']}\t{row['model']}\t"
            f"{row['gt_artifacts']}\t{row['matched_artifacts']}\t"
            f"{row['macro_cover']:.4f}\t{row['micro_cover']:.4f}\t"
            f"{row['processed_samples']}"
        )


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Eval root {eval_root} does not exist.")

    modalities = discover_modalities(eval_root, args.modality)
    if not modalities:
        print("[WARN] No modalities found to process.", file=sys.stderr)
        return

    combined_rows: List[Dict[str, object]] = []

    for modality in modalities:
        calculator = CoverCalculator(
            eval_root=eval_root,
            gt_root=args.gt_root,
            modalities=[modality],
            models=args.model,
            benchmark_tasks=args.benchmark_task,
            benchmark_aliases=BENCHMARK_ALIAS_MAP,
            output_path=None,
            verbose=args.verbose,
        )
        modality_rows = calculator.run()
        if not modality_rows:
            print(
                f"[WARN] No COVER rows computed for modality={modality}.",
                file=sys.stderr,
            )
            continue

        combined_rows.extend(modality_rows)
        print_summary(modality_rows, modality)

        modality_file = args.output_dir / f"cover_{modality}.csv"
        write_summary_csv(modality_file, modality_rows)
        print(f"[OK] Saved modality summary → {modality_file}", file=sys.stderr)

    if combined_rows:
        combined_path = args.output_dir / args.summary_name
        write_summary_csv(combined_path, combined_rows)
        print(f"[OK] Saved combined summary → {combined_path}", file=sys.stderr)
    else:
        print("[WARN] No COVER rows generated; combined summary skipped.", file=sys.stderr)


if __name__ == "__main__":
    main()
