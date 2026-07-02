#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate per-file and overall accuracy for JSON prediction logs.

Each JSON file should contain either:
- a list of dicts; or
- JSON Lines (one JSON object per line).

We look for:
- `response` (string) — first line starts with "Likely Authentic" or "Likely Manipulated".
- ground truth label in any of: `label`, `sample.label`, `gt`, or `ground_truth`.
  * "real"  -> expected "Likely Authentic"
  * "fake"  -> expected "Likely Manipulated"

Usage:
    python eval_accuracy.py /path/to/folder [-o accuracy_report.csv] [--recursive] [--no-strict] [--single-record-only]

Notes:
- By default, strict mode is enabled: records with missing/invalid response or label are counted as incorrect.
- With --no-strict, invalid records are skipped (not counted in the denominator).
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any


PRED_AUTH = "Likely Authentic"
PRED_FAKE = "Likely Manipulated"


def load_records(fp: Path) -> Tuple[List[dict], str, bool]:
    """Load a file that is either a JSON array, JSON object, or JSONL.

    Returns (records, fmt, had_invalid_lines).
    """
    text = fp.read_text(encoding="utf-8").strip()
    if not text:
        return [], "empty", False
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data, "json_array", False
        # If it's a dict, wrap it to keep behavior predictable.
        if isinstance(data, dict):
            return [data], "json_object", False
        return [], "unknown", False
    except json.JSONDecodeError:
        # JSON Lines fallback
        records: List[dict] = []
        had_invalid = False
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError:
                # Keep going; bad line is ignored.
                had_invalid = True
                continue
        return records, "jsonl", had_invalid


def save_records(fp: Path, records: List[dict], fmt: str) -> None:
    """Write records back to disk in the original format."""
    if fmt == "json_object":
        data: Any = records[0] if records else {}
        fp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    if fmt == "json_array":
        fp.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    if fmt == "jsonl":
        lines = [json.dumps(r, ensure_ascii=False) for r in records]
        fp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return


def get_label(rec: dict) -> Optional[str]:
    """Extract normalized GT label ('real' or 'fake')."""
    def _norm(s: Any) -> Optional[str]:
        if not isinstance(s, str):
            return None
        s = s.strip().lower()
        if s in ("real", "fake"):
            return s
        return None

    # common locations
    for key in ("label", "gt", "ground_truth"):
        v = rec.get(key)
        if (lab := _norm(v)) is not None:
            return lab

    # nested under sample
    sample = rec.get("sample")
    if isinstance(sample, dict):
        v = sample.get("label")
        if (lab := _norm(v)) is not None:
            return lab

    return None


def get_prediction(rec: dict) -> Optional[str]:
    """Return normalized prediction headline:
       'Likely Authentic' or 'Likely Manipulated'.

    Accepts cases where the prediction appears at the end of the response.
    """
    resp = rec.get("response")
    if not isinstance(resp, str) or not resp.strip():
        return None

    # Prefer the last occurrence to handle "Overall Assessment" at the end.
    lines = [ln.strip() for ln in resp.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        low = line.lower().lstrip("#").strip()
        if low.startswith("likely authentic") or "likely authentic" in low:
            return PRED_AUTH
        if low.startswith("likely manipulated") or "likely manipulated" in low:
            return PRED_FAKE
    return None


def expected_from_label(label: str) -> Optional[str]:
    if label == "real":
        return PRED_AUTH
    if label == "fake":
        return PRED_FAKE
    return None


def evaluate_file(
    fp: Path,
    strict: bool = True,
    single_record_only: bool = False,
    write_pred: bool = False,
    pred_field: str = "prediction",
    overwrite_pred: bool = False,
) -> Optional[Dict[str, Any]]:
    """Compute metrics for a single file.

    If `single_record_only` is True, files that contain multiple records (JSON array/JSONL)
    are treated as aggregate logs and skipped.
    """
    recs, fmt, had_invalid = load_records(fp)
    if single_record_only and len(recs) != 1:
        return None
    total = len(recs)
    evaluated = 0
    correct = 0
    skipped = 0
    wrote_pred = False

    for r in recs:
        gt = get_label(r)
        pred = get_prediction(r)
        exp = expected_from_label(gt) if gt is not None else None

        if write_pred and pred is not None:
            if overwrite_pred or pred_field not in r:
                r[pred_field] = pred
                wrote_pred = True

        if gt is None or pred is None or exp is None:
            if strict:
                # Count invalid as incorrect and evaluated
                evaluated += 1
            else:
                skipped += 1
            continue

        evaluated += 1
        if pred == exp:
            correct += 1

    denom = evaluated if not strict else max(evaluated, 1)
    accuracy = (correct / denom) if denom else 0.0

    if write_pred and wrote_pred and not had_invalid and fmt in ("json_object", "json_array", "jsonl"):
        save_records(fp, recs, fmt)

    return {
        "file": fp.name,
        "path": str(fp),
        "n_records": total,
        "n_evaluated": evaluated,
        "n_skipped": skipped,
        "n_correct": correct,
        "accuracy": accuracy,
    }


def main():
    ap = argparse.ArgumentParser(description="Compute per-file and overall accuracy from JSON logs.")
    ap.add_argument("data_dir", type=str, help="Folder containing *.json files.")
    ap.add_argument("-o", "--output", type=str, default="accuracy_report.csv", help="CSV path for the summary.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders for *.json files.")
    strict_group = ap.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Count invalid/missing records as incorrect (default).",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Skip invalid/missing records instead of counting them as incorrect.",
    )
    ap.add_argument("--quiet", action="store_true", help="Only print overall summary (no per-file lines).")
    ap.add_argument(
        "--single-record-only",
        action="store_true",
        help="Only evaluate JSON files that contain exactly 1 record; skip aggregate JSON/JSONL files.",
    )
    ap.add_argument(
        "--write-pred",
        action="store_true",
        help="Write parsed prediction headline into each record (default field: prediction).",
    )
    ap.add_argument(
        "--pred-field",
        type=str,
        default="prediction",
        help="Field name to store prediction headline (default: prediction).",
    )
    ap.add_argument(
        "--overwrite-pred",
        action="store_true",
        help="Overwrite existing prediction field if present.",
    )
    args = ap.parse_args()

    base = Path(args.data_dir)
    if not base.exists():
        raise SystemExit(f"Path not found: {base}")

    if args.recursive:
        files = sorted(p for p in base.rglob("*.json") if p.is_file())
    else:
        files = sorted(p for p in base.glob("*.json") if p.is_file())

    if not files:
        raise SystemExit("No JSON files found.")

    rows: List[Dict[str, Any]] = []
    overall_total = 0
    overall_eval = 0
    overall_correct = 0
    overall_skipped = 0
    skipped_files = 0

    for fp in files:
        res = evaluate_file(
            fp,
            strict=args.strict,
            single_record_only=args.single_record_only,
            write_pred=args.write_pred,
            pred_field=args.pred_field,
            overwrite_pred=args.overwrite_pred,
        )
        if res is None:
            skipped_files += 1
            continue
        rows.append(res)
        overall_total += res["n_records"]
        overall_eval += res["n_evaluated"]
        overall_correct += res["n_correct"]
        overall_skipped += res["n_skipped"]

    # Write CSV
    out = Path(args.output)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "path", "n_records", "n_evaluated", "n_skipped", "n_correct", "accuracy"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        # Overall row
        denom = overall_eval if not args.strict else max(overall_eval, 1)
        overall_acc = (overall_correct / denom) if denom else 0.0
        w.writerow({
            "file": "__OVERALL__",
            "path": "",
            "n_records": overall_total,
            "n_evaluated": overall_eval,
            "n_skipped": overall_skipped,
            "n_correct": overall_correct,
            "accuracy": overall_acc,
        })

    # Pretty print summary to stdout
    try:
        if not args.quiet:
            print("\nPer-file accuracy:\n")
            for r in rows:
                print(f"{r['file']:40s} | acc={r['accuracy']:.4f} | "
                      f"correct={r['n_correct']}/{r['n_evaluated']} (skipped={r['n_skipped']}, total={r['n_records']})")
        if skipped_files:
            print(f"\nSkipped aggregate/multi-record files: {skipped_files}")
        print("\nOverall:")
        denom = overall_eval if not args.strict else max(overall_eval, 1)
        overall_acc = (overall_correct / denom) if denom else 0.0
        print(f"accuracy={overall_acc:.4f} | correct={overall_correct}/{overall_eval} "
              f"(skipped={overall_skipped}, total={overall_total})")
        print(f"\nCSV written to: {out.resolve()}")
    except BrokenPipeError:
        # Allow piping to tools like `head` without crashing.
        sys.exit(0)


if __name__ == "__main__":
    main()
