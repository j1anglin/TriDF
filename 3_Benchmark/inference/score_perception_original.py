#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_MCQ_LETTERS = tuple("ABCDE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score original TRIDF perception MC/TF JSONL outputs against question JSON ground_truth fields.",
    )
    parser.add_argument("--benchmark", choices=("mc", "tf"), required=True)
    parser.add_argument("--questions-dir", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--details-out", type=Path, default=None)
    parser.add_argument(
        "--include-unmatched-details",
        action="store_true",
        help="Include unmatched prediction records in details JSONL.",
    )
    return parser.parse_args()


def load_json_any(path: Path) -> List[object]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        items: List[object] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "questions", "data", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return []


def iter_prediction_records(root: Path) -> Iterator[Tuple[Path, int, dict]]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict) and "response" in record:
                        yield path, line_no, record
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "response" in payload:
            yield path, 1, payload
        elif isinstance(payload, list):
            for idx, item in enumerate(payload):
                if isinstance(item, dict) and "response" in item:
                    yield path, idx, item


def normalize_question_modality(value: object) -> str:
    text = str(value or "").strip().lower()
    return {"img": "image", "vid": "video", "aud": "audio"}.get(text, text)


def make_question_id(stem: str, index: int) -> str:
    return f"{stem}-{index}"


def load_questions(questions_dir: Path) -> tuple[Dict[Tuple[str, int], dict], Dict[str, dict]]:
    by_file_index: Dict[Tuple[str, int], dict] = {}
    by_question_id: Dict[str, dict] = {}

    for path in sorted(questions_dir.glob("*.json")):
        payload = load_json_any(path)
        for idx, raw in enumerate(payload):
            if not isinstance(raw, dict):
                continue
            question_id = make_question_id(path.stem, idx)
            row = {
                "question_id": question_id,
                "question_file": path.name,
                "question_stem": path.stem,
                "question_index": idx,
                "task": raw.get("task") or "unknown_task",
                "modality": normalize_question_modality(raw.get("modality")),
                "question_type": raw.get("question_type") or "unknown",
                "artifact_type": raw.get("artifact_type") or "unknown",
                "artifact": raw.get("artifact"),
                "question": raw.get("question"),
                "sample_path": raw.get("sample_path"),
                "options": raw.get("options"),
                "ground_truth": raw.get("ground_truth"),
            }
            by_file_index[(path.name, idx)] = row
            by_question_id[question_id] = row

    return by_file_index, by_question_id


def extract_prediction_key(record: dict) -> tuple[Optional[Tuple[str, int]], Optional[str]]:
    raw_index = record.get("question_index")
    raw_file = record.get("question_file")
    if raw_file is not None and raw_index is not None:
        try:
            return (Path(str(raw_file)).name, int(raw_index)), None
        except (TypeError, ValueError):
            pass

    sample = record.get("sample")
    if isinstance(sample, dict):
        sample_id = str(sample.get("sample_id") or "").strip()
        if sample_id:
            return None, sample_id

    question_id = str(record.get("question_id") or "").strip()
    if question_id:
        return None, question_id
    return None, None


def _ordered_letters(letters: Iterable[str]) -> List[str]:
    allowed = {str(letter).strip().upper() for letter in letters if str(letter).strip()}
    return [letter for letter in LETTERS if letter in allowed]


def mc_option_letters(question: dict) -> List[str]:
    question_text = str(question.get("question") or "")
    parsed = re.findall(r"^\s*([A-Z])\.", question_text, flags=re.M)
    if parsed:
        return _ordered_letters(parsed)

    options = question.get("options")
    if isinstance(options, list) and options:
        # Original TRIDF MCQ JSON stores artifact options only; the prompt adds
        # one final "None of the options are correct" choice.
        return list(LETTERS[: min(len(options) + 1, len(LETTERS))])

    gt = normalize_mc_answer(question.get("ground_truth"), DEFAULT_MCQ_LETTERS)
    if gt:
        max_index = max(LETTERS.index(letter) for letter in gt if letter in LETTERS)
        return list(LETTERS[: max(max_index + 1, len(DEFAULT_MCQ_LETTERS))])
    return list(DEFAULT_MCQ_LETTERS)


def normalize_mc_answer(value: object, allowed_letters: Optional[Sequence[str]] = None) -> Optional[List[str]]:
    allowed = set(allowed_letters or DEFAULT_MCQ_LETTERS)
    if isinstance(value, list):
        letters = [str(item).strip().upper() for item in value if str(item).strip()]
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        head = text.splitlines()[0].strip()
        if not head:
            return None
        none_selected = bool(re.search(r"\bnone\b", head, flags=re.IGNORECASE))
        cleaned = re.sub(r"^(answer|final answer|prediction)\s*[:：]\s*", "", head, flags=re.IGNORECASE)
        if "," in cleaned:
            letters = [token.strip().upper().strip(".;:()[]{}") for token in cleaned.split(",")]
        else:
            letters = re.findall(r"(?<![A-Za-z])[A-Z](?![A-Za-z])", cleaned.upper())
        if none_selected:
            letters.append("E")
    else:
        return None

    normalized = _ordered_letters(letter for letter in letters if letter in allowed)
    return normalized


def score_mc_answer(prediction: List[str], ground_truth: List[str], option_letters: Sequence[str]) -> float:
    option_set = set(option_letters)
    gt_set = {letter for letter in ground_truth if letter in option_set}
    pred_set = {letter for letter in prediction if letter in option_set}
    m = len(option_set)
    k = len(gt_set)
    if m == 0:
        return 0.0

    correct_weight = (1.0 / k) if k else 0.0
    incorrect_weight = (1.0 / (m - k)) if m > k else 0.0
    correct_selected = len(pred_set & gt_set)
    incorrect_selected = len(pred_set - gt_set)
    return (correct_selected * correct_weight) - (incorrect_selected * incorrect_weight)


def normalize_tf_answer(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return None
    head = text.splitlines()[0].strip()
    match = re.match(r"^(yes|no)\b[\\s\\S]*", head)
    if not match:
        return None
    return match.group(1) == "yes"


def normalize_answer(benchmark: str, value: object):
    if benchmark == "mc":
        return normalize_mc_answer(value)
    return normalize_tf_answer(value)


def new_group_stats() -> dict:
    return {
        "total_questions": 0,
        "matched_predictions": 0,
        "valid_predictions": 0,
        "correct": 0,
        "score_sum": 0.0,
        "invalid_predictions": 0,
        "missing_predictions": 0,
    }


def add_question_count(groups: dict, question: dict) -> None:
    for group_name, key in (
        ("by_task", question["task"]),
        ("by_modality", question["modality"]),
        ("by_question_type", question["question_type"]),
        ("by_artifact_type", question["artifact_type"]),
        ("by_question_file", question["question_file"]),
    ):
        groups[group_name][str(key)]["total_questions"] += 1


def add_prediction_count(groups: dict, question: dict, *, valid: bool, correct: bool, score_value: float) -> None:
    for group_name, key in (
        ("by_task", question["task"]),
        ("by_modality", question["modality"]),
        ("by_question_type", question["question_type"]),
        ("by_artifact_type", question["artifact_type"]),
        ("by_question_file", question["question_file"]),
    ):
        stats = groups[group_name][str(key)]
        stats["matched_predictions"] += 1
        if valid:
            stats["valid_predictions"] += 1
        else:
            stats["invalid_predictions"] += 1
        if correct:
            stats["correct"] += 1
        stats["score_sum"] += score_value


def finalize_stats(stats: dict) -> dict:
    matched = stats.get("matched_predictions", 0)
    valid = stats.get("valid_predictions", 0)
    total = stats.get("total_questions", 0)
    out = dict(stats)
    score_sum = float(out.get("score_sum", out["correct"]))
    out["accuracy"] = (score_sum / matched) if matched else 0.0
    out["valid_accuracy"] = (score_sum / valid) if valid else 0.0
    out["coverage_accuracy"] = (score_sum / total) if total else 0.0
    return out


def finalize_groups(groups: dict) -> dict:
    finalized: dict = {}
    for group_name, values in groups.items():
        rows = []
        for key, stats in sorted(values.items()):
            row = {"name": key, **finalize_stats(stats)}
            row["missing_predictions"] = row["total_questions"] - row["matched_predictions"]
            rows.append(row)
        finalized[group_name] = rows
    return finalized


def score(
    *,
    benchmark: str,
    questions_dir: Path,
    predictions_root: Path,
    details_out: Optional[Path],
    include_unmatched_details: bool,
) -> dict:
    by_file_index, by_question_id = load_questions(questions_dir)
    groups = {
        "by_task": defaultdict(new_group_stats),
        "by_modality": defaultdict(new_group_stats),
        "by_question_type": defaultdict(new_group_stats),
        "by_artifact_type": defaultdict(new_group_stats),
        "by_question_file": defaultdict(new_group_stats),
    }
    for question in by_question_id.values():
        add_question_count(groups, question)

    stats = {
        "benchmark": benchmark,
        "questions_dir": str(questions_dir),
        "predictions_root": str(predictions_root),
        "total_questions": len(by_question_id),
        "total_predictions": 0,
        "matched_predictions": 0,
        "valid_predictions": 0,
        "correct": 0,
        "score_sum": 0.0,
        "invalid_predictions": 0,
        "missing_answers": 0,
        "missing_prediction_key": 0,
        "duplicate_predictions_skipped": 0,
    }
    seen_question_ids: set[str] = set()

    details_handle = None
    if details_out:
        details_out.parent.mkdir(parents=True, exist_ok=True)
        details_handle = details_out.open("w", encoding="utf-8")

    try:
        for pred_path, record_index, record in iter_prediction_records(predictions_root):
            stats["total_predictions"] += 1
            file_key, question_id = extract_prediction_key(record)
            question = by_file_index.get(file_key) if file_key else None
            if question is None and question_id:
                question = by_question_id.get(question_id)
            if question is None:
                if question_id is None and file_key is None:
                    stats["missing_prediction_key"] += 1
                else:
                    stats["missing_answers"] += 1
                if details_handle and include_unmatched_details:
                    details_handle.write(
                        json.dumps(
                            {
                                "matched": False,
                                "prediction_file": str(pred_path),
                                "record_index": record_index,
                                "question_key": list(file_key) if file_key else question_id,
                                "response": record.get("response"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            qid = str(question["question_id"])
            if qid in seen_question_ids:
                stats["duplicate_predictions_skipped"] += 1
                continue
            seen_question_ids.add(qid)

            if benchmark == "mc":
                options = mc_option_letters(question)
                pred = normalize_mc_answer(record.get("response"), options)
                gt = normalize_mc_answer(question.get("ground_truth"), options)
                score_value = score_mc_answer(pred, gt, options) if pred is not None and gt is not None else 0.0
            else:
                options = None
                pred = normalize_tf_answer(record.get("response"))
                gt = normalize_tf_answer(question.get("ground_truth"))
                score_value = 1.0 if pred is not None and gt is not None and pred == gt else 0.0
            valid = pred is not None and gt is not None
            correct = bool(valid and pred == gt)

            stats["matched_predictions"] += 1
            if valid:
                stats["valid_predictions"] += 1
            else:
                stats["invalid_predictions"] += 1
            if correct:
                stats["correct"] += 1
            stats["score_sum"] += score_value

            add_prediction_count(groups, question, valid=valid, correct=correct, score_value=score_value)

            if details_handle:
                details_handle.write(
                    json.dumps(
                        {
                            "matched": True,
                            "prediction_file": str(pred_path),
                            "record_index": record_index,
                            "question_id": qid,
                            "question_file": question["question_file"],
                            "question_index": question["question_index"],
                            "task": question["task"],
                            "modality": question["modality"],
                            "artifact_type": question["artifact_type"],
                            "sample_path": question["sample_path"],
                            "ground_truth": gt,
                            "prediction": pred,
                            "score": score_value,
                            "option_letters": options,
                            "correct": correct,
                            "response": record.get("response"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if details_handle:
            details_handle.close()

    stats["missing_predictions"] = stats["total_questions"] - stats["matched_predictions"]
    summary = finalize_stats(stats)
    summary["groups"] = finalize_groups(groups)
    return summary


def main() -> None:
    args = parse_args()
    summary = score(
        benchmark=args.benchmark,
        questions_dir=args.questions_dir.resolve(),
        predictions_root=args.predictions_root.resolve(),
        details_out=args.details_out.resolve() if args.details_out else None,
        include_unmatched_details=args.include_unmatched_details,
    )
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
