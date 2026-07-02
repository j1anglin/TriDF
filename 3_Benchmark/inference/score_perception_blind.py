#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_MCQ_LETTERS = tuple("ABCDE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score blind perception MC/TF outputs by question_id.")
    parser.add_argument("--benchmark", choices=["mc", "tf"], required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def iter_prediction_records(root: Path) -> Iterator[dict]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict) and "response" in record:
                        yield record
            continue
        if path.suffix != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "response" in payload:
            yield payload
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and "response" in item:
                    yield item


def load_answers(path: Path) -> Dict[str, dict]:
    answers: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id") or "").strip()
            if not question_id:
                continue
            answers[question_id] = row
    return answers


def extract_question_id(record: dict) -> str:
    question_id = str(record.get("question_id") or "").strip()
    if question_id:
        return question_id
    sample = record.get("sample")
    if isinstance(sample, dict):
        question_id = str(sample.get("question_id") or sample.get("sample_id") or "").strip()
    return question_id


def _ordered_letters(letters: Iterable[str]) -> List[str]:
    allowed = {str(letter).strip().upper() for letter in letters if str(letter).strip()}
    return [letter for letter in LETTERS if letter in allowed]


def mc_option_letters(answer_row: dict) -> List[str]:
    question_text = str(answer_row.get("question") or answer_row.get("prompt") or "")
    parsed = re.findall(r"^\s*([A-Z])\.", question_text, flags=re.M)
    if parsed:
        return _ordered_letters(parsed)

    options = answer_row.get("options")
    if isinstance(options, list) and options:
        return list(LETTERS[: min(len(options) + 1, len(LETTERS))])

    gt = normalize_mc_answer(answer_row.get("ground_truth"), DEFAULT_MCQ_LETTERS)
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
    return (len(pred_set & gt_set) * correct_weight) - (len(pred_set - gt_set) * incorrect_weight)


def normalize_tf_answer(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return None
    head = text.splitlines()[0].strip().split(" ", 1)[0]
    if head == "yes":
        return True
    if head == "no":
        return False
    return None


def score_records(benchmark: str, records: Iterable[dict], answers: Dict[str, dict]) -> dict:
    stats = {
        "benchmark": benchmark,
        "total_predictions": 0,
        "matched_answers": 0,
        "correct": 0,
        "score_sum": 0.0,
        "invalid_predictions": 0,
        "missing_answers": 0,
        "duplicate_predictions_skipped": 0,
    }
    by_task = defaultdict(lambda: {"matched_answers": 0, "correct": 0, "score_sum": 0.0, "invalid_predictions": 0})
    seen_question_ids: set[str] = set()

    for record in records:
        stats["total_predictions"] += 1
        question_id = extract_question_id(record)
        if not question_id:
            stats["missing_answers"] += 1
            continue
        if question_id in seen_question_ids:
            stats["duplicate_predictions_skipped"] += 1
            continue
        seen_question_ids.add(question_id)
        answer_row = answers.get(question_id)
        if answer_row is None:
            stats["missing_answers"] += 1
            continue

        source_task = str(answer_row.get("source_task") or "unknown_task")
        stats["matched_answers"] += 1
        by_task[source_task]["matched_answers"] += 1

        response = record.get("response")
        truth = answer_row.get("ground_truth")
        if benchmark == "mc":
            options = mc_option_letters(answer_row)
            pred = normalize_mc_answer(response, options)
            gt = normalize_mc_answer(truth, options)
            score_value = score_mc_answer(pred, gt, options) if pred is not None and gt is not None else 0.0
        else:
            pred = normalize_tf_answer(response)
            gt = normalize_tf_answer(truth)
            score_value = 1.0 if pred is not None and gt is not None and pred == gt else 0.0

        if pred is None or gt is None:
            stats["invalid_predictions"] += 1
            by_task[source_task]["invalid_predictions"] += 1
            continue
        stats["score_sum"] += score_value
        by_task[source_task]["score_sum"] += score_value
        if pred == gt:
            stats["correct"] += 1
            by_task[source_task]["correct"] += 1

    matched = stats["matched_answers"]
    stats["accuracy"] = (stats["score_sum"] / matched) if matched else 0.0
    stats["by_task"] = [
        {
            "source_task": task,
            "matched_answers": values["matched_answers"],
            "correct": values["correct"],
            "score_sum": values["score_sum"],
            "invalid_predictions": values["invalid_predictions"],
            "accuracy": (values["score_sum"] / values["matched_answers"]) if values["matched_answers"] else 0.0,
        }
        for task, values in sorted(by_task.items())
    ]
    return stats


def main() -> None:
    args = parse_args()
    answers = load_answers(args.answers.resolve())
    summary = score_records(
        args.benchmark,
        iter_prediction_records(args.predictions_root.resolve()),
        answers,
    )
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
