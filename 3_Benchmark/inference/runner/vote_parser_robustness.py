from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Pattern, Sequence, Tuple

from inference.utils.jsonio import write_json_atomic


@dataclass
class TaskSample:
    task: str
    sample_id: str
    modality: str
    prompt: str
    fake_path: str
    relative_fake_path: str
    label: str = "analysis"
    file_sha256: Optional[str] = None
    media_meta: Optional[Dict[str, object]] = None

    def to_json(self) -> Dict[str, object]:
        return {
            "task": self.task,
            "sample_id": self.sample_id,
            "modality": self.modality,
            "prompt": self.prompt,
            "fake_path": str(self.fake_path),
            "relative_fake_path": str(self.relative_fake_path),
            "label": self.label,
            "file_sha256": self.file_sha256,
            "media_meta": self.media_meta,
        }


@dataclass
class ModelResponse:
    model_id: str
    sample: TaskSample
    response: str
    latency_ms: float
    fallback_count: int = 0
    final_seed: int = 0
    system_hint: Optional[str] = None
    usage_metadata: Optional[Dict[str, object]] = None

    def to_json(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "sample": self.sample.to_json(),
            "response": self.response,
            "timestamp": time.time(),
            "fallback_count": self.fallback_count,
            "final_seed": self.final_seed,
            "system_hint": self.system_hint,
            "usage_metadata": self.usage_metadata,
        }


SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]")


def _sanitize_token(text: str, fallback: str = "sample") -> str:
    token = SAFE_TOKEN_RE.sub("-", str(text or "").strip())
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token or fallback


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


MAPPING_PROMPT = _load_mapping_prompt()


def _extract_artifacts(prompt: str) -> List[str]:
    matches = re.findall(r"^\*\s*\*\*(.*?)\*\*", prompt, flags=re.M)
    return [m.strip() for m in matches if m.strip()]


ARTIFACTS = _extract_artifacts(MAPPING_PROMPT)


def _normalize_artifact(name: str) -> str:
    # Make matching robust to minor formatting differences: spaces, '&' vs 'and', punctuation.
    s = name.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


_CANONICAL_BY_NORM = {_normalize_artifact(a): a for a in ARTIFACTS}


def _parse_boolish(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        s = value.strip().strip("\"'").lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0"):
            return False
    return None


def _extract_json_object_substring(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


_LINE_RE = re.compile(
    r"^\s*[\-*]?\s*`?\s*\"?\s*(?P<key>[^:\n\r]+?)\s*\"?\s*`?\s*:\s*\"?\s*(?P<val>true|false)\s*\"?\s*,?\s*$",
    flags=re.IGNORECASE,
)


def parse_artifact_map(response_text: str) -> Dict[str, bool]:
    """Parse a model's response into {ArtifactName: bool} for known artifacts."""
    out: Dict[str, bool] = {}
    text = (response_text or "").strip()
    if not text:
        return out

    # Try JSON (some models output a JSON object).
    for candidate in (text, _extract_json_object_substring(text) or ""):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                norm = _normalize_artifact(str(k))
                canonical = _CANONICAL_BY_NORM.get(norm)
                if not canonical:
                    continue
                b = _parse_boolish(v)
                if b is None:
                    continue
                out[canonical] = b
            if out:
                return out

    # Fallback: parse `Artifact: True/False` lines.
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        key = m.group("key").strip()
        val = m.group("val").strip()
        canonical = _CANONICAL_BY_NORM.get(_normalize_artifact(key))
        if not canonical:
            continue
        b = _parse_boolish(val)
        if b is None:
            continue
        out[canonical] = b

    return out


def _iter_records(dir_path: Path) -> Iterator[Tuple[Path, Dict[str, object]]]:
    for path in sorted(dir_path.rglob("*.json")):
        if not path.is_file():
            continue
        if path.name.startswith("summary__"):
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict):
            yield path, obj


def _sample_key(record: Dict[str, object]) -> Optional[Tuple[str, str]]:
    sample = record.get("sample")
    if not isinstance(sample, dict):
        return None
    task = str(sample.get("task") or "").strip()
    sample_id = str(sample.get("sample_id") or "").strip()
    if not task or not sample_id:
        return None
    return task, sample_id


def _record_to_sample(record: Dict[str, object]) -> Optional[TaskSample]:
    sample = record.get("sample")
    if not isinstance(sample, dict):
        return None
    try:
        return TaskSample(
            task=str(sample.get("task") or ""),
            sample_id=str(sample.get("sample_id") or ""),
            modality=str(sample.get("modality") or "text"),
            prompt=str(sample.get("prompt") or ""),
            fake_path=str(sample.get("fake_path") or ""),
            relative_fake_path=str(sample.get("relative_fake_path") or ""),
            label=str(sample.get("label") or "analysis"),
            file_sha256=sample.get("file_sha256"),
            media_meta=sample.get("media_meta"),
        )
    except Exception:
        return None


def _output_path(sample: TaskSample, output_dir: Path, layout: str) -> Path:
    if layout == "task-dir":
        return output_dir / (sample.task or "unknown_task") / f"{sample.sample_id}.json"
    if layout == "flat":
        return output_dir / f"{sample.sample_id}.json"
    safe_task = _sanitize_token(sample.task or "eval_hallucination")
    safe_sample = _sanitize_token(sample.sample_id or "sample")
    return output_dir / f"{safe_task}__{safe_sample}.json"


def _summary_path(task_name: str, output_dir: Path, layout: str) -> Optional[Path]:
    if layout == "task-sample":
        safe_task = _sanitize_token(task_name or "eval_hallucination")
        return output_dir / f"summary__{safe_task}.json"
    if layout == "task-dir":
        safe_task = _sanitize_token(task_name or "eval_hallucination")
        return output_dir / f"{safe_task}.json"
    return None


def _format_response(values: Dict[str, bool], artifacts: Sequence[str]) -> str:
    return "\n".join([f"{a}: {'True' if values.get(a) else 'False'}" for a in artifacts])


def _vote(
    artifact: str,
    votes: Dict[str, Optional[bool]],
    *,
    tie_break: str,
    prefer_order: Sequence[str],
) -> bool:
    trues = sum(1 for v in votes.values() if v is True)
    falses = sum(1 for v in votes.values() if v is False)
    if trues > falses:
        return True
    if falses > trues:
        return False

    # Tie or no votes.
    if tie_break in ("true", "false"):
        return tie_break == "true"
    if tie_break.startswith("prefer:"):
        preferred = tie_break.split(":", 1)[1]
        v = votes.get(preferred)
        if v is not None:
            return bool(v)

    # Default deterministic fallback: first available in prefer_order, else False.
    for name in prefer_order:
        v = votes.get(name)
        if v is not None:
            return bool(v)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voting-based mapping by aggregating parser_robustness outputs (gemini, gpt5-mini, qwen3)."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("./rebuttal_alignment_exp/parser_robustness"),
        help="Directory containing per-parser subfolders (e.g. gemini, gpt5-mini, qwen3).",
    )
    parser.add_argument(
        "--parsers",
        nargs="*",
        default=("gemini", "gpt5-mini", "qwen3"),
        help="Subfolders to vote over (default: gemini gpt5-mini qwen3).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--output-layout",
        type=str,
        default="task-sample",
        choices=("task-sample", "task-dir", "flat"),
    )
    parser.add_argument(
        "--tie-break",
        type=str,
        default="false",
        choices=("false", "true", "prefer:gemini", "prefer:gpt5-mini", "prefer:qwen3"),
        help="What to do if votes tie due to missing/invalid parser outputs.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="model_id to store in output; default is derived from parser list.",
    )
    parser.add_argument(
        "--include-vote-metadata",
        action="store_true",
        help="If set, adds vote_metadata to each output record.",
    )
    return parser.parse_args()


def main() -> None:
    if not ARTIFACTS:
        raise RuntimeError("No artifacts found in mapping_prompt; cannot vote.")

    args = parse_args()
    input_root = args.input_root.resolve()
    if args.output_dir is None:
        args.output_dir = input_root / "voting"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parser_names = [p for p in args.parsers if p]
    if not parser_names:
        raise RuntimeError("--parsers is empty")

    model_id = args.model_id
    if not model_id:
        model_id = "voting:" + "+".join(parser_names)

    # Build index: (task, sample_id) -> {parser_name: record}
    index: Dict[Tuple[str, str], Dict[str, Dict[str, object]]] = {}
    scanned = 0
    for name in parser_names:
        dir_path = input_root / name
        if not dir_path.exists():
            print(f"[WARN] Missing parser folder: {dir_path}")
            continue
        for _, record in _iter_records(dir_path):
            key = _sample_key(record)
            if not key:
                continue
            index.setdefault(key, {})[name] = record
            scanned += 1

    keys = sorted(index.keys())
    if args.max_samples is not None:
        keys = keys[: max(0, args.max_samples)]

    print(f"[INFO] Loaded {len(keys)} unique sample(s) from {len(parser_names)} parser(s); scanned {scanned} record(s).")
    if not keys:
        return

    task_records: Dict[str, List[Dict[str, object]]] = {}

    for key in keys:
        per_parser = index.get(key, {})
        # Pick a base sample deterministically.
        base_record: Optional[Dict[str, object]] = None
        for name in parser_names:
            if name in per_parser:
                base_record = per_parser[name]
                break
        if base_record is None:
            continue
        sample_obj = _record_to_sample(base_record)
        if sample_obj is None:
            continue

        out_path = _output_path(sample_obj, output_dir, args.output_layout)
        if args.skip_existing and out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    task_records.setdefault(sample_obj.task or "unknown_task", []).append(existing)
                    print(f"[INFO] Skipping existing result: {out_path}")
                    continue
            except Exception:
                pass

        start = time.time()

        parsed_maps: Dict[str, Dict[str, bool]] = {}
        for name, record in per_parser.items():
            resp = record.get("response")
            parsed_maps[name] = parse_artifact_map(str(resp or ""))

        voted: Dict[str, bool] = {}
        vote_meta: Dict[str, object] = {}

        for artifact in ARTIFACTS:
            votes: Dict[str, Optional[bool]] = {}
            for name in parser_names:
                if name not in parsed_maps:
                    votes[name] = None
                    continue
                votes[name] = parsed_maps[name].get(artifact)

            value = _vote(artifact, votes, tie_break=args.tie_break, prefer_order=parser_names)
            voted[artifact] = value

            if args.include_vote_metadata:
                vote_meta[artifact] = {
                    "true": sum(1 for v in votes.values() if v is True),
                    "false": sum(1 for v in votes.values() if v is False),
                    "votes": {k: v for k, v in votes.items() if v is not None},
                }

        response_text = _format_response(voted, ARTIFACTS)
        latency_ms = (time.time() - start) * 1000.0

        record_out = ModelResponse(
            model_id=model_id,
            sample=sample_obj,
            response=response_text,
            latency_ms=latency_ms,
            fallback_count=0,
            final_seed=0,
            system_hint=None,
            usage_metadata=None,
        ).to_json()

        if args.include_vote_metadata:
            record_out["vote_metadata"] = {
                "parsers": parser_names,
                "tie_break": args.tie_break,
                "per_artifact": vote_meta,
            }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(record_out, out_path)
        task_records.setdefault(sample_obj.task or "unknown_task", []).append(record_out)
        print(f"[INFO] Wrote per-sample result: {out_path}")

    if args.output_layout in ("task-sample", "task-dir"):
        for task_name, records in task_records.items():
            summary_path = _summary_path(task_name, output_dir, args.output_layout)
            if summary_path is None:
                continue
            write_json_atomic(records, summary_path)
            print(f"[INFO] Saved task summary: {summary_path}")


if __name__ == "__main__":
    main()
