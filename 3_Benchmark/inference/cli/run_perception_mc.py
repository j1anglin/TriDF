from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from textwrap import dedent
from typing import List

from inference.dataio.mc_questions import collect_question_files, normalize_modalities
from inference.runner.mc_eval import evaluate_question_files, instantiate_wrapper
from inference.runner.common import maybe_localize
from inference.utils.media import require_video_support

MC_RECOMMENDED_MODELS = [
    "llava-hf/llava-onevision-qwen2-72b-ov-hf",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "OpenGVLab/InternVL2_5-26B",
    "OpenGVLab/InternVL2_5-8B",
    "OpenGVLab/InternVL2_5-38B",
    "OpenGVLab/InternVL2_5-78B",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
]

SYSTEM_HINT = dedent(
    """
You are a forensic media authenticity inspector.

Task:
- Given a sample (e.g., image, audio, video) and a list of artifact options labeled A–E, select all options that are clearly present.

Output Constraints:
1) The FIRST LINE must be exactly one line with no spaces or trailing characters.
   - A comma-separated subset of uppercase letters, e.g., `A,C,E`
   - Do NOT output `None` under any circumstances.
2) Select only options directly supported by clear, observable evidence. If uncertain, exclude them.
3) Consider ONLY the provided option set; ignore anything outside it.

Validation Rules:
- Allowed option set: {A, B, C, D, E}
- Allowed outputs (regex): ^(?:[A-E](?:,[A-E])*)$
- Use uppercase letters only; commas as separators; no spaces.
"""
).strip()

warnings.filterwarnings("ignore", message="Unused or unrecognized kwargs: batch_num_images.")
warnings.filterwarnings("ignore", message="Unused or unrecognized kwargs: batch_num_frames.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate perception MC question JSONs with registered multimodal model wrappers.",
    )
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--questions-dir", type=Path, default=default_root / "benchmark_perception_mc")
    parser.add_argument("--data-root", type=Path, default=Path("/project/aimm/benchmark"))
    parser.add_argument("--output-dir", type=Path, default=default_root / "runs" / "perception_mc")
    parser.add_argument("--model", type=str, default=MC_RECOMMENDED_MODELS[0])
    parser.add_argument("--modalities", nargs="+", default=["img", "vid"])
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=None,
        help="Optional minimum number of tokens to request during generation.",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--per-gpu-max-memory-gib", type=int, default=None)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=None,
        help="Number of frames to sample from videos. If not specified, uses each model's built-in default (InternVL: 8, LLaVA/VILA/Qwen: 16).",
    )
    parser.add_argument("--cache-dir", type=str, default="/project/aimm/benchmark/models")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Enable Hugging Face CPU/disk offload when loading large models.",
    )
    parser.add_argument("--system-hint", type=str, default=SYSTEM_HINT)
    parser.add_argument("--user-prefix", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--question-files",
        nargs="+",
        default=None,
        help="Optional list of question JSON files (names or paths) to evaluate exclusively.",
    )
    parser.add_argument(
        "--preserve-response-whitespace",
        action="store_true",
        help="Do not strip leading or trailing whitespace from model responses.",
    )
    parser.add_argument(
        "--clear-gpu-cache-every",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache() after every N processed samples (0 disables).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    questions_dir = args.questions_dir.resolve()
    data_root = (args.data_root or questions_dir.parent).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    normalized_modalities = normalize_modalities(args.modalities)
    if "video" in normalized_modalities:
        require_video_support()

    question_files = collect_question_files(questions_dir, normalized_modalities)
    if args.question_files:
        requested: List[Path] = []
        for entry in args.question_files:
            candidate = Path(entry)
            if candidate.is_file():
                requested.append(candidate.resolve())
            else:
                resolved = (questions_dir / entry).resolve()
                if resolved.is_file():
                    requested.append(resolved)
                else:
                    print(f"[WARN] Requested question file not found: {entry}")
        if requested:
            available_map = {path.resolve(): path for path in question_files}
            selected: List[Path] = []
            for req in requested:
                match = available_map.get(req)
                if match is None and req.exists():
                    match = req
                if match and match not in selected:
                    selected.append(match)
                elif not req.exists():
                    print(f"[WARN] Requested question file missing on disk: {req}")
            if selected:
                question_files = selected
            else:
                print("[ERROR] No matching question files found for the requested list.")
                return

    if not question_files:
        print("[ERROR] No MC question JSON files found for the requested modalities.")
        return

    requested_model_id = args.model
    resolved_model_id = (
        maybe_localize(requested_model_id, args.cache_dir) if args.cache_dir else requested_model_id
    )
    wrapper = instantiate_wrapper(
        resolved_model_id,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        gpus=args.gpus,
        per_gpu_max_memory_gib=args.per_gpu_max_memory_gib,
        flash_attn=args.flash_attn,
        cache_dir=args.cache_dir,
        offline=args.offline,
        video_max_frames=args.video_max_frames,
        system_hint=args.system_hint,
        user_prefix=args.user_prefix,
        offload=args.offload,
    )
    if args.min_new_tokens is not None:
        wrapper.generation_overrides = {
            **(wrapper.generation_overrides or {}),
            "min_new_tokens": args.min_new_tokens,
        }
    if args.preserve_response_whitespace:
        wrapper.preserve_whitespace = True

    if resolved_model_id != requested_model_id:
        print(f"[INFO] Resolved model '{requested_model_id}' to local path '{resolved_model_id}'.")
    print(f"[INFO] Evaluating model {requested_model_id} on {len(question_files)} MC files.")

    evaluate_question_files(
        question_files,
        requested_model_id=requested_model_id,
        resolved_model_id=resolved_model_id,
        questions_root=questions_dir,
        output_root=output_dir,
        wrapper=wrapper,
        allowed_modalities=normalized_modalities,
        data_root=data_root,
        tasks_filter=args.tasks,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        max_retries=args.max_retries,
        base_seed=args.seed,
        clear_gpu_cache_every=args.clear_gpu_cache_every,
    )

    print("[INFO] MC evaluation complete.")


if __name__ == "__main__":
    main()
