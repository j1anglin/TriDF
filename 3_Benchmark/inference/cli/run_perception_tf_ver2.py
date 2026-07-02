from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from textwrap import dedent
from typing import List

from inference.dataio.tf_questions import collect_question_files, normalize_modalities
from inference.runner.tf_eval import evaluate_question_files, instantiate_wrapper
from inference.runner.common import maybe_localize
from inference.utils.media import require_video_support

TF_RECOMMENDED_MODELS = [
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
- Given a single yes/no question asking whether a specific artifact appears in a specified region of a sample (image/audio/video), answer strictly "yes" or "no".

Output Constraints:
1) The FIRST LINE must be exactly one of: yes or no (lowercase).
   - No punctuation, spaces, or trailing characters.
   - Do NOT output `None` under any circumstances.

Decision Rules:
- Treat yes/no as equally likely.
- If the ROI is fully out-of-frame, fully occluded, or too low-res to perceive basic shape → answer "no".
- If any meaningful part of the ROI is visible, judge based on the visible portion; do not auto-"no" solely due to partial occlusion.
- Answer "yes" if either (a) a clear, distinctive cue of the named artifact is present in the ROI, or (b) two or more consistent subtle cues are present; otherwise answer "no".

Validation Rules:
- Allowed outputs (regex): ^(?:yes|no)$

Examples:
- "Does banding appear on the shoulder in the image?" → yes/no
- "Is there motion blur in the subject's face?" → yes/no
"""
).strip()

warnings.filterwarnings("ignore", message="Unused or unrecognized kwargs: batch_num_images.")
warnings.filterwarnings("ignore", message="Unused or unrecognized kwargs: batch_num_frames.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate perception TF question JSONs with the registered multimodal model wrappers.",
    )
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--questions-dir",
        type=Path,
        default=default_root / "benchmark_perception_tf",
        help="Directory containing perception TF question JSON files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/project/aimm/benchmark"),
        help="Root directory that holds the media assets referenced by the question JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "runs" / "perception_tf_ver2",
        help="Directory where per-model JSONL outputs will be written.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=TF_RECOMMENDED_MODELS[0],
        help="Model id to evaluate.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["img", "vid"],
        help="Modalities to keep from the question JSON entries (e.g., img vid).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional list of task names to keep.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples processed per JSON file.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per response.",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=None,
        help="Optional minimum number of tokens to request during generation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum number of retries when generation yields empty/refusal responses.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed used when retrying generations.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype hint (auto, bf16, fp16, fp32).",
    )
    parser.add_argument("--device-map", default="auto", help="Device map forwarded to Transformers.")
    parser.add_argument("--gpus", default=None, help="Optional GPU id list used for device map planning.")
    parser.add_argument(
        "--per-gpu-max-memory-gib",
        type=int,
        default=None,
        help="Optional per-GPU memory cap forwarded to from_pretrained.",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Request flash_attention_2 when the backend supports it.",
    )
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=None,
        help="Number of frames to sample from videos. If not specified, uses each model's built-in default (InternVL: 8, LLaVA/VILA/Qwen: 16).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/project/aimm/benchmark/models",
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Enable local_files_only mode when loading models and processors.",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Enable Hugging Face CPU/disk offload when loading large models.",
    )
    parser.add_argument(
        "--system-hint",
        type=str,
        default=SYSTEM_HINT,
        help="Optional system prompt appended to every conversation.",
    )
    parser.add_argument(
        "--user-prefix",
        type=str,
        default=None,
        help="Optional prefix prepended to the textual portion of each question.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-file outputs instead of skipping them.",
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=None,
        help="Optional extra root to prepend to each sample_path.",
    )
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
        print("[ERROR] No TF question JSON files found for the requested modalities.")
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
    print(f"[INFO] Evaluating model {requested_model_id} on {len(question_files)} TF files.")

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
        sample_root=args.sample_root,
        overwrite=args.overwrite,
        max_retries=args.max_retries,
        base_seed=args.seed,
        clear_gpu_cache_every=args.clear_gpu_cache_every,
    )

    print("[INFO] TF evaluation complete.")


if __name__ == "__main__":
    main()
