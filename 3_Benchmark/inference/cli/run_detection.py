from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch

from inference.dataio.detection import (
    DETECTION_SYSTEM_HINT,
    discover_detection_tasks,
    load_detection_samples,
    materialize_previews,
    persist_task_manifests,
)
from inference.runner.detection import maybe_localize, run_detection_inference
from inference.utils.media import require_video_support
from inference.utils.torch_helpers import build_loading_kwargs, resolve_torch_dtype



DEFAULT_MODELS = [
    "llava-hf/llava-onevision-qwen2-72b-ov-hf",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "OpenGVLab/InternVL2-40B",
    "OpenGVLab/InternVL2_5-26B",
    "OpenGVLab/InternVL2_5-8B",
    "OpenGVLab/InternVL2_5-38B",
    "OpenGVLab/InternVL2_5-78B",
    "Efficient-Large-Model/VILA1.5-40b",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TypeB OEQ benchmark models (modular version)")
    parser.add_argument("--benchmark-root", type=Path, default=Path("/project/aimm/benchmark"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("./runs/inference/typeb_oeq"),
    )
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--only-tasks", nargs="*", default=None)
    parser.add_argument(
        "--collect-name",
        type=str,
        default="fake_collect.csv",
        help="Name of the primary sample manifest (e.g., fake_collect.csv or collect.csv).",
    )
    parser.add_argument(
        "--real-collect-name",
        type=str,
        default="real_collect.csv",
        help="Name of the real sample manifest; set empty or use --skip-real to ignore.",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Do not load real samples even if a real_collect CSV exists.",
    )
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "balanced", "balanced_low_0", "sequential", "cuda", "cpu"],
    )
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--per-gpu-max-memory-gib", type=int, default=None)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=None,
        help="Number of frames to sample from videos. If not specified, uses each model's built-in default (InternVL: 8, LLaVA/VILA/Qwen: 16).",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default="/project/aimm/benchmark/models")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-verify-media", action="store_true")
    parser.add_argument(
        "--preview-policy", type=str, default="auto", choices=["auto", "force", "skip"],
    )
    parser.add_argument(
        "--model-list",
        type=Path,
        default=None,
        help="Optional text file containing model ids (one per line) overriding --models",
    )
    return parser.parse_args()


def _load_model_list(default_models: List[str], model_list_path: Path | None) -> List[str]:
    if model_list_path is None:
        return default_models
    if not model_list_path.exists():
        raise FileNotFoundError(f"Model list file not found: {model_list_path}")
    with model_list_path.open("r", encoding="utf-8") as handle:
        models = [line.strip() for line in handle if line.strip() and not line.strip().startswith("#")]
    return models or default_models


def main() -> None:
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    benchmark_root: Path = args.benchmark_root
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    task_dirs = discover_detection_tasks(benchmark_root)
    if args.only_tasks:
        only = set(args.only_tasks)
        task_dirs = [d for d in task_dirs if d.name in only]

    real_collect_name = None if (args.skip_real or not args.real_collect_name) else args.real_collect_name
    include_real = bool(real_collect_name) and not args.skip_real
    tasks = load_detection_samples(
        task_dirs,
        max_samples=args.max_samples,
        verify_media=not args.skip_verify_media,
        include_real=include_real,
        collect_name=args.collect_name,
        real_collect_name=real_collect_name,
    )
    if not tasks:
        raise RuntimeError("No tasks discovered for evaluation.")

    if any(sample.modality == "video" for samples in tasks.values() for sample in samples):
        require_video_support()

    persist_task_manifests(tasks, output_root)

    if args.preview_policy != "skip":
        preview_root = output_root / "previews"
        materialize_previews(tasks, preview_root, policy=args.preview_policy)

    load_kwargs = build_loading_kwargs(
        device_map=args.device_map,
        gpus=args.gpus,
        per_gpu_max_memory_gib=args.per_gpu_max_memory_gib,
        flash_attn=bool(args.flash_attn),
        cache_dir=args.cache_dir,
        offline=bool(args.offline),
    )

    model_ids = _load_model_list(args.models, args.model_list)
    model_ids = [maybe_localize(model_id, args.cache_dir) if args.cache_dir else model_id for model_id in model_ids]

    run_detection_inference(
        tasks,
        model_ids=model_ids,
        output_root=output_root,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=resolve_torch_dtype(args.dtype),
        load_kwargs=load_kwargs,
        video_max_frames=args.video_max_frames,
        max_retries=args.max_retries,
        base_seed=args.seed,
        system_hint=DETECTION_SYSTEM_HINT,
    )


if __name__ == "__main__":
    main()
