import argparse
import os
from pathlib import Path
import sys

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from inference.dataio.detection import DETECTION_SYSTEM_HINT
from inference.dataio.fetch_samples import fetch_samples_for_task
from inference.dataio.perception import PERCEPTION_SYSTEM_HINT
from inference.cli.run_perception_mc import SYSTEM_HINT as PERCEPTION_MC_SYSTEM_HINT
from inference.cli.run_perception_tf_ver2 import SYSTEM_HINT as PERCEPTION_TF_SYSTEM_HINT
from inference.runner.gemini_batch_runner import run_gemini_batch_job
from inference.wrappers.registry import build_model_wrapper

def main():
    parser = argparse.ArgumentParser(description="Run Gemini Batch Inference Job")
    parser.add_argument("--model-id", required=True, help="Gemini model ID (e.g., gemini-1.5-pro-latest)")
    parser.add_argument("--task-name", required=True, help="Name of the benchmark task")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save results")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max new tokens for generation")
    parser.add_argument("--max-samples", type=int, help="Limit the number of samples to process")
    parser.add_argument("--system-hint", type=str, default=None, help="Optional system instruction for the model")
    parser.add_argument("--user-prefix", type=str, default=None, help="Optional prefix prepended to prompts")
    parser.add_argument("--batch-size", type=int, default=200, help="Number of requests per Gemini batch job")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between job status polls")
    parser.add_argument(
        "--max-parallel-batches",
        type=int,
        default=1,
        help="Maximum number of Gemini batch operations to run concurrently",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Override Gemini API key for this run")
    
    # Add arguments to match the other runners for compatibility with run_full_benchmark.sh
    parser.add_argument("--benchmark-root", type=str, default="/workspace", help="Benchmark root directory")
    parser.add_argument("--models", type=str, help="Model ID (alternative to --model-id)")
    parser.add_argument("--model", type=str, help="Model ID (alternative to --model-id)")
    parser.add_argument("--cache-dir", type=str, help="Cache directory (ignored)")
    parser.add_argument("--offline", action="store_true", help="Offline mode (ignored)")
    parser.add_argument("--flash-attn", action="store_true", help="Use flash attention (ignored)")
    parser.add_argument("--question-files", type=str, help="Question files (used by fetch_samples)")
    parser.add_argument("--questions-dir", type=str, help="Questions directory (used by fetch_samples)")
    parser.add_argument("--data-root", type=str, help="Data root (used by fetch_samples)")
    parser.add_argument(
        "--collect-name",
        type=str,
        default="fake_collect.csv",
        help="Collection name (used by fetch_samples)",
    )
    parser.add_argument("--skip-real", action="store_true", help="Skip real samples (used by fetch_samples)")
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Exclude audio samples from the batch request.",
    )
    parser.add_argument(
        "--only-tasks",
        nargs="*",
        default=None,
        help="Optional subset of task directory names to evaluate (e.g., aud_voice_cloning).",
    )


    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.api_key:
        os.environ.setdefault("GEMINI_API_KEY", args.api_key)
        os.environ.setdefault("GOOGLE_API_KEY", args.api_key)
        os.environ.setdefault("GOOGLE_GENERATIVE_AI_API_KEY", args.api_key)
    
    model_id = args.model_id or args.models or args.model

    print("Building Gemini wrapper...")
    wrapper = build_model_wrapper(
        model_id=model_id,
        max_new_tokens=args.max_new_tokens,
    )

    task_key = args.task_name.lower()
    if args.system_hint:
        wrapper.system_hint = args.system_hint
    elif task_key.startswith("typeb_oeq") and DETECTION_SYSTEM_HINT:
        wrapper.system_hint = DETECTION_SYSTEM_HINT
    elif task_key.startswith("typea_oeq") and PERCEPTION_SYSTEM_HINT:
        wrapper.system_hint = PERCEPTION_SYSTEM_HINT
    elif task_key.startswith("perception_mc") and PERCEPTION_MC_SYSTEM_HINT:
        wrapper.system_hint = PERCEPTION_MC_SYSTEM_HINT
    elif task_key.startswith("perception_tf") and PERCEPTION_TF_SYSTEM_HINT:
        wrapper.system_hint = PERCEPTION_TF_SYSTEM_HINT

    if args.user_prefix:
        wrapper.user_prefix = args.user_prefix

    print(f"Fetching samples for task: {args.task_name}...")
    requested_max = args.max_samples
    fetch_limit = None if (args.skip_audio and requested_max is not None) else requested_max

    samples = fetch_samples_for_task(
        args.task_name,
        max_samples=fetch_limit,
        question_files=args.question_files,
        questions_dir=args.questions_dir,
        data_root=args.data_root,
        collect_name=args.collect_name,
        skip_real=args.skip_real,
        benchmark_root=args.benchmark_root,
        tasks_filter=args.only_tasks,
    )

    if args.skip_audio:
        before = len(samples)
        samples = [sample for sample in samples if getattr(sample, "modality", "").lower() != "audio"]
        removed = before - len(samples)
        if removed:
            print(f"[INFO] Skip-audio enabled: removed {removed} audio sample(s).")
        if not samples:
            print("[WARN] Skip-audio removed all samples; nothing to submit.")

    if requested_max is not None and samples:
        per_task_counts = {}
        limited_samples = []
        for sample in samples:
            task_name = getattr(sample, "task", "unknown_task")
            seen = per_task_counts.get(task_name, 0)
            if seen >= requested_max:
                continue
            limited_samples.append(sample)
            per_task_counts[task_name] = seen + 1
        if len(limited_samples) < len(samples):
            print(
                f"[INFO] Applying per-task max_samples={requested_max}: "
                f"trimmed {len(samples) - len(limited_samples)} sample(s)."
            )
        samples = limited_samples

    run_gemini_batch_job(
        wrapper=wrapper,
        samples=samples,
        output_dir=args.output_dir,
        task_name=args.task_name,
        batch_size=args.batch_size,
        poll_interval=args.poll_interval,
        max_parallel_batches=args.max_parallel_batches,
    )

if __name__ == "__main__":
    main()
