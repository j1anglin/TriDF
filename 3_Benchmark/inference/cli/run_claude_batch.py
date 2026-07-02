import argparse
import os
from pathlib import Path
import sys

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from inference.dataio.fetch_samples import fetch_samples_for_task  # noqa: E402
from inference.dataio.detection import DETECTION_SYSTEM_HINT  # noqa: E402
from inference.dataio.perception import PERCEPTION_SYSTEM_HINT  # noqa: E402
from inference.cli.run_perception_mc import SYSTEM_HINT as PERCEPTION_MC_SYSTEM_HINT  # noqa: E402
from inference.cli.run_perception_tf_ver2 import SYSTEM_HINT as PERCEPTION_TF_SYSTEM_HINT  # noqa: E402
from inference.runner.claude_batch_runner import run_claude_batch_job  # noqa: E402
from inference.wrappers.registry import build_model_wrapper  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run Claude Messages Batch Inference Job")
    parser.add_argument("--model-id", required=True, help="Claude model ID (e.g., claude-3-opus-20240229)")
    parser.add_argument("--task-name", required=True, help="Name of the benchmark task")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save results")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens for generation")
    parser.add_argument("--max-samples", type=int, help="Limit the number of samples to process")
    parser.add_argument("--system-hint", type=str, default=None, help="Optional system instruction")
    parser.add_argument("--user-prefix", type=str, default=None, help="Optional prefix for prompts")
    parser.add_argument("--batch-size", type=int, default=25, help="Number of requests per batch chunk")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between status polls")
    parser.add_argument(
        "--max-parallel-batches",
        type=int,
        default=1,
        help="Maximum number of Claude batch operations to run concurrently",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Override Claude/Anthropic API key for this run")

    # Compatibility arguments to align with other runners.
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
    parser.add_argument("--skip-audio", action="store_true", help="Exclude audio samples from the batch request.")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", args.api_key)
        os.environ.setdefault("CLAUDE_API_KEY", args.api_key)

    model_id = args.model_id or args.models or args.model

    print("Building Claude batch wrapper...")
    wrapper = build_model_wrapper(
        model_id=model_id,
        max_new_tokens=args.max_new_tokens,
    )
    if args.system_hint:
        wrapper.system_hint = args.system_hint
    if args.user_prefix:
        wrapper.user_prefix = args.user_prefix

    task_key = args.task_name.lower()
    if not wrapper.system_hint:
        if task_key.startswith("typeb_oeq") and DETECTION_SYSTEM_HINT:
            wrapper.system_hint = DETECTION_SYSTEM_HINT
        elif task_key.startswith("typea_oeq") and PERCEPTION_SYSTEM_HINT:
            wrapper.system_hint = PERCEPTION_SYSTEM_HINT
        elif task_key.startswith("perception_mc") and PERCEPTION_MC_SYSTEM_HINT:
            wrapper.system_hint = PERCEPTION_MC_SYSTEM_HINT
        elif task_key.startswith("perception_tf") and PERCEPTION_TF_SYSTEM_HINT:
            wrapper.system_hint = PERCEPTION_TF_SYSTEM_HINT

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
    )

    if args.skip_audio:
        before = len(samples)
        samples = [s for s in samples if (getattr(s, "modality", "").lower() != "audio")]
        removed = before - len(samples)
        if removed:
            print(f"[INFO] Skip-audio enabled: removed {removed} audio sample(s).")
        if not samples:
            print("[WARN] Skip-audio removed all samples; nothing to submit.")

    if args.max_samples is not None and samples:
        per_task_counts = {}
        limited_samples = []
        for sample in samples:
            task_name = getattr(sample, "task", "unknown_task")
            count = per_task_counts.get(task_name, 0)
            if count >= args.max_samples:
                continue
            limited_samples.append(sample)
            per_task_counts[task_name] = count + 1
        if len(limited_samples) < len(samples):
            print(
                f"[INFO] Applying per-task max_samples={args.max_samples}: "
                f"trimmed {len(samples) - len(limited_samples)} sample(s)."
            )
        samples = limited_samples

    run_claude_batch_job(
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
