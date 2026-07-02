# TriDF Benchmark

Official implementation of **TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection** (CVPR 2026).

[Paper](https://arxiv.org/abs/2512.10652) · [Project Page](https://j1anglin.github.io/TriDF/) · [Code](https://github.com/j1anglin/TriDF) · [Dataset](https://huggingface.co/datasets/j1anglin/TriDF)

This repo contains the benchmark code — inference wrappers, evaluators, and
runner scripts. Task data, question files, and ground-truth artifact
annotations are published separately as a HuggingFace dataset (see
[Data](#data)); this repo does not contain any media files.

## Citation

```bibtex
@article{jiang2025tridf,
  title={TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection},
  author={Jiang-Lin, Jian-Yu and Huang, Kang-Yang and Zou, Ling and Lo, Ling and Yang, Sheng-Ping and Tseng, Yu-Wen and Lin, Kun-Hsiang and Chen, Chia-Ling and Ta, Yu-Ting and Wang, Yan-Tsung and others},
  journal={arXiv preprint arXiv:2512.10652},
  year={2025}
}
```

## Layout

```text
tridf/
├── 3_Benchmark/        # Python package: inference, wrappers, evaluators
├── scripts/            # Entrypoints for baselines, evaluation, and data setup
├── models/             # Local model cache (populated by you, not this repo)
├── runs/               # Inference outputs and evaluation scores
└── logs/               # Execution logs
```

`1_DATA/` and `2_GT_Final/` are not part of this repo — download them from
HuggingFace and place them at the repo root (see below), or point `DATA_ROOT`
/ `--gt-root` at wherever you put them.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Data

Task data (`1_DATA/`) and ground truth (`2_GT_Final/`) are hosted on
HuggingFace: [**`j1anglin/TriDF`**](https://huggingface.co/datasets/j1anglin/TriDF).

```bash
huggingface-cli download j1anglin/TriDF --repo-type dataset --local-dir .
```

This places `1_DATA/` and `2_GT_Final/` directly at the repo root, matching
what `scripts/env.sh` expects (`DATA_ROOT=${TRIDF_ROOT}/1_DATA`).

### Restoring CelebAMask-HQ real photos

143 real (unmodified) reference photos in `1_DATA/img_face_swapping/Real_DATA/`
originate from [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
and have been withheld from the HuggingFace dataset in respect of its
non-commercial-research-only, no-re-hosting terms. The restoration tool and
instructions ship with the dataset itself — see `tool/` and the
"Considerations for Using the Data" section in the
[dataset README](https://huggingface.co/datasets/j1anglin/TriDF).

## Run Baselines

TypeB OEQ, image tasks, 10 samples per task:

```bash
bash scripts/run_typeb_oeq_baseline.sh Qwen/Qwen3-VL-8B-Instruct 10
```

TypeA OEQ:

```bash
bash scripts/run_typea_oeq_baseline.sh Qwen/Qwen3-VL-8B-Instruct 10
```

Perception TF:

```bash
bash scripts/run_perception_tf_baseline.sh Qwen/Qwen3-VL-8B-Instruct 10
```

Perception MC:

```bash
bash scripts/run_perception_mc_baseline.sh Qwen/Qwen3-VL-8B-Instruct 10
```

To run all samples, pass an empty second argument:

```bash
bash scripts/run_typeb_oeq_baseline.sh Qwen/Qwen3-VL-8B-Instruct ""
```

Useful environment overrides:

```bash
export TYPEB_OEQ_MODALITY_FILTER=all   # image, video, audio, all
export TYPEA_OEQ_MODALITY_FILTER=all
export MODALITIES="img vid aud"
export OFFLINE=1                       # use local model cache only
```

Commercial perception data is merged into the standard TF/MC question sets, so
use the same perception baseline scripts above.

## Evaluation

**TypeB OEQ accuracy** (output → `runs/scoring/OEQ_score/`):

```bash
bash scripts/eval_typeb_oeq_accuracy.sh Qwen3-VL-8B-Instruct
# STRICT=0 to skip invalid predictions instead of counting them as wrong
```

**OEQ artifact metrics** — COVER / CHAIR / Hal / F0.5 via LLM mapper (output → `runs/scoring/OEQ_score/`).
Both TypeA and TypeB are evaluated in a single run by default.

With GPT (default):

```bash
bash scripts/eval_oeq_artifacts.sh Qwen3-VL-8B-Instruct
# requires OPENAI_API_KEY; mapper defaults to gpt-5-mini
```

With Gemini:

```bash
MAPPER_BACKEND=gemini bash scripts/eval_oeq_artifacts.sh Qwen3-VL-8B-Instruct
# requires GEMINI_API_KEY; mapper defaults to gemini-3.1-flash-lite
```

To restrict to a single task or modality:

```bash
BENCHMARK_TASK=typea_oeq bash scripts/eval_oeq_artifacts.sh Qwen3-VL-8B-Instruct
# MODALITIES="image"    to restrict to a single modality (default: "image video")
# MAPPER_MODEL=<model>  to override the default mapper model
```

**Perception TF / MC accuracy** (output → `runs/scoring/TFQ_score/` and `MCQ_score/`):

```bash
bash scripts/eval_perception_original.sh tf Qwen3-VL-8B-Instruct
bash scripts/eval_perception_original.sh mc Qwen3-VL-8B-Instruct
```

## Notes

- Model weights are intentionally not included; set `OFFLINE=1` to use a local cache under `models/`.
- Generated outputs under `runs/` are not pre-populated.
