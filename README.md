# Algospeak Detection — Dual BERTweet Classifier

A four-class content moderation system for detecting algospeak (coded language used to evade automated filters) on social media. Built with a dual BERTweet architecture and contrastive learning.

**Live demo:** [timagonch/algospeak-classifier](https://huggingface.co/spaces/timagonch/algospeak-classifier) on Hugging Face Spaces

---

## Classes

| Label | Class | Description |
|---|---|---|
| 0 | Allowed | Benign content |
| 1 | Obscene Language | Slurs, hate speech, offensive attacks |
| 2 | Mature Content | Explicit content, drugs, self-harm, incitement |
| 3 | Algospeak | Coded evasion of class 1 or 2 |

---

## Architecture

**Dual BERTweet** (`vinai/bertweet-base`, 270M params × 2):

- **Supervised encoder** — receives `"[CLASS_LABEL]: text"` during training (acts as teacher, discarded after)
- **Unsupervised encoder** — receives raw text only; used exclusively at inference
- **Loss** — Supervised InfoNCE (cross-encoder, like CLIP): pulls same-class embeddings together, pushes different-class apart
- **Inference** — cosine similarity to class prototypes (average unsupervised embedding per class); algospeak prototype uses inverse deny-term frequency weighting

---

## Project Structure

```
├── poc/                          # Main pipeline scripts
│   ├── src/
│   │   ├── model.py              # DualEncoderModel + InfoNCE loss
│   │   ├── encoder_prep.py       # Tokenize splits → .pt files
│   │   ├── train.py              # Training loop (fp16, early stopping, resumable)
│   │   └── inference.py          # Prototype inference + full eval metrics
│   └── config.yaml               # Model config
│
├── experiments/
│   └── four_class/               # Controlled four-class experiment (best model)
│       ├── src/
│       │   ├── model.py
│       │   ├── encoder_prep.py
│       │   ├── train.py
│       │   ├── inference.py
│       │   ├── prepare_splits.py # Combine + group-aware split
│       │   ├── generate_algospeak.py  # GPT-4-turbo synthetic generation
│       │   └── utils.py          # Deny-term detection, inflection-aware regex
│       ├── data/
│       │   ├── deny_list.txt     # Merged deny list (class1 + class2 terms)
│       │   ├── algospeak_hints.json   # Known community substitution forms
│       │   └── splits/           # train/val/test CSVs
│       ├── results/              # Per-run metrics, confusion matrices, ROC curves
│       ├── docs/
│       │   └── temperature_ablation.md
│       └── config.yaml
│
├── build_dataset.py              # Full dataset assembly pipeline (stages 1+2)
├── reclassify.py                 # Deny-list override reclassification
├── synthetic_data.py             # Original synthetic data generation
├── ingest_new_data.py            # Deduplicate + identify algospeak candidates
│
├── Algospeak_experiment/
│   ├── deny_list_class1.txt      # 115 slur/hate speech terms
│   └── deny_list_class2.txt      # 521 explicit content terms
│
└── llm_audit/
    ├── reclassify_full.py        # GPT-4o-mini bulk reclassification
    └── analyze_reclassified.py
```

---

## Setup

```bash
# Requires Python 3.12+ and uv
uv sync

# CUDA recommended (RTX 4070 or equivalent); CPU works but is slow
```

Create a `.env` file:
```env
OPENAI_API_KEY=your_key_here   # for synthetic generation
HF_TOKEN=your_token_here       # for HF Spaces deployment
```

---

## Running the Four-Class Experiment Pipeline

```bash
# 1. Prepare splits (combines reclassified + synthetic data, group-aware split)
uv run python experiments/four_class/src/prepare_splits.py

# 2. Tokenize splits -> .pt files
uv run python experiments/four_class/src/encoder_prep.py

# 3. Train (--fresh to start clean; --temperature to override tau)
uv run python experiments/four_class/src/train.py --fresh

# 4. Inference + metrics (run before next --fresh or checkpoint is overwritten)
uv run python experiments/four_class/src/inference.py --notes "describe run"
```

Results saved to `experiments/four_class/results/run_YYYYMMDD_HHMM/`.

---

## Key Results

| Experiment | Test Acc | Macro F1 | Algospeak F1 |
|---|---|---|---|
| Full dataset, Run 4 (Apr 13, ~13k/class) | **89.4%** | — | — |
| 3-class experiment (Apr 16) | 89.2% | 89.0% | **93.8%** |
| Four-class controlled, tau=0.15 (Apr 22) | 80.7% | 80.8% | 90.5% |

**Temperature ablation** (four-class controlled experiment):

| tau | Test Acc | Algospeak F1 |
|---|---|---|
| 0.07 | 72.1% | 81.4% |
| 0.10 | 79.2% | 90.3% |
| **0.15** | **80.7%** | **90.5%** |
| 0.20 | 82.4% | 91.6% |

tau=0.15 chosen over tau=0.20 despite lower aggregate metrics — tau=0.20 misclassified *"gonna unalive myself fr fr cant take this anymore"* as Allowed.

---

## Deployment

Model weights are stored in `timagonch/algospeak-classifier-model` (HF model repo, ~1.1GB via LFS).
Space code is in `timagonch/algospeak-classifier` (HF Spaces).

To upload updated weights after retraining:
```python
from huggingface_hub import upload_file
REPO = "timagonch/algospeak-classifier-model"
upload_file("experiments/four_class/checkpoints/best_model.pt", "best_model.pt", repo_id=REPO, repo_type="model")
upload_file("experiments/four_class/results/run_YYYYMMDD_HHMM/prototypes.npy", "prototypes.npy", repo_id=REPO, repo_type="model")
```

---

## Environment

- Python 3.12, managed with `uv`
- PyTorch with CUDA (RTX 4070 Laptop GPU, 8.6GB VRAM)
- BERTweet: `vinai/bertweet-base`
- Emoji tokenization: `emoji==0.6.0` (required — BERTweet was trained with emoji converted to text descriptions)
