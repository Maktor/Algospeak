"""
inference.py

Inference and full evaluation for the dual BERTweet model.

Inference uses only the unsupervised encoder:
  1. Build class prototypes from the training set (average embedding per class).
  2. For a new post: encode -> cosine similarity to each prototype -> argmax = class.

Evaluation produces:
  - Accuracy (overall + per-class)
  - Precision, Recall, F1 (per-class, macro, weighted)
  - Confusion matrix (saved as PNG)
  - ROC curves + AUC per class (saved as PNG)
  - Full metrics saved to JSON

Usage:
    uv run python poc/src/inference.py
"""

import sys
import json
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import emoji
matplotlib.use("Agg")  # non-interactive backend for saving figures

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

sys.path.insert(0, str(Path(__file__).parent))
from model import DualEncoderModel, BERTweetEncoder

BASE_DIR = Path(__file__).resolve().parent.parent.parent

CLASS_PREFIX = {
    0: "Allowed:",
    1: "Offensive Language:",
    2: "Mature Content:",
    3: "Algospeak:",
}

CLASS_NAMES = ["Allowed", "Offensive Language", "Mature Content", "Algospeak"]


def load_config() -> dict:
    with open(BASE_DIR / "poc" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_unsupervised_encoder(ckpt_path: Path, cfg: dict, device: torch.device):
    """Load the full dual model from checkpoint, return only the unsupervised encoder."""
    model = DualEncoderModel(cfg["model_name"], cfg["temperature"])
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model.unsupervised


def load_dataset(path: Path) -> TensorDataset:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return TensorDataset(
        data["unsup_ids"],
        data["unsup_mask"],
        data["labels"],
    )


def get_embeddings(
    encoder:  BERTweetEncoder,
    dataset:  TensorDataset,
    batch_sz: int,
    device:   torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run all samples through the unsupervised encoder. Returns (embeddings, labels)."""
    loader = DataLoader(dataset, batch_size=batch_sz, shuffle=False, num_workers=2)
    all_embs, all_labels = [], []

    with torch.no_grad():
        for unsup_ids, unsup_mask, labels in loader:
            unsup_ids  = unsup_ids.to(device)
            unsup_mask = unsup_mask.to(device)
            embs = encoder(unsup_ids, unsup_mask)
            all_embs.append(embs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.vstack(all_embs), np.concatenate(all_labels)


def build_prototypes(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    num_classes: int,
    weights:    np.ndarray | None = None,
) -> np.ndarray:
    """Average embedding per class -> [num_classes, D] prototype matrix.

    weights: optional per-sample weight array (same length as labels).
             If provided, class 3 prototype uses weighted mean; other classes
             use uniform mean regardless.
    """
    D = embeddings.shape[1]
    prototypes = np.zeros((num_classes, D), dtype=np.float32)
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() > 0:
            embs = embeddings[mask]
            if weights is not None and cls == 3:
                w = weights[mask]
                w = w / w.sum()
                proto = (embs * w[:, None]).sum(axis=0)
            else:
                proto = embs.mean(axis=0)
            prototypes[cls] = proto / (np.linalg.norm(proto) + 1e-8)
    return prototypes


def load_class3_weights(train_csv_path: Path, synth_csv_path: Path,
                        labels: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency weights for class 3 training examples.

    Joins train.csv with synthetic_algospeak.csv on text=algospeak_text to
    recover the deny_terms_transformed for each class 3 row, then assigns
    weight = 1 / term_count so underrepresented algospeak types contribute more
    to the prototype.
    """
    train_df = pd.read_csv(train_csv_path).dropna(subset=["text"]).reset_index(drop=True)
    synth_df = pd.read_csv(synth_csv_path)
    text_to_term = dict(zip(
        synth_df["algospeak_text"].astype(str).str.strip(),
        synth_df["deny_terms_transformed"].astype(str),
    ))

    # Build term list for every training row (non-class-3 rows get "")
    terms = []
    for i, row in train_df.iterrows():
        if labels[i] == 3:
            terms.append(text_to_term.get(str(row["text"]).strip(), "unknown"))
        else:
            terms.append("")

    # Count occurrences among class 3 only
    class3_terms = [t for t, lbl in zip(terms, labels) if lbl == 3]
    term_counts = pd.Series(class3_terms).value_counts().to_dict()

    weights = np.ones(len(labels), dtype=np.float32)
    for i, (term, lbl) in enumerate(zip(terms, labels)):
        if lbl == 3:
            weights[i] = 1.0 / term_counts.get(term, 1)

    matched = sum(1 for t, lbl in zip(terms, labels) if lbl == 3 and t != "unknown")
    total_c3 = int((labels == 3).sum())
    print(f"  Class 3 weight lookup: {matched}/{total_c3} matched to deny terms "
          f"({100*matched/total_c3:.1f}%)")
    print(f"  Unique deny terms in class 3 training: {len(term_counts)}")
    return weights


def predict(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cosine similarity of each embedding to each prototype.
    Returns (predicted_labels, score_matrix [N, num_classes]).
    Scores are softmax-normalized cosine similarities — used for ROC curves.
    """
    # cosine similarity: embeddings are already L2-normalized, prototypes also normalized
    sim = embeddings @ prototypes.T                           # [N, num_classes]
    scores = torch.softmax(torch.tensor(sim / 0.1), dim=-1).numpy()  # [N, num_classes]
    preds  = sim.argmax(axis=1)
    return preds, scores


# ─────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {out_path}")


def plot_roc_curves(y_true, scores, num_classes: int, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    for cls in range(num_classes):
        y_bin = (y_true == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, scores[:, cls])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[cls], lw=2,
                label=f"{CLASS_NAMES[cls]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ROC curves saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────

def evaluate_split(
    encoder:    BERTweetEncoder,
    prototypes: np.ndarray,
    split:      str,
    cfg:        dict,
    device:     torch.device,
    results_dir: Path,
) -> dict:
    print(f"\n--- Evaluating {split} split ---")
    dataset = load_dataset(BASE_DIR / cfg["prepared_dir"] / f"{split}.pt")
    embs, labels = get_embeddings(encoder, dataset, cfg["batch_size"], device)
    preds, scores = predict(embs, prototypes)

    # Save per-sample predictions CSV
    csv_df = pd.read_csv(BASE_DIR / cfg[f"{split}_csv"])
    csv_df = csv_df.dropna(subset=["text"]).reset_index(drop=True)
    pred_df = pd.DataFrame({
        "text":            csv_df["text"].astype(str),
        "true_label":      [CLASS_NAMES[i] for i in labels],
        "predicted_label": [CLASS_NAMES[i] for i in preds],
        "correct":         labels == preds,
    })
    pred_df.to_csv(results_dir / f"predictions_{split}.csv", index=False)
    print(f"  Predictions saved -> {results_dir / f'predictions_{split}.csv'}")

    acc    = accuracy_score(labels, preds)
    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, output_dict=True
    )

    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    plot_confusion_matrix(labels, preds, results_dir / f"confusion_matrix_{split}.png")
    plot_roc_curves(labels, scores, cfg["num_classes"], results_dir / f"roc_curves_{split}.png")

    aucs = {}
    for cls in range(cfg["num_classes"]):
        y_bin = (labels == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, scores[:, cls])
        aucs[CLASS_NAMES[cls]] = round(auc(fpr, tpr), 4)

    return {
        "split":           split,
        "accuracy":        round(acc, 4),
        "macro_f1":        round(report["macro avg"]["f1-score"], 4),
        "weighted_f1":     round(report["weighted avg"]["f1-score"], 4),
        "per_class":       {
            CLASS_NAMES[i]: {
                "precision": round(report[CLASS_NAMES[i]]["precision"], 4),
                "recall":    round(report[CLASS_NAMES[i]]["recall"], 4),
                "f1":        round(report[CLASS_NAMES[i]]["f1-score"], 4),
            }
            for i in range(cfg["num_classes"])
        },
        "auc_per_class":   aucs,
        "mean_auc":        round(np.mean(list(aucs.values())), 4),
    }


def classify_text(text: str, encoder, prototypes, tokenizer, max_length, device) -> dict:
    """Classify a single raw text string. Returns predicted class and similarity scores."""
    enc = tokenizer(
        emoji.demojize(text), padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    with torch.no_grad():
        emb = encoder(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    emb = emb.cpu().numpy()

    sim    = emb @ prototypes.T
    scores = torch.softmax(torch.tensor(sim / 0.1), dim=-1).numpy()[0]
    pred   = int(sim.argmax())

    return {
        "predicted_class": pred,
        "predicted_label": CLASS_NAMES[pred],
        "scores":          {CLASS_NAMES[i]: round(float(scores[i]), 4)
                            for i in range(len(CLASS_NAMES))},
    }


def main():
    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir    = BASE_DIR / cfg["checkpoint_dir"]
    results_dir = BASE_DIR / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load unsupervised encoder
    encoder = load_unsupervised_encoder(ckpt_dir / "best_model.pt", cfg, device)

    # Build prototypes from training set
    print("\nBuilding class prototypes from training set...")
    train_ds = load_dataset(BASE_DIR / cfg["prepared_dir"] / "train.pt")
    train_embs, train_labels = get_embeddings(encoder, train_ds, cfg["batch_size"], device)
    synth_csv = BASE_DIR / "Algospeak_experiment" / "synthetic_algospeak.csv"
    weights = load_class3_weights(BASE_DIR / cfg["train_csv"], synth_csv, train_labels)
    prototypes = build_prototypes(train_embs, train_labels, cfg["num_classes"], weights=weights)
    np.save(results_dir / "prototypes.npy", prototypes)
    print(f"  Prototypes saved -> {results_dir / 'prototypes.npy'}")

    # Evaluate val and test splits
    all_results = []
    for split in ["val", "test"]:
        result = evaluate_split(encoder, prototypes, split, cfg, device, results_dir)
        all_results.append(result)

    # Save metrics
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll metrics saved -> {metrics_path}")

    # Summary
    print("\n=== SUMMARY ===")
    for r in all_results:
        print(f"{r['split']:6s} | acc={r['accuracy']:.4f} | macro_f1={r['macro_f1']:.4f} | mean_auc={r['mean_auc']:.4f}")

    # Quick example inference
    print("\n=== Example inference ===")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    examples = [
        "I had a great day today, went for a walk in the park.",
        "I'm going to k!ll that n!gga if he shows up again.",
        "she posted an onlyfans link in her bio",
        "gonna unalive myself fr fr cant take this anymore",
    ]
    for text in examples:
        result = classify_text(text, encoder, prototypes, tokenizer, cfg["max_length"], device)
        print(f"  [{result['predicted_label']}] {text[:70]}")


if __name__ == "__main__":
    main()
