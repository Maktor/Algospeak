"""
experiments/four_class/src/inference.py

Inference and full evaluation for the four-class dual BERTweet experiment.
Saves timestamped results to experiments/four_class/results/run_YYYYMMDD_HHMM/.

Usage:
    uv run python experiments/four_class/src/inference.py --notes "run 1 - τ=0.1"
"""

import sys
import json
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import emoji
import shutil
from datetime import datetime
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parents[3]  # project root
EXP_DIR  = Path(__file__).resolve().parents[1]  # experiments/four_class

sys.path.insert(0, str(Path(__file__).parent))
from model import DualEncoderModel, BERTweetEncoder

CLASS_NAMES = ["Allowed", "Obscene Language", "Mature Content", "Algospeak"]


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else EXP_DIR / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_unsupervised_encoder(ckpt_path: Path, cfg: dict, device: torch.device):
    model = DualEncoderModel(cfg["model_name"], cfg["temperature"])
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model.unsupervised


def load_dataset(path: Path) -> TensorDataset:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return TensorDataset(data["unsup_ids"], data["unsup_mask"], data["labels"])


def get_embeddings(encoder, dataset, batch_sz, device):
    loader = DataLoader(dataset, batch_size=batch_sz, shuffle=False, num_workers=0)
    all_embs, all_labels = [], []
    with torch.no_grad():
        for unsup_ids, unsup_mask, labels in loader:
            embs = encoder(unsup_ids.to(device), unsup_mask.to(device))
            all_embs.append(embs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_embs), np.concatenate(all_labels)


def build_prototypes(embeddings, labels, num_classes, weights=None, algospeak_cls=3):
    D = embeddings.shape[1]
    prototypes = np.zeros((num_classes, D), dtype=np.float32)
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() > 0:
            embs = embeddings[mask]
            if weights is not None and cls == algospeak_cls:
                w = weights[mask]
                w = w / w.sum()
                proto = (embs * w[:, None]).sum(axis=0)
            else:
                proto = embs.mean(axis=0)
            prototypes[cls] = proto / (np.linalg.norm(proto) + 1e-8)
    return prototypes


def load_class3_weights(train_csv_path: Path, synth_csv_path: Path, labels: np.ndarray, algospeak_cls: int = 3):
    train_df = pd.read_csv(train_csv_path).dropna(subset=["text"]).reset_index(drop=True)

    if len(train_df) != len(labels):
        raise ValueError(
            f"train.csv has {len(train_df)} rows but train.pt has {len(labels)} labels. "
            "Re-run encoder_prep.py to rebuild the .pt files from the current CSVs."
        )

    if "deny_terms_transformed" in train_df.columns:
        terms = [
            str(train_df.loc[i, "deny_terms_transformed"]) if labels[i] == algospeak_cls and pd.notna(train_df.loc[i, "deny_terms_transformed"]) else ""
            for i in range(len(train_df))
        ]
    else:
        synth_df = pd.read_csv(synth_csv_path)
        text_to_term = dict(zip(
            synth_df["algospeak_text"].astype(str).str.strip(),
            synth_df["deny_terms_transformed"].astype(str),
        ))
        terms = [
            text_to_term.get(str(train_df.loc[i, "text"]).strip(), "unknown") if labels[i] == algospeak_cls else ""
            for i in range(len(train_df))
        ]

    algo_terms  = [t for t, lbl in zip(terms, labels) if lbl == algospeak_cls]
    term_counts = pd.Series(algo_terms).value_counts().to_dict()
    weights     = np.ones(len(labels), dtype=np.float32)
    for i, (term, lbl) in enumerate(zip(terms, labels)):
        if lbl == algospeak_cls:
            weights[i] = 1.0 / term_counts.get(term, 1)

    matched    = sum(1 for t, lbl in zip(terms, labels) if lbl == algospeak_cls and t not in ("", "unknown"))
    total_algo = int((labels == algospeak_cls).sum())
    if total_algo > 0:
        print(f"  Algospeak weight lookup: {matched}/{total_algo} matched ({100*matched/total_algo:.1f}%)")
        print(f"  Unique deny terms in algospeak training: {len(term_counts)}")
    return weights


def predict(embeddings, prototypes):
    sim    = embeddings @ prototypes.T
    scores = torch.softmax(torch.tensor(sim / 0.1), dim=-1).numpy()
    preds  = sim.argmax(axis=1)
    return preds, scores


def plot_confusion_matrix(y_true, y_pred, out_path, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix -> {out_path}")


def plot_roc_curves(y_true, scores, num_classes, out_path, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for cls in range(num_classes):
        y_bin = (y_true == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, scores[:, cls])
        ax.plot(fpr, tpr, color=colors[cls], lw=2,
                label=f"{class_names[cls]} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ROC curves -> {out_path}")


def evaluate_split(encoder, prototypes, split, cfg, device, results_dir, class_names):
    print(f"\n--- Evaluating {split} split ---")
    dataset = load_dataset(BASE_DIR / cfg["prepared_dir"] / f"{split}.pt")
    embs, labels = get_embeddings(encoder, dataset, cfg["batch_size"], device)
    preds, scores = predict(embs, prototypes)

    csv_df = pd.read_csv(BASE_DIR / cfg[f"{split}_csv"]).dropna(subset=["text"]).reset_index(drop=True)
    pd.DataFrame({
        "text":            csv_df["text"].astype(str),
        "true_label":      [class_names[i] for i in labels],
        "predicted_label": [class_names[i] for i in preds],
        "correct":         labels == preds,
    }).to_csv(results_dir / f"predictions_{split}.csv", index=False)

    acc    = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    plot_confusion_matrix(labels, preds, results_dir / f"confusion_matrix_{split}.png", class_names)
    plot_roc_curves(labels, scores, cfg["num_classes"], results_dir / f"roc_curves_{split}.png", class_names)

    aucs = {}
    for cls in range(cfg["num_classes"]):
        y_bin = (labels == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, scores[:, cls])
        aucs[class_names[cls]] = round(auc(fpr, tpr), 4)

    return {
        "split":       split,
        "accuracy":    round(acc, 4),
        "macro_f1":    round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class":   {class_names[i]: {"precision": round(report[class_names[i]]["precision"], 4),
                                          "recall":    round(report[class_names[i]]["recall"], 4),
                                          "f1":        round(report[class_names[i]]["f1-score"], 4)}
                        for i in range(cfg["num_classes"])},
        "auc_per_class": aucs,
        "mean_auc":    round(np.mean(list(aucs.values())), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if not args.notes:
        print("WARNING: no --notes provided. Strongly recommended for tracking runs.")

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    global CLASS_NAMES
    if "class_labels" in cfg:
        CLASS_NAMES = [cfg["class_labels"].get(i) or cfg["class_labels"].get(str(i)) for i in range(cfg["num_classes"])]

    run_id       = datetime.now().strftime("%Y%m%d_%H%M")
    results_base = BASE_DIR / cfg["results_dir"]
    run_dir      = results_base / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    ckpt_dir  = BASE_DIR / cfg["checkpoint_dir"]
    ckpt_path = ckpt_dir / "best_model.pt"
    encoder   = load_unsupervised_encoder(ckpt_path, cfg, device)

    ckpt_raw  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_meta = {"epoch": int(ckpt_raw.get("epoch", -1)), "val_loss": round(float(ckpt_raw.get("val_loss", -1)), 6)}

    print("\nBuilding class prototypes from training set...")
    train_ds = load_dataset(BASE_DIR / cfg["prepared_dir"] / "train.pt")
    train_embs, train_labels = get_embeddings(encoder, train_ds, cfg["batch_size"], device)

    synth_csv     = EXP_DIR / "data" / "synthetic_algospeak.csv"
    algospeak_cls = cfg["num_classes"] - 1
    weights       = load_class3_weights(BASE_DIR / cfg["train_csv"], synth_csv, train_labels, algospeak_cls=algospeak_cls)
    prototypes    = build_prototypes(train_embs, train_labels, cfg["num_classes"], weights=weights, algospeak_cls=algospeak_cls)

    np.save(run_dir / "prototypes.npy", prototypes)
    np.save(results_base / "prototypes.npy", prototypes)
    print(f"  Prototypes saved -> {run_dir / 'prototypes.npy'}")

    dataset_info = {}
    for split in ["train", "val", "test"]:
        split_df = pd.read_csv(BASE_DIR / cfg[f"{split}_csv"]).dropna(subset=["text"])
        label_col = "label" if "label" in split_df.columns else "classification"
        counts = split_df[label_col].value_counts().sort_index().to_dict()
        dataset_info[split] = {"total": len(split_df), "classes": {f"class_{int(k)}": int(v) for k, v in counts.items()}}

    all_results = []
    for split in ["val", "test"]:
        result = evaluate_split(encoder, prototypes, split, cfg, device, run_dir, CLASS_NAMES)
        all_results.append(result)

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    shutil.copy(metrics_path, results_base / "metrics.json")
    print(f"\nMetrics -> {metrics_path}")

    run_info = {
        "run_id":     run_id,
        "date":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes":      args.notes,
        "checkpoint": ckpt_meta,
        "config":     {k: v for k, v in cfg.items() if not k.endswith("_csv") and k != "prepared_dir"},
        "dataset":    dataset_info,
        "results":    {r["split"]: {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                                     "mean_auc": r["mean_auc"], "per_class": r["per_class"]}
                       for r in all_results},
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Run info  -> {run_dir / 'run_info.json'}")

    print("\n=== SUMMARY ===")
    for r in all_results:
        print(f"{r['split']:6s} | acc={r['accuracy']:.4f} | macro_f1={r['macro_f1']:.4f} | mean_auc={r['mean_auc']:.4f}")
    print(f"\nRun saved to: {run_dir}")

    print("\n=== Example inference ===")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    examples = [
        "I had a great day today, went for a walk in the park.",
        "I'm going to k!ll that n!gga if he shows up again.",
        "she posted an onlyfans link in her bio",
        "gonna unalive myself fr fr cant take this anymore",
    ]
    for text in examples:
        enc = tokenizer(emoji.demojize(text), padding="max_length", truncation=True,
                        max_length=cfg["max_length"], return_tensors="pt")
        with torch.no_grad():
            emb = encoder(enc["input_ids"].to(device), enc["attention_mask"].to(device)).cpu().numpy()
        pred = int((emb @ prototypes.T).argmax())
        print(f"  [{CLASS_NAMES[pred]}] {text[:70]}")


if __name__ == "__main__":
    main()
