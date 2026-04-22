"""
experiments/four_class/src/encoder_prep.py

Tokenizes train/val/test splits for the four-class dual BERTweet experiment.
Reads from experiments/four_class/data/splits/ and writes .pt files to
experiments/four_class/data/prepared/.

Accepts both 'label' and 'classification' as the class column name.

Usage:
    uv run python experiments/four_class/src/encoder_prep.py
    uv run python experiments/four_class/src/encoder_prep.py --config path/to/config.yaml
"""

import sys
import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import emoji

BASE_DIR = Path(__file__).resolve().parents[3]  # project root
EXP_DIR  = Path(__file__).resolve().parents[1]  # experiments/four_class

DEFAULT_CLASS_PREFIX = {
    0: "Allowed:",
    1: "Obscene Language:",
    2: "Mature Content:",
    3: "Algospeak:",
}


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else EXP_DIR / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_labels(df: pd.DataFrame) -> list:
    if "label" in df.columns:
        return df["label"].astype(int).tolist()
    if "classification" in df.columns:
        return df["classification"].astype(int).tolist()
    raise ValueError(f"No 'label' or 'classification' column found. Columns: {df.columns.tolist()}")


def prepare_split(df, tokenizer, max_length, split_name, class_prefix=None):
    if class_prefix is None:
        class_prefix = DEFAULT_CLASS_PREFIX
    texts  = [emoji.demojize(t) for t in df["text"].astype(str).tolist()]
    labels = get_labels(df)

    sup_texts = [f"{class_prefix[lbl]} {txt}" for lbl, txt in zip(labels, texts)]

    print(f"  [{split_name}] Tokenizing supervised inputs ({len(sup_texts)} posts)...")
    sup_enc = tokenizer(
        sup_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    print(f"  [{split_name}] Tokenizing unsupervised inputs ({len(texts)} posts)...")
    unsup_enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "sup_ids":    sup_enc["input_ids"],
        "sup_mask":   sup_enc["attention_mask"],
        "unsup_ids":  unsup_enc["input_ids"],
        "unsup_mask": unsup_enc["attention_mask"],
        "labels":     torch.tensor(labels, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    class_labels = cfg.get("class_labels", {})
    CLASS_PREFIX = {int(k): f"{v}:" for k, v in class_labels.items()} if class_labels else DEFAULT_CLASS_PREFIX

    model_name   = cfg["model_name"]
    max_length   = cfg["max_length"]
    prepared_dir = BASE_DIR / cfg["prepared_dir"]
    prepared_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    for split in ["train", "val", "test"]:
        csv_path = BASE_DIR / cfg[f"{split}_csv"]
        print(f"\nPreparing {split} split from {csv_path}...")

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str)

        labels = get_labels(df)
        from collections import Counter
        class_dist = dict(sorted(Counter(labels).items()))
        print(f"  {len(df)} posts | class dist: {class_dist}")

        data = prepare_split(df, tokenizer, max_length, split, class_prefix=CLASS_PREFIX)

        out_path = prepared_dir / f"{split}.pt"
        torch.save(data, out_path)
        print(f"  Saved -> {out_path}")

    print("\nEncoder prep complete.")


if __name__ == "__main__":
    main()
