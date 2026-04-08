"""
encoder_prep.py

Tokenizes train/val/test splits for the dual BERTweet model.

For each post, two tokenized versions are produced:
  supervised:   "[CLASS_LABEL]: <text>"  (e.g. "Algospeak: gonna unalive myself")
  unsupervised: "<text>"                 (raw text, no label)

Both are saved as tensors in .pt files so train.py can load them without
re-tokenizing each run.

Usage:
    uv run python poc/src/encoder_prep.py
"""

import sys
import yaml
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import emoji

BASE_DIR = Path(__file__).resolve().parent.parent.parent

CLASS_PREFIX = {
    0: "Allowed:",
    1: "Offensive Language:",
    2: "Mature Content:",
    3: "Algospeak:",
}


def load_config() -> dict:
    with open(BASE_DIR / "poc" / "config.yaml") as f:
        return yaml.safe_load(f)


def prepare_split(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    split_name: str,
) -> dict:
    texts  = [emoji.demojize(t) for t in df["text"].astype(str).tolist()]
    labels = df["classification"].astype(int).tolist()

    # Supervised: prepend the ground-truth class label
    sup_texts = [f"{CLASS_PREFIX[lbl]} {txt}" for lbl, txt in zip(labels, texts)]

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
    cfg = load_config()

    model_name   = cfg["model_name"]
    max_length   = cfg["max_length"]
    prepared_dir = BASE_DIR / cfg["prepared_dir"]
    prepared_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {model_name}")
    # use_fast=False for BERTweet — the fast tokenizer has known issues with emoji/special chars
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    for split in ["train", "val", "test"]:
        csv_path = BASE_DIR / cfg[f"{split}_csv"]
        print(f"\nPreparing {split} split from {csv_path}...")

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str)

        class_dist = df["classification"].value_counts().sort_index().to_dict()
        print(f"  {len(df)} posts | class dist: {class_dist}")

        data = prepare_split(df, tokenizer, max_length, split)

        out_path = prepared_dir / f"{split}.pt"
        torch.save(data, out_path)
        print(f"  Saved -> {out_path}")

    print("\nEncoder prep complete.")


if __name__ == "__main__":
    main()
