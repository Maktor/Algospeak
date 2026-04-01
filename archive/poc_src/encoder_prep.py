#!/usr/bin/env python3
"""
Data Preparation for Dual-Encoder RBE

Loads the pre-built balanced splits from build_dataset.py (data/splits/)
and converts them to PyTorch tensors for training.

Classes:
  0 = Allowed
  1 = Offensive Language
  2 = Mature Content
  3 = Algospeak

Input:  <project_root>/data/splits/train.csv, val.csv, test.csv
Output: poc/data/prepared/train.pt, val.pt, test.pt
"""

import sys
import logging
import torch
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "Allowed", 1: "Offensive Language", 2: "Mature Content", 3: "Algospeak"}


def load_split(path: Path) -> pd.DataFrame:
    """Load a CSV split, validate it has 'text' and 'classification' columns."""
    if not path.exists():
        logger.error(f"Split file not found: {path}")
        logger.error("Run build_dataset.py --stage 1 and --stage 2 first.")
        sys.exit(1)

    df = pd.read_csv(path)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)

    if "classification" not in df.columns:
        logger.error(f"'classification' column missing in {path}")
        sys.exit(1)

    df["label"] = df["classification"].astype(int)

    # Warn about any unexpected class values
    known = set(LABEL_NAMES.keys())
    unknown = set(df["label"].unique()) - known
    if unknown:
        logger.warning(f"Unexpected class values in {path.name}: {unknown} — these rows will be kept")

    return df


def log_distribution(name: str, df: pd.DataFrame):
    counts = df["label"].value_counts().sort_index()
    logger.info(f"{name}: {len(df)} posts")
    for cls, n in counts.items():
        label = LABEL_NAMES.get(cls, f"class_{cls}")
        logger.info(f"  Class {cls} ({label}): {n}")


def df_to_pt(df: pd.DataFrame) -> dict:
    return {
        "texts":  df["text"].tolist(),
        "labels": torch.tensor(df["label"].values, dtype=torch.long),
    }


def main():
    repo_root   = Path(__file__).parent.parent.parent
    splits_dir  = repo_root / "data" / "splits"
    prepared_dir = Path(__file__).parent.parent / "data" / "prepared"
    prepared_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Loading splits from {splits_dir}")

    for split_name in ["train", "val", "test"]:
        df = load_split(splits_dir / f"{split_name}.csv")
        log_distribution(split_name, df)
        torch.save(df_to_pt(df), prepared_dir / f"{split_name}.pt")
        logger.info(f"  Saved → {prepared_dir / f'{split_name}.pt'}")

    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
