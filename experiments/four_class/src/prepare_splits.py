"""
experiments/four_class/src/prepare_splits.py

Combines reclassified_unfinished.csv + synthetic_algospeak.csv into a
balanced dataset and produces group-aware train/val/test splits.

Steps:
  1. Load both CSVs and normalize columns to: text, label, pair_id, deny_terms_transformed
  2. Link synthetic algospeak rows back to their source posts in reclassified
     by matching on text, so source posts get the same pair_id as their
     generated algospeak counterpart (prevents leakage across splits).
  3. Balance classes to the minimum class count by downsampling — paired
     posts are always kept; unpaired posts fill remaining slots.
  4. Group-aware split: posts sharing a pair_id land in the same split.
  5. Write train/val/test CSVs to experiments/four_class/data/splits/.

Usage:
    uv run python experiments/four_class/src/prepare_splits.py
    uv run python experiments/four_class/src/prepare_splits.py --val-ratio 0.1 --test-ratio 0.1 --seed 42
"""

import argparse
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
EXP_DIR  = Path(__file__).resolve().parents[1]

DATA_DIR    = EXP_DIR / "data"
SPLITS_DIR  = DATA_DIR / "splits"
RECLASSIFIED_CSV = DATA_DIR / "reclassified_unfinished.csv"
SYNTHETIC_CSV    = DATA_DIR / "synthetic_algospeak.csv"


# ── Load and normalize ─────────────────────────────────────────────────────────

def load_reclassified(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"classification": "label", "pair_ID": "pair_id"})
    df = df[["text", "label", "pair_id"]].copy()
    df["deny_terms_transformed"] = None
    df["text"] = df["text"].astype(str).str.strip()
    df = df.dropna(subset=["text"])
    df = df[df["text"] != ""]
    return df.reset_index(drop=True)


def load_synthetic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame({
        "text":                  df["algospeak_text"].astype(str).str.strip(),
        "label":                 df["label"].astype(int),
        "pair_id":               df["pair_id"].astype(str),
        "deny_terms_transformed": df["deny_terms_transformed"].astype(str),
        "_original_text":        df["original_text"].astype(str).str.strip(),
    })
    return out.dropna(subset=["text"]).reset_index(drop=True)


# ── Link synthetic pair_ids back to source posts ───────────────────────────────

def link_pair_ids(reclassified: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """
    For each synthetic row, find the matching source post in reclassified by
    text and assign the same pair_id — so the pair stays together in splits.
    """
    orig_to_pair = dict(zip(synthetic["_original_text"], synthetic["pair_id"]))
    updated = 0
    for i, row in reclassified.iterrows():
        if pd.isna(row["pair_id"]) and row["text"] in orig_to_pair:
            reclassified.at[i, "pair_id"] = orig_to_pair[row["text"]]
            updated += 1
    print(f"  Linked {updated} source posts to their synthetic pair_id.")
    return reclassified


# ── Balance classes ────────────────────────────────────────────────────────────

def balance_classes(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    counts  = df["label"].value_counts().sort_index()
    target  = int(counts.min())
    print(f"\nClass counts before balancing: {counts.to_dict()}")
    print(f"Target per class: {target}")

    kept = []
    for cls in sorted(df["label"].unique()):
        cls_df  = df[df["label"] == cls].copy()
        paired  = cls_df[cls_df["pair_id"].notna()]
        unpaired = cls_df[cls_df["pair_id"].isna()]

        n_paired = len(paired)
        if n_paired >= target:
            # More paired posts than target — sample from paired only
            kept.append(paired.sample(n=target, random_state=int(rng.integers(1e6))))
            print(f"  class {cls}: kept {target} (sampled from {n_paired} paired, had {len(unpaired)} unpaired)")
        else:
            n_unpaired_needed = target - n_paired
            if len(unpaired) >= n_unpaired_needed:
                sampled = unpaired.sample(n=n_unpaired_needed, random_state=int(rng.integers(1e6)))
            else:
                sampled = unpaired  # take all available
            kept.append(pd.concat([paired, sampled]))
            actual = n_paired + len(sampled)
            print(f"  class {cls}: kept {actual} ({n_paired} paired + {len(sampled)} unpaired)")

    balanced = pd.concat(kept).reset_index(drop=True)
    print(f"\nBalanced distribution: {balanced['label'].value_counts().sort_index().to_dict()}")
    return balanced


# ── Group-aware split ──────────────────────────────────────────────────────────

def group_aware_split(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assign posts to train/val/test such that all posts sharing a pair_id
    land in the same split. Stratifies at the group level by class.
    """
    # Build groups: pair_id → list of row indices
    # Unpaired posts each get a unique singleton group key
    groups: dict[str, list] = defaultdict(list)
    for i, row in df.iterrows():
        pid = row["pair_id"]
        if pd.isna(pid) or str(pid).strip() in ("", "nan"):
            key = f"_solo_{i}"
        else:
            key = str(pid)
        groups[key].append(i)

    # For each group, determine its "primary class" for stratification
    group_keys = list(groups.keys())
    group_class = []
    for key in group_keys:
        idx = groups[key]
        labels = df.loc[idx, "label"].tolist()
        # Prefer the non-algospeak class label for stratification
        non_algo = [l for l in labels if l != 3]
        group_class.append(non_algo[0] if non_algo else labels[0])

    group_df = pd.DataFrame({"key": group_keys, "cls": group_class})

    # Stratified split at group level per class
    train_idx, val_idx, test_idx = [], [], []

    for cls in sorted(group_df["cls"].unique()):
        cls_groups = group_df[group_df["cls"] == cls]["key"].tolist()
        rng.shuffle(cls_groups := np.array(cls_groups))
        cls_groups = cls_groups.tolist()

        n      = len(cls_groups)
        n_val  = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        n_train = n - n_val - n_test

        train_idx.extend(cls_groups[:n_train])
        val_idx.extend(cls_groups[n_train:n_train + n_val])
        test_idx.extend(cls_groups[n_train + n_val:])

    def collect(keys):
        rows = []
        for k in keys:
            rows.extend(groups[k])
        return df.loc[rows].copy()

    return collect(train_idx), collect(val_idx), collect(test_idx)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio",  type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading reclassified_unfinished.csv...")
    reclassified = load_reclassified(RECLASSIFIED_CSV)
    print(f"  {len(reclassified)} rows | class dist: {reclassified['label'].value_counts().sort_index().to_dict()}")

    print("\nLoading synthetic_algospeak.csv...")
    synthetic = load_synthetic(SYNTHETIC_CSV)
    print(f"  {len(synthetic)} rows (all class 3)")

    print("\nLinking synthetic pair_ids to source posts...")
    reclassified = link_pair_ids(reclassified, synthetic)

    print("\nCombining datasets...")
    synthetic_clean = synthetic.drop(columns=["_original_text"])
    combined = pd.concat([reclassified, synthetic_clean], ignore_index=True)
    combined["label"] = combined["label"].astype(int)
    print(f"  {len(combined)} total rows | class dist: {combined['label'].value_counts().sort_index().to_dict()}")

    print("\nBalancing classes...")
    balanced = balance_classes(combined, rng)

    print("\nSplitting (group-aware, stratified)...")
    train_df, val_df, test_df = group_aware_split(balanced, args.val_ratio, args.test_ratio, rng)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = SPLITS_DIR / f"{name}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        dist = df["label"].value_counts().sort_index().to_dict()
        print(f"  {name:5s}: {len(df):>5} rows | class dist: {dist} -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
