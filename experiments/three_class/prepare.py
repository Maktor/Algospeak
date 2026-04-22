#!/usr/bin/env python3
"""
experiments/three_class/prepare.py

Builds the 3-class experiment dataset and train/val/test splits from scratch.

Class mapping:
  0 (Allowed)          -> 0 (Allowed)
  1 (Offensive Lang.)  -> 1 (Harmful Content)
  2 (Mature Content)   -> 1 (Harmful Content)
  3 (Algospeak)        -> 2 (Algospeak)

Data sources:
  - data/splits/class012_pool_reclassified.csv      (LLM-reclassified class 0/1/2 posts)
  - data/splits/class012_pairs_reclassified.csv     (class 1/2 originals + algospeak pairs)
  - Algospeak_experiment/synthetic_algospeak.csv    (extra unpaired algospeak, not reclassified)

Dataset construction (1:1:1 balanced — N = total algospeak available):
  - Class 2 (Algospeak): all paired algospeak + unpaired algospeak from synthetic file
  - Class 1 (Harmful):   N sampled from pair originals (all pairs maintained for split integrity)
  - Class 0 (Allowed):   N sampled from pool class 0

Outputs:
  experiments/three_class/data/final_dataset.csv  (full dataset, shareable)
  experiments/three_class/data/splits/train.csv
  experiments/three_class/data/splits/val.csv
  experiments/three_class/data/splits/test.csv

Usage:
  uv run python experiments/three_class/prepare.py
"""

import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
EXP_DIR   = Path(__file__).parent
DATA_DIR  = EXP_DIR / "data"
OUT_DIR   = DATA_DIR / "splits"

POOL_CSV   = BASE_DIR / "data" / "splits" / "class012_pool_reclassified.csv"
PAIRS_CSV  = BASE_DIR / "data" / "splits" / "class012_pairs_reclassified.csv"
ORIG_CSV   = BASE_DIR / "final_classified_data" / "classified_data.csv"
SYNTH_CSV  = BASE_DIR / "Algospeak_experiment" / "synthetic_algospeak.csv"

VAL_FRAC  = 0.1
TEST_FRAC = 0.1
SEED      = 42
RARE_TERM_THRESHOLD = 5

CLASS_LABELS = {0: "Allowed", 1: "Harmful Content", 2: "Algospeak"}


# ─────────────────────────────────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────────────────────────────────
def build_final_dataset() -> pd.DataFrame:
    pool  = pd.read_csv(POOL_CSV,  encoding="utf-8-sig")
    pairs = pd.read_csv(PAIRS_CSV, encoding="utf-8-sig")
    orig  = pd.read_csv(ORIG_CSV,  encoding="utf-8-sig")

    source_lookup = dict(zip(orig["text"].astype(str).str.strip(), orig["source"]))

    pair_orig_texts = set(pairs["text"].astype(str).str.strip())
    rows = []

    # ── Class 2 (Algospeak) — paired + unpaired synthetic algospeak ───
    # Paired: from pairs CSV (have a matching Harmful original)
    paired_algo_texts = set()
    for i, row in pairs.iterrows():
        pid = f"pair_{i:05d}"
        paired_algo_texts.add(str(row["algospeak_text"]).strip())
        rows.append({
            "text":                   row["algospeak_text"],
            "classification":         2,
            "source":                 "synthetic",
            "pair_id":                pid,
            "techniques":             row.get("techniques"),
            "deny_terms_transformed": row.get("deny_terms_transformed"),
        })

    # Unpaired: from synthetic_algospeak.csv (no reclassified original — algospeak text only)
    synth = pd.read_csv(SYNTH_CSV, encoding="utf-8-sig")
    unpaired = synth[
        ~synth["algospeak_text"].astype(str).str.strip().isin(paired_algo_texts)
    ]
    logger.info(f"  Algospeak: {len(pairs)} paired + {len(unpaired)} unpaired = {len(pairs)+len(unpaired)} total")
    for _, row in unpaired.iterrows():
        rows.append({
            "text":                   row["algospeak_text"],
            "classification":         2,
            "source":                 "synthetic",
            "pair_id":                None,
            "techniques":             row.get("techniques"),
            "deny_terms_transformed": row.get("deny_terms_transformed"),
        })

    N = len(pairs) + len(unpaired)  # total algospeak — sets N for 1:1:1
    logger.info(f"N = {N}/class (1:1:1 balanced)")

    # ── Class 1 (Harmful Content) — all pair originals ────────────────
    # All pairs maintain split integrity with their algospeak counterpart.
    # Supplement with pool class 1/2 if needed to reach N.
    for i, row in pairs.iterrows():
        pid = f"pair_{i:05d}"
        rows.append({
            "text":                   row["text"],
            "classification":         1,
            "source":                 source_lookup.get(str(row["text"]).strip(), "unknown"),
            "pair_id":                pid,
            "techniques":             None,
            "deny_terms_transformed": None,
        })

    n_harmful_so_far = len(pairs)
    if n_harmful_so_far < N:
        n_needed = N - n_harmful_so_far
        pool_c12 = pool[
            pool["classification"].isin([1, 2]) &
            (~pool["text"].astype(str).str.strip().isin(pair_orig_texts))
        ]
        n_pool = min(n_needed, len(pool_c12))
        logger.info(f"  Harmful: {n_harmful_so_far} pair originals + {n_pool} pool posts to reach N")
        for _, row in pool_c12.sample(n=n_pool, random_state=SEED).iterrows():
            rows.append({
                "text":                   row["text"],
                "classification":         1,
                "source":                 source_lookup.get(str(row["text"]).strip(), "unknown"),
                "pair_id":                None,
                "techniques":             None,
                "deny_terms_transformed": None,
            })
    else:
        logger.info(f"  Harmful: {n_harmful_so_far} pair originals (at N)")

    # ── Class 0 (Allowed) — sample N from pool class 0 ────────────────
    pool_c0 = pool[
        (pool["classification"] == 0) &
        (~pool["text"].astype(str).str.strip().isin(pair_orig_texts))
    ]
    n_allowed = min(N, len(pool_c0))
    if n_allowed < N:
        logger.warning(f"Only {len(pool_c0)} Allowed posts available, need {N}. Using all.")
    logger.info(f"  Allowed: {n_allowed} pool posts")

    for _, row in pool_c0.sample(n=n_allowed, random_state=SEED).iterrows():
        rows.append({
            "text":                   row["text"],
            "classification":         0,
            "source":                 source_lookup.get(str(row["text"]).strip(), "unknown"),
            "pair_id":                None,
            "techniques":             None,
            "deny_terms_transformed": None,
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    logger.info(f"Final dataset: {len(df)} rows")
    for cls, label in CLASS_LABELS.items():
        n = (df["classification"] == cls).sum()
        logger.info(f"  Class {cls} ({label}): {n}")

    return df


# ─────────────────────────────────────────────────────────────────────
# GROUP-AWARE STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────────────────
def group_aware_split(df: pd.DataFrame) -> tuple:
    rng = np.random.default_rng(SEED)

    # Deny term frequency across algospeak rows (class 2)
    term_counts: Counter = Counter()
    for val in df.loc[df["classification"] == 2, "deny_terms_transformed"].dropna():
        term_counts.update(str(val).split("|"))
    rare_terms = {t for t, c in term_counts.items() if c < RARE_TERM_THRESHOLD}

    pair_ids = df["pair_id"].dropna().unique()

    pair_meta = {}
    for pid in pair_ids:
        rows = df[df["pair_id"] == pid]
        force = False
        for val in rows["deny_terms_transformed"].dropna():
            if any(t.strip() in rare_terms for t in str(val).split("|")):
                force = True
                break
        tech = "unknown"
        techs = []
        for val in rows["techniques"].dropna():
            techs.extend(str(val).split("|"))
        if techs:
            global_tech_counts = Counter(
                t for v in df["techniques"].dropna() for t in str(v).split("|")
            )
            tech = min(set(techs), key=lambda t: global_tech_counts.get(t, 0))
        pair_meta[pid] = {"force_train": force, "technique": tech}

    forced = [pid for pid, m in pair_meta.items() if m["force_train"]]
    free   = [pid for pid, m in pair_meta.items() if not m["force_train"]]
    logger.info(f"  Pairs forced to train (rare terms): {len(forced)}")
    logger.info(f"  Pairs available for stratified split: {len(free)}")

    by_technique = {}
    for pid in free:
        tech = pair_meta[pid]["technique"]
        by_technique.setdefault(tech, []).append(pid)

    n_val_pairs  = int(len(free) * VAL_FRAC)
    n_test_pairs = int(len(free) * TEST_FRAC)

    val_pool, test_pool = [], []
    for tech, pids in by_technique.items():
        rng.shuffle(pids)
        n_t = max(1, round(len(pids) * TEST_FRAC))
        n_v = max(1, round(len(pids) * VAL_FRAC))
        test_pool.extend(pids[:n_t])
        val_pool.extend(pids[n_t:n_t + n_v])

    rng.shuffle(test_pool); rng.shuffle(val_pool)
    test_pair_ids = set(test_pool[:n_test_pairs])
    val_pair_ids  = set(val_pool[:n_val_pairs]) - test_pair_ids

    def get_split(row):
        pid = row.get("pair_id") if "pair_id" in row.index else None
        if pd.notna(pid):
            if pid in test_pair_ids: return "test"
            if pid in val_pair_ids:  return "val"
            return "train"
        return None

    df["_split"] = df.apply(get_split, axis=1)

    unpaired_idx = df[df["_split"].isna()].index.tolist()
    rng.shuffle(unpaired_idx)
    n_up = len(unpaired_idx)
    n_up_test = int(n_up * TEST_FRAC)
    n_up_val  = int(n_up * VAL_FRAC)
    for i, idx in enumerate(unpaired_idx):
        if i < n_up_test:            df.at[idx, "_split"] = "test"
        elif i < n_up_test+n_up_val: df.at[idx, "_split"] = "val"
        else:                        df.at[idx, "_split"] = "train"

    train = df[df["_split"] == "train"].drop(columns=["_split"])
    val   = df[df["_split"] == "val"].drop(columns=["_split"])
    test  = df[df["_split"] == "test"].drop(columns=["_split"])
    return train, val, test


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build and save final dataset
    df = build_final_dataset()
    final_path = DATA_DIR / "final_dataset.csv"
    df.to_csv(final_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved -> {final_path}")

    # Split
    logger.info("\nBuilding train/val/test splits...")
    train, val, test = group_aware_split(df)

    # Verify no cross-split pairs
    text_to_split = {t: "train" for t in train["text"].astype(str).str.strip()}
    text_to_split.update({t: "val"   for t in val["text"].astype(str).str.strip()})
    text_to_split.update({t: "test"  for t in test["text"].astype(str).str.strip()})
    pairs_df = df[df["pair_id"].notna()][["text", "pair_id"]]
    crossed = sum(
        1 for pid, grp in pairs_df.groupby("pair_id")
        if len({text_to_split.get(str(t).strip()) for t in grp["text"]} - {None}) > 1
    )
    logger.info(f"  Pairs crossing splits: {crossed}")

    # Write splits
    train_cols = ["text", "classification", "deny_terms_transformed"]
    val_test_cols = ["text", "classification"]
    train.reindex(columns=train_cols).to_csv(OUT_DIR / "train.csv", index=False)
    val.reindex(columns=val_test_cols).to_csv(OUT_DIR / "val.csv",   index=False)
    test.reindex(columns=val_test_cols).to_csv(OUT_DIR / "test.csv",  index=False)

    logger.info(f"\nSplits written to {OUT_DIR}")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts = split["classification"].value_counts().sort_index()
        label_str = " | ".join(f"{CLASS_LABELS[c]}={counts.get(c, 0)}" for c in sorted(CLASS_LABELS))
        logger.info(f"  {name}: {len(split)} rows — {label_str}")

    logger.info("\nNext:")
    logger.info("  uv run python poc/src/encoder_prep.py --config experiments/three_class/config.yaml")
    logger.info("  uv run python poc/src/train.py --config experiments/three_class/config.yaml --fresh")
    logger.info("  uv run python poc/src/inference.py --config experiments/three_class/config.yaml --notes '3-class experiment'")


if __name__ == "__main__":
    main()
