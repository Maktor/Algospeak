#!/usr/bin/env python3
"""
Balanced Dataset Pipeline — 25/25/25/25 across classes 0/1/2/3

Two-stage workflow:

  Stage 1: Prepare
    - Filters short posts (< MIN_WORDS words)
    - Samples a balanced class 0/1/2 pool
    - Writes algospeak_sources.csv — class 1+2 posts queued for synthetic generation
    - Writes data/splits/class012_pool.csv for inspection

  Stage 2: Build
    - Loads filtered class 0/1/2 pool
    - Loads synthetic_algospeak.csv (class 3, must exist)
    - Removes data leakage: excludes algospeak source posts from class 1/2 training data
    - Balances all four classes to N = min(available_class3, --target)
    - Splits into train / val / test with balanced class distribution in EACH split
    - Writes data/splits/train.csv, val.csv, test.csv, full_dataset.csv

Usage:
  # Stage 1 — prepare sources, choose how many algospeak posts to generate
  python build_dataset.py --stage 1 --target 10000

  # (then run synthetic_data.py on algospeak_sources.csv to generate class 3)
  #   Update INPUT_FILE in synthetic_data.py to point at algospeak_sources.csv
  #   and run: python synthetic_data.py
  #   Estimated cost for 10k posts with gpt-4o: ~$55

  # Stage 2 — build final splits once synthetic_algospeak.csv is ready
  python build_dataset.py --stage 2 --target 10000

  # Both stages in sequence (only works if synthetic data already exists)
  python build_dataset.py --stage all --target 10000

Options:
  --target N        Target posts per class (default: 10000)
  --min-words N     Minimum word count to keep a post (default: 5)
  --val-frac F      Fraction of data for validation (default: 0.1)
  --test-frac F     Fraction of data for test (default: 0.1)
  --seed N          Random seed (default: 42)
  --stage {1,2,all} Which stage to run (default: all)
"""

import re
import csv
import json
import random
import logging
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────
BASE_DIR              = Path(__file__).parent
CLASSIFIED_CSV        = BASE_DIR / "final_classified_data" / "classified_data.csv"
SYNTHETIC_CSV         = BASE_DIR / "Algospeak_experiment" / "synthetic_algospeak_v2.csv"
OUTPUT_DIR            = BASE_DIR / "data" / "splits"
POOL_CSV              = OUTPUT_DIR / "class012_pool.csv"
ALGO_SOURCES_CSV      = OUTPUT_DIR / "algospeak_sources.csv"

CLASS_LABELS = {0: "Allowed", 1: "Offensive Language", 2: "Mature Content", 3: "Algospeak"}


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(str(text).split())


def load_classified(min_words: int) -> pd.DataFrame:
    """Load classified_data.csv, drop NaN texts, filter short posts."""
    logger.info(f"Loading {CLASSIFIED_CSV}...")
    df = pd.read_csv(CLASSIFIED_CSV)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)

    before = len(df)
    df = df[df["text"].apply(word_count) >= min_words].copy()
    removed = before - len(df)
    logger.info(f"  Loaded {before} posts → kept {len(df)} after min {min_words}-word filter "
                f"(removed {removed}, {removed/before*100:.1f}%)")

    logger.info("  Class distribution after filter:")
    for cls, label in CLASS_LABELS.items():
        if cls == 3:
            continue
        n = (df["classification"] == cls).sum()
        logger.info(f"    Class {cls} ({label}): {n:,}")

    return df



def load_synthetic(min_words: int) -> pd.DataFrame | None:
    """Load synthetic_algospeak.csv, filter short posts, return as class 3."""
    if not SYNTHETIC_CSV.exists():
        logger.warning(f"synthetic_algospeak.csv not found at {SYNTHETIC_CSV}")
        return None

    df = pd.read_csv(SYNTHETIC_CSV)
    if "algospeak_text" not in df.columns:
        logger.error("synthetic_algospeak.csv missing 'algospeak_text' column")
        return None

    df = df.dropna(subset=["algospeak_text"])
    df["text"] = df["algospeak_text"].astype(str)
    before = len(df)
    df = df[df["text"].apply(word_count) >= min_words].copy()
    df["classification"] = 3

    logger.info(f"  Loaded {before} synthetic posts → {len(df)} after min {min_words}-word filter")
    return df[["text", "classification"]]


def balanced_sample(df_by_class: dict, n: int, seed: int) -> pd.DataFrame:
    """Sample exactly n posts per class (or all if fewer available)."""
    parts = []
    for cls, df in df_by_class.items():
        actual = min(n, len(df))
        if actual < n:
            logger.warning(f"  Class {cls}: only {len(df)} posts available, using all (target was {n})")
        sampled = df.sample(actual, random_state=seed)
        parts.append(sampled)
    return pd.concat(parts, ignore_index=True)


def stratified_split(df: pd.DataFrame, val_frac: float, test_frac: float,
                     seed: int) -> tuple:
    """
    Split df into train/val/test maintaining class balance in EACH split.
    Uses two-pass stratified splitting so class ratios are preserved.
    """
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df["classification"], random_state=seed
    )
    # val_frac is relative to full dataset, so adjust for train_val size
    val_frac_adjusted = val_frac / (1 - test_frac)
    train, val = train_test_split(
        train_val, test_size=val_frac_adjusted, stratify=train_val["classification"],
        random_state=seed
    )
    return train, val, test


def print_split_stats(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    logger.info("\n  Split sizes and class distribution:")
    logger.info(f"  {'Split':<8} {'Total':>7}  " +
                "  ".join(f"Cls{c}" for c in sorted(CLASS_LABELS.keys())))
    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts = split["classification"].value_counts().sort_index()
        row = f"  {name:<8} {len(split):>7}"
        for c in sorted(CLASS_LABELS.keys()):
            row += f"  {counts.get(c, 0):>5}"
        logger.info(row)


# ─────────────────────────────────────────────────────────────────────
# STAGE 1 — Prepare sources
# ─────────────────────────────────────────────────────────────────────

def stage1(args):
    """
    Select which class 1+2 posts to convert to algospeak (class 3).
    Outputs algospeak_sources.csv for synthetic_data.py to process.
    Also writes class012_pool.csv — the class 0/1/2 posts available for training,
    with algospeak source posts removed to prevent data leakage.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Prepare algospeak sources")
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    df = load_classified(args.min_words)

    # Pick args.target posts evenly from class 1 and class 2 as algospeak sources.
    # These will be fed to synthetic_data.py to generate class 3 algospeak versions.
    # They are excluded from the class 1/2 training pool to prevent leakage
    # (model must not train on originals and test on their algospeak versions).
    violation_pool = df[df["classification"].isin([1, 2])]
    per_class = args.target // 2

    sources_c1 = violation_pool[violation_pool["classification"] == 1].sample(
        min(per_class, (violation_pool["classification"] == 1).sum()),
        random_state=args.seed
    )
    sources_c2 = violation_pool[violation_pool["classification"] == 2].sample(
        min(args.target - len(sources_c1),
            (violation_pool["classification"] == 2).sum()),
        random_state=args.seed
    )
    all_sources = pd.concat([sources_c1, sources_c2], ignore_index=True)

    logger.info(f"\n  Selected {len(all_sources)} algospeak source posts "
                f"({len(sources_c1)} class 1, {len(sources_c2)} class 2)")

    all_sources[["text", "classification"]].to_csv(ALGO_SOURCES_CSV, index=False)
    logger.info(f"  Written to: {ALGO_SOURCES_CSV}")

    # Write class 0/1/2 pool with source posts removed (leakage prevention)
    source_texts = set(all_sources["text"].tolist())
    pool = df[~df["text"].isin(source_texts)][["text", "classification"]]
    pool.to_csv(POOL_CSV, index=False)
    logger.info(f"  Written class 0/1/2 pool ({len(pool):,} posts) to: {POOL_CSV}")

    logger.info("\n  Available posts per class after source exclusion:")
    for cls in [0, 1, 2]:
        n = (pool["classification"] == cls).sum()
        logger.info(f"    Class {cls} ({CLASS_LABELS[cls]}): {n:,}")
    logger.info(f"    Class 3 ({CLASS_LABELS[3]}): {len(all_sources)} (after generation)")

    logger.info(f"\n  Next steps:")
    logger.info(f"  1. Run synthetic_data.py --overwrite to generate class 3 algospeak:")
    logger.info(f"       python synthetic_data.py --input data/splits/algospeak_sources.csv "
                f"--model gpt-4o-mini --overwrite")
    logger.info(f"  2. Once done, build the final splits:")
    logger.info(f"       python build_dataset.py --stage 2 --target {args.target}")


# ─────────────────────────────────────────────────────────────────────
# STAGE 2 — Build final splits
# ─────────────────────────────────────────────────────────────────────

def stage2(args):
    """
    Build the final balanced 25/25/25/25 train/val/test splits.
    Requires:
      - data/splits/class012_pool.csv (from stage 1)
      - Algospeak_experiment/synthetic_algospeak.csv (from synthetic_data.py)
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Build balanced splits")
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # ── Load class 0/1/2 pool ────────────────────────────────────────
    if not POOL_CSV.exists():
        logger.error(f"class012_pool.csv not found. Run stage 1 first.")
        return

    pool = pd.read_csv(POOL_CSV)
    pool = pool.dropna(subset=["text"])
    pool["text"] = pool["text"].astype(str)
    logger.info(f"Loaded class 0/1/2 pool: {len(pool):,} posts")

    # ── Reclaim skipped algospeak sources back into the pool ─────────
    # synthetic_data.py skips posts with no deny-list terms. Those posts were
    # excluded from the pool in stage 1 but never got algospeak versions —
    # they'd just be wasted. Find them and return them to their original class.
    if ALGO_SOURCES_CSV.exists():
        sources = pd.read_csv(ALGO_SOURCES_CSV)
        sources["text"] = sources["text"].astype(str)

        class3_raw = load_synthetic(args.min_words)
        transformed_texts = set(class3_raw["text"].tolist()) if class3_raw is not None else set()

        # A source post is "skipped" if its original text doesn't appear in the
        # synthetic output. (synthetic_data.py writes original_text for each row.)
        if SYNTHETIC_CSV.exists():
            synth_df = pd.read_csv(SYNTHETIC_CSV)
            if "original_text" in synth_df.columns:
                processed_originals = set(synth_df["original_text"].dropna().astype(str))
            else:
                processed_originals = set()
        else:
            processed_originals = set()

        skipped_sources = sources[~sources["text"].isin(processed_originals)].copy()
        if len(skipped_sources) > 0:
            logger.info(f"\nReclaiming {len(skipped_sources)} skipped algospeak source posts "
                        f"back into class 1/2 pool")
            logger.info(f"  (These were designated for algospeak generation but had no deny-list "
                        f"terms, so synthetic_data.py skipped them.)")
            pool = pd.concat([pool, skipped_sources[["text", "classification"]]],
                             ignore_index=True)
            logger.info(f"  Pool size after reclaim: {len(pool):,}")
        else:
            logger.info("No skipped sources to reclaim — all designated posts were processed.")

    # ── Load class 3 synthetic data ──────────────────────────────────
    class3 = load_synthetic(args.min_words)
    if class3 is None or len(class3) == 0:
        logger.error("No class 3 data available. Generate synthetic algospeak first.")
        return

    logger.info(f"Class 3 available: {len(class3)} posts")

    # ── Determine N ──────────────────────────────────────────────────
    # N is min(user target, available class3, smallest class 0/1/2 pool)
    pool_counts = {cls: (pool["classification"] == cls).sum() for cls in [0, 1, 2]}
    n_available = {0: pool_counts[0], 1: pool_counts[1], 2: pool_counts[2], 3: len(class3)}
    n = min(args.target, *n_available.values())

    logger.info(f"\nTarget N per class: {args.target}")
    logger.info(f"Available per class: {n_available}")
    logger.info(f"Final N per class: {n}")

    if n < args.target:
        limiting = min(n_available, key=n_available.get)
        logger.warning(f"  N capped at {n} (limited by class {limiting} — "
                       f"{CLASS_LABELS[limiting]})")
        logger.warning("  To increase N: generate more synthetic data or reclassify more posts.")

    # ── Sample balanced subsets ──────────────────────────────────────
    df_by_class = {
        cls: pool[pool["classification"] == cls][["text", "classification"]]
        for cls in [0, 1, 2]
    }
    df_by_class[3] = class3

    balanced = balanced_sample(df_by_class, n, args.seed)
    balanced = balanced.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    logger.info(f"\nBalanced dataset: {len(balanced)} total posts ({n} per class)")

    # ── Split into train / val / test ────────────────────────────────
    train, val, test = stratified_split(balanced, args.val_frac, args.test_frac, args.seed)
    print_split_stats(train, val, test)

    # Verify balance
    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts = split["classification"].value_counts()
        if counts.max() - counts.min() > max(3, len(split) * 0.01):
            logger.warning(f"  {name} split may not be perfectly balanced: {counts.to_dict()}")

    # ── Write outputs ────────────────────────────────────────────────
    full_path    = OUTPUT_DIR / "full_dataset.csv"
    train_path   = OUTPUT_DIR / "train.csv"
    val_path     = OUTPUT_DIR / "val.csv"
    test_path    = OUTPUT_DIR / "test.csv"

    balanced.to_csv(full_path, index=False)
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    logger.info(f"\nWritten:")
    logger.info(f"  {full_path}  ({len(balanced)} rows)")
    logger.info(f"  {train_path}  ({len(train)} rows)")
    logger.info(f"  {val_path}  ({len(val)} rows)")
    logger.info(f"  {test_path}  ({len(test)} rows)")
    logger.info("\n  Next: run poc/src/encoder_prep.py to tokenize and prepare for training.")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build balanced 25/25/25/25 dataset")
    parser.add_argument("--stage", choices=["1", "2", "all"], default="all",
                        help="Which stage to run (default: all)")
    parser.add_argument("--target", type=int, default=10000,
                        help="Target posts per class (default: 10000)")
    parser.add_argument("--min-words", type=int, default=5,
                        help="Minimum word count to keep a post (default: 5)")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--test-frac", type=float, default=0.1,
                        help="Fraction of data for test (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    logger.info(f"Config: target={args.target} per class, min_words={args.min_words}, "
                f"val={args.val_frac}, test={args.test_frac}, seed={args.seed}")

    if args.stage in ("1", "all"):
        stage1(args)

    if args.stage in ("2", "all"):
        stage2(args)

    if args.stage == "all":
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
