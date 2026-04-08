#!/usr/bin/env python3
"""
reclassify.py

Applies deny-list rule overrides on top of the team's manual classifications.

Rules (in priority order):
  1. Post contains a class 1 deny-list term → classification = 1
  2. Post contains a class 2 deny-list term (and NOT class 1) → classification = 2
  3. No deny-list match → keep the team's original classification

The team's label is authoritative when the deny lists are silent — posts with no
deny-list hits are never auto-reassigned to class 0.

Usage:
    uv run python reclassify.py
    uv run python reclassify.py --input data/splits/reclassified_012.csv --output data/splits/reclassified_final.csv
"""

import re
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
DEFAULT_IN  = BASE_DIR / "data" / "splits" / "reclassified_012.csv"
DEFAULT_OUT = BASE_DIR / "data" / "splits" / "reclassified_final.csv"
DENY1_PATH  = BASE_DIR / "Algospeak_experiment" / "deny_list_class1.txt"
DENY2_PATH  = BASE_DIR / "Algospeak_experiment" / "deny_list_class2.txt"

CLASS_LABELS = {0: "Allowed", 1: "Offensive Language", 2: "Mature Content"}


# ─────────────────────────────────────────────────────────────────────
# DENY LIST LOADING
# ─────────────────────────────────────────────────────────────────────

def load_deny_list(path: Path) -> list[str]:
    with open(path, encoding="utf-8-sig") as f:
        terms = [line.strip().lower() for line in f if line.strip()]
    unique = sorted(set(terms))
    logger.info(f"  Loaded {len(unique)} terms from {path.name}")
    return unique


# ─────────────────────────────────────────────────────────────────────
# MATCHING
# ─────────────────────────────────────────────────────────────────────

def _term_pattern(term: str) -> re.Pattern:
    """
    Build a compiled regex for a single deny-list term.

    - Single-word or hyphenated terms (start/end with alphanumeric):
        use lookahead/lookbehind word boundaries + suffix tolerance.
    - Terms starting with a non-alphanumeric char (e.g. "(mass) genocide"):
        use simple escaped substring match — no word boundaries at edges.
    """
    esc = re.escape(term)
    starts_word = term[0].isalnum() or term[0] == "-"
    ends_word   = term[-1].isalnum() or term[-1] == "-"

    prefix = r'(?<!\w)' if starts_word else ''
    suffix = r"(?:s|'s|es|d|ed|ing|er)?(?!\w)" if ends_word else ''
    return re.compile(prefix + esc + suffix, re.IGNORECASE)


def _build_patterns(terms: list[str]) -> list[tuple[str, re.Pattern]]:
    return [(t, _term_pattern(t)) for t in terms]


def any_match(text: str, patterns: list[tuple[str, re.Pattern]]) -> bool:
    for _, pat in patterns:
        if pat.search(text):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────
# RECLASSIFICATION
# ─────────────────────────────────────────────────────────────────────

def reclassify(df: pd.DataFrame,
               pats1: list[tuple[str, re.Pattern]],
               pats2: list[tuple[str, re.Pattern]]) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str)

    new_labels = []
    for text, orig in zip(df["text"], df["classification"]):
        t = text.lower()
        if any_match(t, pats1):
            new_labels.append(1)
        elif any_match(t, pats2):
            new_labels.append(2)
        else:
            new_labels.append(int(orig))

    df["classification"] = new_labels
    return df


# ─────────────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────────────

def print_stats(before: pd.Series, after: pd.Series, total: int):
    changed = (before != after).sum()
    logger.info(f"\n  Total posts:   {total:,}")
    logger.info(f"  Changed:       {changed:,}  ({changed / total * 100:.1f}%)")
    logger.info(f"  Unchanged:     {total - changed:,}")

    logger.info("\n  Class distribution (before → after):")
    for cls, label in CLASS_LABELS.items():
        b = (before == cls).sum()
        a = (after  == cls).sum()
        delta = a - b
        sign  = "+" if delta >= 0 else ""
        logger.info(f"    Class {cls} ({label}): {b:>7,} → {a:>7,}  ({sign}{delta:,})")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Apply deny-list overrides to team classifications")
    parser.add_argument("--input",  type=Path, default=DEFAULT_IN,
                        help=f"Input CSV (default: {DEFAULT_IN})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help=f"Output CSV (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RECLASSIFY: applying deny-list overrides")
    logger.info("=" * 60)

    # Load deny lists
    logger.info("\nLoading deny lists...")
    terms1 = load_deny_list(DENY1_PATH)
    terms2 = load_deny_list(DENY2_PATH)
    pats1  = _build_patterns(terms1)
    pats2  = _build_patterns(terms2)

    # Load input
    logger.info(f"\nLoading {args.input}...")
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["classification"] = df["classification"].astype(int)
    logger.info(f"  Loaded {len(df):,} posts")

    before = df["classification"].copy()

    # Reclassify
    logger.info("\nApplying rules...")
    df = reclassify(df, pats1, pats2)

    print_stats(before, df["classification"], len(df))

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"\nWritten: {args.output}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
