#!/usr/bin/env python3
"""
Build rules_exemplars_dual_encoder.jsonl

Generates the ruleset used by poc/src/ruleset_matcher.py.

For each deny-list term, assigns it to class 1 (Offensive Language) or class 2
(Mature Content) based on which class has more matching posts in classified_data.csv.
Exemplars are real posts extracted from the training data where the rule fires —
following the RBE paper (Clarke et al., 2022) Section 3.2 exactly.

Class 0 (Allowed) rules use manually defined benign topic-category keywords.
Class 3 (Algospeak) rules are added separately after MLM discovery.

Output: ruleset+examplers/rules_exemplars_dual_encoder.jsonl

Usage:
  python build_ruleset.py
  python build_ruleset.py --exemplars 5   # exemplars per rule (default: 5)
  python build_ruleset.py --min-posts 3   # min posts to create a rule (default: 3)
  python build_ruleset.py --dry-run       # print stats without writing file
"""

import re
import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DENY_LIST_PATH  = BASE_DIR / "Algospeak_experiment" / "deny_list.txt"
DATA_CSV_PATH   = BASE_DIR / "final_classified_data" / "classified_data.csv"
OUTPUT_PATH     = BASE_DIR / "ruleset+examplers" / "rules_exemplars_dual_encoder.jsonl"

# ─────────────────────────────────────────────────────────────────────
# CLASS 0 — ALLOWED CONTENT
# Topic-category keywords manually chosen to fire ONLY on benign posts.
# Strategy: pick terms specific to everyday benign topics that will never
# appear in violation posts. Verified by spot-checking against the data.
# ─────────────────────────────────────────────────────────────────────
CLASS_0_PATTERNS = {
    # Food & cooking
    "recipe":     "food/cooking",
    "dinner":     "food/cooking",
    "breakfast":  "food/cooking",
    "lunch":      "food/cooking",
    "baked":      "food/cooking",
    "chef":       "food/cooking",
    "restaurant": "food/cooking",
    "grilled":    "food/cooking",
    # Sports
    "touchdown":     "sports",
    "championship":  "sports",
    "playoffs":      "sports",
    "tournament":    "sports",
    "scored":        "sports",
    "athlete":       "sports",
    "stadium":       "sports",
    # Entertainment
    "album":     "entertainment",
    "concert":   "entertainment",
    "episode":   "entertainment",
    "trailer":   "entertainment",
    "streaming": "entertainment",
    "festival":  "entertainment",
    "playlist":  "entertainment",
    # Daily life & positive
    "birthday":    "daily life",
    "vacation":    "daily life",
    "graduation":  "daily life",
    "workout":     "daily life",
    "commute":     "daily life",
    "weekend":     "daily life",
    "celebrate":   "daily life",
    "grateful":    "daily life",
    "excited":     "daily life",
    "proud":       "daily life",
    # Nature & pets
    "puppy":    "nature/pets",
    "sunset":   "nature/pets",
    "garden":   "nature/pets",
    "hiking":   "nature/pets",
    "camping":  "nature/pets",
    "flowers":  "nature/pets",
    # Arts & creativity
    "painting":   "arts",
    "photography": "arts",
    "drawing":    "arts",
    "writing":    "arts",
    "poetry":     "arts",
}

# ─────────────────────────────────────────────────────────────────────
# MANUAL CLASS OVERRIDES
# Force specific deny-list terms to a particular class (1 or 2), overriding
# the frequency-based assignment. Use when the term is clearly domain-specific
# regardless of what the count says.
# ─────────────────────────────────────────────────────────────────────
CLASS_OVERRIDES = {
    # Slurs → always class 1 (offensive language)
    "nigga": 1, "nigger": 1, "faggot": 1, "faggots": 1, "chink": 1, "spic": 1,
    "gook": 1, "beaner": 1, "coon": 1, "jigaboo": 1, "kike": 1, "paki": 1,
    "sand-nigger": 1, "timber-nigger": 1, "swamp-guinea": 1, "wetback": 1,
    "zipperhead": 1, "slanteye": 1, "buddhahead": 1, "camel-jockey": 1,
    "curry-muncher": 1, "dune-coon": 1, "petrol-sniffer": 1, "spearchucker": 1,
    "pikey": 1, "towelhead": 1, "cheese-monkey": 1, "nigerian-hurricane": 1,
    # Hate/extremism → class 1
    "hitler": 1, "nazi": 1, "swaztika": 1, "white power": 1, "supremacist": 1,
    "holocaust": 1, "concentration camp": 1, "hamas": 1, "communism": 1,
    "antisemitism": 1,
    # Sexual terms → always class 2 (mature content)
    "porn": 2, "pornhub": 2, "porno": 2, "pornography": 2, "xxx": 2,
    "onlyfans": 2, "only fans": 2, "blowjob": 2, "handjob": 2, "anal": 2,
    "orgasm": 2, "ejaculate": 2, "masturbation": 2, "masturbates": 2,
    "cunnilingus": 2, "fellatio": 2, "vagina": 2, "penis": 2, "clitoris": 2,
    "erotic": 2, "naked": 2, "nude": 2,
    # Drug terms → class 2 (mature/substance)
    "cocaine": 2, "marijuana": 2, "fentanyl": 2, "crack": 2, "lsd": 2,
    "ketamine": 2, "psilocybin": 2, "methamphetamine": 2, "heroin": 2,
    # Self-harm / suicide → class 2
    "suicide": 2, "self-harm": 2, "anorexia": 2, "suicide note": 2,
    "commit suicide": 2, "kill myself": 2, "kill oneself": 2, "suicidal": 2,
    "overdose": 2, "overdosed": 2, "pro-anorexia": 2, "eating disorder": 2,
}


def load_deny_list(path: Path) -> list:
    """Load deny list terms one per line (same logic as synthetic_data.py)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    terms = [line.strip().lower() for line in lines if line.strip()]
    return sorted(set(terms))


def word_boundary_match(term: str, text_lower: str) -> bool:
    """Case-insensitive word-boundary match (same as ruleset_matcher.py)."""
    return bool(re.search(r'\b' + re.escape(term) + r'\b', text_lower))


def count_and_sample(df_by_class: dict, term: str, n_exemplars: int,
                     used_exemplars: set) -> tuple:
    """
    For each class, count how many posts contain the term and collect exemplars.
    Uses vectorized pandas str.contains for speed across 200k+ posts.

    Returns:
        counts: dict of class_id -> match count
        exemplars: dict of class_id -> list of sampled post texts
    """
    pat = r'\b' + re.escape(term) + r'\b'
    counts = {}
    candidates = {}

    for cls, df in df_by_class.items():
        mask = df["text_lower"].str.contains(pat, regex=True, na=False)
        matches = df.loc[mask, "text"].tolist()
        counts[cls] = len(matches)
        available = [t for t in matches if t not in used_exemplars]
        sampled = random.sample(available, min(n_exemplars, len(available)))
        candidates[cls] = sampled

    return counts, candidates


def assign_class(term: str, counts: dict) -> int | None:
    """
    Determine which target class (1 or 2) to assign to a deny-list term.

    Priority:
    1. Manual override in CLASS_OVERRIDES
    2. Frequency: whichever of class 1 / class 2 has more matches
    3. If tied or both zero, return None (skip this term)
    """
    if term in CLASS_OVERRIDES:
        return CLASS_OVERRIDES[term]

    c1 = counts.get(1, 0)
    c2 = counts.get(2, 0)

    if c1 == 0 and c2 == 0:
        return None          # no matches in violation classes — skip
    if c1 > c2:
        return 1
    if c2 > c1:
        return 2
    # Tie: use whichever has a match (both equal and > 0 → default to 1)
    return 1


def build_deny_rules(deny_terms: list, df_by_class: dict,
                     n_exemplars: int, min_posts: int) -> list:
    """
    Build class 1 and class 2 rules from deny-list terms.

    Each term becomes at most one rule targeting the class it most strongly
    co-occurs with. Exemplars are drawn from classified_data.csv posts of
    that class where the pattern fires.
    """
    rules = []
    used_exemplars = set()   # No exemplar can belong to two rules (per paper)
    rule_counter = 1
    skipped_low = 0
    skipped_no_match = 0

    for i, term in enumerate(deny_terms):
        if (i + 1) % 50 == 0:
            logger.info(f"  Processing term {i+1}/{len(deny_terms)}...")
        counts, candidates = count_and_sample(df_by_class, term,
                                              n_exemplars, used_exemplars)
        target_class = assign_class(term, counts)

        if target_class is None:
            skipped_no_match += 1
            continue

        exemplars = candidates[target_class]
        if len(exemplars) < min_posts:
            skipped_low += 1
            continue

        # Claim these exemplars so no other rule reuses them
        for ex in exemplars:
            used_exemplars.add(ex)

        class_label = "Offensive Language" if target_class == 1 else "Mature Content"
        rules.append({
            "rule_id": f"R{rule_counter}",
            "target_class": target_class,
            "description": f"{class_label} — contains '{term}'",
            "trigger_logic": {"type": "keyword", "pattern": term},
            "exemplars": exemplars,
        })
        rule_counter += 1

    logger.info(f"Deny-list rules created: {len(rules)}")
    logger.info(f"  Skipped (no matches in violation classes): {skipped_no_match}")
    logger.info(f"  Skipped (fewer than {min_posts} usable exemplars): {skipped_low}")
    return rules, used_exemplars


def build_class0_rules(df_class0: pd.DataFrame, n_exemplars: int,
                       min_posts: int, used_exemplars: set,
                       start_counter: int) -> list:
    """
    Build class 0 (Allowed) rules using manually defined benign topic keywords.

    Exemplars are class 0 posts from classified_data.csv where the keyword fires.
    """
    rules = []
    rule_counter = start_counter

    for pattern, topic in CLASS_0_PATTERNS.items():
        pat = r'\b' + re.escape(pattern) + r'\b'
        mask = df_class0["text_lower"].str.contains(pat, regex=True, na=False)
        matches = df_class0.loc[mask, "text"].tolist()
        available = [t for t in matches if t not in used_exemplars]

        if len(available) < min_posts:
            logger.debug(f"Class 0 '{pattern}': only {len(available)} exemplars — skipping")
            continue

        exemplars = random.sample(available, min(n_exemplars, len(available)))
        for ex in exemplars:
            used_exemplars.add(ex)

        rules.append({
            "rule_id": f"R{rule_counter}",
            "target_class": 0,
            "description": f"Allowed content — {topic} topic — contains '{pattern}'",
            "trigger_logic": {"type": "keyword", "pattern": pattern},
            "exemplars": exemplars,
        })
        rule_counter += 1

    logger.info(f"Class 0 rules created: {len(rules)}")
    return rules


def main():
    parser = argparse.ArgumentParser(description="Build rules_exemplars_dual_encoder.jsonl")
    parser.add_argument("--exemplars", type=int, default=5,
                        help="Max exemplars per rule (default: 5)")
    parser.add_argument("--min-posts", type=int, default=3,
                        help="Min posts needed to create a rule (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible exemplar selection")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing output file")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load data ────────────────────────────────────────────────────
    logger.info("Loading deny list...")
    deny_terms = load_deny_list(DENY_LIST_PATH)
    logger.info(f"  {len(deny_terms)} terms loaded")

    logger.info("Loading classified_data.csv...")
    df = pd.read_csv(DATA_CSV_PATH)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    logger.info(f"  {len(df)} posts loaded")
    logger.info(f"  Class distribution: {df['classification'].value_counts().to_dict()}")

    # Pre-compute lowercase column once (avoids re-lowercasing per term)
    logger.info("Pre-computing lowercase text column...")
    df["text_lower"] = df["text"].str.lower()

    # Group DataFrames by class for fast lookup
    df_by_class = {cls: df[df["classification"] == cls] for cls in [0, 1, 2]}

    # ── Build deny-list rules (class 1 + 2) ─────────────────────────
    logger.info("\nBuilding deny-list rules (class 1 + 2)...")
    deny_rules, used_exemplars = build_deny_rules(
        deny_terms, df_by_class, args.exemplars, args.min_posts
    )

    # Count class distribution of deny rules
    class1_count = sum(1 for r in deny_rules if r["target_class"] == 1)
    class2_count = sum(1 for r in deny_rules if r["target_class"] == 2)
    logger.info(f"  Class 1 rules: {class1_count}")
    logger.info(f"  Class 2 rules: {class2_count}")

    # ── Build class 0 rules ──────────────────────────────────────────
    logger.info("\nBuilding class 0 (Allowed) rules...")
    class0_rules = build_class0_rules(
        df_by_class[0], args.exemplars, args.min_posts,
        used_exemplars, start_counter=len(deny_rules) + 1
    )

    # ── Combine all rules ────────────────────────────────────────────
    all_rules = deny_rules + class0_rules

    # ── Summary ──────────────────────────────────────────────────────
    by_class = defaultdict(int)
    for r in all_rules:
        by_class[r["target_class"]] += 1

    logger.info("\n" + "=" * 50)
    logger.info("RULESET SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total rules: {len(all_rules)}")
    for cls in sorted(by_class.keys()):
        label = {0: "Allowed", 1: "Offensive Language",
                 2: "Mature Content", 3: "Algospeak"}.get(cls, str(cls))
        logger.info(f"  Class {cls} ({label}): {by_class[cls]} rules")
    logger.info("  Class 3 (Algospeak): 0 rules — add after MLM discovery")
    logger.info("  Total unique exemplars used: "
                f"{sum(len(r['exemplars']) for r in all_rules)}")

    if args.dry_run:
        logger.info("\nDry run — no file written.")
        return

    # ── Write output ─────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rule in all_rules:
            f.write(json.dumps(rule, ensure_ascii=False) + "\n")

    logger.info(f"\nWritten: {OUTPUT_PATH}")
    logger.info("Next: run poc/src/ruleset_matcher.py to generate exemplar pairs.")
    logger.info("After algospeak discovery: add class 3 rules manually or via")
    logger.info("  merge_algo_rules.py (not yet written).")


if __name__ == "__main__":
    main()
