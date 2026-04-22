"""
experiments/four_class/src/analyze_candidates.py

Pre-generation analysis script. Reads the reclassified CSV and reports:
  - How many unpaired class 1 / class 2 posts have detectable deny terms
  - Which deny terms appear and how often (to ensure good technique coverage)
  - How many posts from each class are needed to balance class 3 (algospeak)
    up to the class 2 count (the limiting class)
  - A recommended balanced selection, with term diversity stats

Does NOT call the API or modify any files. Run this before generate_algospeak.py
to understand what you're working with.

Usage:
    uv run python experiments/four_class/src/analyze_candidates.py \\
        --input experiments/four_class/data/reclassified_unfinished.csv
"""

import sys
import io
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Import shared detection utilities (inflection-aware, consistent with generation)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DENY_LIST_TERMS, detect_deny_terms

# ── Greedy selection ───────────────────────────────────────────────────────────
def _greedy_select(df: pd.DataFrame, target: int) -> pd.DataFrame:
    """
    Greedy coverage-maximization selection.

    At each step scores every remaining candidate post as:
        sum(1 / (times_term_already_covered + 1))  for each deny term in the post

    Posts containing rare/uncovered terms score highest, so underrepresented
    terms get pulled in early. Ties are broken by the pre-shuffle (random_state=42).
    """
    term_coverage: Counter = Counter()
    selected_idx: list = []
    pool = df.sample(frac=1, random_state=42).copy()

    while len(selected_idx) < target and len(pool) > 0:
        scores = pool["deny_terms"].apply(
            lambda terms: sum(1.0 / (term_coverage[t] + 1) for t in terms)
        )
        best = scores.idxmax()
        selected_idx.append(best)
        term_coverage.update(pool.loc[best, "deny_terms"])
        pool = pool.drop(best)

    return df.loc[selected_idx]


# ── Main ───────────────────────────────────────────────────────────────────────
def analyze(input_file: str, save_candidates: str | None = None):
    df = pd.read_csv(input_file, encoding="utf-8-sig")
    print(f"\nLoaded {len(df)} rows from {input_file}")
    print(f"Columns: {df['classification'].value_counts().sort_index().to_dict()}\n")

    # ── Class counts and target ────────────────────────────────────────────────
    counts = df["classification"].value_counts().sort_index()
    class2_total = int(counts.get(2, 0))
    class3_total = int(counts.get(3, 0))
    needed_total = max(0, class2_total - class3_total)
    needed_per_class = needed_total // 2
    remainder = needed_total % 2

    print("=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    for cls, cnt in counts.items():
        print(f"  class {cls}: {cnt:>5} rows")
    print(f"\n  Limiting class (class 2):         {class2_total}")
    print(f"  Existing algospeak (class 3):     {class3_total}")
    print(f"  Algospeak needed to balance:      {needed_total}")
    print(f"  Target from class 1:              {needed_per_class}")
    print(f"  Target from class 2:              {needed_per_class + remainder}")
    print()

    # ── Unpaired candidates ────────────────────────────────────────────────────
    unpaired = df[df["pair_ID"].isna() & df["classification"].isin([1, 2])].copy()
    paired   = df[df["pair_ID"].notna() & df["classification"].isin([1, 2])]

    print("=" * 60)
    print("PAIRING STATUS (class 1 + 2 only)")
    print("=" * 60)
    for cls in [1, 2]:
        n_paired   = int((paired["classification"] == cls).sum())
        n_unpaired = int((unpaired["classification"] == cls).sum())
        print(f"  class {cls}:  {n_paired} already paired  |  {n_unpaired} unpaired (candidates)")
    print()

    # ── Deny-term detection on unpaired posts ─────────────────────────────────
    print("Detecting deny terms in unpaired candidates...", flush=True)
    unpaired = unpaired.copy()
    unpaired["deny_terms"] = unpaired["text"].astype(str).apply(detect_deny_terms)
    unpaired["n_terms"]    = unpaired["deny_terms"].apply(len)
    unpaired["has_terms"]  = unpaired["n_terms"] > 0

    print()
    print("=" * 60)
    print("DENY-TERM COVERAGE IN UNPAIRED CANDIDATES")
    print("=" * 60)
    for cls in [1, 2]:
        sub      = unpaired[unpaired["classification"] == cls]
        usable   = sub[sub["has_terms"]]
        unusable = sub[~sub["has_terms"]]
        print(f"  class {cls}:  {len(usable):>4} usable (have deny terms)  |  "
              f"{len(unusable):>4} unusable (no deny terms detected)")
    print()

    # ── Per-class term frequency ───────────────────────────────────────────────
    for cls in [1, 2]:
        sub    = unpaired[(unpaired["classification"] == cls) & (unpaired["has_terms"])]
        target = needed_per_class + (remainder if cls == 2 else 0)
        usable = len(sub)
        shortfall = max(0, target - usable)

        print("=" * 60)
        print(f"CLASS {cls} — top deny terms across usable candidates (n={usable})")
        print("=" * 60)

        term_counter: Counter = Counter()
        for terms in sub["deny_terms"]:
            term_counter.update(terms)

        for term, cnt in term_counter.most_common(30):
            bar = "█" * min(cnt, 40)
            print(f"  {term:<30} {cnt:>4}  {bar}")

        print()
        print(f"  Need {target} from class {cls} — have {usable} usable candidates", end="")
        if shortfall:
            print(f"  *** SHORTFALL: {shortfall} short ***")
        else:
            print(f"  (surplus: {usable - target})")
        print()

    # ── Greedy diverse selection ───────────────────────────────────────────────
    print("=" * 60)
    print("RECOMMENDED BALANCED SELECTION (greedy term-diversity)")
    print("=" * 60)
    print("  Algorithm: at each step, pick the post whose deny terms are")
    print("  currently least-covered — rare terms get priority over saturated ones.")
    print()

    selected_rows = []
    for cls in [1, 2]:
        target = needed_per_class + (remainder if cls == 2 else 0)
        sub    = unpaired[(unpaired["classification"] == cls) & (unpaired["has_terms"])].copy()

        selected = _greedy_select(sub, target)
        selected_rows.append(selected)

        sel_terms: Counter = Counter()
        for terms in selected["deny_terms"]:
            sel_terms.update(terms)

        # Compare against naive (sort by n_terms) to show improvement
        naive = sub.sample(frac=1, random_state=42).sort_values("n_terms", ascending=False).head(target)
        naive_terms: Counter = Counter()
        for terms in naive["deny_terms"]:
            naive_terms.update(terms)

        print(f"  Class {cls}: selected {len(selected)} of {len(sub)} usable")
        print(f"  Unique deny terms — greedy: {len(sel_terms)}  |  naive (most-terms-first): {len(naive_terms)}")
        print(f"  Top terms in selection:")
        for term, cnt in sel_terms.most_common(20):
            print(f"    {term:<30} {cnt:>4}")
        print()

    print()

    # ── Overall term diversity across full selection ───────────────────────────
    if selected_rows:
        all_selected = pd.concat(selected_rows)
        all_terms: Counter = Counter()
        for terms in all_selected["deny_terms"]:
            all_terms.update(terms)

        print("=" * 60)
        print(f"COMBINED SELECTION SUMMARY")
        print("=" * 60)
        print(f"  Total selected:          {len(all_selected)}")
        print(f"  Unique deny terms:       {len(all_terms)}")
        print(f"  Total term occurrences:  {sum(all_terms.values())}")
        print(f"  Avg terms per post:      {sum(all_terms.values()) / len(all_selected):.2f}")
        print()

        # Terms with 0 coverage — in deny list but not found in selection
        missing_terms = [t for t in DENY_LIST_TERMS if t not in all_terms]
        if missing_terms:
            print(f"  Deny-list terms with NO coverage in selection ({len(missing_terms)}):")
            for t in missing_terms:
                print(f"    {t}")
        else:
            print("  All deny-list terms have at least one post in selection.")

        # Save selected candidates if requested
        if save_candidates:
            out_path = Path(save_candidates)
            all_selected.drop(columns=["deny_terms", "n_terms", "has_terms"]).to_csv(
                out_path, index=False, encoding="utf-8-sig"
            )
            print(f"\n  Saved {len(all_selected)} selected candidates → {out_path}")

    print()
    print("=" * 60)
    print("PROJECTED CLASS DISTRIBUTION AFTER GENERATION")
    print("=" * 60)
    final_class3 = class3_total + needed_total
    for cls, cnt in counts.items():
        label = f"class {cls}"
        marker = " ← target" if cls == 2 else (" ← projected" if cls == 3 else "")
        display = final_class3 if cls == 3 else cnt
        print(f"  {label}: {display:>5}{marker}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze class 1/2 candidates for algospeak generation."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the reclassified CSV.",
    )
    parser.add_argument(
        "--save-candidates", default=None, metavar="PATH",
        help="If set, write the greedy-selected candidates to this CSV path. "
             "Pass that file to generate_algospeak.py --input to generate only the diverse set.",
    )
    args = parser.parse_args()
    analyze(args.input, save_candidates=args.save_candidates)
