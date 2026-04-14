"""
build_deny_hints.py

Reads algospeak_dictionary.json and deny_list_class1.txt + deny_list_class2.txt.
For each deny-list term, extracts known substitutions per technique from the dictionary.
Writes deny_term_hints.json: {deny_term: {pictorial: [...], abbreviation: [...], paraphrase: [...], unknown_spelling: [...]}}

Run: uv run python build_deny_hints.py
"""

import json
import re
import os
import unicodedata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_FILE = os.path.join(BASE_DIR, "algospeak_dictionary.json")
DENY_LIST_FILES = [
    os.path.join(BASE_DIR, "Algospeak_experiment", "deny_list_class1.txt"),
    os.path.join(BASE_DIR, "Algospeak_experiment", "deny_list_class2.txt"),
]
OUTPUT_FILE = os.path.join(BASE_DIR, "deny_term_hints.json")


# ─────────────────────────────────────────────────────────
# TECHNIQUE CLASSIFICATION
# ─────────────────────────────────────────────────────────

def _is_emoji_char(c: str) -> bool:
    cp = ord(c)
    cat = unicodedata.category(c)
    return (
        cat in ('So', 'Sk', 'Sm') or
        0x1F300 <= cp <= 0x1FAFF or   # misc symbols, pictographs, supplemental
        0x2600 <= cp <= 0x27BF or     # misc symbols (☠, ✡, etc.)
        0x1F000 <= cp <= 0x1F02F or   # mahjong
        0x1F0A0 <= cp <= 0x1F0FF or   # playing cards
        0x1F1E0 <= cp <= 0x1F1FF or   # regional indicator (flags)
        cp in (0x200D, 0xFE0F,        # ZWJ, variation selector
               0x20E3,                 # combining enclosing keycap
               0x1F3FB, 0x1F3FC, 0x1F3FD, 0x1F3FE, 0x1F3FF)  # skin tones
    )


def _has_emoji(s: str) -> bool:
    return any(_is_emoji_char(c) for c in s)


# Matches character substitution patterns used in unknown_spelling
_SYMBOL_SUB = re.compile(r'[@$!*]|\b[0-9]')
_DIGIT_IN_WORD = re.compile(r'[a-zA-Z][0-9]|[0-9][a-zA-Z]')  # e.g., b0mb, ch0k3d


def _is_abbreviation(s: str) -> bool:
    """Short all-uppercase token (2-6 alpha chars), possibly with dots/dashes."""
    alpha = re.sub(r'[^a-zA-Z]', '', s)
    return (
        2 <= len(alpha) <= 6 and
        alpha == alpha.upper() and
        not _has_emoji(s) and
        not _SYMBOL_SUB.search(s) and
        not _DIGIT_IN_WORD.search(s)
    )


def classify_key(key: str) -> str:
    """
    Classify an algospeak dictionary key into one of four technique buckets.
    Returns: 'pictorial' | 'abbreviation' | 'unknown_spelling' | 'paraphrase'
    """
    if _has_emoji(key):
        return 'pictorial'
    if _is_abbreviation(key):
        return 'abbreviation'
    if _SYMBOL_SUB.search(key) or _DIGIT_IN_WORD.search(key):
        return 'unknown_spelling'
    return 'paraphrase'


# ─────────────────────────────────────────────────────────
# MEANING → DENY TERM MATCHING
# ─────────────────────────────────────────────────────────

def parse_meanings(meaning_str: str) -> list[str]:
    """
    Split a meaning string on ' / ' and ',' to extract individual terms.
    Returns normalized lowercase strings.
    """
    # Split on " / " first, then optionally on " & "
    parts = re.split(r'\s*/\s*|\s*&\s*', meaning_str)
    return [p.strip().lower() for p in parts if p.strip()]


def match_to_deny_terms(meaning_str: str, deny_set: set[str]) -> list[str]:
    """
    Given a meaning string from the dictionary, return all deny-list terms it maps to.
    Handles multi-value meanings like "gun / firearms" or "shit / poop".
    """
    matched = []
    for candidate in parse_meanings(meaning_str):
        if candidate in deny_set:
            matched.append(candidate)
    return matched


# ─────────────────────────────────────────────────────────
# LOAD FILES
# ─────────────────────────────────────────────────────────

def load_deny_list(path: str) -> set[str]:
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = f.read().splitlines()
    return {line.strip().lower() for line in lines if line.strip()}


def load_dictionary(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────
# BUILD HINTS
# ─────────────────────────────────────────────────────────

TECHNIQUES = ['pictorial', 'abbreviation', 'unknown_spelling', 'paraphrase']


def build_hints(dictionary: dict, deny_set: set[str]) -> dict:
    """
    Returns {deny_term: {technique: [substitution, ...]}} for all deny-list terms
    that have at least one dictionary entry.
    """
    hints: dict[str, dict[str, list[str]]] = {}

    for algospeak_key, entries in dictionary.items():
        technique = classify_key(algospeak_key)

        for entry in entries:
            meaning = entry.get('meaning', '')
            matched_terms = match_to_deny_terms(meaning, deny_set)

            for deny_term in matched_terms:
                if deny_term not in hints:
                    hints[deny_term] = {t: [] for t in TECHNIQUES}

                bucket = hints[deny_term][technique]
                if algospeak_key not in bucket:
                    bucket.append(algospeak_key)

    return hints


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    deny_set = set()
    for f in DENY_LIST_FILES:
        loaded = load_deny_list(f)
        print(f"Loading deny list from: {f} ({len(loaded)} terms)")
        deny_set |= loaded
    print(f"  {len(deny_set)} deny-list terms loaded total.")

    print(f"Loading dictionary from: {DICT_FILE}")
    dictionary = load_dictionary(DICT_FILE)
    print(f"  {len(dictionary)} dictionary entries loaded.")

    hints = build_hints(dictionary, deny_set)

    # Report coverage
    covered = len(hints)
    total = len(deny_set)
    print(f"\nCoverage: {covered}/{total} deny-list terms have at least one dictionary hint.")

    by_technique = {t: 0 for t in TECHNIQUES}
    for term_hints in hints.values():
        for t in TECHNIQUES:
            if term_hints[t]:
                by_technique[t] += 1

    print("Terms with hints per technique:")
    for t, count in by_technique.items():
        print(f"  {t:20s}: {count}")

    # Strip empty technique buckets for cleanliness
    compact = {
        term: {t: subs for t, subs in buckets.items() if subs}
        for term, buckets in hints.items()
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {OUTPUT_FILE}")

    # Preview a few entries
    print("\nSample entries:")
    for term in list(compact.keys())[:15]:
        print(f"  {term!r}: {compact[term]}")


if __name__ == '__main__':
    main()
