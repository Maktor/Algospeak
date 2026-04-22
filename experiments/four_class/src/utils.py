"""
experiments/four_class/src/utils.py

Shared deny-term detection utilities used by both analyze_candidates.py
and generate_algospeak.py. Keeping detection logic in one place ensures
analysis and generation always match exactly.

Inflection coverage:
  - Basic suffixes:  s, 's, es, d, ed, ing, en, er
  - y → ies/ied/ying  (tranny→trannies, pussy→pussies, sexy→sexier)
  - Doubled final consonant before ing/ed/er  (stab→stabbing, shit→shitting)
"""

import re
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_SRC_DIR  = Path(__file__).resolve().parent
_DATA_DIR = _SRC_DIR.parent / "data"

DENY_LIST_FILES = [
    _DATA_DIR / "deny_list.txt",
]

# ── Deny list loading ──────────────────────────────────────────────────────────
def load_deny_list(path: Path) -> list:
    if not path.exists():
        print(f"WARNING: deny list not found at {path}")
        return []
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    return [ln.strip().lower() for ln in lines if ln.strip()]

DENY_LIST_TERMS: list = sorted(set(
    t for p in DENY_LIST_FILES for t in load_deny_list(p)
))

# ── Pattern builder ────────────────────────────────────────────────────────────
_VOWELS = set("aeiou")

def _build_term_pattern(term: str) -> str:
    """
    Build a regex that matches a deny term and its common inflected forms.

    Covers:
      - Basic:  s, 's, es, d, ed, ing, en, er
      - y-drop: tranny→trannies, pussy→pussies, sexy→sexier/sexiest
      - Doubled final consonant: stab→stabbing, shit→shitting, run→running
    """
    esc = re.escape(term)
    patterns = [rf"\b{esc}(?:s|'s|es|d|ed|ing|en|er)?\b"]

    # y → ies / ied / ying / ier / iest  (only when stem ends in consonant+y)
    if term.endswith("y") and len(term) > 1 and term[-2] not in _VOWELS:
        stem = re.escape(term[:-1])
        patterns.append(rf"\b{stem}(?:ies|ied|ying|ier|iest)\b")

    # Doubled final consonant before ing/ed/er
    # Applies to CVC-pattern words (consonant-vowel-consonant),
    # excluding terms ending in w, x, h, y (those don't double).
    if (len(term) >= 3
            and term[-1] not in _VOWELS
            and term[-2] in _VOWELS
            and term[-3] not in _VOWELS
            and term[-1] not in "wxhy"):
        doubled = re.escape(term + term[-1])
        patterns.append(rf"\b{doubled}(?:ing|ed|er)\b")

    return "|".join(f"(?:{p})" for p in patterns) if len(patterns) > 1 else patterns[0]


# Pre-compile one pattern per deny term at import time
_DENY_PATTERNS: dict = {
    term: re.compile(_build_term_pattern(term), re.IGNORECASE)
    for term in DENY_LIST_TERMS if term
}

# ── Markup protection ──────────────────────────────────────────────────────────
_PROTECT_RE = re.compile(r"https?://\S+|@\w+|#\w+")

def protect_markup(text: str) -> tuple:
    """Replace @mentions, #hashtags, URLs with __PN__ placeholders."""
    tokens, counter = {}, [0]
    def replacer(m):
        tok = m.group()
        for ph, val in tokens.items():
            if val == tok:
                return ph
        ph = f"__P{counter[0]}__"
        tokens[ph] = tok
        counter[0] += 1
        return ph
    return _PROTECT_RE.sub(replacer, text), tokens

def restore_markup(text: str, tokens: dict) -> str:
    for ph, orig in tokens.items():
        text = text.replace(ph, orig)
    return text

# ── Detection ──────────────────────────────────────────────────────────────────
def detect_deny_terms(text: str) -> list:
    """
    Return deny-list terms (canonical forms) found in text.
    Protects @mentions, #hashtags, URLs before matching.
    Catches exact forms plus common inflections (see _build_term_pattern).
    """
    masked, _ = protect_markup(text)
    found = [term for term, pat in _DENY_PATTERNS.items() if pat.search(masked)]
    return sorted(set(found), key=len, reverse=True)

def get_untransformed(text: str, terms: list) -> list:
    """
    Return which of `terms` are still present (untransformed) in `text`.
    Uses the same inflection-aware patterns as detect_deny_terms.
    """
    masked, _ = protect_markup(text)
    return [t for t in terms if _DENY_PATTERNS[t].search(masked)]
