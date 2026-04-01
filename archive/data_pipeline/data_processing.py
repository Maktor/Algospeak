#!/usr/bin/env python3
"""
Data Processing Module for Algospeak Detection

This script processes social media posts to detect algospeak (coded/obfuscated language)
and moderate content using rule-based pattern matching and dictionary lookup.

FUNCTIONALITY:
- Loads algospeak dictionary from JSON file
- Detects moderation violations (self-harm, threats, violence, sexual content, hate, etc.)
- Identifies algospeak terms using substring matching and regex patterns
- Categorizes posts as "clean", "questionable", or "moderated"
- Outputs labeled data to JSONL and categorized text files

RUN:
    python data_processing.py
    # Processes posts.txt using algospeak_dictionary.json
    # Outputs: generated_data/clean_posts.txt, generated_data/questionable_posts.txt, generated_data/moderated_posts.txt, generated_data/labeled_posts.jsonl

CUSTOMIZE:
    run(input_file="your_posts.txt", dict_file="your_dictionary.json")
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional


# -----------------------------
# Utilities
# -----------------------------

VS_CHARS_RE = re.compile(r"[\uFE0E\uFE0F]")  # emoji variation selectors

def normalize_text(s: str) -> str:
    # NFKC helps normalize funky unicode variants
    s = unicodedata.normalize("NFKC", s)
    # Remove emoji variation selectors so matching is consistent
    s = VS_CHARS_RE.sub("", s)
    return s

def casefold(s: str) -> str:
    return normalize_text(s).casefold()

def is_wordish(term: str) -> bool:
    # "wordish" = contains letters/digits
    return any(ch.isalnum() for ch in term)

def compile_word_term_pattern(term: str) -> re.Pattern:
    """
    Builds a safer pattern for word-like algospeak tokens:
    - whole word matching (boundaries)
    - allows common separators (space/_/-) inside the term if term contains them
    """
    t = casefold(term)

    # Escape literal characters first
    escaped = re.escape(t)

    # Optional flexibility: allow separators between letters for common stylings
    # Example: "un-alive", "un_alive", "un alive" should match "unalive" term if you include it.
    # If your dictionary has "unalive" and the post has "un-alive", this helps.
    escaped = escaped.replace(r"\-", r"[\s_\-]*").replace(r"\_", r"[\s_\-]*").replace(r"\ ", r"[\s_\-]*")

    # Word boundaries: \b is imperfect for unicode, but good enough for English + digits.
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)

@dataclass
class Match:
    kind: str                  # "moderation" or "algospeak"
    category: str              # e.g., "self_harm", "sexual", "benign"
    token: str                 # matched token/pattern keyword or algospeak term
    meaning: Optional[str] = None

# -----------------------------
# Moderation regex rules (tune as you go)
# -----------------------------

MOD_RULES: Dict[str, List[re.Pattern]] = {
    # CRITICAL / HIGH PRIORITY
    "self_harm": [
        re.compile(r"\b(kms|ctb)\b", re.IGNORECASE),
        re.compile(r"\b(kill\s+myself|end\s+it\s+all|not\s+worth\s+living)\b", re.IGNORECASE),
        re.compile(r"\b(self[\s\-]?harm|cut\s+myself|hurt\s+myself|suicid(e|al))\b", re.IGNORECASE),
    ],
    "threats": [
        re.compile(r"\b(i\s*(will|['’]?ll)\s*(kill|hurt|shoot|stab))\b", re.IGNORECASE),
        re.compile(r"\b(death\s+threat)\b", re.IGNORECASE),
        re.compile(r"\brip\s+your\b", re.IGNORECASE),
    ],
    "violence": [
        # Note: keep "kill" out of here as a standalone — too many idioms (“killed it”).
        re.compile(r"\b(murder|shoot|stab|execute|slaughter|bloodbath|assault)\b", re.IGNORECASE),
        re.compile(r"\b(rape|sexual\s+assault|gang\s+rape)\b", re.IGNORECASE),
    ],
    "sexual": [
        re.compile(r"\b(porn|pornography|sex\s*tape|sex\s*video|xxx|nsfw)\b", re.IGNORECASE),
        re.compile(r"\b(nude|naked|topless|onlyfans)\b", re.IGNORECASE),
        re.compile(r"\b(blowjob|orgasm|masturbat|creampie)\b", re.IGNORECASE),
    ],
    "hate": [
        # WARNING: "hate" alone is too broad, don’t use it as a trigger by itself.
        re.compile(r"\b(racist|racism|homophobic|transphobic|antisemitic|islamophobic|xenophobic)\b", re.IGNORECASE),
        re.compile(r"\b(supremacist|nazi|klan)\b", re.IGNORECASE),
        re.compile(r"\bslur\b", re.IGNORECASE),
    ],
    "harassment": [
        re.compile(r"\b(doxx?|swat(ting)?|stalking|harass(ment)?|cyberbully(ing)?)\b", re.IGNORECASE),
        re.compile(r"\b(raid|cancel|expose)\b", re.IGNORECASE),
    ],
    "spam": [
        re.compile(r"\b(click\s+here|buy\s+now|limited\s+offer|make\s+money\s+fast|act\s+now)\b", re.IGNORECASE),
    ],
}

# Severity ranking
SEVERITY = ["self_harm", "threats", "violence", "sexual", "hate", "harassment", "spam"]

# Label policy
# - moderated: any of self_harm / threats / violence / sexual / hate  OR multiple medium signals
# - questionable: algospeak (non-benign) OR harassment/spam OR weak single signals
# - clean: nothing
MODERATED_CATS = {"self_harm", "threats", "violence", "sexual", "hate"}


# -----------------------------
# Algospeak dictionary handling
# -----------------------------

MEANING_TO_CAT = [
    ("self_harm", re.compile(r"\b(suicide|suicidal|self[\s\-]?harm|kill\s+myself)\b", re.IGNORECASE)),
    ("violence",  re.compile(r"\b(kill|murder|shoot|stab|rape|assault|blood)\b", re.IGNORECASE)),
    ("sexual",    re.compile(r"\b(porn|sex|nude|naked|onlyfans|xxx|nsfw)\b", re.IGNORECASE)),
    ("hate",      re.compile(r"\b(nazi|klan|racis(t|m)|antisemit|homophob|transphob|slur)\b", re.IGNORECASE)),
    ("drugs",     re.compile(r"\b(weed|coke|heroin|opioid|meth|fentanyl|drug)\b", re.IGNORECASE)),
]

def meaning_category(meaning: str) -> str:
    m = meaning or ""
    for cat, rx in MEANING_TO_CAT:
        if rx.search(m):
            return cat
    return "benign"

def load_algospeak(filepath: str) -> Tuple[Dict[str, List[str]], List[Tuple[str, re.Pattern]]]:
    """
    Returns:
      term -> [meaning1, meaning2, ...]
      compiled patterns for wordish terms
    """
    raw = json.load(open(filepath, "r", encoding="utf-8"))
    term_to_meanings: Dict[str, List[str]] = {}
    word_patterns: List[Tuple[str, re.Pattern]] = []

    for term, entries in raw.items():
        term_n = normalize_text(term)
        meanings: List[str] = []
        if isinstance(entries, list):
            for e in entries:
                if isinstance(e, dict) and e.get("meaning"):
                    meanings.append(e["meaning"])
        if meanings:
            term_to_meanings[term_n] = meanings
            if is_wordish(term_n):
                word_patterns.append((term_n, compile_word_term_pattern(term_n)))

    return term_to_meanings, word_patterns


def detect_algospeak(text: str,
                    term_to_meanings: Dict[str, List[str]],
                    word_patterns: List[Tuple[str, re.Pattern]]) -> List[Match]:
    """
    Only triggers when the actual term appears in the text.
    - emoji/non-word terms: substring search
    - word terms: regex boundary matching (handles "un-alive" variants a bit)
    """
    t_norm = normalize_text(text)
    t_cf = t_norm.casefold()

    matches: List[Match] = []

    # 1) Non-word terms (mostly emoji / symbols): substring search
    # To keep this fast, iterate over terms but only those that are not wordish.
    for term, meanings in term_to_meanings.items():
        if is_wordish(term):
            continue
        if term and term in t_norm:
            for meaning in meanings:
                cat = meaning_category(meaning)
                matches.append(Match(kind="algospeak", category=cat, token=term, meaning=meaning))

    # 2) Word terms: regex
    for term, rx in word_patterns:
        if rx.search(t_cf):
            for meaning in term_to_meanings.get(term, []):
                cat = meaning_category(meaning)
                matches.append(Match(kind="algospeak", category=cat, token=term, meaning=meaning))

    return matches


# -----------------------------
# Moderation detection
# -----------------------------

def detect_moderation(text: str) -> List[Match]:
    t = normalize_text(text)
    hits: List[Match] = []
    for cat, patterns in MOD_RULES.items():
        for rx in patterns:
            if rx.search(t):
                hits.append(Match(kind="moderation", category=cat, token=rx.pattern))
    return hits


# -----------------------------
# Label decision logic
# -----------------------------

def choose_label(mod_hits: List[Match], algo_hits: List[Match]) -> Tuple[str, float, List[str]]:
    """
    Returns: (label, confidence, reasons)
    label in {"clean", "questionable", "moderated"}
    confidence is a rough heuristic 0..1
    """
    reasons: List[str] = []

    mod_cats = {m.category for m in mod_hits}
    algo_cats = {a.category for a in algo_hits if a.category != "benign"}

    # Hard rules first
    if mod_cats & MODERATED_CATS:
        worst = next((c for c in SEVERITY if c in mod_cats), "high")
        reasons.append(f"moderation-hit:{worst}")
        return "moderated", 0.95, reasons

    # Algospeak that decodes to harmful topics -> questionable (or moderated if you want)
    if algo_cats:
        worst = next((c for c in SEVERITY if c in algo_cats), "algospeak")
        reasons.append(f"algospeak-harmful:{worst}")
        # if you prefer “harmful algospeak” to be moderated, change label here
        return "questionable", 0.80, reasons

    # Medium-risk moderation-only categories
    if "harassment" in mod_cats or "spam" in mod_cats:
        worst = "harassment" if "harassment" in mod_cats else "spam"
        reasons.append(f"moderation-hit:{worst}")
        conf = 0.75 if worst == "harassment" else 0.60
        return "questionable", conf, reasons

    # Benign algospeak (flags, identity emoji, etc.) should not automatically taint a post
    if any(a.category == "benign" for a in algo_hits):
        reasons.append("algospeak-benign")
        # Still “clean” unless you want to route benign coded language as questionable
        return "clean", 0.55, reasons

    return "clean", 0.90, ["no-hits"]


def label_post(text: str,
               term_to_meanings: Dict[str, List[str]],
               word_patterns: List[Tuple[str, re.Pattern]]) -> Dict[str, Any]:
    mod_hits = detect_moderation(text)
    algo_hits = detect_algospeak(text, term_to_meanings, word_patterns)
    label, conf, reasons = choose_label(mod_hits, algo_hits)

    return {
        "text": text,
        "label": label,
        "confidence": round(conf, 3),
        "reasons": reasons,
        "moderation_hits": [m.__dict__ for m in mod_hits],
        "algospeak_hits": [a.__dict__ for a in algo_hits],
    }


def run(input_file="posts.txt", dict_file="algospeak_dictionary.json"):
    term_to_meanings, word_patterns = load_algospeak(dict_file)

    # Ensure output directory exists
    os.makedirs("generated_data", exist_ok=True)

    out_clean = open("generated_data/clean_posts.txt", "w", encoding="utf-8")
    out_q = open("generated_data/questionable_posts.txt", "w", encoding="utf-8")
    out_mod = open("generated_data/moderated_posts.txt", "w", encoding="utf-8")
    out_jsonl = open("generated_data/labeled_posts.jsonl", "w", encoding="utf-8")

    counts = {"clean": 0, "questionable": 0, "moderated": 0}

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rec = label_post(text, term_to_meanings, word_patterns)
            out_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

            label = rec["label"]
            counts[label] += 1

            if label == "clean":
                out_clean.write(text + "\n")
            elif label == "questionable":
                out_q.write(text + "\n")
            else:
                out_mod.write(text + "\n")

    out_clean.close()
    out_q.close()
    out_mod.close()
    out_jsonl.close()

    print("Done:", counts)


if __name__ == "__main__":
    run()
