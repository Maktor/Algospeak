"""
synthetic_data_v2.py

Improved algospeak generator. Key changes over synthetic_data.py:

1. PER-TERM TECHNIQUE ASSIGNMENT: each deny term in a post gets its own independently
   sampled technique, producing mixed-technique posts that mirror real-world algospeak.

2. DICTIONARY-ANCHORED HINTS (from deny_term_hints.json built by build_deny_hints.py):
   - Pictorial: STRICTLY JSON-only. Term only gets pictorial if it has a known emoji in
     the hints file. If not, its technique is re-rolled. Same emoji every time (temp=0.0
     for pictorial terms).
   - Abbreviation: constrained to the known options from the hints file when available.
   - Paraphrase: soft-steered toward known community paraphrases when available.

3. Outputs to separate files so v1 data is never overwritten:
   - Algospeak_experiment/synthetic_algospeak_v2.csv
   - Algospeak_experiment/transformation_log_v2.csv
"""

import pandas as pd
import openai
import os
import re
import json
import time
import random
from dotenv import load_dotenv
import argparse

load_dotenv()

# ────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(BASE_DIR, "Algospeak_experiment")
INPUT_FILE = os.path.join(BASE_DIR, "data", "splits", "algospeak_sources.csv")
OUTPUT_FILE = os.path.join(EXPERIMENT_DIR, "synthetic_algospeak_v2.csv")
DENY_LIST_FILE = os.path.join(EXPERIMENT_DIR, "deny_list.txt")
TRANSFORMATION_LOG_FILE = os.path.join(EXPERIMENT_DIR, "transformation_log_v2.csv")
HINTS_FILE = os.path.join(BASE_DIR, "deny_term_hints.json")

os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# ────────────────────────────────────────────────
# OPENAI CLIENT
# ────────────────────────────────────────────────
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env or environment variables!")
print("API key loaded (starts with):", api_key[:7] + "...")

client = openai.OpenAI(api_key=api_key)

DEFAULT_MODEL = "gpt-4-turbo"
MAX_TERMS_PER_POST = 6       # cap on how many deny terms to transform per post
MAX_RETRIES = 3
MAX_UNIQUE_SUBSTITUTIONS = 6

# ────────────────────────────────────────────────
# ALGOSPEAK TECHNIQUES
# ────────────────────────────────────────────────
ALGOSPEAK_TECHNIQUES = [
    "unknown_spelling",
    "known_harmless",
    "abbreviation",
    "pictorial",
    "paraphrase",
    "phonetic",
]

TECHNIQUE_DESCRIPTIONS = {
    "unknown_spelling": (
        "Change spelling to an obscure or symbolic form using numbers, symbols, or unusual "
        "character substitutions. The result should look garbled/coded but still be "
        "recognizable to humans. Vary your approach — use leet-speak, symbol replacements, "
        "unicode lookalikes, repeated letters, or mixed strategies. "
        "Examples: abortion → @b0rt!0n, kill → k!ll, sex → $3x, suicide → su1c1d3, "
        "murder → murd3r, rape → r@pe, cocaine → c0k3, hate → h8, dead → d34d, "
        "gun → g*n, drugs → dr*gs, shooting → sh00ting."
    ),
    "known_harmless": (
        "Replace the sensitive word with an existing harmless word that sounds or looks "
        "similar. The replacement must be a real, innocent-sounding word — not a made-up "
        "one. Examples: porn → corn, sex → seggs, kill → keel, drugs → rugs, gun → fun, "
        "rape → grape, cocaine → co-cane, shooting → shouting, weed → seed, "
        "murder → birder, suicide → sidewalk, assault → a-salt."
    ),
    "abbreviation": (
        "Shorten the sensitive term to an abbreviation, acronym, coded shorthand, or "
        "'the ___ word' construction. Examples: sexual assault → SA, "
        "suicide → 'the s word', white supremacy → WS, marijuana → MJ, dead → D-E-D, "
        "rape → 'the r-word', murder → 'the m-word', marijuana → 420, "
        "shooting → 'the bang bang thing', cocaine → 'the white stuff'."
    ),
    "pictorial": (
        "Replace the sensitive word with EXACTLY ONE emoji that visually or conceptually "
        "represents it. The emoji used must come from the DICTIONARY LOCKS listed in the "
        "term assignments — do not substitute any other emoji."
    ),
    "paraphrase": (
        "Substitute the sensitive word or phrase with a euphemistic paraphrase that "
        "conveys the same meaning indirectly. Can be a single word or a short phrase. "
        "Examples: kill → unalive, die → go to sleep forever, suicide → self-deletion, "
        "rape → forceful encounter, murder → permanent vacation, "
        "shoot → go bang bang, dead → no longer with us, drugs → party favors, "
        "suicide → checking out early, gun → the strap, cocaine → nose candy, "
        "assault → unwanted situation."
    ),
    "phonetic": (
        "Replace the sensitive word with something that sounds phonetically similar when "
        "spoken aloud, even if spelled completely differently. "
        "Examples: lesbian → le dollar bean, LGBT → leg booty, suicide → sewer slide, "
        "sex → secks, killed → kilt, negro → knee grow, nigga → knee guh, "
        "murder → murda, cocaine → co-cane, shooting → shootin, rape → grapes, "
        "assault → a-salt, marijuana → mary jane."
    ),
}

# Base temperature per technique — used to compute mixed-post temperature
TECHNIQUE_TEMPERATURE = {
    "unknown_spelling": 0.4,
    "known_harmless":   0.45,
    "abbreviation":     0.35,
    "pictorial":        0.0,   # always deterministic — strictly JSON-locked
    "paraphrase":       0.25,
    "phonetic":         0.3,
}

# ────────────────────────────────────────────────
# HINTS
# ────────────────────────────────────────────────
def load_hints(path: str) -> dict:
    if not os.path.exists(path):
        print(f"WARNING: hints file not found at {path}. Run build_deny_hints.py first.")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded hints for {len(data)} deny-list terms from {path}")
    return data


DENY_TERM_HINTS = load_hints(HINTS_FILE)


def get_hints_for_term(deny_term: str, technique: str) -> list:
    term_hints = DENY_TERM_HINTS.get(deny_term.lower(), {})
    return term_hints.get(technique, [])


# ────────────────────────────────────────────────
# MARKUP PROTECTION (@mentions, #hashtags, URLs)
# ────────────────────────────────────────────────
_PROTECT_PATTERN = re.compile(r'https?://\S+|@\w+|#\w+')


def protect_markup(text: str) -> tuple:
    tokens = {}
    counter = [0]

    def replacer(m):
        token = m.group()
        for ph, val in tokens.items():
            if val == token:
                return ph
        ph = f'__PROTECTED_{counter[0]}__'
        tokens[ph] = token
        counter[0] += 1
        return ph

    masked = _PROTECT_PATTERN.sub(replacer, text)
    return masked, tokens


def restore_markup(text: str, tokens: dict) -> str:
    for placeholder, original in tokens.items():
        text = text.replace(placeholder, original)
    return text


# ────────────────────────────────────────────────
# DENY LIST
# ────────────────────────────────────────────────
def load_deny_list(path: str) -> list:
    if not os.path.exists(path):
        print(f"WARNING: deny list not found at {path}. Using empty list.")
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    tokens = [line.strip().lower() for line in lines if line.strip()]
    unique_terms = sorted(set(tokens))
    print(f"Loaded {len(unique_terms)} deny-list terms.")
    return unique_terms


DENY_LIST_TERMS = load_deny_list(DENY_LIST_FILE)


# ────────────────────────────────────────────────
# TRANSFORMATION LOG
# ────────────────────────────────────────────────
def load_transformation_log(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
            for col in ['deny_term', 'technique', 'algospeak_output']:
                if col not in df.columns:
                    df[col] = ""
            print(f"Loaded {len(df)} rows from transformation log.")
            return df
        except Exception as e:
            print(f"Error loading transformation log: {e}. Starting fresh.")
    return pd.DataFrame(columns=['deny_term', 'technique', 'algospeak_output'])


def get_seen_outputs(log_df: pd.DataFrame, deny_term: str, technique: str) -> list:
    mask = (
        (log_df['deny_term'].str.lower() == deny_term.lower()) &
        (log_df['technique'] == technique)
    )
    seen = log_df.loc[mask, 'algospeak_output'].dropna().tolist()
    return list(dict.fromkeys(seen))


def append_to_transformation_log(path: str, entries: list):
    if not entries:
        return
    df_new = pd.DataFrame(entries)
    if os.path.exists(path):
        df_new.to_csv(path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df_new.to_csv(path, mode='w', header=True, index=False, encoding='utf-8-sig')


def get_untransformed_terms(text: str, terms: list) -> list:
    masked, _ = protect_markup(text)
    lower = masked.lower()
    return [t for t in terms if re.search(r'\b' + re.escape(t) + r'\b', lower)]


# ────────────────────────────────────────────────
# DENY TERM DETECTION
# ────────────────────────────────────────────────
def get_deny_terms_in_text(text: str) -> list:
    """
    Detect deny-list terms using word-boundary matching.
    Uses suffix-tolerant patterns so plural/inflected forms are also caught
    (e.g. 'dicks' matches 'dick', 'cunts' matches 'cunt').
    Returns canonical deny-list terms (not the inflected forms).
    """
    masked, _ = protect_markup(text)
    lower = masked.lower()
    found = [
        term for term in DENY_LIST_TERMS
        if term and re.search(
            r'\b' + re.escape(term) + r"(?:s|'s|es|d|ed|ing|er)?\b",
            lower
        )
    ]
    return sorted(set(found), key=len, reverse=True)


# ────────────────────────────────────────────────
# PER-TERM TECHNIQUE SAMPLING
# ────────────────────────────────────────────────
def sample_term_techniques(deny_terms: list) -> dict:
    """
    Independently sample a technique for each deny term.
    Pictorial is only assigned when the term has a known emoji in the hints file.
    If pictorial is sampled but no hint exists, re-roll without pictorial.
    Returns {deny_term: technique}.
    """
    result = {}
    non_pictorial = [t for t in ALGOSPEAK_TECHNIQUES if t != 'pictorial']
    for term in deny_terms:
        technique = random.choice(ALGOSPEAK_TECHNIQUES)
        if technique == 'pictorial' and not get_hints_for_term(term, 'pictorial'):
            technique = random.choice(non_pictorial)
        result[term] = technique
    return result


def get_mixed_temperature(term_techniques: dict) -> float:
    """
    Compute API temperature for a mixed-technique post.
    Averages each term's base temperature. Pictorial contributes 0.0 (fully locked).
    """
    temps = [TECHNIQUE_TEMPERATURE[t] for t in term_techniques.values()]
    return round(sum(temps) / len(temps), 2) if temps else 0.35


# ────────────────────────────────────────────────
# PROMPT BUILDERS
# ────────────────────────────────────────────────
def build_term_assignment_line(term: str, technique: str, log_df: pd.DataFrame) -> str:
    """
    Build a single assignment line for one term, including any hint or cycling constraint.
    Format: "term" → TECHNIQUE | <constraint>
    """
    hints = get_hints_for_term(term, technique)
    seen = get_seen_outputs(log_df, term, technique)
    constraint = ""

    if technique == 'pictorial' and hints:
        constraint = f"use EXACTLY {hints[0]} — never anything else"
    elif technique == 'abbreviation' and hints:
        options_str = ", ".join(f'"{h}"' for h in hints)
        constraint = f"use one of: {options_str}"
    elif technique == 'paraphrase' and hints:
        options_str = ", ".join(f'"{h}"' for h in hints[:4])
        constraint = f"prefer one of: {options_str}"
    elif len(seen) >= MAX_UNIQUE_SUBSTITUTIONS:
        options_str = ", ".join(f'"{s}"' for s in seen[:MAX_UNIQUE_SUBSTITUTIONS])
        constraint = f"CYCLING — pick one of: {options_str}"
    elif seen:
        seen_str = ", ".join(f'"{s}"' for s in seen)
        constraint = f"avoid already-used: {seen_str}"

    suffix = f" | {constraint}" if constraint else ""
    return f'  "{term}" → {technique.upper()}{suffix}'


def build_technique_reference(techniques_used: set) -> str:
    """Include technique descriptions only for techniques actually in use this post."""
    lines = []
    for technique in ALGOSPEAK_TECHNIQUES:
        if technique in techniques_used:
            lines.append(f"- {technique.upper()}: {TECHNIQUE_DESCRIPTIONS[technique]}")
    return "\n".join(lines)


def build_system_prompt(term_techniques: dict, log_df: pd.DataFrame) -> str:
    assignment_lines = [
        build_term_assignment_line(term, technique, log_df)
        for term, technique in term_techniques.items()
    ]
    assignment_block = "\n".join(assignment_lines)
    reference_block = build_technique_reference(set(term_techniques.values()))

    return f"""You are an expert in algospeak.

CRITICAL RULES:
1. Transform ONLY the deny-list terms listed below. Do NOT touch any other word.
2. Do NOT modify __PROTECTED_N__ tokens — these are @mentions, #hashtags, or URLs that must stay exactly as written.
3. Leave grammar, sentence structure, tone, and all non-deny words 100% identical.
4. Each term has its own assigned technique — apply them independently.
5. If a term appears multiple times, transform every occurrence the same way.
6. Also transform any plural, possessive, or inflected form of a listed term found in the text (e.g. if transforming "dick", also transform "dicks"; if transforming "kill", also transform "kills").
7. Keep the output length and tone extremely close to the original.

TERM → TECHNIQUE ASSIGNMENTS:
{assignment_block}

TECHNIQUE REFERENCE (only for techniques assigned above):
{reference_block}

OUTPUT FORMAT — return a JSON object with exactly these two fields:
{{
  "transformed": "<the full transformed text>",
  "substitutions": {{"<original_term>": "<what you replaced it with>", ...}}
}}
Return ONLY the JSON. No markdown fences, no explanation."""


def build_user_prompt(text: str, term_techniques: dict) -> str:
    terms_numbered = "\n".join(
        f"  {i+1}. \"{t}\"" for i, t in enumerate(term_techniques.keys())
    )
    n = len(term_techniques)
    return (
        f"Transform ALL {n} term(s) below using their individually assigned techniques:\n"
        f"{terms_numbered}\n\n"
        f"Text:\n{text}"
    )


def calc_max_tokens(text: str) -> int:
    word_count = len(text.split())
    return min(800, max(150, int(word_count * 4)))


# ────────────────────────────────────────────────
# API CALL WITH RETRY
# ────────────────────────────────────────────────
def to_algospeak(text: str, term_techniques: dict,
                 log_df: pd.DataFrame, model: str) -> tuple:
    """
    Call GPT to produce an algospeak version of `text`.
    term_techniques: {deny_term: technique} — each term's assigned technique.
    Returns (transformed_text, substitutions_dict).
    """
    system_prompt = build_system_prompt(term_techniques, log_df)
    user_prompt = build_user_prompt(text, term_techniques)
    temperature = get_mixed_temperature(term_techniques)
    max_tokens = calc_max_tokens(text)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            raw = response.choices[0].message.content.strip()

            try:
                data = json.loads(raw)
                transformed = data.get("transformed", "").strip()
                substitutions = data.get("substitutions", {})
                if not transformed:
                    raise ValueError("Empty transformed field in JSON response")
                return transformed, substitutions
            except (json.JSONDecodeError, ValueError):
                return raw, {}

        except openai.RateLimitError:
            wait = (2 ** attempt) * 5
            print(f"  Rate limit (attempt {attempt + 1}/{MAX_RETRIES}), waiting {wait}s...")
            time.sleep(wait)
            last_error = "rate_limit"
        except openai.APIError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
                last_error = str(e)
            else:
                raise

    raise RuntimeError(f"Max retries exceeded. Last error: {last_error}")


# ────────────────────────────────────────────────
# CSV I/O
# ────────────────────────────────────────────────
def append_to_output_csv(new_data: dict, output_file: str):
    df_new = pd.DataFrame([new_data])
    if os.path.exists(output_file):
        df_new.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df_new.to_csv(output_file, mode='w', header=True, index=False, encoding='utf-8-sig')


# ────────────────────────────────────────────────
# MAIN PIPELINE
# ────────────────────────────────────────────────
def process_csv(limit_rows=None, model=DEFAULT_MODEL, overwrite=False, input_file=None):
    active_input = input_file if input_file else INPUT_FILE
    print(f"\n{'='*60}")
    print(f"Model:   {model}")
    print(f"Input:   {active_input}")
    print(f"Output:  {OUTPUT_FILE}")
    print(f"Log:     {TRANSFORMATION_LOG_FILE}")
    print(f"{'='*60}\n")

    if not os.path.exists(active_input):
        raise FileNotFoundError(f"Input CSV not found at {active_input}")

    df_input = pd.read_csv(active_input, encoding='utf-8-sig')
    print("Columns found:", df_input.columns.tolist())
    print(df_input.head(3).to_string(index=False))
    print("-" * 60)

    text_column = df_input.columns[0]
    print(f"-> Using column '{text_column}' as source text\n")

    if overwrite:
        for f in [OUTPUT_FILE, TRANSFORMATION_LOG_FILE]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Overwrite: deleted {f}")

    log_df = load_transformation_log(TRANSFORMATION_LOG_FILE)

    processed_originals = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE, encoding='utf-8-sig')
            if 'original_text' in df_existing.columns:
                processed_originals = set(
                    df_existing['original_text'].astype(str).str.strip()
                )
            print(f"Resuming: {len(processed_originals)} posts already processed.\n")
        except Exception as e:
            print(f"Could not load existing output ({e}). Starting fresh.\n")

    if limit_rows is not None:
        print(f"TEST MODE -> processing first {limit_rows} rows only\n")
        df_input = df_input.head(limit_rows)
    else:
        print("FULL MODE -> processing entire file\n")

    total = len(df_input)
    generated = 0
    skipped_no_deny = 0
    skipped_existing = 0
    errors = 0

    for idx, row in df_input.iterrows():
        original = str(row[text_column]).strip()

        if not original or original.lower() == 'nan':
            continue

        if original in processed_originals and not overwrite:
            skipped_existing += 1
            continue

        deny_terms_detected = get_deny_terms_in_text(original)
        if not deny_terms_detected:
            skipped_no_deny += 1
            print(f"[{idx+1}/{total}] SKIP (no deny terms): {original[:80]}")
            continue

        # Independently assign a technique to each term (capped at MAX_TERMS_PER_POST)
        terms_to_use = deny_terms_detected[:MAX_TERMS_PER_POST]
        term_techniques = sample_term_techniques(terms_to_use)
        temp = get_mixed_temperature(term_techniques)

        techniques_summary = ", ".join(
            f"{t}={tech.upper()}" for t, tech in term_techniques.items()
        )
        print(
            f"[{idx+1}/{total}] temp={temp} | "
            f"{len(term_techniques)}/{len(deny_terms_detected)} terms | "
            f"{techniques_summary}"
        )
        print(f"  Original:  {original[:120]}")

        try:
            masked_text, markup_tokens = protect_markup(original)

            algospeak_masked, substitutions = to_algospeak(
                text=masked_text,
                term_techniques=term_techniques,
                log_df=log_df,
                model=model,
            )

            # Retry any terms GPT silently skipped, preserving each term's technique
            missed = get_untransformed_terms(
                restore_markup(algospeak_masked, markup_tokens),
                list(term_techniques.keys()),
            )
            if missed:
                print(f"  GPT missed {missed} — retrying on partial result...")
                missed_techniques = {t: term_techniques[t] for t in missed}
                algospeak_masked, substitutions2 = to_algospeak(
                    text=algospeak_masked,
                    term_techniques=missed_techniques,
                    log_df=log_df,
                    model=model,
                )
                substitutions.update(substitutions2)

            algospeak_version = restore_markup(algospeak_masked, markup_tokens)

            if algospeak_version.strip() == original.strip():
                print(f"  SKIP (GPT returned unchanged text)")
                errors += 1
                continue

            print(f"  Algospeak: {algospeak_version[:120]}")

            # Store technique as pipe-separated list of unique techniques used
            techniques_used_str = "|".join(
                sorted(set(term_techniques.values()))
            )

            append_to_output_csv(
                {
                    'original_text': original,
                    'algospeak_text': algospeak_version,
                    'techniques': techniques_used_str,
                    'deny_terms_detected': "|".join(deny_terms_detected),
                    'deny_terms_transformed': "|".join(term_techniques.keys()),
                },
                OUTPUT_FILE,
            )

            # Log individual term → substitution pairs per technique
            new_log_entries = [
                {
                    'deny_term': term,
                    'technique': term_techniques[term],
                    'algospeak_output': sub,
                }
                for term in term_techniques
                if (sub := substitutions.get(term, ""))
                and sub not in get_seen_outputs(log_df, term, term_techniques[term])
            ]
            if new_log_entries:
                append_to_transformation_log(TRANSFORMATION_LOG_FILE, new_log_entries)
                log_df = pd.concat(
                    [log_df, pd.DataFrame(new_log_entries)],
                    ignore_index=True,
                )

            generated += 1
            time.sleep(0.4)

        except Exception as e:
            errors += 1
            print(f"  ERROR on row {idx+1}: {e}")

    print(f"\n{'='*60}")
    print(f"  Generated:            {generated}")
    print(f"  Skipped (no deny):    {skipped_no_deny}")
    print(f"  Skipped (existing):   {skipped_existing}")
    print(f"  Errors:               {errors}")
    print(f"  Output CSV:           {OUTPUT_FILE}")
    print(f"  Transformation log:   {TRANSFORMATION_LOG_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mixed-technique algospeak variants using dictionary-anchored hints."
    )
    parser.add_argument(
        '--test', nargs='?', const=100, type=int,
        help="Test mode: process only the first N rows (default: 100 if flag given with no value).",
    )
    parser.add_argument(
        '--model', type=str, default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help="Delete and regenerate output CSV and transformation log from scratch.",
    )
    parser.add_argument(
        '--input', type=str, default=None,
        help="Path to input CSV (default: data/splits/algospeak_sources.csv).",
    )
    args = parser.parse_args()
    process_csv(limit_rows=args.test, model=args.model, overwrite=args.overwrite,
                input_file=args.input)
