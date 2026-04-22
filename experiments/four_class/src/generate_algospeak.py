"""
experiments/four_class/src/generate_algospeak.py

Synthetic algospeak generator for the four-class experiment.

Reads the manually reclassified CSV from experiments/four_class/data/,
generates algospeak variants for class 1 and class 2 posts, and writes
results to experiments/four_class/data/synthetic_algospeak.csv.

Key design decisions vs the original synthetic_data.py:
  - Preserves harmful intent: explicit system-prompt rule prevents GPT from
    sanitizing slurs or replacing hate speech with inclusive language.
  - Slur-aware technique routing: paraphrase and known_harmless are excluded
    for slurs unless a confirmed community hint exists in deny_term_hints.json.
    Slurs default to unknown_spelling / abbreviation / phonetic.
  - pair_id propagation: each generated row inherits the source post's pair_id,
    or gets a new one assigned (gen_<hex>) if the source had none.
  - Distribution-aware: prints class counts at startup so you can see exactly
    how many more rows are needed before running.

Expected input columns:
    text       — post text
    label      — integer class label (0, 1, 2, 3)
    pair_id    — (optional) existing pair id; NaN / empty if unpaired

Output CSV columns:
    original_text, algospeak_text, label (=3), source_label,
    pair_id, techniques, deny_terms_detected, deny_terms_transformed

Usage:
    uv run python experiments/four_class/src/generate_algospeak.py \\
        --input experiments/four_class/data/<file>.csv

    uv run python experiments/four_class/src/generate_algospeak.py \\
        --input experiments/four_class/data/<file>.csv --test 20

    uv run python experiments/four_class/src/generate_algospeak.py \\
        --input experiments/four_class/data/<file>.csv --overwrite
"""

import sys
import io
import os
import json
import time
import random
import argparse
import uuid
from pathlib import Path

import pandas as pd
import openai
from dotenv import load_dotenv

# Windows console can't encode most emoji — reconfigure stdout before any prints.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

load_dotenv()

# Shared detection utilities — same inflection-aware logic as analyze_candidates.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DENY_LIST_TERMS, detect_deny_terms, get_untransformed, protect_markup, restore_markup

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
EXP_DIR     = SCRIPT_DIR.parent                     # experiments/four_class
BASE_DIR    = EXP_DIR.parents[1]                    # project root
DATA_DIR    = EXP_DIR / "data"
HINTS_FILE  = DATA_DIR / "algospeak_hints.json"     # four_class experiment hints

DEFAULT_OUTPUT = DATA_DIR / "synthetic_algospeak.csv"
DEFAULT_LOG    = DATA_DIR / "transformation_log.csv"

ALGOSPEAK_LABEL  = 3   # class label assigned to all generated rows
MAX_TERMS_PER_POST  = 6
MAX_RETRIES         = 3
MAX_UNIQUE_SUBS     = 6
DEFAULT_MODEL       = "gpt-4-turbo"

# ── OpenAI ─────────────────────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file.")
print(f"API key loaded: {api_key[:7]}...")
client = openai.OpenAI(api_key=api_key)

# ── Slur set ───────────────────────────────────────────────────────────────────
# These are identity-targeting slurs from the class 1 deny list.
# For these terms, paraphrase and known_harmless are excluded unless a
# confirmed community hint exists — GPT will otherwise sanitize them.
SLUR_TERMS = {
    "arsehole", "bastard", "beaner", "bellend", "bitch", "buddhahead",
    "camel-jockey", "cheese-monkey", "chink", "coon", "cunt", "curry-muncher",
    "darkie", "dune-coon", "faggot", "fuck", "fuckhead", "fucktard",
    "fuckwad", "fuckwit", "gook", "jigaboo", "kike", "motherfucker",
    "nigger", "paki", "petrol-sniffer", "pikey", "retard", "retarded",
    "sand-nigger", "shit", "shithead", "slanteye", "slut", "spearchucker",
    "spic", "swamp-guinea", "timber-nigger", "towelhead", "tranny", "twat",
    "wetback", "white power", "whore", "zipperhead", "cracker", "dyke",
    "ass", "asshole", "dick", "fucking",
}

# ── Techniques ─────────────────────────────────────────────────────────────────
ALL_TECHNIQUES = [
    "unknown_spelling", "known_harmless", "abbreviation",
    "pictorial", "paraphrase", "repurposing", "phonetic",
]
# Slurs stay restricted to these unless hints unlock more
SLUR_SAFE_TECHNIQUES = ["unknown_spelling", "abbreviation", "phonetic"]

TECHNIQUE_DESCRIPTIONS = {
    "unknown_spelling": (
        "Change spelling to an obscure or symbolic form using numbers, symbols, or unusual "
        "character substitutions. The result should look garbled/coded but still be "
        "recognizable to humans. Vary your approach.\n"
        "Examples: kill → k!ll, sex → $3x, suicide → su1c1d3, nigger → n1gg3r, "
        "faggot → f@gg0t, tranny → tr@nny, bitch → b1tch, retard → r3t4rd, "
        "rape → r@pe, cocaine → c0k3, shit → $h!t, fuck → f*ck."
    ),
    "known_harmless": (
        "Replace the sensitive word with a REAL, innocent-sounding word that looks or "
        "sounds similar. Never invent a word.\n"
        "Examples: porn → corn, rape → grape, kill → keel, "
        "cocaine → co-cane, murder → birder."
    ),
    "abbreviation": (
        "Shorten the term to an abbreviation, acronym, or 'the ___ word/slur' construction.\n"
        "Examples: sexual assault → SA, suicide → 'the s-word', nigger → 'the n-word', "
        "faggot → 'the f-slur', tranny → 't-slur', retard → 'the r-word', "
        "rape → 'the r-word', murder → 'the m-word'."
    ),
    "pictorial": (
        "Replace the sensitive word with EXACTLY ONE emoji that visually or conceptually "
        "represents it. The emoji must come from the DICTIONARY LOCKS in the term "
        "assignments — do not use any other emoji."
    ),
    "paraphrase": (
        "Substitute the sensitive word with a community-used coded paraphrase.\n"
        "ONLY valid forms: kill/murder/suicide → unalive, suicide/die → exit the chat.\n"
        "Do NOT use this technique for any other term."
    ),
    "repurposing": (
        "Replace the sensitive word with an existing, unrelated word or phrase that the "
        "community has adopted as a code — no spelling changes, no phonetic tricks, "
        "just an existing word used with a new meaning.\n"
        "Examples: kill → end, killing → ending, porn → fun movie, "
        "sex → spicy time, cocaine → snow, marijuana → honey, "
        "vibrator → spicy eggplant, murdered → redrumed."
    ),
    "phonetic": (
        "Replace the sensitive word with something that sounds phonetically similar "
        "when spoken aloud.\n"
        "Examples: suicide → sewer slide, sex → secks, killed → kilt, "
        "negro → knee grow, nigga → knee guh, murder → murda, rape → grapes, "
        "tranny → granny, shit → shish."
    ),
}

TECHNIQUE_TEMPERATURE = {
    "unknown_spelling": 0.4,
    "known_harmless":   0.45,
    "abbreviation":     0.35,
    "pictorial":        0.0,
    "paraphrase":       0.25,
    "repurposing":      0.4,
    "phonetic":         0.3,
}

# ── Hints ──────────────────────────────────────────────────────────────────────
def load_hints(path: Path) -> dict:
    if not path.exists():
        print(f"WARNING: hints file not found at {path}. Run build_deny_hints.py first.")
        return {}
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded hints for {len(data)} deny-list terms.")
    return data

DENY_TERM_HINTS = load_hints(HINTS_FILE)

def get_hints(term: str, technique: str) -> list:
    return DENY_TERM_HINTS.get(term.lower(), {}).get(technique, [])

print(f"Deny list: {len(DENY_LIST_TERMS)} terms total.")

# ── Technique sampling ─────────────────────────────────────────────────────────
def sample_techniques(deny_terms: list) -> dict:
    """
    Independently sample a technique per deny term.

    Slur terms: restricted to always-safe set (unknown_spelling, abbreviation, phonetic).
      - pictorial unlocked if a hint emoji exists.
      - paraphrase unlocked only if a confirmed community hint exists.
      - known_harmless stays excluded for slurs (too likely to sanitize without guidance).

    Non-slur terms: full technique set, with pictorial gated on having a hint emoji.
    """
    result = {}
    for term in deny_terms:
        is_slur = term.lower() in SLUR_TERMS

        if is_slur:
            eligible = list(SLUR_SAFE_TECHNIQUES)
            if get_hints(term, 'pictorial'):
                eligible.append('pictorial')
            if get_hints(term, 'paraphrase'):
                eligible.append('paraphrase')
            technique = random.choice(eligible)
        else:
            technique = random.choice(ALL_TECHNIQUES)
            if technique == 'pictorial' and not get_hints(term, 'pictorial'):
                technique = random.choice([t for t in ALL_TECHNIQUES if t != 'pictorial'])

        result[term] = technique
    return result

def mixed_temperature(term_techniques: dict) -> float:
    temps = [TECHNIQUE_TEMPERATURE[t] for t in term_techniques.values()]
    return round(sum(temps) / len(temps), 2) if temps else 0.35

# ── Transformation log ─────────────────────────────────────────────────────────
def load_log(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
            print(f"Loaded {len(df)} rows from transformation log.")
            return df
        except Exception as e:
            print(f"Could not load log ({e}) — starting fresh.")
    return pd.DataFrame(columns=['deny_term', 'technique', 'algospeak_output'])

def get_seen(log_df: pd.DataFrame, term: str, technique: str) -> list:
    mask = (log_df['deny_term'].str.lower() == term.lower()) & (log_df['technique'] == technique)
    return log_df.loc[mask, 'algospeak_output'].dropna().tolist()

def append_log(path: Path, entries: list):
    if not entries:
        return
    df = pd.DataFrame(entries)
    if path.exists():
        df.to_csv(path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(path, mode='w', header=True, index=False, encoding='utf-8-sig')

# ── Prompt building ────────────────────────────────────────────────────────────
def build_assignment_line(term: str, technique: str, log_df: pd.DataFrame) -> str:
    hints = get_hints(term, technique)
    seen  = get_seen(log_df, term, technique)
    constraint = ""

    if technique == 'pictorial' and hints:
        constraint = f"use EXACTLY {hints[0]} — no other emoji"
    elif technique == 'abbreviation' and hints:
        constraint = f"use one of: {', '.join(repr(h) for h in hints)}"
    elif technique == 'paraphrase' and hints:
        constraint = f"prefer one of: {', '.join(repr(h) for h in hints[:4])}"
    elif len(seen) >= MAX_UNIQUE_SUBS:
        constraint = f"CYCLING — pick one of: {', '.join(repr(s) for s in seen[:MAX_UNIQUE_SUBS])}"
    elif seen:
        constraint = f"avoid already-used: {', '.join(repr(s) for s in seen)}"

    return f'  "{term}" → {technique.upper()}' + (f" | {constraint}" if constraint else "")

def build_system_prompt(term_techniques: dict, log_df: pd.DataFrame) -> str:
    assignments = "\n".join(
        build_assignment_line(t, tech, log_df) for t, tech in term_techniques.items()
    )
    techniques_used = set(term_techniques.values())
    reference = "\n".join(
        f"- {t.upper()}: {TECHNIQUE_DESCRIPTIONS[t]}"
        for t in ALL_TECHNIQUES if t in techniques_used
    )

    return f"""You are an expert in algospeak — the coded language social media users developed to evade automated content moderation.

CRITICAL RULES:
1. Transform ONLY the deny-list terms listed below. Do NOT touch any other word.
2. Do NOT modify __PN__ placeholder tokens — these represent @mentions, #hashtags, or URLs.
3. Leave grammar, sentence structure, tone, and all non-deny words 100% identical.
4. Each term has its own assigned technique — apply them independently.
5. If a term appears multiple times, transform every occurrence the same way.
6. Also transform any plural, possessive, or inflected form of a listed term (e.g. if "kill" is listed, also transform "kills", "killing", "killed").
7. NEVER sanitize, correct, or euphemize in a way that removes the harmful meaning.
   This is content moderation research — the output must still clearly communicate
   the original harmful meaning in a coded form that a human reader would understand.
   For slurs specifically: produce a coded evasion form ("tr@nny", "t-slur", "n-word", "f@gg0t")
   and NEVER substitute inclusive or clinical language ("transgender", "gay person", etc.).

TERM → TECHNIQUE ASSIGNMENTS:
{assignments}

TECHNIQUE REFERENCE:
{reference}

OUTPUT FORMAT — return a JSON object with exactly these two keys:
{{
  "transformed": "<full transformed text>",
  "substitutions": {{"<original_term>": "<what you replaced it with>", ...}}
}}
Return ONLY the JSON. No markdown fences, no explanation."""

def build_user_prompt(text: str, term_techniques: dict) -> str:
    numbered = "\n".join(f'  {i+1}. "{t}"' for i, t in enumerate(term_techniques))
    return (
        f"Transform ALL {len(term_techniques)} term(s) below using their assigned techniques:\n"
        f"{numbered}\n\nText:\n{text}"
    )

# ── API call ───────────────────────────────────────────────────────────────────
def call_api(text: str, term_techniques: dict, log_df: pd.DataFrame, model: str) -> tuple:
    sys_prompt  = build_system_prompt(term_techniques, log_df)
    user_prompt = build_user_prompt(text, term_techniques)
    temperature = mixed_temperature(term_techniques)
    max_tokens  = min(800, max(150, len(text.split()) * 4))

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            raw = resp.choices[0].message.content.strip()
            try:
                data = json.loads(raw)
                transformed = data.get("transformed", "").strip()
                subs = data.get("substitutions", {})
                if not transformed:
                    raise ValueError("empty transformed field")
                return transformed, subs
            except (json.JSONDecodeError, ValueError):
                return raw, {}

        except openai.RateLimitError:
            wait = (2 ** attempt) * 5
            print(f"  Rate limit — waiting {wait}s...")
            time.sleep(wait)
            last_error = "rate_limit"
        except openai.APIError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
                last_error = str(e)
            else:
                raise

    raise RuntimeError(f"Max retries exceeded. Last error: {last_error}")

# ── Output ─────────────────────────────────────────────────────────────────────
def append_output(row: dict, path: Path):
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(path, mode='w', header=True, index=False, encoding='utf-8-sig')

def make_pair_id() -> str:
    return f"gen_{uuid.uuid4().hex[:8]}"

# ── Main ───────────────────────────────────────────────────────────────────────
def run(input_file: str, output_file: Path, log_file: Path,
        limit: int, model: str, overwrite: bool):

    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input not found: {input_file}")

    df_in = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"\nInput: {len(df_in)} rows  |  columns: {df_in.columns.tolist()}")

    if 'text' not in df_in.columns:
        raise ValueError(f"Expected a 'text' column. Found: {df_in.columns.tolist()}")
    # Accept either 'label' or 'classification' as the class column
    # Accept either 'label' or 'classification' as the class column
    if 'label' not in df_in.columns and 'classification' in df_in.columns:
        df_in = df_in.rename(columns={'classification': 'label'})
    if 'label' not in df_in.columns:
        raise ValueError(f"Expected a 'label' or 'classification' column. Found: {df_in.columns.tolist()}")
    # Accept either 'pair_id' or 'pair_ID'
    if 'pair_id' not in df_in.columns and 'pair_ID' in df_in.columns:
        df_in = df_in.rename(columns={'pair_ID': 'pair_id'})

    # ── Class distribution ─────────────────────────────────────────────────────
    counts = df_in['label'].value_counts().sort_index()
    print(f"\nClass distribution in input:")
    for cls, cnt in counts.items():
        print(f"  class {cls}: {cnt} rows")

    class2_count = int(counts.get(2, 0))
    if class2_count == 0:
        raise ValueError("No class 2 rows in input — cannot determine target count.")

    print(f"\nTarget count per class (class 2 = limiting class): {class2_count}")

    # How many algospeak rows already exist per source class?
    existing_by_source: dict = {1: 0, 2: 0}
    if output_file.exists() and not overwrite:
        try:
            df_ex = pd.read_csv(output_file, encoding='utf-8-sig')
            if 'source_label' in df_ex.columns:
                for cls in [1, 2]:
                    existing_by_source[cls] = int((df_ex['source_label'] == cls).sum())
        except Exception:
            pass

    for cls in [1, 2]:
        need = max(0, class2_count - existing_by_source[cls])
        print(f"  class {cls} algospeak: {existing_by_source[cls]} existing, need {need} more to reach target")

    # ── Overwrite ──────────────────────────────────────────────────────────────
    if overwrite:
        for f in [output_file, log_file]:
            if f.exists():
                f.unlink()
                print(f"Overwrite: deleted {f}")

    log_df = load_log(log_file)

    already_done: set = set()
    if output_file.exists():
        try:
            df_ex = pd.read_csv(output_file, encoding='utf-8-sig')
            if 'original_text' in df_ex.columns:
                already_done = set(df_ex['original_text'].astype(str).str.strip())
            print(f"Resuming: {len(already_done)} posts already processed.")
        except Exception as e:
            print(f"Could not read existing output ({e}) — starting fresh.")

    # ── Candidate selection ────────────────────────────────────────────────────
    candidates = df_in[df_in['label'].isin([1, 2])].copy()
    if limit is not None:
        print(f"\nTEST MODE: processing first {limit} candidates only.")
        candidates = candidates.head(limit)

    print(f"\n{'='*60}")
    print(f"Model:   {model}")
    print(f"Input:   {input_file}")
    print(f"Output:  {output_file}")
    print(f"Log:     {log_file}")
    print(f"{'='*60}\n")

    generated, skipped_no_deny, skipped_existing, errors = 0, 0, 0, 0
    total = len(candidates)

    for i, (idx, row) in enumerate(candidates.iterrows()):
        original = str(row['text']).strip()
        if not original or original.lower() == 'nan':
            continue

        if original in already_done and not overwrite:
            skipped_existing += 1
            continue

        source_label = int(row['label'])

        deny_terms = detect_deny_terms(original)
        if not deny_terms:
            skipped_no_deny += 1
            print(f"[{i+1}/{total}] SKIP (no deny terms): {original[:80]}")
            continue

        terms_to_use   = deny_terms[:MAX_TERMS_PER_POST]
        term_techniques = sample_techniques(terms_to_use)
        temp            = mixed_temperature(term_techniques)
        tech_summary    = ", ".join(f"{t}={tech.upper()}" for t, tech in term_techniques.items())

        print(f"[{i+1}/{total}] cls={source_label} temp={temp} | {tech_summary}")
        print(f"  Orig: {original[:110]}")

        # pair_id: inherit from source row, or mint a new one
        raw_pid = row.get('pair_id', None) if 'pair_id' in row.index else None
        if pd.isna(raw_pid) or str(raw_pid).strip() in ('', 'nan'):
            pair_id = make_pair_id()
        else:
            pair_id = str(raw_pid).strip()

        try:
            masked, markup = protect_markup(original)

            algo_masked, subs = call_api(masked, term_techniques, log_df, model)

            # Retry any terms GPT silently skipped
            missed = get_untransformed(restore_markup(algo_masked, markup), list(term_techniques.keys()))
            if missed:
                print(f"  GPT missed {missed} — retrying...")
                missed_tech = {t: term_techniques[t] for t in missed}
                algo_masked, subs2 = call_api(algo_masked, missed_tech, log_df, model)
                subs.update(subs2)

            algospeak_text = restore_markup(algo_masked, markup)

            if algospeak_text.strip() == original.strip():
                print("  SKIP (GPT returned unchanged text)")
                errors += 1
                continue

            print(f"  Algo: {algospeak_text[:110]}")

            techniques_str = "|".join(sorted(set(term_techniques.values())))

            append_output({
                'original_text':          original,
                'algospeak_text':         algospeak_text,
                'label':                  ALGOSPEAK_LABEL,
                'source_label':           source_label,
                'pair_id':                pair_id,
                'techniques':             techniques_str,
                'deny_terms_detected':    "|".join(deny_terms),
                'deny_terms_transformed': "|".join(term_techniques.keys()),
            }, output_file)

            # Update transformation log with novel substitutions only
            new_entries = [
                {'deny_term': t, 'technique': term_techniques[t], 'algospeak_output': subs.get(t, "")}
                for t in term_techniques
                if subs.get(t, "") and subs.get(t, "") not in get_seen(log_df, t, term_techniques[t])
            ]
            if new_entries:
                append_log(log_file, new_entries)
                log_df = pd.concat([log_df, pd.DataFrame(new_entries)], ignore_index=True)

            already_done.add(original)
            generated += 1
            time.sleep(0.4)

        except Exception as e:
            errors += 1
            print(f"  ERROR row {i+1}: {e}")

    print(f"\n{'='*60}")
    print(f"  Generated:           {generated}")
    print(f"  Skipped (no deny):   {skipped_no_deny}")
    print(f"  Skipped (existing):  {skipped_existing}")
    print(f"  Errors:              {errors}")
    print(f"  Output:              {output_file}")
    print(f"  Log:                 {log_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic algospeak for the four-class experiment."
    )
    parser.add_argument(
        '--input', required=True,
        help="Path to input CSV (manually reclassified data).",
    )
    parser.add_argument(
        '--output', default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        '--log', default=str(DEFAULT_LOG),
        help=f"Transformation log CSV path (default: {DEFAULT_LOG}).",
    )
    parser.add_argument(
        '--test', nargs='?', const=20, type=int,
        help="Test mode: process only the first N class-1/2 rows (default 20 if flag given with no value).",
    )
    parser.add_argument(
        '--model', default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help="Delete existing output and log, regenerate from scratch.",
    )
    args = parser.parse_args()

    run(
        input_file=args.input,
        output_file=Path(args.output),
        log_file=Path(args.log),
        limit=args.test,
        model=args.model,
        overwrite=args.overwrite,
    )
