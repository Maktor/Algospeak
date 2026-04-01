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
OUTPUT_FILE = os.path.join(EXPERIMENT_DIR, "synthetic_algospeak.csv")
DENY_LIST_FILE = os.path.join(EXPERIMENT_DIR, "deny_list.txt")
TRANSFORMATION_LOG_FILE = os.path.join(EXPERIMENT_DIR, "transformation_log.csv")

os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# ────────────────────────────────────────────────
# OPENAI CLIENT
# ────────────────────────────────────────────────
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env or environment variables!")
print("API key loaded (starts with):", api_key[:7] + "...")

client = openai.OpenAI(api_key=api_key)

DEFAULT_MODEL = "gpt-4o"
MAX_TERMS_TO_TRANSFORM = 6   # At most this many deny terms per post (naturalness)
MAX_RETRIES = 3              # API retry attempts on transient errors
MAX_UNIQUE_SUBSTITUTIONS = 6 # After this many unique substitutions per term+technique,
                             # cycle through existing ones instead of inventing new ones

# ────────────────────────────────────────────────
# ALGOSPEAK TECHNIQUES + TEMPERATURES
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
        "represents it. Use only a single emoji per term — never stack multiple emojis. "
        "Examples: gun → 🔫, death → 💀, drugs → 💊, sex → 🍆, knife → 🔪, "
        "weed → 🌿, cocaine → ❄️, suicide → 🪢, shooting → 💥, hate → 🤬, "
        "blood → 🩸, bomb → 💣."
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

TECHNIQUE_TEMPERATURE = {
    "unknown_spelling": 0.4,
    "known_harmless":   0.45,
    "abbreviation":     0.35,
    "pictorial":        0.4,
    "paraphrase":       0.25,
    "phonetic":         0.3,
}

# Per-technique cap on how many deny terms to transform per post.
# Techniques that add length/words (paraphrase) or are visually noisy (pictorial)
# use lower limits so the post doesn't look unnatural.
TECHNIQUE_MAX_TERMS = {
    "unknown_spelling": 6,
    "known_harmless":   6,
    "abbreviation":     6,
    "phonetic":         5,
    "pictorial":        4,
    "paraphrase":       3,
}

# ────────────────────────────────────────────────
# MARKUP PROTECTION (@mentions, #hashtags, URLs)
# ────────────────────────────────────────────────
_PROTECT_PATTERN = re.compile(r'https?://\S+|@\w+|#\w+')


def protect_markup(text: str) -> tuple:
    """
    Replace @mentions, #hashtags, and URLs with stable placeholders before
    sending to GPT. Same token always maps to the same placeholder, so all
    occurrences are restored correctly in one pass.

    Returns (masked_text, {placeholder: original_token}).
    """
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
    """Restore __PROTECTED_N__ placeholders to their original values."""
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
    # Each line is one term/phrase — preserve spaces within phrases
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
    # Deduplicate while preserving order (dict.fromkeys trick)
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
    """
    Return any terms from the list that still appear unchanged in `text`.
    Used after a GPT call to detect terms GPT silently skipped.
    Strips markup placeholders before checking so they don't interfere.
    """
    masked, _ = protect_markup(text)
    lower = masked.lower()
    return [t for t in terms if re.search(r'\b' + re.escape(t) + r'\b', lower)]


# ────────────────────────────────────────────────
# DENY TERM DETECTION
# ────────────────────────────────────────────────
def get_deny_terms_in_text(text: str) -> list:
    """
    Detect deny-list terms using word-boundary matching.
    Strips @mentions, #hashtags, and URLs first so terms inside them
    (e.g. 'kill' inside '@killzone') are not falsely detected.
    """
    masked, _ = protect_markup(text)
    lower = masked.lower()
    found = [
        term for term in DENY_LIST_TERMS
        if term and re.search(r'\b' + re.escape(term) + r'\b', lower)
    ]
    return sorted(set(found), key=len, reverse=True)


# ────────────────────────────────────────────────
# PROMPT BUILDERS
# ────────────────────────────────────────────────
def build_system_prompt(technique: str, deny_terms_to_transform: list,
                        log_df: pd.DataFrame) -> str:
    technique_desc = TECHNIQUE_DESCRIPTIONS[technique]

    variety_lines = []   # terms still building up unique substitutions
    cycling_lines = []   # terms that have hit the cap — reuse from existing set

    for term in deny_terms_to_transform:
        seen = get_seen_outputs(log_df, term, technique)
        if 0 < len(seen) < MAX_UNIQUE_SUBSTITUTIONS:
            seen_str = ", ".join(f'"{s}"' for s in seen)
            variety_lines.append(
                f'  - "{term}": avoid these already-used substitutions: {seen_str}'
            )
        elif len(seen) >= MAX_UNIQUE_SUBSTITUTIONS:
            options_str = ", ".join(f'"{s}"' for s in seen[:MAX_UNIQUE_SUBSTITUTIONS])
            cycling_lines.append(
                f'  - "{term}": pick exactly one of these (do NOT invent anything new): {options_str}'
            )

    constraints_block = ""
    if variety_lines:
        constraints_block += (
            "\n\nNEGATIVE CONSTRAINTS (invent something new — avoid all of these):\n"
            + "\n".join(variety_lines)
        )
    if cycling_lines:
        constraints_block += (
            "\n\nCYCLING MODE (enough variety collected — reuse one of the listed options):\n"
            + "\n".join(cycling_lines)
        )

    return f"""You are an expert in algospeak.

CRITICAL RULES:
1. Transform ONLY the listed deny-list terms. Do NOT touch any other word.
2. Do NOT modify __PROTECTED_N__ tokens — these are @mentions, #hashtags, or URLs that must stay exactly as written.
3. Leave grammar, sentence structure, tone, and all non-deny words 100% identical.

VARIETY: Invent a fresh substitution for each term. Never reuse anything from the negative constraints below.

Assigned technique: {technique.upper()}
Description: {technique_desc}

Rules:
1. Apply ONLY the assigned technique. Do not mix techniques.
2. Transform ALL listed deny-list terms. Do not skip any.
3. If a term appears multiple times, transform every occurrence the same way.
4. Keep the output length and tone extremely close to the original.{constraints_block}

OUTPUT FORMAT — return a JSON object with exactly these two fields:
{{
  "transformed": "<the full transformed text>",
  "substitutions": {{"<original_term>": "<what you replaced it with>", ...}}
}}
Return ONLY the JSON. No markdown fences, no explanation."""


def build_user_prompt(text: str, technique: str, deny_terms_to_transform: list) -> str:
    terms_numbered = "\n".join(
        f"  {i+1}. \"{t}\"" for i, t in enumerate(deny_terms_to_transform)
    )
    return (
        f"Technique: {technique.upper()}\n\n"
        f"Transform ALL {len(deny_terms_to_transform)} term(s) listed below:\n{terms_numbered}\n\n"
        f"Text:\n{text}"
    )


def calc_max_tokens(text: str) -> int:
    word_count = len(text.split())
    return min(800, max(150, int(word_count * 4)))


# ────────────────────────────────────────────────
# API CALL WITH RETRY
# ────────────────────────────────────────────────
def to_algospeak(text: str, technique: str, deny_terms_to_transform: list,
                 log_df: pd.DataFrame, model: str) -> tuple:
    """
    Call GPT to produce an algospeak version of `text`.
    Returns (transformed_text, substitutions_dict).

    substitutions_dict maps each original deny term to what GPT replaced it with,
    e.g. {"kill": "k!ll", "gun": "🔫"}. Used to log individual substitutions
    so the deduplication constraints are accurate on future runs.

    Retries on rate-limit / transient API errors with exponential backoff.
    Falls back to treating the raw response as plain text if JSON parsing fails.
    """
    system_prompt = build_system_prompt(technique, deny_terms_to_transform, log_df)
    user_prompt = build_user_prompt(text, technique, deny_terms_to_transform)
    temperature = TECHNIQUE_TEMPERATURE.get(technique, 0.5)
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
                # GPT didn't follow JSON format — treat whole response as plain text
                return raw, {}

        except openai.RateLimitError:
            wait = (2 ** attempt) * 5  # 5s, 10s, 20s
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
    print(f"→ Using column '{text_column}' as source text\n")

    # Delete existing files first so log_df and processed_originals load clean.
    if overwrite:
        for f in [OUTPUT_FILE, TRANSFORMATION_LOG_FILE]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Overwrite: deleted {f}")

    log_df = load_transformation_log(TRANSFORMATION_LOG_FILE)

    # Resume: build set of already-processed originals from existing output CSV.
    # On re-runs without --overwrite, any post already in the output is skipped
    # so we never duplicate LLM calls or overwrite existing results.
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
        print(f"TEST MODE → processing first {limit_rows} rows only\n")
        df_input = df_input.head(limit_rows)
    else:
        print("FULL MODE → processing entire file\n")

    total = len(df_input)
    generated = 0
    skipped_no_deny = 0
    skipped_existing = 0
    errors = 0

    for idx, row in df_input.iterrows():
        original = str(row[text_column]).strip()

        if not original or original.lower() == 'nan':
            continue

        # Skip posts already processed in a previous run
        if original in processed_originals and not overwrite:
            skipped_existing += 1
            continue

        # Detect deny terms (ignores content inside @mentions / #hashtags / URLs)
        deny_terms_detected = get_deny_terms_in_text(original)
        if not deny_terms_detected:
            skipped_no_deny += 1
            print(f"[{idx+1}/{total}] SKIP (no deny terms): {original[:80]}")
            continue

        technique = random.choice(ALGOSPEAK_TECHNIQUES)
        temp = TECHNIQUE_TEMPERATURE[technique]
        term_cap = TECHNIQUE_MAX_TERMS[technique]

        deny_terms_to_transform = deny_terms_detected[:term_cap]

        print(
            f"[{idx+1}/{total}] technique={technique.upper()} | temp={temp} | "
            f"transforming {len(deny_terms_to_transform)}/{len(deny_terms_detected)} "
            f"terms: {deny_terms_to_transform}"
        )
        print(f"  Original:  {original[:120]}")

        try:
            # Protect @mentions, #hashtags, URLs from modification
            masked_text, markup_tokens = protect_markup(original)

            algospeak_masked, substitutions = to_algospeak(
                text=masked_text,
                technique=technique,
                deny_terms_to_transform=deny_terms_to_transform,
                log_df=log_df,
                model=model,
            )

            # Check if GPT silently skipped any terms and retry once on just those.
            # We work on the masked text so placeholders stay intact through the retry.
            missed = get_untransformed_terms(
                restore_markup(algospeak_masked, markup_tokens),
                deny_terms_to_transform,
            )
            if missed:
                print(f"  GPT missed {missed} — retrying on partial result...")
                algospeak_masked, substitutions2 = to_algospeak(
                    text=algospeak_masked,
                    technique=technique,
                    deny_terms_to_transform=missed,
                    log_df=log_df,
                    model=model,
                )
                substitutions.update(substitutions2)

            # Restore protected markup in GPT's output
            algospeak_version = restore_markup(algospeak_masked, markup_tokens)

            # Skip if GPT produced no change at all
            if algospeak_version.strip() == original.strip():
                print(f"  SKIP (GPT returned unchanged text)")
                errors += 1
                continue

            print(f"  Algospeak: {algospeak_version[:120]}")

            append_to_output_csv(
                {
                    'original_text': original,
                    'algospeak_text': algospeak_version,
                    'technique': technique,
                    'deny_terms_detected': "|".join(deny_terms_detected),
                    'deny_terms_transformed': "|".join(deny_terms_to_transform),
                },
                OUTPUT_FILE,
            )

            # Log individual term → substitution pairs (not the full post).
            # Skip substitutions already in the log — cycling reuses existing entries
            # and we don't want duplicates inflating the seen count.
            new_log_entries = [
                {'deny_term': term, 'technique': technique, 'algospeak_output': sub}
                for term in deny_terms_to_transform
                if (sub := substitutions.get(term, ""))
                and sub not in get_seen_outputs(log_df, term, technique)
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
        description="Generate single-technique algospeak variants for posts containing deny-list terms."
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
        help="Path to input CSV (default: data/splits/algospeak_sources.csv). "
             "Must have a 'text' column.",
    )
    args = parser.parse_args()
    process_csv(limit_rows=args.test, model=args.model, overwrite=args.overwrite,
                input_file=args.input)
