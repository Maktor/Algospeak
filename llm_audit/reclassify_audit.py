"""
llm_audit/reclassify_audit.py

Audits reclassified_012.csv by re-classifying each post with an LLM.
Outputs a comparison CSV showing where the LLM disagrees with the
team's original labels.

Usage:
    uv run python llm_audit/reclassify_audit.py
    uv run python llm_audit/reclassify_audit.py --provider openai --model gpt-4-turbo --target 100 --overwrite
    uv run python llm_audit/reclassify_audit.py --provider ollama --model qwen2.5:14b --target 500 --overwrite
"""

import argparse
import csv
import os
import time
from pathlib import Path
from collections import defaultdict

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR    = Path(__file__).resolve().parent.parent
INPUT_FILE  = BASE_DIR / "data" / "splits" / "reclassified_012.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "llm_audit.csv"

LABELS = {
    0: "Allowed",
    1: "Obscene Language",
    2: "Mature Content",
    4: "Uncertain",
   -1: "Parse Error",
}

OUTPUT_COLS = [
    "text", "source",
    "old_classification", "new_classification",
    "old_label", "new_label",
    "changed",
]

SYSTEM_PROMPT = """\
You are a content moderation classifier. Classify the social media post below into exactly one of these categories:

0 = Allowed
   Benign posts. Also includes posts that merely discuss or report on bad events neutrally
   (e.g. news headlines, factual statements about violence or crime).
   Example: "the president was assassinated today" → 0 (reporting a fact)

1 = Obscene Language
   Posts containing slurs, hate speech, or explicit offensive language used as an attack
   against a person or group.
   Example: "you stupid [slur]" → 1

2 = Mature Content
   Posts containing sexual content, drug use, self-harm, OR direct calls for / incitement
   of violence against people.
   Example: "those people in the middle east should all be killed" → 2

4 = Uncertain
   Use this only if the post is genuinely ambiguous and you cannot confidently assign
   it to class 0, 1, or 2. For example, if the meaning depends on context you do not have,
   or if the post contains a term that could be offensive or benign depending on usage.
   Example: "I need to fix my tranny" → 4 (could be car transmission or a slur)

Respond with a single integer only: 0, 1, 2, or 4. No explanation, no punctuation.\
"""


def make_client(provider: str) -> openai.OpenAI:
    if provider == "ollama":
        return openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set in .env")
        return openai.OpenAI(api_key=api_key)


def classify(client: openai.OpenAI, model: str, text: str) -> int:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        temperature=0,
        max_tokens=5,
    )
    raw = response.choices[0].message.content.strip()
    first = raw[0] if raw else ""
    return int(first) if first in {"0", "1", "2", "4"} else -1


def main():
    parser = argparse.ArgumentParser(description="LLM reclassification audit")
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama",
                        help="LLM provider (default: ollama)")
    parser.add_argument("--model",    type=str, default=None,
                        help="Model name (default: qwen2.5:7b for ollama, gpt-4o-mini for openai)")
    parser.add_argument("--target",   type=int, default=100,
                        help="Number of posts to process (default: 100)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    # Default models per provider
    if args.model is None:
        args.model = "qwen2.5:7b" if args.provider == "ollama" else "gpt-4-turbo"

    if OUTPUT_FILE.exists() and not args.overwrite:
        raise SystemExit(f"Output file already exists: {OUTPUT_FILE}\nUse --overwrite to replace it.")

    df = pd.read_csv(INPUT_FILE, nrows=args.target)
    print(f"Provider : {args.provider} / {args.model}")
    print(f"Loaded   : {len(df)} rows from {INPUT_FILE.name}")
    print(f"Output   → {OUTPUT_FILE}\n")

    client = make_client(args.provider)

    rows = []
    for i, row in enumerate(df.itertuples(), start=1):
        t0 = time.time()
        new_cls = classify(client, args.model, row.text)
        elapsed = time.time() - t0

        old_cls = int(row.classification)
        changed = old_cls != new_cls

        rows.append({
            "text":               row.text,
            "source":             row.source,
            "old_classification": old_cls,
            "new_classification": new_cls,
            "old_label":          LABELS.get(old_cls, "Unknown"),
            "new_label":          LABELS.get(new_cls, "Unknown"),
            "changed":            changed,
        })

        status = "CHANGED" if changed else "same"
        print(f"[{i:>4}/{args.target}] {status:7s}  {old_cls}→{new_cls}  ({elapsed:.1f}s)  {row.text[:60]!r}")

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLS)
    out_df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL)

    # ── Summary ──────────────────────────────────────────────────────
    total_changed = out_df["changed"].sum()
    uncertain     = (out_df["new_classification"] == 4).sum()
    errors        = (out_df["new_classification"] == -1).sum()
    transitions   = defaultdict(int)
    for r in rows:
        if r["changed"]:
            transitions[f"{r['old_classification']}→{r['new_classification']}"] += 1

    print(f"\n{'─'*50}")
    print(f"Total processed : {len(rows)}")
    print(f"Changed         : {total_changed} ({100*total_changed/len(rows):.1f}%)")
    print(f"Uncertain (4)   : {uncertain} ({100*uncertain/len(rows):.1f}%)")
    print(f"Parse errors    : {errors}")
    if transitions:
        print("Transitions:")
        for k, v in sorted(transitions.items()):
            print(f"  {k}: {v}")
    print(f"Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
