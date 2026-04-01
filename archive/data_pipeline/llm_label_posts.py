#!/usr/bin/env python3
"""
Robust LLM labeling for algospeak (BROAD) using OpenAI Responses API + dictionary as supporting evidence.

Definition (BROAD):
- "Algospeak" = intentional coded/obfuscated language (euphemism, emoji substitutions, leetspeak, masking like p*rn,
  spacing, initials, etc.). It does NOT have to be sensitive-content-related.

IMPORTANT:
- The dictionary is SUPPORTING evidence only. A match is not sufficient. Context decides.
- If unclear, label is_coded="unsure" (do not guess).

Install (uv):
  uv add openai pydantic python-dotenv

Run:
  uv run python llm_label_posts.py --input posts_10000.txt --model gpt-5-nano --limit 500 --overwrite
  uv run python llm_label_posts.py --input posts.txt --model gpt-5-nano --overwrite

Dictionary:
  default: algospeak_dictionary.json (same folder), or set --dict /path/to/algospeak_dictionary.json
"""

#!/usr/bin/env python3
"""
Robust LLM Labeling for Algospeak Detection

This script uses OpenAI's GPT models to label social media posts for algospeak (intentional
coded/obfuscated language). It provides detailed categorization including domain, mechanism,
confidence scores, and reasoning.

FUNCTIONALITY:
- Loads algospeak dictionary for context
- Calls OpenAI API with structured prompts for consistent labeling
- Handles retries, fallbacks, and error recovery
- Outputs labeled data in JSONL format with rich metadata
- Categorizes posts into clean/algospeak/manual review buckets

RUN EXAMPLES:
    # Basic run with defaults
    uv run python llm_label_posts.py --input posts_10000.txt --model gpt-5-nano

    # Limited run for testing
    uv run python llm_label_posts.py --input posts.txt --model gpt-4o-mini --limit 100 --overwrite

    # Full production run
    uv run python llm_label_posts.py --input posts_10000.txt --model gpt-5-nano --limit 500 --overwrite

OUTPUT FILES:
- generated_data/llm_labeled_posts.jsonl: Detailed labeling results
- generated_data/clean_posts.txt: Posts labeled as clean (no algospeak)
- generated_data/algospeak_yes.txt: Posts confirmed to contain algospeak
- generated_data/needs_manual_review.txt: Posts requiring human review
- generated_data/error_posts.txt: Posts that failed processing with error details
- clean_posts.txt: Posts labeled as clean (no algospeak)
- algospeak_yes.txt: Posts with confirmed algospeak
- needs_manual_review.txt: Ambiguous or low-confidence cases
- error_posts.txt: Posts that failed to label

REQUIREMENTS:
- OpenAI API key in .env file
- openai, pydantic, python-dotenv packages
- Algospeak dictionary file (algospeak_dictionary.json)

CONFIGURATION:
- Model selection: gpt-5-nano (fast), gpt-4o-mini (fallback)
- Confidence thresholds and retry logic
- Structured output schema for consistent results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import unicodedata
from typing import Any, Dict, List, Literal, Optional, Set

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


# -----------------------------
# Structured output schema (BROAD)
# -----------------------------
Alg = Literal["yes", "no", "unsure"]

Domain = Literal[
    "sexual",
    "violence",
    "self_harm",
    "drugs",
    "hate",
    "profanity",
    "politics",
    "platforms",
    "commerce",
    "health",
    "other",
    "none",
]

Mechanism = Literal[
    "euphemism",
    "emoji_sub",
    "leetspeak",
    "masking",
    "spacing",
    "homograph",
    "initialism",
    "phonetic",
    "other",
]


class AlgoLabel(BaseModel):
    is_coded: Alg
    domain: Domain
    mechanism: Mechanism
    confidence: float = Field(ge=0.0, le=1.0)
    spans: List[str] = Field(default_factory=list)  # exact token(s) from post
    decoded: List[str] = Field(default_factory=list)  # short guesses like ["porn"], ["tiktok"]
    reason: str = Field(description="Brief explanation of the label decision")
    
    model_config = {"json_schema_extra": {
        "required": ["is_coded", "domain", "mechanism", "confidence", "reason"]
    }}


# -----------------------------
# Few-shot examples (short + high signal)
# -----------------------------
FEW_SHOT_EXAMPLES = r"""
Examples:
1) "link in bio for 🌽" → is_coded=yes, domain=sexual, mechanism=emoji_sub, confidence=0.9
2) "seggs ed 😭" → is_coded=yes, domain=sexual, mechanism=euphemism, confidence=0.85
3) "unalive" → is_coded=yes, domain=violence, mechanism=euphemism, confidence=0.9
4) "that movie was 🔥🔥" → is_coded=no (normal hype slang), domain=none, confidence=0.85
5) "🍉 is good" → is_coded=no (literal fruit), domain=none, confidence=0.9

Context matters most. Dictionary hits alone ≠ algospeak.
""".strip()


SYSTEM_PROMPT = (
    "You label social media posts for ALGOSPEAK (BROAD): intentional coded/obfuscated language.\n"
    "This includes euphemisms, emoji substitutions, leetspeak, masking (p*rn), spacing, homographs, initials, etc.\n\n"
    "Dictionary hits are SUPPORTING EVIDENCE only. A hit is NOT sufficient; context decides.\n"
    "Many emojis/slang are normal (🔥 hype, 💀 laughter). Do NOT label coded unless the post supports a coded/obfuscated intent.\n"
    "If you cannot justify from the text alone, set is_coded='unsure' (do not guess).\n\n"
    "Output STRICT JSON ONLY matching the schema keys:\n"
    "is_coded, domain, mechanism, confidence, spans, decoded, dictionary_hits, reason.\n\n"
    "Rules:\n"
    "- spans must list the exact triggering token(s)/emoji/word(s) seen in the post.\n"
    "- If is_coded='no' => domain='none' and mechanism='other'.\n"
    "- decoded should be short (0-3 items) and only when you can justify.\n\n"
    "Confidence Scoring (0.0-1.0):\n"
    "- 0.90-1.00: Very clear, unambiguous algospeak with strong textual/contextual support. "
    "Multiple corroborating signals or a well-known coded term used in context.\n"
    "- 0.75-0.89: Clear algospeak with good evidence. Intent is fairly obvious despite minor ambiguity.\n"
    "- 0.65-0.74: Plausible algospeak with some context uncertainty or weaker supporting signals. "
    "Could reasonably be interpreted as algospeak but not definitive.\n"
    "- Below 0.65: Use is_coded='unsure' instead. Insufficient confidence to confidently label as 'yes'.\n\n"
    + FEW_SHOT_EXAMPLES
)

FALLBACK_SYSTEM = SYSTEM_PROMPT + "\n\nReturn STRICT JSON ONLY. No extra text."


# -----------------------------
# Filtering (keep emoji-only posts; drop empty/url-only/very short non-emoji)
# -----------------------------
URL_ONLY_RE = re.compile(r"^\s*(https?://\S+)\s*$", re.I)
MALFORMED_URL_RE = re.compile(r"\.(com|org|net|io|co|uk|tv|info|edu|gov)", re.I)
TWITTER_HANDLE_RE = re.compile(r"^@\w+$")
REPEATED_CHAR_RE = re.compile(r"^(.)\1{4,}$")  # 5+ repeated chars


def has_symbol_or_emoji(text: str) -> bool:
    for ch in text:
        if unicodedata.category(ch) == "So":
            return True
    return False


def is_garbage(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    
    # Skip formal URLs
    if URL_ONLY_RE.match(t):
        return True
    
    # Skip malformed URLs (contain .com/.org etc but no proper protocol)
    if MALFORMED_URL_RE.search(t) and not ("http://" in t or "https://" in t):
        # Check if it's >80% domain-like (mostly non-word chars and dots)
        if sum(1 for c in t if c in ".:/@-") / len(t) > 0.4:
            return True
    
    # Skip Twitter handles only
    if TWITTER_HANDLE_RE.match(t):
        return True
    
    # Skip repeated single character (spam)
    if REPEATED_CHAR_RE.match(t):
        return True
    
    # Skip if only numbers/codes
    if t.replace(" ", "").replace("-", "").replace("_", "").isdigit():
        return True
    
    # Skip if >70% symbols/punctuation/emoji (likely spam/noise)
    symbol_count = sum(1 for c in t if not c.isalnum() and not c.isspace())
    if len(t) > 2 and symbol_count / len(t) > 0.7:
        # Exception: keep if has actual emoji (could be expressive)
        if has_symbol_or_emoji(t):
            return False
        return True
    
    # Keep posts with emoji even if short
    if has_symbol_or_emoji(t):
        return False
    
    # Otherwise, drop if very short (less than 4 chars of actual content)
    return len(t) < 4


# -----------------------------
# Resume / hashing
# -----------------------------
SUCCESS_LABELS = {"clean", "questionable", "moderated"}  # keep for compatibility


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def load_success_hashes(jsonl_path: str) -> Set[str]:
    """Only treat SUCCESS rows as done. Errors should be retried next run."""
    done: Set[str] = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("final_label") in SUCCESS_LABELS:
                    h = rec.get("hash")
                    if isinstance(h, str) and h:
                        done.add(h)
            except Exception:
                continue
    return done


# -----------------------------
# Dictionary helpers (supporting evidence)
# -----------------------------
def load_algospeak_dict(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # expecting {token: meaning}
    out: Dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if k is None:
                continue
            ks = str(k)
            if not ks:
                continue
            out[ks] = "" if v is None else str(v)
    return out


def find_dictionary_hits(text: str, d: Dict[str, str], max_hits: int) -> List[Dict[str, str]]:
    """
    Simple substring hits; later you can upgrade to:
      - word boundaries for alphanumerics
      - normalization rules
      - trie/Aho-Corasick for speed
    """
    hits: List[Dict[str, str]] = []
    if not d:
        return hits

    # tiny speed win: only consider tokens whose first char appears
    present_chars = set(text)
    for token, meaning in d.items():
        if not token:
            continue
        if token[0] not in present_chars:
            continue
        if token in text:
            hits.append({"token": token, "meaning": meaning})
            if len(hits) >= max_hits:
                break
    return hits


# -----------------------------
# OpenAI helpers
# -----------------------------
def is_gpt5_model(model: str) -> bool:
    return model.lower().startswith("gpt-5")


def get_output_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", None)
    return txt if isinstance(txt, str) else ""


def is_empty_output_error(msg: str) -> bool:
    return ("Raw output: ''" in msg) or ('Raw output: ""' in msg)


def _parse_must_exist(resp: Any) -> AlgoLabel:
    parsed = getattr(resp, "output_parsed", None)
    if parsed is None:
        raw = get_output_text(resp)
        raise ValueError(f"Structured parse returned None. Raw output: {raw[:600]!r}")
    return parsed


def structured_kwargs_for_model(model: str, effort: str) -> Dict[str, Any]:
    """
    GPT-5 family rejects temperature; use reasoning controls.
    For non-GPT-5, keep it simple.
    """
    if is_gpt5_model(model):
        return {
            "reasoning": {"effort": effort},
            "text": {"verbosity": "low"},
        }
    return {
        "temperature": 0.2,
    }


def call_structured(
    client: OpenAI,
    post_text: str,
    dict_hits: List[Dict[str, str]],
    model: str,
    max_output_tokens: int,
    effort: str,
) -> AlgoLabel:
    user_payload = (
        f"POST:\n{post_text}\n\n"
        f"DICT_HITS (supporting evidence; may be irrelevant):\n{json.dumps(dict_hits, ensure_ascii=False)}"
    )
    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        text_format=AlgoLabel,
        max_output_tokens=max_output_tokens,
        **structured_kwargs_for_model(model, effort),
    )
    return _parse_must_exist(resp)


def call_fallback_json(
    client: OpenAI,
    post_text: str,
    dict_hits: List[Dict[str, str]],
    model: str,
    max_output_tokens: int,
    effort: str,
) -> AlgoLabel:
    user_payload = (
        f"POST:\n{post_text}\n\n"
        f"DICT_HITS (supporting evidence; may be irrelevant):\n{json.dumps(dict_hits, ensure_ascii=False)}"
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": FALLBACK_SYSTEM},
            {"role": "user", "content": user_payload},
        ],
        max_output_tokens=max_output_tokens,
        **structured_kwargs_for_model(model, effort),
    )

    raw = get_output_text(resp).strip()
    if not raw:
        raise ValueError("Fallback JSON parse failed. Raw output: ''")

    json_text = raw
    if not (json_text.startswith("{") and json_text.endswith("}")):
        m = re.search(r"\{.*\}", json_text, flags=re.S)
        if m:
            json_text = m.group(0)

    try:
        data = json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Fallback JSON parse failed. Raw output: {raw[:600]!r}") from e

    try:
        return AlgoLabel.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Fallback JSON validation failed: {e}. Raw: {raw[:600]!r}") from e


def label_one_robust(
    client: OpenAI,
    post_text: str,
    dict_hits: List[Dict[str, str]],
    model: str,
    max_output_tokens: int,
    effort: str,
    structured_attempts_before_fallback: int,
) -> AlgoLabel:
    last_structured_err: Optional[Exception] = None

    for _ in range(structured_attempts_before_fallback):
        try:
            return call_structured(client, post_text, dict_hits, model, max_output_tokens, effort)
        except Exception as e:
            last_structured_err = e

    try:
        return call_fallback_json(client, post_text, dict_hits, model, max_output_tokens, effort)
    except Exception as e:
        raise RuntimeError(f"Structured failed ({last_structured_err}); fallback failed ({e})") from e


def retry_labeling(
    client: OpenAI,
    post_text: str,
    dict_hits: List[Dict[str, str]],
    model: str,
    max_output_tokens: int,
    effort: str,
    max_retries: int,
    structured_attempts_before_fallback: int,
) -> AlgoLabel:
    delay = 1.0
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            return label_one_robust(
                client=client,
                post_text=post_text,
                dict_hits=dict_hits,
                model=model,
                max_output_tokens=max_output_tokens,
                effort=effort,
                structured_attempts_before_fallback=structured_attempts_before_fallback,
            )
        except Exception as e:
            last_err = e
            time.sleep(delay + random.random() * 0.5)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"Failed after {max_retries} retries. Last error: {last_err}")


def label_one_with_retry_and_cascade(
    client: OpenAI,
    post_text: str,
    dict_hits: List[Dict[str, str]],
    model: str,
    fallback_model: str,
    max_output_tokens: int,
    effort: str,
    max_retries: int,
    structured_attempts_before_fallback: int,
) -> AlgoLabel:
    try:
        return retry_labeling(
            client, post_text, dict_hits, model, max_output_tokens, effort, max_retries, structured_attempts_before_fallback
        )
    except Exception as e_primary:
        if not is_empty_output_error(str(e_primary)):
            raise
        return retry_labeling(
            client, post_text, dict_hits, fallback_model, max_output_tokens, effort, max_retries, structured_attempts_before_fallback
        )


# -----------------------------
# Output logic (dataset-building friendly)
# -----------------------------
def is_low_confidence(llm: AlgoLabel, threshold: float) -> bool:
    return llm.confidence < threshold


def needs_manual_review(llm: AlgoLabel, threshold: float) -> bool:
    return (llm.is_coded == "unsure") or is_low_confidence(llm, threshold)


def final_bucket(llm: AlgoLabel) -> str:
    # Keep final buckets simple: clean vs questionable
    if llm.is_coded == "no":
        return "clean"
    return "questionable"


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="posts.txt")
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--fallback-model", default="gpt-4o-mini")

    parser.add_argument("--dict", default="algospeak_dictionary.json", help="JSON {token: meaning}")
    parser.add_argument("--dict-max-hits", type=int, default=8, help="Max dictionary hits per post (lower = faster)")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-jsonl", default="generated_data/llm_labeled_posts.jsonl")
    parser.add_argument("--max-output-tokens", type=int, default=400, help="Reduced for speed/cost (was 700)")
    parser.add_argument("--min-confidence-review", type=float, default=0.65)
    parser.add_argument("--sleep", type=float, default=0.05, help="Reduced for faster throughput (was 0.10)")
    parser.add_argument("--max-retries", type=int, default=5, help="Reduced to save cost (was 10)")
    parser.add_argument("--effort", default="low", choices=["low", "medium", "high"], help="Keep at low for speed")
    parser.add_argument("--structured-attempts", type=int, default=1, help="Skip fallback for speed (was 2)")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env or your environment.")

    # Load dictionary (supporting evidence)
    algodict: Dict[str, str] = {}
    if args.dict and os.path.exists(args.dict):
        try:
            algodict = load_algospeak_dict(args.dict)
            print(f"Loaded dictionary: {args.dict} ({len(algodict)} entries)")
        except Exception as e:
            print(f"WARNING: failed to load dictionary at {args.dict}: {e}. Continuing with empty dict.")
    else:
        print(f"WARNING: dictionary file not found at {args.dict}. Continuing with empty dict.")

    client = OpenAI()

    with open(args.input, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        raw_posts = [line.strip() for line in f if line.strip()]

    posts = [p for p in raw_posts if not is_garbage(p)]
    if args.limit is not None:
        posts = posts[: args.limit]

    print(
        f"Input lines: {len(raw_posts)} | Kept after filter: {len(posts)} | "
        f"Model: {args.model} | Fallback: {args.fallback_model}"
    )

    # Output paths
    out_clean_path = "generated_data/clean_posts.txt"
    out_yes_path = "generated_data/algospeak_yes.txt"
    out_review_path = "generated_data/needs_manual_review.txt"
    out_error_path = "generated_data/error_posts.txt"

    if args.overwrite:
        for path in [args.out_jsonl, out_clean_path, out_yes_path, out_review_path, out_error_path]:
            if os.path.exists(path):
                os.remove(path)

    # Ensure output directories exist
    for path in [args.out_jsonl, out_clean_path, out_yes_path, out_review_path, out_error_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    already_done = load_success_hashes(args.out_jsonl)
    if already_done:
        print(f"Resume: {len(already_done)} successful posts already labeled (will skip those).")

    out_jsonl = open(args.out_jsonl, "a", encoding="utf-8")
    out_clean = open(out_clean_path, "a", encoding="utf-8")
    out_yes = open(out_yes_path, "a", encoding="utf-8")
    out_review = open(out_review_path, "a", encoding="utf-8")
    out_error = open(out_error_path, "a", encoding="utf-8")

    counts: Dict[str, int] = {
        "processed": 0,
        "skipped": 0,
        "clean": 0,
        "algospeak_yes": 0,
        "review": 0,
        "errors": 0,
    }

    for i, post in enumerate(posts, 1):
        h = stable_hash(post)
        if h in already_done:
            counts["skipped"] += 1
            continue

        dict_hits = find_dictionary_hits(post, algodict, max_hits=args.dict_max_hits)

        try:
            llm = label_one_with_retry_and_cascade(
                client=client,
                post_text=post,
                dict_hits=dict_hits,
                model=args.model,
                fallback_model=args.fallback_model,
                max_output_tokens=args.max_output_tokens,
                effort=args.effort,
                max_retries=args.max_retries,
                structured_attempts_before_fallback=args.structured_attempts,
            )

            # Enforce rule: no => none/other (in case model slips)
            if llm.is_coded == "no":
                llm.domain = "none"
                llm.mechanism = "other"
                llm.decoded = []

            needs_review_flag = needs_manual_review(llm, args.min_confidence_review)

            rec: Dict[str, Any] = {
                "hash": h,
                "text": post,
                "is_coded": llm.is_coded,
                "domain": llm.domain,
                "mechanism": llm.mechanism,
                "confidence": llm.confidence,
                "spans": llm.spans,
                "decoded": llm.decoded,
                "reasoning": llm.reason,
                "needs_review": needs_review_flag,
                "model_used": args.model,
            }
            out_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_jsonl.flush()

            already_done.add(h)

            # Mutually exclusive buckets
            if needs_review_flag:
                out_review.write(post + "\n")
                out_review.flush()
                counts["review"] += 1
            elif llm.is_coded == "yes":
                out_yes.write(post + "\n")
                out_yes.flush()
                counts["algospeak_yes"] += 1
            elif llm.is_coded == "no":
                out_clean.write(post + "\n")
                out_clean.flush()
                counts["clean"] += 1
            else:
                # Fallback: uncertain but somehow passed review threshold (shouldn't happen)
                out_review.write(post + "\n")
                out_review.flush()
                counts["review"] += 1

            counts["processed"] += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

        except Exception as e:
            counts["errors"] += 1
            err_rec = {
                "text": post,
                "is_coded": "error",
                "confidence": 0.0,
                "needs_review": True,
                "reasoning": f"LLM labeling failed: {str(e)[:200]}",
                "model_used": args.model,
            }
            out_jsonl.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
            out_jsonl.flush()
            out_error.write(post + "\n")
            out_error.flush()

        if i % 50 == 0:
            print(
                f"Progress {i}/{len(posts)} | processed={counts['processed']} skipped={counts['skipped']} "
                f"errors={counts['errors']} yes={counts['algospeak_yes']} review={counts['review']}"
            )

    out_jsonl.close()
    out_clean.close()
    out_yes.close()
    out_review.close()
    out_error.close()

    print("Done:", counts)
    print(f"Wrote/updated: {args.out_jsonl}, {out_clean_path}, {out_yes_path}, {out_review_path}, {out_error_path}")


if __name__ == "__main__":
    main()
