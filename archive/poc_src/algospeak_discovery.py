#!/usr/bin/env python3
"""
MLM-based Algospeak Candidate Discovery

Implements the euphemism detection approach from:
  Zhu et al. (2021) "Self-Supervised Euphemism Detection and Identification for Content Moderation"

Pipeline:
  1. Load violation posts (class 1 + 2) from classified_data.csv as the corpus
  2. For each target keyword, extract sentences containing it
  3. Mask the target keyword with [MASK] in each sentence
  4. Filter masked sentences using BERT MLM (keep only "informative" ones)
  5. Rank candidate words by their cumulative MLM probability across all informative contexts
  6. Save ranked candidates for human validation

Why violation posts only:
  Algospeak is a transformation of violating content, so we look for words that
  appear in the same sentence-level contexts as known violation terms.
  Benign posts dilute the signal without contributing meaningful candidates.

Usage:
  cd poc
  python src/algospeak_discovery.py                        # Full run
  python src/algospeak_discovery.py --sample 200           # Quick test: 200 sentences/keyword
  python src/algospeak_discovery.py --top-k 30             # Return top 30 candidates
  python src/algospeak_discovery.py --mlm-threshold 3      # Looser informative filter
"""

import json
import re
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Target keywords: the "real" forbidden terms that algospeak
# is designed to replace. Sourced from the deny list categories
# in the paper (self-harm, violence, sexual, slurs).
# ─────────────────────────────────────────────────────────────
TARGET_KEYWORDS = [
    # Self-harm / suicide
    "suicide", "self harm", "cutting", "hang", "die",

    # Violence
    "kill", "murder", "shoot", "stab", "attack", "assault", "beat", "torture",

    # General profanity (high algospeak frequency)
    "fuck", "fucking", "shit", "bitch", "cunt", "ass", "pussy",
    "dick", "cock", "cum", "bastard",

    # Sexual
    "sex", "rape", "porn", "whore", "slut", "anal",

    # Slurs — racial / ethnic
    "nigger", "nigga", "kike", "coon", "cracker", "chink",
    "spic", "paki", "beaner", "gook", "wetback", "tranny", "dyke",

    # Slurs — ableist / identity
    "retarded", "faggot",

    # Ideology / hate
    "nazi", "genocide",

    # Identity
    "gay", "lesbian", "trans",

    # Drugs / weapons
    "gun", "drug", "weed", "crack", "cocaine",
]


def load_corpus(data_csv_path):
    """
    Load violation posts (class 1 = Offensive, class 2 = Mature) from classified_data.csv.

    We restrict to violation classes because:
    - Algospeak transforms violation content to evade detection
    - Allowed posts are noise that dilutes the MLM signal
    """
    logger.info(f"Loading corpus from {data_csv_path}")
    df = pd.read_csv(data_csv_path)

    # classification column is numeric: 0=Allowed, 1=Offensive, 2=Mature
    df_violations = df[df['classification'].isin([1, 2])].copy()

    texts = df_violations['text'].dropna().astype(str).tolist()
    logger.info(f"Loaded {len(texts)} violation posts (class 1 Offensive + class 2 Mature)")
    return texts


def extract_masked_sentences(texts, target_keywords, max_per_keyword=500):
    """
    For each target keyword, find sentences that contain it and create masked versions.

    Masking replaces the target keyword with BERT's [MASK] token so BERT must
    predict what word fits in that context — surfacing semantically similar terms.

    Args:
        texts: list of post strings
        target_keywords: list of known forbidden terms to mask
        max_per_keyword: cap on how many sentences to collect per keyword
                         (prevents one keyword from dominating processing time)

    Returns:
        dict: keyword -> list of (original_text, masked_text) tuples
    """
    masked_by_keyword = defaultdict(list)

    for keyword in target_keywords:
        count = 0
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        for text in texts:
            if pattern.search(text):
                # Replace keyword with [MASK] — case-insensitive
                masked = pattern.sub('[MASK]', text)
                masked_by_keyword[keyword].append((text, masked))
                count += 1
                if count >= max_per_keyword:
                    break

        logger.info(f"  '{keyword}': {count} sentences found")

    total = sum(len(v) for v in masked_by_keyword.values())
    logger.info(f"Total masked sentences: {total}")
    return masked_by_keyword


def load_bert(model_name="bert-base-uncased"):
    """Load BERT tokenizer and masked language model."""
    logger.info(f"Loading BERT: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"BERT loaded on {device}")
    return tokenizer, model, device


def get_top_predictions(masked_text, tokenizer, model, device, top_k=100):
    """
    Run BERT MLM on a single masked sentence.

    Returns the top-k predicted tokens and their probabilities at the [MASK] position.
    Returns empty list if [MASK] is not found or an error occurs.

    Args:
        masked_text: string containing exactly one [MASK] token
        top_k: how many predictions to return
    """
    try:
        inputs = tokenizer(
            masked_text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        ).to(device)

        mask_token_id = tokenizer.mask_token_id
        mask_positions = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]

        if len(mask_positions) == 0:
            return []

        mask_pos = mask_positions[0].item()

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        probs = torch.softmax(logits[0, mask_pos, :], dim=-1)
        top_probs, top_ids = torch.topk(probs, top_k)

        predictions = [
            (tokenizer.decode([tid.item()]).strip(), prob.item())
            for tid, prob in zip(top_ids, top_probs)
        ]
        return predictions

    except Exception as e:
        logger.warning(f"Skipping sentence due to error: {e}")
        return []


def is_informative_context(predictions, target_keywords, threshold=5):
    """
    Determine if a masked sentence is informative (Zhu et al. Section IV-A).

    A sentence is informative if BERT's top-`threshold` predictions include
    any of the target keywords — meaning the context strongly implies one of
    our forbidden terms. Generic sentences like "Why is [MASK] so hard?" get
    filtered out because BERT predicts random words, not target keywords.

    Args:
        predictions: list of (token, prob) from get_top_predictions
        threshold: how many top predictions to check (Zhu et al. use t=5)
    """
    top_tokens = {pred[0].lower() for pred in predictions[:threshold]}

    for keyword in target_keywords:
        # Multi-word keywords: check if any component word appears
        for word in keyword.lower().split():
            if word in top_tokens:
                return True
    return False


def load_checkpoint(checkpoint_path):
    """Load per-keyword results saved so far. Returns empty dict if no checkpoint."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Resuming from checkpoint: {len(data)} keywords already done — {list(data.keys())}")
        # JSON serializes tuples as lists; convert back
        return {kw: [tuple(pair) for pair in pairs] for kw, pairs in data.items()}
    return {}


def save_checkpoint(checkpoint_path, per_keyword):
    """Save completed keyword results to disk for resume on interrupt."""
    with open(checkpoint_path, 'w') as f:
        json.dump(per_keyword, f)


def generate_candidates(masked_by_keyword, tokenizer, model, device,
                        target_keywords, output_dir, mlm_threshold=5, top_k=50):
    """
    Core candidate generation loop (Zhu et al. Section IV-A, Stage 3).

    For each keyword's masked sentences:
      1. Get BERT's top predictions for the [MASK] position
      2. Filter to informative sentences only (is_informative_context)
      3. Accumulate prediction probabilities across all informative sentences
      4. Rank candidates by total accumulated score

    Checkpoints after each keyword so Ctrl+C resumes where it left off.

    Returns:
        per_keyword: dict of keyword -> [(candidate_word, score), ...]
        global_scores: dict of word -> total_score across all keywords
    """
    checkpoint_path = Path(output_dir) / "checkpoint.json"
    per_keyword = load_checkpoint(checkpoint_path)
    global_scores = defaultdict(float)

    # Rebuild global scores from already-completed keywords
    for kw_results in per_keyword.values():
        for word, score in kw_results:
            global_scores[word] += score

    target_set = {kw.lower() for kw in target_keywords}
    stop_tokens = {'the', 'a', 'an', 'is', 'it', 'i', 'to', 'of', 'in',
                   'and', 'or', 'that', 'this', 'my', 'me', 'you', 'he', 'she'}

    keywords_todo = [kw for kw in masked_by_keyword if kw not in per_keyword]
    logger.info(f"{len(per_keyword)} keywords already done, {len(keywords_todo)} remaining")

    for keyword in keywords_todo:
        sentence_pairs = masked_by_keyword[keyword]
        keyword_scores = defaultdict(float)
        informative_count = 0

        logger.info(f"Processing '{keyword}' ({len(sentence_pairs)} sentences)...")

        for original, masked in tqdm(sentence_pairs, desc=f"  {keyword}", unit="sent", leave=False):
            predictions = get_top_predictions(masked, tokenizer, model, device, top_k=100)

            if not predictions:
                continue

            if not is_informative_context(predictions, target_keywords, mlm_threshold):
                continue

            informative_count += 1

            for token, prob in predictions[:top_k]:
                token_clean = token.lower().strip()
                if token_clean in target_set:
                    continue
                if token_clean.startswith('##'):
                    continue
                if len(token_clean) < 2 or token_clean in stop_tokens:
                    continue
                if not any(c.isalpha() for c in token_clean):
                    continue
                keyword_scores[token_clean] += prob

        logger.info(f"  '{keyword}': {informative_count}/{len(sentence_pairs)} informative contexts")

        ranked = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        per_keyword[keyword] = ranked[:top_k]

        for token, score in keyword_scores.items():
            global_scores[token] += score

        # Checkpoint after each keyword — safe to interrupt after this point
        save_checkpoint(checkpoint_path, per_keyword)
        logger.info(f"  Checkpoint saved ({len(per_keyword)}/{len(masked_by_keyword)} keywords done)")

    return per_keyword, global_scores


def save_results(per_keyword, global_scores, output_dir, top_k=50):
    """
    Save discovery results to CSV and a draft JSONL ruleset.

    Outputs:
      - algospeak_candidates.csv: per-keyword ranked candidates for human validation
      - global_candidates.csv: top candidates across all keywords combined
      - candidate_rules_draft.jsonl: draft rules for ruleset_matcher (needs human exemplars)

    The 'validated' column in CSVs is left blank for a human to fill in (yes/no).
    Validated terms + hand-written exemplars get merged into:
      ruleset+examplers/rules_exemplars_dual_encoder.jsonl as class 3 rules.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # ── Per-keyword candidates ──────────────────────────────
    rows = []
    for keyword, candidates in per_keyword.items():
        for rank, (word, score) in enumerate(candidates[:top_k], 1):
            rows.append({
                'target_keyword': keyword,
                'rank': rank,
                'candidate_word': word,
                'mlm_score': round(score, 6),
                'validated': '',      # Human fills: yes / no
                'notes': ''
            })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "algospeak_candidates.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved per-keyword candidates: {csv_path}")

    # ── Global ranking (across all keywords) ───────────────
    global_ranked = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    global_df = pd.DataFrame([
        {'rank': i + 1, 'candidate_word': word,
         'total_mlm_score': round(score, 6), 'validated': '', 'notes': ''}
        for i, (word, score) in enumerate(global_ranked)
    ])
    global_path = output_dir / "global_candidates.csv"
    global_df.to_csv(global_path, index=False)
    logger.info(f"Saved global candidates: {global_path}")

    # ── Draft JSONL rules for ruleset_matcher ──────────────
    # Each validated term becomes a class-3 rule.
    # 'exemplars' is left empty — humans add real example sentences after validation.
    jsonl_path = output_dir / "candidate_rules_draft.jsonl"
    with open(jsonl_path, 'w') as f:
        for rank, (word, score) in enumerate(global_ranked, 1):
            rule_draft = {
                'rule_id': f'algo_{rank:03d}',
                'target_class': 3,                      # Algospeak class
                'trigger_logic': {'pattern': word},
                'exemplars': [],                        # Human fills these after validation
                'mlm_score': round(score, 6),
                'validated': False                      # Human sets to True after review
            }
            f.write(json.dumps(rule_draft) + '\n')
    logger.info(f"Saved draft rules: {jsonl_path}")

    # ── Summary ────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("DISCOVERY COMPLETE — Next steps:")
    logger.info("=" * 60)
    logger.info(f"1. Open {csv_path}")
    logger.info("   - Review each candidate word")
    logger.info("   - Mark 'validated' column: yes / no")
    logger.info(f"2. For validated terms, add exemplar sentences in {jsonl_path}")
    logger.info("   - Exemplars are real sentences showing the algospeak in context")
    logger.info("   - Aim for 3-5 exemplars per term")
    logger.info("3. Merge validated rules into:")
    logger.info("   ruleset+examplers/rules_exemplars_dual_encoder.jsonl")
    logger.info("   as class 3 (Algospeak) entries")
    logger.info("4. Re-run encoder_prep.py and ruleset_matcher.py to include class 3")


def main():
    parser = argparse.ArgumentParser(
        description="MLM-based Algospeak Candidate Discovery (Zhu et al. 2021)"
    )
    parser.add_argument(
        '--sample', type=int, default=500,
        help='Max sentences per keyword to process (default: 500, use ~100 for quick test)'
    )
    parser.add_argument(
        '--top-k', type=int, default=50,
        help='Number of top candidates to return per keyword (default: 50)'
    )
    parser.add_argument(
        '--mlm-threshold', type=int, default=5,
        help='MLM informative context threshold — Zhu et al. optimal is 5 (default: 5)'
    )
    parser.add_argument(
        '--model', type=str, default='bert-base-uncased',
        help='BERT model for MLM (default: bert-base-uncased)'
    )
    args = parser.parse_args()

    # ── Paths ──────────────────────────────────────────────
    repo_root = Path(__file__).parent.parent.parent
    data_csv = repo_root / "final_classified_data" / "classified_data.csv"
    output_dir = Path(__file__).parent.parent / "data" / "synthetic"

    # Ensure output directory exists (gitignored, won't exist on fresh clone)
    output_dir.mkdir(exist_ok=True, parents=True)

    if not data_csv.exists():
        logger.error(f"Data file not found: {data_csv}")
        logger.info("Ensure classified_data.csv exists in final_classified_data/")
        return

    # ── Step 1: Load corpus ────────────────────────────────
    logger.info("Step 1: Loading corpus...")
    texts = load_corpus(data_csv)

    # ── Step 2: Extract masked sentences ──────────────────
    logger.info("Step 2: Extracting masked sentences...")
    masked_by_keyword = extract_masked_sentences(
        texts, TARGET_KEYWORDS, max_per_keyword=args.sample
    )

    total = sum(len(v) for v in masked_by_keyword.values())
    if total == 0:
        logger.error("No masked sentences found. Check that classified_data.csv has violation posts.")
        return

    # ── Step 3: Load BERT ──────────────────────────────────
    logger.info("Step 3: Loading BERT MLM model...")
    tokenizer, model, device = load_bert(args.model)

    # ── Step 4: Generate candidates ────────────────────────
    logger.info("Step 4: Generating algospeak candidates...")
    logger.info(f"  MLM threshold (t): {args.mlm_threshold}")
    logger.info(f"  Top-k candidates:  {args.top_k}")
    per_keyword, global_scores = generate_candidates(
        masked_by_keyword, tokenizer, model, device,
        TARGET_KEYWORDS,
        output_dir=output_dir,
        mlm_threshold=args.mlm_threshold,
        top_k=args.top_k
    )

    # ── Step 5: Save results ───────────────────────────────
    logger.info("Step 5: Saving results...")
    save_results(per_keyword, global_scores, output_dir, top_k=args.top_k)

    # Clean up checkpoint now that final results are saved
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint deleted (run complete)")

    logger.info("✅ Algospeak discovery complete!")


if __name__ == "__main__":
    main()
