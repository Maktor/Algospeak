# Synthetic Algospeak Data Generation — Rules & Design

This document covers every rule, constraint, and known limitation of `synthetic_data.py`.
The full dataset has not yet been run with the version of the script that includes all the
updates described here, so behavior at scale may surface new edge cases.

---

## Overview

The pipeline reads posts from `data/splits/algospeak_sources.csv`, detects deny-list terms in each post,
picks a random algospeak technique, and asks GPT to rewrite only the flagged terms using
that technique. The result is saved to `synthetic_algospeak.csv`.

The goal is to produce realistic, varied synthetic algospeak that resembles how real users
evade content moderation — not every word substituted, not the same substitution every time.

---

## 1. Deny List

**File:** `Algospeak_experiment/deny_list.txt`

### Loading rules
- One term or phrase per line. Spaces within a line are preserved — multi-word phrases
  like `"sexual assault"` or `"eating me up"` are treated as a single atomic token, not
  split into individual words.
- Lines are lowercased and deduplicated on load.
- BOM-safe encoding (`utf-8-sig`).

### Why this matters
The original loading code replaced all separators (including newlines) with spaces and
then split on spaces. This shredded multi-word phrases into individual tokens, causing
common words like `"me"`, `"i"`, `"a"`, `"the"`, `"as"` to appear as deny-list terms
and fire on nearly every post. This was fixed by switching to line-by-line parsing.

### Deny list curation
- Broad single-word entries that match too aggressively in normal text have been removed:
  `human`, `help`, `words`, `story`, `bio`, `save`, `link`, `comments`, `political`,
  `platforms`, `lord`, `money`, `amazon`, `honey`, `black`, `insta`, `twitter`,
  `tiktok`, `profile`, `donation`, `share`, `linked`, `linktree`.

---

## 2. Markup Protection

Before any deny-term detection or GPT call, `@mentions`, `#hashtags`, and `https://` URLs
are extracted from the post and replaced with stable placeholders (`__PROTECTED_0__`,
`__PROTECTED_1__`, etc.).

**Why:** Without this, a term like `"kill"` inside `@killzone` would be detected as a
deny term and GPT would attempt to transform the mention, breaking the link. Placeholders
are restored from GPT's output after the call.

GPT is explicitly instructed in the system prompt: *"Do NOT modify __PROTECTED_N__ tokens."*

The same token always maps to the same placeholder, so if `@user` appears three times in
a post, all three occurrences are correctly restored.

---

## 3. Deny Term Detection

Detection uses word-boundary regex (`\b`) rather than plain substring matching.

**Why plain substring matching fails:**
- `"me" in "sometimes"` → True (false positive)
- `"i" in "suicide"` → True (false positive)
- `"as" in "assault"` → True (false positive)

With word-boundary matching, `\bme\b` only fires when `"me"` appears as a standalone
word, not embedded inside another word.

`re.escape` is applied to each term before building the pattern, so terms containing
special regex characters (parentheses, slashes, apostrophes) are handled safely.

Detection runs on the markup-protected version of the text, so deny terms inside
`@mentions` or URLs are not picked up.

---

## 4. Technique Selection & Partial Transformation

### Partial transformation (naturalness rule)
Real algospeak authors do not transform every single sensitive word in a post. They
typically obfuscate only the word(s) they expect will get them flagged.

**Rule:** The technique is selected first, then ALL detected terms up to the
`TECHNIQUE_MAX_TERMS` cap for that technique are transformed. Terms are sorted
longest-first (most specific first) so that multi-word phrases like `"sexual assault"`
are prioritized over single words like `"assault"`.

**Per-technique term limits** (`TECHNIQUE_MAX_TERMS`):
- `unknown_spelling` → 6
- `known_harmless` → 6
- `abbreviation` → 6
- `phonetic` → 5
- `pictorial` → 4
- `paraphrase` → 3 (paraphrases add words/length; more than 3 makes posts sound unnatural)

### Technique assignment
One technique is assigned randomly per post. The six techniques are:

| Technique | Description |
|---|---|
| `unknown_spelling` | Leet-speak, symbol substitution, unicode lookalikes, character scrambling |
| `known_harmless` | Replace with a real innocent word that sounds or looks similar |
| `abbreviation` | Shorten to acronym, coded shorthand, or "the ___ word" construction |
| `pictorial` | Replace with exactly one emoji — no stacking |
| `paraphrase` | Euphemistic multi-word or single-word substitute |
| `phonetic` | Replace with something that sounds the same when spoken aloud |

---

## 5. Algospeak Techniques — Rules & Examples

### unknown_spelling
Change the spelling using numbers, symbols, or unusual character substitutions.
The result should look garbled but still be recognizable to humans.

Strategies: leet-speak, symbol-only substitution, unicode lookalikes, repeated letters,
mixed strategies.

Examples: `abortion → @b0rt!0n`, `kill → k!ll`, `sex → $3x`, `suicide → su1c1d3`,
`murder → murd3r`, `rape → r@pe`, `cocaine → c0k3`, `hate → h8`, `dead → d34d`,
`gun → g*n`, `drugs → dr*gs`, `shooting → sh00ting`

Temperature: **0.4**

---

### known_harmless
Replace with a real, innocent-sounding word that sounds or looks similar.
The replacement must be a word that actually exists in the dictionary.

Examples: `porn → corn`, `sex → seggs`, `kill → keel`, `drugs → rugs`, `gun → fun`,
`rape → grape`, `cocaine → co-cane`, `shooting → shouting`, `weed → seed`,
`murder → birder`, `suicide → sidewalk`, `assault → a-salt`

Temperature: **0.45**

---

### abbreviation
Shorten to an abbreviation, acronym, coded shorthand, or "the ___ word" construction.

Examples: `sexual assault → SA`, `suicide → the s word`, `white supremacy → WS`,
`marijuana → MJ`, `dead → D-E-D`, `rape → the r-word`, `murder → the m-word`,
`marijuana → 420`, `shooting → the bang bang thing`, `cocaine → the white stuff`

Temperature: **0.35**

---

### pictorial
Replace with **exactly one emoji** that visually or conceptually represents the term.
Never stack multiple emojis for a single term.

Examples: `gun → 🔫`, `death → 💀`, `drugs → 💊`, `sex → 🍆`, `knife → 🔪`,
`weed → 🌿`, `cocaine → ❄️`, `suicide → 🪢`, `shooting → 💥`, `hate → 🤬`,
`blood → 🩸`, `bomb → 💣`

Temperature: **0.4**

---

### paraphrase
Substitute with a euphemistic phrase or single word that conveys the same meaning
indirectly. Can be one word or a short phrase.

Examples: `kill → unalive`, `die → go to sleep forever`, `suicide → self-deletion`,
`rape → forceful encounter`, `murder → permanent vacation`, `shoot → go bang bang`,
`dead → no longer with us`, `drugs → party favors`, `suicide → checking out early`,
`gun → the strap`, `cocaine → nose candy`, `assault → unwanted situation`

Temperature: **0.25**

---

### phonetic
Replace with something that sounds phonetically similar when spoken aloud, even if
spelled completely differently.

Examples: `lesbian → le dollar bean`, `LGBT → leg booty`, `suicide → sewer slide`,
`sex → secks`, `killed → kilt`, `negro → knee grow`, `nigga → knee guh`,
`murder → murda`, `cocaine → co-cane`, `shooting → shootin`, `rape → grapes`,
`assault → a-salt`, `marijuana → mary jane`

Temperature: **0.3**

---

## 6. Variety & Deduplication System

### Transformation log
Every time a term is successfully transformed, the individual substitution (e.g.,
`kill → k!ll`) is recorded in `transformation_log.csv` with the technique used.
The log is loaded at startup and used to build negative constraints for GPT.

**Important:** Earlier versions of the script stored the full transformed post as the
`algospeak_output` for each term instead of just the substitution. This made the
deduplication system inoperative. The fix records only the individual term substitution.
If you have a log file generated before this fix, run with `--overwrite` to clear it.

### Cycling system
Each term+technique pair has a cap of `MAX_UNIQUE_SUBSTITUTIONS = 6` unique substitutions.

| Seen count | GPT instruction |
|---|---|
| 0 | Full creative freedom — no constraints |
| 1–5 | Negative constraint — avoid all previously used substitutions |
| 6+ | Cycling mode — pick one of the 6 existing substitutions, do not invent new ones |

This prevents GPT from producing increasingly poor quality substitutions as it runs out
of obvious options, and keeps generation fast.

The seen count is based on **unique** substitutions (deduplicated). Cycling reuse entries
are not re-logged, so the count stays stable.

---

## 7. Resume & Overwrite Behavior

### Resume (default)
On re-runs without `--overwrite`, posts whose `original_text` already appears in
`synthetic_algospeak.csv` are skipped entirely. This means:
- Crashed or interrupted runs can be safely restarted.
- The same post is never sent to the LLM twice.
- The transformation log is preserved and continues accumulating variety data.

### Overwrite (`--overwrite` flag)
Both `synthetic_algospeak.csv` and `transformation_log.csv` are deleted **before**
anything is loaded into memory. This ensures a truly clean run — no old substitutions
carry over as constraints.

**Bug (fixed):** An earlier version deleted the files after loading the log into memory,
so `log_df` still held the old data even though the file was gone. The delete now happens
first.

---

## 8. GPT Call & Retry Logic

### Structured output
GPT is asked to return a JSON object:
```json
{
  "transformed": "<full transformed text>",
  "substitutions": {"<original_term>": "<what it was replaced with>", ...}
}
```
If GPT does not follow the JSON format, the raw response is treated as plain text and the
`substitutions` dict falls back to empty (meaning the transformation log won't record
individual term entries for that post).

### API retry (transient errors)
On rate limit or API errors, the call retries up to `MAX_RETRIES = 3` times with
exponential backoff: 5s, 10s, 20s.

### Missed term retry
**Known limitation:** GPT sometimes silently skips one or more of the terms it was asked
to transform, particularly when:
- A term is short or ambiguous in context.
- A longer phrase that subsumes the term was already transformed (e.g., `"sexual assault"`
  transformed as a phrase, but standalone `"assault"` in another sentence was skipped).
- The technique is difficult to apply to a specific word and GPT gives up on it.

**Fix:** After each GPT call, `get_untransformed_terms` scans the output with
word-boundary regex to find any requested terms still present unchanged. If any are found,
one additional call is made using the already-partially-transformed text as input,
targeting only the missed terms. Substitutions from both calls are merged before saving.

This does not guarantee 100% transformation — if GPT skips a term twice, it is left
untransformed. The `deny_terms_transformed` column in the output CSV reflects what was
actually requested, not what was confirmed transformed.

---

## 9. Output CSV Schema

`synthetic_algospeak.csv`

| Column | Description |
|---|---|
| `original_text` | The original post text |
| `algospeak_text` | The transformed post with algospeak applied |
| `technique` | The algospeak technique used |
| `deny_terms_detected` | All deny terms found in the post, pipe-separated |
| `deny_terms_transformed` | The subset of terms passed to GPT, pipe-separated |

Note: `deny_terms_detected` may contain more terms than `deny_terms_transformed` due
to the partial transformation rule. Not all detected terms are necessarily present in
the algospeak output.

---

## 10. Configuration Constants

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `gpt-4o` | OpenAI model used. Override with `--model` |
| `TECHNIQUE_MAX_TERMS` | see section 4 | Per-technique cap on how many terms to transform per post |
| `MAX_RETRIES` | `3` | API retry attempts on transient errors |
| `MAX_UNIQUE_SUBSTITUTIONS` | `6` | Unique substitutions per term+technique before cycling |

---

## 11. Running the Script

```bash
# Normal run (resumes from where it left off)
python synthetic_data.py

# Test run on first 100 rows
python synthetic_data.py --test

# Test run on first N rows
python synthetic_data.py --test 50

# Use a specific model
python synthetic_data.py --model gpt-4o-mini

# Full reset — deletes output CSV and transformation log, regenerates everything
python synthetic_data.py --overwrite
```

---

## 12. Known Limitations Summary

| Issue | Status |
|---|---|
| GPT skips terms silently | Partially mitigated — one auto-retry on missed terms |
| GPT ignores JSON format | Gracefully handled — falls back to plain text, log entry skipped |
| Partial transformation means not all deny terms are substituted | By design — mirrors real algospeak behavior |
| Per-technique term limits | Implemented — `TECHNIQUE_MAX_TERMS` dict replaces old global cap |
| Old transformation log entries (full-post format) pollute constraints | Fix: run with `--overwrite` to clear the log |
| Full dataset not yet run with current script version | Pending — behavior at scale unknown |
