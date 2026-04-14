# Experiment Log

Tracks all training runs for the Algospeak classifier. Intended for use in the project paper.

---

## Run 1 — Baseline (Old Architecture)
**Date:** ~2026-03-28  
**Branch:** pre-gen2  
**Architecture:** Original POC (ruleset-based + early dual encoder attempt)  
**Data:** classified_data.csv, no synthetic class 3, no group-aware split  
**Results:**
- Test accuracy: ~78%
- Macro F1: ~77.8%
- Notes: Ruled out ruleset/exemplar approach. Transitioned to dual BERTweet with InfoNCE loss.

---

## Run 2 — Gen2 First Attempt
**Date:** 2026-04-06  
**Branch:** gen2  
**Architecture:** Dual BERTweet (vinai/bertweet-base), supervised InfoNCE loss, prototype-based inference  
**Data:** 10,000 per class (40,000 total), old deny lists, no group-aware split  
**Config:** lr=2e-5, temperature=0.07, batch=32, patience=5, fp16  
**Results:**
- Test accuracy: 79.88%
- Macro F1: 80.39%
- Mean AUC: 94.53%
- Per-class F1: Allowed=0.708, Offensive Language=0.948, Mature Content=0.702, Algospeak=0.858
- Notes: Offensive Language improved significantly from reclassification. Mature Content and Allowed weak.

---

## Run 3 — Group-Aware Splits, 13k per class
**Date:** 2026-04-12  
**Branch:** gen2  
**Architecture:** Dual BERTweet (same as Run 2)  
**Data:** 13,260 per class (53,040 total), group-aware train/val/test split (pairs original + algospeak versions), num_workers=2  
**Config:** lr=2e-5, temperature=0.07, batch=32, patience=3, fp16  
**Results:**
- Val accuracy: 86.8%, Macro F1: 86.7%, Mean AUC: 96.9%
- Test accuracy: 85.9%, Macro F1: 85.9%, Mean AUC: 96.7%
- Per-class test F1: Allowed=0.823, Offensive Language=0.917, Mature Content=0.810, Algospeak=0.885
- Best epoch: 14 of 17
- Notes: Group-aware splitting prevents train/test leakage. num_workers=2 caused overnight hang on subsequent retraining attempt (Windows multiprocessing issue).

---

## Run 4 — num_workers=0 Fix, Weighted Prototypes
**Date:** 2026-04-13  
**Branch:** gen2  
**Architecture:** Dual BERTweet (same as Run 2), weighted class 3 prototype at inference  
**Data:** Same as Run 3 (13,260 per class, 53,040 total, group-aware split)  
**Config:** lr=2e-5, temperature=0.07, batch=32, patience=2, fp16, num_workers=0  
**Inference:** Class 3 prototype weighted by inverse deny-term frequency (underrepresented algospeak types contribute more)  
**Results:**
- Val accuracy: 90.2%
- Test accuracy: TBD (run inference to get exact numbers)
- Best epoch: 20 (hit max epochs — model still improving, could benefit from more epochs)
- Notes: num_workers=0 fixed Windows training hang. Accuracy improved significantly (+4.3% val vs Run 3). Weighted prototype added to better represent rare algospeak types (e.g. "unalive") vs dominant slur substitutions. Known weakness: class 3 synthetic data skewed toward slur substitutions — self-harm algospeak underrepresented.

---

## Known Weaknesses (as of Run 4)
- **"unalive"** and other self-harm algospeak not detected — class 3 training data dominated by slur substitutions (nigger=1013, faggot=787, tranny=715 examples vs kill=75)
- **Mature Content recall weak** (~73%) — some sexual/violence posts leak into Allowed
- **Allowed precision weak** (~72%) — some harmful posts classified as Allowed

## Next Steps
- Generate more diverse synthetic data targeting self-harm algospeak (add "unalive", "sewerslide" to deny_term_hints.json)
- Test with more epochs (current max=20, model hadn't converged at Run 4)
- Consider realistic class distribution test (70% Allowed, 10% each for 1/2/3)
