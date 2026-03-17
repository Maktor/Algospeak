# Complete Feedback Loop Guide: From Phase 1 Training to Phase 2 Fine-tuning

## The Goal: Model Learns from Your Feedback ✅

Yes! The feedback **DOES** make the model better. Here's exactly how it works:

---

## The Full Workflow

### **Phase 1: Initial Training (What you run first)**

```bash
cd poc
python src/encoder_prep.py       # Prepare data
python src/ruleset_matcher.py    # Match rules to texts
python src/train.py              # Train model on 50K original data
python src/inference.py          # Get predictions
```

**Outputs:**
- ✅ `models/checkpoints/best_model.pt` — Trained model
- ✅ `results/test_results_phase1.csv` — Predictions + similarity scores

---

### **Phase 2: Human Feedback (YOU ANNOTATE)**

This is where your corrections feed back into model improvement!

#### **Step 1: Export Flagged Items**

```bash
python src/feedback.py --export --threshold 0.85
```

**Output:** `results/flagged_for_review_0.85_phase1.csv`

Format:
```csv
id,text,label,similarity,corrected_label
0,"fuck this shit",1,0.89,
1,"hello world",0,0.45,
3,"damn son",1,0.87,
...
```

#### **Step 2: YOU Manually Correct Labels**

Open the CSV in Excel or Google Sheets:

```
BEFORE:                          AFTER:
id | text | predicted | correct | correct_label
3  | damn | 1         | ????    | 2  ← YOU FILL THIS
15 | dumb | 1         | ????    | 1
```

- **Label 0** = Allowed
- **Label 1** = Offensive Language
- **Label 2** = Mature Content
- Leave blank if unsure

Save as: `flagged_for_review_0.85_phase1_annotated.csv`

#### **Step 3: Import Your Corrections**

```bash
python src/feedback.py --import flagged_for_review_0.85_phase1_annotated.csv
```

**Output:** `data/feedback/feedback.jsonl`

This file contains your corrections in JSON format:
```json
{"id": 3, "text": "damn son", "original_label": 1, "corrected_label": 2, "similarity": 0.87}
{"id": 15, "text": "this is dumb", "original_label": 1, "corrected_label": 1, "similarity": 0.92}
...
```

**What the model learns:**
- Item #3: "It wasn't Offensive Language, it was Mature Content!"
- Item #15: "You were right about this one!"

---

### **Phase 2: Retrain with Your Feedback (MODEL LEARNS!)**

Now the model **retrains** using:
1. **Original 50K training data** (unchanged)
2. **Your corrections** (additional feedback signal)

```bash
python src/train.py --phase 2
```

**What happens:**
- Loads Phase 1 checkpoint as warm start
- Mixes original training data + your corrections
- Fine-tunes for 3 epochs (faster than Phase 1 since it's a warm start)
- Saves new model: `models/checkpoints/phase2_model.pt`

**Result:** Model learns from your feedback! ✅

---

### **Phase 3: Compare Performance (Validate Improvement)**

Run new inference with Phase 2 model:

```bash
python src/inference.py --model phase2_model.pt
```

**Outputs:**
- ✅ `results/test_results_phase2.csv` — New predictions
- ✅ Compare with Phase 1: `results/test_results_phase1.csv`

**Compare:**
```
Phase 1 accuracy:  89.5%
Phase 2 accuracy:  91.2%  ← IMPROVED! 🎉
```

The model got better because it learned from your corrections!

---

## Complete Pipeline (End-to-End)

```bash
# =============  PHASE 1 ===============
cd poc
python src/encoder_prep.py
python src/ruleset_matcher.py
python src/train.py                    # Saves: best_model.pt
python src/inference.py                # Saves: test_results_phase1.csv

# (At this point, check: results/flagged_for_review_0.85_phase1.csv)

# ============= HUMAN REVIEW ============
# Edit flagged_for_review_0.85_phase1.csv in Excel
# Fill corrected_label column
# Save as flagged_for_review_0.85_phase1_annotated.csv

# ============ PHASE 2: MODEL LEARNS ==========
python src/feedback.py --import flagged_for_review_0.85_phase1_annotated.csv
python src/train.py --phase 2          # Saves: phase2_model.pt
python src/inference.py --model phase2_model.pt  # Saves: test_results_phase2.csv

# ============ COMPARE RESULTS ============
# Open Excel:
# - test_results_phase1.csv
# - test_results_phase2.csv
# Compare predictions & accuracy!
```

---

## Example: How the Feedback Loop Works

### **Scenario:**
You review 50 flagged items. You find:
- Item #12: Was predicted "Offensive" but should be "Mature" ✗
- Item #27: Was predicted "Offensive" and IS "Offensive" ✓
- Item #45: Was predicted "Allowed" but should be "Offensive" ✗

### **What Happens:**

**Phase 1 Model:**
```
(Text embedding, Rule embedding) --feedforward--> Similarity = 0.78
Decision: "Offensive" (based on similarity threshold)
```

**Phase 2 Model (WITH YOUR FEEDBACK):**
```
Same (Text, Rule) pair
BUT: Model has learned from your correction!
The weights adjusted so that this specific pair now produces similarity = 0.65
Decision: "Mature" (matches your correction!) ✓
```

---

## Expected Results

### **Phase 1 → Phase 2 Improvements:**
- ✅ **Reduced false positives**: Fewer items wrongly flagged as "Offensive"
- ✅ **Better class distinction**: Model learns subtle differences (Mature vs. Offensive)
- ✅ **Adapted embeddings**: BERT weights fine-tune to your specific data patterns

### **Iteration:**
You can repeat this cycle:
1. Phase 2 inference → identify remaining errors
2. Human review → collect more corrections
3. Phase 2b retrain → further improvement

Each iteration makes the model better!

---

## Quick Reference: Commands

| Step | Command | Output |
|------|---------|--------|
| **Phase 1 Setup** | `python src/encoder_prep.py` | train/val/test splits |
| | `python src/ruleset_matcher.py` | exemplar pairs |
| **Phase 1 Train** | `python src/train.py` | `best_model.pt` |
| **Phase 1 Infer** | `python src/inference.py` | `test_results_phase1.csv` |
| **Export Flagged** | `python src/feedback.py --export` | `flagged_for_review_0.85_phase1.csv` |
| **Import Feedback** | `python src/feedback.py --import <file>` | `feedback.jsonl` |
| **Phase 2 Train** | `python src/train.py --phase 2` | `phase2_model.pt` |
| **Phase 2 Infer** | `python src/inference.py --model phase2_model.pt` | `test_results_phase2.csv` |
| **Analyze Feedback** | `python src/feedback.py --analyze` | Error breakdown |

---

## Key Point: YES, The Model Learns! ✅

| Phase | Data | Model State |
|-------|------|-------------|
| **Phase 1** | 50K original comments | Initial unfrozen training |
| **Phase 2** | 50K original + your corrections | Fine-tuned with feedback signals |

The model **retrains** with your corrections mixed in, so it learns from your annotations. Your feedback literally becomes part of the training data for Phase 2!
