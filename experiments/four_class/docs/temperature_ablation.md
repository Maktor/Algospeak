# Temperature Ablation — Four-Class Dual BERTweet

**Date:** 2026-04-22  
**Model:** Dual BERTweet (InfoNCE contrastive loss)  
**Dataset:** ~874 samples/class after balancing (small dataset)

## What temperature controls

In contrastive learning, temperature (τ) controls how sharp the similarity gradients are during training. Lower τ = tighter clusters with steeper loss gradients. Higher τ = softer, more spread-out clusters. With small datasets, lower temperatures risk overfitting because the model is pushed to memorize exact positions of training examples rather than learning generalizable structure. Higher temperatures act as regularization.

## Runs

All runs used the same train/val/test splits (seed=42, group-aware stratified split). `--fresh` used for every run so no checkpoints were shared.

| Run | τ | Epoch stopped | Test acc | Macro F1 | Mean AUC | Algospeak F1 |
|-----|---|---------------|----------|----------|----------|--------------|
| 1   | 0.10 | 20 (ES) | 0.7918 | 0.7957 | 0.9452 | 0.9032 |
| 2   | 0.07 | 14 (ES) | 0.7214 | 0.7256 | 0.8979 | 0.8138 |
| 3   | 0.15 | 10 (ES) | 0.8065 | 0.8083 | 0.9351 | 0.9045 |
| 4   | 0.20 | 9  (ES) | 0.8240 | 0.8252 | 0.9345 | 0.9161 |

ES = early stopping triggered (patience=5 on val_acc).

## Key observations

- **τ=0.07 was the worst** — too sharp for this dataset size, model failed to generalize. "gonna unalive myself" classified as `[Allowed]`.
- **Trend:** test acc and macro F1 improved monotonically from τ=0.07 → τ=0.20.
- **AUC peaked at τ=0.10** and declined slightly after — probability calibration got slightly worse as accuracy improved.
- **τ=0.20 aggregate metrics are the best**, but the model misclassified "gonna unalive myself fr fr cant take this anymore" as `[Allowed]` instead of `[Algospeak]`. This is one of the most canonical algospeak examples (suicide-related coded language) and represents exactly the failure mode this system is trying to prevent.

## Why we chose τ=0.15

τ=0.15 correctly classifies all four example inference probes including "unalive myself" → `[Algospeak]`. The marginal aggregate metric gain at τ=0.20 (+0.0175 test acc, +0.0116 Algospeak F1) does not outweigh a high-profile miss on a safety-critical algospeak pattern. In deployment, a false negative on suicide-related algospeak is a worse failure than a small drop in overall accuracy.

τ=0.15 is also more conservative — if we collect more data later, retraining at a slightly lower temperature would be safer since larger datasets can support tighter clusters.

## Example inference at τ=0.15 (run 3)

| Text | Prediction | Correct? |
|------|-----------|----------|
| I had a great day today, went for a walk in the park. | Allowed | ✓ |
| I'm going to k!ll that n!gga if he shows up again. | Algospeak | ✓ |
| she posted an onlyfans link in her bio | Mature Content | ✓ |
| gonna unalive myself fr fr cant take this anymore | Algospeak | ✓ |

## Final config

```yaml
temperature: 0.15
```

Checkpoint: `experiments/four_class/checkpoints/best_model.pt` (epoch 10, val_loss=0.0166, val_acc=0.8642)
