# Dual-Encoder RBE PoC - Quick Start

## Overview
Proof-of-concept for Rule-Based Embeddings (RBE) dual-encoder system with human feedback loop.

## Setup
1. Branch: `feature/dual-encoder-poc`
2. Data: 50K stratified sample from 200K comments
3. Model: Unfrozen `sentence-transformers/all-miniLM-L6-v2`
4. Training: Contrastive loss, 5 epochs (~2.5-3.5 hrs total on CPU)

## Quick Run
```bash
cd poc
python src/encoder_prep.py
python src/ruleset_matcher.py  
python src/train.py
python src/inference.py
python src/evaluate.py
```

## Key Files
- `config.yaml`: Hyperparameters
- `.claude.md`: Architecture docs
- `src/`: All Python code
- `results/`: Outputs and metrics

## Expected Results
- Training: Loss decreases, validation accuracy improves
- Inference: Similarity scores generated for all test items
- Evaluation: ROC curves, confusion matrix, per-class metrics

## Classes
- 0: Allowed (65%)
- 1: Offensive Language (31%)
- 2: Mature Content (26%)
