#!/usr/bin/env python3
"""
Inference and Threshold Tuning for Dual-Encoder RBE PoC

Run inference on test set and test multiple thresholds.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import yaml
from pathlib import Path
import logging
import json
from tqdm import tqdm
import sys
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from dual_encoder import create_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_class_exemplars():
    """Load and group exemplars by class from ruleset"""
    ruleset_path = Path(__file__).parent.parent.parent / "ruleset+examplers" / "rules_exemplars_dual_encoder.jsonl"
    
    class_exemplars = {0: "", 1: "", 2: ""}  # Default empty for classes without rules
    
    if not ruleset_path.exists():
        logger.warning(f"Ruleset not found: {ruleset_path}, using empty exemplars")
        return class_exemplars
    
    with open(ruleset_path, 'r') as f:
        for line in f:
            rule = json.loads(line)
            target_class = rule['target_class']
            exemplars = rule['exemplars']
            # Concatenate all exemplars for this class
            class_exemplars[target_class] += " ".join(exemplars) + " "
    
    # Strip trailing spaces
    for cls in class_exemplars:
        class_exemplars[cls] = class_exemplars[cls].strip()
    
    logger.info(f"Loaded class exemplars: { {k: len(v.split()) for k,v in class_exemplars.items()} } words per class")
    return class_exemplars

def run_multiclass_inference(model, texts, class_exemplars, device):
    """Run multiclass inference: compute similarity to each class exemplar"""
    model.eval()
    predictions = []
    all_similarities = []  # For each text, list of 3 similarities (one per class)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 32), desc="Multiclass Inference"):
            batch_texts = texts[i:i+32]
            
            # Encode texts once
            text_embeddings = model.encode_text(batch_texts)
            
            # Compute similarities to each class exemplar
            batch_sims = []
            for cls in [0, 1, 2]:
                exemplar = class_exemplars[cls]
                if not exemplar:  # Empty exemplar
                    exemplar = "neutral content"  # Fallback
                
                rule_embedding = model.encode_rules([exemplar] * len(batch_texts))
                sim = F.cosine_similarity(text_embeddings, rule_embedding)
                batch_sims.append(sim.cpu().numpy())
            
            # batch_sims is list of 3 arrays, each of shape (batch_size,)
            # Transpose to get per-text similarities
            for j in range(len(batch_texts)):
                text_sims = [batch_sims[cls][j] for cls in range(3)]
                all_similarities.append(text_sims)
                pred_class = text_sims.index(max(text_sims))  # Argmax
                predictions.append(pred_class)
    
    return predictions, all_similarities

def run_inference(model, texts, exemplars, device):
    """Run inference on texts and exemplars"""
    model.eval()
    similarities = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 32), desc="Inference"):
            batch_texts = texts[i:i+32]
            batch_exemplars = exemplars[i:i+32]
            
            # Encode
            text_embeddings = model.encode_text(batch_texts)
            rule_embeddings = model.encode_rules(batch_exemplars)
            
            # Compute cosine similarity
            sim = F.cosine_similarity(text_embeddings, rule_embeddings)
            similarities.extend(sim.cpu().numpy().tolist())
    
    return similarities

def main():
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument('--model', type=str, default='best_model.pt', 
                       choices=['best_model.pt', 'phase2_model.pt'],
                       help='Which checkpoint to use (Phase 1 or Phase 2)')
    args = parser.parse_args()
    
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Paths
    prepared_dir = Path(__file__).parent.parent / "data" / "prepared"
    checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load test data
    logger.info("Loading test data...")
    test_pairs_path = prepared_dir / "test_pairs" / "pairs.pt"
    if not test_pairs_path.exists():
        logger.error(f"Test pairs not found: {test_pairs_path}")
        logger.info("Please run ruleset_matcher.py first")
        sys.exit(1)
    
    test_data = torch.load(test_pairs_path)
    texts = test_data['texts']
    exemplars = test_data['exemplars']
    labels = test_data['labels'].numpy()
    
    logger.info(f"Loaded {len(texts)} test items")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model = create_model(config)
    checkpoint_path = checkpoint_dir / args.model
    
    if not checkpoint_path.exists():
        logger.error(f"Model checkpoint not found: {checkpoint_path}")
        logger.info(f"Available options: best_model.pt (Phase 1), phase2_model.pt (Phase 2)")
        sys.exit(1)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    
    # Determine phase from checkpoint name
    phase = 2 if 'phase2' in args.model else 1
    logger.info(f"Running inference with Phase {phase} model")
    
    # Load class exemplars for multiclass prediction
    logger.info("Loading class exemplars...")
    class_exemplars = load_class_exemplars()
    
    # Run multiclass inference
    logger.info("Running multiclass inference...")
    predictions, all_similarities = run_multiclass_inference(model, texts, class_exemplars, device)
    
    # Compute multiclass metrics
    logger.info(f"\n📊 Multiclass Metrics (Phase {phase}):")
    accuracy = accuracy_score(labels, predictions)
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    conf_matrix = confusion_matrix(labels, predictions)
    logger.info("Confusion Matrix:")
    logger.info(f"Predicted →  Allowed  Offensive  Mature")
    logger.info(f"Actual ↓")
    class_names = ['Allowed', 'Offensive', 'Mature']
    for i, row in enumerate(conf_matrix):
        logger.info(f"  {class_names[i]:8} {row[0]:8} {row[1]:8} {row[2]:8}")
    
    logger.info("\nClassification Report:")
    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    logger.info(report)
    
    # Save multiclass results
    multiclass_df = pd.DataFrame({
        'id': range(len(texts)),
        'text': texts,
        'true_label': labels,
        'pred_label': predictions,
        'sim_allowed': [s[0] for s in all_similarities],
        'sim_offensive': [s[1] for s in all_similarities],
        'sim_mature': [s[2] for s in all_similarities]
    })
    
    multiclass_path = results_dir / f"multiclass_results_phase{phase}.csv"
    multiclass_df.to_csv(multiclass_path, index=False)
    logger.info(f"Saved multiclass results: {multiclass_path}")
    
    # Save multiclass metrics
    multiclass_metrics = {
        'accuracy': float(accuracy),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report
    }
    metrics_path = results_dir / f"multiclass_metrics_phase{phase}.json"
    with open(metrics_path, 'w') as f:
        json.dump(multiclass_metrics, f, indent=2)
    logger.info(f"Saved multiclass metrics: {metrics_path}")
    
    # Run inference (original similarity-based)
    logger.info("Running original similarity inference...")
    similarities = run_inference(model, texts, exemplars, device)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'id': range(len(texts)),
        'text': texts,
        'label': labels,
        'similarity': similarities
    })
    
    # Save all results with phase suffix
    results_path = results_dir / f"test_results_phase{phase}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved all results: {results_path}")
    
    # Analyze thresholds
    logger.info(f"\nThreshold analysis (Phase {phase}):")
    threshold_stats = []
    
    for threshold in config['thresholds']:
        flagged = (results_df['similarity'] > threshold).sum()
        flagged_ratio = flagged / len(results_df)
        
        logger.info(f"  k={threshold}: {flagged} flagged ({flagged_ratio:.2%})")
        
        threshold_stats.append({
            'threshold': threshold,
            'flagged_count': int(flagged),
            'flagged_ratio': float(flagged_ratio)
        })
        
        # Export flagged items for this threshold
        flagged_df = results_df[results_df['similarity'] > threshold].copy()
        flagged_path = results_dir / f"flagged_for_review_{threshold}_phase{phase}.csv"
        # Add empty column for corrections
        flagged_df['corrected_label'] = ''
        flagged_df[['id', 'text', 'label', 'similarity', 'corrected_label']].to_csv(flagged_path, index=False)
    
    # Save threshold stats
    stats_path = results_dir / f"threshold_stats_phase{phase}.json"
    with open(stats_path, 'w') as f:
        json.dump(threshold_stats, f, indent=2)
    logger.info(f"Saved threshold stats: {stats_path}")
    
    logger.info("✅ Inference complete!")
    
    if phase == 1:
        logger.info(f"\n📝 Next steps for human feedback loop:")
        logger.info(f"  1. Export flagged: python src/feedback.py --export --threshold 0.85")
        logger.info(f"  2. Edit CSV file: flagged_for_review_0.85_phase1.csv")
        logger.info(f"     - Fill 'corrected_label' column (0/1/2)")
        logger.info(f"  3. Import feedback: python src/feedback.py --import flagged_for_review_0.85_phase1_annotated.csv")
        logger.info(f"  4. Retrain Phase 2: python src/train.py --phase 2")
        logger.info(f"  5. New inference: python src/inference.py --model phase2_model.pt")
        logger.info(f"  6. Compare results!")
    elif phase == 2:
        logger.info(f"\n✅ Phase 2 inference complete!")
        logger.info(f"Compare metrics:")
        logger.info(f"  - Phase 1: {results_dir / 'test_results_phase1.csv'}")
        logger.info(f"  - Phase 2: {results_dir / 'test_results_phase2.csv'}")
        logger.info(f"  → Should see improved accuracy on your feedback!")

if __name__ == "__main__":
    main()
