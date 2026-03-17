#!/usr/bin/env python3
"""
Evaluation Metrics and Analysis for Dual-Encoder RBE PoC

Generate confusion matrices, ROC curves, and per-class metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    roc_auc_score, accuracy_score, precision_recall_curve
)
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_results():
    """Load inference results"""
    results_path = Path(__file__).parent.parent / "results" / "test_results.csv"
    if not results_path.exists():
        logger.error(f"Results not found: {results_path}")
        logger.info("Please run inference.py first")
        return None
    
    return pd.read_csv(results_path)

def get_predictions_at_threshold(similarities, threshold):
    """Convert similarities to binary predictions at threshold"""
    # Simple: if similarity > threshold, predict class 1 (uncertain), else class 0
    return (similarities > threshold).astype(int)

def compute_metrics(results_df, threshold=0.85):
    """Compute classification metrics at a given threshold"""
    pred = get_predictions_at_threshold(results_df['similarity'], threshold)
    
    # Create binary labels: uncertain (1) vs certain (0)
    true_binary = (results_df['label'] > 0).astype(int)  # Class 0 = certain (allowed), 1+ = uncertain
    
    accuracy = accuracy_score(true_binary, pred)
    
    # Per-class metrics
    report = classification_report(true_binary, pred, output_dict=True)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'report': report
    }

def plot_similarity_distribution(results_df, results_dir):
    """Plot distribution of similarity scores"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    class_names = {0: 'Allowed', 1: 'Offensive', 2: 'Mature'}
    
    for i, class_id in enumerate([0, 1, 2]):
        class_sims = results_df[results_df['label'] == class_id]['similarity']
        axes[i].hist(class_sims, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[i].set_title(f'{class_names[class_id]} (n={len(class_sims)})')
        axes[i].set_xlabel('Similarity Score')
        axes[i].set_ylabel('Count')
        axes[i].axvline(0.85, color='red', linestyle='--', label='k=0.85')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / "similarity_distribution.png", dpi=100)
    logger.info(f"Saved similarity distribution plot")
    plt.close()

def plot_confusion_matrix(results_df, threshold, results_dir):
    """Plot confusion matrix at threshold"""
    # Simple binary classification: flagged vs not flagged
    pred = get_predictions_at_threshold(results_df['similarity'], threshold)
    true = (results_df['label'] > 0).astype(int)
    
    cm = confusion_matrix(true, pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    threshold_label = f"Confusion Matrix (k={threshold})"
    plt.title(threshold_label)
    plt.colorbar(im)
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Certain', 'Flagged'])
    plt.yticks(tick_marks, ['Certain (True)', 'Flagged (True)'])
    
    # Annotate
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.tight_layout()
    plt.savefig(results_dir / f"confusion_matrix_k{threshold}.png", dpi=100)
    logger.info(f"Saved confusion matrix for k={threshold}")
    plt.close()

def plot_roc_curve(results_df, results_dir):
    """Plot ROC curve for multi-class classification"""
    from sklearn.preprocessing import label_binarize
    
    # Binarize labels
    y_true = label_binarize(results_df['label'], classes=[0, 1, 2])
    y_score = results_df['similarity'].values
    
    # Compute ROC for each class (one-vs-rest)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    class_names = {0: 'Allowed', 1: 'Offensive', 2: 'Mature'}
    colors = ['blue', 'red', 'orange']
    
    for i, (class_id, color) in enumerate(zip([0, 1, 2], colors)):
        # Create binary label for this class
        y_class = (results_df['label'] == class_id).astype(int)
        
        # Compute ROC
        fpr, tpr, _ = roc_curve(y_class, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{class_names[class_id]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - One-vs-Rest Classification')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curves.png", dpi=100)
    logger.info(f"Saved ROC curves")
    plt.close()

def main():
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load results
    logger.info("Loading inference results...")
    results_df = load_results()
    if results_df is None:
        return
    
    # Compute metrics at key thresholds
    logger.info("Computing metrics...")
    metrics_by_threshold = {}
    
    for threshold in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        metrics = compute_metrics(results_df, threshold)
        metrics_by_threshold[threshold] = metrics['report']
        
        flagged = (results_df['similarity'] > threshold).sum()
        logger.info(f"  k={threshold}: {flagged} flagged ({100*flagged/len(results_df):.1f}%)")
    
    # Save metrics
    metrics_path = results_dir / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        # Convert numpy types for JSON serialization
        metrics_serializable = {}
        for k, v in metrics_by_threshold.items():
            metrics_serializable[str(k)] = {
                str(k2): float(v2) if isinstance(v2, (int, np.integer, float, np.floating)) else v2
                for k2, v2 in v.items()
            }
        json.dump(metrics_serializable, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")
    
    # Visualizations
    logger.info("Generating visualizations...")
    plot_similarity_distribution(results_df, results_dir)
    plot_confusion_matrix(results_df, 0.85, results_dir)
    plot_roc_curve(results_df, results_dir)
    
    # Summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total test items: {len(results_df)}")
    logger.info(f"  Class 0 (Allowed): {(results_df['label'] == 0).sum()}")
    logger.info(f"  Class 1 (Offensive): {(results_df['label'] == 1).sum()}")
    logger.info(f"  Class 2 (Mature): {(results_df['label'] == 2).sum()}")
    logger.info(f"  Similarity range: [{results_df['similarity'].min():.3f}, {results_df['similarity'].max():.3f}]")
    logger.info(f"  Similarity mean: {results_df['similarity'].mean():.3f}")
    logger.info(f"  Similarity std: {results_df['similarity'].std():.3f}")
    
    logger.info("✅ Evaluation complete!")

if __name__ == "__main__":
    main()
