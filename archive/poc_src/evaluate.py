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

CLASS_NAMES = ['Allowed', 'Offensive', 'Mature', 'Algospeak']

def load_results():
    """Load multiclass inference results — prefers phase2, falls back to phase1."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    for phase in [2, 1]:
        results_path = results_dir / f"multiclass_results_phase{phase}.csv"
        if results_path.exists():
            logger.info(f"Loading results from {results_path}")
            return pd.read_csv(results_path)

    logger.error("No results found. Please run inference.py first.")
    return None

def compute_metrics(results_df):
    """Compute 4-class classification metrics."""
    true = results_df['true_label']
    pred = results_df['pred_label']

    accuracy = accuracy_score(true, pred)
    report = classification_report(true, pred, target_names=CLASS_NAMES, labels=[0, 1, 2, 3], output_dict=True)

    return {'accuracy': accuracy, 'report': report}

def plot_similarity_distribution(results_df, results_dir):
    """Plot distribution of per-class similarity scores for each true class."""
    sim_cols = {
        0: 'sim_allowed',
        1: 'sim_offensive',
        2: 'sim_mature',
        3: 'sim_algospeak',
    }
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for i, class_id in enumerate([0, 1, 2, 3]):
        col = sim_cols[class_id]
        if col not in results_df.columns:
            axes[i].set_title(f'{CLASS_NAMES[class_id]} (no data)')
            continue
        class_sims = results_df[results_df['true_label'] == class_id][col]
        axes[i].hist(class_sims, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axes[i].set_title(f'{CLASS_NAMES[class_id]} (n={len(class_sims)})')
        axes[i].set_xlabel('Similarity Score')
        axes[i].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(results_dir / "similarity_distribution.png", dpi=100)
    logger.info("Saved similarity distribution plot")
    plt.close()

def plot_confusion_matrix(results_df, results_dir):
    """Plot 4-class confusion matrix."""
    true = results_df['true_label']
    pred = results_df['pred_label']

    cm = confusion_matrix(true, pred, labels=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (4-class)")
    plt.colorbar(im)

    tick_marks = np.arange(4)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=100)
    logger.info("Saved confusion matrix")
    plt.close()

def plot_roc_curve(results_df, results_dir):
    """Plot one-vs-rest ROC curves for all 4 classes."""
    sim_cols = {
        0: 'sim_allowed',
        1: 'sim_offensive',
        2: 'sim_mature',
        3: 'sim_algospeak',
    }
    colors = ['blue', 'red', 'orange', 'green']

    fig, ax = plt.subplots(figsize=(10, 8))

    for class_id, color in zip([0, 1, 2, 3], colors):
        col = sim_cols[class_id]
        if col not in results_df.columns:
            continue
        y_class = (results_df['true_label'] == class_id).astype(int)
        y_score = results_df[col].values
        fpr, tpr, _ = roc_curve(y_class, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{CLASS_NAMES[class_id]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - One-vs-Rest (4-class)')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "roc_curves.png", dpi=100)
    logger.info("Saved ROC curves")
    plt.close()

def main():
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load results
    logger.info("Loading inference results...")
    results_df = load_results()
    if results_df is None:
        return
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(results_df)
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")

    # Save metrics
    metrics_path = results_dir / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj
        json.dump(_convert(metrics), f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    # Visualizations
    logger.info("Generating visualizations...")
    plot_similarity_distribution(results_df, results_dir)
    plot_confusion_matrix(results_df, results_dir)
    plot_roc_curve(results_df, results_dir)

    # Summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total test items: {len(results_df)}")
    for i, name in enumerate(CLASS_NAMES):
        count = (results_df['true_label'] == i).sum()
        logger.info(f"  Class {i} ({name}): {count}")
    
    logger.info("✅ Evaluation complete!")

if __name__ == "__main__":
    main()
