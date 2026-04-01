#!/usr/bin/env python3
"""
Human Feedback Pipeline for Dual-Encoder RBE PoC

Export flagged items for annotation, re-import corrected labels.
"""

import pandas as pd
import json
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_flagged(threshold=0.85):
    """Export flagged items for human review"""
    results_dir = Path(__file__).parent.parent / "results"
    results_path = results_dir / "test_results_phase1.csv"
    
    if not results_path.exists():
        logger.error(f"Results not found: {results_path}")
        logger.info("Please run inference.py first")
        return
    
    logger.info(f"Loading results from {results_path}")
    results_df = pd.read_csv(results_path)
    
    # Filter flagged items
    flagged_df = results_df[results_df['similarity'] > threshold].copy()
    logger.info(f"Found {len(flagged_df)} flagged items (similarity > {threshold})")
    
    # Add columns for human review
    flagged_df['corrected_label'] = ''
    flagged_df['review_notes'] = ''
    
    # Export
    export_path = results_dir / f"flagged_for_review_{threshold}.csv"
    flagged_df[['id', 'text', 'label', 'similarity', 'corrected_label', 'review_notes']].to_csv(export_path, index=False)
    
    logger.info(f"✅ Exported {len(flagged_df)} flagged items to: {export_path}")
    logger.info(f"Next step: Open the CSV file and fill in corrected_label column")

def import_feedback(annotated_csv_path):
    """Import corrected labels from annotated CSV"""
    annotated_path = Path(annotated_csv_path)
    
    if not annotated_path.exists():
        logger.error(f"Annotated CSV not found: {annotated_path}")
        return
    
    logger.info(f"Loading annotated CSV from {annotated_path}")
    annotated_df = pd.read_csv(annotated_path)
    
    # Validate columns
    required_cols = ['id', 'corrected_label']
    if not all(col in annotated_df.columns for col in required_cols):
        logger.error(f"Missing required columns: {required_cols}")
        return
    
    # Filter out rows with empty corrected_label
    feedback_df = annotated_df[annotated_df['corrected_label'].notna()].copy()
    feedback_df = feedback_df[feedback_df['corrected_label'] != '']
    
    logger.info(f"Extracted {len(feedback_df)} annotations from {len(annotated_df)} rows")
    
    # Create feedback objects
    feedback_list = []
    
    for _, row in feedback_df.iterrows():
        try:
            corrected_label = int(row['corrected_label'])
            if corrected_label not in [0, 1, 2]:
                logger.warning(f"Invalid label {corrected_label} for id {row['id']}, skipping")
                continue
            
            feedback_obj = {
                'id': int(row['id']),
                'text': row['text'],
                'original_label': int(row['label']),
                'corrected_label': corrected_label,
                'similarity': float(row['similarity']),
                'review_notes': row.get('review_notes', '')
            }
            feedback_list.append(feedback_obj)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing row {row['id']}: {e}")
            continue
    
    # Save feedback
    feedback_dir = Path(__file__).parent.parent / "data" / "feedback"
    feedback_dir.mkdir(exist_ok=True, parents=True)
    
    feedback_path = feedback_dir / "feedback.jsonl"
    
    with open(feedback_path, 'w') as f:
        for obj in feedback_list:
            f.write(json.dumps(obj) + '\n')
    
    # Summary statistics
    logger.info(f"✅ Imported {len(feedback_list)} corrections to: {feedback_path}")
    
    # Analyze corrections
    corrections = sum(1 for f in feedback_list if f['original_label'] != f['corrected_label'])
    agreements = len(feedback_list) - corrections
    
    logger.info(f"Model agreements: {agreements}/{len(feedback_list)} ({100*agreements/len(feedback_list):.1f}%)")
    logger.info(f"Corrections needed: {corrections}/{len(feedback_list)} ({100*corrections/len(feedback_list):.1f}%)")
    
    # Per-class breakdown
    class_names = {0: 'Allowed', 1: 'Offensive', 2: 'Mature'}
    for class_id in [0, 1, 2]:
        count = sum(1 for f in feedback_list if f['corrected_label'] == class_id)
        logger.info(f"  {class_names[class_id]}: {count} items")

def analyze_feedback():
    """Analyze feedback to understand model errors"""
    feedback_dir = Path(__file__).parent.parent / "data" / "feedback"
    feedback_path = feedback_dir / "feedback.jsonl"
    
    if not feedback_path.exists():
        logger.error(f"Feedback file not found: {feedback_path}")
        logger.info("Please import feedback first using --import")
        return
    
    logger.info(f"Analyzing feedback from {feedback_path}")
    
    feedback_list = []
    with open(feedback_path, 'r') as f:
        for line in f:
            feedback_list.append(json.loads(line))
    
    logger.info(f"Total feedback items: {len(feedback_list)}")
    
    # Confusion analysis
    confusion = {}
    for f in feedback_list:
        orig = f['original_label']
        corrected = f['corrected_label']
        key = f"{orig} -> {corrected}"
        confusion[key] = confusion.get(key, 0) + 1
    
    logger.info("Confusion patterns:")
    class_names = {0: 'Allowed', 1: 'Offensive', 2: 'Mature'}
    for key, count in sorted(confusion.items(), key=lambda x: x[1], reverse=True):
        orig, corrected = key.split(' -> ')
        logger.info(f"  {class_names[int(orig)]} -> {class_names[int(corrected)]}: {count} cases")

def main():
    parser = argparse.ArgumentParser(description="Human feedback pipeline")
    parser.add_argument('--export', action='store_true', help='Export flagged items for review')
    parser.add_argument('--threshold', type=float, default=0.85, help='Similarity threshold')
    parser.add_argument('--import', type=str, dest='import_path', help='Import annotated CSV path')
    parser.add_argument('--analyze', action='store_true', help='Analyze feedback patterns')
    
    args = parser.parse_args()
    
    if args.export:
        export_flagged(args.threshold)
    elif args.import_path:
        import_feedback(args.import_path)
    elif args.analyze:
        analyze_feedback()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
