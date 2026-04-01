#!/usr/bin/env python3
"""
Quick reference: Run the full Dual-Encoder RBE PoC pipeline

Usage:
  cd poc
  python run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_directories():
    """Create all gitignored output directories upfront so scripts never fail on missing folders."""
    poc_dir = Path(__file__).parent.parent
    dirs = [
        poc_dir / "data" / "prepared",
        poc_dir / "data" / "synthetic",   # MLM discovery output + future synthetic algospeak
        poc_dir / "data" / "feedback",
        poc_dir / "models" / "checkpoints",
        poc_dir / "results",
    ]
    for d in dirs:
        d.mkdir(exist_ok=True, parents=True)
        logger.info(f"  ✅ {d.relative_to(poc_dir)}")
    logger.info("All output directories ready.")

def run_command(cmd, description):
    """Run a shell command and log output"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Step: {description}")
    logger.info(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
    if result.returncode != 0:
        logger.error(f"❌ Failed: {description}")
        sys.exit(1)
    
    logger.info(f"✅ Complete: {description}")

def main():
    logger.info("Starting Dual-Encoder RBE PoC Pipeline")
    logger.info("This will take ~3-4 hours on CPU (can run overnight)")

    # Pre-flight: create all gitignored output directories
    logger.info("\nCreating output directories...")
    ensure_directories()

    # Step 0: Algospeak candidate discovery (MLM-based, Zhu et al. 2021)
    # Outputs ranked candidate words to data/synthetic/ for human validation.
    # Run with --sample 100 for a quick test; full run uses default 500.
    run_command(
        "python src/algospeak_discovery.py --sample 100",
        "Algospeak Discovery (MLM candidate generation — quick sample)"
    )

    # Step 1: Data preparation
    run_command(
        "python src/encoder_prep.py",
        "Data Preparation (downsample + split)"
    )
    
    # Step 2: Ruleset matching
    run_command(
        "python src/ruleset_matcher.py",
        "Ruleset Matching (exemplar pairing)"
    )
    
    # Step 3: Training
    run_command(
        "python src/train.py",
        "Training (unfrozen encoders, ~2.5-3.5 hrs)"
    )
    
    # Step 4: Inference
    run_command(
        "python src/inference.py",
        "Inference (similarity computation + thresholding)"
    )
    
    # Step 5: Evaluation
    run_command(
        "python src/evaluate.py",
        "Evaluation (ROC curves + metrics)"
    )
    
    logger.info("\n" + "="*60)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Check results/:")
    logger.info("   - test_results.csv: All predictions + similarity scores")
    logger.info("   - flagged_for_review_0.85.csv: Items for human review")
    logger.info("   - roc_curves.png: ROC plots for evaluation")
    logger.info("   - similarity_distribution.png: Score distributions by class")
    logger.info("\n2. Human review (optional):")
    logger.info("   - Open flagged_for_review_0.85.csv in Excel")
    logger.info("   - Fill 'corrected_label' column (0/1/2)")
    logger.info("   - Save as flagged_for_review_0.85_annotated.csv")
    logger.info("   - Run: python src/feedback.py --import flagged_for_review_0.85_annotated.csv")
    logger.info("\n3. Review outputs:")
    logger.info("   - results/evaluation_metrics.json: Per-threshold statistics")
    logger.info("   - models/checkpoints/training_log.json: Training history")

if __name__ == "__main__":
    main()
