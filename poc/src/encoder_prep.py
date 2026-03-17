#!/usr/bin/env python3
"""
Data Preparation for Dual-Encoder RBE PoC

Loads classified_data.csv, downsamples to 50K stratified, maps labels, splits train/val/test.
"""

import pandas as pd
import yaml
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    # Find config.yaml in parent directory of src/
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def map_labels(df):
    """Map textual classification labels to integers"""
    # Create mapping: text labels -> numeric IDs (0=Allowed, 1=Offensive, 2=Mature)
    label_map = {
        "Allowed": 0,
        "Offensive Language": 1, 
        "Mature Content": 2,
        0: 0,  # Handle already-numeric labels
        1: 1,
        2: 2,
    }
    
    # Convert 'classification' column to numeric 'label' column
    df['label'] = df['classification'].map(label_map)
    
    # Handle any unmapped labels by removing rows with NaN labels
    if df['label'].isna().any():
        unmapped = df[df['label'].isna()]['classification'].unique()
        logger.warning(f"Found unmapped labels: {unmapped}")
        df = df.dropna(subset=['label'])  # Remove rows with unknown labels
        df['label'] = df['label'].astype(int)  # Convert to integer type
    
    logger.info(f"Label mapping complete")
    logger.info(f"Class distribution after mapping:\n{df['label'].value_counts()}")
    return df

def downsample_stratified(df, target_size, random_state=42):
    """Downsample to target_size while preserving class distribution"""
    # Skip if dataset is already smaller than target
    if len(df) <= target_size:
        logger.info(f"Dataset already <= {target_size}, no downsampling needed")
        return df
    
    # Use stratified sampling to keep class ratios balanced after downsampling
    # stratify=df['label'] ensures each class maintains its % in the smaller sample
    df_sample, _ = train_test_split(
        df, 
        test_size=len(df) - target_size,  # Discard (len(df) - target_size) rows
        random_state=random_state,
        stratify=df['label']  # Keep class distribution proportional
    )
    
    logger.info(f"Downsampled from {len(df)} to {len(df_sample)} comments")
    logger.info(f"New class distribution:\n{df_sample['label'].value_counts(normalize=True)}")
    return df_sample

def split_data(df, train_split=0.7, val_split=0.15, test_split=0.15, random_state=42):
    """Split into train/val/test preserving class distribution in each split"""
    # Verify split percentages sum to 100%
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1"
    
    # STEP 1: Split data into training (70%) and temporary (30% for val+test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=val_split + test_split,
        random_state=random_state,
        stratify=df['label']  # Each split gets balanced class distribution
    )
    
    # STEP 2: Split the temporary 30% into validation (15%) and test (15%)
    val_size = val_split / (val_split + test_split)  # Compute proportion within temp_df
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_size,  # test = 50% of temp_df, val = 50% of temp_df
        random_state=random_state,
        stratify=temp_df['label']  # Keep balanced class distribution
    )
    
    logger.info(f"Train: {len(train_df)} ({len(train_df)/len(df):.1%})")
    logger.info(f"Val: {len(val_df)} ({len(val_df)/len(df):.1%})")
    logger.info(f"Test: {len(test_df)} ({len(test_df)/len(df):.1%})")
    
    # Log class distribution per split
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(f"{split_name} class dist: {split_df['label'].value_counts(normalize=True).to_dict()}")
    
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df, data_dir):
    """Save data splits as PyTorch tensors for fast loading during training"""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)  # Create directory if needed
    
    # Helper function: convert DataFrame rows to PyTorch tensors
    def df_to_tensor(df):
        texts = df['text'].tolist()  # Extract text list
        labels = torch.tensor(df['label'].values, dtype=torch.long)  # Convert labels to tensor
        return {'texts': texts, 'labels': labels}
    
    torch.save(df_to_tensor(train_df), data_dir / "train.pt")
    torch.save(df_to_tensor(val_df), data_dir / "val.pt") 
    torch.save(df_to_tensor(test_df), data_dir / "test.pt")
    
    logger.info(f"Saved splits to {data_dir}")
    logger.info(f"  - train.pt: {len(train_df)} items")
    logger.info(f"  - val.pt: {len(val_df)} items")
    logger.info(f"  - test.pt: {len(test_df)} items")

def main():
    """Main pipeline: load data → map labels → downsample → split → save"""
    config = load_config()
    
    # Define file paths
    repo_root = Path(__file__).parent.parent.parent  # Navigate to project root
    data_csv = repo_root / "final_classified_data" / "classified_data.csv"  # Raw classified data
    prepared_dir = Path(__file__).parent.parent / "data" / "prepared"  # Output directory
    
    # Load raw CSV data
    logger.info(f"Loading data from {data_csv}")
    if not data_csv.exists():
        logger.error(f"Data file not found: {data_csv}")
        sys.exit(1)  # Exit if file not found
    
    df = pd.read_csv(data_csv)  # Load dataset
    logger.info(f"Loaded {len(df)} comments")
    logger.info(f"Original class distribution:\n{df['classification'].value_counts()}")
    
    # STEP 1: Convert text labels ("Allowed", "Offensive Language", etc.) to integers (0, 1, 2)
    df = map_labels(df)
    
    # STEP 2: Reduce dataset from ~200K to 50K while keeping class ratios balanced
    df = downsample_stratified(df, config['data_size'])
    
    # STEP 3: Split into 70% train, 15% val, 15% test (all with balanced classes)
    train_df, val_df, test_df = split_data(
        df, 
        config['train_split'], 
        config['val_split'], 
        config['test_split']
    )
    
    # STEP 4: Save each split as PyTorch tensors for fast loading in training
    save_splits(train_df, val_df, test_df, prepared_dir)
    
    logger.info("✅ Data preparation complete!")

if __name__ == "__main__":
    main()
