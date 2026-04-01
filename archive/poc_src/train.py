#!/usr/bin/env python3
"""
Training Loop for Dual-Encoder RBE PoC

Train unfrozen encoders with contrastive loss.
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
import yaml
from pathlib import Path
import logging
import json
from tqdm import tqdm
import sys

from dual_encoder import create_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#this method is used to create the dataset class for the training loop. It takes in the texts, exemplars, and labels and creates a PyTorch dataset that can be used with a DataLoader for batching during training.
class TextRuleDataset(Dataset):
    """Dataset for text-rule exemplar pairs"""
    
    def __init__(self, texts_list, exemplars_list, labels_list):
        self.texts = texts_list
        self.exemplars = exemplars_list
        self.labels = torch.tensor(labels_list, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'exemplars': self.exemplars[idx],
            'label': self.labels[idx]
        }

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

#this method is used to load the dataset for the training loop. It takes in the split name and the prepared directory, and returns a PyTorch dataset that can be used with a DataLoader for batching during training.
def load_dataset(split_name, prepared_dir):
    """Load dataset split from saved pairs"""
    split_path = prepared_dir / f"{split_name}_pairs" / "pairs.pt"
    
    if not split_path.exists():
        logger.error(f"Dataset not found: {split_path}")
        logger.info("Please run ruleset_matcher.py first")
        sys.exit(1)
    
    data = torch.load(split_path)
    return TextRuleDataset(data['texts'], data['exemplars'], data['labels'])


def load_feedback(feedback_dir):
    """
    Load human feedback corrections from feedback.jsonl

    This method loads the human feedback corrections from a JSONL file.
    The file is expected to be located in the feedback directory.

    Parameters
    ----------
    feedback_dir : str
        The directory containing the feedback JSONL file.

    Returns
    -------
    A TextRuleDataset object containing the human feedback corrections.
    If the file is not found, None is returned.
    """
    feedback_path = Path(feedback_dir) / "feedback.jsonl"
    
    if not feedback_path.exists():
        logger.warning(f"Feedback file not found: {feedback_path}")
        return None
    
    feedback_list = []
    with open(feedback_path, 'r') as f:
        for line in f:
            feedback_list.append(json.loads(line))
    
    logger.info(f"Loaded {len(feedback_list)} feedback items")
    
    # Convert to dataset format
    texts = [f['text'] for f in feedback_list]
    exemplars = [""] * len(texts)  # Placeholder; will be filled by ruleset_matcher if needed
    labels = [f['corrected_label'] for f in feedback_list]  # Use corrected labels!
    
    return TextRuleDataset(texts, exemplars, labels)

def merge_datasets(*datasets):
    """Merge multiple datasets"""
    all_texts = []
    all_exemplars = []
    all_labels = []
    
    for ds in datasets:
        if ds is not None:
            all_texts.extend(ds.texts)
            all_exemplars.extend(ds.exemplars)
            all_labels.extend(ds.labels.tolist())
    
    return TextRuleDataset(all_texts, all_exemplars, all_labels)

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        texts = batch['text']
        exemplars = batch['exemplars']
        labels = batch['label'].to(device)
        
        # Encode
        text_embeddings = model.encode_text(texts)
        rule_embeddings = model.encode_rules(exemplars)
        
        # Forward pass
        _, loss = model(text_embeddings, rule_embeddings, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for batch in pbar:
            texts = batch['text']
            exemplars = batch['exemplars']
            labels = batch['label'].to(device)
            
            # Encode
            text_embeddings = model.encode_text(texts)
            rule_embeddings = model.encode_rules(exemplars)
            
            # Forward pass
            _, loss = model(text_embeddings, rule_embeddings, labels)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(val_loader)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Dual-Encoder RBE PoC")
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2], help='Training phase')
    args = parser.parse_args()
    
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Paths
    prepared_dir = Path(__file__).parent.parent / "data" / "prepared"
    feedback_dir = Path(__file__).parent.parent / "data" / "feedback"
    checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    if args.phase == 1:
        logger.info("="*60)
        logger.info("PHASE 1: Training on original data (unfrozen encoders)")
        logger.info("="*60)
        
        # Load datasets
        logger.info("Loading datasets...")
        train_dataset = load_dataset("train", prepared_dir)
        val_dataset = load_dataset("val", prepared_dir)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        checkpoint_path = checkpoint_dir / "best_model.pt"
        
    elif args.phase == 2:
        logger.info("="*60)
        logger.info("PHASE 2: Fine-tuning with human feedback")
        logger.info("="*60)
        
        # Load original training data + human feedback
        logger.info("Loading datasets...")
        train_dataset = load_dataset("train", prepared_dir)
        feedback_dataset = load_feedback(feedback_dir)
        
        if feedback_dataset is None:
            logger.error("Phase 2 requires feedback. Please annotate items first:")
            logger.error("  1. python src/feedback.py --export --threshold 0.85")
            logger.error("  2. Edit flagged_for_review_0.85.csv (add corrected_label)")
            logger.error("  3. python src/feedback.py --import flagged_for_review_0.85_annotated.csv")
            sys.exit(1)
        
        # Merge training data with feedback
        train_dataset = merge_datasets(train_dataset, feedback_dataset)
        logger.info(f"Merged dataset size: {len(train_dataset)}")
        
        val_dataset = load_dataset("val", prepared_dir)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        checkpoint_path = checkpoint_dir / "phase2_model.pt"
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    # For Phase 2, load Phase 1 checkpoint as warm start
    if args.phase == 2:
        phase1_checkpoint = checkpoint_dir / "best_model.pt"
        if phase1_checkpoint.exists():
            logger.info(f"Loading Phase 1 checkpoint: {phase1_checkpoint}")
            model.load_state_dict(torch.load(phase1_checkpoint, map_location='cpu'))
        else:
            logger.warning("Phase 1 checkpoint not found, training from scratch")
    
    model = model.to(device)
    
    # Optimizer and scheduler (linear warmup for first 10% steps, then cosine decay)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )

    total_steps = len(train_loader) * config['num_epochs']
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    history = []
    
    num_epochs = config['num_epochs']
    patience = config.get('early_stopping_patience', 2)
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        
        logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'phase': args.phase,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"✅ Saved best model: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No val improvement for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered")
                break
    
    # Save training history
    history_path = checkpoint_dir / f"training_log_phase{args.phase}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training log: {history_path}")
    
    logger.info("✅ Training complete!")
    
    if args.phase == 1:
        logger.info("\nNext steps:")
        logger.info("  1. Run inference: python src/inference.py")
        logger.info("  2. Export flagged items: python src/feedback.py --export --threshold 0.85")
        logger.info("  3. Annotate in Excel: edit flagged_for_review_0.85.csv")
        logger.info("  4. Import feedback: python src/feedback.py --import flagged_for_review_0.85_annotated.csv")
        logger.info("  5. Retrain Phase 2: python src/train.py --phase 2")
    elif args.phase == 2:
        logger.info("\nNext steps:")
        logger.info("  1. Run new inference: python src/inference.py --model phase2_model.pt")
        logger.info("  2. Compare Phase 1 vs Phase 2 results!")
        logger.info("  3. (Optionally) Repeat feedback loop for more rounds")

if __name__ == "__main__":
    main()
