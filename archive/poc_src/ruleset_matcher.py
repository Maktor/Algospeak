#!/usr/bin/env python3
"""
Ruleset Matcher for Dual-Encoder RBE PoC

Loads rules from JSONL, matches them against texts, concatenates exemplars.
"""

import json
import re
import random
import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RulesetMatcher:
    """Matches texts against keyword rules and returns concatenated exemplars.
    
    Purpose: For each text, find which rules fire (keyword match),
    then concatenate their exemplars to create rule input for the dual encoder.
    """
    
    def __init__(self, rules_jsonl_path):
        """Initialize with rules from JSONL file"""
        self.rules = {}  # Dict mapping rule_id -> {pattern, target_class, exemplars}
        self.rules_by_class = {0: [], 1: [], 2: [], 3: []}  # Track rules per class (0/1/2/3)
        self.all_exemplars = []  # Flat exemplar pool for random fallback
        self._load_rules(rules_jsonl_path)
    
    def _load_rules(self, rules_path):
        """Load rules from JSONL file and index them by rule_id and class"""
        rules_path = Path(rules_path)
        if not rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {rules_path}")
        
        # Read JSONL file (one JSON object per line)
        with open(rules_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                rule = json.loads(line)  # Parse JSON
                rule_id = rule['rule_id']
                target_class = rule['target_class']  # Which class this rule implies (0/1/2)
                pattern = rule['trigger_logic']['pattern'].lower()  # Keyword to match
                exemplars = rule['exemplars']  # Example phrases for this rule
                
                # Store rule with lowercase pattern for case-insensitive matching
                self.rules[rule_id] = {
                    'target_class': target_class,
                    'pattern': pattern,
                    'exemplars': exemplars
                }
                self.all_exemplars.extend(exemplars)
                # Track which rules belong to each class (for fallback if no rules fire)
                self.rules_by_class[target_class].append(rule_id)
        
        logger.info(f"Loaded {len(self.rules)} rules from {rules_path}")
        for class_id in sorted(self.rules_by_class.keys()):
            count = len(self.rules_by_class[class_id])
            logger.info(f"  - Class {class_id}: {count} rules")
        if len(self.all_exemplars) == 0:
            logger.warning("No exemplars found in ruleset – texts with no fired rules will get an empty exemplar string")
    
    def get_fired_rules(self, text):
        """Find all rules that fire (keyword match) for a given text.
        
        Returns: list of rule_ids that matched
        """
        text_lower = text.lower()  # Case-insensitive matching
        fired = []
        
        # Check each rule's pattern against the text
        for rule_id, rule_info in self.rules.items():
            pattern = rule_info['pattern']  # The keyword to match
            # Word-boundary match to avoid partial-word false positives
            if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                fired.append(rule_id)  # Rule fires!
        
        return fired
    
    def get_exemplars_for_text(self, text):
        """Get concatenated exemplars from all fired rules for a text.
        
        If no rules fire, randomly sample exemplars from the entire ruleset.
        This becomes the 'rule input' to the dual encoder.
        """
        # Step 1: Find which rules apply to this text
        fired_rules = self.get_fired_rules(text)
        
        # Step 2: Collect exemplars from fired rules. If nothing matches, randomly
        # sample fallback exemplars from the full ruleset.
        if not fired_rules:
            if self.all_exemplars:
                sample_size = min(10, len(self.all_exemplars))
                exemplar_list = random.sample(self.all_exemplars, sample_size)
            else:
                exemplar_list = []
        else:
            # Use exemplars from all rules that fired
            exemplar_list = []
            for rule_id in fired_rules:
                exemplar_list.extend(self.rules[rule_id]['exemplars'])
        
        # Step 3: Concatenate exemplars (limit to 10 to avoid too-long sequences)
        # This becomes the rule encoder's input
        concatenated = " . ".join(exemplar_list[:10])
        return concatenated
    
    def process_dataset(self, texts, labels):
        """Process a list of texts and labels to create exemplar pairs.
        
        For each text, generate the corresponding exemplar string.
        Returns list of exemplar strings (one per text).
        """
        exemplars_list = []
        
        # Generate exemplars for each text
        for i, text in enumerate(texts):
            exemplars = self.get_exemplars_for_text(text)  # Find matching rules and concatenate exemplars
            exemplars_list.append(exemplars)
            
            # Log progress every 1000 items
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} texts")
        
        return exemplars_list
    
    def save_pairs(self, texts, exemplars, labels, output_path):
        """Save text-exemplar-label pairs to PyTorch tensor for fast loading.
        
        Each row: (text, exemplar_string, class_label)
        This is what the training loop will load and encode.
        """
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)  # Create directory if needed
        
        # Bundle data into dict for PyTorch serialization
        data = {
            'texts': texts,  # List of original text strings
            'exemplars': exemplars,  # List of exemplar strings (one per text)
            'labels': labels  # List of class labels (0/1/2)
        }
        
        # Save as PyTorch .pt file (binary format, fast loading)
        torch.save(data, output_path / "pairs.pt")
        logger.info(f"Saved pairs to {output_path / 'pairs.pt'}")

def main():
    """Main pipeline: load rules → match rules to data → save pairs"""
    # Define paths
    repo_root = Path(__file__).parent.parent.parent
    rules_path = repo_root / "ruleset+examplers" / "rules_exemplars_dual_encoder.jsonl"
    prepared_dir = Path(__file__).parent.parent / "data" / "prepared"
    
    # Initialize the rule matcher (loads rules from JSONL)
    matcher = RulesetMatcher(rules_path)
    
    # Load the data splits that were prepared by encoder_prep.py
    logger.info("Loading data splits...")
    train_data = torch.load(prepared_dir / "train.pt")  # Has 'texts' and 'labels' keys
    val_data = torch.load(prepared_dir / "val.pt")
    test_data = torch.load(prepared_dir / "test.pt")
    
    # For each split (train/val/test), generate exemplars and save pairs
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        logger.info(f"Processing {split_name} split...")
        
        # Generate exemplar string for each text by matching rules
        exemplars = matcher.process_dataset(split_data['texts'], split_data['labels'])
        
        # Save the (text, exemplar, label) pairs to disk for training
        matcher.save_pairs(
            split_data['texts'],
            exemplars,
            split_data['labels'],
            prepared_dir / f"{split_name}_pairs"
        )
    
    logger.info("✅ Ruleset matching complete!")

if __name__ == "__main__":
    main()
