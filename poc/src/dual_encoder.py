#!/usr/bin/env python3
"""
Dual Encoder Architecture for RBE PoC

Rule Encoder + Text Encoder with Contrastive Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class DualEncoder(nn.Module):
    """
    This model has TWO encoders:
      - A text encoder: turns a social media post into a list of numbers (embedding)
      - A rule encoder: turns an algospeak rule/example into a list of numbers (embedding)
    Then it compares the two embeddings to decide if the post matches the rule.
    """
    
    def __init__(self, model_name, embedding_dim=384, margin=0.5):
        # Call the parent class (nn.Module) setup — required for PyTorch models
        super(DualEncoder, self).__init__()
        
        # Load the same pre-trained language model for both encoders
        # SentenceTransformer converts text into a fixed-size vector of numbers
        logger.info(f"Loading model: {model_name}")
        self.text_encoder = SentenceTransformer(model_name)  # Encodes the social media post
        self.rule_encoder = SentenceTransformer(model_name)  # Encodes the algospeak rule
        
        self.embedding_dim = embedding_dim  # Size of each embedding vector (default: 384 numbers)
        self.margin = margin                # Minimum gap we want between similar vs. different pairs
        
        # This "similarity head" is a small neural network that takes both embeddings
        # glued together and outputs a single score between 0 and 1:
        #   ~1.0 = post likely matches the rule (algospeak detected)
        #   ~0.0 = post does not match the rule
        self.similarity_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),  # Combine both embeddings (768 numbers) → 256 numbers
            nn.ReLU(),                           # Keep only positive values (adds non-linearity)
            nn.Dropout(0.1),                     # Randomly zero out 10% of values during training to prevent overfitting
            nn.Linear(256, 1),                   # Compress 256 numbers down to a single score
            nn.Sigmoid()                         # Squish the score into the range [0, 1]
        )
        
        logger.info(f"Model initialized with embedding_dim={embedding_dim}, margin={margin}")
    
    def encode_text(self, texts):
        """
        Converts one or more social media posts into embedding vectors.
        Each post becomes a list of 384 numbers that captures its meaning.
        """
        # If a single string is passed instead of a list, wrap it in a list
        if isinstance(texts, str):
            texts = [texts]
        # Encode the texts and return them as PyTorch tensors (GPU-compatible number arrays)
        embeddings = self.text_encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embeddings
    
    def encode_rules(self, rule_exemplars):
        """
        Converts one or more algospeak rules/examples into embedding vectors.
        Works the same way as encode_text but uses the rule encoder.
        """
        # If a single string is passed instead of a list, wrap it in a list
        if isinstance(rule_exemplars, str):
            rule_exemplars = [rule_exemplars]
        # Encode the rules and return them as PyTorch tensors
        embeddings = self.rule_encoder.encode(rule_exemplars, convert_to_tensor=True, show_progress_bar=False)
        return embeddings
    
    def forward(self, text_embeddings, rule_embeddings, labels=None):
        """
        The main pass through the model. Given a post embedding and a rule embedding,
        compute how similar they are. If labels are provided (during training),
        also compute how wrong the model was (the loss).

        Args:
            text_embeddings: vectors for the posts  — shape: (batch_size, 384)
            rule_embeddings: vectors for the rules  — shape: (batch_size, 384)
            labels: correct class labels (only used during training)

        Returns:
            similarity_scores: a score per pair showing how similar they are
            loss: how wrong the model was (only returned during training)
        """
        # Glue the two embeddings side-by-side into one long vector per pair
        # e.g. [post_vector | rule_vector] → shape: (batch_size, 768)
        concatenated = torch.cat([text_embeddings, rule_embeddings], dim=1)
        
        # Pass the combined vector through the similarity head to get a score per pair
        similarity_scores = self.similarity_head(concatenated)  # shape: (batch_size, 1)
        
        # If no labels given (inference/prediction mode), just return the scores
        if labels is None:
            return similarity_scores
        
        # During training: also compute how wrong the predictions were
        loss = self.contrastive_loss(similarity_scores, labels)
        
        return similarity_scores, loss
    
    def contrastive_loss(self, similarities, labels):
        """
        Calculates the training loss using contrastive learning:
          - If two items are the SAME class → their similarity score should be HIGH
          - If two items are DIFFERENT classes → their similarity score should be LOW
        This teaches the model to group similar things together and push different things apart.

        Args:
            similarities: predicted similarity scores — shape: (batch_size, 1)
            labels: true class labels — shape: (batch_size,)

        Returns:
            loss: a single number representing how wrong the model was
        """
        batch_size = labels.size(0)
        
        # Expand labels into a grid so we can compare every label against every other label
        # labels_expanded[i][j] == labels_expanded[j][i], forming a (batch_size x batch_size) matrix
        labels_expanded = labels.unsqueeze(0).expand(batch_size, batch_size)
        
        # Create a matrix where 1.0 = same class pair, 0.0 = different class pair
        labels_pairs = (labels_expanded == labels_expanded.t()).float()
        
        # Expand similarity scores into a (batch_size x batch_size) matrix for pairwise comparison
        sim_expanded = similarities.expand(batch_size, batch_size)
        sim_pairs = sim_expanded * sim_expanded.t()
        
        # Penalize same-class pairs that have LOW similarity (they should be close)
        pos_pairs = labels_pairs * (1 - sim_pairs)
        
        # Penalize different-class pairs that have HIGH similarity (they should be far apart)
        neg_pairs = (1 - labels_pairs) * sim_pairs
        
        # Average the total penalty across all pairs in the batch
        loss = (pos_pairs.sum() + neg_pairs.sum()) / (batch_size * batch_size)
        
        return loss
    
    def similarity_score(self, text_embedding, rule_embedding):
        """
        Computes cosine similarity between a post embedding and a rule embedding.
        Cosine similarity measures the angle between two vectors:
          1.0 = pointing in the exact same direction (very similar)
          0.0 = perpendicular (unrelated)
         -1.0 = opposite directions (very different)
        """
        return F.cosine_similarity(text_embedding, rule_embedding, dim=-1)
    
    def get_device(self):
        """
        Returns which device (CPU or GPU) the model is currently running on.
        Useful for making sure input data is sent to the same device as the model.
        """
        # Peek at the first parameter of the model to find out its device
        return next(self.parameters()).device

def create_model(config):
    """
    A helper/factory function that builds a DualEncoder model from a config dictionary.
    Instead of calling DualEncoder(...) directly with all arguments,
    you can pass a config dict like:
      { 'model_name': '...', 'embedding_dim': 384, 'margin': 0.5 }
    """
    return DualEncoder(
        model_name=config['model_name'],
        embedding_dim=config['embedding_dim'],
        margin=config['margin']
    )
