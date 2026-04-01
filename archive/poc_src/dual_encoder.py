#!/usr/bin/env python3
"""
Dual Encoder Architecture for RBE PoC

Rule Encoder + Text Encoder with Contrastive Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class BertEncoder(nn.Module):
    """Small wrapper that turns raw text into pooled BERT embeddings."""

    def __init__(self, model_name, max_length=256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = self.model(**encoded)
        token_embeddings = outputs.last_hidden_state

        # Mean pooling with attention mask to ignore padding tokens.
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * attention_mask, dim=1)
        counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        return summed / counts

class DualEncoder(nn.Module):
    """
    This model has TWO encoders:
      - A text encoder: turns a social media post into a list of numbers (embedding)
      - A rule encoder: turns an algospeak rule/example into a list of numbers (embedding)
    Then it compares the two embeddings to decide if the post matches the rule.
    """
    
    def __init__(self, model_name, embedding_dim=768, margin=0.5, temperature=0.07, max_length=256):
        # Call the parent class (nn.Module) setup — required for PyTorch models
        super(DualEncoder, self).__init__()

        # Load two BERT encoders so text and rules can be learned separately.
        logger.info(f"Loading model: {model_name}")
        self.text_encoder = BertEncoder(model_name, max_length=max_length)  # Encodes the social media post
        self.rule_encoder = BertEncoder(model_name, max_length=max_length)  # Encodes the algospeak rule

        hidden_size = self.text_encoder.model.config.hidden_size
        if embedding_dim != hidden_size:
            logger.warning(
                f"Configured embedding_dim={embedding_dim} does not match model hidden_size={hidden_size}; "
                f"using hidden_size={hidden_size} instead"
            )
            embedding_dim = hidden_size

        self.embedding_dim = embedding_dim  # Size of each embedding vector
        self.margin = margin                # Kept for backward compat; not used by SupCon loss
        self.temperature = temperature      # Temperature for supervised contrastive loss

        logger.info(f"Model initialized with embedding_dim={embedding_dim}, temperature={temperature}")
    
    def encode_text(self, texts):
        """
        Converts one or more social media posts into embedding vectors.
        Each post becomes a list of 384 numbers that captures its meaning.
        """
        return self.text_encoder(texts)
    
    def encode_rules(self, rule_exemplars):
        """
        Converts one or more algospeak rules/examples into embedding vectors.
        Works the same way as encode_text but uses the rule encoder.
        """
        return self.rule_encoder(rule_exemplars)
    
    def forward(self, text_embeddings, rule_embeddings, labels=None):
        """
        The main pass through the model. Given a post embedding and a rule embedding,
        compute how similar they are. If labels are provided (during training),
        also compute how wrong the model was (the loss).

        Args:
            text_embeddings: vectors for the posts  — shape: (batch_size, emb_dim)
            rule_embeddings: vectors for the rules  — shape: (batch_size, emb_dim)
            labels: correct class labels (only used during training)

        Returns:
            similarity_scores: a score per pair showing how similar they are
            loss: how wrong the model was (only returned during training)
        """
        # Per-pair cosine similarity between each text and its matched rule.
        similarity_scores = F.cosine_similarity(text_embeddings, rule_embeddings, dim=-1)

        if labels is None:
            return similarity_scores

        loss = self.contrastive_loss(text_embeddings, rule_embeddings, labels)
        return similarity_scores, loss

    def contrastive_loss(self, text_embeddings, rule_embeddings, labels):
        """
        Supervised contrastive loss (Khosla et al. 2020) for 4-class training.

        Each sample's representation is the mean of its text and rule embeddings.
        Within the batch, pairs that share the same class label are positives;
        all other pairs are negatives. The loss pulls same-class representations
        together and pushes different-class representations apart.

        Args:
            text_embeddings: shape (B, emb_dim)
            rule_embeddings: shape (B, emb_dim)
            labels: integer class labels, shape (B,)

        Returns:
            loss: scalar
        """
        device = text_embeddings.device
        B = text_embeddings.size(0)

        # Average text and rule embeddings, then L2-normalize.
        # This is the representation for each sample in the batch.
        embeddings = F.normalize((text_embeddings + rule_embeddings) / 2.0, dim=-1)  # (B, D)

        # Pairwise cosine similarity matrix (dot product on normalized vectors).
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)

        # Mask: True on the diagonal (self-pairs to exclude).
        mask_self = torch.eye(B, dtype=torch.bool, device=device)

        # Positive mask: same class label, excluding self.
        labels_col = labels.unsqueeze(1)          # (B, 1)
        labels_row = labels.unsqueeze(0)          # (1, B)
        mask_pos = (labels_col == labels_row) & ~mask_self  # (B, B)

        # Numerical stability: subtract row-wise max before exp.
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values.detach()

        exp_sim = torch.exp(sim_matrix) * (~mask_self).float()  # zero out self-pairs

        sum_all = exp_sim.sum(dim=1)                              # (B,)
        sum_pos = (exp_sim * mask_pos.float()).sum(dim=1)         # (B,)

        # Only compute loss for anchors that have at least one positive in the batch.
        has_pos = mask_pos.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        num_pos = mask_pos.float().sum(dim=1).clamp(min=1.0)  # avoid div-by-zero
        loss = -(1.0 / num_pos[has_pos]) * torch.log(sum_pos[has_pos] / sum_all[has_pos].clamp(min=1e-8))
        return loss.mean()
    
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
        margin=config['margin'],
        temperature=config.get('temperature', 0.07),
        max_length=config.get('max_length', 256)
    )
