"""
model.py

Dual BERTweet architecture for algospeak content moderation.

Two independent BERTweet encoders trained jointly with supervised InfoNCE loss:
  - supervised encoder:   receives "[CLASS_LABEL]: text" — class-aware during training
  - unsupervised encoder: receives raw text only — the inference model

At inference, only the unsupervised encoder is used. Its embeddings are compared
to class prototypes (built from training data) via cosine similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class BERTweetEncoder(nn.Module):
    """
    Wraps vinai/bertweet-base and returns an L2-normalized CLS token embedding.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        return F.normalize(cls_emb, dim=-1)           # L2 normalize -> cosine-ready


class DualEncoderModel(nn.Module):
    """
    Two independent BERTweet encoders trained with supervised InfoNCE loss.

    supervised encoder:
        Input: "[CLASS_LABEL]: <text>"  (e.g. "Offensive Language: I hate you")
        Produces class-aware embeddings during training.
        Discarded after training.

    unsupervised encoder:
        Input: raw text
        Trained (via InfoNCE) to match the supervised encoder's embedding space.
        Used exclusively at inference.
    """

    def __init__(self, model_name: str, temperature: float):
        super().__init__()
        self.supervised   = BERTweetEncoder(model_name)
        self.unsupervised = BERTweetEncoder(model_name)
        self.temperature  = temperature

    def forward(
        self,
        sup_ids:    torch.Tensor,
        sup_mask:   torch.Tensor,
        unsup_ids:  torch.Tensor,
        unsup_mask: torch.Tensor,
        labels:     torch.Tensor,
    ):
        e_s = self.supervised(sup_ids, sup_mask)        # [B, D]
        e_u = self.unsupervised(unsup_ids, unsup_mask)  # [B, D]
        loss = supervised_infonce_loss(e_s, e_u, labels, self.temperature)
        return loss, e_s, e_u


def supervised_infonce_loss(
    e_s:         torch.Tensor,
    e_u:         torch.Tensor,
    labels:      torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Cross-encoder supervised InfoNCE loss.

    For each unsupervised embedding e_u_i:
      Positives: all supervised embeddings e_s_j where label_j == label_i
      Negatives: all supervised embeddings e_s_j where label_j != label_i

    Loss = mean_i [ -log( sum_{j: pos} exp(sim_ij/τ) / sum_j exp(sim_ij/τ) ) ]

    Both e_s and e_u are L2-normalized so sim = dot product = cosine similarity.

    Args:
        e_s:         [B, D] supervised encoder embeddings
        e_u:         [B, D] unsupervised encoder embeddings
        labels:      [B]    integer class labels
        temperature: scalar τ (typically 0.07)

    Returns:
        Scalar loss.
    """
    # Similarity matrix: unsupervised queries supervised keys — [B, B]
    sim = torch.mm(e_u, e_s.T) / temperature

    # Positive mask: True where label_j == label_i — [B, B]
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

    # Numerical stability: subtract row max before exp
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim  = torch.exp(sim)
    pos_sum  = (exp_sim * pos_mask).sum(dim=1)   # [B]
    all_sum  = exp_sim.sum(dim=1)                # [B]

    # Skip samples with no positives in this batch (shouldn't happen at batch_size >= num_classes)
    valid = pos_sum > 0
    if not valid.any():
        return torch.tensor(0.0, requires_grad=True, device=e_s.device)

    loss = -torch.log(pos_sum[valid] / all_sum[valid])
    return loss.mean()
