"""
experiments/four_class/src/model.py

Dual BERTweet architecture for algospeak content moderation.
Identical to poc/src/model.py — copied here so the four_class experiment
is fully self-contained and changes don't affect the main poc run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class BERTweetEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_emb, dim=-1)


class DualEncoderModel(nn.Module):
    def __init__(self, model_name: str, temperature: float):
        super().__init__()
        self.supervised   = BERTweetEncoder(model_name)
        self.unsupervised = BERTweetEncoder(model_name)
        self.temperature  = temperature

    def forward(self, sup_ids, sup_mask, unsup_ids, unsup_mask, labels):
        e_s = self.supervised(sup_ids, sup_mask)
        e_u = self.unsupervised(unsup_ids, unsup_mask)
        loss = supervised_infonce_loss(e_s, e_u, labels, self.temperature)
        return loss, e_s, e_u


def supervised_infonce_loss(e_s, e_u, labels, temperature):
    sim = torch.mm(e_u, e_s.T) / temperature
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    all_sum = exp_sim.sum(dim=1)
    valid = pos_sum > 0
    if not valid.any():
        return torch.tensor(0.0, requires_grad=True, device=e_s.device)
    loss = -torch.log(pos_sum[valid] / all_sum[valid])
    return loss.mean()
