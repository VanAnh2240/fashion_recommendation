"""
BPR-MF — Bayesian Personalized Ranking Matrix Factorization
Baseline model: không dùng graph, không dùng embedding ảnh.
Chỉ học từ interaction matrix (customer_id, article_id).
Input : (user_idx, pos_item_idx, neg_item_idx)  ← BPR triplet
Output: logit per (user, item)                  ← dot(e_u, e_i) / scale
        caller tự sigmoid khi cần score [0,1]
"""
import torch
import torch.nn as nn


class BPRModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        logit_scale: float = 1.0, 
    ):
        super().__init__()
        self.n_users       = n_users
        self.n_items       = n_items
        self.embedding_dim = embedding_dim
        self.logit_scale   = logit_scale

        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.1)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Trả logit (chưa sigmoid)."""
        e_u = self.user_emb(user_idx)
        e_i = self.item_emb(item_idx)
        return (e_u * e_i).sum(dim=-1) / self.logit_scale

    def bpr_loss(self, user_idx, pos_item_idx, neg_item_idx) -> torch.Tensor:
        pos_score = self.forward(user_idx, pos_item_idx)
        neg_score = self.forward(user_idx, neg_item_idx)
        return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

    @torch.no_grad()
    def predict(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        return self.forward(user_idx, item_idx)