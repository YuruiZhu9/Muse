"""Trainable Two-Tower recall model with LLM semantic enhancement."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UserTower(nn.Module):
    """Encodes user features + LLM state embeddings into a user vector."""

    def __init__(self, state_dim: int = 2560, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, user_state_mean: torch.Tensor) -> torch.Tensor:
        x = self.state_proj(user_state_mean)
        x = F.relu(x)
        return F.normalize(self.fc(x), dim=-1)


class ItemTower(nn.Module):
    """Encodes item semantic embeddings into an item vector."""

    def __init__(self, semantic_dim: int = 2560, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.sem_proj = nn.Linear(semantic_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, item_semantic: torch.Tensor) -> torch.Tensor:
        x = self.sem_proj(item_semantic)
        x = F.relu(x)
        return F.normalize(self.fc(x), dim=-1)


class TwoTowerModel(nn.Module):
    """Trainable two-tower model for recall stage."""

    def __init__(self, state_dim: int = 2560, semantic_dim: int = 2560,
                 hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.user_tower = UserTower(state_dim, hidden_dim, output_dim)
        self.item_tower = ItemTower(semantic_dim, hidden_dim, output_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, user_state_mean: torch.Tensor,
                item_semantic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_tower(user_state_mean)
        item_emb = self.item_tower(item_semantic)
        return user_emb, item_emb

    def compute_loss(self, user_embs: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        """In-batch sampled softmax (InfoNCE) loss."""
        logits = torch.matmul(user_embs, item_embs.T) / self.temperature.exp()
        labels = torch.arange(len(user_embs), device=user_embs.device)
        return F.cross_entropy(logits, labels)


def train_two_tower(
    model: TwoTowerModel,
    user_state_embs: np.ndarray,        # (num_users, 5, 2560)
    item_semantic_embs: np.ndarray,     # (num_items, 2560)
    user_item_pairs: List[Tuple[int, int]],  # (user_id, item_id) positive pairs
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> TwoTowerModel:
    """Train the two-tower model with in-batch negatives."""
    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    user_states_t = torch.tensor(user_state_embs, dtype=torch.float32)
    item_sems_t = torch.tensor(item_semantic_embs, dtype=torch.float32)

    n_pairs = len(user_item_pairs)
    if n_pairs == 0:
        return model

    for epoch in range(epochs):
        indices = np.random.permutation(n_pairs)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_pairs, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_users = [user_item_pairs[i][0] for i in batch_idx]
            batch_items = [user_item_pairs[i][1] for i in batch_idx]

            user_mean = user_states_t[batch_users].mean(dim=1).to(DEVICE)
            item_sem = item_sems_t[batch_items].to(DEVICE)

            user_embs, item_embs = model(user_mean, item_sem)
            loss = model.compute_loss(user_embs, item_embs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  [two-tower] epoch={epoch+1}/{epochs} loss={avg_loss:.4f}")

    model.eval()
    return model


def build_two_tower_index(
    model: TwoTowerModel,
    item_semantic_embs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-compute all item embeddings using the trained item tower."""
    model.eval()
    item_sems_t = torch.tensor(item_semantic_embs, dtype=torch.float32)
    all_item_embs: List[np.ndarray] = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(item_sems_t), batch_size):
            batch = item_sems_t[start:start + batch_size].to(DEVICE)
            embs = model.item_tower(batch).cpu().numpy()
            all_item_embs.append(embs)
    item_emb_matrix = np.concatenate(all_item_embs, axis=0)
    return item_emb_matrix
