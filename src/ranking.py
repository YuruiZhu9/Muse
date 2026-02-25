"""
MuseRecSys - Ranking Layer Module

This module implements the StateEnhancedRankingModel, which combines:
1. Deep Interest Network (DIN) for base recommendation
2. LLM Semantic Enhancement for rich user/item representations
3. MMoE (Multi-gate Mixture-of-Experts) for multi-task learning (CTR + CVR)

Author: MuseRecSys Team
Date: 2025-02-25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# Configuration flag to control LLM feature usage
# Set to False to disable LLM semantic features for faster inference
ENABLE_LLM_FEATURES = True


class StateEnhancedRankingModel(nn.Module):
    """
    State-Enhanced Ranking Model with DIN + LLM Semantic Features + MMoE

    Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        Input Layer                              │
    │  user_id, hist_item_ids, target_item_id, features              │
    └──────────────────────┬──────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                      │
        ▼                                      ▼
┌─────────────────────┐         ┌───────────────────────────┐
│  Part 1: DIN Chain  │         │ Part 2: LLM Enhancement   │
│  - Target Embedding │         │ (if ENABLE_LLM_FEATURES)  │
│  - History Embeds   │         │ - User State Projection   │
│  - Attention Pool   │         │ - Item Semantic Project   │
│  - Base Concat      │         │ - Semantic Attention      │
└──────────┬──────────┘         └───────────┬───────────────┘
           │                                 │
           └───────────────┬─────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  Fusion Layer  │
                  │  Dynamic concat│
                  └────────┬───────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  Part 3: MMoE  │
                  │  - Experts     │
                  │  - Gates       │
                  └────────┬───────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  Output Tasks  │
                  │  - CTR (0-1)   │
                  │  - CVR (0-1)   │
                  └────────────────┘

    Args:
        num_users (int): Number of unique users in the dataset
        num_items (int): Number of unique items in the dataset
        user_feature_dim (int): Dimension of user-side features
        item_feature_dim (int): Dimension of item-side features
        embedding_dim (int): Embedding dimension for ID features (default: 64)
        history_len (int): Maximum length of user history (default: 50)
        llm_semantic_dim (int): Dimension of LLM semantic embeddings (default: 2560)
        llm_proj_dim (int): Projection dimension for LLM features (default: 64)
        num_experts (int): Number of expert networks in MMoE (default: 4)
        expert_hidden_dim (int): Hidden dimension for expert networks (default: 128)
        enable_llm_features (bool): Whether to enable LLM semantic features
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 64,
        history_len: int = 50,
        llm_semantic_dim: int = 2560,
        llm_proj_dim: int = 64,
        num_experts: int = 4,
        expert_hidden_dim: int = 128,
        enable_llm_features: bool = True
    ):
        super(StateEnhancedRankingModel, self).__init__()

        # Store configuration
        self.embedding_dim = embedding_dim
        self.history_len = history_len
        self.llm_semantic_dim = llm_semantic_dim
        self.llm_proj_dim = llm_proj_dim
        self.num_experts = num_experts
        self.enable_llm_features = enable_llm_features

        # ===================================================================
        # Part 1: Base DIN (Deep Interest Network) Components
        # ===================================================================

        # ID Embeddings: Map sparse IDs to dense vectors
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # DIN Attention Layer
        # Computes attention weights between target item and historical items
        # Formula: attention_score = target^T * W * history + b
        self.attention_hidden_dim = embedding_dim * 4  # As per DIN paper
        self.attention_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, self.attention_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.attention_hidden_dim, 1)
        )

        # Base MLP layers after concatenation
        # Concat: [seq_embed, target_embed, user_embed, user_features, item_features]
        base_concat_dim = (
            embedding_dim +      # Sequence embedding (attention pooled)
            embedding_dim +      # Target item embedding
            embedding_dim +      # User embedding
            user_feature_dim +   # Additional user features
            item_feature_dim     # Additional item features
        )

        # ===================================================================
        # Part 2: LLM Semantic Enhancement Components
        # ===================================================================

        if self.enable_llm_features:
            # Projection layers: Map high-dimensional LLM embeddings to compact space
            # user_state_embs: (batch, 5, 2560) -> (batch, 5, 64)
            self.user_state_projection = nn.Linear(llm_semantic_dim, llm_proj_dim)

            # item_semantic_embs: (batch, 2560) -> (batch, 64)
            self.item_semantic_projection = nn.Linear(llm_semantic_dim, llm_proj_dim)

            # Semantic Attention: Fuses user states with target item semantics
            # Query: Target item semantic (batch, 1, 64)
            # Key/Value: User state embeddings (batch, 5, 64)
            self.semantic_attention_hidden_dim = llm_proj_dim * 2
            self.semantic_attention_layer = nn.Sequential(
                nn.Linear(llm_proj_dim * 2, self.semantic_attention_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.semantic_attention_hidden_dim, 1)
            )

            # Calculate final fusion dimension
            # Base concat + Fused User Semantic + Target Item Semantic
            self.fusion_dim = base_concat_dim + (2 * llm_proj_dim)
        else:
            # Without LLM features, only use base concat
            self.fusion_dim = base_concat_dim

        # ===================================================================
        # Part 3: MMoE (Multi-gate Mixture-of-Experts) Components
        # ===================================================================

        # Expert Networks: Shared feature extractors
        # Each expert learns different patterns from the fused representation
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden_dim, expert_hidden_dim),
                nn.ReLU()
            )
            for _ in range(num_experts)
        ])

        # Gate Networks: Task-specific routing
        # Each gate learns to weight experts differently for each task
        # We have 2 tasks: CTR (Click-Through Rate) and CVR (Conversion Rate)
        self.num_tasks = 2
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, num_experts),
                nn.Softmax(dim=-1)  # Softmax for expert weights
            )
            for _ in range(self.num_tasks)
        ])

        # Tower Networks: Task-specific prediction layers
        # Each tower takes the weighted combination of expert outputs
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden_dim, expert_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden_dim // 2, 1),
                nn.Sigmoid()  # Output probability in [0, 1]
            )
            for _ in range(self.num_tasks)
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)

    def din_attention(
        self,
        target_embed: torch.Tensor,
        history_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute DIN-style attention between target and historical items.

        DIN (Deep Interest Network) attention mechanism computes the relevance
        between the target item and each item in the user's history. This allows
        the model to focus on the most relevant historical items.

        Args:
            target_embed: Target item embedding, shape (batch, embedding_dim)
            history_embeds: Historical item embeddings, shape (batch, seq_len, embedding_dim)
            mask: Boolean mask for padding positions, shape (batch, seq_len)
                  True for valid positions, False for padding

        Returns:
            Weighted sequence embedding, shape (batch, embedding_dim)

        Attention Computation Steps:
        1. Expand target to match history sequence length
        2. Concatenate target with each history item
        3. Pass through attention network to get scores
        4. Apply mask (if provided) and softmax to get weights
        5. Weighted sum of history embeddings
        """
        batch_size, seq_len, embed_dim = history_embeds.shape

        # Step 1: Expand target embedding to match sequence length
        # target_embed: (batch, embed_dim) -> (batch, seq_len, embed_dim)
        target_expanded = target_embed.unsqueeze(1).expand(-1, seq_len, -1)

        # Step 2: Concatenate target and history for attention computation
        # concat: (batch, seq_len, 2 * embed_dim)
        concat_input = torch.cat([target_expanded, history_embeds], dim=-1)

        # Step 3: Compute attention scores
        # scores: (batch, seq_len, 1)
        scores = self.attention_layer(concat_input).squeeze(-1)  # (batch, seq_len)

        # Step 4: Apply mask for padded positions
        if mask is not None:
            # Set masked positions to very negative value (~0 after softmax)
            scores = scores.masked_fill(~mask, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Step 5: Compute weighted sum of history embeddings
        # attention_weights: (batch, seq_len) -> (batch, seq_len, 1)
        # weighted sum: (batch, embed_dim)
        weighted_embed = torch.sum(
            history_embeds * attention_weights.unsqueeze(-1),
            dim=1
        )

        return weighted_embed

    def llm_semantic_enhancement(
        self,
        user_state_embs: torch.Tensor,
        item_semantic_embs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute LLM semantic enhancement features.

        This module processes LLM-generated semantic embeddings for users and items,
        then fuses them using a semantic attention mechanism.

        Args:
            user_state_embs: LLM user state embeddings, shape (batch, 5, 2560)
                            Represents 5 aspects of user state (e.g., preferences,
                            intent, context, behavior, demographics)
            item_semantic_embs: LLM item semantic embeddings, shape (batch, 2560)
                               Rich semantic representation of the target item

        Returns:
            fused_user_semantic: Fused user semantic embedding, shape (batch, 64)
            target_item_semantic: Projected target item semantic, shape (batch, 64)

        Processing Steps:
        1. Project high-dimensional LLM embeddings to compact space
        2. Apply semantic attention: target item queries user states
        3. Return fused representation and target item semantics
        """
        batch_size = user_state_embs.shape[0]

        # Step 1: Project LLM embeddings to compact space
        # user_state_embs: (batch, 5, 2560) -> (batch, 5, 64)
        user_state_proj = self.user_state_projection(user_state_embs)

        # item_semantic_embs: (batch, 2560) -> (batch, 64)
        target_item_proj = self.item_semantic_projection(item_semantic_embs)

        # Step 2: Semantic Attention
        # Use target item as query to attend to user's state aspects
        # Query: target_item_proj (batch, 64) -> (batch, 1, 64)
        # Key/Value: user_state_proj (batch, 5, 64)
        target_expanded = target_item_proj.unsqueeze(1)  # (batch, 1, 64)

        # Concatenate query with each user state for attention scoring
        # user_state_expanded: (batch, 5, 64) -> (batch, 5, 64) for broadcasting
        concat_input = torch.cat([
            target_expanded.expand(-1, 5, -1),  # (batch, 5, 64)
            user_state_proj  # (batch, 5, 64)
        ], dim=-1)  # (batch, 5, 128)

        # Compute semantic attention scores
        semantic_scores = self.semantic_attention_layer(concat_input).squeeze(-1)  # (batch, 5)
        semantic_weights = F.softmax(semantic_scores, dim=-1)  # (batch, 5)

        # Step 3: Fused user semantic representation
        # Weighted sum of user states based on target item relevance
        fused_user_semantic = torch.sum(
            user_state_proj * semantic_weights.unsqueeze(-1),
            dim=1
        )  # (batch, 64)

        return fused_user_semantic, target_item_proj

    def mmoe_forward(
        self,
        fused_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MMoE (Multi-gate Mixture-of-Experts).

        MMoE enables multi-task learning by having shared expert networks
        and task-specific gate networks. Each gate learns to weigh experts
        differently for its task.

        Args:
            fused_features: Fused input features, shape (batch, fusion_dim)

        Returns:
            Dictionary containing task predictions:
            - 'ctr': Click-Through Rate prediction, shape (batch, 1)
            - 'cvr': Conversion Rate prediction, shape (batch, 1)

        MMoE Computation Steps:
        1. Pass input through all expert networks
        2. For each task:
           a. Compute gate weights for experts
           b. Weight expert outputs by gate weights
           c. Pass weighted combination through task tower
        3. Return task predictions
        """
        batch_size = fused_features.shape[0]

        # Step 1: Compute expert outputs
        # experts: list of (batch, expert_hidden_dim)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(fused_features))

        # Stack expert outputs: (num_experts, batch, expert_hidden_dim)
        expert_outputs = torch.stack(expert_outputs, dim=0)

        # Transpose for easier processing: (batch, num_experts, expert_hidden_dim)
        expert_outputs = expert_outputs.transpose(0, 1)

        # Step 2: Compute task outputs
        task_outputs = {}
        task_names = ['ctr', 'cvr']

        for task_idx, task_name in enumerate(task_names):
            # Step 2a: Compute gate weights for this task
            # gate: (batch, num_experts)
            gate_weights = self.gates[task_idx](fused_features)

            # Step 2b: Weighted combination of expert outputs
            # gate_weights: (batch, num_experts, 1)
            # expert_outputs: (batch, num_experts, expert_hidden_dim)
            # weighted_expert_output: (batch, expert_hidden_dim)
            weighted_expert_output = torch.sum(
                expert_outputs * gate_weights.unsqueeze(-1),
                dim=1
            )

            # Step 2c: Pass through task tower
            task_output = self.towers[task_idx](weighted_expert_output)
            task_outputs[task_name] = task_output

        return task_outputs

    def forward(
        self,
        user_id: torch.Tensor,
        hist_item_ids: torch.Tensor,
        target_item_id: torch.Tensor,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        user_state_embs: Optional[torch.Tensor] = None,
        item_semantic_embs: Optional[torch.Tensor] = None,
        enable_llm_features: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the StateEnhancedRankingModel.

        Args:
            user_id: User ID tensor, shape (batch,)
            hist_item_ids: Historical item IDs, shape (batch, history_len)
            target_item_id: Target item ID to rank, shape (batch,)
            user_features: Additional user features, shape (batch, user_feature_dim)
            item_features: Additional item features, shape (batch, item_feature_dim)
            user_state_embs: LLM user state embeddings, shape (batch, 5, 2560)
                            Required if enable_llm_features=True
            item_semantic_embs: LLM item semantic embeddings, shape (batch, 2560)
                               Required if enable_llm_features=True
            enable_llm_features: Override the default LLM feature setting.
                               If None, uses self.enable_llm_features

        Returns:
            Dictionary containing:
            - 'ctr': Click-Through Rate prediction, shape (batch, 1)
            - 'cvr': Conversion Rate prediction, shape (batch, 1)

        Processing Flow:
        1. Embedding lookup for IDs
        2. DIN attention on historical items
        3. Base feature concatenation
        4. LLM semantic enhancement (if enabled)
        5. MMoE multi-task prediction
        """
        # Determine if LLM features should be used
        use_llm = enable_llm_features if enable_llm_features is not None else self.enable_llm_features

        batch_size = user_id.shape[0]

        # ===================================================================
        # Part 1: Base DIN Chain
        # ===================================================================

        # ID Embeddings
        user_embed = self.user_embedding(user_id)  # (batch, embedding_dim)
        target_embed = self.item_embedding(target_item_id)  # (batch, embedding_dim)

        # History item embeddings
        # hist_item_ids: (batch, history_len)
        history_embeds = self.item_embedding(hist_item_ids)  # (batch, history_len, embedding_dim)

        # Create mask for padded positions (assuming padding_id = 0)
        hist_mask = hist_item_ids != 0  # (batch, history_len)

        # DIN Attention: Weighted pooling of historical items
        # sequence embedding captures relevant aspects of user history
        seq_embed = self.din_attention(target_embed, history_embeds, hist_mask)

        # Base concatenation: [seq_embed, target_embed, user_embed, user_features, item_features]
        base_concat = torch.cat([
            seq_embed,          # (batch, embedding_dim)
            target_embed,       # (batch, embedding_dim)
            user_embed,         # (batch, embedding_dim)
            user_features,      # (batch, user_feature_dim)
            item_features       # (batch, item_feature_dim)
        ], dim=-1)  # (batch, base_concat_dim)

        # ===================================================================
        # Part 2: LLM Semantic Enhancement (Conditional)
        # ===================================================================

        if use_llm:
            if user_state_embs is None or item_semantic_embs is None:
                raise ValueError(
                    "user_state_embs and item_semantic_embs must be provided "
                    "when enable_llm_features=True"
                )

            # Compute LLM semantic enhancements
            fused_user_semantic, target_item_semantic = self.llm_semantic_enhancement(
                user_state_embs, item_semantic_embs
            )

            # Dynamic fusion: concat base features with LLM semantic features
            fused_features = torch.cat([
                base_concat,
                fused_user_semantic,   # (batch, llm_proj_dim)
                target_item_semantic   # (batch, llm_proj_dim)
            ], dim=-1)  # (batch, base_concat_dim + 2*llm_proj_dim)
        else:
            # Without LLM features, use base concat directly
            fused_features = base_concat

        # ===================================================================
        # Part 3: MMoE Multi-Task Prediction
        # ===================================================================

        # Pass through MMoE to get task predictions
        task_outputs = self.mmoe_forward(fused_features)

        return task_outputs

    def predict(
        self,
        user_id: torch.Tensor,
        hist_item_ids: torch.Tensor,
        target_item_id: torch.Tensor,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        user_state_embs: Optional[torch.Tensor] = None,
        item_semantic_embs: Optional[torch.Tensor] = None,
        enable_llm_features: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for inference. Returns CTR and CVR predictions.

        Args:
            Same as forward()

        Returns:
            ctr_pred: CTR prediction tensor, shape (batch, 1)
            cvr_pred: CVR prediction tensor, shape (batch, 1)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                user_id=user_id,
                hist_item_ids=hist_item_ids,
                target_item_id=target_item_id,
                user_features=user_features,
                item_features=item_features,
                user_state_embs=user_state_embs,
                item_semantic_embs=item_semantic_embs,
                enable_llm_features=enable_llm_features
            )
            return outputs['ctr'], outputs['cvr']


class RankingLoss(nn.Module):
    """
    Combined loss function for multi-task ranking.

    Computes weighted combination of CTR and CVR losses.
    """

    def __init__(self, ctr_weight: float = 1.0, cvr_weight: float = 1.0):
        """
        Args:
            ctr_weight: Weight for CTR loss
            cvr_weight: Weight for CVR loss
        """
        super(RankingLoss, self).__init__()
        self.ctr_weight = ctr_weight
        self.cvr_weight = cvr_weight
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        ctr_pred: torch.Tensor,
        cvr_pred: torch.Tensor,
        ctr_label: torch.Tensor,
        cvr_label: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            ctr_pred: CTR prediction, shape (batch, 1)
            cvr_pred: CVR prediction, shape (batch, 1)
            ctr_label: CTR ground truth (click or not), shape (batch, 1)
            cvr_label: CVR ground truth (conversion or not), shape (batch, 1)

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary containing individual loss values
        """
        ctr_loss = self.bce_loss(ctr_pred, ctr_label)
        cvr_loss = self.bce_loss(cvr_pred, cvr_label)

        total_loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss

        loss_dict = {
            'total': total_loss.item(),
            'ctr': ctr_loss.item(),
            'cvr': cvr_loss.item()
        }

        return total_loss, loss_dict


# ============================================================================
# Utility Functions
# ============================================================================

def create_ranking_model(
    num_users: int,
    num_items: int,
    user_feature_dim: int,
    item_feature_dim: int,
    config: Optional[Dict] = None
) -> StateEnhancedRankingModel:
    """
    Factory function to create a ranking model with default or custom config.

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        user_feature_dim: Dimension of user features
        item_feature_dim: Dimension of item features
        config: Optional configuration dictionary to override defaults

    Returns:
        Initialized StateEnhancedRankingModel
    """
    default_config = {
        'embedding_dim': 64,
        'history_len': 50,
        'llm_semantic_dim': 2560,
        'llm_proj_dim': 64,
        'num_experts': 4,
        'expert_hidden_dim': 128,
        'enable_llm_features': ENABLE_LLM_FEATURES
    }

    if config is not None:
        default_config.update(config)

    model = StateEnhancedRankingModel(
        num_users=num_users,
        num_items=num_items,
        user_feature_dim=user_feature_dim,
        item_feature_dim=item_feature_dim,
        **default_config
    )

    return model


def get_model_size(model: StateEnhancedRankingModel) -> Dict[str, int]:
    """
    Calculate model size statistics.

    Args:
        model: The ranking model

    Returns:
        Dictionary with total parameters and trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the StateEnhancedRankingModel.
    """
    # Model configuration
    num_users = 10000
    num_items = 50000
    user_feature_dim = 32
    item_feature_dim = 32
    batch_size = 16
    history_len = 50

    # Create model
    model = create_ranking_model(
        num_users=num_users,
        num_items=num_items,
        user_feature_dim=user_feature_dim,
        item_feature_dim=item_feature_dim
    )

    print("=" * 60)
    print("StateEnhancedRankingModel - Example Usage")
    print("=" * 60)

    # Print model size
    size_info = get_model_size(model)
    print(f"\nModel Size:")
    print(f"  Total parameters: {size_info['total_params']:,}")
    print(f"  Trainable parameters: {size_info['trainable_params']:,}")

    # Create dummy input
    user_id = torch.randint(0, num_users, (batch_size,))
    hist_item_ids = torch.randint(0, num_items, (batch_size, history_len))
    target_item_id = torch.randint(0, num_items, (batch_size,))
    user_features = torch.randn(batch_size, user_feature_dim)
    item_features = torch.randn(batch_size, item_feature_dim)
    user_state_embs = torch.randn(batch_size, 5, 2560)  # LLM embeddings
    item_semantic_embs = torch.randn(batch_size, 2560)  # LLM embeddings

    # Forward pass
    print(f"\nInput shapes:")
    print(f"  user_id: {user_id.shape}")
    print(f"  hist_item_ids: {hist_item_ids.shape}")
    print(f"  target_item_id: {target_item_id.shape}")
    print(f"  user_features: {user_features.shape}")
    print(f"  item_features: {item_features.shape}")
    print(f"  user_state_embs: {user_state_embs.shape}")
    print(f"  item_semantic_embs: {item_semantic_embs.shape}")

    # With LLM features
    outputs = model(
        user_id=user_id,
        hist_item_ids=hist_item_ids,
        target_item_id=target_item_id,
        user_features=user_features,
        item_features=item_features,
        user_state_embs=user_state_embs,
        item_semantic_embs=item_semantic_embs,
        enable_llm_features=True
    )

    print(f"\nOutput shapes (with LLM features):")
    print(f"  CTR prediction: {outputs['ctr'].shape}")
    print(f"  CVR prediction: {outputs['cvr'].shape}")
    print(f"  CTR sample values: {outputs['ctr'][:3].squeeze().tolist()}")

    # Without LLM features
    outputs_no_llm = model(
        user_id=user_id,
        hist_item_ids=hist_item_ids,
        target_item_id=target_item_id,
        user_features=user_features,
        item_features=item_features,
        enable_llm_features=False
    )

    print(f"\nOutput shapes (without LLM features):")
    print(f"  CTR prediction: {outputs_no_llm['ctr'].shape}")
    print(f"  CVR prediction: {outputs_no_llm['cvr'].shape}")

    # Loss computation
    ctr_label = torch.randint(0, 2, (batch_size, 1)).float()
    cvr_label = torch.randint(0, 2, (batch_size, 1)).float()

    loss_fn = RankingLoss()
    total_loss, loss_dict = loss_fn(
        outputs['ctr'], outputs['cvr'],
        ctr_label, cvr_label
    )

    print(f"\nLoss values:")
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  CTR loss: {loss_dict['ctr']:.4f}")
    print(f"  CVR loss: {loss_dict['cvr']:.4f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
