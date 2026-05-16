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
        attention_hidden_dim (Optional[int]): Hidden dimension of DIN attention MLP.
            Defaults to 4 * embedding_dim when not provided.
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
        attention_hidden_dim: Optional[int] = None,
        num_experts: int = 4,
        expert_hidden_dim: int = 128,
        enable_llm_features: bool = True,
        padding_idx: int = 0,
        semantic_num_heads: int = 4,
        semantic_dropout: float = 0.0,
    ):
        super(StateEnhancedRankingModel, self).__init__()

        # Store configuration
        self.embedding_dim = embedding_dim
        self.history_len = history_len
        self.llm_semantic_dim = llm_semantic_dim
        self.llm_proj_dim = llm_proj_dim
        self.attention_hidden_dim = attention_hidden_dim or (embedding_dim * 4)
        self.num_experts = num_experts
        self.enable_llm_features = enable_llm_features
        self.padding_idx = padding_idx
        self.semantic_num_heads = semantic_num_heads
        self.semantic_dropout = semantic_dropout

        if llm_proj_dim % semantic_num_heads != 0:
            raise ValueError(
                f"llm_proj_dim ({llm_proj_dim}) must be divisible by semantic_num_heads ({semantic_num_heads})"
            )
        self.semantic_head_dim = llm_proj_dim // semantic_num_heads

        # ===================================================================
        # Part 1: Base DIN (Deep Interest Network) Components
        # ===================================================================

        # ID Embeddings: Map sparse IDs to dense vectors
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # Reserve padding_idx for masked history positions.
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=padding_idx)

        # DIN Attention Layer
        # Computes attention weights between target item and historical items.
        # Strict DIN local activation input: [q, k, q-k, q*k]
        self.attention_layer = nn.Sequential(
            nn.Linear(embedding_dim * 4, self.attention_hidden_dim),
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
        # Note: For A/B testing compatibility, we always initialize with LLM projection layers
        # even if enable_llm_features is False. This allows dynamic switching at runtime.

        # Projection layers: Map high-dimensional LLM embeddings to compact space
        # user_state_embs: (batch, 5, 2560) -> (batch, 5, 64)
        self.user_state_projection = nn.Linear(llm_semantic_dim, llm_proj_dim)

        # item_semantic_embs: (batch, 2560) -> (batch, 64)
        self.item_semantic_projection = nn.Linear(llm_semantic_dim, llm_proj_dim)

        # Semantic Cross-Attention:
        # - target item projected semantic acts as the query
        # - user semantic slots act as key/value
        # - learnable slot embeddings encode the identity/order of the 5 semantic fields
        self.semantic_slot_embedding = nn.Embedding(5, llm_proj_dim)
        self.semantic_q_proj = nn.Linear(llm_proj_dim, llm_proj_dim)
        self.semantic_k_proj = nn.Linear(llm_proj_dim, llm_proj_dim)
        self.semantic_v_proj = nn.Linear(llm_proj_dim, llm_proj_dim)
        self.semantic_out_proj = nn.Linear(llm_proj_dim, llm_proj_dim)
        self.semantic_user_norm = nn.LayerNorm(llm_proj_dim)
        self.semantic_item_norm = nn.LayerNorm(llm_proj_dim)
        self.semantic_output_norm = nn.LayerNorm(llm_proj_dim)
        self.semantic_attention_dropout = nn.Dropout(semantic_dropout)

        # Residual gate keeps the target-aware branch stable by interpolating it
        # with a masked mean pooled user semantic baseline.
        self.semantic_residual_gate = nn.Sequential(
            nn.Linear(llm_proj_dim * 3, llm_proj_dim),
            nn.ReLU(),
            nn.Linear(llm_proj_dim, llm_proj_dim),
            nn.Sigmoid()
        )
        self._latest_semantic_attention_weights: Optional[torch.Tensor] = None

        # Calculate final fusion dimension
        # Always use the larger dimension (with LLM features) for A/B testing compatibility
        # Base concat + Fused User Semantic + Target Item Semantic
        self.fusion_dim = base_concat_dim + (2 * llm_proj_dim)

        # Store base concat dim for dynamic padding when LLM features are disabled
        self.base_concat_dim = base_concat_dim
        self.llm_proj_dim = llm_proj_dim

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
        2. Build strict DIN pairwise features [q, k, q-k, q*k]
        3. Pass through attention network to get scores
        4. Apply mask (if provided) and softmax to get weights
        5. Weighted sum of history embeddings
        """
        concat_input = self._build_din_attention_input(target_embed, history_embeds)

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

    def _build_din_attention_input(
        self,
        target_embed: torch.Tensor,
        history_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Build strict DIN local activation inputs: [q, k, q-k, q*k].

        Args:
            target_embed: Target item embedding, shape (batch, embedding_dim)
            history_embeds: Historical item embeddings, shape (batch, seq_len, embedding_dim)

        Returns:
            Tensor of shape (batch, seq_len, 4 * embedding_dim)
        """
        _, seq_len, _ = history_embeds.shape

        # Broadcast the target item against each historical behavior.
        target_expanded = target_embed.unsqueeze(1).expand(-1, seq_len, -1)

        return torch.cat(
            [
                target_expanded,
                history_embeds,
                target_expanded - history_embeds,
                target_expanded * history_embeds
            ],
            dim=-1
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean over valid positions only.

        Args:
            values: Tensor of shape (batch, seq_len, dim)
            mask: Boolean mask of shape (batch, seq_len)

        Returns:
            Tensor of shape (batch, dim)
        """
        mask_float = mask.unsqueeze(-1).to(dtype=values.dtype)
        summed = torch.sum(values * mask_float, dim=1)
        denom = torch.clamp(mask_float.sum(dim=1), min=1.0)
        return summed / denom

    def _semantic_cross_attention(
        self,
        target_item_proj: torch.Tensor,
        user_state_proj: torch.Tensor,
        state_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Target-aware multi-head cross-attention from item semantics to user semantic slots.

        Args:
            target_item_proj: Projected item semantic tensor, shape (batch, llm_proj_dim)
            user_state_proj: Projected user state tensor, shape (batch, 5, llm_proj_dim)
            state_mask: Valid-state mask, shape (batch, 5)

        Returns:
            attn_output: Target-aware semantic summary, shape (batch, llm_proj_dim)
            attn_weights: Attention weights, shape (batch, num_heads, 1, 5)
        """
        batch_size, seq_len, _ = user_state_proj.shape

        q = self.semantic_q_proj(target_item_proj).view(
            batch_size, self.semantic_num_heads, 1, self.semantic_head_dim
        )
        k = self.semantic_k_proj(user_state_proj).view(
            batch_size, seq_len, self.semantic_num_heads, self.semantic_head_dim
        ).transpose(1, 2)
        v = self.semantic_v_proj(user_state_proj).view(
            batch_size, seq_len, self.semantic_num_heads, self.semantic_head_dim
        ).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.semantic_head_dim ** 0.5)
        attn_mask = state_mask.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(~attn_mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights * attn_mask.to(dtype=attn_weights.dtype)
        attn_weights = attn_weights / torch.clamp(attn_weights.sum(dim=-1, keepdim=True), min=1e-9)
        attn_weights = self.semantic_attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v).squeeze(2)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.llm_proj_dim)
        attn_output = self.semantic_out_proj(attn_output)

        return attn_output, attn_weights

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
        batch_size, seq_len, _ = user_state_embs.shape

        # Step 1: Build a semantic-slot mask from the raw embeddings.
        # Zero vectors are treated as missing semantic slots.
        state_mask = user_state_embs.abs().sum(dim=-1) > 0  # (batch, 5)

        # Step 2: Project semantic features into a compact interaction space.
        user_state_proj = self.user_state_projection(user_state_embs)  # (batch, 5, 64)
        target_item_proj = self.item_semantic_projection(item_semantic_embs)  # (batch, 64)

        # Learnable slot embeddings keep the 5 semantic fields distinguishable.
        slot_ids = torch.arange(seq_len, device=user_state_embs.device)
        slot_embeds = self.semantic_slot_embedding(slot_ids).unsqueeze(0).expand(batch_size, -1, -1)

        mask_float = state_mask.unsqueeze(-1).to(dtype=user_state_proj.dtype)
        user_state_proj = (user_state_proj + slot_embeds) * mask_float
        user_state_proj = self.semantic_user_norm(user_state_proj)
        user_state_proj = user_state_proj * mask_float
        target_item_proj = self.semantic_item_norm(target_item_proj)

        # Step 3: Target-aware multi-head cross-attention.
        attn_output, attn_weights = self._semantic_cross_attention(
            target_item_proj=target_item_proj,
            user_state_proj=user_state_proj,
            state_mask=state_mask
        )
        self._latest_semantic_attention_weights = attn_weights.detach()

        # Step 4: Residual gating between attention output and a robust masked mean baseline.
        pooled_user_semantic = self._masked_mean(user_state_proj, state_mask)
        semantic_gate = self.semantic_residual_gate(
            torch.cat([attn_output, pooled_user_semantic, target_item_proj], dim=-1)
        )
        fused_user_semantic = semantic_gate * attn_output + (1.0 - semantic_gate) * pooled_user_semantic
        fused_user_semantic = self.semantic_output_norm(fused_user_semantic)

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

        # Create mask for padded positions.
        hist_mask = hist_item_ids != self.padding_idx  # (batch, history_len)

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
            # Without LLM features, pad with zeros to match model's input dimension
            # This allows A/B testing with the same model instance
            batch_size = base_concat.shape[0]
            device = base_concat.device
            dtype = base_concat.dtype

            # Zero padding for LLM features
            llm_padding = torch.zeros(batch_size, 2 * self.llm_proj_dim, dtype=dtype, device=device)

            fused_features = torch.cat([
                base_concat,
                llm_padding  # (batch, 2*llm_proj_dim) - zero padding
            ], dim=-1)  # (batch, base_concat_dim + 2*llm_proj_dim)

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
    """Combined loss for multi-task ranking. Supports BCE (binary) and MSE (regression)."""

    def __init__(self, ctr_weight: float = 1.0, cvr_weight: float = 1.0, regression: bool = False):
        super(RankingLoss, self).__init__()
        self.ctr_weight = ctr_weight
        self.cvr_weight = cvr_weight
        self.regression = regression
        self.ctr_loss_fn = nn.MSELoss() if regression else nn.BCELoss()
        self.cvr_loss_fn = nn.BCELoss()

    def forward(
        self,
        ctr_pred: torch.Tensor,
        cvr_pred: torch.Tensor,
        ctr_label: torch.Tensor,
        cvr_label: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ctr_loss = self.ctr_loss_fn(ctr_pred, ctr_label)
        cvr_loss = self.cvr_loss_fn(cvr_pred, cvr_label)
        total_loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss
        return total_loss, {'total': total_loss.item(), 'ctr': ctr_loss.item(), 'cvr': cvr_loss.item()}


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
        'attention_hidden_dim': None,
        'num_experts': 4,
        'expert_hidden_dim': 128,
        'enable_llm_features': ENABLE_LLM_FEATURES,
        'padding_idx': 0,
        'semantic_num_heads': 4,
        'semantic_dropout': 0.0,
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
