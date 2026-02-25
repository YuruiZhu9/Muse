"""
MuseRecSys Recall Layer Module

This module implements the recall layer of the recommendation system with 4 channels:
1. Two-Tower Model Recall - Simulated traditional two-tower model
2. Item2Item Recall - Item-based Collaborative Filtering using co-occurrence
3. Hot Recall - Popular/trending items
4. LLM Semantic Recall - LLM-powered semantic search with 5-to-3 vector merge logic

Author: MuseRecSys Team
Date: 2025-02-25
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import random

from .config import (
    ENABLE_LLM_FEATURES,
    EMB_DIM as ITEM_SEMANTIC_EMB_DIM,
    RECALL_CHANNEL_CONFIG,
    RECALL_FUSE_SIZE,
    USER_STATE_VECTOR_5,
    USER_STATE_VECTOR_3
)

# Default top_k per channel from config
RECALL_PER_CHANNEL = RECALL_CHANNEL_CONFIG.get("two_tower", {}).get("top_k", 150)
FINAL_CANDIDATE_SIZE = RECALL_FUSE_SIZE
from .data_loader import DataLoader


class HybridRecall:
    """
    Hybrid Recall System with 4 recall channels.

    This class implements a multi-channel recall strategy that combines:
    - Two-Tower Model (simulated)
    - Item2Item Collaborative Filtering
    - Hot/Popular Items
    - LLM Semantic Recall (conditional based on ENABLE_LLM_FEATURES)

    The results from all channels are fused using a round-robin strategy
    to create a diverse final candidate set.
    """

    def __init__(self, data_loader: DataLoader, enable_llm_features: bool = True):
        """
        Initialize the Hybrid Recall system.

        Args:
            data_loader: DataLoader instance for accessing data
            enable_llm_features: Override for ENABLE_LLM_FEATURES config.
                               If None, uses config value.
        """
        self.data_loader = data_loader
        self.enable_llm_features = enable_llm_features if enable_llm_features is not None else ENABLE_LLM_FEATURES

        # Initialize channel-specific components
        self._item_co_occurrence: Optional[Dict[Tuple[int, int], int]] = None
        self._hot_items: Optional[List[int]] = None
        self._llm_faiss_index = None  # Faiss index for semantic search
        self._item_semantic_embs: Optional[np.ndarray] = None

        # Pre-load data for efficiency
        self._initialize_channels()

    def _initialize_channels(self):
        """
        Initialize and pre-load data for all recall channels.
        """
        # Load Item2Item co-occurrence data
        self._item_co_occurrence = self.data_loader.load_item_co_occurrence()

        # Load hot items
        self._hot_items = self.data_loader.load_hot_items()

        # Initialize LLM semantic recall if enabled
        if self.enable_llm_features:
            self._initialize_llm_channel()

    def _initialize_llm_channel(self):
        """
        Initialize the LLM semantic recall channel.

        This sets up a Faiss IndexFlatIP (Inner Product) index for
        fast similarity search on item semantic embeddings.
        """
        try:
            import faiss

            # Load item semantic embeddings (2560-dim vectors)
            self._item_semantic_embs = self.data_loader.load_item_semantic_embs()

            if self._item_semantic_embs is not None and len(self._item_semantic_embs) > 0:
                # Create Faiss index with inner product (IP) distance
                # IndexFlatIP is exact search, no approximation
                embedding_dim = self._item_semantic_embs.shape[1] if len(self._item_semantic_embs.shape) > 1 else ITEM_SEMANTIC_EMB_DIM

                # Normalize embeddings for cosine similarity via inner product
                faiss.normalize_L2(self._item_semantic_embs)

                # Create the index
                self._llm_faiss_index = faiss.IndexFlatIP(embedding_dim)
                self._llm_faiss_index.add(self._item_semantic_embs.astype('float32'))

                print(f"[LLM Recall] Initialized Faiss index with {self._item_semantic_embs.shape[0]} items, "
                      f"dim={embedding_dim}")
            else:
                print("[LLM Recall] Warning: No semantic embeddings found, LLM channel disabled")
                self.enable_llm_features = False

        except ImportError:
            print("[LLM Recall] Warning: Faiss not installed, LLM channel disabled")
            self.enable_llm_features = False
        except Exception as e:
            print(f"[LLM Recall] Error initializing LLM channel: {e}")
            self.enable_llm_features = False

    # ========================================================================
    # Channel 1: Two-Tower Model Recall
    # ========================================================================

    def _two_tower_recall(self, user_id: int, top_k: int = RECALL_PER_CHANNEL) -> List[Tuple[int, float]]:
        """
        Channel 1: Two-Tower Model Recall.

        Simulates a traditional two-tower (user tower + item tower) model.
        In production, this would use trained embeddings for user and items.
        For now, returns random candidates with pseudo-scores.

        Args:
            user_id: The user ID to recall for
            top_k: Number of items to recall

        Returns:
            List of (item_id, score) tuples
        """
        num_items = self.data_loader.num_items

        if num_items == 0:
            return []

        # Simulate recall with random sampling + scoring
        # In production: user_emb = user_tower(user_id)
        #              scores = item_tower_emb @ user_emb

        candidates = random.sample(range(num_items), min(top_k, num_items))

        # Generate pseudo-scores (higher = better)
        results = [(item_id, random.random()) for item_id in candidates]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    # ========================================================================
    # Channel 2: Item2Item Recall (ItemCF)
    # ========================================================================

    def _item2item_recall(self, user_id: int, top_k: int = RECALL_PER_CHANNEL) -> List[Tuple[int, float]]:
        """
        Channel 2: Item2Item Recall using Item-based Collaborative Filtering.

        Uses co-occurrence statistics to find items similar to user's history.

        Algorithm:
        1. Get user's interaction history
        2. For each item in history, find similar items via co-occurrence
        3. Aggregate and score candidates

        Args:
            user_id: The user ID to recall for
            top_k: Number of items to recall

        Returns:
            List of (item_id, score) tuples
        """
        # Get user's interaction history
        user_history = self.data_loader.get_user_history(user_id)

        if not user_history:
            # No history: return random items
            return self._two_tower_recall(user_id, top_k)

        # Score aggregation: item -> total co-occurrence score
        candidate_scores: Dict[int, float] = {}

        for history_item in user_history:
            # Find items that co-occur with this history item
            for (item_i, item_j), co_occ_count in self._item_co_occurrence.items():
                if item_i == history_item or item_j == history_item:
                    similar_item = item_j if item_i == history_item else item_i

                    # Skip items already in user history
                    if similar_item in user_history:
                        continue

                    # Aggregate score (could use TF-IDF weighting in production)
                    candidate_scores[similar_item] = candidate_scores.get(similar_item, 0) + co_occ_count

        # Convert to list and sort
        results = [(item_id, score) for item_id, score in candidate_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k] if results else self._two_tower_recall(user_id, top_k)

    # ========================================================================
    # Channel 3: Hot Recall
    # ========================================================================

    def _hot_recall(self, user_id: int, top_k: int = RECALL_PER_CHANNEL) -> List[Tuple[int, float]]:
        """
        Channel 3: Hot/Popular Items Recall.

        Returns trending items based on global popularity.
        Useful for cold-start users and diversity.

        Args:
            user_id: The user ID (not used, but kept for interface consistency)
            top_k: Number of items to recall

        Returns:
            List of (item_id, score) tuples
        """
        if not self._hot_items:
            return []

        # Score based on position (higher position = higher score)
        results = [(item_id, 1.0 - (i / len(self._hot_items)))
                   for i, item_id in enumerate(self._hot_items[:top_k])]

        return results

    # ========================================================================
    # Channel 4: LLM Semantic Recall (KEY INNOVATION)
    # ========================================================================

    def _llm_semantic_recall(self, user_id: int, top_k: int = RECALL_PER_CHANNEL) -> List[Tuple[int, float]]:
        """
        Channel 4: LLM Semantic Recall - KEY INNOVATION.

        This is the core LLM-powered recall channel that uses:
        1. Faiss IndexFlatIP for fast similarity search
        2. Item semantic embeddings (2560-dim vectors)
        3. 5-to-3 vector merge logic for user state

        5-to-3 Merge Logic:
        -------------------
        The user has 5 state vectors from LLM analysis:
            1. long_term_intent - User's long-term interests
            2. life_stage - User's life stage characteristics
            3. psychological_demand - Current psychological needs
            4. retrieval_suggestions - Suggestions for retrieval
            5. interest_growth_points - Areas for interest exploration

        These are merged into 3 query vectors:
            V_profile  = Mean(long_term_intent, life_stage)
                        -> Represents stable user profile

            V_intent   = Mean(psychological_demand, retrieval_suggestions)
                        -> Represents current intent/context

            V_explore  = interest_growth_points
                        -> Directly used for exploration

        Search Strategy:
        ----------------
        For each of the 3 query vectors, perform Top-K search via Faiss.
        Then aggregate results with deduplication.

        Args:
            user_id: The user ID to recall for
            top_k: Number of items to recall per query vector (total results may vary)

        Returns:
            List of (item_id, score) tuples
        """
        # Check if LLM features are enabled
        if not self.enable_llm_features:
            return []

        # Check if Faiss index is ready
        if self._llm_faiss_index is None:
            return []

        # Get user state (the 5 vectors)
        user_states = self.data_loader.load_user_states()
        if user_id not in user_states:
            return []

        state = user_states[user_id]

        # Verify all 5 vectors exist
        required_vectors = USER_STATE_VECTOR_5
        if not all(v in state for v in required_vectors):
            print(f"[LLM Recall] Warning: User {user_id} missing required state vectors")
            return []

        # ====================================================================
        # Step 1: 5-to-3 Vector Merge
        # ====================================================================

        # V_profile = Mean(long_term_intent, life_stage)
        v_profile = self._mean_vectors([
            state["long_term_intent"],
            state["life_stage"]
        ])

        # V_intent = Mean(psychological_demand, retrieval_suggestions)
        v_intent = self._mean_vectors([
            state["psychological_demand"],
            state["retrieval_suggestions"]
        ])

        # V_explore = interest_growth_points (direct use, no merge)
        v_explore = state["interest_growth_points"]

        # Store the 3 merged vectors
        query_vectors = {
            "V_profile": v_profile,
            "V_intent": v_intent,
            "V_explore": v_explore
        }

        # ====================================================================
        # Step 2: Faiss Search for Each Query Vector
        # ====================================================================

        all_results: Dict[int, float] = {}

        # Search for each of the 3 query vectors
        for vector_name, query_vector in query_vectors.items():
            # Prepare query for Faiss (reshape and normalize)
            query = query_vector.reshape(1, -1).astype('float32')

            # Normalize for cosine similarity via inner product
            import faiss
            faiss.normalize_L2(query)

            # Search top_k items
            k = min(top_k, self._llm_faiss_index.ntotal)
            scores, indices = self._llm_faiss_index.search(query, k)

            # Aggregate results
            for idx, (item_id, score) in enumerate(zip(indices[0], scores[0])):
                if item_id >= 0:  # Valid item
                    # Combine scores from different query vectors
                    # Use max to keep the best match
                    if item_id not in all_results or score > all_results[item_id]:
                        all_results[item_id] = score

        # ====================================================================
        # Step 3: Convert to Result List
        # ====================================================================

        results = [(item_id, score) for item_id, score in all_results.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k * 3]  # Return up to 3x top_k since we searched 3 vectors

    @staticmethod
    def _mean_vectors(vectors: List[np.ndarray]) -> np.ndarray:
        """
        Compute the mean of multiple vectors.

        Args:
            vectors: List of numpy arrays to average

        Returns:
            Mean vector as numpy array
        """
        return np.mean(vectors, axis=0)

    # ========================================================================
    # Main Recall Method
    # ========================================================================

    def recall(self, user_id: int, top_k: int = RECALL_PER_CHANNEL) -> Dict[str, List[Tuple[int, float]]]:
        """
        Execute recall from all enabled channels.

        Args:
            user_id: The user ID to recall for
            top_k: Number of items to recall per channel

        Returns:
            Dictionary with channel names as keys and lists of (item_id, score) as values:
            {
                "two_tower": [(item_id, score), ...],
                "item2item": [(item_id, score), ...],
                "hot": [(item_id, score), ...],
                "llm_semantic": [(item_id, score), ...]  # Only if enabled
            }
        """
        results = {}

        # Channel 1: Two-Tower
        results["two_tower"] = self._two_tower_recall(user_id, top_k)

        # Channel 2: Item2Item
        results["item2item"] = self._item2item_recall(user_id, top_k)

        # Channel 3: Hot
        results["hot"] = self._hot_recall(user_id, top_k)

        # Channel 4: LLM Semantic (conditional)
        if self.enable_llm_features:
            llm_results = self._llm_semantic_recall(user_id, top_k)
            if llm_results:
                results["llm_semantic"] = llm_results

        return results

    # ========================================================================
    # Recall Fusion
    # ========================================================================

    def fuse_recall_results(
        self,
        recall_dict: Dict[str, List[Tuple[int, float]]],
        final_size: int = FINAL_CANDIDATE_SIZE
    ) -> List[int]:
        """
        Fuse recall results from multiple channels using round-robin selection.

        Round-Robin Strategy:
        ----------------------
        Given N channels each with M candidates:
        Select items in round-robin order:
            ch1[0], ch2[0], ch3[0], ch4[0], ch1[1], ch2[1], ch3[1], ch4[1], ...

        This ensures:
        1. Diversity - Items from all channels are represented
        2. Balance - No single channel dominates
        3. Quality - Top items from each channel are prioritized

        Deduplication:
        --------------
        If an item appears in multiple channels, only the first occurrence is kept.
        Subsequent occurrences are skipped.

        Args:
            recall_dict: Dictionary of channel results from recall() method
            final_size: Target number of final candidates

        Returns:
            List of item IDs (duplicates removed)
        """
        # Get list of channel names
        channel_names = list(recall_dict.keys())
        num_channels = len(channel_names)

        if num_channels == 0:
            return []

        # Convert channel results to item lists (for round-robin)
        channel_items = [recall_dict[ch] for ch in channel_names]

        # Track seen items for deduplication
        seen = set()
        final_candidates = []

        # Round-robin selection
        max_rounds = max(len(items) for items in channel_items) if channel_items else 0

        for round_idx in range(max_rounds):
            for ch_idx in range(num_channels):
                # Stop if we've reached target size
                if len(final_candidates) >= final_size:
                    break

                # Get next item from this channel
                if round_idx < len(channel_items[ch_idx]):
                    item_id, _ = channel_items[ch_idx][round_idx]

                    # Add if not already seen
                    if item_id not in seen:
                        seen.add(item_id)
                        final_candidates.append(item_id)

            # Early exit if target reached
            if len(final_candidates) >= final_size:
                break

        return final_candidates


# ============================================================================
# Convenience Functions
# ============================================================================

def create_recall_system(data_path: str = "data/", enable_llm_features: bool = None) -> HybridRecall:
    """
    Convenience function to create a HybridRecall instance.

    Args:
        data_path: Path to data directory
        enable_llm_features: Override LLM feature flag

    Returns:
        Initialized HybridRecall instance
    """
    data_loader = DataLoader(data_path)
    return HybridRecall(data_loader, enable_llm_features)
