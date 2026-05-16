"""
MuseRecSys Re-ranking Layer

This module implements diversity-aware re-ranking strategies for recommendation results.
It provides two main approaches:
1. Heuristic-based sliding window re-ranking (category diversity)
2. DPP (Determinantal Point Process) based re-ranking (semantic diversity)

Author: MuseRecSys Team
Date: 2025-02-25
"""

import numpy as np
from typing import Iterable, List, Dict, Tuple, Optional
from collections import defaultdict


class ReRanker:
    """
    Re-ranking module for improving recommendation diversity.

    This class implements two re-ranking strategies:
    - Heuristic: Sliding window based category diversity control
    - DPP: Determinantal Point Process for semantic diversity

    Attributes:
        data_loader: Data loader instance for accessing item metadata
        strategy: Re-ranking strategy ('heuristic', 'dpp', or 'auto')
    """

    def __init__(self, data_loader, strategy: str = 'dpp'):
        """
        Initialize the ReRanker.

        Args:
            data_loader: Data loader instance with access to item features
            strategy: Re-ranking strategy ('heuristic', 'dpp', or 'auto')
        """
        self.data_loader = data_loader
        self.strategy = strategy.lower()

        # Validate strategy
        valid_strategies = ['heuristic', 'dpp', 'auto']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of {valid_strategies}")

    @staticmethod
    def _normalize_embeddings(semantic_embs: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if semantic_embs is None:
            return None
        semantic_embs = np.asarray(semantic_embs, dtype=np.float32)
        if semantic_embs.ndim != 2:
            raise ValueError("semantic_embs must be a 2D array")
        norms = np.linalg.norm(semantic_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return semantic_embs / norms

    def _filter_seen_items(
        self,
        ranked_items: List[Dict],
        scores: Optional[np.ndarray] = None,
        semantic_embs: Optional[np.ndarray] = None,
        seen_item_ids: Optional[Iterable[int]] = None,
    ) -> Tuple[List[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
        if not ranked_items:
            return [], scores, semantic_embs

        seen_set = set(int(item_id) for item_id in seen_item_ids) if seen_item_ids is not None else set()
        keep_indices = [
            index for index, item in enumerate(ranked_items)
            if int(item.get("item_id")) not in seen_set
        ]

        filtered_items = [ranked_items[index] for index in keep_indices]
        filtered_scores = None if scores is None else np.asarray(scores)[keep_indices]
        filtered_embs = None if semantic_embs is None else np.asarray(semantic_embs)[keep_indices]
        return filtered_items, filtered_scores, filtered_embs

    @staticmethod
    def _tail_same_category_count(selected_items: List[Dict], category_id) -> int:
        count = 0
        for item in reversed(selected_items):
            if item.get("category_id") != category_id:
                break
            count += 1
        return count

    def _apply_list_rules(
        self,
        ranked_items: List[Dict],
        semantic_embs: Optional[np.ndarray] = None,
        final_size: int = 50,
        prefix_diversity_top_n: int = 5,
        max_prefix_same_category: int = 2,
        max_consecutive_same_category: int = 1,
        max_adjacent_semantic_similarity: float = 0.92,
    ) -> List[Dict]:
        if not ranked_items:
            return []

        normalized_embs = self._normalize_embeddings(semantic_embs)
        remaining_indices = list(range(len(ranked_items)))
        selected_indices: List[int] = []
        selected_items: List[Dict] = []
        prefix_category_counts: Dict[int, int] = defaultdict(int)

        def respects_constraints(candidate_index: int, relax_prefix: bool, relax_semantic: bool) -> bool:
            candidate = ranked_items[candidate_index]
            category_id = candidate.get("category_id")

            if (
                not relax_prefix
                and len(selected_items) < prefix_diversity_top_n
                and category_id is not None
                and prefix_category_counts[category_id] >= max_prefix_same_category
            ):
                return False

            if (
                category_id is not None
                and self._tail_same_category_count(selected_items, category_id) >= max_consecutive_same_category
            ):
                return False

            if (
                not relax_semantic
                and normalized_embs is not None
                and selected_indices
            ):
                last_selected_index = selected_indices[-1]
                similarity = float(np.dot(normalized_embs[candidate_index], normalized_embs[last_selected_index]))
                if similarity >= max_adjacent_semantic_similarity:
                    return False

            return True

        while remaining_indices and len(selected_items) < final_size:
            chosen_index = None

            for relax_prefix, relax_semantic in (
                (False, False),
                (False, True),
                (True, True),
            ):
                for candidate_index in remaining_indices:
                    if respects_constraints(candidate_index, relax_prefix=relax_prefix, relax_semantic=relax_semantic):
                        chosen_index = candidate_index
                        break
                if chosen_index is not None:
                    break

            if chosen_index is None:
                chosen_index = remaining_indices[0]

            remaining_indices.remove(chosen_index)
            selected_indices.append(chosen_index)
            selected_item = ranked_items[chosen_index]
            selected_items.append(selected_item)

            category_id = selected_item.get("category_id")
            if category_id is not None and len(selected_items) <= prefix_diversity_top_n:
                prefix_category_counts[category_id] += 1

        return selected_items

    def heuristic_rerank(
        self,
        ranked_items: List[Dict],
        window_size: int = 3,
        max_same_category: int = 3
    ) -> List[Dict]:
        """
        Heuristic-based re-ranking using sliding window diversity control.

        This method ensures that no more than `max_same_category` items with the
        same category ID appear in any sliding window of size `window_size`.
        When a constraint violation is detected, the violating item is swapped
        with a later item that satisfies the constraint.

        Algorithm:
        1. Iterate through ranked items maintaining a sliding window
        2. For each position, check category counts in the current window
        3. If max_same_category is exceeded, find a swap candidate from remaining items
        4. Swap the violating item with the candidate
        5. Continue until all positions are processed

        Args:
            ranked_items: List of ranked items, each as a dict with at least 'item_id' and 'category_id'
            window_size: Size of the sliding window (default: 3)
            max_same_category: Maximum allowed items of same category in window (default: 3)

        Returns:
            Re-ranked list of items with improved category diversity

        Example:
            >>> items = [
            ...     {'item_id': 1, 'category_id': 'A'},
            ...     {'item_id': 2, 'category_id': 'A'},
            ...     {'item_id': 3, 'category_id': 'B'},
            ...     {'item_id': 4, 'category_id': 'A'},
            ... ]
            >>> reranker = ReRanker(data_loader)
            >>> reranked = reranker.heuristic_rerank(items, window_size=3, max_same_category=2)
        """
        if not ranked_items:
            return []

        # Create a copy to avoid modifying the original list
        result = ranked_items.copy()
        n = len(result)

        # Create a mapping of item_id to category_id for quick lookup
        item_to_category = {}
        for item in result:
            item_id = item.get('item_id')
            category_id = item.get('category_id')
            if item_id is not None and category_id is not None:
                item_to_category[item_id] = category_id

        # Helper function to get category of an item
        def get_category(item):
            return item.get('category_id')

        # Process each position in the ranked list
        for i in range(n):
            # Define the sliding window ending at position i
            window_start = max(0, i - window_size + 1)
            window_end = i + 1

            # Count categories in the current window
            category_count = defaultdict(int)
            for j in range(window_start, window_end):
                cat = get_category(result[j])
                if cat:
                    category_count[cat] += 1

            # Check if any category exceeds the limit
            current_category = get_category(result[i])
            if current_category and category_count[current_category] > max_same_category:
                # Need to swap this item with a later item of different category
                swap_found = False

                # Look for a swap candidate from position i+1 onwards
                for j in range(i + 1, n):
                    candidate_category = get_category(result[j])

                    # Check if swapping would resolve the violation
                    if candidate_category and candidate_category != current_category:
                        # Simulate the swap and check if window constraint would be satisfied
                        # After swap, the candidate item would be at position i
                        # Check if adding candidate_category would violate the constraint
                        temp_count = category_count.copy()
                        temp_count[current_category] -= 1  # Remove current item
                        temp_count[candidate_category] += 1  # Add candidate item

                        # Check if constraint is satisfied
                        if all(count <= max_same_category for count in temp_count.values()):
                            # Perform the swap
                            result[i], result[j] = result[j], result[i]
                            swap_found = True
                            break

                # If no suitable swap found, try to find any item with different category
                if not swap_found:
                    for j in range(i + 1, n):
                        candidate_category = get_category(result[j])
                        if candidate_category and candidate_category != current_category:
                            # Swap anyway (may not fully satisfy constraint but improves diversity)
                            result[i], result[j] = result[j], result[i]
                            break

        return result

    def dpp_rerank(
        self,
        ranked_items: List[Dict],
        scores: np.ndarray,
        semantic_embs: np.ndarray,
        final_size: int = 50,
        lambda_diversity: float = 0.5
    ) -> List[Dict]:
        """
        DPP-based re-ranking using Fast Greedy MAP Inference.

        Determinantal Point Processes (DPP) provide an elegant way to balance
        relevance (quality) and diversity (dissimilarity). The DPP selects a
        subset of items that maximizes the determinant of a kernel matrix,
        which naturally promotes diverse items.

        The scoring function combines:
        - Diversity score: Based on semantic similarity (kernel matrix)
        - Relevance score: Based on the original ranking scores

        Final Score = lambda_diversity * diversity_score + (1 - lambda_diversity) * relevance_score

        Algorithm (Fast Greedy MAP):
        1. Build kernel matrix from semantic embeddings (cosine similarity)
        2. Scale kernel by relevance scores to create score-scaled kernel
        3. Initialize with the highest scoring item
        4. Iteratively add items that maximize marginal gain
        5. Marginal gain = det(S ∪ {item}) / det(S) where S is current set

        Args:
            ranked_items: List of ranked items, each as a dict with at least 'item_id'
            scores: Relevance scores for each item (shape: [n_items])
            semantic_embs: Semantic embeddings for each item (shape: [n_items, embedding_dim])
            final_size: Target size of re-ranked list (default: 50)
            lambda_diversity: Balance between diversity and relevance (default: 0.5)
                             0.0 = pure relevance, 1.0 = pure diversity

        Returns:
            Re-ranked list of items with balanced relevance and diversity

        Example:
            >>> items = [{'item_id': 1}, {'item_id': 2}, ...]
            >>> scores = np.array([0.9, 0.8, ...])
            >>> embs = np.random.rand(len(items), 2560)  # 2560-dim embeddings
            >>> reranker = ReRanker(data_loader)
            >>> reranked = reranker.dpp_rerank(items, scores, embs, final_size=50)
        """
        if not ranked_items:
            return []

        n = len(ranked_items)
        final_size = min(final_size, n)

        # Ensure scores is a numpy array
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        # Ensure semantic_embs is a 2D numpy array
        if not isinstance(semantic_embs, np.ndarray):
            semantic_embs = np.array(semantic_embs)

        # Normalize embeddings for cosine similarity
        # L2 normalization makes dot product equivalent to cosine similarity
        norms = np.linalg.norm(semantic_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_embs = semantic_embs / norms

        # Compute kernel matrix using cosine similarity
        # K[i,j] = cosine_similarity(emb[i], emb[j])
        kernel_matrix = self.compute_kernel_matrix(normalized_embs)

        # Scale kernel by relevance scores
        # This creates the score-scaled kernel: L(i,j) = s(i) * K(i,j) * s(j)
        # where s(i) is the relevance score
        score_scaled_kernel = np.outer(scores, scores) * kernel_matrix

        # Apply lambda weighting
        # When lambda is high, diversity is more important
        # When lambda is low, relevance is more important
        # We adjust the kernel based on lambda
        if lambda_diversity >= 1.0:
            # Pure diversity: use kernel matrix only
            adjusted_kernel = kernel_matrix
        elif lambda_diversity <= 0.0:
            # Pure relevance: use diagonal matrix (no diversity consideration)
            adjusted_kernel = np.diag(scores ** 2)
        else:
            # Balance between diversity and relevance
            # Interpolate between score-scaled kernel and relevance-only diagonal
            diversity_kernel = kernel_matrix
            relevance_kernel = np.diag(np.ones(n))
            adjusted_kernel = (lambda_diversity * score_scaled_kernel +
                             (1 - lambda_diversity) * np.diag(scores ** 2))

        # Run Fast Greedy MAP inference to select items
        selected_indices = self.fast_greedy_map(
            adjusted_kernel,
            scores,
            final_size
        )

        # Return selected items in order of selection
        return [ranked_items[i] for i in selected_indices]

    def compute_kernel_matrix(self, embs: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix from embeddings using cosine similarity.

        The kernel matrix K is defined as:
        K[i,j] = cosine_similarity(emb[i], emb[j])

        Since embeddings are L2-normalized, cosine similarity simplifies to dot product:
        K[i,j] = emb[i] · emb[j]

        The resulting kernel matrix is:
        - Symmetric: K[i,j] = K[j,i]
        - Positive semi-definite
        - Diagonal elements are 1 (self-similarity)

        Args:
            embs: L2-normalized embedding matrix (shape: [n_items, embedding_dim])

        Returns:
            Kernel matrix of shape [n_items, n_items]

        Note:
            Input embeddings should be L2-normalized for correct cosine similarity.
            Use: embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        """
        # Compute cosine similarity kernel
        # For normalized embeddings, dot product = cosine similarity
        kernel = np.dot(embs, embs.T)

        # Ensure numerical stability (clip to valid range)
        kernel = np.clip(kernel, -1.0, 1.0)

        return kernel

    def fast_greedy_map(
        self,
        kernel_matrix: np.ndarray,
        scores: np.ndarray,
        top_k: int
    ) -> List[int]:
        """
        Fast Greedy MAP (Maximum A Posteriori) inference for DPP.

        This algorithm greedily selects items to maximize the determinant of the
        kernel submatrix, which is equivalent to maximizing the DPP probability.

        The algorithm maintains:
        - Selected set S (indices of chosen items)
        - Kernel submatrix L_S (kernel matrix restricted to S)

        At each iteration, it selects the item that maximizes the marginal gain:
        gain(i) = det(L_S ∪ {i}) / det(L_S)

        Using the matrix determinant lemma, this can be computed efficiently:
        det(L_S ∪ {i}) = det(L_S) * (1 - L_i,S * inv(L_S) * L_S,i)

        Therefore: gain(i) = 1 - L_i,S * inv(L_S) * L_S,i

        Args:
            kernel_matrix: Kernel matrix (shape: [n_items, n_items])
            scores: Relevance scores (shape: [n_items])
            top_k: Number of items to select

        Returns:
            List of selected indices in order of selection
        """
        n = kernel_matrix.shape[0]
        top_k = min(top_k, n)

        if top_k == 0:
            return []

        # Initialize
        selected = []  # Selected indices
        remaining = set(range(n))  # Remaining candidate indices

        # Pre-compute for efficient updates
        # We'll use a greedy approach with efficient determinant updates

        # Select first item: highest scoring item
        # (This handles the pure relevance case and provides a good starting point)
        first_idx = int(np.argmax(scores))
        selected.append(first_idx)
        remaining.remove(first_idx)

        # Build the kernel submatrix incrementally
        # L_S will be the kernel matrix restricted to selected indices
        L_S = np.array([[kernel_matrix[first_idx, first_idx]]])

        # Iteratively select remaining items
        for _ in range(1, top_k):
            if not remaining:
                break

            best_idx = None
            best_gain = -np.inf

            # Compute marginal gain for each remaining item
            for idx in remaining:
                # Use matrix determinant lemma for efficient computation
                # gain(i) = det(L_S ∪ {i}) / det(L_S)
                #        = kernel[i,i] - L_i,S * inv(L_S) * L_S,i

                # Extract the row/column for this item from kernel matrix
                L_i_S = kernel_matrix[np.array(selected), idx]  # Column
                L_S_i = kernel_matrix[idx, np.array(selected)]  # Row

                try:
                    # Compute inv(L_S) * L_S_i
                    # Using solve is more numerically stable than explicit inverse
                    inv_L_S_times_L_S_i = np.linalg.solve(L_S, L_S_i)

                    # Compute the marginal gain
                    # gain = K[i,i] - L_i,S * inv(L_S) * L_S,i
                    marginal_gain = kernel_matrix[idx, idx] - np.dot(L_i_S, inv_L_S_times_L_S_i)

                    # Ensure non-negative (numerical stability)
                    marginal_gain = max(0, marginal_gain)

                    if marginal_gain > best_gain:
                        best_gain = marginal_gain
                        best_idx = idx

                except np.linalg.LinAlgError:
                    # Singular matrix, skip this candidate
                    continue

            if best_idx is not None:
                # Add the best item to selected set
                selected.append(best_idx)
                remaining.remove(best_idx)

                # Update L_S (expand the submatrix)
                # Add new row and column for the selected item
                new_row = kernel_matrix[best_idx, np.array(selected)]
                new_col = kernel_matrix[np.array(selected), best_idx]

                # Expand L_S
                new_L_S = np.zeros((len(selected), len(selected)))
                new_L_S[:-1, :-1] = L_S
                new_L_S[-1, :] = new_row
                new_L_S[:, -1] = new_col

                L_S = new_L_S
            else:
                # No valid item found (shouldn't happen with proper kernel)
                break

        return selected

    def rerank(
        self,
        ranked_items: List[Dict],
        scores: Optional[np.ndarray] = None,
        semantic_embs: Optional[np.ndarray] = None,
        final_size: int = 50,
        seen_item_ids: Optional[Iterable[int]] = None,
        filter_seen_items: bool = True,
        prefix_diversity_top_n: int = 5,
        max_prefix_same_category: int = 2,
        max_consecutive_same_category: int = 1,
        max_adjacent_semantic_similarity: float = 0.92,
        **kwargs
    ) -> List[Dict]:
        """
        Main re-ranking method that routes to the appropriate strategy.

        Args:
            ranked_items: List of ranked items
            scores: Relevance scores (required for DPP)
            semantic_embs: Semantic embeddings (required for DPP)
            **kwargs: Additional arguments passed to the specific re-ranking method

        Returns:
            Re-ranked list of items
        """
        if filter_seen_items:
            ranked_items, scores, semantic_embs = self._filter_seen_items(
                ranked_items=ranked_items,
                scores=scores,
                semantic_embs=semantic_embs,
                seen_item_ids=seen_item_ids,
            )

        if not ranked_items:
            return []

        rerank_candidate_size = min(len(ranked_items), max(final_size * 2, prefix_diversity_top_n * 2, final_size))
        heuristic_kwargs = {
            key: kwargs[key]
            for key in ("window_size", "max_same_category")
            if key in kwargs
        }
        dpp_kwargs = {
            key: kwargs[key]
            for key in ("lambda_diversity",)
            if key in kwargs
        }

        if self.strategy == 'heuristic':
            base_ranked_items = self.heuristic_rerank(ranked_items, **heuristic_kwargs)
        elif self.strategy == 'dpp':
            if scores is None or semantic_embs is None:
                raise ValueError("DPP strategy requires both scores and semantic_embs")
            base_ranked_items = self.dpp_rerank(
                ranked_items,
                scores,
                semantic_embs,
                final_size=rerank_candidate_size,
                **dpp_kwargs
            )
        else:  # 'auto' - choose based on available data
            if scores is not None and semantic_embs is not None:
                base_ranked_items = self.dpp_rerank(
                    ranked_items,
                    scores,
                    semantic_embs,
                    final_size=rerank_candidate_size,
                    **dpp_kwargs
                )
            else:
                base_ranked_items = self.heuristic_rerank(ranked_items, **heuristic_kwargs)

        emb_by_item_id = None
        if semantic_embs is not None:
            emb_by_item_id = {
                int(item["item_id"]): np.asarray(semantic_embs[index], dtype=np.float32)
                for index, item in enumerate(ranked_items)
            }
            base_semantic_embs = np.stack(
                [emb_by_item_id[int(item["item_id"])] for item in base_ranked_items],
                axis=0
            )
        else:
            base_semantic_embs = None

        constrained_items = self._apply_list_rules(
            ranked_items=base_ranked_items,
            semantic_embs=base_semantic_embs,
            final_size=final_size,
            prefix_diversity_top_n=prefix_diversity_top_n,
            max_prefix_same_category=max_prefix_same_category,
            max_consecutive_same_category=max_consecutive_same_category,
            max_adjacent_semantic_similarity=max_adjacent_semantic_similarity,
        )

        return constrained_items[:final_size]


# ============================================================================
# Utility Functions
# ============================================================================

def compute_mmr(
    query_emb: np.ndarray,
    item_embs: np.ndarray,
    lambda_param: float = 0.5,
    selected_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Compute Maximal Marginal Relevance (MMR) scores.

    MMR is another diversity-aware re-ranking approach that balances:
    - Relevance: Similarity to query
    - Diversity: Dissimilarity to already selected items

    MMR(i) = lambda * relevance(i) - (1 - lambda) * max(similarity(i, j) for j in selected)

    Args:
        query_emb: Query embedding (shape: [embedding_dim])
        item_embs: Item embeddings (shape: [n_items, embedding_dim])
        lambda_param: Balance between relevance and diversity (default: 0.5)
        selected_indices: Indices of already selected items (default: None)

    Returns:
        MMR scores for each item (shape: [n_items])
    """
    # Normalize embeddings
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    item_norms = item_embs / (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-10)

    # Compute relevance (similarity to query)
    relevance = np.dot(item_norms, query_norm)

    if selected_indices is None or len(selected_indices) == 0:
        # No items selected yet, return relevance scores
        return relevance

    # Compute diversity (maximum similarity to any selected item)
    selected_embs = item_norms[selected_indices]
    similarities = np.dot(item_norms, selected_embs.T)
    max_similarity = np.max(similarities, axis=1)

    # Compute MMR scores
    mmr_scores = lambda_param * relevance - (1 - lambda_param) * max_similarity

    return mmr_scores


class MultiCriteriaReRanker(ReRanker):
    """
    Extended ReRanker that supports multiple criteria for re-ranking.

    This class extends the base ReRanker to support:
    - Multiple diversity dimensions (category, semantic, temporal)
    - Custom scoring functions
    - Hybrid re-ranking strategies
    """

    def __init__(self, data_loader, strategy: str = 'dpp'):
        super().__init__(data_loader, strategy)

    def multi_criteria_rerank(
        self,
        ranked_items: List[Dict],
        scores: np.ndarray,
        semantic_embs: np.ndarray,
        category_weights: Optional[Dict[str, float]] = None,
        final_size: int = 50,
        lambda_diversity: float = 0.5,
        lambda_category: float = 0.3
    ) -> List[Dict]:
        """
        Multi-criteria re-ranking combining semantic and category diversity.

        Args:
            ranked_items: List of ranked items
            scores: Relevance scores
            semantic_embs: Semantic embeddings
            category_weights: Optional weights for different categories
            final_size: Target size of re-ranked list
            lambda_diversity: Weight for semantic diversity
            lambda_category: Weight for category diversity

        Returns:
            Re-ranked list with multiple diversity considerations
        """
        # Start with DPP re-ranking for semantic diversity
        reranked = self.dpp_rerank(
            ranked_items,
            scores,
            semantic_embs,
            final_size=min(final_size * 2, len(ranked_items)),  # Get more candidates
            lambda_diversity=lambda_diversity
        )

        # Apply heuristic re-ranking for category diversity
        if lambda_category > 0:
            reranked = self.heuristic_rerank(
                reranked[:final_size],
                window_size=5,
                max_same_category=max(1, int(5 * (1 - lambda_category)))
            )

        return reranked[:final_size]
