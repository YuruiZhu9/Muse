"""
Offline evaluation helpers for KuaiRec-based recommendation experiments.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


def _compute_ild(pred_lists: List[List[int]], item_vectors: Dict[int, np.ndarray]) -> float:
    """Intra-List Diversity: 1 - mean pairwise cosine similarity within each list."""
    scores: List[float] = []
    for items in pred_lists:
        vecs = []
        for item_id in items:
            v = item_vectors.get(int(item_id))
            if v is not None:
                vecs.append(v)
        if len(vecs) < 2:
            continue
        stacked = np.stack(vecs)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sim_matrix = np.dot(stacked, stacked.T) / (norms @ norms.T)
        n = sim_matrix.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        scores.append(float(1.0 - np.mean(sim_matrix[triu_idx])))
    return float(np.mean(scores)) if scores else float("nan")


class KuaiRecEvaluator:
    """Compute ranking metrics from user-level predictions and holdout labels."""

    def __init__(self, ground_truth: Dict[int, Sequence[int]], catalog_size: int | None = None):
        self.ground_truth = {
            int(user_id): [int(item_id) for item_id in items]
            for user_id, items in ground_truth.items()
            if items
        }
        self.catalog_size = catalog_size

    def evaluate(
        self,
        predictions: Dict[int, Sequence[int]],
        k: int = 10,
        item_vectors: Dict[int, np.ndarray] | None = None,
    ) -> Dict[str, float]:
        ndcg_scores: List[float] = []
        hit_scores: List[float] = []
        mrr_scores: List[float] = []
        recall_scores: List[float] = []
        covered_items = set()
        all_pred_lists: List[List[int]] = []

        for user_id, true_items in self.ground_truth.items():
            pred_items = [int(item_id) for item_id in predictions.get(user_id, [])[:k]]
            covered_items.update(pred_items)
            all_pred_lists.append(pred_items)

            true_set = set(int(item_id) for item_id in true_items)
            if not true_set:
                continue

            hits = [1.0 if item_id in true_set else 0.0 for item_id in pred_items]
            dcg = sum(hit / np.log2(index + 2) for index, hit in enumerate(hits))
            idcg = sum(1.0 / np.log2(index + 2) for index in range(min(len(true_set), k)))
            ndcg_scores.append(float(dcg / idcg) if idcg > 0 else 0.0)

            hit_scores.append(1.0 if any(hits) else 0.0)

            reciprocal_rank = 0.0
            for index, item_id in enumerate(pred_items):
                if item_id in true_set:
                    reciprocal_rank = 1.0 / float(index + 1)
                    break
            mrr_scores.append(reciprocal_rank)

            hit_count = sum(1 for item_id in pred_items if item_id in true_set)
            recall_scores.append(float(hit_count) / float(len(true_set)))

        user_count = max(len(self.ground_truth), 1)
        coverage_base = self.catalog_size if self.catalog_size else len(covered_items) or 1

        metrics: Dict[str, float] = {
            f"ndcg@{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            f"hit_rate@{k}": float(np.mean(hit_scores)) if hit_scores else 0.0,
            f"mrr@{k}": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            f"recall@{k}": float(np.mean(recall_scores)) if recall_scores else 0.0,
            f"coverage@{k}": float(len(covered_items)) / float(coverage_base),
            "evaluated_users": float(user_count),
        }
        if item_vectors:
            metrics[f"ild@{k}"] = _compute_ild(all_pred_lists, item_vectors)
        return metrics


def summarize_prediction_overlap(
    left: Dict[int, Iterable[int]],
    right: Dict[int, Iterable[int]],
) -> Dict[str, float]:
    """Return simple overlap diagnostics between two user-level recommendation sets."""

    user_ids = sorted(set(left.keys()) | set(right.keys()))
    if not user_ids:
        return {"mean_jaccard": 0.0, "mean_overlap_count": 0.0}

    jaccard_scores: List[float] = []
    overlap_counts: List[float] = []

    for user_id in user_ids:
        left_set = set(int(item_id) for item_id in left.get(user_id, []))
        right_set = set(int(item_id) for item_id in right.get(user_id, []))
        union = left_set | right_set
        intersection = left_set & right_set

        overlap_counts.append(float(len(intersection)))
        jaccard_scores.append(float(len(intersection)) / float(len(union)) if union else 0.0)

    return {
        "mean_jaccard": float(np.mean(jaccard_scores)),
        "mean_overlap_count": float(np.mean(overlap_counts)),
    }
