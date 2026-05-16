import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.config import (
    DEVICE,
    EMB_DIM,
    ENABLE_LLM_FEATURES,
    MAX_HISTORY_LEN,
    RANKING_CONFIG,
    RANKING_TOP_K,
    RECALL_FUSE_SIZE,
    RERANK_CONFIG,
)
from src.data_loader_kuairec import KuaiRecDataLoader
from src.ranking import StateEnhancedRankingModel
from src.recall import HybridRecall
from src.rerank import ReRanker


class KuaiRecSmokePipeline:
    def __init__(self, data_loader: KuaiRecDataLoader, enable_llm_features: Optional[bool] = None):
        self.data_loader = data_loader
        self.enable_llm_features = ENABLE_LLM_FEATURES if enable_llm_features is None else enable_llm_features

        print("=" * 60)
        print("Initializing KuaiRec Smoke Pipeline...")
        print("=" * 60)

        self.recall = HybridRecall(data_loader=self.data_loader, enable_llm_features=self.enable_llm_features)
        self.ranking_model = StateEnhancedRankingModel(
            num_users=self.data_loader.num_users,
            num_items=self.data_loader.num_items + 1,
            user_feature_dim=32,
            item_feature_dim=32,
            embedding_dim=RANKING_CONFIG["user_id_dim"],
            history_len=MAX_HISTORY_LEN,
            llm_semantic_dim=EMB_DIM,
            llm_proj_dim=RANKING_CONFIG["llm_proj_dim"],
            attention_hidden_dim=RANKING_CONFIG["attention_hidden_dim"],
            num_experts=RANKING_CONFIG["num_experts"],
            expert_hidden_dim=RANKING_CONFIG["expert_hidden_dim"],
            enable_llm_features=self.enable_llm_features,
            semantic_num_heads=RANKING_CONFIG.get("semantic_num_heads", 4),
            semantic_dropout=RANKING_CONFIG.get("semantic_dropout", 0.0),
        ).to(DEVICE)
        self.reranker = ReRanker(self.data_loader, strategy=RERANK_CONFIG["strategy"])

    def run_for_user(self, local_user_id: int, enable_llm_features: Optional[bool] = None) -> Dict:
        use_llm = self.enable_llm_features if enable_llm_features is None else enable_llm_features
        timing = {}
        results: Dict = {}

        start = time.time()
        original_recall_flag = self.recall.enable_llm_features
        self.recall.enable_llm_features = use_llm
        try:
            recall_results = self.recall.recall(local_user_id)
        finally:
            self.recall.enable_llm_features = original_recall_flag
        timing["recall"] = time.time() - start
        results["recall_results"] = recall_results

        fused_candidates = self.recall.fuse_recall_results(recall_results, final_size=RECALL_FUSE_SIZE)
        if not fused_candidates:
            results["final_recommendations"] = []
            results["timing"] = timing
            return results

        start = time.time()
        user_features = self.data_loader.get_user_features(local_user_id)
        user_history = self.data_loader.get_user_history(local_user_id, MAX_HISTORY_LEN)
        hist_padded = [0] * MAX_HISTORY_LEN
        if user_history:
            offset_history = [int(item_id) + 1 for item_id in user_history[-MAX_HISTORY_LEN:]]
            hist_padded[-len(offset_history):] = offset_history
        batch_size = len(fused_candidates)

        user_ids = torch.tensor([local_user_id] * batch_size, dtype=torch.long).to(DEVICE)
        hist_item_ids = torch.tensor([hist_padded] * batch_size, dtype=torch.long).to(DEVICE)
        target_item_ids = torch.tensor([int(item_id) + 1 for item_id in fused_candidates], dtype=torch.long).to(DEVICE)
        user_feat_tensor = torch.tensor(
            [user_features.feature_vector.tolist()] * batch_size,
            dtype=torch.float32
        ).to(DEVICE)

        item_feat_tensor = torch.zeros(batch_size, 32, dtype=torch.float32).to(DEVICE)
        for index, item_id in enumerate(fused_candidates):
            item_feat = self.data_loader.get_item_features(item_id)
            item_feat_tensor[index] = torch.tensor(item_feat.feature_vector, dtype=torch.float32)

        user_state_embs = None
        item_semantic_embs = None
        if use_llm:
            user_state = self.data_loader.get_user_state_embs(local_user_id)
            user_state_embs = torch.tensor(user_state, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1).to(DEVICE)
            item_semantic_np = self.data_loader.get_batch_item_semantic_embs(fused_candidates)
            item_semantic_embs = torch.tensor(item_semantic_np, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            rank_outputs = self.ranking_model(
                user_id=user_ids,
                hist_item_ids=hist_item_ids,
                target_item_id=target_item_ids,
                user_features=user_feat_tensor,
                item_features=item_feat_tensor,
                user_state_embs=user_state_embs,
                item_semantic_embs=item_semantic_embs,
                enable_llm_features=use_llm,
            )

        ctr_scores = rank_outputs["ctr"].squeeze(-1).detach().cpu().numpy()
        ranking_results = []
        for item_id, score in zip(fused_candidates, ctr_scores):
            item_feat = self.data_loader.get_item_features(item_id)
            ranking_results.append({
                "item_id": item_id,
                "original_item_id": self.data_loader.original_item_id(item_id),
                "score": float(score),
                "category_id": item_feat.category_id,
            })
        ranking_results.sort(key=lambda row: row["score"], reverse=True)
        ranking_results = ranking_results[:RANKING_TOP_K]
        timing["ranking"] = time.time() - start
        results["ranking_results"] = ranking_results

        start = time.time()
        rerank_items = ranking_results
        if rerank_items:
            rerank_scores = np.array([row["score"] for row in rerank_items], dtype=np.float32)
            rerank_semantic = self.data_loader.get_batch_item_semantic_embs([row["item_id"] for row in rerank_items])
            rerank_kwargs = {
                "final_size": RERANK_CONFIG["final_size"],
                "seen_item_ids": set(user_history),
                "filter_seen_items": RERANK_CONFIG.get("filter_seen_items", True),
                "prefix_diversity_top_n": RERANK_CONFIG.get("prefix_diversity_top_n", 5),
                "max_prefix_same_category": RERANK_CONFIG.get("max_prefix_same_category", 2),
                "max_consecutive_same_category": RERANK_CONFIG.get("max_consecutive_same_category", 1),
                "max_adjacent_semantic_similarity": RERANK_CONFIG.get("max_adjacent_semantic_similarity", 0.92),
            }
            if self.reranker.strategy == "dpp":
                rerank_kwargs["lambda_diversity"] = RERANK_CONFIG["lambda_diversity"]
                rerank_items = self.reranker.rerank(
                    rerank_items,
                    scores=rerank_scores,
                    semantic_embs=rerank_semantic,
                    **rerank_kwargs,
                )
            else:
                rerank_kwargs["window_size"] = RERANK_CONFIG["window_size"]
                rerank_kwargs["max_same_category"] = RERANK_CONFIG["max_same_category"]
                rerank_items = self.reranker.rerank(
                    rerank_items,
                    **rerank_kwargs,
                )

        timing["reranking"] = time.time() - start
        results["final_recommendations"] = rerank_items
        results["timing"] = timing
        results["original_user_id"] = self.data_loader.original_user_id(local_user_id)
        return results


def main():
    inference_file = Path(os.getenv("KUAIREC_USER_INFERENCE_FILE", "LLM_part/user_inferences_big_train.jsonl"))
    interaction_file = Path(os.getenv("KUAIREC_INTERACTION_FILE", "Datasets/KuaiRec 2.0/data/big_matrix.csv"))
    user_limit = int(os.getenv("KUAIREC_USER_LIMIT", "50"))
    eval_users = int(os.getenv("KUAIREC_EVAL_USERS", "5"))
    output_file = Path(os.getenv("KUAIREC_PIPELINE_OUTPUT", "LLM_part/kuairec_smoke_pipeline_results.json"))

    print(f"Using inference file: {inference_file}")
    print(f"Using interaction file: {interaction_file}")

    loader = KuaiRecDataLoader(
        interaction_file=interaction_file,
        user_inference_file=inference_file,
        user_limit=user_limit,
        emb_dim=EMB_DIM,
    )
    print(loader)

    pipeline = KuaiRecSmokePipeline(loader, enable_llm_features=True)

    all_results: List[Dict] = []
    local_user_ids = list(range(min(eval_users, loader.num_users)))

    for local_user_id in local_user_ids:
        print("\n" + "=" * 60)
        print(f"Evaluating local user {local_user_id} / original user {loader.original_user_id(local_user_id)}")
        print("=" * 60)

        llm_results = pipeline.run_for_user(local_user_id, enable_llm_features=True)
        base_results = pipeline.run_for_user(local_user_id, enable_llm_features=False)

        llm_items = [row["original_item_id"] for row in llm_results["final_recommendations"]]
        base_items = [row["original_item_id"] for row in base_results["final_recommendations"]]
        overlap = sorted(set(llm_items) & set(base_items))

        summary = {
            "local_user_id": local_user_id,
            "original_user_id": loader.original_user_id(local_user_id),
            "llm_enabled": {
                "recommendations": llm_items,
                "timing": llm_results["timing"],
            },
            "llm_disabled": {
                "recommendations": base_items,
                "timing": base_results["timing"],
            },
            "comparison": {
                "overlap_count": len(overlap),
                "overlap_items": overlap,
            },
        }
        all_results.append(summary)

        print(f"LLM enabled top-5: {llm_items[:5]}")
        print(f"LLM disabled top-5: {base_items[:5]}")
        print(f"Overlap count: {len(overlap)}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Smoke pipeline results saved to: {output_file}")


if __name__ == "__main__":
    main()
