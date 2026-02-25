"""
MuseRecSys - LLM状态感知的多阶段推荐系统推理管线

主控流水线：串联数据层、召回层、精排层、重排层，并支持A/B测试对比

Author: MuseRecSys Team
Date: 2025-02-25
"""

import time
import sys
from typing import Dict, List, Tuple, Optional

# Check for required dependencies
try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install it with: pip install numpy")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: torch is required. Install it with: pip install torch")
    sys.exit(1)

# Check for faiss (optional)
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("[Warning] faiss not installed. LLM semantic recall will be disabled.")
    print("         Install it with: pip install faiss-cpu")

from src.config import (
    ENABLE_LLM_FEATURES,
    NUM_USERS,
    NUM_ITEMS,
    EMB_DIM,
    MAX_HISTORY_LEN,
    RECALL_CHANNEL_CONFIG,
    RECALL_FUSE_SIZE,
    RANKING_CONFIG,
    RANKING_TOP_K,
    RERANK_CONFIG,
    DEVICE,
    set_llm_mode,
    get_llm_mode
)
from src.data_loader import DataLoader

# Import recall layer
try:
    from src.recall import HybridRecall
except Exception as e:
    print(f"Error importing recall module: {e}")
    sys.exit(1)

# Import ranking layer
try:
    from src.ranking import StateEnhancedRankingModel
except Exception as e:
    print(f"Error importing ranking module: {e}")
    sys.exit(1)

# Import reranking layer
try:
    from src.rerank import ReRanker
except Exception as e:
    print(f"Error importing reranking module: {e}")
    sys.exit(1)


class MuseRecSysPipeline:
    """
    MuseRecSys 推荐系统推理管线

    完整流程：
    1. Data Layer: 加载用户/物品特征及LLM语义向量
    2. Recall Layer: 4路并行召回（含LLM语义通道）
    3. Ranking Layer: State-Enhanced DIN + MMoE精排
    4. Re-ranking Layer: 多样性重排（DPP或启发式）

    支持通过 ENABLE_LLM_FEATURES 开关控制LLM模块的启用/禁用
    """

    def __init__(
        self,
        num_users: int = NUM_USERS,
        num_items: int = NUM_ITEMS,
        emb_dim: int = EMB_DIM,
        enable_llm_features: bool = None
    ):
        """
        初始化推荐系统管线

        Args:
            num_users: 用户数量
            num_items: 物品数量
            emb_dim: LLM语义向量维度
            enable_llm_features: 是否启用LLM特征（None则根据faiss是否可用自动决定）
        """
        # Auto-detect LLM feature availability
        if enable_llm_features is None:
            enable_llm_features = HAS_FAISS and ENABLE_LLM_FEATURES
        elif enable_llm_features and not HAS_FAISS:
            print("[Warning] LLM features requested but faiss is not installed. Disabling LLM features.")
            enable_llm_features = False

        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.enable_llm_features = enable_llm_features

        print("=" * 60)
        print("Initializing MuseRecSys Pipeline...")
        print("=" * 60)

        # Step 1: Initialize Data Layer
        print("\n[1/4] Initializing Data Layer...")
        self.data_loader = DataLoader(
            num_users=num_users,
            num_items=num_items,
            emb_dim=emb_dim
        )
        print(f"  Data loaded: {self.data_loader}")

        # Step 2: Initialize Recall Layer
        print("\n[2/4] Initializing Recall Layer...")
        self.recall = HybridRecall(
            data_loader=self.data_loader,
            enable_llm_features=enable_llm_features
        )
        print(f"  Recall channels: {list(RECALL_CHANNEL_CONFIG.keys())}")
        print(f"  LLM Features: {'ENABLED' if enable_llm_features else 'DISABLED (faiss not installed)'}")

        # Step 3: Initialize Ranking Layer
        print("\n[3/4] Initializing Ranking Layer...")
        self.ranking_model = StateEnhancedRankingModel(
            num_users=num_users,
            num_items=num_items,
            user_feature_dim=32,  # DataLoader feature vector dimension
            item_feature_dim=32,
            embedding_dim=RANKING_CONFIG['user_id_dim'],
            history_len=MAX_HISTORY_LEN,
            llm_semantic_dim=emb_dim,
            llm_proj_dim=RANKING_CONFIG['llm_proj_dim'],
            num_experts=RANKING_CONFIG['num_experts'],
            expert_hidden_dim=RANKING_CONFIG['expert_hidden_dim'],
            enable_llm_features=enable_llm_features
        ).to(DEVICE)
        print(f"  Ranking model: State-Enhanced DIN + MMoE")
        print(f"  Device: {DEVICE}")

        # Step 4: Initialize Re-ranking Layer
        print("\n[4/4] Initializing Re-ranking Layer...")
        self.reranker = ReRanker(
            data_loader=self.data_loader,
            strategy=RERANK_CONFIG['strategy']
        )
        print(f"  Re-ranking strategy: {RERANK_CONFIG['strategy']}")

        print("\n" + "=" * 60)
        print("Pipeline initialization complete!")
        print("=" * 60)

    def run_for_user(
        self,
        user_id: int,
        enable_llm_features: Optional[bool] = None
    ) -> Dict:
        """
        为单个用户运行完整的推荐流程

        Args:
            user_id: 用户ID
            enable_llm_features: 是否启用LLM特征（None则使用初始化时的设置）

        Returns:
            Dict containing:
                - recall_results: 召回结果
                - ranking_results: 精排结果
                - final_recommendations: 最终推荐列表
                - timing: 各阶段耗时
        """
        if enable_llm_features is None:
            enable_llm_features = self.enable_llm_features

        # Disable LLM if faiss is not available
        if enable_llm_features and not HAS_FAISS:
            enable_llm_features = False

        timing = {}
        results = {}

        # ====================================================================
        # Stage 1: Recall
        # ====================================================================
        start_time = time.time()
        recall_results = self.recall.recall(user_id)
        timing['recall'] = time.time() - start_time
        results['recall_results'] = recall_results

        # Fuse recall results
        fused_candidates = self.recall.fuse_recall_results(
            recall_results,
            final_size=RECALL_FUSE_SIZE
        )

        if not fused_candidates:
            print(f"[Warning] No candidates recalled for user {user_id}")
            return results

        # ====================================================================
        # Stage 2: Ranking
        # ====================================================================
        start_time = time.time()

        # Prepare ranking inputs
        user_features = self.data_loader._user_features[user_id]
        user_history = self.data_loader.get_user_history(user_id, MAX_HISTORY_LEN)

        # Pad history to fixed length
        hist_padded = user_history + [0] * (MAX_HISTORY_LEN - len(user_history))

        # Prepare batch inputs for all candidates
        batch_size = len(fused_candidates)

        # Convert to tensors
        user_ids = torch.tensor([user_id] * batch_size, dtype=torch.long).to(DEVICE)
        hist_item_ids = torch.tensor([hist_padded] * batch_size, dtype=torch.long).to(DEVICE)
        target_item_ids = torch.tensor(fused_candidates, dtype=torch.long).to(DEVICE)

        # User features
        user_feat_tensor = torch.tensor(
            [user_features.feature_vector.tolist()] * batch_size,
            dtype=torch.float32
        ).to(DEVICE)

        # Item features
        item_feat_tensor = torch.zeros(batch_size, 32, dtype=torch.float32).to(DEVICE)
        for i, item_id in enumerate(fused_candidates):
            item_feat = self.data_loader._item_features[item_id]
            item_feat_tensor[i] = torch.tensor(
                item_feat.feature_vector,
                dtype=torch.float32
            )

        # LLM features (if enabled)
        user_state_embs = None
        item_semantic_embs = None

        if enable_llm_features:
            # User state embeddings: [batch, 5, 2560]
            user_state = self.data_loader.get_user_state_embs(user_id)
            # Expand to batch dimension: (5, 2560) -> (batch, 5, 2560)
            user_state_embs = torch.tensor(
                user_state,  # Already numpy array
                dtype=torch.float32
            ).unsqueeze(0).expand(batch_size, -1, -1).to(DEVICE)

            # Item semantic embeddings: [batch, 2560]
            item_semantics = []
            for item_id in fused_candidates:
                item_sem = self.data_loader.get_item_semantic_embs(item_id)
                item_semantics.append(item_sem)
            # Convert list to numpy array first, then to tensor (faster)
            item_semantic_np = np.stack(item_semantics)
            item_semantic_embs = torch.tensor(
                item_semantic_np,
                dtype=torch.float32
            ).to(DEVICE)

        # Run ranking model
        with torch.no_grad():
            rank_outputs = self.ranking_model(
                user_id=user_ids,
                hist_item_ids=hist_item_ids,
                target_item_id=target_item_ids,
                user_features=user_feat_tensor,
                item_features=item_feat_tensor,
                user_state_embs=user_state_embs,
                item_semantic_embs=item_semantic_embs,
                enable_llm_features=enable_llm_features
            )

        # Extract CTR scores for ranking
        ctr_scores = rank_outputs['ctr'].squeeze(-1).cpu().numpy()

        # Sort by CTR score
        ranked_indices = np.argsort(-ctr_scores)
        ranked_items = []

        for idx in ranked_indices[:RANKING_TOP_K]:
            item_id = fused_candidates[idx]
            item_features = self.data_loader._item_features[item_id]
            ranked_items.append({
                'item_id': item_id,
                'score': float(ctr_scores[idx]),
                'category_id': item_features.category_id,
                'ctr_score': float(ctr_scores[idx]),
                'cvr_score': float(rank_outputs['cvr'][idx].item())
            })

        timing['ranking'] = time.time() - start_time
        results['ranking_results'] = ranked_items

        # ====================================================================
        # Stage 3: Re-ranking
        # ====================================================================
        start_time = time.time()

        # Prepare data for re-ranking
        item_ids = [item['item_id'] for item in ranked_items]
        scores = np.array([item['score'] for item in ranked_items])

        # Get semantic embeddings for DPP
        semantic_embs = None
        if RERANK_CONFIG['strategy'] == 'dpp':
            semantic_embs = np.array([
                self.data_loader.get_item_semantic_embs(item_id)
                for item_id in item_ids
            ])

        # Run re-ranking
        if RERANK_CONFIG['strategy'] == 'dpp' and semantic_embs is not None:
            reranked_items = self.reranker.dpp_rerank(
                ranked_items=ranked_items,
                scores=scores,
                semantic_embs=semantic_embs,
                final_size=RERANK_CONFIG['final_size'],
                lambda_diversity=RERANK_CONFIG['lambda_diversity']
            )
        else:
            reranked_items = self.reranker.heuristic_rerank(
                ranked_items=ranked_items,
                window_size=RERANK_CONFIG['window_size'],
                max_same_category=RERANK_CONFIG['max_same_category']
            )
            reranked_items = reranked_items[:RERANK_CONFIG['final_size']]

        timing['reranking'] = time.time() - start_time
        results['final_recommendations'] = reranked_items
        results['timing'] = timing

        return results

    def run_ab_test(
        self,
        user_id: int,
        final_size: int = 20
    ) -> Dict:
        """
        A/B测试：对比启用/禁用LLM特征的推荐效果

        Args:
            user_id: 用户ID
            final_size: 最终推荐列表长度

        Returns:
            Dict containing results for both modes
        """
        print("=" * 60)
        print(f"A/B Test for User {user_id}")
        print("=" * 60)

        ab_results = {}

        # ====================================================================
        # Run with LLM Features ENABLED (only if faiss is available)
        # ====================================================================
        if HAS_FAISS:
            print("\n--- Running with LLM Features ENABLED ---")
            set_llm_mode(True)
            self.recall.enable_llm_features = True

            start_time = time.time()
            results_llm = self.run_for_user(user_id, enable_llm_features=True)
            time_llm = time.time() - start_time

            final_llm = results_llm.get('final_recommendations', [])
            item_ids_llm = [item['item_id'] for item in final_llm[:final_size]]

            print(f"Time: {time_llm:.3f}s")
            print(f"Top-{final_size} recommendations: {item_ids_llm}")
            print(f"Timing breakdown: {results_llm.get('timing', {})}")

            ab_results['llm_enabled'] = {
                'recommendations': item_ids_llm,
                'scores': [item['score'] for item in final_llm[:final_size]],
                'timing': results_llm.get('timing', {}),
                'total_time': time_llm
            }
        else:
            print("\n--- Skipping LLM Enabled test (faiss not installed) ---")
            ab_results['llm_enabled'] = None

        # ====================================================================
        # Run with LLM Features DISABLED
        # ====================================================================
        print("\n--- Running with LLM Features DISABLED ---")
        set_llm_mode(False)
        self.recall.enable_llm_features = False

        start_time = time.time()
        results_base = self.run_for_user(user_id, enable_llm_features=False)
        time_base = time.time() - start_time

        final_base = results_base.get('final_recommendations', [])
        item_ids_base = [item['item_id'] for item in final_base[:final_size]]

        print(f"Time: {time_base:.3f}s")
        print(f"Top-{final_size} recommendations: {item_ids_base}")
        print(f"Timing breakdown: {results_base.get('timing', {})}")

        ab_results['llm_disabled'] = {
            'recommendations': item_ids_base,
            'scores': [item['score'] for item in final_base[:final_size]],
            'timing': results_base.get('timing', {}),
            'total_time': time_base
        }

        # ====================================================================
        # Comparison (only if LLM test was run)
        # ====================================================================
        if ab_results['llm_enabled'] is not None:
            print("\n--- Comparison ---")

            # Overlap analysis
            set_llm = set(item_ids_llm)
            set_base = set(item_ids_base)
            overlap = set_llm & set_base
            unique_llm = set_llm - set_base
            unique_base = set_base - set_llm

            print(f"Overlap: {len(overlap)} items ({len(overlap)/final_size*100:.1f}%)")
            print(f"Unique to LLM: {len(unique_llm)} items - {list(unique_llm)[:5]}")
            print(f"Unique to Base: {len(unique_base)} items - {list(unique_base)[:5]}")

            # Timing comparison
            print(f"\nTiming Comparison:")
            print(f"  LLM Enabled:  {time_llm:.3f}s")
            print(f"  LLM Disabled: {time_base:.3f}s")
            print(f"  Difference:   {time_llm - time_base:+.3f}s ({(time_llm/time_base-1)*100:+.1f}%)")

            ab_results['comparison'] = {
                'overlap_count': len(overlap),
                'overlap_pct': len(overlap) / final_size * 100,
                'unique_llm': list(unique_llm),
                'unique_base': list(unique_base),
                'time_diff': time_llm - time_base,
                'time_diff_pct': (time_llm / time_base - 1) * 100
            }

        # Restore original setting
        set_llm_mode(self.enable_llm_features)
        self.recall.enable_llm_features = self.enable_llm_features

        print("\n" + "=" * 60)

        return ab_results


def main():
    """
    主函数：演示MuseRecSys推理管线的完整流程
    """
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  MuseRecSys - LLM状态感知推荐系统推理管线".center(56) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")

    # Check faiss availability
    if not HAS_FAISS:
        print("[Info] faiss is not installed. Running without LLM semantic recall.")
        print("      To enable LLM features, install faiss: pip install faiss-cpu")
        print()

    # Create pipeline (auto-detect LLM feature availability)
    pipeline = MuseRecSysPipeline(
        num_users=100,
        num_items=500,
        emb_dim=2560,
        enable_llm_features=None  # Auto-detect based on faiss availability
    )

    # Run test for a sample user
    sample_user_id = 42
    print(f"\nRunning pipeline for user {sample_user_id}...")

    results = pipeline.run_for_user(sample_user_id)
    final_rec = results.get('final_recommendations', [])

    print(f"\n=== Results for User {sample_user_id} ===")
    print(f"Recall channels: {list(results.get('recall_results', {}).keys())}")
    print(f"Ranked items: {len(results.get('ranking_results', []))}")
    print(f"Final recommendations: {len(final_rec)}")
    print(f"Timing: {results.get('timing', {})}")

    if final_rec:
        print(f"\nTop 10 recommendations:")
        for i, item in enumerate(final_rec[:10]):
            print(f"  {i+1}. Item {item['item_id']} (score: {item['score']:.4f}, category: {item['category_id']})")

    # Run A/B test only if faiss is available
    if HAS_FAISS:
        print("\n" + "=" * 60)
        print("Running A/B Test...")
        ab_results = pipeline.run_ab_test(sample_user_id, final_size=20)

        # Print summary
        print("\n" + "=" * 60)
        print("A/B Test Summary")
        print("=" * 60)
        if ab_results['llm_enabled']:
            print(f"\nLLM Enabled Recommendations:")
            print(f"  {ab_results['llm_enabled']['recommendations']}")
            print(f"\nLLM Disabled Recommendations:")
            print(f"  {ab_results['llm_disabled']['recommendations']}")
            print(f"\nOverlap: {ab_results['comparison']['overlap_count']}/20 items")
            print(f"Time Impact: {ab_results['comparison']['time_diff_pct']:+.1f}%")

    print("\n" + "=" * 60)
    print("Pipeline execution complete!")
    print("=" * 60)
    print("\n")


if __name__ == "__main__":
    main()
