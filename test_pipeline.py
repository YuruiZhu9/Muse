"""
MuseRecSys 测试脚本

验证推荐系统推理管线的各个模块是否正常工作

Author: MuseRecSys Team
Date: 2025-02-25
"""

import sys
import numpy as np
import torch

print("=" * 60)
print("MuseRecSys - Module Testing")
print("=" * 60)

# Test 1: Import modules
print("\n[Test 1] Importing modules...")
try:
    from src.config import ENABLE_LLM_FEATURES, DEVICE, NUM_USERS, NUM_ITEMS
    print(f"  Config: ENABLE_LLM_FEATURES={ENABLE_LLM_FEATURES}")
    print(f"  Device: {DEVICE}")
except Exception as e:
    print(f"  [FAILED] {e}")
    sys.exit(1)

# Test 2: Data Loader
print("\n[Test 2] Testing Data Loader...")
try:
    from src.data_loader import DataLoader

    loader = DataLoader(num_users=100, num_items=500, emb_dim=2560)
    print(f"  Created: {loader}")

    # Test user features
    user_feat = loader.get_user_features(0)
    print(f"  User 0: age={user_feat.age}, gender={user_feat.gender}")

    # Test item features
    item_feat = loader.get_item_features(0)
    print(f"  Item 0: category_id={item_feat.category_id}")

    # Test user state embeddings
    user_state = loader.get_user_state_embs(0)
    print(f"  User 0 state shape: {user_state.shape}")

    # Test item semantic embeddings
    item_sem = loader.get_item_semantic_embs(0)
    print(f"  Item 0 semantic shape: {item_sem.shape}")

    print("  [PASSED]")
except Exception as e:
    print(f"  [FAILED] {e}")
    import traceback
    traceback.print_exc()

# Test 3: Recall Layer
print("\n[Test 3] Testing Recall Layer...")
try:
    from src.recall import HybridRecall

    recall = HybridRecall(data_loader=loader, enable_llm_features=False)
    results = recall.recall(0, top_k=50)

    print(f"  Channels: {list(results.keys())}")
    for channel, items in results.items():
        print(f"    {channel}: {len(items)} items")

    # Test fusion
    fused = recall.fuse_recall_results(results, final_size=50)
    print(f"  Fused candidates: {len(fused)} items")

    print("  [PASSED]")
except Exception as e:
    print(f"  [FAILED] {e}")
    import traceback
    traceback.print_exc()

# Test 4: Ranking Layer
print("\n[Test 4] Testing Ranking Layer...")
try:
    from src.ranking import StateEnhancedRankingModel

    model = StateEnhancedRankingModel(
        num_users=100,
        num_items=501,
        user_feature_dim=32,
        item_feature_dim=32,
        enable_llm_features=False
    )

    # Test forward pass
    batch_size = 16
    user_id = torch.tensor([0] * batch_size)
    hist_item_ids = torch.randint(0, 501, (batch_size, 20))
    target_item_id = torch.randint(1, 501, (batch_size,))
    user_features = torch.randn(batch_size, 32)
    item_features = torch.randn(batch_size, 32)

    outputs = model(
        user_id=user_id,
        hist_item_ids=hist_item_ids,
        target_item_id=target_item_id,
        user_features=user_features,
        item_features=item_features,
        enable_llm_features=False
    )

    print(f"  CTR output shape: {outputs['ctr'].shape}")
    print(f"  CVR output shape: {outputs['cvr'].shape}")
    print("  [PASSED]")
except Exception as e:
    print(f"  [FAILED] {e}")
    import traceback
    traceback.print_exc()

# Test 5: Re-ranking Layer
print("\n[Test 5] Testing Re-ranking Layer...")
try:
    from src.rerank import ReRanker

    reranker = ReRanker(data_loader=loader, strategy='heuristic')

    # Mock ranked items
    ranked_items = [
        {'item_id': i, 'category_id': i % 10, 'score': 1.0 - i * 0.01}
        for i in range(50)
    ]

    # Test heuristic rerank
    reranked = reranker.heuristic_rerank(ranked_items, window_size=3, max_same_category=3)
    print(f"  Heuristic rerank: {len(reranked)} items")

    # Test DPP rerank
    scores = np.array([item['score'] for item in ranked_items])
    semantic_embs = np.random.randn(len(ranked_items), 2560)

    reranked_dpp = reranker.dpp_rerank(
        ranked_items[:20],
        scores[:20],
        semantic_embs[:20],
        final_size=10
    )
    print(f"  DPP rerank: {len(reranked_dpp)} items")

    print("  [PASSED]")
except Exception as e:
    print(f"  [FAILED] {e}")
    import traceback
    traceback.print_exc()

# Test 6: Full Pipeline (without LLM)
print("\n[Test 6] Testing Full Pipeline (without LLM)...")
try:
    from main import MuseRecSysPipeline

    pipeline = MuseRecSysPipeline(
        num_users=100,
        num_items=500,
        emb_dim=2560,
        enable_llm_features=False
    )

    results = pipeline.run_for_user(0, enable_llm_features=False)
    final_rec = results.get('final_recommendations', [])

    print(f"  Recall results: {len(results.get('recall_results', {}))} channels")
    print(f"  Ranking results: {len(results.get('ranking_results', []))} items")
    print(f"  Final recommendations: {len(final_rec)} items")
    print(f"  Timing: {results.get('timing', {})}")

    if final_rec:
        print(f"  Top 5 items: {[r['item_id'] for r in final_rec[:5]]}")

    print("  [PASSED]")
except Exception as e:
    print(f"  [FAILED] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
