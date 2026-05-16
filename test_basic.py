"""
MuseRecSys 简单测试脚本（不含 LLM 功能）

验证推荐系统推理管线的基础模块是否正常工作
不依赖 faiss，可以直接在 PyCharm 中运行

Author: MuseRecSys Team
Date: 2025-02-25
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("MuseRecSys - 基础功能测试")
print("=" * 60)

# Test 1: Check dependencies
print("\n[Test 1] 检查依赖...")
try:
    import numpy as np
    print(f"  numpy: OK (version {np.__version__})")
except ImportError:
    print("  numpy: FAILED - 请运行: pip install numpy")
    sys.exit(1)

try:
    import torch
    print(f"  torch: OK (version {torch.__version__})")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("  torch: FAILED - 请运行: pip install torch")
    sys.exit(1)

try:
    import faiss
    print(f"  faiss: OK (version {faiss.__version__})")
    HAS_FAISS = True
except ImportError:
    print("  faiss: NOT INSTALLED (LLM语义召回功能将被禁用)")
    print("        安装方法: pip install faiss-cpu")
    HAS_FAISS = False

# Test 2: Import modules
print("\n[Test 2] 导入模块...")
try:
    from src.config import NUM_USERS, NUM_ITEMS, EMB_DIM, DEVICE
    print(f"  config: OK (users={NUM_USERS}, items={NUM_ITEMS}, emb_dim={EMB_DIM}, device={DEVICE})")
except Exception as e:
    print(f"  config: FAILED - {e}")
    sys.exit(1)

try:
    from src.data_loader import DataLoader
    print("  data_loader: OK")
except Exception as e:
    print(f"  data_loader: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from src.recall import HybridRecall
    print("  recall: OK")
except Exception as e:
    print(f"  recall: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from src.ranking import StateEnhancedRankingModel
    print("  ranking: OK")
except Exception as e:
    print(f"  ranking: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from src.rerank import ReRanker
    print("  rerank: OK")
except Exception as e:
    print(f"  rerank: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Data Loader
print("\n[Test 3] 测试数据加载...")
try:
    loader = DataLoader(num_users=100, num_items=500, emb_dim=2560)
    print(f"  DataLoader创建成功: {loader}")

    # Test user features
    user_feat = loader.get_user_features(0)
    print(f"  User 0: age={user_feat.age}, gender={user_feat.gender}")

    # Test item features
    item_feat = loader.get_item_features(0)
    print(f"  Item 0: category_id={item_feat.category_id}")

    # Test user history
    history = loader.get_user_history(0, max_len=10)
    print(f"  User 0 history: {len(history)} items")

    # Test embeddings
    user_state = loader.get_user_state_embs(0)
    print(f"  User 0 state shape: {user_state.shape}")

    item_sem = loader.get_item_semantic_embs(0)
    print(f"  Item 0 semantic shape: {item_sem.shape}")

    print("  [PASSED]")
except Exception as e:
    print(f"  [FAILED] {e}")
    import traceback
    traceback.print_exc()

# Test 4: Recall Layer (without LLM)
print("\n[Test 4] 测试召回层...")
try:
    recall = HybridRecall(data_loader=loader, enable_llm_features=False)
    results = recall.recall(0, top_k=50)

    print(f"  召回通道: {list(results.keys())}")
    for channel, items in results.items():
        print(f"    {channel}: {len(items)} items")
        history_set = set(loader.get_user_history(0))
        assert all(item_id not in history_set for item_id, _ in items), (
            f"channel {channel} should filter items from user history"
        )

    # Test fusion
    fused = recall.fuse_recall_results(results, final_size=50)
    print(f"  融合后候选: {len(fused)} items")
    print("  [PASSED]")
except Exception as e:
    print(f"  [FAILED] {e}")
    import traceback
    traceback.print_exc()

# Test 5: Ranking Layer
print("\n[Test 5] 测试精排层...")
try:
    model = StateEnhancedRankingModel(
        num_users=100,
        num_items=501,
        user_feature_dim=32,
        item_feature_dim=32,
        enable_llm_features=False
    )

    # Verify DIN local activation uses the strict feature order [q, k, q-k, q*k]
    target_embed = torch.tensor([[1.0, 2.0]])
    history_embeds = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])
    din_inputs = model._build_din_attention_input(target_embed, history_embeds)
    expected_din_inputs = torch.tensor([
        [
            [1.0, 2.0, 3.0, 4.0, -2.0, -2.0, 3.0, 8.0],
            [1.0, 2.0, 5.0, 6.0, -4.0, -4.0, 5.0, 12.0]
        ]
    ])
    assert torch.allclose(din_inputs, expected_din_inputs), "DIN attention input is not [q, k, q-k, q*k]"

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

    # Test semantic cross-attention masking: zero semantic slots must receive zero attention mass.
    llm_model = StateEnhancedRankingModel(
        num_users=10,
        num_items=11,
        user_feature_dim=32,
        item_feature_dim=32,
        llm_semantic_dim=8,
        llm_proj_dim=8,
        semantic_num_heads=2,
        enable_llm_features=True
    )
    user_state_embs = torch.randn(1, 5, 8)
    user_state_embs[:, 3:, :] = 0.0
    item_semantic_embs = torch.randn(1, 8)
    fused_user_semantic, target_item_semantic = llm_model.llm_semantic_enhancement(
        user_state_embs=user_state_embs,
        item_semantic_embs=item_semantic_embs
    )
    attn_weights = llm_model._latest_semantic_attention_weights
    assert attn_weights is not None, "semantic attention weights were not recorded"
    assert fused_user_semantic.shape == (1, 8)
    assert target_item_semantic.shape == (1, 8)
    assert torch.allclose(attn_weights[..., 3:], torch.zeros_like(attn_weights[..., 3:]), atol=1e-6), (
        "masked semantic slots should receive zero attention weight"
    )
    print("  [PASSED]")
except Exception as e:
    print(f"  [FAILED] {e}")
    import traceback
    traceback.print_exc()

# Test 6: Re-ranking Layer (heuristic)
print("\n[Test 6] 测试重排层...")
try:
    reranker = ReRanker(data_loader=loader, strategy='heuristic')

    # Mock ranked items
    ranked_items = [
        {'item_id': i, 'category_id': i % 10, 'score': 1.0 - i * 0.01}
        for i in range(50)
    ]

    reranked = reranker.heuristic_rerank(ranked_items, window_size=3, max_same_category=3)
    print(f"  启发式重排: {len(reranked)} items")

    # Test lightweight list rules on top of base reranking
    constrained_items = [
        {'item_id': 0, 'category_id': 0, 'score': 0.99},
        {'item_id': 1, 'category_id': 0, 'score': 0.98},
        {'item_id': 2, 'category_id': 0, 'score': 0.97},
        {'item_id': 3, 'category_id': 1, 'score': 0.96},
        {'item_id': 4, 'category_id': 2, 'score': 0.95},
    ]
    constrained_embs = np.array([
        [1.0, 0.0, 0.0],
        [0.999, 0.001, 0.0],
        [0.998, 0.002, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    constrained_scores = np.array([item['score'] for item in constrained_items], dtype=np.float32)

    constrained_reranked = reranker.rerank(
        constrained_items,
        scores=constrained_scores,
        semantic_embs=constrained_embs,
        final_size=4,
        seen_item_ids={0},
        filter_seen_items=True,
        prefix_diversity_top_n=3,
        max_prefix_same_category=1,
        max_consecutive_same_category=1,
        max_adjacent_semantic_similarity=0.95,
        window_size=3,
        max_same_category=3,
    )

    constrained_ids = [item['item_id'] for item in constrained_reranked]
    constrained_categories = [item['category_id'] for item in constrained_reranked]
    assert 0 not in constrained_ids, "seen items should be removed before reranking"
    assert len(constrained_categories[:3]) == len(set(constrained_categories[:3])), (
        "prefix results should avoid over-homogeneous categories"
    )
    for left, right in zip(constrained_categories, constrained_categories[1:]):
        assert left != right, "consecutive results should not share the same category under the configured rule"
    print("  [PASSED]")
except Exception as e:
    print(f"  [FAILED] {e}")
    import traceback
    traceback.print_exc()

# Test 7: Full Pipeline (without LLM)
print("\n[Test 7] 测试完整流程...")
try:
    from main import MuseRecSysPipeline

    pipeline = MuseRecSysPipeline(
        num_users=100,
        num_items=500,
        emb_dim=2560,
        enable_llm_features=False  # 禁用 LLM 功能
    )

    results = pipeline.run_for_user(0, enable_llm_features=False)
    final_rec = results.get('final_recommendations', [])

    print(f"  召回结果: {len(results.get('recall_results', {}))} 个通道")
    print(f"  精排结果: {len(results.get('ranking_results', []))} 个物品")
    print(f"  最终推荐: {len(final_rec)} 个物品")
    print(f"  耗时: {results.get('timing', {})}")

    if final_rec:
        print(f"  Top 5 推荐: {[r['item_id'] for r in final_rec[:5]]}")

    print("  [PASSED]")
except Exception as e:
    print(f"  [FAILED] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)

# Summary
print("\n[摘要]")
print(f"- 基础功能: 正常")
if HAS_FAISS:
    print(f"- LLM语义召回: 可用 (faiss已安装)")
    print(f"  提示: 运行 'python main.py' 可以进行A/B测试")
else:
    print(f"- LLM语义召回: 不可用 (faiss未安装)")
    print(f"  启用LLM功能: pip install faiss-cpu")
print("\n")
