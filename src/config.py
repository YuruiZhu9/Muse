"""
MuseRecSys 全局配置文件

核心配置:
- ENABLE_LLM_FEATURES: 控制是否启用LLM语义特征
  当为False时，系统回退到传统基于ID的推荐模式
  用于A/B测试对比LLM特征的效果
"""

# ============================================================================
# 全局实验开关 - 控制LLM模块的启用/禁用
# ============================================================================
ENABLE_LLM_FEATURES = True  # 设置为False可回退到传统推荐模式


# ============================================================================
# 数据配置
# ============================================================================
NUM_USERS = 1000          # Mock用户数量
NUM_ITEMS = 5000          # Mock物品数量
EMB_DIM = 2560            # LLM语义向量维度
MAX_HISTORY_LEN = 20      # 最大历史行为序列长度

# 类别配置（基于KuaiRec数据集）
NUM_CATEGORIES = 50       # 物品类别数量
NUM_TAGS = 100            # 标签数量


# ============================================================================
# 召回配置
# ============================================================================
RECALL_CHANNEL_CONFIG = {
    "two_tower": {
        "enabled": True,
        "top_k": 150,
        "weight": 1.0
    },
    "item2item": {
        "enabled": True,
        "top_k": 150,
        "weight": 1.0
    },
    "hot": {
        "enabled": True,
        "top_k": 150,
        "weight": 0.8
    },
    "llm_semantic": {
        "enabled": True,  # 实际启用时受ENABLE_LLM_FEATURES控制
        "top_k": 150,
        "weight": 1.2,
        "faiss_index_type": "IndexFlatIP"  # 内积索引
    }
}

RECALL_FUSE_SIZE = 100    # 召回融合后的最终候选数量


# ============================================================================
# 精排配置
# ============================================================================
RANKING_CONFIG = {
    # Embedding维度
    "user_id_dim": 64,
    "item_id_dim": 64,
    "category_id_dim": 32,

    # DIN相关
    "attention_hidden_dim": 128,

    # LLM语义降维
    "llm_proj_dim": 128,  # 2560 -> 128 (grid-search optimal: heads=4)
    "semantic_num_heads": 4,
    "semantic_dropout": 0.0,

    # MMoE配置
    "num_experts": 4,
    "expert_hidden_dim": 128,
    "num_tasks": 2,  # CTR, CVR
}

RANKING_TOP_K = 50        # 精排后保留的物品数量


# ============================================================================
# 重排配置
# ============================================================================
RERANK_CONFIG = {
    "strategy": "dpp",  # "heuristic" or "dpp"

    # 启发式规则配置
    "window_size": 3,
    "max_same_category": 3,

    # DPP配置
    "lambda_diversity": 0.5,  # 多样性权重
    "final_size": 20,          # 最终推荐列表长度

    # Lightweight list rules
    "filter_seen_items": True,
    "prefix_diversity_top_n": 5,
    "max_prefix_same_category": 2,
    "max_consecutive_same_category": 1,
    "max_adjacent_semantic_similarity": 0.92,
}


# ============================================================================
# Faiss配置 (LLM语义召回)
# ============================================================================
FAISS_CONFIG = {
    "index_type": "IndexFlatIP",  # 内积索引，适合归一化向量
    "nlist": 100,                 # IVF参数
    "nprobe": 10,                 # IVF搜索时访问的聚类中心数
}


# ============================================================================
# 用户状态向量配置 (5变3逻辑)
# ============================================================================
# 5个原始状态向量名称
USER_STATE_VECTOR_5 = [
    "long_term_intent",        # 长期兴趣意图
    "life_stage",              # 生命周期阶段
    "psychological_demand",    # 心理需求
    "retrieval_suggestions",   # 检索建议
    "interest_growth_points"   # 兴趣增长点
]

# 3个合并后的向量名称及合并逻辑
USER_STATE_VECTOR_3 = [
    "V_profile",   # 用户画像向量 = Mean(long_term_intent, life_stage)
    "V_intent",    # 意图向量 = Mean(psychological_demand, retrieval_suggestions)
    "V_explore"    # 探索向量 = interest_growth_points (直接使用)
]


# ============================================================================
# 设备配置
# ============================================================================
# Lazy import torch to avoid import errors
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = "cpu"  # Fallback if torch is not installed
    print("[Config] Warning: PyTorch not installed, using CPU fallback")


# ============================================================================
# 辅助函数：动态切换LLM模式
# ============================================================================
def set_llm_mode(enabled: bool):
    """
    动态切换LLM特征模式

    Args:
        enabled: True启用LLM特征，False回退到传统模式
    """
    global ENABLE_LLM_FEATURES
    ENABLE_LLM_FEATURES = enabled
    print(f"[Config] LLM Features {'ENABLED' if enabled else 'DISABLED'}")


def get_llm_mode() -> bool:
    """获取当前LLM模式状态"""
    return ENABLE_LLM_FEATURES


if __name__ == "__main__":
    print("=" * 50)
    print("MuseRecSys Configuration")
    print("=" * 50)
    print(f"LLM Features: {'ENABLED' if ENABLE_LLM_FEATURES else 'DISABLED'}")
    print(f"Device: {DEVICE}")
    print(f"Num Users: {NUM_USERS}")
    print(f"Num Items: {NUM_ITEMS}")
    print(f"Embedding Dim: {EMB_DIM}")
    print("=" * 50)
