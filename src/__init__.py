"""
MuseRecSys - LLM状态感知的多阶段推荐系统推理管线

模块结构:
- data_loader.py: 数据层，提供Mock数据和特征接口
- recall.py: 召回层，实现4路并行召回通道
- ranking.py: 精排层，实现State-Enhanced DIN + MMoE模型
- rerank.py: 重排层，实现启发式规则和DPP多样性算法
"""

__version__ = "1.0.0"
__author__ = "MuseRecSys Team"

# Lazy imports to avoid dependency issues
# DataLoader can be imported directly as it has minimal dependencies
from .data_loader import (
    DataLoader,
    UserFeatures,
    ItemFeatures,
    UserHistory,
    create_dataloader
)

__all__ = [
    "DataLoader",
    "UserFeatures",
    "ItemFeatures",
    "UserHistory",
    "create_dataloader",
]

# Optional imports for other modules (require torch, faiss, etc.)
def _lazy_import():
    """Lazy import for modules with external dependencies."""
    global HybridRecall, StateEnhancedRankingModel, RankingLoss
    global create_ranking_model, get_model_size, ENABLE_LLM_FEATURES, ReRanker

    from .recall import HybridRecall
    from .ranking import (
        StateEnhancedRankingModel,
        RankingLoss,
        create_ranking_model,
        get_model_size,
        ENABLE_LLM_FEATURES
    )
    from .rerank import ReRanker

    __all__.extend([
        "HybridRecall",
        "StateEnhancedRankingModel",
        "RankingLoss",
        "create_ranking_model",
        "get_model_size",
        "ENABLE_LLM_FEATURES",
        "ReRanker",
    ])
