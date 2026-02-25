# MuseRecSys - LLM状态感知的多阶段推荐系统推理管线

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目概述

MuseRecSys 是一个**模块化、高可扩展的推荐系统推理管线**，核心创新点在于引入 **LLM 编码的用户状态语义向量**，作为召回和精排阶段的增强特征。

### 核心特性

- **LLM 状态感知**：引入离线生成的 LLM 语义向量（2560维）作为召回和精排的增强特征
- **可插拔设计**：通过 `ENABLE_LLM_FEATURES` 全局开关控制是否启用 LLM 模块，支持严格的 A/B 测试
- **多阶段架构**：
  - 数据层 (Data Layer)
  - 召回层 (Recall Layer) - 4路并行召回
  - 精排层 (Ranking Layer) - State-Enhanced DIN + MMoE
  - 重排层 (Re-ranking Layer) - DPP多样性算法
- **完整 Mock 数据**：基于 KuaiRec 数据集结构，支持离线测试
- **智能依赖检测**：自动检测 faiss 是否安装，优雅降级

## 项目结构

```
MuseRecSys/
├── src/
│   ├── __init__.py          # 模块初始化
│   ├── config.py            # 全局配置
│   ├── data_loader.py       # 数据层（Mock数据生成）
│   ├── recall.py            # 召回层（4路召回）
│   ├── ranking.py           # 精排层（DIN+MMoE）
│   └── rerank.py            # 重排层（DPP/启发式）
├── main.py                  # 主控流水线
├── test_basic.py            # 基础功能测试
├── requirements.txt         # 核心依赖
├── requirements-llm.txt     # LLM功能额外依赖
└── README.md                # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
# 安装核心依赖（必需）
pip install -r requirements.txt

# 可选：安装 faiss 以启用 LLM 语义召回功能
pip install -r requirements-llm.txt
```

### 2. 运行测试

```bash
# 基础功能测试（不含 LLM）
python test_basic.py
```

### 3. 运行完整流水线

```bash
# 运行主程序
python main.py
```

## 环境要求

- Python 3.8+
- numpy >= 1.21.0
- torch >= 2.0.0
- faiss-cpu >= 1.7.0 (可选，用于 LLM 语义召回)

## 配置说明

全局配置在 `src/config.py` 中：

```python
# LLM特征开关
ENABLE_LLM_FEATURES = True  # 设为False回退到传统模式

# 数据配置
NUM_USERS = 1000
NUM_ITEMS = 5000
EMB_DIM = 2560

# 召回配置
RECALL_FUSE_SIZE = 100

# 精排配置
RANKING_TOP_K = 50

# 重排配置
RERANK_CONFIG = {
    'strategy': 'dpp',
    'lambda_diversity': 0.5,
    'final_size': 20
}
```

## 运行示例

### 基础推荐（不含 LLM）

```python
from main import MuseRecSysPipeline

# 创建管线（自动检测 faiss 是否可用）
pipeline = MuseRecSysPipeline(
    num_users=100,
    num_items=500,
    emb_dim=2560,
    enable_llm_features=False  # 明确禁用 LLM
)

# 运行推荐
results = pipeline.run_for_user(user_id=0)
final_rec = results['final_recommendations']
print(f"推荐结果: {[r['item_id'] for r in final_rec]}")
```

### A/B 测试（需要 faiss）

```python
# 创建管线并运行 A/B 测试
pipeline = MuseRecSysPipeline(enable_llm_features=True)
ab_results = pipeline.run_ab_test(user_id=42, final_size=20)

# 查看对比结果
print("LLM启用:", ab_results['llm_enabled']['recommendations'])
print("LLM禁用:", ab_results['llm_disabled']['recommendations'])
print("重叠率:", ab_results['comparison']['overlap_pct'])
```

## 故障排除

### faiss 未安装

如果看到以下提示：
```
[Warning] faiss not installed. LLM semantic recall will be disabled.
```

系统会自动禁用 LLM 功能并使用基础推荐模式。要启用 LLM 功能：

```bash
pip install faiss-cpu
```

### torch 不可用

```bash
pip install torch>=2.0.0
```

## 架构设计

### 召回层 4 通道

| 通道 | 描述 | LLM相关 |
|------|------|---------|
| Channel 1 | Two-Tower 双塔模型 | ✗ |
| Channel 2 | Item2Item 协同过滤 | ✗ |
| Channel 3 | Hot 热门榜单 | ✗ |
| Channel 4 | **LLM 语义召回** | ✓ |

### LLM 语义召回 (5变3逻辑)

```
V_profile  = Mean(long_term_intent, life_stage)
V_intent   = Mean(psychological_demand, retrieval_suggestions)
V_explore  = interest_growth_points
```

使用 Faiss IndexFlatIP 进行 Top-N 检索。

## 扩展性

系统采用模块化设计，支持：

- **新增召回通道**：在 `HybridRecall` 类中添加新方法
- **自定义精排模型**：继承 `StateEnhancedRankingModel`
- **替换重排策略**：实现 `ReRanker` 的子类

## 下一步计划

1. 集成真实的 LLM 推理（LLM_part/）
2. 替换为 KuaiRec 真实数据
3. 添加离线评估指标
4. 性能优化（批处理、量化）

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 致谢

- KuaiRec 数据集: https://kuairec.com
- Faiss 向量检索: https://github.com/facebookresearch/faiss
- PyTorch: https://pytorch.org/

---

**MuseRecSys Team** - 2025
