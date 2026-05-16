# KuaiRec 数据集与 LLM_part 联动使用指南

**文档版本:** v1.0
**创建日期:** 2026-02-26
**适用项目:** MuseRecSys

---

## 目录

1. [系统概述](#1-系统概述)
2. [LLM_part 功能分析](#2-llmpart-功能分析)
3. [时间序列划分策略](#3-时间序列划分策略)
4. [完整工作流设计](#4-完整工作流设计)
5. [代码实现方案](#5-代码实现方案)
6. [数据格式规范](#6-数据格式规范)
7. [集成到推荐系统](#7-集成到推荐系统)

---

## 1. 系统概述

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    KuaiRec 数据集与 LLM_part 联动系统                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐│
│  │ KuaiRec 原始 │    │ DeepSeek API│    │ Qwen 微调   │    │ Qwen        ││
│  │   数据      │───▶│  生成训练   │───▶│   模型      │───▶│ Embedding   ││
│  │             │    │    数据     │    │            │    │   编码      ││
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘│
│         │                  │                  │                  │       │
│         ▼                  ▼                  ▼                  ▼       │
│   big_matrix.csv    user_inferences    qwen3_finetuned   user_state.npy │
│   user_features        .jsonl              (LoRA)        item_semantic.npy
│   item_categories                                            │
│                                                                ▼       │
│                    ┌─────────────────────────────────────────────────┐  │
│                    │           MuseRecSys Pipeline                  │  │
│                    │  (2560维向量参与召回/精排/重排)                  │  │
│                    └─────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 三个核心功能

| 功能 | 目的 | 输入 | 输出 | 状态 |
|------|------|------|------|------|
| **功能1** | 生成训练数据 | KuaiRec 原始数据 | `user_inferences.jsonl` | ✅ 已实现 |
| **功能2** | 微调 Qwen 模型 | `user_inferences.jsonl` | `qwen3_finetuned/` | ✅ 已实现 |
| **功能3** | 生成语义向量 | KuaiRec + 微调模型 | `.npy` 向量文件 | 🔄 待完善 |

### 1.3 核心挑战

**时间序列划分问题:**
- KuaiRec 提供 2020-07-05 至 2020-09-05 (约 62 天) 的数据
- 不能简单地把所有数据当作历史序列
- 需要模拟真实推荐场景：**在时刻 t，只能使用 t 之前的历史**

---

## 2. LLM_part 功能分析

### 2.1 功能1: DeepSeek API 生成训练数据

**文件:** `LLMapi_for_generate.py`

**核心类:** `KuaiRecInferenceGenerator`

**系统 Prompt 分析:**

该 Prompt 设计非常完善，要求输出 **5 维用户状态分析**：

```json
{
  "user_basic_information": {
    "user_id": "用户ID",
    "user_active_degree": "活跃度",
    "gender": "性别"
  },
  "user_status_analysis": {
    "long_term_intent": {
      "description": "长期兴趣意图（兴趣星系描述）"
    },
    "psychological_demand": {
      "core_demand": "核心心理角色",
      "immediate_need": "长期稳定 + 近期凸显需求"
    },
    "life_stage_hypothesis": {
      "stage": "生活阶段",
      "confidence": "置信度（高/中/低）",
      "key_attributes": ["关键标签"]
    },
    "interest_growth_points": {
      "emerging_signals": ["潜在领域"],
      "bridge_concepts": ["连接概念"]
    }
  },
  "retrieval_suggestions": {
    "explicit_queries": ["直接搜索词"],
    "implicit_keywords": ["召回标签"]
  }
}
```

**与 MuseRecSys 的映射:**

| LLM 输出 | MuseRecSys 字段 | 用途 |
|----------|-----------------|------|
| `long_term_intent.description` | `long_term_intent` (文本) | 可解释性 |
| `psychological_demand.core_demand` | - | 用户画像分析 |
| `life_stage_hypothesis.stage` | `life_stage` (文本) | 生命周期 |
| `retrieval_suggestions.implicit_keywords` | `retrieval_suggestions` | 召回优化 |

**当前问题:**
- ❌ Mock 数据不符合 KuaiRec 格式
- ❌ 时间戳格式不匹配
- ❌ 缺少时间窗口处理逻辑

### 2.2 功能2: Qwen 模型微调

**文件:** `SFT_Qwen1.7b/finetune_qwen4b.py`

**模型:** Qwen3-1.7B + LoRA

**训练配置:**
```python
MODEL_ID = "Qwen/Qwen3-1.7B"
DATA_PATH = "trandata.jsonl"
OUTPUT_DIR = "./qwen3_finetuned"

# LoRA 配置
r=16, lora_alpha=32, lora_dropout=0.05
target_modules="all-linear"

# 训练参数
per_device_train_batch_size=2
gradient_accumulation_steps=4
learning_rate=2e-4
num_train_epochs=3
```

**数据格式要求:**
```json
{
  "messages": [
    {"role": "user", "content": "用户上下文..."},
    {"role": "assistant", "content": "LLM 分析结果 JSON..."}
  ]
}
```

**当前问题:**
- ❌ 数据格式与功能 1 输出不匹配
- ❌ 缺少数据验证
- ❌ 无早停和验证集

### 2.3 功能3: Embedding 编码

**文件:** `Embedding_Qwen4B/inference_embedding.py`

**模型:** Qwen3-Embedding-4B (4-bit 量化)

**输出维度:** 2560

**代码分析:**
```python
def mean_pooling(model_output, attention_mask):
    # Mean Pooling - 标准的句向量获取方法
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(...)
    return torch.sum(token_embeddings * input_mask_expanded, 1) / ...
```

**当前问题:**
- ❌ 只支持单句输入，无批处理
- ❌ 无缓存机制
- ❌ 输出未持久化

---

## 3. 时间序列划分策略

### 3.1 问题定义

**场景:** 在第 t 天进行推荐，只能使用 t 天之前的历史数据

**挑战:**
- KuaiRec 数据覆盖 62 天（2020-07-05 至 2020-09-05）
- 用户行为随时间变化（兴趣漂移）
- 需要模拟真实推荐系统的"信息约束"

### 3.2 数据划分方案

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    KuaiRec 时间轴 (62 天)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Day 1   Day 7  Day 14  Day 21  Day 28  Day 35  Day 42  Day 49  Day 62 │
│   ▲       ▲       ▲       ▼       ▼       ▼       ▼       ▼       ▼    │
│   │       │       │       │       │       │       │       │       │    │
│   │       │       │       └───────┴───────┴───────┴───────┴───────┘    │
│   │       │       │                      │                          │
│  冷启动   │       │               测试集 (28天)                      │
│   期      │       │               用于评估推荐效果                     │
│          │       │                                                  │
│    训练集窗口                                           │
│    (滑动窗口)                                           │
│                                                         │
│    每个采样点:                                               │
│    - 历史窗口: 7天/14天/21天                                    │
│    - 预测目标: 历史窗口后1天                                    │
│                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 时间窗口设计

#### 方案 A: 滑动窗口采样 (推荐)

```python
class TimeWindowSampler:
    """时间窗口采样器 - 模拟真实推荐场景"""

    def __init__(
        self,
        data_path: str,
        train_window_days: int = 14,    # 历史窗口：14天
        predict_gap_days: int = 1,      # 预测目标：后1天
        slide_stride_days: int = 7,     # 滑动步长：7天
        min_interactions: int = 5       # 最小交互次数
    ):
        """
        参数说明:
        - train_window_days: 作为"已知历史"的天数
        - predict_gap_days: 预测目标与历史窗口的间隔
        - slide_stride_days: 窗口滑动步长
        - min_interactions: 用户最少交互次数（过滤冷启动）
        """
        self.train_window_days = train_window_days
        self.predict_gap_days = predict_gap_days
        self.slide_stride_days = slide_stride_days
        self.min_interactions = min_interactions

    def sample_time_windows(self, user_id: int) -> List[Dict]:
        """
        为单个用户生成时间窗口样本

        返回:
        [
            {
                "sample_id": "u123_20200719",
                "user_id": 123,
                "history_start": "2020-07-05",
                "history_end": "2020-07-18",  # 14天历史
                "predict_date": "2020-07-19",  # 预测日
                "history_interactions": [...],  # 历史交互
                "predict_interactions": [...]   # 目标交互（用于验证）
            },
            ...
        ]
        """
```

#### 方案 B: 固定日期划分

```python
# 固定划分日期（简化版）
DATE_SPLITS = {
    "train_start": "2020-07-05",   # 数据起始日
    "train_end": "2020-08-08",     # 训练集结束（35天）
    "valid_start": "2020-08-09",   # 验证集开始
    "valid_end": "2020-08-22",     # 验证集结束（14天）
    "test_start": "2020-08-23",    # 测试集开始
    "test_end": "2020-09-05"       # 测试集结束（14天）
}

# 问题：这种划分丢失了"时间约束"的真实性
```

### 3.4 用户状态的时间敏感性

**用户状态随时间演变:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    用户状态随时间的演变                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Day 1-7:      Day 8-14:      Day 15-21:      Day 22-28:                │
│  冷启动期      初步形成        兴趣稳定        兴趣漂移                    │
│                                                                          │
│  ┌─────┐      ┌──────┐       ┌──────┐        ┌──────┐                  │
│  │探索 │  →   │聚焦   │   →   │深耕  │    →   │拓展   │                  │
│  │多类 │      │考研  │       │考研  │        │考研+  │                  │
│  │内容 │      │为主  │       │细分  │        │生活  │                  │
│  └─────┘      └──────┘       └──────┘        └──────┘                  │
│                                                                          │
│  LLM 推理结果应该反映这种演变:                                            │
│  - Day 7: "用户兴趣分散，正在探索..."                                    │
│  - Day 14: "用户聚焦考研学习..."                                         │
│  - Day 21: "用户深入考研专业课..."                                       │
│  - Day 28: "用户在备考基础上开始关注生活品质..."                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**关键洞察:**
- **不同时间点的用户状态是不同的**
- **LLM 推理应该基于"截止到该时刻"的历史**
- **不能使用未来信息进行推理** (Data Leakage)

---

## 4. 完整工作流设计

### 4.1 数据准备阶段

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        阶段 1: 数据准备                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  输入: KuaiRec 原始数据                                                  │
│    - big_matrix.csv (12,530,806 条交互)                                 │
│    - user_features.csv (7,176 条用户特征)                               │
│    - kuairec_caption_category.csv (物品元数据)                          │
│                                                                          │
│  处理步骤:                                                                │
│    1. 数据清洗与过滤                                                     │
│    2. 时间序列划分                                                       │
│    3. 用户-物品索引构建                                                  │
│    4. 时间窗口采样                                                       │
│                                                                          │
│  输出:                                                                    │
│    - time_windows.jsonl (时间窗口样本)                                   │
│    - user_index.pkl (用户索引)                                          │
│    - item_index.pkl (物品索引)                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 LLM 训练数据生成阶段

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     阶段 2: LLM 训练数据生成                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  输入: time_windows.jsonl                                                │
│                                                                          │
│  处理流程:                                                                │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  遍历每个时间窗口样本                                           │   │
│  │    ↓                                                            │   │
│  │  构建 Prompt (只使用 history_end 之前的交互)                     │   │
│  │    ↓                                                            │   │
│  │  调用 DeepSeek API                                              │   │
│  │    ↓                                                            │   │
│  │  解析 JSON 响应                                                 │   │
│  │    ↓                                                            │   │
│  │  保存到 trandata.jsonl                                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  输出: trandata.jsonl (格式化后的微调数据)                               │
│                                                                          │
│  数据格式:                                                               │
│  {                                                                      │
│    "messages": [                                                        │
│      {"role": "user", "content": "用户上下文..."},                     │
│      {"role": "assistant", "content": "LLM分析结果JSON..."}             │
│    ],                                                                   │
│    "metadata": {                                                        │
│      "user_id": 123,                                                   │
│      "sample_id": "u123_20200719",                                     │
│      "history_end": "2020-07-18"                                       │
│    }                                                                    │
│  }                                                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 模型微调阶段

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        阶段 3: Qwen 模型微调                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  输入: trandata.jsonl                                                    │
│                                                                          │
│  微调配置:                                                                │
│    - 基础模型: Qwen/Qwen3-1.7B                                          │
│    - 微调方法: LoRA (r=16, alpha=32)                                    │
│    - 训练轮数: 3 epochs                                                 │
│    - 批次大小: 2 (per device) × 4 (gradient accumulation) = 8          │
│    - 学习率: 2e-4                                                       │
│                                                                          │
│  输出: qwen3_finetuned/                                                 │
│    - adapter_config.json                                                │
│    - adapter_model.safetensors                                          │
│    - tokenizer files                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 向量生成阶段

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    阶段 4: 语义向量生成                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  4.1 用户状态向量生成                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  输入: KuaiRec 用户历史 (截止到某时刻) + 微调后的 Qwen          │    │
│  │  流程:                                                           │    │
│  │    1. 构建 Prompt (使用用户历史)                                 │    │
│  │    2. 调用微调后的 Qwen 生成用户状态分析                          │    │
│  │    3. 提取 5 个维度的文本描述                                     │    │
│  │    4. 使用 Qwen3-Embedding-4B 编码为 5×2560 向量                  │    │
│  │  输出: user_state_embs.npy (N_users × 5 × 2560)                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  4.2 物品语义向量生成                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  输入: 物品 caption + Qwen3-Embedding-4B                        │    │
│  │  流程:                                                           │    │
│  │    1. 构建物品描述文本 (caption + 分类 + 标签)                   │    │
│  │    2. 使用 Qwen3-Embedding-4B 编码                               │    │
│  │  输出: item_semantic_embs.npy (N_items × 2560)                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.5 推荐系统集成阶段

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    阶段 5: 推荐系统集成                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  向量文件位置:                                                            │
│    - LLM_part/user_state_embs.npy                                       │
│    - LLM_part/item_semantic_embs.npy                                    │
│                                                                          │
│  集成方式:                                                                │
│    src/data_loader.py 修改:                                             │
│      ```python                                                          │
│      def get_user_state_embs(self, user_id: int) -> np.ndarray:         │
│          # 从预生成的 npy 文件加载                                      │
│          user_idx = self.user_id_to_idx[user_id]                       │
│          return self.user_state_embs[user_idx]  # (5, 2560)            │
│      ```                                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 代码实现方案

### 5.1 时间窗口采样器实现

```python
# LLM_part/time_window_sampler.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

class TimeWindowSampler:
    """
    KuaiRec 时间窗口采样器

    核心功能: 为每个用户生成多个"历史窗口-预测日"样本
    """

    DATE_FORMAT = "%Y-%m-%d"
    DATA_START_DATE = "2020-07-05"
    DATA_END_DATE = "2020-09-05"

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        train_window_days: int = 14,
        predict_gap_days: int = 1,
        slide_stride_days: int = 7,
        min_history_interactions: int = 5,
        min_predict_interactions: int = 1
    ):
        """
        Args:
            interactions_df: KuaiRec 交互数据 (必须包含 date 列)
            train_window_days: 历史窗口天数
            predict_gap_days: 预测目标与历史窗口的间隔天数
            slide_stride_days: 滑动窗口步长
            min_history_interactions: 历史窗口最少交互次数
            min_predict_interactions: 预测日最少交互次数
        """
        self.df = interactions_df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"], format="%Y%m%d")

        self.train_window_days = train_window_days
        self.predict_gap_days = predict_gap_days
        self.slide_stride_days = slide_stride_days
        self.min_history_interactions = min_history_interactions
        self.min_predict_interactions = min_predict_interactions

        # 预处理: 按用户和时间排序
        self.df = self.df.sort_values(["user_id", "timestamp"])

        # 计算时间范围
        self.min_date = self.df["date"].min()
        self.max_date = self.df["date"].max()

    def generate_sample_dates(self) -> List[Dict]:
        """
        生成所有可能的采样日期点

        Returns:
            [{"predict_date": "2020-07-19", "history_start": "2020-07-05"}, ...]
        """
        sample_dates = []
        current_date = self.min_date + timedelta(days=self.train_window_days)

        while current_date + timedelta(days=self.predict_gap_days) <= self.max_date:
            history_start = current_date - timedelta(days=self.train_window_days)
            predict_date = current_date + timedelta(days=self.predict_gap_days)

            sample_dates.append({
                "predict_date": predict_date.strftime(self.DATE_FORMAT),
                "history_start": history_start.strftime(self.DATE_FORMAT),
                "history_end": current_date.strftime(self.DATE_FORMAT)
            })

            current_date += timedelta(days=self.slide_stride_days)

        return sample_dates

    def sample_user_windows(self, user_id: int) -> List[Dict]:
        """
        为单个用户生成时间窗口样本

        这是核心方法，确保:
        1. 只使用 history_end 之前的数据作为输入
        2. predict_date 的数据用于验证
        """
        user_df = self.df[self.df["user_id"] == user_id].copy()

        if len(user_df) < self.min_history_interactions:
            return []

        sample_dates = self.generate_sample_dates()
        windows = []

        for sample_info in sample_dates:
            predict_date = pd.to_datetime(sample_info["predict_date"])
            history_end = pd.to_datetime(sample_info["history_end"])
            history_start = pd.to_datetime(sample_info["history_start"])

            # 历史窗口交互 (只使用 history_end 之前的数据)
            history_mask = (
                (user_df["date"] >= history_start) &
                (user_df["date"] <= history_end)
            )
            history_interactions = user_df[history_mask]

            # 预测日交互 (用于验证)
            predict_mask = user_df["date"] == predict_date
            predict_interactions = user_df[predict_mask]

            # 过滤条件
            if (len(history_interactions) < self.min_history_interactions or
                len(predict_interactions) < self.min_predict_interactions):
                continue

            windows.append({
                "sample_id": f"u{user_id}_{sample_info['predict_date']}",
                "user_id": user_id,
                "history_start": sample_info["history_start"],
                "history_end": sample_info["history_end"],
                "predict_date": sample_info["predict_date"],
                "history_interactions": history_interactions.to_dict("records"),
                "predict_interactions": predict_interactions.to_dict("records"),
                "num_history": len(history_interactions),
                "num_predict": len(predict_interactions)
            })

        return windows

    def sample_all_users(
        self,
        max_users: Optional[int] = None,
        output_path: str = "time_windows.jsonl"
    ) -> List[Dict]:
        """
        为所有用户生成时间窗口样本

        Args:
            max_users: 最大用户数 (用于快速测试)
            output_path: 输出文件路径
        """
        unique_users = self.df["user_id"].unique()

        if max_users:
            unique_users = unique_users[:max_users]

        all_windows = []

        for user_id in unique_users:
            windows = self.sample_user_windows(user_id)
            all_windows.extend(windows)

        # 保存到 jsonl
        with open(output_path, 'w', encoding='utf-8') as f:
            for window in all_windows:
                f.write(json.dumps(window, ensure_ascii=False) + '\n')

        print(f"生成 {len(all_windows)} 个时间窗口样本，保存至 {output_path}")
        return all_windows


# 使用示例
if __name__ == "__main__":
    # 加载 KuaiRec 数据
    interactions = pd.read_csv("Datasets/KuaiRec 2.0/data/big_matrix.csv")

    # 创建采样器
    sampler = TimeWindowSampler(
        interactions_df=interactions,
        train_window_days=14,  # 14天历史
        predict_gap_days=1,    # 预测后1天
        slide_stride_days=7,   # 每7天滑动一次
        min_history_interactions=5
    )

    # 生成样本 (先测试 100 个用户)
    windows = sampler.sample_all_users(max_users=100)
```

### 5.2 LLM 数据生成器改造

```python
# LLM_part/llm_data_generator.py
import json
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
import time

class KuaiRecLLMDataGenerator:
    """
    KuaiRec LLM 训练数据生成器

    核心改进:
    1. 支持时间窗口约束
    2. 输出标准微调格式
    3. 自动重试和错误处理
    """

    def __init__(
        self,
        api_key: str,
        item_metadata_path: str,
        system_prompt_path: str = "prompts/user_analysis_prompt.txt"
    ):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.item_metadata = self._load_item_metadata(item_metadata_path)
        self.system_prompt = self._load_system_prompt(system_prompt_path)

    def _load_item_metadata(self, path: str) -> Dict:
        """加载物品元数据"""
        df = pd.read_csv(path)
        metadata = {}
        for _, row in df.iterrows():
            metadata[row["video_id"]] = {
                "caption": row.get("caption", ""),
                "first_level": row.get("first_level_category_name", ""),
                "second_level": row.get("second_level_category_name", ""),
                "third_level": row.get("third_level_category_name", ""),
                "tags": row.get("topic_tag", [])
            }
        return metadata

    def build_user_context(
        self,
        history_interactions: List[Dict],
        user_features: Dict
    ) -> str:
        """
        构建用户上下文 (只使用历史窗口内的数据)

        关键: 确保不包含任何 predict_date 之后的信息
        """
        context = f"=== 用户特征 ===\n"
        for key, value in user_features.items():
            context += f"{key}: {value}\n"

        context += "\n=== 历史交互记录 (按时间顺序) ===\n"

        for i, interaction in enumerate(history_interactions, 1):
            video_id = interaction["video_id"]
            watch_ratio = interaction.get("watch_ratio", 0)
            timestamp = interaction.get("time", "")

            # 获取物品元数据
            item_meta = self.item_metadata.get(video_id, {})

            context += f"\n交互 {i}:\n"
            context += f"  视频ID: {video_id}\n"
            context += f"  观看比例: {watch_ratio:.2f}\n"
            context += f"  交互时间: {timestamp}\n"

            if item_meta.get("first_level"):
                context += f"  分类: {item_meta['first_level']}"
                if item_meta.get("second_level"):
                    context += f" > {item_meta['second_level']}"
                context += "\n"

            if item_meta.get("caption"):
                caption = item_meta["caption"][:100]
                context += f"  描述: {caption}...\n"

        context += "\n请基于以上历史记录，分析用户当前的状态。"
        return context

    def generate_training_sample(
        self,
        time_window: Dict,
        user_features: Dict
    ) -> Dict:
        """
        为单个时间窗口生成训练样本

        Args:
            time_window: 时间窗口样本 (来自 TimeWindowSampler)
            user_features: 用户特征

        Returns:
            标准微调格式: {"messages": [...], "metadata": {...}}
        """
        user_context = self.build_user_context(
            time_window["history_interactions"],
            user_features
        )

        # 调用 DeepSeek API
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_context}
            ],
            temperature=0.3,
            max_tokens=6000
        )

        assistant_message = response.choices[0].message.content

        # 清理可能的 markdown 标记
        if assistant_message.startswith("```json"):
            assistant_message = assistant_message[7:]
        if assistant_message.startswith("```"):
            assistant_message = assistant_message[3:]
        if assistant_message.endswith("```"):
            assistant_message = assistant_message[:-3]
        assistant_message = assistant_message.strip()

        return {
            "messages": [
                {"role": "user", "content": user_context},
                {"role": "assistant", "content": assistant_message}
            ],
            "metadata": {
                "user_id": time_window["user_id"],
                "sample_id": time_window["sample_id"],
                "history_end": time_window["history_end"],
                "predict_date": time_window["predict_date"],
                "num_history": time_window["num_history"]
            }
        }

    def batch_generate(
        self,
        time_windows_path: str,
        user_features_path: str,
        output_path: str = "trandata.jsonl",
        max_samples: int = None,
        delay: float = 1.0
    ):
        """
        批量生成训练数据
        """
        # 加载时间窗口样本
        time_windows = []
        with open(time_windows_path, 'r', encoding='utf-8') as f:
            for line in f:
                time_windows.append(json.loads(line))

        if max_samples:
            time_windows = time_windows[:max_samples]

        # 加载用户特征
        user_features_df = pd.read_csv(user_features_path)
        user_features_map = {
            row["user_id"]: row.to_dict()
            for _, row in user_features_df.iterrows()
        }

        # 生成训练数据
        with open(output_path, 'w', encoding='utf-8') as f:
            for window in tqdm(time_windows, desc="生成训练数据"):
                user_id = window["user_id"]
                user_feat = user_features_map.get(user_id, {})

                try:
                    sample = self.generate_training_sample(window, user_feat)
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"\n错误: user_id={user_id}, {e}")
                    # 写入错误标记
                    error_sample = {
                        "messages": [
                            {"role": "user", "content": "错误"},
                            {"role": "assistant", "content": "{}"}
                        ],
                        "metadata": {"error": str(e)}
                    }
                    f.write(json.dumps(error_sample) + '\n')

                time.sleep(delay)

        print(f"\n训练数据生成完成: {output_path}")
```

### 5.3 Embedding 向量生成器

```python
# LLM_part/embedding_generator.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict
from tqdm import tqdm

class SemanticEmbeddingGenerator:
    """
    语义向量生成器

    功能:
    1. 用户状态向量 (5 × 2560)
    2. 物品语义向量 (2560)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        use_4bit: bool = True
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本为向量

        Args:
            texts: 文本列表

        Returns:
            embeddings: (len(texts), 2560) 归一化向量
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**encoded)

        # Mean Pooling
        embeddings = self._mean_pooling(
            outputs,
            encoded["attention_mask"]
        )

        # L2 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        return torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_user_state_embeddings(
        self,
        user_state_texts: Dict[int, Dict[str, str]],
        output_path: str = "user_state_embs.npy"
    ) -> np.ndarray:
        """
        生成用户状态向量

        Args:
            user_state_texts: {
                user_id: {
                    "long_term_intent": "用户长期兴趣描述...",
                    "life_stage": "用户生命周期描述...",
                    "psychological_demand": "心理需求描述...",
                    "retrieval_suggestions": "检索建议...",
                    "interest_growth_points": "兴趣增长点..."
                }
            }

        Returns:
            user_state_embs: (N_users, 5, 2560)
        """
        user_ids = sorted(user_state_texts.keys())
        all_embeddings = []

        # 字段顺序必须与 MuseRecSys config 定义一致
        field_order = [
            "long_term_intent",
            "life_stage",
            "psychological_demand",
            "retrieval_suggestions",
            "interest_growth_points"
        ]

        for user_id in tqdm(user_ids, desc="生成用户状态向量"):
            user_texts = user_state_texts[user_id]
            texts = [user_texts.get(f, "") for f in field_order]

            # 批量编码 5 个维度
            embeddings = self.encode_texts(texts)  # (5, 2560)
            all_embeddings.append(embeddings)

        all_embeddings = np.stack(all_embeddings, axis=0)  # (N, 5, 2560)

        # 保存
        np.save(output_path, all_embeddings)
        print(f"用户状态向量已保存: {output_path}, shape={all_embeddings.shape}")

        return all_embeddings

    def generate_item_semantic_embeddings(
        self,
        item_metadata: Dict[int, str],
        output_path: str = "item_semantic_embs.npy"
    ) -> np.ndarray:
        """
        生成物品语义向量

        Args:
            item_metadata: {
                video_id: "视频描述文本 (caption + 分类 + 标签)"
            }

        Returns:
            item_semantic_embs: (N_items, 2560)
        """
        item_ids = sorted(item_metadata.keys())
        texts = [item_metadata[iid] for iid in item_ids]

        # 批量编码
        embeddings = self.encode_texts(texts)  # (N, 2560)

        # 保存
        np.save(output_path, embeddings)
        print(f"物品语义向量已保存: {output_path}, shape={embeddings.shape}")

        return embeddings


# 使用示例
if __name__ == "__main__":
    generator = SemanticEmbeddingGenerator()

    # 示例: 生成物品向量
    item_texts = {
        0: "颜值随拍视频，展示日常穿搭技巧",
        1: "美食制作教程，家常菜快手做法",
        # ...
    }

    item_embs = generator.generate_item_semantic_embeddings(
        item_texts,
        "item_semantic_embs.npy"
    )
```

---

## 6. 数据格式规范

### 6.1 时间窗口样本格式

```json
{
  "sample_id": "u123_20200719",
  "user_id": 123,
  "history_start": "2020-07-05",
  "history_end": "2020-07-18",
  "predict_date": "2020-07-19",
  "history_interactions": [
    {
      "user_id": 123,
      "video_id": 3649,
      "play_duration": 13838,
      "video_duration": 10867,
      "watch_ratio": 1.27,
      "time": "2020-07-05 00:08:23.438",
      "date": "20200705",
      "timestamp": 1593878903.438
    }
    // ... 更多历史交互
  ],
  "predict_interactions": [
    // predict_date 的实际交互 (用于验证)
  ],
  "num_history": 45,
  "num_predict": 3
}
```

### 6.2 微调训练数据格式

```json
{
  "messages": [
    {
      "role": "user",
      "content": "=== 用户特征 ===\nuser_id: 123\nuser_active_degree: high_active\n...\n\n=== 历史交互记录 ===\n交互 1:\n  视频ID: 3649\n  ..."
    },
    {
      "role": "assistant",
      "content": "{\"user_basic_information\": {\"user_id\": \"123\", ...}, \"user_status_analysis\": {...}, \"retrieval_suggestions\": {...}}"
    }
  ],
  "metadata": {
    "user_id": 123,
    "sample_id": "u123_20200719",
    "history_end": "2020-07-18",
    "predict_date": "2020-07-19",
    "num_history": 45
  }
}
```

### 6.3 向量文件格式

**用户状态向量:**
```
文件: user_state_embs.npy
形状: (N_users, 5, 2560)
数据类型: float32
归一化: L2 normalized

维度 0 (5) 的顺序:
- [0]: long_term_intent
- [1]: life_stage
- [2]: psychological_demand
- [3]: retrieval_suggestions
- [4]: interest_growth_points
```

**物品语义向量:**
```
文件: item_semantic_embs.npy
形状: (N_items, 2560)
数据类型: float32
归一化: L2 normalized
```

---

## 7. 集成到推荐系统

### 7.1 DataLoader 修改

```python
# src/data_loader_kuairec.py (修改版)
import numpy as np
import pandas as pd
from typing import Dict, List

class KuaiRecDataLoader:
    """
    KuaiRec 数据加载器 - 支持预生成的 LLM 向量
    """

    def __init__(
        self,
        data_path: str,
        llm_vector_path: str = "LLM_part/",
        max_users: int = None,
        max_items: int = None
    ):
        self.data_path = data_path
        self.llm_vector_path = llm_vector_path

        # 加载原始数据
        self._load_kuairec_data(max_users, max_items)

        # 加载预生成的 LLM 向量
        self._load_llm_vectors()

    def _load_llm_vectors(self):
        """加载预生成的语义向量"""
        import os

        user_embs_path = os.path.join(
            self.llm_vector_path,
            "user_state_embs.npy"
        )
        item_embs_path = os.path.join(
            self.llm_vector_path,
            "item_semantic_embs.npy"
        )

        try:
            self.user_state_embs = np.load(user_embs_path)
            self.item_semantic_embs = np.load(item_embs_path)
            print(f"✅ LLM 向量加载成功")
            print(f"   用户状态: {self.user_state_embs.shape}")
            print(f"   物品语义: {self.item_semantic_embs.shape}")
        except FileNotFoundError as e:
            print(f"⚠️ LLM 向量未找到: {e}")
            print("   使用 Mock 向量...")
            self._init_mock_vectors()

    def get_user_state_embs(self, user_id: int) -> np.ndarray:
        """
        获取用户状态向量

        Returns:
            (5, 2560) - 5 个维度的语义向量
        """
        if user_id not in self.user_id_to_idx:
            # 返回零向量
            return np.zeros((5, 2560), dtype=np.float32)

        user_idx = self.user_id_to_idx[user_id]
        return self.user_state_embs[user_idx]

    def get_item_semantic_embs(self, item_id: int) -> np.ndarray:
        """
        获取物品语义向量

        Returns:
            (2560,) - 物品语义向量
        """
        if item_id not in self.item_id_to_idx:
            return np.zeros(2560, dtype=np.float32)

        item_idx = self.item_id_to_idx[item_id]
        return self.item_semantic_embs[item_idx]
```

### 7.2 完整运行流程

```bash
# 步骤 1: 生成时间窗口样本
python LLM_part/time_window_sampler.py \
    --data-path "Datasets/KuaiRec 2.0/data/big_matrix.csv" \
    --train-window-days 14 \
    --slide-stride-days 7 \
    --max-users 1000 \
    --output "LLM_part/time_windows.jsonl"

# 步骤 2: 生成微调训练数据 (使用 DeepSeek API)
python LLM_part/llm_data_generator.py \
    --time-windows "LLM_part/time_windows.jsonl" \
    --item-metadata "Datasets/KuaiRec 2.0/data/kuairec_caption_category.csv" \
    --user-features "Datasets/KuaiRec 2.0/data/user_features.csv" \
    --output "LLM_part/trandata.jsonl" \
    --max-samples 1000 \
    --api-key "$DEEPSEEK_API_KEY"

# 步骤 3: 微调 Qwen 模型
python LLM_part/SFT_Qwen1.7b/finetune_qwen4b.py \
    --data-path "LLM_part/trandata.jsonl" \
    --output-dir "LLM_part/qwen3_finetuned" \
    --num-epochs 3

# 步骤 4: 生成语义向量
# 4.1 使用微调后的模型生成用户状态文本
python LLM_part/generate_user_states.py \
    --model "LLM_part/qwen3_finetuned" \
    --time-windows "LLM_part/time_windows.jsonl" \
    --output "LLM_part/user_state_texts.json"

# 4.2 使用 Embedding 模型编码
python LLM_part/embedding_generator.py \
    --user-states "LLM_part/user_state_texts.json" \
    --item-metadata "Datasets/KuaiRec 2.0/data/kuairec_caption_category.csv" \
    --output-dir "LLM_part/"

# 步骤 5: 运行推荐系统
python main.py \
    --data-path "Datasets/KuaiRec 2.0" \
    --llm-vector-path "LLM_part/" \
    --enable-llm-features
```

---

## 8. 关键注意事项

### 8.1 时间序列约束

**必须遵守的原则:**
- ✅ 只使用 `history_end` 之前的数据作为 LLM 输入
- ❌ 不能使用 `predict_date` 之后的信息
- ❌ 不能在训练时"看到"测试数据

### 8.2 成本优化

**DeepSeek API 成本:**
- deepseek-chat: ¥1 / 1M tokens (输入)
- 估计每个样本: ~2000 tokens (输入) + ~1500 tokens (输出)
- 1000 个样本成本: ~¥3-5

**优化建议:**
1. 先用少量样本 (100-500) 验证流程
2. 只对活跃用户生成样本
3. 考虑使用更小的历史窗口 (7 天)
4. 批量 API 调用 (减少请求次数)

### 8.3 数据质量验证

**验证检查点:**
1. 时间戳顺序正确
2. 历史窗口不包含预测日数据
3. 用户-物品 ID 映射一致
4. JSON 格式有效
5. 向量维度正确

---

## 9. 总结

### 9.1 完整流程图

```
KuaiRec 原始数据 (62天)
         │
         ▼
┌─────────────────────┐
│  时间窗口采样        │  ← 核心创新：模拟真实推荐场景
│  (14天历史 → 1天预测)│
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  DeepSeek API        │  ← 成本：~¥3-5/1000样本
│  生成训练数据        │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Qwen3-1.7B 微调     │  ← LoRA 高效微调
│  (一次性投入)        │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  微调模型推理        │  ← 可重复使用
│  + Embedding 编码    │
└─────────────────────┘
         │
         ▼
   2560维语义向量
         │
         ▼
┌─────────────────────┐
│  MuseRecSys Pipeline │
│  (召回/精排/重排)     │
└─────────────────────┘
```

### 9.2 下一步行动

| 优先级 | 任务 | 文件 |
|--------|------|------|
| P0 | 实现时间窗口采样器 | `LLM_part/time_window_sampler.py` |
| P0 | 改造 LLM 数据生成器 | `LLM_part/llm_data_generator.py` |
| P1 | 实现向量生成器 | `LLM_part/embedding_generator.py` |
| P1 | 修改 DataLoader | `src/data_loader_kuairec.py` |
| P2 | 添加验证脚本 | `LLM_part/validate_pipeline.py` |

---

**文档结束**

*本文档详细说明了 KuaiRec 数据集与 LLM_part 的联动使用方案，核心创新在于时间窗口采样策略，确保了推荐场景的真实性。*
