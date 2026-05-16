# KuaiRec 2.0 数据集分析与使用指南

**分析日期:** 2026-02-26
**数据来源:** https://kuairec.com/
**数据路径:** `C:\Users\53025\MuseRecSys\Datasets\KuaiRec 2.0`

---

## 一、数据集概述

### 1.1 数据集基本信息

KuaiRec 是一个**完全观测**的用户 - 物品交互数据集，源自快手视频应用的真实日志。

**核心特点:**
- 🔹 首个包含**完全观测交互矩阵**的推荐系统数据集
- 🔹 包含一个**小矩阵** (1,411 用户，99.6% 密度) - 近乎全观测
- 🔹 包含一个**大矩阵** (7,176 用户，1,253 万交互) - 部分观测
- 🔹 丰富的用户特征、物品特征、社交网络数据
- 🔹 许可证：**CC BY-SA 4.0** (需署名，相同方式共享)

### 1.2 数据规模统计

| 数据文件 | 记录数 | 说明 |
|---------|--------|------|
| `big_matrix.csv` | 12,530,806 条 | 大矩阵 - 主交互数据 |
| `small_matrix.csv` | 4,676,570 条 | 小矩阵 - 1411 用户的全观测数据 |
| `user_features.csv` | 7,176 条 | 用户特征 |
| `item_categories.csv` | 10,728 条 | 物品标签 |
| `item_daily_features.csv` | 343,341 条 | 物品每日统计 |
| `social_network.csv` | 472 条 | 社交网络 |
| `kuairec_caption_category.csv` | - | 物品详情/视频描述 |

---

## 二、数据文件详细说明

### 2.1 big_matrix.csv - 主交互矩阵

**记录数:** 12,530,806 条 (约 1253 万)
**用户数:** 7,176
**时间范围:** 2020-07-05 至 2020-09-05 (约 2 个月)

| 字段名 | 类型 | 说明 | 示例值 |
|--------|------|------|--------|
| `user_id` | int | 用户 ID | 0, 1, 2... |
| `video_id` | int | 视频 ID | 3649, 9598... |
| `play_duration` | int | 播放时长 (毫秒) | 13838 (13.8 秒) |
| `video_duration` | int | 视频总时长 (毫秒) | 10867 |
| `time` | datetime | 交互时间 | 2020-07-05 00:08:23.438 |
| `date` | int | 日期 | 20200705 |
| `timestamp` | float | Unix 时间戳 | 1593878903.438 |
| `watch_ratio` | float | 观看比例 (=play_duration/video_duration) | 1.27 (>1 表示重复观看) |

**关键字段说明:**

```python
# watch_ratio 解读
watch_ratio < 0.3   # 浅层交互 (用户快速跳过)
watch_ratio 0.3-0.8  # 中等交互 (部分观看)
watch_ratio > 0.8    # 深层交互 (完整观看)
watch_ratio > 1.0    # 重复观看
```

**数据分布统计 (来自官方 Notebook):**
- 视频时长中位数：约 10,000ms (10 秒)
- watch_ratio 均值：约 0.8 (80% 的视频被观看到结束)
- 用户日均播放次数：呈长尾分布

---

### 2.2 small_matrix.csv - 小矩阵 (全观测)

**记录数:** 4,676,570 条
**用户数:** 1,411
**特点:** 99.6% 密度 - 几乎包含所有曝光记录

**用途:**
- 无偏推荐算法研究
- 因果推断研究
- 反事实评估 (Counterfactual Evaluation)

**字段:** 与 big_matrix 相同

---

### 2.3 user_features.csv - 用户特征

**记录数:** 7,176 条 (覆盖大矩阵所有用户)

| 字段名 | 类型 | 说明 | 示例值 |
|--------|------|------|--------|
| `user_id` | int | 用户 ID | 0, 1, 2... |
| `user_active_degree` | str | 用户活跃度 | "high_active", "full_active" |
| `is_lowactive_period` | int | 是否低活跃期 | 0/1 |
| `is_live_streamer` | int | 是否主播 | 0/1 |
| `is_video_author` | int | 是否视频作者 | 0/1 |
| `follow_user_num` | int | 关注数 | 5, 386... |
| `follow_user_num_range` | str | 关注数区间 | "(0,10]", "(250,500]" |
| `fans_user_num` | int | 粉丝数 | 0, 4... |
| `fans_user_num_range` | str | 粉丝数区间 | "[1,10)", "(10,50]" |
| `friend_user_num` | int | 好友数 | 0, 2... |
| `register_days` | int | 注册天数 | 107, 327... |
| `onehot_feat0~17` | int/float | One-hot 编码特征 | 0/1 |

**特征统计:**
- 用户活跃度：`full_active` > `high_active` > `low_active`
- 平均关注数：呈长尾分布，大部分用户关注较少
- 粉丝数：高度稀疏，大部分用户无粉丝

---

### 2.4 item_categories.csv - 物品标签

**记录数:** 10,728 条

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `video_id` | int | 视频 ID |
| `feat` | list | 标签 ID 列表 |

**示例:**
```
video_id=0: feat=[8]
video_id=1: feat=[27, 9]
video_id=2: feat=[9]
```

**标签统计:**
- 平均每个视频标签数：1-2 个
- 标签 ID 范围：2-31 (共约 30 个标签类别)

---

### 2.5 kuairec_caption_category.csv - 视频详情

**最丰富的物品元数据来源**

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `video_id` | int | 视频 ID |
| `manual_cover_text` | str | 封面文字 |
| `caption` | str | 视频描述/文案 |
| `topic_tag` | list | 话题标签 |
| `first_level_category_id` | int | 一级分类 ID |
| `first_level_category_name` | str | 一级分类名称 |
| `second_level_category_id` | int | 二级分类 ID |
| `second_level_category_name` | str | 二级分类名称 |
| `third_level_category_id` | int | 三级分类 ID |
| `third_level_category_name` | str | 三级分类名称 |

**分类体系示例:**
```
一级分类：颜值 (ID:8)
  └── 二级分类：颜值随拍 (ID:673)

一级分类：高新数码 (ID:27)
  └── 二级分类：无

一级分类：喜剧 (ID:9)
  └── 二级分类：搞笑互动 (ID:727)

一级分类：摄影 (ID:26)
  └── 二级分类：主题摄影 (ID:686)
      └── 三级分类：景物摄影 (ID:2434)

一级分类：时尚 (ID:5)
  └── 二级分类：营销售卖 (ID:737)
      └── 三级分类：女装 (ID:2596)

一级分类：明星娱乐 (ID:6)
  └── 二级分类：娱乐八卦 (ID:667)
      └── 三级分类：饭制 (ID:2375)

一级分类：情感 (ID:19)
一级分类：美食 (ID:12)
  └── 二级分类：美食日常 (ID:292)
      └── 三级分类：美食分享 (ID:1461)
```

**话题标签示例:**
```
["五爱市场", "感谢快手我要上热门", "搞笑"]
["刘耀文", "宋亚轩", "文轩", "顾子璇是樱桃吖"]
["灵魂属性大揭秘"]
```

---

### 2.6 item_daily_features.csv - 物品每日统计

**记录数:** 343,341 条 (物品 × 日期)

**核心字段 (58 列):**

| 字段组 | 字段名示例 | 说明 |
|--------|-----------|------|
| **基础信息** | `video_id`, `date`, `author_id` | 视频 ID、日期、作者 ID |
| **视频属性** | `video_duration`, `video_width`, `video_height` | 时长、分辨率 |
| **曝光数据** | `show_cnt`, `show_user_num` | 曝光次数/人数 |
| **播放数据** | `play_cnt`, `play_user_num`, `play_duration` | 播放次数/时长 |
| **完播数据** | `complete_play_cnt`, `valid_play_cnt` | 完播次数、有效播放 |
| **互动数据** | `like_cnt`, `comment_cnt`, `share_cnt` | 点赞/评论/分享 |
| **转化数据** | `follow_cnt`, `collect_cnt`, `download_cnt` | 关注/收藏/下载 |
| **负反馈** | `report_cnt`, `reduce_similar_cnt` | 举报/减少相似 |

**典型用途:**
- 物品热门度时序分析
- 多目标优化 (播放 + 互动 + 转化)
- 负反馈建模

---

### 2.7 social_network.csv - 社交网络

**记录数:** 472 条 (有社交关系的用户)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `user_id` | int | 用户 ID |
| `friend_list` | list | 好友 ID 列表 |

**示例:**
```
user_id=3371: friend_list=[2975]
user_id=5352: friend_list=[4202, 7126]
```

**社交图统计:**
- 平均好友数：< 2 (高度稀疏)
- 最大好友数：有限

---

## 三、数据加载代码

### 3.1 基础加载 (loaddata.py)

```python
import pandas as pd

# 加载交互矩阵
big_matrix = pd.read_csv("data/big_matrix.csv")
small_matrix = pd.read_csv("data/small_matrix.csv")

# 加载社交网络
social_network = pd.read_csv("data/social_network.csv")
social_network["friend_list"] = social_network["friend_list"].map(eval)

# 加载物品特征
item_categories = pd.read_csv("data/item_categories.csv")
item_categories["feat"] = item_categories["feat"].map(eval)

# 加载用户特征
user_features = pd.read_csv("data/user_features.csv")

# 加载物品每日特征
item_daily_features = pd.read_csv("data/item_daily_features.csv")
```

### 3.2 推荐数据加载器 (可集成到项目)

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class KuaiRecData:
    """KuaiRec 数据集容器"""
    interactions: pd.DataFrame
    user_features: pd.DataFrame
    item_features: pd.DataFrame
    item_daily: pd.DataFrame
    social_network: pd.DataFrame

class KuaiRecLoader:
    """KuaiRec 数据加载器"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self._load_all_data()

    def _load_all_data(self) -> KuaiRecData:
        """加载所有数据"""
        return KuaiRecData(
            interactions=pd.read_csv(f"{self.data_path}/data/big_matrix.csv"),
            user_features=pd.read_csv(f"{self.data_path}/data/user_features.csv"),
            item_features=self._load_item_features(),
            item_daily=pd.read_csv(f"{self.data_path}/data/item_daily_features.csv"),
            social_network=self._load_social_network()
        )

    def _load_item_features(self) -> pd.DataFrame:
        """加载物品特征 (合并 categories 和 caption)"""
        categories = pd.read_csv(f"{self.data_path}/data/item_categories.csv")
        categories["feat"] = categories["feat"].map(eval)

        caption = pd.read_csv(f"{self.data_path}/data/kuairec_caption_category.csv")

        # 合并两个表
        return caption.merge(categories, on="video_id", how="left")

    def _load_social_network(self) -> pd.DataFrame:
        """加载社交网络"""
        df = pd.read_csv(f"{self.data_path}/data/social_network.csv")
        df["friend_list"] = df["friend_list"].map(eval)
        return df

    def get_user_history(self, user_id: int) -> pd.DataFrame:
        """获取用户历史交互"""
        return self.data.interactions[
            self.data.interactions["user_id"] == user_id
        ].sort_values("timestamp")

    def get_item_stats(self, video_id: int) -> pd.DataFrame:
        """获取物品统计信息"""
        return self.data.item_daily[
            self.data.item_daily["video_id"] == video_id
        ]

    def get_user_profile(self, user_id: int) -> pd.Series:
        """获取用户画像"""
        profile = self.data.user_features[
            self.data.user_features["user_id"] == user_id
        ].iloc[0]
        return profile

    def create_user_item_matrix(self, sparse: bool = True) -> pd.DataFrame:
        """创建用户 - 物品交互矩阵"""
        df = self.data.interactions

        # 聚合用户 - 物品交互
        pivot = df.pivot_table(
            index="user_id",
            columns="video_id",
            values="watch_ratio",
            aggfunc="mean"
        )

        return pivot

    def compute_watch_ratio_labels(self, threshold: float = 0.8) -> pd.DataFrame:
        """
        计算二分类标签 (正样本：观看比例 >= threshold)

        Returns:
            DataFrame with columns: user_id, video_id, label
        """
        df = self.data.interactions.copy()
        df["label"] = (df["watch_ratio"] >= threshold).astype(int)
        return df[["user_id", "video_id", "label"]]
```

---

## 四、MuseRecSys 项目数据使用思路

### 4.1 整体集成方案

```
┌─────────────────────────────────────────────────────────────────┐
│                    KuaiRec 数据集                                │
├─────────────────────────────────────────────────────────────────┤
│  big_matrix.csv    → 用户交互历史 (play_duration, watch_ratio)   │
│  user_features.csv → 用户特征 (年龄、性别、活跃度等)              │
│  item_*.csv        → 物品特征 (类别、标签、描述)                  │
│  social_network    → 社交关系 (可选)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  数据处理与特征工程                              │
├─────────────────────────────────────────────────────────────────┤
│  1. 用户历史序列构建 (按时间排序的 video_id 列表)                │
│  2. 用户特征向量化 (数值特征 + One-hot)                         │
│  3. 物品特征向量化 (类别编码 + 标签 Multi-hot)                   │
│  4. LLM 语义向量生成 (用 caption 生成 2560 维向量)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MuseRecSys 推荐系统                             │
├─────────────────────────────────────────────────────────────────┤
│  Recall Layer  → 4 路召回 (用真实用户历史)                        │
│  Ranking Layer → DIN+MMoE (用真实特征训练)                      │
│  ReRank Layer  → DPP 多样性重排                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4.2 数据层改造 (src/data_loader.py)

**当前问题:** 目前使用 Mock 数据，需要替换为真实数据

**改造方案:**

```python
# src/data_loader_kuairec.py (新建)
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class UserFeatures:
    user_id: int
    age: Optional[int]
    gender: Optional[int]
    user_active_degree: float
    feature_vector: np.ndarray

@dataclass
class ItemFeatures:
    item_id: int
    category_id: int
    tags: List[str]
    video_duration: int
    feature_vector: np.ndarray

class KuaiRecDataLoader:
    """KuaiRec 真实数据加载器"""

    def __init__(
        self,
        data_path: str,
        emb_dim: int = 2560,
        max_users: Optional[int] = None,
        max_items: Optional[int] = None
    ):
        """
        Args:
            data_path: KuaiRec 数据目录
            emb_dim: LLM 语义向量维度
            max_users: 最大用户数 (用于子集测试)
            max_items: 最大物品数
        """
        self.data_path = data_path
        self.emb_dim = emb_dim
        self.max_users = max_users
        self.max_items = max_items

        self._interactions: Optional[pd.DataFrame] = None
        self._user_features: Dict[int, UserFeatures] = {}
        self._item_features: Dict[int, ItemFeatures] = {}
        self._user_history: Dict[int, List[int]] = {}
        self._user_state_embs: Optional[np.ndarray] = None
        self._item_semantic_embs: Optional[np.ndarray] = None

        self._load_data()

    def _load_data(self):
        """加载所有数据"""
        print("Loading KuaiRec data...")

        # 1. 加载交互数据
        self._load_interactions()

        # 2. 加载用户特征
        self._load_user_features()

        # 3. 加载物品特征
        self._load_item_features()

        # 4. 构建用户历史
        self._build_user_history()

        # 5. 加载/生成 LLM 语义向量
        self._load_llm_embeddings()

        print(f"Data loaded: {len(self._user_features)} users, "
              f"{len(self._item_features)} items")

    def _load_interactions(self):
        """加载交互数据并过滤"""
        df = pd.read_csv(f"{self.data_path}/data/big_matrix.csv")

        # 过滤有效交互 (watch_ratio > 0.3 视为有效)
        df = df[df["watch_ratio"] > 0.3].copy()

        # 限制用户和物品数量 (用于快速测试)
        if self.max_users:
            valid_users = df["user_id"].unique()[:self.max_users]
            df = df[df["user_id"].isin(valid_users)]

        if self.max_items:
            valid_items = df["video_id"].unique()[:self.max_items]
            df = df[df["video_id"].isin(valid_items)]

        self._interactions = df
        print(f"  Interactions: {len(df)} records")

    def _load_user_features(self):
        """加载用户特征并转换为项目格式"""
        df = pd.read_csv(f"{self.data_path}/data/user_features.csv")

        # 映射活跃度为数值
        active_map = {
            "low_active": 0.2,
            "high_active": 0.6,
            "full_active": 1.0
        }

        for _, row in df.iterrows():
            user_id = row["user_id"]

            # 构造 32 维特征向量
            feature_vector = self._encode_user_features(row)

            self._user_features[user_id] = UserFeatures(
                user_id=user_id,
                age=None,  # KuaiRec 无年龄信息
                gender=None,  # KuaiRec 无性别信息
                user_active_degree=active_map.get(
                    row["user_active_degree"], 0.5
                ),
                feature_vector=feature_vector
            )

    def _encode_user_features(self, row: pd.Series) -> np.ndarray:
        """
        将用户特征编码为 32 维向量

        特征设计:
        [0-4]   活跃度相关 (user_active_degree, is_lowactive_period, ...)
        [5-9]   社交特征 (follow, fans, friend 数量)
        [10-14] 创作特征 (is_live_streamer, is_video_author)
        [15-29] onehot 特征 (取前 15 个)
        [30-31] 补充特征
        """
        features = np.zeros(32, dtype=np.float32)

        # 活跃度
        active_map = {"low_active": 0.2, "high_active": 0.6, "full_active": 1.0}
        features[0] = active_map.get(row["user_active_degree"], 0.5)
        features[1] = row.get("is_lowactive_period", 0)
        features[2] = row.get("is_live_streamer", 0)
        features[3] = row.get("is_video_author", 0)

        # 社交特征 (归一化)
        features[5] = np.log1p(row.get("follow_user_num", 0)) / 10
        features[6] = np.log1p(row.get("fans_user_num", 0)) / 10
        features[7] = np.log1p(row.get("friend_user_num", 0)) / 5

        # 注册天数
        features[8] = np.log1p(row.get("register_days", 0)) / 10

        # onehot 特征
        for i in range(15):
            col = f"onehot_feat{i}"
            if col in row:
                features[10 + i] = row[col]

        return features

    def _load_item_features(self):
        """加载物品特征"""
        # 加载物品分类
        df = pd.read_csv(f"{self.data_path}/data/item_categories.csv")
        df["feat"] = df["feat"].map(eval)

        # 加载物品详情
        detail = pd.read_csv(
            f"{self.data_path}/data/kuairec_caption_category.csv"
        )

        # 合并
        df = df.merge(detail, on="video_id", how="left")

        # 限制物品数量
        if self.max_items:
            df = df.head(self.max_items)

        for _, row in df.iterrows():
            item_id = row["video_id"]

            # 构造 32 维特征向量
            feature_vector = self._encode_item_features(row)

            # 获取分类 ID (使用一级分类)
            category_id = row.get("first_level_category_id", 0)
            if pd.isna(category_id):
                category_id = 0

            # 获取标签
            tags = row.get("topic_tag", [])
            if isinstance(tags, float):  # NaN 情况
                tags = []

            # 视频时长 (毫秒)
            duration = row.get("video_duration", 0)
            if pd.isna(duration):
                duration = 0

            self._item_features[item_id] = ItemFeatures(
                item_id=item_id,
                category_id=int(category_id),
                tags=tags,
                video_duration=int(duration),
                feature_vector=feature_vector
            )

    def _encode_item_features(self, row: pd.Series) -> np.ndarray:
        """
        将物品特征编码为 32 维向量

        特征设计:
        [0-4]   分类特征 (一级、二级、三级分类 embedding)
        [5-14]  标签 Multi-hot (取前 10 个标签)
        [15-19] 视频属性 (时长、分辨率)
        [20-31] 预留
        """
        features = np.zeros(32, dtype=np.float32)

        # 分类 ID (归一化)
        features[0] = row.get("first_level_category_id", 0) / 30
        features[1] = row.get("second_level_category_id", 0) / 800
        features[2] = row.get("third_level_category_id", 0) / 3000

        # 视频属性
        duration = row.get("video_duration", 0)
        features[15] = np.log1p(duration) / 15  # 归一化
        features[16] = row.get("video_width", 0) / 1000
        features[17] = row.get("video_height", 0) / 2000

        return features

    def _build_user_history(self):
        """构建用户历史交互序列"""
        for user_id in self._user_features.keys():
            user_interactions = self._interactions[
                self._interactions["user_id"] == user_id
            ].sort_values("timestamp")

            history = user_interactions["video_id"].tolist()
            self._user_history[user_id] = history

    def _load_llm_embeddings(self):
        """
        加载 LLM 语义向量

        两种方案:
        1. 预计算: 使用 Qwen3-Embedding-4B 离线生成
        2. Mock: 随机生成 (当前回退方案)
        """
        num_users = len(self._user_features)
        num_items = len(self._item_features)

        # 尝试加载预计算的向量
        llm_user_path = f"{self.data_path}/llm_user_embs.npy"
        llm_item_path = f"{self.data_path}/llm_item_embs.npy"

        try:
            self._user_state_embs = np.load(llm_user_path)
            self._item_semantic_embs = np.load(llm_item_path)
            print("  LLM embeddings loaded from cache")
        except FileNotFoundError:
            # 使用 Mock 向量
            print("  LLM embeddings not found, using mock vectors")
            np.random.seed(42)
            self._user_state_embs = np.random.randn(
                num_users, 5, self.emb_dim
            ).astype(np.float32)
            self._item_semantic_embs = np.random.randn(
                num_items, self.emb_dim
            ).astype(np.float32)

            # L2 归一化
            self._user_state_embs /= (
                np.linalg.norm(self._user_state_embs, axis=2, keepdims=True)
                + 1e-8
            )
            self._item_semantic_embs /= (
                np.linalg.norm(self._item_semantic_embs, axis=1, keepdims=True)
                + 1e-8
            )

    # ============ 对外接口 (保持与原有接口兼容) ============

    def get_user_features(self, user_id: int) -> UserFeatures:
        return self._user_features[user_id]

    def get_item_features(self, item_id: int) -> ItemFeatures:
        return self._item_features[item_id]

    def get_user_history(self, user_id: int, max_len: int = 20) -> List[int]:
        history = self._user_history.get(user_id, [])
        return history[-max_len:] if len(history) > max_len else history

    def get_user_state_embs(self, user_id: int) -> np.ndarray:
        user_idx = list(self._user_features.keys()).index(user_id)
        return self._user_state_embs[user_idx]

    def get_item_semantic_embs(self, item_id: int) -> np.ndarray:
        item_idx = list(self._item_features.keys()).index(item_id)
        return self._item_semantic_embs[item_idx]
```

---

### 4.3 LLM 语义向量生成方案

#### 方案 A: 使用 Qwen3-Embedding-4B 离线生成

**用户状态向量 (5 维) 生成:**

```python
# llm_offline/generate_user_states.py
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import json

def build_user_prompt(history_items: List[dict]) -> str:
    """
    构建用户状态生成 prompt

    输入: 用户历史观看的视频列表
    输出: JSON 格式的用户状态
    """
    prompt = """你是一个推荐系统专家。请分析以下用户历史观看行为，生成 5 个维度的用户状态向量描述:

用户历史观看视频 (按时间顺序):
"""
    for item in history_items[-20:]:  # 取最近 20 个
        prompt += f"""
- 视频{item['video_id']}:
  分类：{item.get('category_name', '未知')}
  标签：{item.get('tags', [])}
  观看比例：{item.get('watch_ratio', 0):.2f}
"""

    prompt += """
请用 JSON 格式输出以下 5 个维度的描述 (每维度 50-100 字):
{
  "long_term_intent": "用户的长期兴趣主题",
  "life_stage": "用户生命周期阶段特征",
  "psychological_demand": "当前心理需求",
  "retrieval_suggestions": "检索建议关键词",
  "interest_growth_points": "潜在兴趣探索方向"
}
"""
    return prompt

def generate_user_states(
    data_path: str,
    output_path: str,
    batch_size: int = 32
):
    """生成所有用户的状态向量"""

    # 加载模型
    model_name = "Qwen/Qwen3-Embedding-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载用户历史
    interactions = pd.read_csv(f"{data_path}/data/big_matrix.csv")
    item_info = pd.read_csv(f"{data_path}/data/kuairec_caption_category.csv")

    # 为每个用户构建 prompt
    user_prompts = []
    user_ids = []

    for user_id, group in interactions.groupby("user_id"):
        user_history = group.merge(item_info, on="video_id", how="left")
        prompt = build_user_prompt(user_history.to_dict("records"))
        user_prompts.append(prompt)
        user_ids.append(user_id)

    # 批量生成 embedding
    all_states = []

    for i in range(0, len(user_prompts), batch_size):
        batch_prompts = user_prompts[i:i+batch_size]

        encoded = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        all_states.append(embeddings.cpu().numpy())

    # 保存
    all_states = np.vstack(all_states)
    np.save(output_path, all_states)
```

**物品语义向量生成:**

```python
# llm_offline/generate_item_embeddings.py
def build_item_prompt(item_row: pd.Series) -> str:
    """构建物品语义生成 prompt"""
    prompt = f"""视频信息:
- 标题/封面文字：{item_row.get('manual_cover_text', '无')}
- 描述：{item_row.get('caption', '无')}
- 话题标签：{item_row.get('topic_tag', [])}
- 分类：{item_row.get('first_level_category_name', '未知')} >
        {item_row.get('second_level_category_name', '未知')}
"""
    return prompt

# 类似用户状态生成的流程
```

#### 方案 B: 使用 Caption 直接 Embedding (简化版)

```python
# 直接使用 caption 文本生成 embedding
def generate_item_embedding_from_caption(
    caption: str,
    model,
    tokenizer
) -> np.ndarray:
    """从视频描述生成语义向量"""
    encoded = tokenizer(
        caption,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**encoded)
        embedding = outputs.last_hidden_state.mean(dim=1)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding.cpu().numpy()[0]
```

---

### 4.4 训练数据构建

#### 正负样本构建

```python
# src/train_data_builder.py
class KuaiRecTrainDataBuilder:
    """构建训练数据"""

    def __init__(self, data_loader: KuaiRecDataLoader):
        self.data_loader = data_loader

    def build_ranking_dataset(
        self,
        watch_ratio_threshold: float = 0.8,
        negative_sampling_ratio: int = 4
    ):
        """
        构建精排模型训练数据

        正样本：watch_ratio >= threshold
        负样本：从用户未交互物品中采样
        """
        interactions = self.data_loader._interactions

        # 正样本
        positives = interactions[
            interactions["watch_ratio"] >= watch_ratio_threshold
        ].copy()
        positives["label_ctr"] = 1

        # 负样本 (用户交互过但 watch_ratio 很低的)
        negatives = interactions[
            interactions["watch_ratio"] < 0.3
        ].copy()
        negatives["label_ctr"] = 0

        # 合并
        train_data = pd.concat([positives, negatives])

        return train_data

    def build_seq_dataset(self, max_seq_len: int = 50):
        """
        构建序列推荐训练数据

        对于每个用户，按时间顺序构建序列:
        Input: [item_1, item_2, ..., item_{t-1}]
        Target: item_t
        """
        sequences = []
        targets = []
        user_ids = []

        for user_id, history in self.data_loader._user_history.items():
            if len(history) < 2:
                continue

            # 滑动窗口构建样本
            for t in range(1, min(len(history), max_seq_len)):
                seq = history[:t]
                target = history[t]

                # Padding
                if len(seq) < max_seq_len:
                    seq = seq + [0] * (max_seq_len - len(seq))

                sequences.append(seq)
                targets.append(target)
                user_ids.append(user_id)

        return {
            "sequences": np.array(sequences),
            "targets": np.array(targets),
            "user_ids": np.array(user_ids)
        }
```

---

### 4.5 评估指标

```python
# src/evaluation.py
class KuaiRecEvaluator:
    """KuaiRec 数据集评估器"""

    def __init__(self, ground_truth: Dict[int, List[int]]):
        """
        Args:
            ground_truth: {user_id: [positive_items]}
        """
        self.ground_truth = ground_truth

    def evaluate(
        self,
        predictions: Dict[int, List[int]],
        k: int = 10
    ) -> Dict[str, float]:
        """
        计算评估指标

        Returns:
            {
                "ndcg@k": float,
                "hit_rate@k": float,
                "mrr": float,
                "coverage": float
            }
        """
        ndcg_scores = []
        hit_scores = []
        mrr_scores = []

        for user_id, pred_items in predictions.items():
            if user_id not in self.ground_truth:
                continue

            true_items = set(self.ground_truth[user_id])
            pred_items = pred_items[:k]

            # NDCG@K
            dcg = 0.0
            idcg = 0.0
            for i, item in enumerate(pred_items):
                if item in true_items:
                    dcg += 1.0 / np.log2(i + 2)

            for i in range(min(len(true_items), k)):
                idcg += 1.0 / np.log2(i + 2)

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

            # Hit Rate@K
            hit = 1.0 if any(item in true_items for item in pred_items) else 0.0
            hit_scores.append(hit)

            # MRR
            mrr = 0.0
            for i, item in enumerate(pred_items):
                if item in true_items:
                    mrr = 1.0 / (i + 1)
                    break
            mrr_scores.append(mrr)

        return {
            "ndcg@k": np.mean(ndcg_scores),
            "hit_rate@k": np.mean(hit_scores),
            "mrr": np.mean(mrr_scores),
            "coverage": len(set().union(*[set(v) for v in predictions.values()]))
                       / len(self.ground_truth)
        }
```

---

## 五、实施步骤

### 阶段 1: 数据准备 (1-2 天)

```bash
# 1. 验证数据完整性
ls Datasets/KuaiRec\ 2.0/data/

# 2. 数据预处理
python src/data_loader_kuairec.py  # 测试数据加载

# 3. 生成 LLM 语义向量 (可选)
python llm_offline/generate_user_states.py
python llm_offline/generate_item_embeddings.py
```

### 阶段 2: 数据层替换 (1 天)

```python
# 修改 main.py 中的数据层导入
# from src.data_loader import DataLoader
from src.data_loader_kuairec import KuaiRecDataLoader

# 修改初始化
# data_loader = DataLoader(num_users=100, num_items=500)
data_loader = KuaiRecDataLoader(
    data_path="Datasets/KuaiRec 2.0",
    max_users=1000,  # 先用子集测试
    max_items=2000
)
```

### 阶段 3: 模型训练 (2-3 天)

```python
# 构建训练数据
builder = KuaiRecTrainDataBuilder(data_loader)
train_data = builder.build_ranking_dataset()

# 训练精排模型
from src.ranking import StateEnhancedRankingModel, RankingLoss

# 训练循环...
```

### 阶段 4: 离线评估 (1 天)

```python
# 构建测试集
test_data = builder.build_ranking_dataset(...)

# 评估
evaluator = KuaiRecEvaluator(ground_truth=test_ground_truth)
metrics = evaluator.evaluate(predictions, k=10)

print(f"NDCG@10: {metrics['ndcg@k']:.4f}")
print(f"Hit Rate@10: {metrics['hit_rate@k']:.4f}")
```

---

## 六、关键注意事项

### 6.1 数据许可证

**CC BY-SA 4.0 要求:**
- ✅ 必须署名 (引用原论文)
- ✅ 相同方式共享 (衍生作品使用相同许可证)
- ✅ 标明修改

**引用格式:**
```
@article{gao2022kuairec,
  title={KuaiRec: A fully-observed dataset for recommendation systems},
  author={Gao, Chongming and Liu, Shijun and Xu, Wen and ...},
  journal={arXiv preprint arXiv:2210.11191},
  year={2022}
}
```

### 6.2 数据规模控制

**推荐配置:**
```python
# 快速测试
max_users=500, max_items=1000

# 中等规模
max_users=2000, max_items=5000

# 全量 (需要足够计算资源)
max_users=7176, max_items=10728
```

### 6.3 内存优化

```python
# 使用 pandas 分块读取
chunk_size = 100000
chunks = pd.read_csv("big_matrix.csv", chunksize=chunk_size)

# 使用 numpy 内存映射
user_embs = np.load("user_embs.npy", mmap_mode="r")
```

---

## 七、总结

### KuaiRec 数据集优势

| 优势 | 说明 |
|------|------|
| 🔹 完全观测 | small_matrix 99.6% 密度，适合无偏研究 |
| 🔹 丰富特征 | 用户特征 + 物品特征 + 社交网络 |
| 🔹 真实场景 | 来自快手真实日志，非模拟数据 |
| 🔹 时序信息 | 精确到毫秒的时间戳 |
| 🔹 多目标 | 播放 + 点赞 + 评论 + 分享 + 负反馈 |

### 与 MuseRecSys 的契合点

| MuseRecSys 模块 | KuaiRec 数据支持 |
|-----------------|-----------------|
| 数据层 | 真实用户/物品特征替换 Mock 数据 |
| 召回层 | 用户历史序列支持所有 4 路召回 |
| 精排层 | 丰富特征支持 DIN+MMoE 训练 |
| 重排层 | 物品分类支持多样性计算 |
| LLM 模块 | Caption 文本支持语义向量生成 |

### 下一步行动

1. **[ ] 数据验证** - 运行 `loaddata.py` 验证数据完整性
2. **[ ] 创建 KuaiRecDataLoader** - 实现真实数据加载器
3. **[ ] 生成 LLM 向量** - 使用 Qwen3-Embedding-4B 生成语义向量
4. **[ ] 集成测试** - 用 KuaiRec 数据运行完整流水线
5. **[ ] 离线评估** - 构建测试集并评估效果

---

**文档结束**

*本分析基于 KuaiRec 2.0 数据集和 MuseRecSys 项目需求*
