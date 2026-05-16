"""
KuaiRec Teacher数据生成脚本

功能：从KuaiRec数据集构建时间快照样本，调用DeepSeek API生成用户状态推断

支持两种模式：
- small_matrix: 63天连续，适合开发调试
- big_matrix: 28天不连续观测，适合正式训练

时间窗口设计（big_matrix）：
- 只在有观测的日期上采样 anchor_date
- 历史只使用更早的已观测日期
- 不做"按自然日连续衰减"的强假设

数据块划分：
- Block 1: 2020-07-05 ~ 2020-07-12
- Block 2: 2020-08-01 ~ 2020-08-10
- Block 3: 2020-08-27 ~ 2020-09-05

训练策略：
- train: Block 1 的日期
- val: Block 2 的日期
- test: Block 3 的日期
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from tqdm import tqdm
import os
import time
import ast
import re
import argparse
from pathlib import Path
from collections import defaultdict

# ==================== 配置 ====================
# 数据路径配置
DATA_DIR = "Datasets/KuaiRec 2.0/data"
RAW_DATA_DIR = "Datasets"

# big_matrix 时间块配置
BIG_MATRIX_BLOCKS = {
    "block1": {"start": "2020-07-05", "end": "2020-07-12", "name": "Block1"},
    "block2": {"start": "2020-08-01", "end": "2020-08-10", "name": "Block2"},
    "block3": {"start": "2020-08-27", "end": "2020-09-05", "name": "Block3"}
}

# 数据划分配置（big_matrix模式）
BIG_MATRIX_SPLITS = {
    "train": ["block1"],           # Block1 作为训练集
    "val": ["block2"],             # Block2 作为验证集
    "test": ["block3"]             # Block3 作为测试集
}

# small_matrix 时间窗口配置（63天连续）
SMALL_MATRIX_SPLITS = {
    "warmup_start": "2020-07-05",
    "warmup_end": "2020-07-18",
    "train_start": "2020-07-19",
    "train_end": "2020-08-22",
    "val_start": "2020-08-23",
    "val_end": "2020-08-29",
    "test_start": "2020-08-30",
    "test_end": "2020-09-05"
}

# 采样配置
MAX_USERS = 50  # 首版先用50个用户测试
MAX_HISTORY_PER_SAMPLE = 30  # 每个快照最多保留30条历史
MIN_INTERACTIONS = 10  # 最少有效交互数才生成样本

# 输出目录
OUTPUT_DIR = "LLM_part"


def load_local_env_file():
    """Best-effort local .env loader to avoid requiring shell exports for one-off runs."""
    env_candidates = [
        Path(".env.local"),
        Path(".env"),
    ]
    for env_path in env_candidates:
        if not env_path.exists():
            continue
        with env_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        return


def sanitize_output_tag(output_tag: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9_-]+", "_", (output_tag or "").strip())
    return tag.strip("_")


def rebalance_samples(
    samples: List[Dict],
    max_samples: Optional[int] = None,
    max_windows_per_user: Optional[int] = None,
) -> List[Dict]:
    """
    按用户分组重排样本，避免简单截断导致样本集中在少数用户。

    规则：
    1. 先保持原有用户顺序与窗口顺序分组
    2. 可选限制每个用户最多保留多少个窗口
    3. 若设置 max_samples，则按用户 round-robin 轮转截断
    """
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    user_order: List[int] = []

    for sample in samples:
        user_id = int(sample["user_id"])
        if user_id not in grouped:
            user_order.append(user_id)
        grouped[user_id].append(sample)

    if max_windows_per_user is not None and max_windows_per_user > 0:
        for user_id in user_order:
            grouped[user_id] = grouped[user_id][:max_windows_per_user]

    if max_samples is None or max_samples <= 0:
        rebalanced = []
        for user_id in user_order:
            rebalanced.extend(grouped[user_id])
        return rebalanced

    result: List[Dict] = []
    offsets = {user_id: 0 for user_id in user_order}

    while len(result) < max_samples:
        added_this_round = False
        for user_id in user_order:
            current_offset = offsets[user_id]
            user_samples = grouped[user_id]
            if current_offset >= len(user_samples):
                continue
            result.append(user_samples[current_offset])
            offsets[user_id] += 1
            added_this_round = True
            if len(result) >= max_samples:
                break
        if not added_this_round:
            break

    return result


class KuaiRecDataLoader:
    """KuaiRec数据集加载器"""

    def __init__(self, matrix_type: str = "big"):
        """
        初始化数据加载器

        Args:
            matrix_type: "big" 或 "small"
        """
        self.matrix_type = matrix_type
        self.matrix = None  # 统一使用 matrix 字段
        self.caption_category = None
        self.multi_category = None
        self.user_features = None
        self.observed_dates = None  # 观测到的日期集合

        # 缓存
        self._video_info_cache = {}
        self._user_profile_cache = {}

    def load_all(self):
        """加载所有数据"""
        print("=" * 60)
        print(f"加载KuaiRec数据集 ({self.matrix_type.upper()} matrix)...")
        print("=" * 60)

        # 1. 加载行为主表
        matrix_file = f"{DATA_DIR}/big_matrix.csv" if self.matrix_type == "big" else f"{DATA_DIR}/small_matrix.csv"
        print(f"\n[1/4] 加载 {os.path.basename(matrix_file)}...")
        self.matrix = pd.read_csv(matrix_file, nrows=None)  # 加载全部

        # 修复类型
        self.matrix['user_id'] = pd.to_numeric(self.matrix['user_id'], errors='coerce')
        self.matrix['video_id'] = pd.to_numeric(self.matrix['video_id'], errors='coerce')
        self.matrix = self.matrix.dropna(subset=['user_id', 'video_id', 'timestamp', 'date'])
        self.matrix['user_id'] = self.matrix['user_id'].astype(int)
        self.matrix['video_id'] = self.matrix['video_id'].astype(int)

        # 转换日期
        self.matrix['date'] = pd.to_datetime(self.matrix['date'], format='%Y%m%d')
        self.matrix['timestamp_dt'] = pd.to_datetime(self.matrix['timestamp'], unit='s')

        print(f"  原始记录数: {len(self.matrix):,}")
        print(f"  日期范围: {self.matrix['date'].min()} ~ {self.matrix['date'].max()}")

        # 提取观测日期集合
        self.observed_dates = sorted(self.matrix['date'].unique())
        print(f"  观测日期数: {len(self.observed_dates)}")
        print(f"  观测日期: {[str(d.date()) for d in self.observed_dates]}")

        # 2. 加载视频语义主表
        print("\n[2/4] 加载 kuairec_caption_category.csv...")
        try:
            self.caption_category = pd.read_csv(f"{RAW_DATA_DIR}/kuairec_caption_category.csv")
        except:
            self.caption_category = pd.read_csv(f"{DATA_DIR}/kuairec_caption_category.csv", engine='python', on_bad_lines='skip')

        # 修复video_id类型
        self.caption_category['video_id'] = pd.to_numeric(self.caption_category['video_id'], errors='coerce')
        self.caption_category = self.caption_category.dropna(subset=['video_id'])
        self.caption_category['video_id'] = self.caption_category['video_id'].astype(int)
        print(f"  视频数: {len(self.caption_category):,}")

        # 3. 加载视频分类补充表
        print("\n[3/4] 加载 video_raw_categories_multi.csv...")
        self.multi_category = pd.read_csv(f"{RAW_DATA_DIR}/video_raw_categories_multi.csv")
        self.multi_category['video_id'] = pd.to_numeric(self.multi_category['video_id'], errors='coerce')
        self.multi_category = self.multi_category.dropna(subset=['video_id'])
        self.multi_category['video_id'] = self.multi_category['video_id'].astype(int)
        print(f"  记录数: {len(self.multi_category):,}")

        # 4. 加载用户背景表
        print("\n[4/4] 加载 user_features_raw.csv...")
        self.user_features = pd.read_csv(f"{RAW_DATA_DIR}/user_features_raw.csv")
        self.user_features['user_id'] = pd.to_numeric(self.user_features['user_id'], errors='coerce')
        self.user_features = self.user_features.dropna(subset=['user_id'])
        self.user_features['user_id'] = self.user_features['user_id'].astype(int)
        print(f"  用户数: {len(self.user_features):,}")

        # 预处理视频信息
        self._build_video_info_cache()

        print(f"\n数据加载完成!")

    def _build_video_info_cache(self):
        """构建视频信息缓存"""
        print("\n构建视频信息缓存...")

        # 处理 caption_category
        for _, row in self.caption_category.iterrows():
            video_id = row['video_id']
            self._video_info_cache[video_id] = {
                'caption': row.get('caption', ''),
                'manual_cover_text': row.get('manual_cover_text', ''),
                'topic_tag': row.get('topic_tag', ''),
                'first_level_category_name': row.get('first_level_category_name', ''),
                'second_level_category_name': row.get('second_level_category_name', ''),
                'third_level_category_name': row.get('third_level_category_name', ''),
            }

        # 处理 multi_category
        multi_filtered = self.multi_category[
            (self.multi_category['category_online'] == 1)
        ].sort_values('prob', ascending=False)

        for video_id, group in multi_filtered.groupby('video_id'):
            if video_id not in self._video_info_cache:
                self._video_info_cache[video_id] = {}

            best = group.iloc[0]
            self._video_info_cache[video_id]['backup_category_path'] = f"{best['root_name']} > {best['parent_name']} > {best['category_name']}"
            self._video_info_cache[video_id]['backup_category_confidence'] = float(best['prob'])

        print(f"  缓存视频数: {len(self._video_info_cache):,}")

    def get_video_info(self, video_id: int) -> Dict:
        """获取视频信息"""
        return self._video_info_cache.get(video_id, {})

    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """获取用户背景信息"""
        if user_id in self._user_profile_cache:
            return self._user_profile_cache[user_id]

        row = self.user_features[self.user_features['user_id'] == user_id]
        if row.empty:
            return None

        row = row.iloc[0]

        # 安全地提取数值字段
        def safe_int(val, default=0):
            if pd.isna(val):
                return default
            try:
                return int(val)
            except:
                return default

        def safe_float(val, default=0.0):
            if pd.isna(val):
                return default
            try:
                return float(val)
            except:
                return default

        profile = {
            'user_id': int(user_id),
            'gender': str(row.get('gender', '')),
            'age_range': str(row.get('age_range', '')),
            'fre_city': str(row.get('fre_city', '')),
            'fre_city_level': str(row.get('fre_city_level', '')),
            'fre_community_type': str(row.get('fre_community_type', '')),
            'phone_brand': str(row.get('phone_brand', '')),
            'phone_model': str(row.get('phone_model', '')),
            'mod_price': safe_float(row.get('mod_price', 0)),
            'platform': str(row.get('platform', '')),
            'register_days': safe_int(row.get('register_days', 0)),
        }

        self._user_profile_cache[user_id] = profile
        return profile


class TimeWindowSampler:
    """时间窗口采样器 - 支持 big_matrix 和 small_matrix"""

    def __init__(self, data_loader: KuaiRecDataLoader):
        self.data_loader = data_loader

    def get_anchor_dates_for_split(self, split: str) -> List[pd.Timestamp]:
        """
        获取指定划分的所有可用 anchor_date

        big_matrix: 基于观测日期块
        small_matrix: 基于连续日期
        """
        if self.data_loader.matrix_type == "big":
            # big_matrix 模式：基于时间块
            blocks = BIG_MATRIX_SPLITS.get(split, [])
            anchor_dates = []

            for block_name in blocks:
                block = BIG_MATRIX_BLOCKS[block_name]
                start_dt = pd.to_datetime(block['start'])
                end_dt = pd.to_datetime(block['end'])

                # 筛选在观测日期中的日期
                for dt in self.data_loader.observed_dates:
                    if start_dt <= dt <= end_dt:
                        anchor_dates.append(dt)

            return sorted(anchor_dates)

        else:
            # small_matrix 模式：基于连续日期
            if split == "train":
                start = SMALL_MATRIX_SPLITS["train_start"]
                end = SMALL_MATRIX_SPLITS["train_end"]
            elif split == "val":
                start = SMALL_MATRIX_SPLITS["val_start"]
                end = SMALL_MATRIX_SPLITS["val_end"]
            else:  # test
                start = SMALL_MATRIX_SPLITS["test_start"]
                end = SMALL_MATRIX_SPLITS["test_end"]

            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)

            return [dt for dt in self.data_loader.observed_dates if start_dt <= dt <= end_dt]

    def get_valid_history_dates(self, anchor_date: pd.Timestamp) -> List[pd.Timestamp]:
        """
        获取某个 anchor_date 之前的所有有效历史日期

        对于 big_matrix：只返回更早的观测日期
        对于 small_matrix：返回所有更早的日期
        """
        valid_dates = []
        for dt in self.data_loader.observed_dates:
            if dt < anchor_date:
                valid_dates.append(dt)
        return valid_dates

    def generate_samples(
        self,
        max_users: int = MAX_USERS,
        split: str = "train"
    ) -> List[Dict]:
        """
        生成时间快照样本
        """
        print(f"\n生成 {split} 样本 ({self.data_loader.matrix_type} matrix)...")

        # 获取该 split 的所有 anchor_date
        anchor_dates = self.get_anchor_dates_for_split(split)
        print(f"  可用anchor_date数: {len(anchor_dates)}")

        if not anchor_dates:
            print(f"  警告: {split} 没有可用的anchor_date!")
            return []

        print(f"  anchor_date范围: {min(anchor_dates).date()} ~ {max(anchor_dates).date()}")

        # 筛选用户（选择交互最多的用户）
        user_interaction_count = self.data_loader.matrix.groupby('user_id').size()
        top_users = user_interaction_count.nlargest(max_users).index.tolist()
        print(f"  选取交互最多的 {len(top_users)} 个用户")

        # 筛选数据
        df = self.data_loader.matrix[
            self.data_loader.matrix['user_id'].isin(top_users)
        ].copy()

        samples = []

        for user_id in tqdm(top_users, desc=f"处理用户 ({split})"):
            user_samples = self._generate_user_samples(
                user_id, df, anchor_dates
            )
            samples.extend(user_samples)

        print(f"  生成样本数: {len(samples)}")
        return samples

    def _generate_user_samples(
        self,
        user_id: int,
        df: pd.DataFrame,
        anchor_dates: List[pd.Timestamp]
    ) -> List[Dict]:
        """为单个用户生成样本"""
        user_data = df[df['user_id'] == user_id].sort_values('timestamp')

        # 获取用户profile
        user_profile = self.data_loader.get_user_profile(user_id)
        if user_profile is None:
            return []

        samples = []

        # 遍历每个anchor_date
        for anchor_date in anchor_dates:
            # 历史：该anchor_date之前的所有观测日期的数据
            history = user_data[user_data['date'] < anchor_date].copy()

            # 标签：该anchor_date当天的数据
            label_day = user_data[user_data['date'] == anchor_date].copy()

            # 检查是否满足条件
            if len(history) >= MIN_INTERACTIONS:
                sample = self._build_sample(
                    user_id=user_id,
                    anchor_date=anchor_date,
                    history=history,
                    label_day=label_day,
                    user_profile=user_profile
                )
                if sample:
                    samples.append(sample)

        return samples

    def _build_sample(
        self,
        user_id: int,
        anchor_date: pd.Timestamp,
        history: pd.DataFrame,
        label_day: pd.DataFrame,
        user_profile: Dict
    ) -> Optional[Dict]:
        """构建单个样本"""
        # 派生watch_signal字段
        def get_watch_signal(watch_ratio):
            if watch_ratio >= 1.0:
                return "强正反馈/可能重复观看"
            elif watch_ratio >= 0.8:
                return "正反馈/较完整消费"
            elif watch_ratio >= 0.6:
                return "弱正反馈/有一定兴趣"
            else:
                return "浅层浏览/弱兴趣"

        # 派生finish_flag字段
        def get_finish_flag(row):
            return row['play_duration'] >= row['video_duration']

        # 构造历史交互列表
        history_interactions = []

        # 取最近的历史
        recent_history = history.tail(MAX_HISTORY_PER_SAMPLE)

        for _, row in recent_history.iterrows():
            video_id = int(row['video_id'])
            video_info = self.data_loader.get_video_info(video_id)

            # 构造视频语义文本
            caption = video_info.get('caption', '')
            cover = video_info.get('manual_cover_text', '')

            if not isinstance(caption, str):
                caption = ''
            if not isinstance(cover, str):
                cover = ''

            item_text = caption if caption else cover
            if item_text == 'UNKNOWN' or not item_text:
                item_text = ''

            # 构造分类路径
            category_parts = [
                video_info.get('first_level_category_name', ''),
                video_info.get('second_level_category_name', ''),
                video_info.get('third_level_category_name', '')
            ]
            category_path = ' > '.join([p for p in category_parts if p])

            backup_category = video_info.get('backup_category_path', '')

            # 解析topic_tag
            topic_tags = []
            tag_str = video_info.get('topic_tag', '')
            if tag_str:
                try:
                    topic_tags = ast.literal_eval(tag_str) if tag_str.startswith('[') else [tag_str]
                except:
                    topic_tags = [tag_str] if tag_str else []

            interaction = {
                'video_id': video_id,
                'timestamp': row['timestamp_dt'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['timestamp_dt']) else '',
                'watch_ratio': round(float(row['watch_ratio']), 3),
                'watch_signal': get_watch_signal(row['watch_ratio']),
                'play_duration': int(row['play_duration']),
                'video_duration': int(row['video_duration']),
                'finish_flag': get_finish_flag(row),
                'item_text': item_text[:200] if item_text else '',
                'topic_tags': topic_tags,
                'category_path': category_path,
                'backup_category_path': backup_category
            }

            history_interactions.append(interaction)

        # 构造标签
        label_day_items = label_day['video_id'].tolist() if not label_day.empty else []

        # 构建完整样本
        sample = {
            'sample_id': f"u{user_id}_d{anchor_date.strftime('%Y%m%d')}",
            'user_id': user_id,
            'anchor_date': anchor_date.strftime('%Y-%m-%d'),
            'history_start': history['date'].min().strftime('%Y-%m-%d') if not history.empty else '',
            'history_end': history['date'].max().strftime('%Y-%m-%d') if not history.empty else '',
            'user_profile': user_profile,
            'history_interactions': history_interactions,
            'label_day_items': [int(x) for x in label_day_items]
        }

        return sample


class TeacherPromptBuilder:
    """Teacher Prompt构建器"""

    def __init__(self, data_loader: KuaiRecDataLoader):
        self.data_loader = data_loader

    def build_context(self, sample: Dict) -> str:
        """将样本转换为LLM可理解的自然语言上下文"""
        profile = sample['user_profile']

        profile_text = f"""用户基本信息:
- 性别: {profile.get('gender', '未知')}
- 年龄段: {profile.get('age_range', '未知')}
- 城市: {profile.get('fre_city', '未知')}
- 城市等级: {profile.get('fre_city_level', '未知')}
- 手机品牌: {profile.get('phone_brand', '未知')}
- 手机价格区间: {profile.get('mod_price', 0)}元
- 注册天数: {profile.get('register_days', 0)}天"""

        interactions_text = "\n用户历史观看记录 (按时间倒序):\n"

        for i, interaction in enumerate(sample['history_interactions'], 1):
            interactions_text += f"""
{i}. 视频ID: {interaction['video_id']}
   观看时间: {interaction['timestamp']}
   观看比例: {interaction['watch_ratio']:.1%} ({interaction['watch_signal']})
   {'✅ 完整看完' if interaction['finish_flag'] else '❌ 未完整看完'}
   视频内容: {interaction['item_text'][:100] if interaction['item_text'] else '无描述'}
   话题标签: {', '.join(interaction['topic_tags']) if interaction['topic_tags'] else '无'}
   分类路径: {interaction['category_path'] or '无'}"""

        label_text = f"""
当日观看视频 (用于参考用户可能的兴趣):
{', '.join([str(x) for x in sample['label_day_items']])} 或 [当日无观看]"""

        context = f"""请分析以下用户的兴趣偏好和行为模式。

{profile_text}
{interactions_text}
{label_text}

请基于以上信息，分析用户的：
1. 长期兴趣意图
2. 当前心理需求
3. 生活阶段
4. 潜在兴趣增长点
5. 检索建议关键词

请严格按照JSON格式输出分析结果。"""

        return context

    def build_messages(self, sample: Dict, system_prompt: str) -> List[Dict]:
        """构建API消息格式"""
        context = self.build_context(sample)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        return messages


# ==================== API调用部分 ====================
from openai import OpenAI


class TeacherInferenceGenerator:
    """Teacher推理生成器"""

    SYSTEM_PROMPT = """
你是专业的用户行为分析专家，兼具严谨的逻辑推理能力和深刻的心理学洞察力。你擅长从用户看似矛盾、碎片化的行为中，梳理出连贯的意图脉络，并挖掘其行为背后的情感补偿机制与潜在需求。

## 任务
基于用户的历史交互记录（包括观看的视频、交互时间、观看时长）以及视频的分类信息，对用户的状态进行深度推理分析。

## 分析原则
1. **辩证包容**：揭示行为的主次关系与内在逻辑，用描述性语言包容多元兴趣。
2. **频率与时效**：区分"长期稳定的倾向"与"近期突发的需求"。
3. **阶段明确性**：若有强信号（密集连贯行为，例如持续关注和消费装修、备孕、考研、求职等等话题），则给出明确阶段；否则归入"稳定期/泛兴趣探索期"。
4. **冷启动处理**：若有效互动少于3条，进行保守推断并注明数据不足。
5. **行为模式归纳**：基于交互时间和观看时长，自然归纳用户的内容消费和使用习惯，但不在输出中显式标注时间标签。

## 输出要求
请严格按照以下JSON格式输出，不要包含任何Markdown格式或额外解释：

{
  "user_basic_information": {
    "user_id": "用户的ID，从输入中提取",
    "user_active_degree": "用户的活跃度，从输入中提取",
    "gender": "用户的性别，从输入中提取"
  },

  "user_status_analysis": {
    "long_term_intent": {
      "description": "遵循'兴趣星系'原则，用一段整合性描述概括用户兴趣全景。建议结构：'用户当前最突出的兴趣是...（核心兴趣簇）。同时，也稳定关注...（次要兴趣簇）。近期，表现出对...的新关注或出现了连接...与...的兴趣苗头（兴趣生长点）。'"
    },
    "psychological_demand": {
      "core_demand": "总结用户使用本平台最根本的心理角色，如'核心的知识获取渠道'、'首选的休闲娱乐方式'或'重要的生活灵感来源'。",
      "immediate_need": "分别判断用户'长期稳定'的和'近期凸显'（基于近期高频行为推断出的）心理需求。若无近期凸显，则只描述长期稳定的心理诉求即可。"
    },
    "life_stage_hypothesis": {
      "stage": "推断的生活阶段。若无强信号，则为'稳定期/泛兴趣探索期'。",
      "confidence": "对此推断的置信度，分为高、中、低。",
      "key_attributes": ["仅在置信度为'高'时，提供1-3个最能定义当前阶段的关键标签（如'备考'、'装修'、'学习冲刺期'）；否则为空数组[]。"]
    },
    "interest_growth_points": {
      "emerging_signals": ["【用于泛化发散】识别用户尚未深度涉足、但与当前兴趣存在逻辑关联的'潜在领域'。禁止直接列举已看过的具体物品，必须是新的类目或话题（如看了编程 -> 推荐'开源社区文化'）。"],
      "bridge_concepts": ["【负责探索和抽象】提炼能够连接用户看似不相关兴趣点的'高阶思维模型'或'生活方式概念'（如连接健身与效率 -> '量化生活'）。"]
    }
  },
  "retrieval_suggestions": {
    "explicit_queries": ["【专注精准精准】推断用户下一步最可能直接在搜索框输入的2-4个'具体名词词组'。要求：可直接匹配数据库标题，落地性强（如'Python爬虫源码'）。"],
    "implicit_keywords": ["【兴趣召回标签】挖掘用户行为背后隐含的、用于扩充召回范围的2-4个'核心标签'。侧重于具体的技能点或内容属性（如看'Python教程' -> 标签'后端开发'、'职场技能'）。"]
  }
}

## 重要说明
- 你的分析应该基于用户的历史行为数据
- 保持推理的连贯性和逻辑性
- 输出必须是严格的JSON格式
- 所有字段都必须包含，即使某些字段内容可能较少
- 适当拆解问题，进行深入思考和分析，兼顾逻辑性、准确性和发散性
"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, messages: List[Dict]) -> Dict:
        """调用API生成推理结果"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3,
                max_tokens=6000
            )

            content = response.choices[0].message.content

            # 清理markdown
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            return {"error": f"JSON解析失败: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}


def generate_teacher_data(
    api_key: str,
    matrix_type: str = "big",
    max_users: int = MAX_USERS,
    split: str = "train",
    delay: float = 1.0,
    resume: bool = True,
    max_samples: Optional[int] = None,
    max_windows_per_user: Optional[int] = None,
    output_tag: str = "",
):
    """
    生成Teacher数据的完整流程
    """
    print("=" * 60)
    print(f"KuaiRec Teacher数据生成 ({matrix_type.upper()} matrix)")
    print("=" * 60)

    # 1. 加载数据
    data_loader = KuaiRecDataLoader(matrix_type=matrix_type)
    data_loader.load_all()

    # 2. 生成时间窗口样本
    sampler = TimeWindowSampler(data_loader)
    samples = sampler.generate_samples(max_users=max_users, split=split)

    if not samples:
        print("没有生成任何样本!")
        return

    samples = rebalance_samples(
        samples,
        max_samples=max_samples,
        max_windows_per_user=max_windows_per_user,
    )

    unique_user_count = len({int(sample["user_id"]) for sample in samples})
    print(f"重排后样本数: {len(samples)}")
    print(f"重排后用户数: {unique_user_count}")
    if max_windows_per_user is not None and max_windows_per_user > 0:
        print(f"每用户最多窗口数: {max_windows_per_user}")

    file_suffix = f"{matrix_type}_{split}"
    clean_tag = sanitize_output_tag(output_tag)
    if clean_tag:
        file_suffix = f"{file_suffix}_{clean_tag}"

    # 保存时间窗口样本
    time_samples_file = f"{OUTPUT_DIR}/time_window_samples_{file_suffix}.jsonl"
    print(f"\n保存时间窗口样本到: {time_samples_file}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(time_samples_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # 3. 构建prompt并调用API
    print("\n构建API请求并调用...")
    prompt_builder = TeacherPromptBuilder(data_loader)

    contexts_file = f"{OUTPUT_DIR}/user_contexts_{file_suffix}.jsonl"
    print(f"保存 context 文件到: {contexts_file}")
    with open(contexts_file, 'w', encoding='utf-8') as f_ctx:
        for sample in samples:
            context = prompt_builder.build_context(sample)
            f_ctx.write(
                json.dumps(
                    {
                        'sample_id': sample['sample_id'],
                        'user_id': sample['user_id'],
                        'anchor_date': sample['anchor_date'],
                        'context': context,
                    },
                    ensure_ascii=False,
                ) + '\n'
            )

    # 检查已完成的样本
    inferences_file = f"{OUTPUT_DIR}/user_inferences_{file_suffix}.jsonl"
    completed_samples = set()
    if resume and os.path.exists(inferences_file):
        print("检测到已有结果文件，开启断点续跑模式...")
        with open(inferences_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    sample_id = result.get('sample_id', '')
                    if sample_id:
                        completed_samples.add(sample_id)
                except:
                    pass
        print(f"  已完成样本数: {len(completed_samples)}")

    # 4. 调用API生成
    print("\n调用API生成Teacher数据...")
    generator = TeacherInferenceGenerator(api_key=api_key)

    with open(inferences_file, 'a', encoding='utf-8') as f_out:
        for sample in tqdm(samples, desc="生成Teacher数据"):
            sample_id = sample['sample_id']

            if sample_id in completed_samples:
                continue

            messages = prompt_builder.build_messages(
                sample,
                TeacherInferenceGenerator.SYSTEM_PROMPT
            )

            result = generator.generate(messages)

            result['sample_id'] = sample_id
            result['user_id'] = sample['user_id']
            result['anchor_date'] = sample['anchor_date']
            result['generated_time'] = time.strftime("%Y-%m-%d %H:%M:%S")

            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

            time.sleep(delay)

    print(f"\nTeacher数据生成完成!")
    print(f"  时间窗口样本: {time_samples_file}")
    print(f"  推理结果: {inferences_file}")


# ==================== 主函数 ====================
if __name__ == "__main__":
    load_local_env_file()
    parser = argparse.ArgumentParser(description="KuaiRec Teacher数据生成")
    parser.add_argument("--api_key", type=str, default="", help="DeepSeek API密钥；未传时尝试读取环境变量 DEEPSEEK_API_KEY / API_KEY")
    parser.add_argument("--matrix", type=str, default="big", choices=["big", "small"], help="矩阵类型")
    parser.add_argument("--max_users", type=int, default=MAX_USERS, help="最大用户数")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="数据划分")
    parser.add_argument("--delay", type=float, default=1.0, help="API调用延迟(秒)")
    parser.add_argument("--no_resume", action="store_true", help="禁用断点续跑")
    parser.add_argument("--max_samples", type=int, default=0, help="最多生成多少条样本，0表示不限制")
    parser.add_argument("--max_windows_per_user", type=int, default=0, help="每个用户最多保留多少个时间窗口，0表示不限制")
    parser.add_argument("--output_tag", type=str, default="", help="输出文件附加标签，便于区分不同实验批次")

    args = parser.parse_args()
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("未提供 DeepSeek API 密钥。请使用 --api_key，或设置 DEEPSEEK_API_KEY / API_KEY 环境变量。")

    generate_teacher_data(
        api_key=api_key,
        matrix_type=args.matrix,
        max_users=args.max_users,
        split=args.split,
        delay=args.delay,
        resume=not args.no_resume,
        max_samples=(args.max_samples if args.max_samples > 0 else None),
        max_windows_per_user=(args.max_windows_per_user if args.max_windows_per_user > 0 else None),
        output_tag=args.output_tag,
    )
