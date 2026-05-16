"""
生成KuaiRec big_matrix时间切片快照数据 - 优化版本
使用向量化操作提高效率
"""
import pandas as pd
import json
import numpy as np

# 配置
DATA_PATH = "Datasets/KuaiRec 2.0/data"
RAW_DATA_PATH = "Datasets"
OUTPUT_FILE = "LLM_part/time_window_samples.jsonl"

print("=" * 60)
print("步骤1: 加载数据")
print("=" * 60)

# 加载big_matrix
print("加载 big_matrix.csv...")
big_matrix = pd.read_csv(f"{DATA_PATH}/big_matrix.csv")
print(f"  总记录数: {len(big_matrix)}")

# 过滤空值
big_matrix = big_matrix.dropna(subset=['time', 'date', 'timestamp'])
big_matrix['date_int'] = big_matrix['date'].astype(int)
print(f"  过滤后: {len(big_matrix)}")

# 加载用户画像
print("加载 user_features_raw.csv...")
user_features = pd.read_csv(f"{RAW_DATA_PATH}/user_features_raw.csv")

# 构建用户画像字典
user_profile_dict = {}
for _, row in user_features.iterrows():
    uid = int(row['user_id'])
    user_profile_dict[uid] = {
        'user_active_degree': str(row.get('user_active_degree', '')),
        'gender': str(row.get('gender', '')),
        'age_range': str(row.get('age_range', '')),
        'follow_user_num': int(row.get('follow_user_num', 0)),
        'fans_user_num': int(row.get('fans_user_num', 0)),
        'fre_province': str(row.get('fre_province', '')),
        'fre_city': str(row.get('fre_city', '')),
        'register_days': int(row.get('register_days', 0))
    }

print("\n" + "=" * 60)
print("步骤2: 分析用户-日期分布")
print("=" * 60)

# 按用户和日期分组
user_date_groups = big_matrix.groupby(['user_id', 'date_int'])
print(f"  用户-日期组合数: {len(user_date_groups)}")

# 获取每个用户的有效日期列表
user_dates = big_matrix.groupby('user_id')['date_int'].apply(lambda x: sorted(x.unique()))
print(f"  用户数: {len(user_dates)}")

# 获取所有唯一日期
all_dates = sorted(big_matrix['date_int'].unique())
print(f"  唯一日期数: {len(all_dates)}")
print(f"  日期: {all_dates}")

# 预先按用户和时间排序
big_matrix = big_matrix.sort_values(['user_id', 'timestamp'])

print("\n" + "=" * 60)
print("步骤3: 生成时间切片样本")
print("=" * 60)

# 从哪个用户开始（修改这里以继续之前的进度）
START_USER = 0  # 从0开始
# START_USER = 3920  # 从这里继续

sample_count = 0
write_interval = 50000

# 使用更高效的方式处理
# 按用户分组处理
with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
    for user_id in user_dates.index:
        if user_id < START_USER:
            continue
        user_dates_list = user_dates[user_id]
        first_date = user_dates_list[0]

        # 获取该用户所有数据
        user_data = big_matrix[big_matrix['user_id'] == user_id]

        # 对每个可能的anchor_date
        for anchor_date in user_dates_list:
            if anchor_date <= first_date:
                continue

            # 分割历史和标签
            history_df = user_data[user_data['date_int'] < anchor_date]
            label_df = user_data[user_data['date_int'] == anchor_date]

            if len(history_df) == 0 or len(label_df) == 0:
                continue

            # 构建样本
            sample = {
                'sample_id': f"u{user_id}_d{anchor_date}",
                'user_id': int(user_id),
                'anchor_date': int(anchor_date),
                'history_start_date': int(history_df['date_int'].min()),
                'history_end_date': int(history_df['date_int'].max()),
                'history_count': len(history_df),
                'label_count': len(label_df),
                'user_profile': user_profile_dict.get(int(user_id), {}),
                'history_items': [],
                'label_items': []
            }

            # 历史交互 - 简化为只保留必要字段
            for _, row in history_df.iterrows():
                sample['history_items'].append({
                    'video_id': int(row['video_id']),
                    'watch_ratio': round(float(row['watch_ratio']), 4),
                    'play_duration': int(row['play_duration']),
                    'video_duration': int(row['video_duration']),
                    'date': int(row['date_int'])
                })

            # 标签交互
            for _, row in label_df.iterrows():
                sample['label_items'].append({
                    'video_id': int(row['video_id']),
                    'watch_ratio': round(float(row['watch_ratio']), 4),
                    'timestamp': float(row['timestamp'])
                })

            # 写入
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            sample_count += 1

            if sample_count % 10000 == 0:
                print(f"  已生成 {sample_count} 条样本...")

print(f"\n完成!")
print(f"  总样本数: {sample_count}")
print(f"  输出文件: {OUTPUT_FILE}")
