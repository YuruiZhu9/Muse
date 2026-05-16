# Prompt 使用说明（精简正式版）

## 1. 目标

这份文档用于说明在当前项目里，如何基于 KuaiRec 数据生成用户兴趣推断结果，并与后续 student / embedding / 推荐 pipeline 对齐。

它不是训练脚本，也不是测试日志，而是一个**长期保留的指导性说明文件**。

---

## 2. 当前推荐的生成目标

当前项目建议将用户兴趣结果统一为 compact 结构，便于：

- student 微调
- 本地推理
- embedding 编码
- 推荐系统接入

推荐输出结构：

```json
{
  "user_basic_information": {
    "user_id": "0",
    "user_active_degree": "high_active",
    "gender": "F"
  },
  "long_term_intent": "...",
  "life_stage": "...",
  "psychological_demand": "...",
  "retrieval_suggestions": "...",
  "interest_growth_points": "...",
  "generated_time": "YYYY-MM-DD HH:MM:SS"
}
```

要求：

- `user_id` 只出现在 `user_basic_information` 内
- `generated_time` 必须位于最后
- 5 个语义字段全部为字符串
- 不输出 Markdown，不输出解释性前言

---

## 3. 数据使用规则

### 3.1 用户交互主表

使用：

- `Datasets/KuaiRec 2.0/data/big_matrix.csv`

字段：

- `user_id`
- `video_id`
- `play_duration`
- `video_duration`
- `time`
- `date`
- `timestamp`
- `watch_ratio`

规则：

- 过滤 `time/date/timestamp` 为空的记录
- 按 `user_id` 分组
- 每个用户按 `timestamp` 升序排列

### 3.2 用户画像主表

使用：

- `Datasets/user_features_raw.csv`

优先字段：

- `user_active_degree`
- `gender`
- `age_range`
- `follow_user_num`
- `fans_user_num`
- `friend_user_num`
- `fre_province`
- `fre_city`
- `fre_city_level`
- `fre_community_type`
- `register_days`
- `phone_brand`
- `mod_price`

### 3.3 视频语义主表

使用：

- `Datasets/KuaiRec 2.0/data/kuairec_caption_category.csv`

优先级：

1. `caption`
2. `manual_cover_text`
3. 分类路径
4. `topic_tag`

### 3.4 分类补充表

使用：

- `Datasets/video_raw_categories_multi.csv`

规则：

- 取 `category_online = 1`
- 优先高 `prob`
- 聚合为：
  - `root_name > parent_name > category_name`

---

## 4. watch_ratio 标签规则

- `watch_ratio >= 1.0`
  - `强正反馈 / 可能重复观看`
- `0.8 <= watch_ratio < 1.0`
  - `正反馈 / 较完整消费`
- `0.6 <= watch_ratio < 0.8`
  - `弱正反馈 / 有一定兴趣`
- `watch_ratio < 0.6`
  - `浅层浏览 / 弱兴趣`

要求：

- 同时保留数值和语义标签
- 不破坏原始时间顺序

---

## 5. 当前实践建议

当前不要把用户的全量原始历史逐条塞进 student。

更推荐：

- 保留全历史统计摘要
- 再保留最近 `20~60` 条高价值明细

原因：

- 上下文长度更可控
- 更适合本地小模型推理
- 更利于 student 稳定输出固定结构

---

## 6. 当前相关代码

- 原始 teacher API 角色定义：`LLM_part/LLMapi_for_generate.py`
- 当前 student schema：`LLM_part/student_schema.py`
- SFT 数据构造：`LLM_part/build_sft_dataset.py`
- 本地推理：`LLM_part/run_local_qwen_inference.py`
- embedding：`LLM_part/Embedding_Qwen4B/inference_embedding.py`

---

## 7. 说明

如果后续需要再次精修 prompt，本文件应继续保留，不应视为临时测试文件删除。
