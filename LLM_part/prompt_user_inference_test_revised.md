# Prompt 使用说明（测试版）

## 1. 用途

这份文档用于小规模测试时快速核对：

- 输入数据来自哪些文件
- item 信息怎么拼
- watch_ratio 怎么解释
- 输出结构应该长什么样

它属于指导性文件，应该长期保留。

---

## 2. 小规模测试建议

测试时建议：

- 先选 `2~3` 个用户
- 每个用户保留 `20~30` 条有效交互
- 历史按 `timestamp` 升序排列
- 尽量覆盖不同强度的 `watch_ratio`

---

## 3. 数据文件

### 交互表

- `Datasets/KuaiRec 2.0/data/small_matrix.csv`

### 用户画像

- `Datasets/user_features_raw.csv`

### item 主语义

- `Datasets/KuaiRec 2.0/data/kuairec_caption_category.csv`

### item 补充类别

- `Datasets/video_raw_categories_multi.csv`

---

## 4. item 语义整合规则

优先级：

1. `caption`
2. `manual_cover_text`
3. `first/second/third_level_category_name`
4. `topic_tag`
5. `root_name > parent_name > category_name`

如果文本缺失：

- 可以只保留分类路径
- 不编造视频标题

---

## 5. 交互历史组织方式

建议每条交互包含：

- 视频 ID
- 交互时间
- 观看比例
- 观看比例语义标签
- 视频文本描述
- 分类路径
- 补充分类

---

## 6. watch_ratio 标签

- `>= 1.0`：强正反馈 / 可能重复观看
- `0.8 ~ 1.0`：正反馈 / 较完整消费
- `0.6 ~ 0.8`：弱正反馈 / 有一定兴趣
- `< 0.6`：浅层浏览 / 弱兴趣

---

## 7. 测试输出要求

测试阶段也建议统一到 compact 结构：

```json
{
  "user_basic_information": {
    "user_id": "...",
    "user_active_degree": "...",
    "gender": "..."
  },
  "long_term_intent": "...",
  "life_stage": "...",
  "psychological_demand": "...",
  "retrieval_suggestions": "...",
  "interest_growth_points": "...",
  "generated_time": "YYYY-MM-DD HH:MM:SS"
}
```

---

## 8. 相关文件

- `LLM_part/student_schema.py`
- `LLM_part/build_sft_dataset.py`
- `LLM_part/run_local_qwen_inference.py`
- `LLM_part/LLMapi_for_generate.py`

---

## 9. 说明

本文件用于测试场景下快速查阅规则，不属于可随时删除的中间文件。
