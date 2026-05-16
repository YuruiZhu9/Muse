# 数据集使用方法——codex版

## 1. 文档说明

这份文档用于说明当前项目中 KuaiRec 数据集的实际使用方式，重点不是理论介绍，而是：

- 项目里哪些 CSV 真正参与了链路
- 各个文件之间如何对齐
- 用户历史行为如何组织成可供 LLM 使用的上下文
- 这些数据最终如何进入推荐系统 pipeline

如果你要重新开始，这份文档可以和 `项目实践方案.md` 配合阅读：

- `项目实践方案.md`：更偏向整体工程链路
- `数据集使用方法——codex版.md`：更偏向数据文件与数据组织方式

---

## 2. 当前项目实际使用的数据文件

### 2.1 用户交互主表

#### `Datasets/KuaiRec 2.0/data/big_matrix.csv`

这是当前主用的交互表。

使用字段：

- `user_id`
- `video_id`
- `play_duration`
- `video_duration`
- `time`
- `date`
- `timestamp`
- `watch_ratio`

用途：

- 构造用户历史行为序列
- 生成 LLM 输入上下文
- 作为推荐系统用户历史来源

处理规则：

- 先过滤 `time`、`date`、`timestamp` 为空的记录
- 再按 `user_id` 分组
- 每个用户内部按 `timestamp` 升序排列

---

### 2.2 用户画像表

#### `Datasets/user_features_raw.csv`

这是给 LLM 用的主画像表。

优先字段：

- `user_id`
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

作用：

- 补全用户背景
- 作为 prompt 中的 `user_basic_information` 和用户画像上下文来源

如果字段缺失：

- 直接跳过
- 不编造

---

### 2.3 推荐模型侧用户特征表

#### `Datasets/KuaiRec 2.0/data/user_features.csv`

这张表不是给 LLM 写 prompt 的，而是给推荐模型侧构造结构化数值特征。

作用：

- 进入 `src/data_loader_kuairec.py`
- 构造 `UserFeatures.feature_vector`

也就是说：

- `user_features_raw.csv` 负责“人类可读画像”
- `user_features.csv` 负责“模型可用数值特征”

---

### 2.4 视频语义主表

#### `Datasets/KuaiRec 2.0/data/kuairec_caption_category.csv`

这是 item 语义信息的主来源。

优先读取字段：

- `caption`
- `manual_cover_text`
- `topic_tag`
- `first_level_category_name`
- `second_level_category_name`
- `third_level_category_name`

语义优先级：

1. 优先 `caption`
2. 如果 `caption` 为空，则退到 `manual_cover_text`
3. 如果 `manual_cover_text = UNKNOWN`，则视为缺失
4. 类别路径优先用三级分类字段拼接

生成 item 主语义时，通常组合为：

- 文本描述
- 主类别路径
- topic_tag

---

### 2.5 视频分类补充表

#### `Datasets/video_raw_categories_multi.csv`

这是一个一对多表，用于在主表文本信息不强时补充类别语义。

处理规则：

- 先筛 `category_online = 1`
- 再按 `prob` 取较高记录
- 按 `video_id` 聚合
- 拼成：
  - `root_name > parent_name > category_name`

作用：

- 补充 item 语义
- 主表已有优质 caption 时，它只作为补充，不喧宾夺主

---

### 2.6 视频日级统计表

#### `Datasets/KuaiRec 2.0/data/item_daily_features.csv`

当前主要使用：

- `video_duration`

作用：

- 给推荐侧构造 `ItemFeatures`
- 作为交互解释中的视频时长信息补充

---

## 3. 当前不作为主输入的数据文件

以下文件当前不是主链路必要输入：

- `Datasets/KuaiRec 2.0/data/social_network.csv`
- `Datasets/KuaiRec 2.0/data/item_categories.csv`

说明：

- 这些表不是完全没价值
- 只是当前 student / teacher / embedding / pipeline 主链路不依赖它们
- 在重新开始阶段，可以先不引入，避免复杂度过高

---

## 4. item 信息如何对齐

对每条用户交互中的 `video_id`，当前项目采用如下对齐方式：

### 第一优先级：主语义表

从 `kuairec_caption_category.csv` 获取：

- `caption`
- `manual_cover_text`
- `first_level_category_name`
- `second_level_category_name`
- `third_level_category_name`
- `topic_tag`

### 第二优先级：补充类别表

当主表文本缺失、过短、或类别信息不足时，从 `video_raw_categories_multi.csv` 补充：

- `root_name`
- `parent_name`
- `category_name`

### 实际给模型看的 item 结构

在当前实现中，一条交互里的视频信息通常组织成：

- 视频 ID
- 交互时间
- 观看深度
- 标题/描述
- 分类路径
- 补充标签

---

## 5. watch_ratio 的用法

当前项目不会只传原始数值，而是会同时传：

- 原始 `watch_ratio`
- 行为强度语义标签

标签规则：

- `watch_ratio >= 1.0`
  - `强正反馈 / 可能重复观看`
- `0.8 <= watch_ratio < 1.0`
  - `正反馈 / 较完整消费`
- `0.6 <= watch_ratio < 0.8`
  - `弱正反馈 / 有一定兴趣`
- `watch_ratio < 0.6`
  - `浅层浏览 / 弱兴趣`

作用：

- 让 LLM 区分不同强度的行为信号
- 帮助 teacher / student 生成更稳定的兴趣摘要

---

## 6. 用户历史行为如何组织

当前不建议把所有历史逐条原样展开给 student。

更合理的做法是两层结构：

### 6.1 全历史摘要

保留全量统计信息，例如：

- 总交互数
- 覆盖天数
- 首次交互时间
- 最近交互时间
- `watch_ratio >= 0.6 / 0.8 / 1.0` 的数量

### 6.2 最近窗口明细

只保留最近一段时间的高价值交互明细，例如：

- 最近 `20~60` 条
- 仍按时间顺序排列
- 保留不同强度的 `watch_ratio`

这样做的原因：

- 不会把上下文撑爆
- 保留长期兴趣的统计轮廓
- 还能保留近期兴趣漂移和短期意图

---

## 7. 当前项目中 teacher / student / embedding 的数据关系

### 7.1 teacher

teacher 负责：

- 读取真实 KuaiRec 数据
- 基于 prompt 生成用户兴趣推断 JSON

teacher 的输入是：

- 用户画像
- 历史行为序列
- watch_ratio 语义标签
- item 文本/类别语义

teacher 的输出是：

- 结构化用户兴趣 JSON

### 7.2 student

student 当前建议输出 compact 版本，只保留 5 个核心语义字段：

- `long_term_intent`
- `life_stage`
- `psychological_demand`
- `retrieval_suggestions`
- `interest_growth_points`

原因：

- 更适合小模型稳定生成
- 更适合后续 embedding 编码
- 更容易规模化批量推理

### 7.3 embedding

embedding 阶段会对两类文本做编码：

- 用户侧 5 个语义字段
- item 侧文本与类别信息

编码维度：

- `2560`

当前项目中已经支持：

- 临时 hash 编码后端
- 真实 `Qwen3-Embedding-4B` 后端

---

## 8. 数据侧最重要的实践原则

如果重新开始，最重要的不是一次性把所有文件都接进来，而是：

1. 先固定最核心的输入文件
2. 先固定用户上下文结构
3. 先固定 student 输出 schema
4. 先用小样本跑通
5. 再逐步扩量

所以当前推荐的最小必要数据子集是：

- `big_matrix.csv`
- `user_features_raw.csv`
- `user_features.csv`
- `kuairec_caption_category.csv`
- `video_raw_categories_multi.csv`
- `item_daily_features.csv`

这 6 个文件已经足够支撑当前主链路。

---

## 9. 当前重新开始时应保留的关键代码

建议保留：

- `LLM_part/LLMapi_for_generate.py`
- `LLM_part/generate_teacher_from_kuairec.py`
- `LLM_part/build_sft_dataset.py`
- `LLM_part/run_local_qwen_inference.py`
- `LLM_part/student_schema.py`
- `LLM_part/Embedding_Qwen4B/inference_embedding.py`
- `LLM_part/SFT_Qwen1.7b/finetune_qwen4b.py`
- `LLM_part/SFT_Qwen1.7b/inference_lora.py`
- `LLM_part/SFT_Qwen1.7b/qwen35_4b_finetuned`
- `src/data_loader_kuairec.py`
- `src/recall.py`
- `src/ranking.py`
- `run_kuairec_llm_smoke.py`

---

## 10. 最后说明

这份文档是恢复版指导文档。

后续如果继续整理项目，应遵循一个原则：

> 指导性文件、架构说明文件、核心源码、必要模型和必要 smoke 脚本应该保留；  
> 缓存、临时样本、批量生成结果、中间 JSONL 和一次性测试文件可以随时清理。
