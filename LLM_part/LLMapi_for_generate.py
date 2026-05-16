import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
from typing import List, Dict, Any
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(env_file: str = ".env") -> None:
    """Load simple KEY=VALUE entries from the project .env file."""
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


class KuaiRecInferenceGenerator:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        use_mock_data: bool = False,
    ):
        """
        初始化DeepSeek API客户端

        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL
            use_mock_data: 是否使用模拟数据（用于测试API调用）
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

        # 加载分类数据（根据你的描述）
        self.category_data = self.load_category_data(use_mock_data=use_mock_data)

        self.system_prompt = """
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



    def load_category_data(self, use_mock_data: bool = False) -> Dict:
        """
        加载视频分类数据

        Args:
            use_mock_data: 是否使用模拟数据（用于测试，无需真实CSV文件）
        """
        if use_mock_data:
            print("⚠️ 使用模拟数据模式（无需CSV文件）")
            return self._generate_mock_category_data()

        # 尝试加载真实数据
        try:
            caption_df = pd.read_csv('kuairec_caption_category.csv')
            multi_category_df = pd.read_csv('video_raw_categories_multi.csv')
            print("✅ 成功加载真实CSV数据")
        except FileNotFoundError as e:
            print(f"⚠️ 未找到CSV文件: {e}")
            print("📝 自动切换到模拟数据模式进行测试")
            return self._generate_mock_category_data()

        # 构建视频ID到分类的映射
        category_map = {}

        # 处理kuairec_caption_category.csv
        for _, row in caption_df.iterrows():
            video_id = row['video_id']
            category_map[video_id] = {
                'first_level': row['first_level_category_name'],
                'second_level': row['second_level_category_name'],
                'third_level': row['third_level_category_name'],
                'caption': row.get('caption', ''),
                'tags': row.get('topic_tag', '')
            }

        # 处理multi-category数据
        for _, row in multi_category_df.iterrows():
            video_id = row['video_id']
            if video_id not in category_map:
                category_map[video_id] = {}
            # 可以添加多级分类信息

        return category_map

    def _generate_mock_category_data(self) -> Dict:
        """生成模拟的视频分类数据（用于测试API调用）- 备战考研+游戏+美食用户画像"""
        mock_categories = [
            # === 考研学习相关（核心兴趣簇）===
            {
                'video_id': 2001,
                'first_level': '知识',
                'second_level': '考研备考',
                'third_level': '英语学习',
                'caption': '考研英语核心词汇3500词详解，附记忆技巧和真题例句，高效备考必备！',
                'tags': '考研,英语,词汇,备考'
            },
            {
                'video_id': 2002,
                'first_level': '知识',
                'second_level': '考研备考',
                'third_level': '数学复习',
                'caption': '考研数学线性代数重点题型突破，老师手把手教你解题思路，干货满满！',
                'tags': '考研,数学,线性代数,解题技巧'
            },
            {
                'video_id': 2003,
                'first_level': '知识',
                'second_level': '考研备考',
                'third_level': '专业课',
                'caption': '计算机考研408数据结构精讲，二叉树遍历算法详解，带你轻松搞定难点！',
                'tags': '考研,计算机,408,数据结构'
            },
            {
                'video_id': 2004,
                'first_level': '知识',
                'second_level': '学习方法',
                'third_level': '时间管理',
                'caption': '考研党必看！如何高效规划每天12小时学习时间，番茄工作法实战经验分享。',
                'tags': '考研,时间管理,学习方法,效率'
            },
            {
                'video_id': 2005,
                'first_level': '知识',
                'second_level': '考研备考',
                'third_level': '政治复习',
                'caption': '2025考研政治马原部分核心考点梳理，选择题高频易错点分析。',
                'tags': '考研,政治,马原,考点梳理'
            },

            # === 游戏娱乐相关（情感补偿机制）===
            {
                'video_id': 3001,
                'first_level': '游戏',
                'second_level': '手机游戏',
                'third_level': '王者荣耀',
                'caption': '王者荣耀上分攻略：打野位意识教学，如何正确带节奏上王者！',
                'tags': '游戏,王者荣耀,攻略,打野'
            },
            {
                'video_id': 3002,
                'first_level': '游戏',
                'second_level': '主机游戏',
                'third_level': 'Switch游戏',
                'caption': '塞尔达传说王国之泪全神庙位置攻略，收集强迫症福音！',
                'tags': '游戏,Switch,塞尔达,攻略'
            },
            {
                'video_id': 3003,
                'first_level': '娱乐',
                'second_level': '游戏实况',
                'third_level': '搞笑解说',
                'caption': '游戏名场面集锦，这些操作看笑我了！原来游戏还能这么玩~',
                'tags': '游戏,搞笑,实况,解说'
            },
            {
                'video_id': 3004,
                'first_level': '游戏',
                'second_level': '电竞资讯',
                'third_level': '比赛解说',
                'caption': 'LPL夏季赛季后赛精彩回顾，这波团战配合简直绝了！',
                'tags': '游戏,电竞,LPL,比赛'
            },

            # === 美食相关（生活调剂/放松）===
            {
                'video_id': 4001,
                'first_level': '生活',
                'second_level': '美食制作',
                'third_level': '快手菜',
                'caption': '考研党宵夜推荐！5分钟搞定番茄鸡蛋面，简单好吃又解馋。',
                'tags': '美食,快手菜,宵夜,考研党'
            },
            {
                'video_id': 4002,
                'first_level': '生活',
                'second_level': '美食探店',
                'third_level': '校园周边',
                'caption': '大学门口隐藏美食盘点，这家炸鸡配酸奶简直绝配，复习完来一份！',
                'tags': '美食,探店,校园,炸鸡'
            },
            {
                'video_id': 4003,
                'first_level': '生活',
                'second_level': '美食制作',
                'third_level': '煲汤养生',
                'caption': '熬夜备考党必学！元气养生汤教程，补气养神提高学习效率。',
                'tags': '美食,煲汤,养生,备考'
            },
            {
                'video_id': 4004,
                'first_level': '生活',
                'second_level': '美食vlog',
                'third_level': '一人食',
                'caption': '考研独居党的一日三餐，简单营养不费时间，学习也要好好吃饭！',
                'tags': '美食,vlog,一人食,考研'
            },

            # === 其他相关内容 ===
            {
                'video_id': 5001,
                'first_level': '知识',
                'second_level': '职业规划',
                'third_level': '研究生生活',
                'caption': '读研真实体验分享：研究生三年到底是怎样的？学弟学妹必看！',
                'tags': '考研,研究生,生活分享,经验'
            },
            {
                'video_id': 5002,
                'first_level': '生活',
                'second_level': '宿舍生活',
                'third_level': '收纳整理',
                'caption': '宿舍书桌改造计划，打造沉浸式学习环境，备考效率翻倍！',
                'tags': '宿舍,收纳,学习环境,改造'
            }
        ]

        # 构建视频ID到分类的映射
        category_map = {}
        for item in mock_categories:
            video_id = item.pop('video_id')
            category_map[video_id] = item

        return category_map

    def prepare_user_context(self, user_id: int, user_features: Dict,
                             interaction_history: List[Dict]) -> str:
        """
        准备用户的上下文信息

        Args:
            user_id: 用户ID
            user_features: 用户特征数据
            interaction_history: 用户交互历史
        """
        context = f"用户ID: {user_id}\n\n"

        # 添加用户特征
        context += "=== 用户特征 ===\n"
        for key, value in user_features.items():
            context += f"{key}: {value}\n"

        # 添加交互历史
        context += "\n=== 交互历史 ===\n"
        for i, interaction in enumerate(interaction_history, 1):
            video_id = interaction['video_id']
            watch_time = interaction['watch_time']
            timestamp = interaction['timestamp']

            # 获取视频分类信息
            category_info = self.category_data.get(video_id, {})

            context += f"\n交互记录 {i}:\n"
            context += f"  视频ID: {video_id}\n"
            context += f"  观看时长: {watch_time}秒\n"
            context += f"  交互时间: {timestamp}\n"
            if category_info:
                context += f"  视频分类: {category_info.get('first_level', '未知')} -> "
                context += f"{category_info.get('second_level', '未知')} -> "
                context += f"{category_info.get('third_level', '未知')}\n"
                if category_info.get('caption'):
                    context += f"  视频描述: {category_info['caption'][:100]}...\n"

        return context

    def generate_inference(self, user_context: str) -> Dict:
        """
        调用API生成推理结果

        Args:
            user_context: 用户上下文信息
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,  # 使用 deepseek-chat 模型   reasoner
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_context + "\n\n请直接以JSON格式输出分析结果，不要包含任何其他文字说明。"}
                ],
                temperature=0.3,
                max_tokens=6000
            )

            # 获取响应内容
            content = response.choices[0].message.content

            # 清理markdown代码块标记（如果存在）
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]  # 去掉 ```json
            if content.startswith("```"):
                content = content[3:]  # 去掉 ```
            if content.endswith("```"):
                content = content[:-3]  # 去掉结尾的 ```
            content = content.strip()

            print(f"\n【清理后的JSON长度】: {len(content)} 字符")

            # 解析JSON响应
            result_json = json.loads(content)
            return result_json

        except json.JSONDecodeError as e:
            print(f"\n⚠️ JSON解析失败: {e}")
            print("💡 建议：检查API返回的内容是否为有效JSON格式")
            return {"error": f"JSON解析失败: {str(e)}"}
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return {"error": str(e)}

    def batch_process(self, user_data_file: str, output_file: str,
                      batch_size: int = 10, delay: float = 0.5):
        """
        批量处理用户数据

        Args:
            user_data_file: 用户数据文件路径
            output_file: 输出文件路径
            batch_size: 批量大小
            delay: API调用延迟（避免速率限制）
        """
        # 加载用户数据
        # 这里需要根据你的数据格式进行调整
        user_data = pd.read_csv(user_data_file)
        user_features = pd.read_csv('user_features_raw.csv')

        # 分组用户交互数据（假设每个用户有多条记录）
        user_groups = user_data.groupby('user_id')

        results = []

        for user_id, user_interactions in tqdm(user_groups, desc="处理用户数据"):
            # 获取用户特征
            user_feature_row = user_features[user_features['user_id'] == user_id]
            user_feature_dict = user_feature_row.iloc[0].to_dict() if not user_feature_row.empty else {}

            # 准备交互历史
            interactions = []
            for _, row in user_interactions.iterrows():
                interactions.append({
                    'video_id': row['video_id'],
                    'watch_time': row.get('watch_time', 0),
                    'timestamp': row.get('timestamp', '')
                })

            # 准备上下文
            context = self.prepare_user_context(user_id, user_feature_dict, interactions)

            # 生成推理
            inference_result = self.generate_inference(context)

            # 添加用户ID到结果
            inference_result['user_id'] = user_id

            results.append(inference_result)

            # 保存中间结果
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(inference_result, ensure_ascii=False) + '\n')

            # 延迟以避免速率限制
            time.sleep(delay)

        print(f"处理完成！结果已保存到: {output_file}")
        return results

    def generate_sample_prompt(self, user_id: int):
        """
        生成示例prompt用于调试
        """
        # 示例用户特征
        sample_features = {
            'user_id': user_id,
            'user_active_degree': 'full activate', #'high_active'、'full_active'、'middle_active'、'UNKNOWN'
            'gender': 'F',
            'age_range': '24-30',
            'follow_user_num': 386,
            'register_days': 327,
            'fre_city': '江门'
        }

        # 示例交互历史
        sample_interactions = [
            {
                'video_id': 2418,
                'watch_time': 120,
                'timestamp': '2024-01-01 10:30:00'
            },
            {
                'video_id': 1000,
                'watch_time': 45,
                'timestamp': '2024-01-01 11:15:00'
            }
        ]

        context = self.prepare_user_context(user_id, sample_features, sample_interactions)

        print("=== System Prompt ===")
        print(self.system_prompt)
        print("\n=== User Context ===")
        print(context)


    def test_api_call(self, user_id: int = 1):
        """
        测试API调用功能，实际调用DeepSeek API并输出完整结果

        用户画像：备战考研的大学生，同时消费游戏和美食内容
        - 考研学习是主要目标（高观看时长，白天时段）
        - 游戏是学习后的情感补偿和放松方式（深夜时段）
        - 美食是生活调剂（一日三餐相关）
        """
        print("=" * 60)
        print("开始测试API调用...")
        print("👤 测试用户画像：备战考研 + 游戏 + 美食")
        print("=" * 60)

        # 示例用户特征 - 符合考研党画像
        sample_features = {
            'user_id': user_id,
            'gender': 'M',  # 男性
            'age_range': '22-25',  # 考研典型年龄段
            'follow_user_num': 156,  # 关注数量不多，专注学习
            'register_days': 89,  # 注册3个月左右，可能是备考开始时注册
            'fre_city': '武汉'  # 高校集中城市
        }

        # 示例交互历史 - 体现"备战考研+游戏+美食"的行为模式
        sample_interactions = [
            # === 上午学习时段 ===
            {
                'video_id': 2001,  # 考研英语词汇
                'watch_time': 285,  # 观看近5分钟，学习投入度高
                'timestamp': '2024-10-15 08:30:00'  # 早上学习时间
            },
            {
                'video_id': 2002,  # 考研数学线性代数
                'watch_time': 520,  # 观看近9分钟，深度学习
                'timestamp': '2024-10-15 10:15:00'
            },
            # === 午休时段 ===
            {
                'video_id': 4001,  # 快手菜教程
                'watch_time': 98,  # 边看边准备午饭
                'timestamp': '2024-10-15 12:00:00'
            },
            # === 下午学习时段 ===
            {
                'video_id': 2003,  # 数据结构专业课
                'watch_time': 450,  # 深度学习专业课
                'timestamp': '2024-10-15 14:20:00'
            },
            {
                'video_id': 2004,  # 时间管理学习方法
                'watch_time': 380,  # 学习如何学习
                'timestamp': '2024-10-15 16:00:00'
            },
            # === 晚饭时段 ===
            {
                'video_id': 4002,  # 探店视频
                'watch_time': 120,  # 放松看探店
                'timestamp': '2024-10-15 18:30:00'
            },
            # === 深夜时段 - 游戏娱乐（学习后的情感补偿）===
            {
                'video_id': 3001,  # 王者荣耀攻略
                'watch_time': 185,  # 看完整个攻略视频
                'timestamp': '2024-10-15 22:30:00'  # 深夜娱乐时间
            },
            {
                'video_id': 3002,  # Switch游戏攻略
                'watch_time': 320,  # 深夜放松时段
                'timestamp': '2024-10-15 23:15:00'
            },
            # === 第二天继续学习 ===
            {
                'video_id': 2005,  # 考研政治
                'watch_time': 265,  # 早上政治复习
                'timestamp': '2024-10-16 08:00:00'
            },
            {
                'video_id': 5001,  # 研究生生活分享
                'watch_time': 420,  # 了解研究生生活，坚定考研目标
                'timestamp': '2024-10-16 09:30:00'
            },
            # === 学习间隙放松 ===
            {
                'video_id': 4004,  # 一人食vlog
                'watch_time': 156,  # 学习间隙看美食vlog放松
                'timestamp': '2024-10-16 12:30:00'
            },
            # === 晚上学习后游戏放松 ===
            {
                'video_id': 3003,  # 游戏搞笑解说
                'watch_time': 145,  # 学习后的纯娱乐
                'timestamp': '2024-10-16 21:45:00'
            },
            {
                'video_id': 5002,  # 宿舍书桌改造
                'watch_time': 290,  # 打造学习环境
                'timestamp': '2024-10-16 15:00:00'
            }
        ]

        # 准备上下文
        context = self.prepare_user_context(user_id, sample_features, sample_interactions)

        print("\n【发送给API的上下文信息】")
        print("-" * 40)
        print(context[:500] + "..." if len(context) > 500 else context)
        print("-" * 40)

        print("\n正在调用DeepSeek API...")
        print(f"使用模型: {self.model}")
        print(f"API地址: {self.client.base_url}")

        # 调用API生成推理
        inference_result = self.generate_inference(context)

        print("\n" + "=" * 60)
        print("【API返回结果】")
        print("=" * 60)

        if "error" in inference_result:
            print(f"❌ API调用失败: {inference_result['error']}")
        else:
            print("✅ API调用成功！")
            print("\n完整JSON结果:")
            print(json.dumps(inference_result, ensure_ascii=False, indent=2))

            # 保存到jsonl文件
            # inference_result['user_id'] = user_id
            inference_result['generated_time'] = time.strftime("%Y-%m-%d %H:%M:%S")

            output_file = "user_inferences.jsonl"
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(inference_result, ensure_ascii=False) + '\n')

            print(f"\n💾 结果已保存到: {output_file}")

        print("\n" + "=" * 60)
        return inference_result


# 使用示例
if __name__ == "__main__":
    load_env_file()

    # 设置API密钥
    API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
    if not API_KEY:
        raise RuntimeError("Please set DEEPSEEK_API_KEY in .env before running this script.")
    BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # 初始化生成器（启用模拟数据模式，无需CSV文件）
    generator = KuaiRecInferenceGenerator(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        use_mock_data=True,
    )

    # 测试API调用
    generator.test_api_call(1)


    # 2. 批量处理数据
    # generator.batch_process(
    #     user_data_file="user_interactions.csv",  # 你的用户交互数据文件
    #     output_file="user_inferences.jsonl",
    #     batch_size=20,
    #     delay=1.0  # 1秒延迟
    # )

    # 3. 读取生成的JSONL文件
    def read_jsonl(file_path: str):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    # 读取示例
    # inferences = read_jsonl("user_inferences.jsonl")
