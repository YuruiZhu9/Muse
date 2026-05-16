from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict


COMPACT_FIELD_NAMES = [
    "long_term_intent",
    "life_stage",
    "psychological_demand",
    "retrieval_suggestions",
    "interest_growth_points",
]

PLACEHOLDER_VALUES = {
    "",
    "字符串",
    "用户ID",
    "user_id",
    "性别",
    "gender",
    "活跃度",
    "活跃程度",
    "user_active_degree",
    "未提供",
    "未知",
}


STUDENT_SYSTEM_PROMPT = """
你是推荐系统中的用户兴趣语义蒸馏模型。

输入是基于 KuaiRec 数据整理出的用户画像、时间有序交互历史、观看深度标签与视频语义信息。
你的任务不是输出长篇解释，而是输出后续 embedding 与推荐 pipeline 直接使用的紧凑语义 JSON。

你必须只输出一个严格可解析的 JSON 对象，不要输出 Markdown、解释、前言、思考过程或任何额外文字。

输出 JSON 的固定结构如下：
{
  "user_basic_information": {
    "user_id": "字符串",
    "user_active_degree": "字符串",
    "gender": "字符串"
  },
  "long_term_intent": "1-2 句，总结长期稳定兴趣、主内容簇与消费主轴。",
  "life_stage": "1 句，对当前阶段或生活状态做保守判断；证据弱时明确写出“保守判断/信号有限/偏探索期”。",
  "psychological_demand": "1-2 句，概括核心心理需求与近期动机。",
  "retrieval_suggestions": "1-2 句，用自然语言串联显式查询倾向和隐式召回关键词。",
  "interest_growth_points": "1-2 句，描述潜在扩展方向、跨兴趣桥接点或下一步可探索主题。",
  "generated_time": "YYYY-MM-DD HH:MM:SS"
}

严格约束：
1. `user_id` 只能出现在 `user_basic_information` 内，不能单独出现在顶层。
2. `generated_time` 必须是最后一个字段。
3. 5 个语义字段必须全部存在，且它们的值都必须是字符串，不能是数组、对象或 null。
4. 不要编造上下文里没有证据的信息；证据不足时使用保守表述。
5. 输出尽量紧凑、信息密度高、适合后续 embedding 编码。
6. 输出语言使用中文。
""".strip()


def build_student_system_prompt() -> str:
    return STUDENT_SYSTEM_PROMPT


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = "；".join(_normalize_text(item) for item in value if _normalize_text(item))
    else:
        text = str(value)
    text = text.replace("\r", "\n").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip("`\"' ")


def _join_non_empty(parts) -> str:
    return "；".join(part for part in (_normalize_text(part) for part in parts) if part)


def _is_placeholder_text(value: Any) -> bool:
    return _normalize_text(value) in PLACEHOLDER_VALUES


def is_legacy_teacher_schema(obj: Dict[str, Any]) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("user_status_analysis"), dict)


def is_compact_student_schema(obj: Dict[str, Any]) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("user_basic_information"), dict)
        and all(field in obj for field in COMPACT_FIELD_NAMES)
    )


def extract_basic_info_from_context(context: str, fallback_user_id: Any) -> Dict[str, str]:
    patterns = {
        "user_id": [r"用户ID:\s*([^\n]+)", r"user_id:\s*([^\n]+)"],
        "user_active_degree": [r"user_active_degree:\s*([^\n]+)", r"活跃度:\s*([^\n]+)", r"用户活跃度:\s*([^\n]+)"],
        "gender": [r"gender:\s*([^\n]+)", r"性别:\s*([^\n]+)"],
    }
    result = {
        "user_id": str(fallback_user_id),
        "user_active_degree": "",
        "gender": "",
    }
    for key, regexes in patterns.items():
        for pattern in regexes:
            match = re.search(pattern, context, flags=re.IGNORECASE)
            if match:
                result[key] = _normalize_text(match.group(1))
                break
    return result


def teacher_full_to_compact(
    obj: Dict[str, Any],
    fallback_user_id: Any = "",
    fallback_active_degree: str = "",
    fallback_gender: str = "",
    generated_time: str = "",
) -> Dict[str, str]:
    basic = obj.get("user_basic_information", {}) if isinstance(obj, dict) else {}
    analysis = obj.get("user_status_analysis", {}) if isinstance(obj, dict) else {}
    retrieval = obj.get("retrieval_suggestions", {}) if isinstance(obj, dict) else {}

    life_stage_hypothesis = analysis.get("life_stage_hypothesis", {}) if isinstance(analysis, dict) else {}
    psychological_demand = analysis.get("psychological_demand", {}) if isinstance(analysis, dict) else {}
    interest_growth_points = analysis.get("interest_growth_points", {}) if isinstance(analysis, dict) else {}

    life_stage_parts = [
        life_stage_hypothesis.get("stage"),
        f"置信度：{_normalize_text(life_stage_hypothesis.get('confidence'))}" if _normalize_text(life_stage_hypothesis.get("confidence")) else "",
        f"关键标签：{'、'.join(_normalize_text(x) for x in life_stage_hypothesis.get('key_attributes', []) if _normalize_text(x))}"
        if isinstance(life_stage_hypothesis.get("key_attributes"), list) and any(_normalize_text(x) for x in life_stage_hypothesis.get("key_attributes", []))
        else "",
    ]

    retrieval_text = _join_non_empty(
        [
            f"显式查询：{'、'.join(_normalize_text(x) for x in retrieval.get('explicit_queries', []) if _normalize_text(x))}"
            if isinstance(retrieval.get("explicit_queries"), list) and any(_normalize_text(x) for x in retrieval.get("explicit_queries", []))
            else "",
            f"隐式关键词：{'、'.join(_normalize_text(x) for x in retrieval.get('implicit_keywords', []) if _normalize_text(x))}"
            if isinstance(retrieval.get("implicit_keywords"), list) and any(_normalize_text(x) for x in retrieval.get("implicit_keywords", []))
            else "",
        ]
    )

    growth_text = _join_non_empty(
        [
            f"增长信号：{'、'.join(_normalize_text(x) for x in interest_growth_points.get('emerging_signals', []) if _normalize_text(x))}"
            if isinstance(interest_growth_points.get("emerging_signals"), list) and any(_normalize_text(x) for x in interest_growth_points.get("emerging_signals", []))
            else "",
            f"桥接概念：{'、'.join(_normalize_text(x) for x in interest_growth_points.get('bridge_concepts', []) if _normalize_text(x))}"
            if isinstance(interest_growth_points.get("bridge_concepts"), list) and any(_normalize_text(x) for x in interest_growth_points.get("bridge_concepts", []))
            else "",
        ]
    )

    return {
        "user_basic_information": {
            "user_id": str(fallback_user_id)
            if _is_placeholder_text(basic.get("user_id"))
            else (_normalize_text(basic.get("user_id")) or str(fallback_user_id)),
            "user_active_degree": _normalize_text(fallback_active_degree)
            if _is_placeholder_text(basic.get("user_active_degree"))
            else (_normalize_text(basic.get("user_active_degree")) or _normalize_text(fallback_active_degree)),
            "gender": _normalize_text(fallback_gender)
            if _is_placeholder_text(basic.get("gender"))
            else (_normalize_text(basic.get("gender")) or _normalize_text(fallback_gender)),
        },
        "long_term_intent": _normalize_text(
            analysis.get("long_term_intent", {}).get("description")
            if isinstance(analysis.get("long_term_intent"), dict)
            else analysis.get("long_term_intent")
        ),
        "life_stage": _join_non_empty(life_stage_parts),
        "psychological_demand": _join_non_empty(
            [
                psychological_demand.get("core_demand") if isinstance(psychological_demand, dict) else "",
                psychological_demand.get("immediate_need") if isinstance(psychological_demand, dict) else "",
            ]
        ),
        "retrieval_suggestions": retrieval_text,
        "interest_growth_points": growth_text,
        "generated_time": _normalize_text(obj.get("generated_time") or obj.get("generated_at") or generated_time),
    }


def _extract_field_best_effort(obj: Any, field_name: str) -> str:
    """Recursively search for a field value in nested dicts."""
    if isinstance(obj, dict):
        if field_name in obj:
            return _normalize_text(obj[field_name])
        for _key, value in obj.items():
            result = _extract_field_best_effort(value, field_name)
            if result:
                return result
    return ""


def _fallback_normalize(
    obj: Dict[str, Any],
    fallback_user_id: Any = "",
    fallback_active_degree: str = "",
    fallback_gender: str = "",
    generated_time: str = "",
) -> Dict[str, str]:
    """Best-effort normalization: extract fields from anywhere in the JSON."""
    basic = obj.get("user_basic_information", {}) if isinstance(obj, dict) else {}
    return {
        "user_basic_information": {
            "user_id": _normalize_text(basic.get("user_id")) or str(fallback_user_id),
            "user_active_degree": _normalize_text(basic.get("user_active_degree")) or _normalize_text(fallback_active_degree),
            "gender": _normalize_text(basic.get("gender")) or _normalize_text(fallback_gender),
        },
        "long_term_intent": _extract_field_best_effort(obj, "long_term_intent"),
        "life_stage": _extract_field_best_effort(obj, "life_stage"),
        "psychological_demand": _extract_field_best_effort(obj, "psychological_demand"),
        "retrieval_suggestions": _extract_field_best_effort(obj, "retrieval_suggestions"),
        "interest_growth_points": _extract_field_best_effort(obj, "interest_growth_points"),
        "generated_time": _normalize_text(obj.get("generated_time", "") or generated_time),
    }


def normalize_compact_result(
    obj: Dict[str, Any],
    fallback_user_id: Any = "",
    fallback_active_degree: str = "",
    fallback_gender: str = "",
    generated_time: str = "",
) -> Dict[str, str]:
    if is_legacy_teacher_schema(obj):
        result = teacher_full_to_compact(
            obj,
            fallback_user_id=fallback_user_id,
            fallback_active_degree=fallback_active_degree,
            fallback_gender=fallback_gender,
            generated_time=generated_time,
        )
    elif is_compact_student_schema(obj):
        basic = obj.get("user_basic_information", {})
        result = {
            "user_basic_information": {
                "user_id": str(fallback_user_id)
                if _is_placeholder_text(basic.get("user_id"))
                else (_normalize_text(basic.get("user_id")) or str(fallback_user_id)),
                "user_active_degree": _normalize_text(fallback_active_degree)
                if _is_placeholder_text(basic.get("user_active_degree"))
                else (_normalize_text(basic.get("user_active_degree")) or _normalize_text(fallback_active_degree)),
                "gender": _normalize_text(fallback_gender)
                if _is_placeholder_text(basic.get("gender"))
                else (_normalize_text(basic.get("gender")) or _normalize_text(fallback_gender)),
            },
            "long_term_intent": _normalize_text(obj.get("long_term_intent")),
            "life_stage": _normalize_text(obj.get("life_stage")),
            "psychological_demand": _normalize_text(obj.get("psychological_demand")),
            "retrieval_suggestions": _normalize_text(obj.get("retrieval_suggestions")),
            "interest_growth_points": _normalize_text(obj.get("interest_growth_points")),
            "generated_time": _normalize_text(obj.get("generated_time") or generated_time),
        }
    else:
        result = _fallback_normalize(
            obj,
            fallback_user_id=fallback_user_id,
            fallback_active_degree=fallback_active_degree,
            fallback_gender=fallback_gender,
            generated_time=generated_time,
        )

    for field in COMPACT_FIELD_NAMES:
        result[field] = _normalize_text(result.get(field))

    if not result["user_basic_information"]["user_id"]:
        result["user_basic_information"]["user_id"] = str(fallback_user_id)

    if not result["generated_time"]:
        result["generated_time"] = _normalize_text(generated_time) or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "user_basic_information": result["user_basic_information"],
        "long_term_intent": result["long_term_intent"],
        "life_stage": result["life_stage"],
        "psychological_demand": result["psychological_demand"],
        "retrieval_suggestions": result["retrieval_suggestions"],
        "interest_growth_points": result["interest_growth_points"],
        "generated_time": result["generated_time"],
    }


def compact_state_texts_from_any(
    obj: Dict[str, Any],
    fallback_user_id: Any = "",
) -> Dict[str, str]:
    normalized = normalize_compact_result(obj, fallback_user_id=fallback_user_id, generated_time="")
    return {field: _normalize_text(normalized.get(field)) for field in COMPACT_FIELD_NAMES}
