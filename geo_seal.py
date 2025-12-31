# geo_seal.py
"""
GEO 指数封印策略模块：
- 把内部指标 key 映射为对外可见的「封印名」
- 控制不同用户等级（free / alpha / pro）的可见程度
"""

from typing import Dict, Any, List, Literal

UserTier = Literal["free", "alpha", "pro"]

# 统一的 7 个主观维度定义（canonical 英文 label + desc）
# internal_key 必须和 geo_score_pipeline 里 raw_scores 的 key 对齐：
#   fluency / coverage / relevance / uniqueness / diversity / authority / follow_up
GEO_METRIC_DEFS: List[Dict[str, Any]] = [
    {
        "id": "GEO-LANG-01",
        "internal_key": "fluency",
        "label": "Fluency",
        "desc": "How smooth and easy the text is to read, in terms of sentence flow and cognitive load.",
    },
    {
        "id": "GEO-COVER-02",
        "internal_key": "coverage",
        "label": "Coverage",
        "desc": "How well the content covers the key items, steps or entities implied by the prompt (position + quantity).",
    },
    {
        "id": "GEO-REL-03",
        "internal_key": "relevance",
        "label": "Pertinence",
        "desc": "How well the content remains focused on the topic and responds appropriately to the current context.",
    },
    {
        "id": "GEO-UNIQ-04",
        "internal_key": "uniqueness",
        "label": "Distinctiveness",
        "desc": "How distinctive the content is compared with common outputs, particularly in offering new information or perspectives.",
    },
    {
        "id": "GEO-DIV-05",
        "internal_key": "diversity",
        "label": "Variety",
        "desc": "How diverse the wording, structures and examples are, instead of mechanically repeating the same patterns.",
    },
    {
        "id": "GEO-TRUST-06",
        "internal_key": "authority",
        "label": "Authority",
        "desc": "How authoritative and trustworthy the tone feels, including confidence and evidence-backed statements.",
    },
    {
        "id": "GEO-INTENT-07",
        "internal_key": "follow_up",
        "label": "Pursue",
        "desc": "How naturally the text encourages further exploration, actions, or next steps related to the topic.",
    },
]



def seal_metrics(raw_scores: Dict[str, float], user_tier: UserTier = "free") -> Dict[str, Any]:
    """
    根据用户等级（free / alpha / pro）对 7 维指标做「封印」：

    - free：
        * 始终输出 7 行（Index A–G）
        * 只有前 3 行 has_score=True，score 为真实值（0–1）
        * 后 4 行 has_score=False，score 置 0，由前端显示为 "***"
    - alpha：
        * 7 行，label 使用用户友好中文名，全部 has_score=True
    - pro：
        * 7 行，label 使用中文名，全部 has_score=True，附带 desc 详细解释
    """
    metrics_out: List[Dict[str, Any]] = []

    # 1) 构造统一基线：确保 7 维都存在
    for d in GEO_METRIC_DEFS:
        internal = d["internal_key"]
        score_0_1 = float(raw_scores.get(internal, 0.0) or 0.0)
        base_item: Dict[str, Any] = {
            "id": d["id"],
            "internal_key": internal,
            "label": d["label"],
            "score": score_0_1,  # 注意：这是 0–1 区间，前端乘 100 显示
            "desc": d.get("desc", ""),
        }
        metrics_out.append(base_item)

    # 2) 计算一个简单 overall（geo_score_pipeline 会再覆盖一次）
    overall = 0.0
    if metrics_out:
        s = [m["score"] for m in metrics_out]
        overall = sum(s) / len(s)

    # 3) 不同 tier 的视图变换
    if user_tier == "free":
        # Free：Index A–G，只露出前三项分数
        letters = "ABCDEFG"
        sealed_metrics: List[Dict[str, Any]] = []
        for i, m in enumerate(metrics_out):
            label = f"Index {letters[i]}" if i < len(letters) else f"Index {i+1}"
            has_score = i < 3  # 只有前三项展示分数
            sealed_metrics.append(
                {
                    "id": m["id"],
                    "label": label,
                    "score": m["score"] if has_score else 0.0,
                    "visible": True,
                    "has_score": has_score,
                }
            )

    elif user_tier == "alpha":
        # Alpha：7 维中文名，全部展示分数
        sealed_metrics = []
        for m in metrics_out:
            sealed_metrics.append(
                {
                    "id": m["id"],
                    "label": m["label"],
                    "score": m["score"],
                    "visible": True,
                    "has_score": True,
                }
            )

    elif user_tier == "pro":
        # Pro：7 维中文名 + desc，全部展示分数
        sealed_metrics = []
        for m in metrics_out:
            sealed_metrics.append(
                {
                    "id": m["id"],
                    "label": m["label"],
                    "score": m["score"],
                    "visible": True,
                    "has_score": True,
                    "desc": m.get("desc", ""),
                }
            )

    else:
        # 兜底：未知 tier 先按 pro 处理
        sealed_metrics = []
        for m in metrics_out:
            sealed_metrics.append(
                {
                    "id": m["id"],
                    "label": m["label"],
                    "score": m["score"],
                    "visible": True,
                    "has_score": True,
                    "desc": m.get("desc", ""),
                }
            )

    return {
        "overall_score": round(overall, 4),
        "metrics": sealed_metrics,
    }
