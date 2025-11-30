# geo_brand.py
"""
品牌信息结构化 & 文本拼装工具 —— 修正版
新增字段：must_expose（可选）
"""

from typing import Dict, List, Union, Any, Tuple
from validators import validate_brand_brief


def _normalize_str_list(value: Union[str, List[str], None]) -> List[str]:
    """
    接受 str / list[str] / None，统一变成去空行后的 list[str]。
    """
    if value is None:
        return []
    if isinstance(value, str):
        items = [line.strip() for line in value.splitlines()]
    else:
        items = [str(v).strip() for v in value]
    return [v for v in items if v]


def build_brand_brief_structured(data: Dict[str, object]) -> str:
    """
    依据结构化字段构建统一的 brand_brief 文本。

    必选字段：
      - brand_name: str
      - category: str
      - target_audience: str
      - core_value: str
      - key_features: List[str] or str
      - differentiators: List[str] or str
      - use_cases: List[str] or str

    可选字段：
      - must_expose: str (释放字段，不强制，但若存在需进入逻辑链)
    """

    brand_name = (data.get("brand_name") or "").strip()
    category = (data.get("category") or "").strip()
    target_audience = (data.get("target_audience") or "").strip()
    core_value = (data.get("core_value") or "").strip()

    key_features = _normalize_str_list(data.get("key_features"))
    differentiators = _normalize_str_list(data.get("differentiators"))
    use_cases = _normalize_str_list(data.get("use_cases"))

    # 🔥 新增字段：可选的期望露出 Must Expose（供 COT 使用）
    must_expose = (data.get("must_expose") or "").strip()

    parts: List[str] = []

    if brand_name:
        parts.append(f"【品牌名称】{brand_name}")
    if category:
        parts.append(f"【所在行业/品类】{category}")
    if target_audience:
        parts.append(f"【目标人群】{target_audience}")
    if core_value:
        parts.append(f"【核心价值主张】{core_value}")

    if key_features:
        feat_text = "\n".join(f"- {f}" for f in key_features)
        parts.append(f"【核心功能/模块】\n{feat_text}")

    if differentiators:
        diff_text = "\n".join(f"- {d}" for d in differentiators)
        parts.append(f"【差异化亮点】\n{diff_text}")

    if use_cases:
        case_text = "\n".join(f"- {c}" for c in use_cases)
        parts.append(f"【典型使用场景】\n{case_text}")

    # 🟢 可选字段：Must Expose
    # 只有存在时才写入（未来 Stage1/Stage2 触发落点用）
    
    if must_expose:
        parts.append(f"【期望露出字段】{must_expose}")

    if not parts:
        return ""

    return "\n".join(parts)


def build_and_validate_brand_brief(data: Dict[str, Any]) -> Tuple[bool, List[str], str]:
    """
    综合入口：
    1）对 brand_brief 的结构化字段执行 validate_brand_brief 校验
    2）校验通过后才执行 build_brand_brief_structured 构建文本
    返回:
        ok: bool                是否通过校验
        errors: list[str]       错误描述列表
        brand_brief_text: str   拼接后的品牌简介文本，用于 Stage1 的上游输入
    """
    ok, errors = validate_brand_brief(data)
    if not ok:
        return False, errors, ""

    brief_text = build_brand_brief_structured(data)
    return True, [], brief_text
