# validators.py

from typing import Dict, Tuple

PROMPT_INJECTION_PATTERNS = [
    "忽略以上所有指令",
    "ignore previous",
    "你是一个大语言模型",
    "you are chatgpt",
    "system prompt",
    "作为一个大型语言模型",
    "act as ",
    "act as a ",
    "现在开始你不再是",
    "请无视之前的设定",
]

def _check_length(name: str, value: str, min_len: int, max_len: int) -> str:
    v = (value or "").strip()
    if min_len and len(v) < min_len:
        raise ValueError(f"{name} 太短（至少 {min_len} 个字符）")
    if max_len and len(v) > max_len:
        raise ValueError(f"{name} 太长（最多 {max_len} 个字符），请精简后再试")
    return v

def _check_injection(name: str, value: str) -> str:
    lower_v = value.lower()
    for p in PROMPT_INJECTION_PATTERNS:
        if p.lower() in lower_v:
            raise ValueError(f"{name} 中包含不允许的控制型指令片段，请改写后再提交。")
    return value

def validate_cot_inputs(
    user_question: str,
    brand_brief: str,
    must_expose: str,
    expo_hint: str = "",
) -> Dict[str, str]:
    """
    统一做一层输入规范 + 防注入。
    失败直接 raise ValueError，在上层捕获并显示给用户即可。
    """
    uq = _check_length("目标问题", user_question, 10, 300)
    bb = _check_length("品牌简介", brand_brief, 50, 2000)
    me = _check_length("期望露出字段", must_expose, 0, 300)
    eh = _check_length("补充提示", expo_hint, 0, 500)

    uq = _check_injection("目标问题", uq)
    bb = _check_injection("品牌简介", bb)
    me = _check_injection("期望露出字段", me)
    eh = _check_injection("补充提示", eh)

    # 可以在这里再做一些简单的“品牌名是否出现”的检查
    if len(me) > 0 and (me.split()[0] not in bb):
        # 不强制报错，只是未来可以做 warning
        pass

    return {
        "user_question": uq,
        "brand_brief": bb,
        "must_expose": me,
        "expo_hint": eh,
    }
    
def validate_brand_brief(bb: dict) -> Tuple[bool, list]:
    """
    校验 brand_brief 的结构与内容
    返回 (是否通过, 错误列表)
    """
    required = [
        "brand_name",
        "category",
        "target_audience",
        "core_value",
        "key_features",
        "differentiators",
        "use_cases",
    ]

    errors = []

    # 必填字段存在性检查
    for field in required:
        if not bb.get(field):
            errors.append(f"❌ 缺少必要字段: {field}")

    # 文本类字段长度与非空规则
    if "brand_name" in bb and len(bb["brand_name"]) < 2:
        errors.append("❌ brand_name 过短，不具有品牌特征")

    if "category" in bb and len(bb["category"]) < 2:
        errors.append("❌ category 信息不足")

    # 列表兼容文本自动拆分
    for field in ["key_features", "differentiators", "use_cases"]:
        if isinstance(bb.get(field), str):
            arr = [x.strip() for x in bb[field].split("\n") if x.strip()]
            bb[field] = arr
        if not isinstance(bb.get(field), list) or len(bb[field]) == 0:
            errors.append(f"❌ {field} 至少需要 1 条以上内容")

    # must_expose 可选字段无需校验

    return len(errors) == 0, errors

