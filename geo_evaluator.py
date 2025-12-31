# -*- coding: utf-8 -*-
# geo_evaluator.py

import json
import os
import re
import time
import hashlib  # ⭐ 新增
import math
from statistics import mean, pstdev
from typing import Dict, Literal, TypedDict, Optional, Any
from openai import OpenAI

from geo_metrics import compression_ratio, type_token_ratio, reading_ease
from geo_report import render_report_html
from geo_seal import seal_metrics   # 顶部加这一行

from providers_groq_gemini import ModelHub
hub = ModelHub()

# ========= 统一 LLM 调用（支持手动 + auto fallback） =========
from pipeline.inference_engine import call_model

# ========= 你需要把这里接到你已有的 DashScope/DeepSeek 调用 =========
def llm_complete(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 32,
    provider: str = "groq",
) -> str:
    """
    评分专用调用（严格短输出）：
    - provider 可选 groq / gemini / grok / deepseek / qwen
    - model_name 作为具体模型名透传
    """
    # 评分提示：只要数字
    sys_hint = "You are a strict grader. Output ONE number only."
    full_prompt = f"{sys_hint}\n\n{prompt}"

    # 统一走 call_model
    return (call_model(
        full_prompt,
        provider=provider,
        temperature=temperature,
        model=model_name,
    ) or "").strip()

# 简单的进程内缓存：同一模型 + 同一问句 + 同一原文 + 同一改写 → 只评一次
_GEO_BASE_CACHE: Dict[str, Dict[str, Any]] = {}

# =========================
# GEO Cache Debug Logging
# =========================
GEO_CACHE_DEBUG = os.getenv("GEO_CACHE_DEBUG", "0").strip() == "1"

def _cache_dbg(tag: str, hit: bool, cache_key: str, extra: dict | None = None):
    """
    仅在 GEO_CACHE_DEBUG=1 时打印 cache hit/miss
    """
    if not GEO_CACHE_DEBUG:
        return
    try:
        ck = (cache_key or "")[:24] + "..." if cache_key else ""
        payload = {"tag": tag, "hit": bool(hit), "key": ck}
        if extra:
            payload.update(extra)
        print("[GEO-CACHE]", json.dumps(payload, ensure_ascii=False))
    except Exception:
        # debug 不影响主流程
        print(f"[GEO-CACHE] tag={tag} hit={hit}")



def _make_cache_key(
    model_ui: str,
    model_name: str,
    user_question: str,
    article_title: str,
    source_text: str,
    rewritten_text: str,
) -> str:
    """
    为单次 GEO 评估生成稳定 key：
    - 同一模型 + 同一 provider + 同一问句 + 同一标题 + 同一原文 + 同一改写 → key 相同
    """
    payload = json.dumps(
        {
            "ui": model_ui,
            "model": model_name,
            "q": user_question,
            "title": article_title,
            "src": source_text,
            "opt": rewritten_text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()



class GeoScore(TypedDict):
    relevance: float
    influence: float
    uniqueness: float
    diversity: float
    subjective_position: float
    subjective_count: float
    follow_up: float
    objective: Dict[str, float]
    geo_score: float
    mode: Literal["single_text", "with_citations"]
    model_used: str
    latency_ms: int
    samples: int
    stddev: Dict[str, float]

# 支持 1–20（可带小数），并避免误抓到更大数字的尾巴
_NUM_RE = re.compile(r'(?<!\d)(?:20(?:\.\d+)?|1?\d(?:\.\d+)?)(?!\d)')


def _clip_1_5(x: float) -> float:
    return max(1.0, min(5.0, x))

def _clip_1_20(x: float) -> float:
    return max(1.0, min(20.0, float(x)))


def _extract_score(text: str) -> Optional[float]:
    """从 LLM 返回中抓取 1~5 的数字；取最后一个匹配以防前面是示例。"""
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None

def load_eval_templates(dir_path: str = "templates/") -> Dict[str, str]:
    def _read(name: str) -> str:
        p = os.path.join(dir_path, name)
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return {
        "relevance": _read("relevance_detailed.txt"),
        "influence": _read("influence_detailed.txt"),
        "uniqueness": _read("uniqueness_detailed.txt"),
        "diversity": _read("diversity_detailed.txt"),
        "subjpos": _read("subjpos_detailed.txt"),
        "subjcount": _read("subjcount_detailed.txt"),
        "follow": _read("follow_detailed.txt"),
    }

def _format_prompt(tpl: str, query: str, answer: str) -> str:
    """
    将模板中示例段落替换为实际的 {query}/{answer}。
    模板若含占位符 {query} / {answer} 最佳；若无占位，也直接附加在末尾。
    """
    has_q = "{query}" in tpl
    has_a = "{answer}" in tpl
    if has_q or has_a:
        return tpl.replace("{query}", query).replace("{answer}", answer)
    return f"{tpl.strip()}\n\nInput User Query:\n{query}\n\nGenerated Answer:\n{answer}\n\nEvaluation Form (scores ONLY):\n- Score:"

FORCE_NUMERIC_SUFFIX = """
You MUST output exactly one Arabic number between 1 and 20.
You MAY use decimals like 16.5.
Do NOT output any words or explanations, only the number.

例如：只输出一个 1 到 20 之间的阿拉伯数字，可以带小数，如：16.5
"""



def evaluate_dimension(
    model_name: str,
    prompt_template: str,
    query: str,
    answer: str,
    provider: str = "groq",
    mode: Literal["single_text", "with_citations"] = "single_text",
    retries: int = 2
) -> float:
    """
    评分严格模式：在模板后追加“只输出数字”提示；解析失败自动重试，最终回退12.0。
    """
    base_prompt = _format_prompt(prompt_template, query, answer)
    prompt = f"{base_prompt.strip()}\n\n{FORCE_NUMERIC_SUFFIX.strip()}"
    last_err = None
    for _ in range(max(1, retries)):
        try:
            text = llm_complete(model_name, prompt, provider=provider, temperature=0.0, max_tokens=12)
            val = _extract_score(text or "")
            if val is not None:
                return _clip_1_20(val)

        except Exception as e:
            last_err = e
            time.sleep(0.2)
    return 12.0

def _sample_scores(
    model_name: str,
    tpl: str,
    query: str,
    answer: str,
    n: int,
    provider: str = "auto"
) -> (float, float):
    vals = [
        evaluate_dimension(model_name, tpl, query, answer, provider=provider)
        for _ in range(max(1, n))
    ]
    return float(mean(vals)), float(pstdev(vals)) if len(vals) > 1 else 0.0

def evaluate_subjective_scores(
    model_name: str,
    query: str,
    answer: str,
    provider: str = "groq",
    samples: int = 1
) -> (Dict[str, float], Dict[str, float]):
    """
    返回：七维平均分、以及对应的标准差（用于诊断稳定性）
    """
    tpls = load_eval_templates()
    means, stdevs = {}, {}

    def _do(key_tpl, out_key):
        m, s = _sample_scores(model_name, tpls[key_tpl], query, answer, samples, provider=provider)
        means[out_key], stdevs[out_key] = m, s

    _do("relevance", "relevance")
    _do("influence", "influence")
    _do("uniqueness", "uniqueness")
    _do("diversity", "diversity")
    _do("subjpos", "subjective_position")
    _do("subjcount", "subjective_count")
    _do("follow", "follow_up")

    print("TEMPLATES LOADED:", tpls.keys())

    return means, stdevs

def _subjective_to_0_100(subj: Dict[str, float]) -> float:
    """
    主观七维平均分（0~20）映射到 0~100：
    你指定的规则：某一维原始打分为 a，则
        score_0_100 = sqrt(5 * a) * 10
    - a 允许为小数，但会被 clamp 到 [1, 5]
    - 例如：
        a = 1  → score ≈ sqrt(5) * 10 ≈ 22
        a = 5  → score = sqrt(25) * 10 = 50
    """
    keys = [
        "relevance",
        "influence",
        "uniqueness",
        "diversity",
        "subjective_position",
        "subjective_count",
        "follow_up",
    ]
    mapped_vals = []

    for k in keys:
        v = subj.get(k, 0.0)
        try:
            v = float(v)
        except Exception:
            # 解析失败时按 1 分处理（极差，但不是 0）
            v = 1.0

        # 限定在 [1, 20] 之间
        if v <= 0.0:
            v = 1.0
        v = max(1.0, min(20.0, v))

        # 根据你指定的公式：score_0_100 = sqrt(5 * a) * 10
        mapped = math.sqrt(5.0 * v) * 10.0
        mapped_vals.append(mapped)

    if not mapped_vals:
        return 0.0

    return float(mean(mapped_vals))


def compute_geo_score(subj: Dict[str, float], obj: Dict[str, float]) -> float:
    """
    总分 = 主观(七维均值的0~100) + 客观附加项（最多+40）
    """
    subjective = _subjective_to_0_100(subj)

    # 客观加分（启发式上限 40）
    cr = obj.get("compression_ratio", 1.0)
    ttr = obj.get("ttr", 0.0)              # 0~1
    fre = obj.get("reading_ease", 0.0)     # 0~100

    bonus = 0.0
    bonus += max(0.0, min(20.0, 20.0 * (1.0 - cr)))    # 更精炼更加分（上限20）
    bonus += max(0.0, min(10.0, fre / 10.0))           # 可读性（上限10）
    bonus += max(0.0, min(10.0, ttr * 100.0 / 50.0))   # TTR=0.5 记满10分

    return float(max(0.0, min(100.0, subjective + bonus)))

def evaluate_geo_score(
    model_name: str,
    query: str,
    src_text: str,
    opt_text: str,
    provider: str = "auto",
    mode: Literal["single_text","with_citations"] = "single_text",
    samples: int = 1
) -> GeoScore:
    """
    对优化稿（opt_text）进行主观七维评审 + 客观指标计算，返回统一结构。
    """
    t0 = time.time()
    subj_means, subj_std = evaluate_subjective_scores(
        model_name, query, opt_text, provider=provider, samples=samples
    )
    obj = {
        "compression_ratio": compression_ratio(src_text, opt_text),
        "ttr": type_token_ratio(opt_text),
        "reading_ease": reading_ease(opt_text, lang="auto"),
    }
    total = compute_geo_score(subj_means, obj)
    dt = int((time.time() - t0) * 1000)

    return GeoScore(
        relevance=subj_means["relevance"],
        influence=subj_means["influence"],
        uniqueness=subj_means["uniqueness"],
        diversity=subj_means["diversity"],
        subjective_position=subj_means["subjective_position"],
        subjective_count=subj_means["subjective_count"],
        follow_up=subj_means["follow_up"],
        objective=obj,
        geo_score=total,
        mode=mode,
        model_used=f"{provider}:{model_name}",
        latency_ms=dt,
        samples=samples,
        stddev=subj_std
    )

def build_geo_summary(
    grade: str,
    sealed_overall: float | None,
    user_tier: str,
    raw_scores: Optional[Dict[str, float]] = None,
    objective: Optional[Dict[str, float]] = None,
) -> str:
    """
    构造 GEO-Score 的自然语言 summary。

    - grade: A~E 等级（来自 sealed_overall）
    - sealed_overall: 0~1 的综合指数
    - user_tier: 'free' | 'alpha' | 'pro' | 'debug'
    - raw_scores: 7 维主观指标（0~1）
        fluency / coverage / relevance / uniqueness / diversity / authority / follow_up
    - objective: 客观指标（这里只使用 ttr / reading_ease，完全不解释 CR）
    """
    # -------- overall index ----------
    if sealed_overall is None:
        idx = None
        idx_str = "–"
    else:
        idx = float(sealed_overall)
        idx_str = f"{idx * 100:.1f}"

    tier = (user_tier or "free").lower().strip()

    # -------- 7 维维度映射 & 分数档位 ----------
    pretty_dim = {
        "fluency": "Fluency",
        "coverage": "Coverage",
        "relevance": "Pertinence",
        "uniqueness": "Distinctiveness",
        "diversity": "Variety",
        "authority": "Authority",
        "follow_up": "Pursue",
    }

    scores: Dict[str, float] = {}
    if raw_scores:
        for k, v in raw_scores.items():
            try:
                scores[k] = float(v)
            except Exception:
                continue

    def _band(v: float) -> str:
        """把 0~1 的得分切成档位标签。"""
        if v >= 0.80:
            return "high"          # 很强 / 表现突出
        if v >= 0.65:
            return "upper_mid"     # 明显偏好
        if v >= 0.50:
            return "mid"           # 中等
        if v >= 0.35:
            return "lower_mid"     # 偏弱
        return "low"               # 较弱

    def _pick_dim_lists():
        """选出相对表现靠前/靠后的维度列表，并区分是否存在“绝对强项”."""
        if not scores:
            return [], [], False

        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = items[:3]
        bottom = list(reversed(items))[:3]

        strong_dims = [k for k, v in top if v >= 0.65]
        weak_dims = [k for k, v in bottom if v <= 0.5]

        # 当所有分数都不高时，用“相对不那么弱”的说法，而不是 strong/weak
        has_real_strength = any(scores[k] >= 0.7 for k in scores.keys())

        return strong_dims, weak_dims, has_real_strength

    strong_dims, weak_dims, has_real_strength = _pick_dim_lists()

    def _fmt_dim_list(keys):
        labels = [pretty_dim.get(k, k) for k in keys]
        if not labels:
            return ""
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"{labels[0]} and {labels[1]}"
        return ", ".join(labels[:-1]) + " and " + labels[-1]

    # -------- 维度级别：name + score 的解释（主要用于 Pro） ----------
    def _dim_comment(key: str, v: float) -> str:
        label = pretty_dim.get(key, key)
        b = _band(v)

        if key == "relevance":
            if b == "high":
                return f"{label} is high: the draft stays closely aligned with the user question."
            if b == "upper_mid":
                return f"{label} is clearly above average, mostly staying on topic with only minor drift."
            if b == "mid":
                return f"{label} is in a middle band: the draft generally matches the question but occasionally drifts."
            if b == "lower_mid":
                return f"{label} is on the weak side, with noticeable off-topic or under-explained parts."
            return f"{label} is low, meaning the draft often misses or only partially answers the core question."

        if key == "coverage":
            if b == "high":
                return f"{label} is high: key points are well covered within the space of the rewrite."
            if b == "upper_mid":
                return f"{label} is above average, covering most of the important aspects."
            if b == "mid":
                return f"{label} is moderate: some core aspects are present, but a few angles are underdeveloped."
            if b == "lower_mid":
                return f"{label} is relatively weak; the draft only touches a subset of the necessary points."
            return f"{label} is low, suggesting large gaps in what a reader would expect to see."

        if key == "uniqueness":
            if b == "high":
                return f"{label} is high: the wording and framing feel distinctive rather than generic."
            if b == "upper_mid":
                return f"{label} is above average, with a noticeable amount of original framing."
            if b == "mid":
                return f"{label} is in a neutral band; the draft feels serviceable but not particularly original."
            if b == "lower_mid":
                return f"{label} is on the low side, with the text feeling quite template-like."
            return f"{label} is low, making the draft look formulaic and hard to differentiate in GEO."

        if key == "diversity":
            if b == "high":
                return f"{label} is high: the draft uses varied structures and perspectives."
            if b == "upper_mid":
                return f"{label} is above average in {label.lower()}, giving the content a richer feel."
            if b == "mid":
                return f"{label} is moderate; the draft mostly repeats a few patterns."
            if b == "lower_mid":
                return f"{label} is relatively weak, with limited variety in examples or angles."
            return f"{label} is low, so the draft feels monotonous and easy to skim past."

        if key == "authority":
            if b == "high":
                return f"{label} is high: the draft feels grounded with clear signals of expertise or credible references."
            if b == "upper_mid":
                return f"{label} is above average, offering some evidence or expert framing."
            if b == "mid":
                return f"{label} is middling; the draft asserts claims but does not always back them with signals of trust."
            if b == "lower_mid":
                return f"{label} is on the weak side, with many claims sounding somewhat unsupported."
            return f"{label} is low, meaning the draft lacks cues that models and readers can treat as trustworthy."

        if key == "follow_up":
            if b == "high":
                return f"{label} is high: the draft naturally opens up clear next questions or actions."
            if b == "upper_mid":
                return f"{label} is above average, offering a few good hooks for follow-up."
            if b == "mid":
                return f"{label} is moderate; follow-up space exists but is not made explicit."
            if b == "lower_mid":
                return f"{label} is relatively weak, with few hints about what to ask or do next."
            return f"{label} is low, so the draft feels like a dead end rather than a step in an ongoing dialogue."

        if key == "fluency":
            if b == "high":
                return f"{label} is high: sentences read smoothly and are easy to parse."
            if b == "upper_mid":
                return f"{label} is above average, with mostly natural flow."
            if b == "mid":
                return f"{label} is in a middle band; some phrases may feel a bit stiff or dense."
            if b == "lower_mid":
                return f"{label} is relatively weak and may slow readers down."
            return f"{label} is low, making the draft feel heavy or awkward to read."

        # fallback 一般说明
        if b == "high":
            return f"{label} is high for this draft."
        if b == "upper_mid":
            return f"{label} is clearly above average."
        if b == "mid":
            return f"{label} is in a middle band."
        if b == "lower_mid":
            return f"{label} is on the weak side."
        return f"{label} is low and needs attention."

    # -------- 客观指标解析：只使用 TTR + Reading Ease，完全不解释 CR ----------
    ttr = None
    fre = None
    if objective:
        try:
            if objective.get("ttr") is not None:
                ttr = float(objective["ttr"])
        except Exception:
            ttr = None
        try:
            if objective.get("reading_ease") is not None:
                fre = float(objective["reading_ease"])
        except Exception:
            fre = None

    def _ttr_phrase(short: bool = False) -> str:
        if ttr is None:
            return ""
        if ttr >= 0.55:
            return "uses fairly diverse wording" if short else (
                "The wording is fairly diverse, which helps the text feel less repetitive."
            )
        if ttr >= 0.35:
            return "keeps a balanced level of wording variety" if short else (
                "The text keeps a balanced level of wording variety, which is usually comfortable for readers and models."
            )
        return "relies on rather repetitive wording" if short else (
            "The text relies on rather repetitive wording, which can make it feel mechanical or generic."
        )

    def _fre_phrase_short() -> str:
        if fre is None:
            return ""
        if fre >= 70:
            return "reads very easily"
        if fre >= 55:
            return "is reasonably easy to read"
        if fre >= 40:
            return "is somewhat dense to read"
        return "feels quite heavy and effortful to read"

    def _fre_phrase_long() -> str:
        if fre is None:
            return ""
        if fre >= 70:
            return "The reading ease score is high, so the draft should feel light and easy for most readers."
        if fre >= 55:
            return "The reading ease score is above average, and most readers can follow the text without much effort."
        if fre >= 40:
            return "The reading ease score is in a middle band: understandable, but some readers may find it a bit dense."
        return "The reading ease score is low, so the draft may feel heavy or cognitively demanding; simplifying sentences would help."

    # -------- Free tier：整体+相对表现+引导升级 ----------
    if tier == "free":
        base = f"GEO-Max rated this draft at grade {grade}"
        if idx_str != "–":
            base += "."
        else:
            base += "."

        detail_parts: list[str] = []

        if scores:
            avg_score = sum(scores.values()) / len(scores)
            if avg_score >= 0.7:
                detail_parts.append("Overall the seven GEO dimensions are in a relatively strong band.")
            elif avg_score >= 0.5:
                detail_parts.append("Overall the seven GEO dimensions sit in a mid band.")
            else:
                detail_parts.append("Overall the seven GEO dimensions are on the weak side and would benefit from a focused revision.")

            if has_real_strength and strong_dims:
                detail_parts.append(
                    f"Within this profile, { _fmt_dim_list(strong_dims) } stand out as relatively better-performing dimensions."
                )
            if weak_dims:
                detail_parts.append(
                    f"{ _fmt_dim_list(weak_dims) } come out as the more constrained dimensions right now."
                )

        tail = " In the free preview you see only a overall index and a coarse dimension profile; full diagnostics are available in GEO Tools Alpha and Pro."
        return base + (" " + " ".join(detail_parts) if detail_parts else "") + tail

    # -------- Alpha tier：中等详细度解释 + TTR/Reading Ease 简要方向 ----------
    if tier == "alpha":
        base = f"GEO-Max rated this draft at grade {grade}"
        if idx_str != "–":
            base += "."
        else:
            base += "."

        parts: list[str] = []

        if scores:
            avg_score = sum(scores.values()) / len(scores)
            if avg_score >= 0.7:
                parts.append("Overall the 7 GEO dimensions are in a fairly strong band for this draft.")
            elif avg_score >= 0.5:
                parts.append("Overall the 7 GEO dimensions are in a middle band.")
            else:
                parts.append("Overall the 7 GEO dimensions are on the weaker side and would benefit from targeted polishing.")

            if has_real_strength and strong_dims:
                parts.append(
                    f"{ _fmt_dim_list(strong_dims) } are the relatively better dimensions."
                )
            if weak_dims:
                parts.append(
                    f"{ _fmt_dim_list(weak_dims) } are the dimensions where improvement would bring the most gain."
                )

        obj_phrases: list[str] = []
        ttr_s = _ttr_phrase(short=True)
        if ttr_s:
            obj_phrases.append(ttr_s)
        fre_s = _fre_phrase_short()
        if fre_s:
            obj_phrases.append(fre_s)

        if obj_phrases:
            parts.append(
                "From the objective side, the text " + ", ".join(obj_phrases) + "."
            )

        tail = " This Alpha view already folds these objective signals into the score."
        return base + (" " + " ".join(parts) if parts else "") + tail

    # -------- Pro tier：维度+分数解释 + TTR/Reading Ease 详细诊断 ----------
    if tier == "pro":
        base = f"GEO-Max rated this draft at grade {grade}"
        if idx_str != "–":
            base += "."
        else:
            base += "."

        dim_comments: list[str] = []
        if scores:
            explained_keys: set[str] = set()
            for k in strong_dims[:2]:
                if k in scores:
                    dim_comments.append(_dim_comment(k, scores[k]))
                    explained_keys.add(k)
            for k in weak_dims[:2]:
                if k in scores and k not in explained_keys:
                    dim_comments.append(_dim_comment(k, scores[k]))
                    explained_keys.add(k)

            if not dim_comments:
                mid_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
                for k, v in mid_items:
                    dim_comments.append(_dim_comment(k, v))

        obj_detail_parts: list[str] = []
        if ttr is not None:
            obj_detail_parts.append(
                f"The type–token ratio is about {ttr:.2f}; {_ttr_phrase(short=False)}"
            )
        if fre is not None:
            obj_detail_parts.append(
                f"The reading ease score is roughly {fre:.1f}/100. {_fre_phrase_long()}"
            )

        tail = " In Pro you get the full 7-dimension breakdown plus these objective diagnostics to guide precise revisions for GEO."

        return (
            base
            + (" " + " ".join(dim_comments) if dim_comments else "")
            + (" " + " ".join(obj_detail_parts) if obj_detail_parts else "")
            + " "
            + tail
        )

    # -------- Debug tier：最简单的说明 ----------
    if tier == "debug":
        base = f"GEO-Max rated this draft at grade {grade}"
        if idx_str != "–":
            base += "."
        else:
            base += "."
        return (
            base
            + " This debug view exposes all 7 subjective dimensions and objective metrics (excluding CR from interpretation) for internal inspection and engine tuning."
        )

    # -------- 兜底 ----------
    if idx_str != "–":
        return f"GEO-Max rated this draft at grade {grade} ."
    return f"GEO-Max rated this draft at grade {grade}."

def _build_anchored_query(user_question: str, article_title: str) -> str:
    """
    将标题/问题作为评分锚点写入 query，最小侵入。
    - 有 title：优先提供 title + question
    - 无 title：退化为原来的 user_question
    """
    q = (user_question or "").strip()
    t = (article_title or "").strip()

    if t and q:
        return f"[Article Title]\n{t}\n\n[User Question]\n{q}".strip()
    if t and not q:
        return f"[Article Title]\n{t}".strip()
    return q

def _estimate_tokens_rough(text: Optional[str]) -> int:
    """
    粗略 token 估算（与前端 approxTokens 思路一致即可）：
    - 英文：按单词数近似
    - 中文：按中文字符数近似
    - 混合：两者相加
    注意：这是“长度置信度”用途，不用于精确计费。
    """
    if not text:
        return 0
    t = str(text).strip()
    if not t:
        return 0

    # 中文字符（CJK Unified Ideographs 等大致范围）
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", t)
    cjk_count = len(cjk_chars)

    # 英文/数字单词
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", t)
    word_count = len(words)

    # 一个经验：中文 1 字≈1 token（粗略），英文 1 词≈1~1.3 token（这里取 1）
    return cjk_count + word_count


def _length_confidence_ct(tokens: int) -> float:
    """
    你的 Ct 定义（按你给出的规则落地）：
    - Hard Threshold: tokens < 400 -> Ct = 0.4
    - Growth: 400~1800 -> Ct = min(1, log(tokens-300)/log(1500))
      （因为 1800-300=1500）
    - Overload: tokens > 2500 -> Ct = 0.95（轻微下调）
    """
    try:
        n = int(tokens)
    except Exception:
        n = 0

    if n < 400:
        return 0.4

    if n <= 1800:
        # 防止 log(<=0)
        x = max(1, n - 300)
        denom = math.log(1800 - 300)  # log(1500)
        if denom <= 0:
            return 1.0
        return min(1.0, math.log(x) / denom)

    # 1800 以上默认满分系数
    ct = 1.0

    # 过载区间：>2500 轻微下调
    if n > 2500:
        ct = 0.95

    return ct

# ============================================================
# 🔁 统一暴露给外部的 Geo 指数入口（给 FastAPI / 前端调用）
#      —— 增加缓存：同一篇内容 + 模型，只评一次，多视图封印
# ============================================================

def geo_score_pipeline(
    user_question: str,
    article_title: str = "",
    source_text: str = "",
    rewritten_text: str = "",
    model_ui: str = "groq",
    model_name: str = "llama-3.3-70b-versatile",
    samples: int = 1,
    user_tier: str = "free",
):
    """
    GEO-Score 对外统一入口：

    - 输入：user_question / article_title(可选) / 原文 / 改写文 + 模型配置 + user_tier
    - 输出：前端直接使用的 geo_score / grade / summary / sealed 视图
    """
    start_ts = time.time()
    try:
        
        anchored_query = _build_anchored_query(user_question=user_question, article_title=article_title)

        # ⭐ 计算本次评估的 key（内容 + 模型 + provider）
        cache_key = _make_cache_key(
            model_ui=model_ui,
            model_name=model_name,
            user_question=user_question,
            article_title=article_title or "",
            source_text=source_text,
            rewritten_text=rewritten_text,
        )

        base = _GEO_BASE_CACHE.get(cache_key)

        # 这些变量在后续会用到（保证无论 cache hit/miss 都存在）
        subj_0_1: Dict[str, float] = {}
        raw_scores_0_100: Dict[str, float] = {}

        if base is None:
            _cache_dbg("score", False, cache_key, {"provider": model_ui, "model": model_name})
            # ================================
            #  缓存未命中：真正跑一遍评估
            # ================================
            result = evaluate_geo_score(
                model_name=model_name,
                query=anchored_query,    
                src_text=source_text,
                opt_text=rewritten_text,
                provider=model_ui,
                mode="single_text",
                samples=samples,
            )

            if hasattr(result, "dict"):
                raw: Dict[str, Any] = result.dict()
            elif isinstance(result, dict):
                raw = result
            else:
                raw = result.__dict__

            subj = {
                "relevance": float(raw.get("relevance", 0.0)),
                "influence": float(raw.get("influence", 0.0)),
                "uniqueness": float(raw.get("uniqueness", 0.0)),
                "diversity": float(raw.get("diversity", 0.0)),
                "subjective_position": float(raw.get("subjective_position", 0.0)),
                "subjective_count": float(raw.get("subjective_count", 0.0)),
                "follow_up": float(raw.get("follow_up", 0.0)),
            }
            obj = raw.get("objective") or {}
            geo_score_0_100 = float(raw.get("geo_score", 0.0))

            # ========= 主观 1~20 → 0~1 / 0~100（与你指定的公式严格对齐） =========
            def _scale_1_20_to_0_1(x: float) -> float:
                """
                主观打分 b ∈ [1,20] 映射到 0~1（与你确认的 1–20 体系严格对齐）：
                    score_0_100 = sqrt(5 * b) * 10
                    score_0_1   = score_0_100 / 100 = sqrt(5 * b) / 10
                关键锚点（与旧 1–5 对应 4/8/12/16/20 完全等价）：
                    b=4  → 0_100≈44.72 → 0_1≈0.4472
                    b=12 → 0_100≈77.46 → 0_1≈0.7746
                    b=20 → 0_100=100   → 0_1=1
                """
                try:
                    v = float(x)
                except Exception:
                    v = 1.0
                if v <= 0.0:
                    v = 1.0
                v = max(1.0, min(20.0, v))
                return math.sqrt(5.0 * v) / 10.0


            
            subj_0_1 = {k: _scale_1_20_to_0_1(v) for k, v in subj.items()}
            subj_0_100 = {k: float(v) * 100.0 for k, v in subj_0_1.items()}

            # ========= 客观 reading_ease → 0~1 =========
            def _scale_0_100_to_0_1(x: float) -> float:
                try:
                    v = float(x) / 100.0
                    return max(0.0, min(1.0, v))
                except Exception:
                    return 0.0

            reading_ease_val = obj.get("reading_ease", None)
            reading_ease_0_1 = (
                _scale_0_100_to_0_1(reading_ease_val)
                if reading_ease_val is not None
                else None
            )

            # Fluency 取值策略：
            # - 优先 reading_ease（可读性）
            # - 缺失/为0 则回退到 pertinence（relevance）
            if reading_ease_0_1 is not None and reading_ease_0_1 > 0.0:
                fluency_0_1 = reading_ease_0_1
            else:
                fluency_0_1 = subj_0_1["relevance"]

            coverage_0_1 = (
                subj_0_1["subjective_position"] + subj_0_1["subjective_count"]
            ) / 2.0

            raw_scores = {
                "fluency": float(fluency_0_1),
                "coverage": float(coverage_0_1),
                "relevance": float(subj_0_1["relevance"]),       # UI: Pertinence
                "uniqueness": float(subj_0_1["uniqueness"]),     # UI: Distinctiveness
                "diversity": float(subj_0_1["diversity"]),       # UI: Variety
                "authority": float(subj_0_1["influence"]),       # UI: Authority
                "follow_up": float(subj_0_1["follow_up"]),       # UI: Pursue
            }
            raw_scores_0_100 = {k: float(v) * 100.0 for k, v in raw_scores.items()}

            # ✅ 关键：把 subj_0_1 / raw_scores_0_100 一并塞进缓存，确保 cache hit 也能 debug
            base = {
                "raw": raw,
                "subj": subj,
                "subj_0_1": subj_0_1,                 # ✅ 新增
                "subj_0_100": subj_0_100,
                "objective": obj,
                "geo_score_0_100": geo_score_0_100,
                "raw_scores": raw_scores,
                "raw_scores_0_100": raw_scores_0_100,  # ✅ 新增
            }
            _GEO_BASE_CACHE[cache_key] = base

        else:
            _cache_dbg("score", True, cache_key, {"provider": model_ui, "model": model_name})
            # ================================
            #  缓存命中：直接复用上一次评估结果
            # ================================
            raw = base["raw"]
            subj = base["subj"]
            subj_0_100 = base["subj_0_100"]
            obj = base["objective"]
            geo_score_0_100 = base["geo_score_0_100"]
            raw_scores = base["raw_scores"]

            # ✅ 关键：subj_0_1 在缓存里优先取；没有就由 subj_0_100 派生（避免为空/报错）
            subj_0_1 = base.get("subj_0_1") or {k: float(v) / 100.0 for k, v in (subj_0_100 or {}).items()}

            # ✅ raw_scores_0_100 同理
            raw_scores_0_100 = base.get("raw_scores_0_100") or {k: float(v) * 100.0 for k, v in (raw_scores or {}).items()}

        # ================================
        # Length Confidence Coefficient (Ct)
        # - 在“打完分(raw_scores)”之后统一乘系数
        # - 在 seal_metrics 之前做，确保展示/overall 也被影响
        # ================================
        text_for_len = (rewritten_text or "").strip() or (source_text or "").strip()
        est_tokens = _estimate_tokens_rough(text_for_len)
        ct = _length_confidence_ct(est_tokens)

        # 对 7 维逐项乘 Ct，并 clamp 到 [0, 1]
        raw_scores = {k: max(0.0, min(1.0, float(v) * ct)) for k, v in (raw_scores or {}).items()}
        raw_scores_0_100 = {k: float(v) * 100.0 for k, v in raw_scores.items()}


        # debug 在封印层按 pro 的视角输出 7 维
        tier_for_seal = "pro" if user_tier == "debug" else user_tier
        sealed_view = seal_metrics(raw_scores, user_tier=tier_for_seal)

        sealed_overall = sealed_view.get("overall_score")
        if sealed_overall is None:
            sealed_overall = max(0.0, min(1.0, float(geo_score_0_100) / 100.0))
            sealed_view["overall_score"] = sealed_overall

        def _map_grade(idx_0_1: float) -> str:
            try:
                s = float(idx_0_1)
            except Exception:
                return "–"
            if s >= 0.85:
                return "A"
            if s >= 0.70:
                return "B"
            if s >= 0.55:
                return "C"
            if s >= 0.40:
                return "D"
            return "E"

        grade = _map_grade(sealed_overall)

        summary = build_geo_summary(
            grade=grade,
            sealed_overall=sealed_overall,
            user_tier=user_tier,
            raw_scores=raw_scores,
            objective=obj,
        )

        latency_ms = (time.time() - start_ts) * 1000.0

        raw_debug = {
            "subjective_raw_1_20": subj,
            "subjective_0_100": subj_0_100,
            "objective": obj,
            "geo_score_raw_0_100": geo_score_0_100,
        }

        out: Dict[str, Any] = {
            "ok": True,
            "error": None,
            "geo_score": geo_score_0_100,
            "grade": grade,
            "summary": summary,
            "subjective": subj,
            "objective": obj,
            "sealed": sealed_view,
            "sealed_overall": sealed_overall,
            "latency_ms": latency_ms,
            "model_used": raw.get("model_used"),
            "samples": raw.get("samples", samples),
            "user_tier": user_tier,
        }

        # ✅ 你已改过：debug_on OR user_tier == "debug"（这里直接给最终版）
        debug_on = os.getenv("GEO_SCORE_DEBUG", "0").strip() == "1"
        want_debug_view = debug_on or (user_tier == "debug")

        if want_debug_view:
            out["raw_scores_0_1"] = raw_scores
            out["raw_scores_0_100"] = raw_scores_0_100
            out["subjective_raw_1_20"] = subj
            out["subjective_scaled_0_1"] = subj_0_1
            out["subjective_scaled_0_100"] = subj_0_100
            # 注意：确保 ct / est_tokens 在上文已计算出来
            out["length_confidence"] = {"ct": ct, "est_tokens": est_tokens}

        if user_tier == "debug":
            out["raw_debug"] = raw_debug

        return out

    except Exception as e:
        latency_ms = (time.time() - start_ts) * 1000.0
        return {
            "ok": False,
            "error": f"geo_score_pipeline error: {e}",
            "geo_score": 0.0,
            "grade": "E",
            "summary": "GEO-Score evaluation failed.",
            "subjective": {},
            "objective": {},
            "sealed": {"overall_score": 0.0, "metrics": []},
            "sealed_overall": 0.0,
            "latency_ms": latency_ms,
            "model_used": None,
            "samples": samples,
            "user_tier": user_tier,
        }



def build_geo_report(project_title: str,
                     query: str,
                     src_text: str,
                     opt_text: str,
                     score: dict) -> str:
    """
    Stage3：将 GEO-Score + 文本前后对比渲染成可分享 HTML。
    """
    return render_report_html(
        project_title=project_title,
        query=query,
        src_text=src_text,
        opt_text=opt_text,
        score=score,
    )
