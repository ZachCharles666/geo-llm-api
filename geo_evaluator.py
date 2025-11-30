# -*- coding: utf-8 -*-
# geo_evaluator.py

import json
import os
import re
import time
from statistics import mean, pstdev
from typing import Dict, Literal, TypedDict, Optional
from openai import OpenAI

from geo_metrics import compression_ratio, type_token_ratio, reading_ease

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

_NUM_RE = re.compile(r'([1-5](?:\.\d+)?)')

def _clip_1_5(x: float) -> float:
    return max(1.0, min(5.0, x))

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
请只输出一个阿拉伯数字，范围 1 到 5。不要输出任何解释或文字，直接输出数字。
例如：4.5
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
    评分严格模式：在模板后追加“只输出数字”提示；解析失败自动重试，最终回退3.0。
    """
    base_prompt = _format_prompt(prompt_template, query, answer)
    prompt = f"{base_prompt.strip()}\n\n{FORCE_NUMERIC_SUFFIX.strip()}"
    last_err = None
    for _ in range(max(1, retries)):
        try:
            text = llm_complete(model_name, prompt, provider=provider, temperature=0.0, max_tokens=12)
            val = _extract_score(text or "")
            if val is not None:
                return _clip_1_5(val)
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    return 3.0

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

    return means, stdevs


def _subjective_to_0_100(subj: Dict[str, float]) -> float:
    # 线性映射 (1~5) → (0~100)
    keys = ["relevance","influence","uniqueness","diversity",
            "subjective_position","subjective_count","follow_up"]
    vals = [subj[k] for k in keys]
    norm = [ (v - 1.0) / 4.0 * 100.0 for v in vals ]
    return float(mean(norm))

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
