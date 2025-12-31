# geo_core.py — GEO-Max 四大核心能力的无 UI 封装
from __future__ import annotations

import os
from typing import Dict, Any, Tuple

from pipeline.inference_engine import call_model
from geo_evaluator import evaluate_geo_score
from geo_impression import (
    impression_word_count,
    impression_pos_count,
    impression_wordpos_count,
    compute_delta,
)
from geo_report import render_report_html
from validators import validate_cot_inputs

# ✅ 唯一来源：统一从 providers_groq_gemini 读取 provider 映射 & 默认模型
from providers_groq_gemini import norm_provider, DEFAULT_MODELS


# ============ 公共工具 ============

def _build_lang_instruction(out_lang: str) -> str:
    """根据 out_lang 生成给大模型看的语言约束"""
    if out_lang == "Chinese":
        return "请使用简体中文输出结果。"
    if out_lang == "English":
        return "Please answer in English."
    # Auto 或其他情况
    return "输出语言请与输入文本的主要语言保持一致；若输入为中文，请使用简体中文输出。"


def split_into_chunks(text: str, max_chars: int) -> list[str]:
    """
    按段落分块，逻辑与 app.py 中保持一致，防止单次调用过长。
    """
    import re

    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paras = re.split(r"\n{2,}", text)
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p.strip()
    if buf:
        chunks.append(buf)
    return chunks


def _load_md_template(name: str) -> str:
    """
    从 geo_prompts 目录加载 markdown 模板，如 cot_stage1.md / cot_stage2.md。
    """
    base = os.path.join(os.path.dirname(__file__), "geo_prompts")
    path = os.path.join(base, f"{name}.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:  # 本地读文件出错时把错误文字直接返回，方便调试
        return f"⚠️ 无法读取模板：{path}\n\n错误：{e}"


_ALLOWED_MD_KEYS = {"USER_QUESTION", "BRAND_BRIEF", "MUST_EXPOSE", "EXPO_HINT", "STAGE1_MD"}


def _fmt_md_template(tpl: str, **vars) -> str:
    """
    兼容 {USER_QUESTION} 这类占位符，避免与 format 的 {} 冲突。
    """
    if not isinstance(tpl, str):
        tpl = str(tpl or "")
    t = tpl.replace("{", "{{").replace("}", "}}")
    for key in _ALLOWED_MD_KEYS:
        t = t.replace("{{" + key + "}}", "{" + key + "}")
    return t.format(**vars)


# ============ 功能 1：内容改写 · GEO-Max 优化稿 ============

def geo_rewrite(
    text: str,
    model_ui: str = "Groq",
    use_chunk: bool = True,
    max_chars: int = 2400,
    out_lang: str = "Auto",
    temperature: float = 0.2,
    rewrite_goal: str = "balanced",  # ✅ NEW: auto / fast / balanced / credibility / seo_push
) -> Tuple[str, str]:
    """
    功能 1：内容改写，生成 GEO-Max 优化稿（多语言 + geo_prompts.json 版本）
    - 支持 2-pass / 3-pass（由 rewrite_goal 决定）
    - 支持长度下限约束（避免输出过短）
    返回: (optimized_text, original_text)
    """
    from pathlib import Path
    import json

    # 0. 输入预处理
    text = (text or "").strip()
    if not text:
        return "⚠️ 请输入原文。", ""

    # 1. 解析模型 provider & 具体模型
    provider = norm_provider(model_ui)
    model = DEFAULT_MODELS.get(provider) or next(iter(DEFAULT_MODELS.values()))

    # 2. 构造语言指令（给大模型看的）
    if (out_lang or "Auto") == "Auto":
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            out_lang = "Chinese"
    lang_instruction = _build_lang_instruction(out_lang)

    # 3. 语言 key（沿用你现有映射，已覆盖 de/es/fr/ja/ko）
    lang_key_map = {
        "Auto": "auto",
        "Chinese": "zh",
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "Japanese": "ja",
        "Korean": "ko",
        "German": "de",
    }
    lang_suffix = lang_key_map.get(out_lang, "auto")

    # 4. 读取 geo_prompts.json
    prompt_file = Path(__file__).with_name("geo_prompts.json")
    prompts: Dict[str, str] = {}
    if prompt_file.exists():
        try:
            prompts = json.loads(prompt_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[geo_rewrite] 读取 geo_prompts.json 失败，将使用内置中文兜底 Prompt：{e}")
            prompts = {}
    else:
        print("[geo_rewrite] geo_prompts.json 未找到，将使用内置中文兜底 Prompt。")

    # 5. 内置兜底 Prompt（保持你原有逻辑）
    fallback_prompt = (
        "你是一名生成式引擎优化（GEO）专家，负责将下面的文本改写为更适合被大模型引用和总结的版本。"
        "在不改变原始观点方向的前提下，提升逻辑清晰度、信息密度与可引用性。"
        "请仅输出改写后的正文，不要添加任何额外说明。\n\n原文：{TEXT}"
    )

    # 6. 选择 2-pass / 3-pass（✅ 第一阶段：让用户选择目标，从而决定 pass 数）
    # 你后续可以在 UI 用更友好的文案映射到这些 goal
    goal = (rewrite_goal or "auto").strip().lower()
    alias = {"seo_boost": "seo_push", "authority_boost": "credibility"}
    goal = alias.get(goal, goal)
    
    if goal in {"fast", "auto"}:
        pass_plan = [1, 2]  # ✅ 2-pass：先结构化与去噪，再做可引用强化
    elif goal in {"balanced"}:
        pass_plan = [1, 2, 3]  # 
    else:
        # credibility / seo_push / deep 等更“重加工”目标，用 3-pass
        pass_plan = [1, 2, 3]

    print(f"[geo_rewrite] rewrite_goal={rewrite_goal}, pass_plan={pass_plan}")

    # 7. 分块逻辑（沿用你的 split_into_chunks）
    if use_chunk:
        chunks = split_into_chunks(text, max_chars)
    else:
        chunks = [text]

    def _get_tpl(pass_id: int) -> str:
        """
        兼容两种 geo_prompts.json 结构：
        A) 扁平：geo_max_p1_zh / geo_max_p2_en ...
        B) 嵌套：geo_rewrite.pass1_core.zh / pass2_geo.en / pass3_query.auto ...
        """
        # 1) 先尝试扁平 key（兼容旧版或你未来可能做的扁平化）
        flat_key = f"geo_max_p{pass_id}_{lang_suffix}"
        flat_old = f"geo_max_{lang_suffix}"
        if isinstance(prompts, dict):
            tpl = prompts.get(flat_key) or prompts.get(flat_old)
            if tpl:
                return tpl

        # 2) 再尝试你现在的嵌套结构（geo_rewrite -> passX -> lang）
        grp = None
        if pass_id == 1:
            grp = "pass1_core"
        elif pass_id == 2:
            grp = "pass2_geo"
        elif pass_id == 3:
            grp = "pass3_query"

        try:
            nested = prompts.get("geo_rewrite", {})
            if isinstance(nested, dict) and grp in nested:
                block = nested.get(grp, {})
                if isinstance(block, dict):
                    tpl2 = block.get(lang_suffix) or block.get("auto")
                    if tpl2:
                        return tpl2
        except Exception:
            pass

        # 3) 最后兜底
        return (
            (prompts.get("geo_rewrite", {}).get("pass1_core", {}).get("zh") if isinstance(prompts, dict) else None)
            or (prompts.get("geo_max_auto") if isinstance(prompts, dict) else None)
            or (prompts.get("geo_max_zh") if isinstance(prompts, dict) else None)
            or fallback_prompt
        )



    def _len_ratio(src: str, out: str) -> float:
        # ✅ 近似 token：用字符长度兜底（你前后端已有粗略 token 计算时，可替换这里）
        s = len((src or "").strip())
        o = len((out or "").strip())
        if s <= 0:
            return 1.0
        return o / float(s)

    outs = []

    # 8. 逐块改写（每块按 pass 计划串行）
    for ck in chunks:
        ck = (ck or "").strip()
        if not ck:
            continue

        cur_text = ck
        for pass_id in pass_plan:
            tpl = _get_tpl(pass_id)

            try:
                prompt = tpl.format(TEXT=cur_text)
            except Exception as e:
                print(f"[geo_rewrite] 模板 format 失败，使用 fallback。错误：{e}")
                prompt = fallback_prompt.format(TEXT=cur_text)

            # 附加语言指令
            lang_instruction_clean = (lang_instruction or "").strip()
            if lang_instruction_clean:
                prompt = lang_instruction_clean + "\n\n" + prompt

            cur_text = (call_model(
                prompt,
                provider=provider,
                model=model,
                temperature=temperature,
            ) or "").strip()

        outs.append(cur_text)

    final = "\n\n".join(outs).strip()
    return final, text


# ============ 功能 2：GEO-Score 主观 + 客观评分（产品模式） ============

def geo_score(
    src_text: str,
    opt_text: str,
    model_ui: str = "Groq",
    samples: int = 1,
) -> Dict[str, Any]:
    """
    功能 2：针对内容进行 GEO-Score 评分（产品模式 single_text）。

    返回：GeoScore 字典（结构与 geo_evaluator.evaluate_geo_score 一致）
    """
    src_text = (src_text or "").strip()
    opt_text = (opt_text or "").strip()
    if not src_text or not opt_text:
        raise ValueError("缺少文本，无法评分。")

    provider = norm_provider(model_ui)
    model = DEFAULT_MODELS.get(provider) or next(iter(DEFAULT_MODELS.values()))

    score_json = evaluate_geo_score(
        model_name=model,
        query="",
        src_text=src_text,
        opt_text=opt_text,
        provider=provider,
        mode="single_text",
        samples=samples,
    )
    return score_json


def geo_score_report_html(
    project_title: str = "GEO-Max 评分报告",
    src_text: str = "",
    opt_text: str = "",
    score_json: Dict[str, Any] | None = None,
) -> str:
    """
    为网站 / 接口生成带 GEO-Score 的 HTML 报告。

    说明：
    - project_title 增加了默认值，方便被其他地方只用 3 个参数调用
    - 如果 score_json 为空，会抛出异常，避免生成空报告
    """
    if not score_json:
        raise ValueError("score_json 为空，请先调用 geo_score。")

    return render_report_html(project_title, src_text, opt_text, score_json)


# ============ 功能 3：论文模式评分 + 引用份额指标 ============

def geo_paper_score(
    query: str,
    answer_with_citations: str,
    model_ui: str = "Groq",
    samples: int = 1,
) -> Dict[str, Any]:
    """
    功能 3（主干）：论文模式评分。
    - 标准来自 templates/ 下的 *detailed.txt 模板
    - answer_with_citations 需要包含 [1][2]… 的引用标记

    返回：GeoScore 字典，mode 会是 "with_citations"。
    """
    query = (query or "").strip()
    answer_with_citations = (answer_with_citations or "").strip()
    if not answer_with_citations:
        raise ValueError("answer_with_citations 为空，无法评分。")

    provider = norm_provider(model_ui)
    model = DEFAULT_MODELS.get(provider) or next(iter(DEFAULT_MODELS.values()))

    score_json = evaluate_geo_score(
        model_name=model,
        query=query,
        src_text=query or answer_with_citations,  # 若无原文，则退化为自评分
        opt_text=answer_with_citations,
        provider=provider,
        mode="with_citations",
        samples=samples,
    )
    return score_json


def geo_paper_impression_single(
    answer_with_citations: str,
    n_sources: int,
    mode: str = "WordPos",
):
    """
    单次分布：某次回答中，各引用源的贡献份额。
    """
    if mode == "WordPos":
        dist = impression_wordpos_count(answer_with_citations, int(n_sources))
    elif mode == "Word":
        dist = impression_word_count(answer_with_citations, int(n_sources))
    else:
        dist = impression_pos_count(answer_with_citations, int(n_sources))
    return dist


def geo_paper_impression_delta(
    before: str,
    after: str,
    n_sources: int,
    target_idx: int,
    mode: str = "WordPos",
):
    """
    前后对比分布：比较优化前/后的引用份额变化。
    """
    if mode == "WordPos":
        res = compute_delta(before, after, int(n_sources), int(target_idx), mode="wordpos")
    elif mode == "Word":
        res = compute_delta(before, after, int(n_sources), int(target_idx), mode="word")
    else:
        res = compute_delta(before, after, int(n_sources), int(target_idx), mode="pos")
    return res


# ============ 功能 4：GEO-CoT 两段式 Markdown 工作流 ============

def geo_cot_stage1(
    user_question: str,
    brand_brief: str,
    must_expose: str,
    model_ui: str = "Groq",
    expo_hint: str = "",
    template_name: str = "cot_stage1",
) -> Tuple[str, str]:
    """
    Stage1：
    - 执行 geo_prompts/{template_name}.md
    - 生成可编辑的 Markdown（第一阶段分析 / 大纲 / 粗稿）

    返回: (stage1_markdown, prompt_used_for_debug)
    """

    # 输入校验：长度限制、防 prompt 注入
    validated = validate_cot_inputs(
        user_question=user_question,
        brand_brief=brand_brief,
        must_expose=must_expose,
        expo_hint=expo_hint,
    )

    # 读取 MD 模板
    tpl = _load_md_template(template_name)

    # 注入变量
    prompt = _fmt_md_template(
        tpl,
        USER_QUESTION=validated["user_question"],
        BRAND_BRIEF=validated["brand_brief"],
        MUST_EXPOSE=validated["must_expose"],
        EXPO_HINT=validated["expo_hint"],
        STAGE1_MD="",  # Stage1 无需注入上一阶段内容
    )

    provider = norm_provider(model_ui)
    model = DEFAULT_MODELS.get(provider) or next(iter(DEFAULT_MODELS.values()))

    out_md = call_model(
        prompt,
        provider=provider,
        model=model,
        temperature=0.2
    ) or ""

    return out_md.strip() or "> ⚠️ Stage1 未产出内容，请检查模板或输入。", prompt


def geo_cot_stage2(
    user_question: str,
    brand_brief: str,
    must_expose: str,
    stage1_md: str,
    model_ui: str = "Groq",
    expo_hint: str = "",
    template_name: str = "cot_stage2",
) -> Tuple[str, str]:
    """
    Stage2：
    - 读取 geo_prompts/{template_name}.md
    - 注入你编辑后的 Stage1 文档
    - 生成最终 Markdown

    返回: (stage2_markdown, prompt_used_for_debug)
    """
    tpl = _load_md_template(template_name)
    prompt = _fmt_md_template(
        tpl,
        USER_QUESTION=(user_question or "").strip(),
        BRAND_BRIEF=(brand_brief or "").strip(),
        MUST_EXPOSE=(must_expose or "").strip(),
        EXPO_HINT=(expo_hint or "").strip(),
        STAGE1_MD=(stage1_md or "").strip(),
    )

    provider = norm_provider(model_ui)
    model = DEFAULT_MODELS.get(provider) or next(iter(DEFAULT_MODELS.values()))

    out_md = call_model(prompt, provider=provider, model=model, temperature=0.2) or ""
    return out_md.strip() or "> ⚠️ Stage2 未产出内容，请检查 Stage1 文档或模板语法。", prompt
