# app.py — GEO-Max 多模型文本优化引擎（Groq/Gemini 可切换 · 带日志）
import os
import re
import time
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

import json  # ✅ 新增
import logging
import gradio as gr

import geo_core  # ✅ 新增：统一从 geo_core 调用四大核心能力
from geo_brand import build_brand_brief_structured
from geo_core import geo_cot_stage1  # 按你实际模块路径来

# =========================
# ★ 修复 Gradio JSON Schema 里 bool 导致的 APIInfoParseError
# =========================
try:
    # 一些 Gradio 版本在解析 Blocks 的 JSON Schema 时，
    # 会把 additionalProperties=True 直接丢给
    # gradio_client.utils._json_schema_to_python_type(True, defs)
    # 然后抛出：APIInfoParseError("Cannot parse schema True")
    import gradio_client.utils as gc_utils  # type: ignore

    _orig_json_schema_to_python_type = gc_utils._json_schema_to_python_type  # type: ignore[attr-defined]

    def _safe_json_schema_to_python_type(schema, defs=None):
        # 如果 schema 本身是布尔值（True / False），这里直接认为是 "Any" 类型，
        # 避免抛出 APIInfoParseError。
        if isinstance(schema, bool):
            return "Any"
        return _orig_json_schema_to_python_type(schema, defs)

    gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type  # type: ignore[attr-defined]
    logging.info("✔ Patched gradio_client.utils._json_schema_to_python_type for bool schema.")
except Exception as e:
    logging.warning(f"⚠ Failed to patch gradio_client.utils._json_schema_to_python_type: {e}")

# =========================
# 业务模块导入
# =========================
from geo_logger import log_run
from geo_report import render_report_html
from geo_impression import (
    impression_word_count,
    impression_pos_count,
    impression_wordpos_count,
    compute_delta,
)
from pipeline.inference_engine import call_model

# =========================
# 日志设置
# =========================
LOG_PATH = Path(__file__).with_name("geo_ui_debug.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("geo-ui")

logger.info("=== GEO-Max UI 启动 ===")
logger.info("Env check: GROQ_API_KEY set? %s",
            "YES" if os.getenv("GROQ_API_KEY") else "NO")
logger.info("Env check: GEMINI_API_KEY set? %s",
            "YES" if os.getenv("GEMINI_API_KEY") else "NO")


# =========================
# Prompt 配置加载（使用 geo_prompts.json · 版本2）
# =========================

# 默认兜底的中文 GEO-Max Prompt（当 geo_prompts.json 读取失败时使用）
DEFAULT_GEO_PROMPT_ZH = """
你是一名生成式引擎优化（GEO）专家，负责将下面的文本改写为更适合被大模型引用和总结的版本。请遵守以下规则：

1. 在不歪曲原意的前提下，提升逻辑清晰度与可读性；
2. 保留对“事实、时间、数据、专有名词、机构名称”等关键信息的准确表述；
3. 避免口语化和过度修辞，保持专业、克制、可靠的语气；
4. 如果原文逻辑存在缺口，可以通过“补足上下文衔接语”的方式弱化断裂感，但不要凭空杜撰事实；
5. 不要输出任何额外解释，只输出一版改写后的正文内容。

下面是需要改写的原文：

{TEXT}
""".strip()

GEO_PROMPTS: Dict[str, str] = {}

def run_cot_stage1_gradio(
    user_question: str,
    brand_name: str,
    category: str,
    target_audience: str,
    core_value: str,
    key_features: str,
    differentiators: str,
    use_cases: str,
    must_expose: str,
    expo_hint: str,
    model_ui: str,
) -> str:
    brand_structured = {
        "brand_name": brand_name,
        "category": category,
        "target_audience": target_audience,
        "core_value": core_value,
        "key_features": key_features,
        "differentiators": differentiators,
        "use_cases": use_cases,
        "must_expose": must_expose,
    }

    brand_brief_text = build_brand_brief_structured(brand_structured)

    stage1_md, _prompt_used = geo_cot_stage1(
        user_question=user_question,
        brand_brief=brand_brief_text,
        must_expose=must_expose,
        model_ui=model_ui,
        expo_hint=expo_hint,
    )

    return stage1_md

def _load_geo_prompts() -> None:
    """从 geo_prompts.json 读取 Prompt 模板。"""
    global GEO_PROMPTS
    prompt_file = Path(__file__).with_name("geo_prompts.json")
    try:
        if prompt_file.exists():
            GEO_PROMPTS = json.loads(prompt_file.read_text(encoding="utf-8"))
            logger.info("geo_prompts.json 加载成功，包含键：%s", list(GEO_PROMPTS.keys()))
        else:
            GEO_PROMPTS = {}
            logger.warning("geo_prompts.json 未找到，使用内置 DEFAULT_GEO_PROMPT_ZH 兜底。")
    except Exception as e:
        GEO_PROMPTS = {}
        logger.error("加载 geo_prompts.json 失败，将使用默认 Prompt：%s", e)

def build_geo_prompt(text: str, lang_instruction: str = "") -> str:
    """
    基于 geo_prompts.json 生成完整 Prompt。
    - 优先使用 geo_prompts.json 中的 'geo_max_zh' 模板；
    - 若读取失败，回退到 DEFAULT_GEO_PROMPT_ZH；
    - 将 {TEXT} 替换为待改写文本；
    - 若有 lang_instruction（语言要求），追加在末尾。
    """
    # 1) 选择模板
    tpl = GEO_PROMPTS.get("geo_max_zh") or DEFAULT_GEO_PROMPT_ZH

    # 2) 填充 {TEXT}
    try:
        prompt = tpl.format(TEXT=text)
    except Exception as e:
        logger.warning("geo_prompts.json 模板 format 失败：%s，改用简单拼接方式。", e)
        prompt = tpl + "\n\n【原文】\n" + text

    # 3) 附加语言指令
    lang_instruction = (lang_instruction or "").strip()
    if lang_instruction:
        prompt += "\n\n" + lang_instruction

    return prompt

# 模块加载时，预先读取一次 geo_prompts.json
_load_geo_prompts()


# =========================
# UI 常量
# =========================
APP_THEME = gr.themes.Soft()
APP_CSS = """
#wrap{max-width:1280px;margin:0 auto}
.tile{border:1px solid #eee;padding:14px;border-radius:12px}
.stack>*{margin-bottom:10px}
.tabs button{font-weight:600}
.footnote{font-size:12px;opacity:.7}
"""

# Provider 映射（UI显示名 -> inference provider key）
PROVIDER_MAP = {
    "Groq": "groq",
    "Gemini": "gemini",
    # 预留，暂不接入：
    "Grok": "groq",
    "通义千问": "qwen",
    "DeepSeek": "deepseek",
    "文心一言": "qwen",
}

# 默认模型
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-pro",
    "qwen": "qwen-turbo",
    "deepseek": "deepseek-chat",
}


def norm_provider(ui_name: str) -> str:
    p = PROVIDER_MAP.get(ui_name, "groq")
    logger.debug("norm_provider: ui=%s -> provider=%s", ui_name, p)
    return p

def _json_keys_to_str(obj):
    """
    递归地将 dict 的 key 转为 str，避免 gr.JSON / orjson 报
    'Dict key must be str' 的错误。
    """
    if isinstance(obj, dict):
        return {str(k): _json_keys_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_keys_to_str(v) for v in obj]
    return obj


def safe_progress(progress, v: float, desc: str = ""):
    try:
        progress(v, desc=desc)
    except Exception:
        # 有时候 gradio progress 在某些环境下会抛错，这里直接忽略
        pass


def _retry_call(fn, times=2, sleep_s=0.4):
    last = None
    for i in range(max(1, times)):
        try:
            logger.debug("retry_call: try %d/%d", i + 1, times)
            return fn()
        except Exception as e:
            last = e
            logger.warning("retry_call error: %s", e)
            time.sleep(sleep_s)
    if last:
        raise last


# =========================
# 工具函数：分块
# =========================
def split_into_chunks(text: str, max_chars: int) -> List[str]:
    text = text.strip()
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

    logger.debug("split_into_chunks: %d chunks", len(chunks))
    return chunks

def _build_lang_instruction(out_lang: str) -> str:
    """
    根据下拉框选择，生成给大模型看的“输出语言要求”说明。
    """
    if out_lang == "Chinese":
        return "请使用简体中文输出结果。"
    if out_lang == "English":
        return "Please answer in English."
    # Auto 或其他情况
    return "输出语言请与输入文本的主要语言保持一致。"



# =========================
# Tab1：GEO-Score 评分（壳层：转发到 geo_core）
# =========================

def run_geo(
    text: str,
    model_ui: str,
    use_chunk: bool,
    max_chars: int,
    out_lang: str = "Auto",
    progress=gr.Progress(),
):
    """
    Tab1 的内容改写功能：
    - 不再在 app.py 里直接调模型
    - 统一转发到 geo_core.geo_rewrite
    """
    logger.info(
        "run_geo (wrapper) called, model_ui=%s, use_chunk=%s, max_chars=%s",
        model_ui,
        use_chunk,
        max_chars,
    )

    safe_progress(progress, 0.05, "准备输入")

    text = (text or "").strip()
    if not text:
        return "⚠️ 请输入原文。", ""

    try:
        safe_progress(progress, 0.25, "调用核心引擎 geo_core.geo_rewrite")

        # ✅ 所有真正的业务逻辑都在 geo_core 里完成
        optimized, original = geo_core.geo_rewrite(
            text=text,
            model_ui=model_ui,
            use_chunk=use_chunk,
            max_chars=max_chars,
            out_lang=out_lang,
            temperature=0.2,
        )

        safe_progress(progress, 0.95, "完成")
        logger.info(
            "run_geo (wrapper) finished, length=%d",
            len(optimized or ""),
        )
        # outputs=[out_text, state_original]
        return optimized, original

    except Exception as e:
        logger.error("run_geo (wrapper) exception: %s", e)
        traceback.print_exc()
        msg = f"⚠️ run_geo 出错：{type(e).__name__} - {e}"
        # 第二个输出用原文兜底，避免前端 State 为空
        return msg, text

def run_score(
    original_text: str,
    optimized_text: str,
    model_ui: str,
    progress=gr.Progress(),
):
    """
    Tab1 的 GEO-Score 评分功能壳层：
    - 不再在 app.py 里直接构造 prompt、调模型
    - 统一调用 geo_core.geo_score
    - 输出两个结果：
        1）Markdown 形式的 JSON（给 score_md 用）
        2）原始 score_json（给 state_score 存起来，导出 HTML 时复用）
    """
    logger.info(
        "run_score (wrapper) called, model_ui=%s",
        model_ui,
    )

    safe_progress(progress, 0.05, "准备评分输入")

    original_text = (original_text or "").strip()
    optimized_text = (optimized_text or "").strip()
    if not original_text or not optimized_text:
        return (
            "⚠️ 请先完成内容改写，再进行评分。",
            {},  # state_score 为空 dict
        )

    try:
        safe_progress(progress, 0.20, "调用核心评分引擎 geo_core.geo_score")

        # ✅ 调用核心评分逻辑（产品模式 single_text）
        score_json = geo_core.geo_score(
            src_text=original_text,
            opt_text=optimized_text,
            model_ui=model_ui,
            samples=1,  # 先固定为 1，有需要再加 UI 选项
        )

        # ✅ 这里换成更友好的 Markdown 展示
        geo_score_value = score_json.get("geo_score", 0.0)

        dims = [
            ("相关性", "relevance"),
            ("影响力", "influence"),
            ("独特性", "uniqueness"),
            ("多样性", "diversity"),
            ("主观立场", "subjective_position"),
            ("主观密度", "subjective_count"),
            ("后续引导", "follow_up"),
        ]

        lines = []
        lines.append(f"### 🌐 GEO-Score 总览")
        lines.append("")
        lines.append(f"**总分：{geo_score_value:.1f} / 100**")
        lines.append("")
        lines.append("| 维度 | 分数 (1-5) |")
        lines.append("| ---- | ---------- |")

        for label, key in dims:
            v = score_json.get(key, None)
            if isinstance(v, (int, float)):
                lines.append(f"| {label} | {v:.1f} |")
            else:
                lines.append(f"| {label} | - |")

        lines.append("")
        lines.append("<details><summary>查看原始 JSON 结果</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(score_json, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("</details>")

        score_md = "\n".join(lines)

        safe_progress(progress, 0.95, "评分完成")
        logger.info("run_score (wrapper) finished")

        # 第二个输出仍然是原始 score_json，给 state_score 用
        return score_md, score_json

    except Exception as e:
        logger.error("run_score (wrapper) exception: %s", e)
        traceback.print_exc()
        msg_md = f"⚠️ run_score 出错：{type(e).__name__} - {e}"
        return msg_md, {}

def export_html_with_score(
    original_text: str,
    optimized_text: str,
    score_json: dict,
    progress=gr.Progress(),
):
    """
    根据已有评分结果，导出带评分报告的 HTML 文件：
    - original_text: 原文
    - optimized_text: GEO-Max 优化稿
    - score_json: run_score 生成并存到 state_score 的评分结果
    返回：
      1）file_html: HTML 文件路径（给 gr.File 使用）
      2）tip: 提示文案
    """
    logger.info("export_html_with_score called")

    safe_progress(progress, 0.05, "准备导出数据")

    original_text = (original_text or "").strip()
    optimized_text = (optimized_text or "").strip()
    if not original_text or not optimized_text:
        return None, "⚠️ 缺少原文或优化稿，无法导出报告。"

    if not isinstance(score_json, dict) or not score_json:
        return None, "⚠️ 尚未计算 GEO-Score，或评分结果为空。请先点击『计算 GEO-Score』。"

    try:
        safe_progress(progress, 0.25, "生成评分报告 HTML")

        # ✅ 调用核心的 HTML 报告生成逻辑
        html_content = geo_core.geo_score_report_html(
            project_title="GEO-Max 评分报告（产品模式）",
            src_text=original_text,
            opt_text=optimized_text,
            score_json=score_json,
        )

        safe_progress(progress, 0.60, "写入临时 HTML 文件")

        # 写入临时文件，交给 gr.File 组件下载
        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".html",
            prefix="geo_max_report_",
        )
        tmp_path = tmp.name
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        tmp.close()

        safe_progress(progress, 0.95, "导出完成")
        logger.info("export_html_with_score finished, path=%s", tmp_path)

        tip = "✅ 报告已生成，请点击下方链接下载。"
        return tmp_path, tip

    except Exception as e:
        logger.error("export_html_with_score exception: %s", e)
        traceback.print_exc()
        tip = f"⚠️ 导出报告时出错：{type(e).__name__} - {e}"
        return None, tip

# =========================
# Tab2：Impression
# =========================
def run_impression_single(answer: str, n_sources: int, mode_sel: str):
    """
    Tab2 单次分布：
    - 现在改为通过 geo_core.geo_paper_impression_single 统一实现
    - UI 仍然返回「提示语 + JSON 分布」
    """
    logger.info("run_impression_single (wrapper) called, mode=%s", mode_sel)

    answer = (answer or "").strip()
    if not answer:
        return "⚠️ 请先在左侧输入带 [1][2]… 的答案文本。", {}

    try:
        dist = geo_core.geo_paper_impression_single(
            answer_with_citations=answer,
            n_sources=int(n_sources or 1),
            mode=mode_sel or "WordPos",
        )
        dist = _json_keys_to_str(dist)
        return "✅ 计算完成", dist

    except Exception as e:
        logger.error("run_impression_single (wrapper) exception: %s", e)
        traceback.print_exc()
        return f"⚠️ 失败：{type(e).__name__} - {e}", {}


def run_impression_delta(
    before: str,
    after: str,
    n_sources: int,
    target_idx: int,
    mode_sel: str,
):
    """
    Tab2 前后版本 Δ：
    - 统一转发给 geo_core.geo_paper_impression_delta
    """
    logger.info("run_impression_delta (wrapper) called, mode=%s", mode_sel)

    before = (before or "").strip()
    after = (after or "").strip()
    if not before or not after:
        return "⚠️ 请先在右侧输入 Before / After 两个答案。", {}

    try:
        res = geo_core.geo_paper_impression_delta(
            before=before,
            after=after,
            n_sources=int(n_sources or 1),
            target_idx=int(target_idx or 1),
            mode=mode_sel or "WordPos",
        )
        res = _json_keys_to_str(res)
        return "✅ 计算完成", res
    except Exception as e:
        logger.error("run_impression_delta (wrapper) exception: %s", e)
        traceback.print_exc()
        return f"⚠️ 失败：{type(e).__name__} - {e}", {}


# =========================
# Tab3：GEO-CoT 两段式 Markdown
# =========================

def _save_md_to_file(md_text: str, filename: str):
    try:
        tmpdir = tempfile.gettempdir()
        path = os.path.join(tmpdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_text or "")
        logger.info("md saved to %s", path)
        return path
    except Exception as e:
        logger.error("save_md_to_file error: %s", e)
        return None

def run_stage1_markdown(
    q: str,
    brand_name: str,
    category: str,
    target_audience: str,
    core_value: str,
    key_features: str,
    differentiators: str,
    use_cases: str,
    must_expose: str,
    expo_hint: str,
    model_ui: str,
    progress=gr.Progress(),
):
    """
    Stage1 壳层（结构化 brand_brief 版本）：
    - 接收结构化品牌信息字段
    - 使用 geo_brand.build_brand_brief_structured 组装 brand_brief 文本
    - 调用 geo_core.geo_cot_stage1 生成可编辑 Markdown
    - 继续返回：Stage1 Markdown、debug prompt 片段、下载路径、提示语
    """
    logger.info("run_stage1_markdown (wrapper) called, model_ui=%s", model_ui)

    q = (q or "").strip()
    must_expose = (must_expose or "").strip()
    expo_hint = (expo_hint or "").strip()

    if not q:
        return "⚠️ 请先填写『目标问题』。", "", None, "⚠️ 缺少目标问题"

    # 结构化品牌信息 → brand_brief 文本
    brand_structured = {
        "brand_name": brand_name or "",
        "category": category or "",
        "target_audience": target_audience or "",
        "core_value": core_value or "",
        "key_features": key_features or "",
        "differentiators": differentiators or "",
        "use_cases": use_cases or "",
        "must_expose": must_expose or "",
    }
    brand_ctx = build_brand_brief_structured(brand_structured).strip()

    if not brand_ctx:
        return (
            "⚠️ 请至少填写品牌基础信息（如品牌名称、行业/品类等）。",
            "",
            None,
            "⚠️ 品牌信息为空",
        )

    try:
        safe_progress(progress, 0.10, "调用 GEO-CoT Stage1 引擎")

        out_md, prompt_used = geo_core.geo_cot_stage1(
            user_question=q,
            brand_brief=brand_ctx,
            must_expose=must_expose,
            model_ui=model_ui,
            expo_hint=expo_hint,
            template_name="cot_stage1",
        )

        if not out_md.strip():
            out_md = "⚠️ Stage1 未产出内容，请重试或检查模板。"

        dl_path = _save_md_to_file(out_md, "geo_stage1_output.md")

        safe_progress(progress, 0.95, "Stage1 完成")
        return out_md, prompt_used[:1200], dl_path, "✅ Stage1 完成：可编辑后进入 Stage 2。"

    except Exception as e:
        logger.error("run_stage1_markdown (wrapper) exception: %s", e)
        traceback.print_exc()
        return f"> ⚠️ Stage1 出错：{type(e).__name__} - {e}", "", None, "⚠️ 执行失败"


def run_stage2_markdown(
    q: str,
    brand_name: str,
    category: str,
    target_audience: str,
    core_value: str,
    key_features: str,
    differentiators: str,
    use_cases: str,
    must_expose: str,
    expo_hint: str,
    model_ui: str,
    stage1_md: str,
    progress=gr.Progress(),
):
    """
    Stage2 壳层（结构化 brand_brief 版本）：
    - 再次使用结构化品牌信息组装 brand_brief 文本
    - 将（可能已被用户编辑过的）Stage1 Markdown 注入模板，生成最终 Markdown
    - 返回最终 Markdown、debug prompt、.md 下载路径、提示
    """
    logger.info("run_stage2_markdown (wrapper) called, model_ui=%s", model_ui)

    q = (q or "").strip()
    must_expose = (must_expose or "").strip()
    expo_hint = (expo_hint or "").strip()
    stage1_md = (stage1_md or "").strip()

    if not stage1_md:
        return "> ⚠️ 请先完成 Stage1，并在必要时进行编辑。", "", None, "⚠️ 缺少 Stage1 文本"

    # 结构化品牌信息 → brand_brief 文本
    brand_structured = {
        "brand_name": brand_name or "",
        "category": category or "",
        "target_audience": target_audience or "",
        "core_value": core_value or "",
        "key_features": key_features or "",
        "differentiators": differentiators or "",
        "use_cases": use_cases or "",
        "must_expose": must_expose or "",
    }
    brand_ctx = build_brand_brief_structured(brand_structured).strip()

    if not brand_ctx:
        return (
            "> ⚠️ 品牌信息为空，请至少填写品牌名称 / 行业 / 目标人群等基础信息。",
            "",
            None,
            "⚠️ 品牌信息为空",
        )

    try:
        safe_progress(progress, 0.15, "调用 GEO-CoT Stage2 引擎")

        out_md, prompt_used = geo_core.geo_cot_stage2(
            user_question=q,
            brand_brief=brand_ctx,
            must_expose=must_expose,
            stage1_md=stage1_md,
            model_ui=model_ui,
            expo_hint=expo_hint,
            template_name="cot_stage2",
        )

        if not out_md.strip():
            out_md = "> ⚠️ Stage2 未产出内容，请检查 Stage1 文档或模板语法。"

        dl_path = _save_md_to_file(out_md, "geo_stage2_output.md")

        safe_progress(progress, 0.95, "Stage2 完成")
        return out_md, prompt_used[:1200], dl_path, "✅ Stage2 完成：右侧可复制/下载。"

    except Exception as e:
        logger.error("run_stage2_markdown (wrapper) exception: %s", e)
        traceback.print_exc()
        return f"> ⚠️ Stage2 出错：{type(e).__name__} - {e}", "", None, "⚠️ 执行失败"



# =========================
# Gradio UI
# =========================
with gr.Blocks(
    title="GEO-Max 多模型文本优化引擎（含评分）",
    analytics_enabled=False,
    theme=APP_THEME,
    css=APP_CSS,
) as demo:
    with gr.Group(elem_id="wrap"):
        gr.Markdown("### GEO-Max · 生成式引擎优化\n极简、稳定：内容改写 + 自动评分。")

        with gr.Tabs(elem_classes=["tabs"]):
            # ---- Tab 1 ----
            with gr.Tab("⚙️ 产品模式（质量评分）"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["stack"]):
                        with gr.Group(elem_classes=["tile"]):
                            inp_text = gr.Textbox(label="✍️ 输入原文", lines=8, show_copy_button=True)
                            model_dd = gr.Dropdown(
                                choices=["Groq", "Gemini", "Grok", "通义千问", "DeepSeek", "文心一言"],
                                value="Groq",
                                label="🧩 选择模型",
                            )
                            # 🌐 输出语言选择
                            lang_dd = gr.Dropdown(
                                choices=["Auto", "Chinese", "English","Spanish","French","Japanese","Korean","German"],
                                value="Auto",
                                label="🌐 Output language",
                            )
                            use_chunk = gr.Checkbox(value=True, label="自动分块（建议开启）")
                            max_chars = gr.Slider(800, 6000, value=2400, step=200, label="单次最大字符数")

                            btn_run = gr.Button("🚀 生成 GEO-Max 优化稿", variant="primary")
                            btn_clear = gr.Button("🧹 清空")
                            gr.Markdown("<div class='footnote'>提示：我们不保存你的文本；评分仅在本地会话内计算。</div>")

                    with gr.Column(scale=1, elem_classes=["stack"]):
                        with gr.Group(elem_classes=["tile"]):
                            out_text = gr.Textbox(
                                label="📈 GEO-Max 优化结果",
                                lines=12,
                                show_copy_button=True,
                            )
                            btn_score = gr.Button("📊 计算 GEO-Score（自动评分）")
                            score_md = gr.Markdown("")
                            with gr.Row():
                                btn_html = gr.Button("导出带评分报告（HTML）")
                                file_html = gr.File(label="下载报告", visible=False)
                            tip = gr.Markdown("")

                state_original = gr.State("")
                state_optimized = gr.State("")
                state_score = gr.State({})

                btn_run.click(
                    fn=run_geo,
                    inputs=[inp_text, model_dd, use_chunk, max_chars, lang_dd],
                    outputs=[out_text, state_original],
                    queue=False,
                )

                out_text.change(
                    lambda x: x,
                    inputs=out_text,
                    outputs=state_optimized,
                    queue=False,
                )
                btn_score.click(
                    fn=run_score,
                    inputs=[state_original, state_optimized, model_dd],
                    outputs=[score_md, state_score],
                    queue=False,
                )
                btn_html.click(
                    fn=export_html_with_score,
                    inputs=[state_original, state_optimized, state_score],
                    outputs=[file_html, tip],
                    queue=False,
                )
                btn_clear.click(
                    lambda: ("", "", "", "", None),
                    None,
                    [inp_text, out_text, score_md, tip, file_html],
                    queue=False,
                )

            # ---- Tab 2 ----
            with gr.Tab("📘 论文模式（with citations）"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["stack"]):
                        with gr.Group(elem_classes=["tile"]):
                            n_sources = gr.Number(value=3, label="来源总数（N）", precision=0)
                            mode_sel = gr.Dropdown(
                                choices=["WordPos", "Word", "Pos"],
                                value="WordPos",
                                label="指标模式",
                            )
                            answer_once = gr.Textbox(
                                label="单次分布：带 [1][2]… 的答案（任一段）",
                                lines=6,
                                show_copy_button=True,
                            )
                            btn_once = gr.Button("📊 计算单次分布", variant="secondary")
                            msg_once = gr.Markdown("")
                            dist_once = gr.JSON(label="分布（和=1）")
                    with gr.Column(scale=1, elem_classes=["stack"]):
                        with gr.Group(elem_classes=["tile"]):
                            before_ans = gr.Textbox(
                                label="Before：带引用的答案",
                                lines=6,
                                show_copy_button=True,
                            )
                            after_ans = gr.Textbox(
                                label="After：带引用的答案",
                                lines=6,
                                show_copy_button=True,
                            )
                            target_idx = gr.Number(value=1, label="目标来源索引（1..N）", precision=0)
                            btn_delta = gr.Button("📈 计算 Δ 提升（After - Before）", variant="primary")
                            msg_delta = gr.Markdown("")
                            res_delta = gr.JSON(label="结果（含 dist_before / dist_after / delta）")

                btn_once.click(
                    fn=run_impression_single,
                    inputs=[answer_once, n_sources, mode_sel],
                    outputs=[msg_once, dist_once],
                    queue=False,
                )
                btn_delta.click(
                    fn=run_impression_delta,
                    inputs=[before_ans, after_ans, n_sources, target_idx, mode_sel],
                    outputs=[msg_delta, res_delta],
                    queue=False,
                )

            # ---- Tab 3 ----
            with gr.Tab("🧠 GEO-CoT（两段式·Markdown 模板）"):
                with gr.Row():
                    # 左侧：输入区（结构化品牌信息）
                    with gr.Column(scale=1, elem_classes=["stack"]):
                        with gr.Group(elem_classes=["tile"]):
                            md_q = gr.Textbox(
                                label="🎯 目标问题",
                                placeholder="例如：推荐几款适合中小企业的 CRM 软件",
                                lines=2,
                            )

                            gr.Markdown("#### 🏷️ 品牌信息（结构化填写）")
                            md_brand_name = gr.Textbox(
                                label="品牌名称（brand_name）",
                                placeholder="例如：超兔 CRM / GEO-Max / ……",
                                lines=1,
                            )
                            md_category = gr.Textbox(
                                label="所在行业 / 品类（category）",
                                placeholder="例如：SaaS / CRM / 制造业数字化 / ……",
                                lines=1,
                            )
                            md_target_audience = gr.Textbox(
                                label="目标人群（target_audience）",
                                placeholder="例如：制造业中小企业老板 / 市场负责人 / ……",
                                lines=1,
                            )
                            md_core_value = gr.Textbox(
                                label="核心价值主张（core_value）",
                                placeholder="一句话解释：这个品牌凭什么值得被推荐？",
                                lines=2,
                            )
                            md_key_features = gr.Textbox(
                                label="核心功能 / 模块（key_features，每行一条）",
                                placeholder="例如：\n- 销售漏斗管理\n- 客户全生命周期跟踪\n- 进销存一体化",
                                lines=3,
                            )
                            md_differentiators = gr.Textbox(
                                label="差异化亮点（differentiators，每行一条）",
                                placeholder="例如：\n- 支持“销售-进销存-生产-财务”一体化\n- 制造业场景深度适配",
                                lines=3,
                            )
                            md_use_cases = gr.Textbox(
                                label="典型使用场景（use_cases，每行一条）",
                                placeholder="例如：\n- 订单驱动生产\n- 多门店分仓发货\n- 大区+经销商协同",
                                lines=3,
                            )
                            md_must_expose = gr.Textbox(
                                label="期望露出字段（must_expose，可选）",
                                placeholder="例如：超兔CRM, 表情包姨姨公众号, 免费试用30天",
                                lines=2,
                            )
                            md_expo_hint = gr.Textbox(
                                label="补充提示（expo_hint，可选，仅给模型看）",
                                placeholder="例如：更偏向实用主义口径；避免过度吹捧；突出一体化链路优势。",
                                lines=2,
                            )

                            md_model = gr.Dropdown(
                                choices=["Groq", "Gemini", "Grok", "DeepSeek", "通义千问", "文心一言"],
                                value="Groq",
                                label="🧩 模型",
                            )

                        with gr.Group(elem_classes=["tile"]):
                            gr.Markdown("#### Stage 1：执行 `cot_stage1.md` → 生成 Markdown（可编辑）")
                            btn_s1 = gr.Button("🚀 运行 Stage 1（Markdown）", variant="primary")
                            s1_md_editable = gr.Textbox(
                                label="📝 Stage1 产出（可编辑 Markdown）",
                                lines=18,
                                show_copy_button=True,
                            )
                            s1_prompt_dbg = gr.Textbox(
                                label="调试：Stage1 最终提示词片段（只读）",
                                lines=5,
                                interactive=False,
                            )
                            s1_download = gr.DownloadButton(label="下载 Stage1 .md", value=None)
                            s1_tip = gr.Markdown("")
                            btn_confirm_s2 = gr.Button(
                                "✅ 使用上方 Markdown 进入 Stage 2",
                                variant="secondary",
                            )

                    # 右侧：Stage2 结果展示
                    with gr.Column(scale=1, elem_classes=["stack"]):
                        with gr.Group(elem_classes=["tile"]):
                            gr.Markdown("#### Stage 2：执行 `cot_stage2.md`（注入你编辑后的 Stage1 文档）")
                            s2_md_view = gr.Markdown(value="> 运行 Stage 2 后，这里显示最终 Markdown")
                            s2_prompt_dbg = gr.Textbox(
                                label="调试：Stage2 最终提示词片段（只读）",
                                lines=5,
                                interactive=False,
                            )
                            s2_download = gr.DownloadButton(label="下载 Stage2 .md", value=None)
                            s2_tip = gr.Markdown("")

                # 事件绑定：Stage1
                btn_s1.click(
                    run_stage1_markdown,
                    inputs=[
                        md_q,
                        md_brand_name,
                        md_category,
                        md_target_audience,
                        md_core_value,
                        md_key_features,
                        md_differentiators,
                        md_use_cases,
                        md_must_expose,
                        md_expo_hint,
                        md_model,
                    ],
                    outputs=[s1_md_editable, s1_prompt_dbg, s1_download, s1_tip],
                    show_progress=True,
                )

                # 事件绑定：Stage2
                btn_confirm_s2.click(
                    run_stage2_markdown,
                    inputs=[
                        md_q,
                        md_brand_name,
                        md_category,
                        md_target_audience,
                        md_core_value,
                        md_key_features,
                        md_differentiators,
                        md_use_cases,
                        md_must_expose,
                        md_expo_hint,
                        md_model,
                        s1_md_editable,
                    ],
                    outputs=[s2_md_view, s2_prompt_dbg, s2_download, s2_tip],
                    show_progress=True,
                )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
