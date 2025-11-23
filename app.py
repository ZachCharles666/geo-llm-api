# app.py — GEO-Max 多模型文本优化引擎（评分增强版 · 极简UI）
import os, json, requests, re, uuid, textwrap, tempfile
from datetime import datetime
from typing import Dict, Any, Tuple, List

import logging, traceback, time, sys
from pathlib import Path

import gradio as gr
from openai import OpenAI

# === 本地模块 ===
from geo_logger import log_run
from geo_evaluator import evaluate_geo_score
from geo_report import render_report_html
from geo_impression import (
    impression_word_count,
    impression_pos_count,
    impression_wordpos_count,
    compute_delta
)

    
# === 调试/日志设置 ===
DEBUG_GEO_COT = True  # 临时打开；定位完成后可置 False
_log_path = Path(__file__).with_name("geo_cot_debug.log")
logging.basicConfig(
    level=logging.DEBUG if DEBUG_GEO_COT else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(_log_path, encoding="utf-8")]
)
def dbg(tag, **kw):
    if not DEBUG_GEO_COT: 
        return
    safe = {k: (str(v)[:800] + "…[trunc]" if isinstance(v, str) and len(v) > 800 else v) for k, v in kw.items()}
    logging.debug(f"[GEO-COT:{tag}] {safe}")

def log_exc(tag):
    logging.error(f"[GEO-COT:{tag}] EXC={traceback.format_exc()}")
    
def safe_progress(p, *args, **kwargs):
    """安全调用 gr.Progress；避免 if p: 触发 __len__ 导致 IndexError"""
    try:
        if p is not None:
            p(*args, **kwargs)
    except Exception as _e:
        # 可选：打印一行调试，不影响主流程
        print("[PROGRESS-IGNORED]", repr(_e))


# --- PATCH START ---
import gradio_client.utils as grc_utils
def _safe_json_schema_to_python_type(schema, defs=None):
    try:
        if isinstance(schema, bool):
            return "Any"
        return grc_utils._json_schema_to_python_type(schema, defs)
    except Exception:
        return "Any"
grc_utils.json_schema_to_python_type = _safe_json_schema_to_python_type
# --- PATCH END ---

def render_cot_markdown(data: dict) -> str:
    """
    将 GEO-CoT 结果转为可读 Markdown（优先渲染 evidence_chain_v2）
    显示结构：
      # 逻辑链
      # 证据链（按节点）
        ## <节点1>
        - 来源类型：<source_type>
        - 可复述事实：<claim>
        - 如何验证：<how_to_verify>
        - 建议资产：<asset>
        > ⚠️ 证据缺口：<gaps>
      # 标题（与证据链节点对应）
    """
    if not isinstance(data, dict):
        return "> 暂无数据"

    def _s(x):  # 安全取字符串
        return (x or "").strip()

    def _render_proof_v2(pf: dict) -> list[str]:
        lines = []
        st = _s(pf.get("source_type"))
        cl = _s(pf.get("claim"))
        hv = _s(pf.get("how_to_verify"))
        asst = _s(pf.get("asset"))
        gp = _s(pf.get("gaps"))
        if st:   lines.append(f"- 来源类型：`{st}`")
        if cl:   lines.append(f"- 可复述事实：{cl}")
        if hv:   lines.append(f"- 如何验证：{hv}")
        if asst: lines.append(f"- 建议资产：{asst}")
        if gp:   lines.append(f"> ⚠️ 证据缺口：{gp}")
        return lines

    def _render_proof_v1(ev: dict) -> list[str]:
        """兼容旧版 evidence_chain -> 简要转写为 v2 风格"""
        if not isinstance(ev, dict):
            return []
        lines = []
        # 简化展示：把常见 key 摘要化
        mapping_keys = ["official","category_tags","products","scenarios",
                        "media_refs","tech_specs","third_party","structure"]
        picked = [k for k in mapping_keys if _s(ev.get(k))]
        if picked:
            lines.append(f"- 来源类型：`mixed(v1)`")
            # 合成一句可复述事实的占位
            lines.append("- 可复述事实：该节点包含官方简介/产品/媒体/技术/第三方等多源证据（v1）")
            lines.append("- 如何验证：按字段到对应页面或文档核对（About/产品页/媒体页/白皮书/百科等）")
            lines.append("- 建议资产：About、FAQ、HowTo、Product JSON-LD、对比表")
        gaps = ev.get("gaps")
        if isinstance(gaps, list) and gaps:
            lines.append(f"> ⚠️ 证据缺口：{'；'.join(str(x) for x in gaps)}")
        elif isinstance(gaps, str) and gaps.strip():
            lines.append(f"> ⚠️ 证据缺口：{gaps.strip()}")
        return lines

    md = []

    # 1) 逻辑链
    logic = data.get("logic_chain") or []
    md.append("# 🧠 逻辑链")
    if logic:
        for i, step in enumerate(logic, 1):
            md.append(f"{i}. {step}")
    else:
        md.append("- （空）")

    # 2) 证据链（优先 v2）
    md.append("\n# 🔗 证据链（按节点）")
    ev2 = data.get("evidence_chain_v2") or []
    ev1 = data.get("evidence_chain") or []

    if ev2:
        for i, item in enumerate(ev2, 1):
            node = _s(item.get("node")) or f"节点{i}"
            proof = item.get("proof") or {}
            md.append(f"\n## {i}. {node}")
            lines = _render_proof_v2(proof)
            md.extend(lines if lines else ["- （该节点暂无可展示字段）"])
    elif ev1:
        # 仅当没有 v2 时才降级展示 v1
        for i, item in enumerate(ev1, 1):
            node = _s(item.get("node")) or f"节点{i}"
            md.append(f"\n## {i}. {node}")
            ev = item.get("evidence") or {}
            lines = _render_proof_v1(ev)
            md.extend(lines if lines else ["- （该节点暂无可展示字段）"])
    else:
        md.append("- （空）")

    # 3) 标题（与证据链节点对应）
    md.append("\n# 🏷️ 标题（与证据链节点对应）")
    titles = data.get("titles_by_node") or []
    # 建立 node -> titles 映射
    node2titles = {}
    for t in titles:
        n = _s(t.get("node"))
        arr = t.get("titles") or []
        if n:
            node2titles[n] = [str(x) for x in arr if _s(x)]

    # 以“已展示的证据链顺序”为准输出标题；若证据为空则按逻辑链顺序
    order_nodes = []
    if ev2:
        order_nodes = [ _s(x.get("node")) or f"节点{i+1}" for i, x in enumerate(ev2) ]
    elif ev1:
        order_nodes = [ _s(x.get("node")) or f"节点{i+1}" for i, x in enumerate(ev1) ]
    elif logic:
        order_nodes = list(logic)

    if order_nodes:
        any_title = False
        for i, node in enumerate(order_nodes, 1):
            arr = node2titles.get(node, [])
            if arr:
                any_title = True
                md.append(f"\n## {i}. {node}")
                for s in arr:
                    md.append(f"- {s}")
        if not any_title:
            md.append("- （证据链节点未找到对应标题）")
    else:
        md.append("- （空）")

    # 不在阅读视图里渲染 raw_text（仅调试框显示）
    return "\n".join(md)




def export_json_file(data: dict, filename: str = "geo_cot_export.json") -> str:
    """把 JSON 落成临时文件，返回路径供 DownloadButton 使用"""
    try:
        tmpdir = tempfile.gettempdir()
        path = os.path.join(tmpdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    except Exception as e:
        return ""

_CIT_RE = re.compile(r"\[\s*(\d+)\s*\]")

# ===== 主题与 CSS（只做视觉层，零侵入业务） =====
APP_THEME = gr.themes.Monochrome(
    primary_hue="indigo", secondary_hue="slate"
).set(
    button_primary_background_fill="linear-gradient(180deg,#6366f1,#4f46e5)",
    button_primary_background_fill_hover="linear-gradient(180deg,#4f46e5,#4338ca)",
    button_primary_text_color="#fff"
)

APP_CSS = """
/* 统一浅灰背景，移除割裂感 */
html, body { height:100%; background:#f5f7fb; color:#0f172a; }
.gradio-container { background:transparent !important;
  font:16px/1.72 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial; }

/* 版心 1280 */
#wrap{ max-width:1280px; margin:0 auto; padding:28px 20px 64px }

/* Tabs：更清晰的选中线与悬停 */
.tabs > div > button{
  border-bottom:2px solid transparent !important; border-radius:0 !important;
}
.tabs > div > button[aria-selected="true"]{
  border-bottom-color:#6366f1 !important; background:#ffffff !important;
}
.tabs > div > button:hover{ background:#eef1ff !important; }

/* 两列外壳：tile 卡片 */
.tile{
  background:#fff; border:1px solid #e7e9ee; border-radius:14px; padding:16px;
  box-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 22px rgba(15,23,42,.04);
}
.tile:hover{ box-shadow:0 2px 6px rgba(15,23,42,.06), 0 12px 32px rgba(15,23,42,.06) }

/* 列内垂直间距 */
.stack > * + *{ margin-top:14px }

/* 表单聚焦态 */
textarea, input, select{
  border-radius:10px !important; border:1px solid #e6e8ef !important; background:#fff !important;
}
textarea:focus, input:focus, select:focus{
  outline:0 !important; border-color:#6366f1 !important; box-shadow:0 0 0 3px rgba(99,102,241,.18) !important;
}

/* 按钮：默认与主行动 */
.gradio-button{
  padding:11px 16px !important; border-radius:10px !important;
  border:1px solid #e7e9ee !important; background:#ffffff !important; color:#0f172a !important;
  transition:transform .12s ease, box-shadow .12s ease, background .12s ease;
}
.gradio-button:hover{ transform: translateY(-1px); box-shadow:0 3px 10px rgba(15,23,42,.08); }

.gradio-button.primary,
.gradio-button[data-testid="button-primary"]{
  background:linear-gradient(180deg,#6366f1,#4f46e5) !important; color:#fff !important; border:none !important;
  box-shadow:0 6px 16px rgba(79,70,229,.30);
}
.gradio-button.primary:hover,
.gradio-button[data-testid="button-primary"]:hover{
  transform: translateY(-1px); box-shadow:0 8px 24px rgba(79,70,229,.34);
}
.gradio-button.primary:active,
.gradio-button[data-testid="button-primary"]:active{
  transform: translateY(0); box-shadow:0 4px 12px rgba(79,70,229,.28);
}

/* 行间距：列更疏一些 */
.gradio-row{ gap:24px !important; }

/* 轻量脚注 */
.footnote{ margin-top:10px; color:#6b7280; font-size:12px; }

/* 隐藏 “Built with Gradio” 页脚 */
footer, #footer, .gradio-container .footer, .built-with, .svelte-1ipelgc { display:none !important; }
"""


def _first_choice(choices):
    """统一安全地取第一条 choice；无则返回 None。兼容 list/tuple/pydantic 对象/None。"""
    try:
        if not choices:
            return None
        # 兼容 pydantic/SDK 对象：优先用迭代器
        it = iter(choices)
        return next(it, None)
    except Exception:
        # 某些对象可 __len__ 但不可迭代；退回索引并保护越界
        try:
            return choices[0] if getattr(choices, "__len__", None) and len(choices) > 0 else None
        except Exception:
            return None

# ===== 引用规整（论文模式需要） =====
def _has_std_cite(text: str) -> bool:
    return bool(_CIT_RE.search(text))

def _needs_fix(text: str) -> bool:
    if "来源待补" in text or "【" in text or "（" in text:  # 中文括号/占位
        return True
    ids = [int(m.group(1)) for m in _CIT_RE.finditer(text)]
    return bool(ids and sorted(set(ids)) != list(range(1, max(ids)+1)))

def normalize_citation_markers(text: str) -> str:
    t = re.sub(r"【\s*(\d+)\s*】", r"[\1]", text)
    t = re.sub(r"（\s*(\d+)\s*）", r"[\1]", t)
    if "来源待补" in t and not _has_std_cite(t):  # 无编号但有占位
        t = t.replace("来源待补", "[1]")
    return t

def maybe_citation_enrich(text: str) -> Tuple[str, bool]:
    if _has_std_cite(text) and not _needs_fix(text):
        return text, False
    t = normalize_citation_markers(text)
    return t, True

def _stringify_keys(d: Dict[int, float]) -> Dict[str, float]:
    return {str(k): float(v) for k, v in (d or {}).items()}

def run_impression_single(answer_with_citations: str, n_sources, mode: str):
    txt = (answer_with_citations or "").strip()
    if not txt:
        return "⚠️ 请输入包含 [1][2]… 的答案文本。", {}
    try:
        n = int(n_sources or 1)
        mode_l = (mode or "WordPos").lower()
        if mode_l.startswith("wordpos"):
            dist, used = impression_wordpos_count(txt, n), "WordPos"
        elif mode_l.startswith("word"):
            dist, used = impression_word_count(txt, n), "Word"
        else:
            dist, used = impression_pos_count(txt, n), "Pos"
        dist_str = _stringify_keys(dist)
        if not dist_str:
            return "⚠️ 未解析到任何 [x] 引用。请确认文本里有 [1][2]… 标注，且 N≥最大编号。", {}
        return f"✅ {used} 分布计算完成（各份额相加=1）。", dist_str
    except Exception as e:
        return f"❌ 解析失败：{e}", {}

def run_impression_delta(before_answer: str, after_answer: str, n_sources, target_idx, mode: str):
    before_txt = (before_answer or "").strip()
    after_txt  = (after_answer or "").strip()
    if not before_txt or not after_txt:
        return "⚠️ 请同时填写 Before 与 After 的“带引用答案”。", {}
    try:
        n = int(n_sources or 1); t = int(target_idx or 1)
        res = compute_delta(before_txt, after_txt, n, t, mode or "WordPos")
        res_out: Dict[str, Any] = {
            "mode": res.get("mode"),
            "n_sources": int(res.get("n_sources", n)),
            "target_idx": int(res.get("target_idx", t)),
            "dist_before": _stringify_keys(res.get("dist_before", {})),
            "dist_after": _stringify_keys(res.get("dist_after", {})),
            "delta": float(res.get("delta", 0.0)),
        }
        msg = f"✅ 目标来源 [{res_out['target_idx']}] 的 {res_out['mode']} 份额提升 Δ = {res_out['delta']:+.4f}（After - Before）。"
        if not res_out["dist_before"] or not res_out["dist_after"]:
            msg = "⚠️ 未解析到 [x] 引用或 N 设置过小，请确认文本中存在 [1][2]… 且 N≥最大编号。"
        return msg, res_out
    except Exception as e:
        return f"❌ 计算失败：{e}", {}

# -----------------------------
# 统一保存目录
SAVE_DIR = os.path.join(os.path.expanduser("~"), "GEO-Reports")
os.makedirs(SAVE_DIR, exist_ok=True)
# -----------------------------
# === 两段式 / 一段式 Prompt 外置化版 ===
def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def load_md_prompt_file(name: str) -> str:
    """
    优先从 geo_prompts_md/{name}.md 读取 Markdown Prompt。
    若不存在，则尝试读取 geo_prompts/{name}.json 的 template 字段。
    均不存在时，返回空串，交由上层用 fallback 兜底。
    """
    base_md = os.path.join(os.path.dirname(__file__), "geo_prompts_md")
    md_path = os.path.join(base_md, f"{name}.md")
    txt = _read_text_file(md_path).strip()
    if txt:
        return txt

    # 兼容你原有 JSON 模板目录
    base_json = os.path.join(os.path.dirname(__file__), "geo_prompts")
    json_path = os.path.join(base_json, f"{name}.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("template") or "").strip()
    except Exception:
        return ""

# 允许的占位符（会被保留为 {NAME}，其它大括号全部转义成 {{ }}）
_ALLOWED_KEYS = {"USER_QUESTION", "BRAND_BRIEF", "MUST_EXPOSE", "EXPO_HINT", "STAGE1_JSON"}

_ALLOWED_KEYS = {"USER_QUESTION", "BRAND_BRIEF", "MUST_EXPOSE", "EXPO_HINT", "STAGE1_JSON"}

def _fmt_prompt(template: str, **vars) -> str:
    """
    安全格式化：先整体转义花括号，再反转义允许占位符。
    这样你的 MD 里即使包含 JSON 例子或大括号，也不会触发 KeyError。
    """
    if not isinstance(template, str):
        template = str(template or "")
    t = template.replace("{", "{{").replace("}", "}}")
    for key in _ALLOWED_KEYS:
        t = t.replace("{{" + key + "}}", "{" + key + "}")
    return t.format(**vars)

# 加载外置模板（内容无需在 JSON 里手工加双大括号）
PROMPT_STAGE1      = load_md_prompt_file("cot_stage1")
PROMPT_STAGE2      = load_md_prompt_file("cot_stage2")
PROMPT_SINGLE_PASS = load_md_prompt_file("cot_single")

# === （可选）模板占位符检查：放在加载之后！！ ===

for name, tmpl in {
    "cot_stage1": PROMPT_STAGE1,
    "cot_stage2": PROMPT_STAGE2,
    "cot_single": PROMPT_SINGLE_PASS
}.items():
    if not tmpl:
        logging.warning(f"[PROMPT CHECK] 模板 {name} 为空或未加载")
        continue
    # 先走一遍“安全放开”的处理，避免把 JSON 示例当占位符
    _safe = _fmt_prompt(tmpl,
                        USER_QUESTION="x",
                        BRAND_BRIEF="x",
                        MUST_EXPOSE="x",
                        EXPO_HINT="x",
                        STAGE1_JSON="{}")
    unknown = [m for m in re.findall(r"\{([A-Za-z0-9_]+)\}", _safe)
               if m not in _ALLOWED_KEYS]
    if unknown:
        logging.warning(f"[PROMPT CHECK] 模板 {name} 含未声明占位符: {unknown}")


# ====== COT Prompt 加载（优先读外部文件） ======
def _load_cot_template(fname: str, fallback: str) -> str:
    """统一加载逻辑，若外部不存在或 template 为空，则使用 fallback"""
    p = os.path.join(os.path.dirname(__file__), fname)
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tpl = (obj.get("template") or "").strip()
        return tpl if tpl else fallback
    except Exception:
        return fallback

# —— 通用逻辑策略版 Fallback —— #
_FALLBACK_STAGE1 = """你是 GEO （Generative Engine Optimization）内容顾问。
请生成一个包含 3–6 步的“逻辑链（logic_chain）”，用于描述一个从“定义问题”到“建立验证机制”的完整 GEO 内容策略过程。
仅输出 JSON：
{
  "logic_chain": ["步骤1","步骤2","步骤3", "步骤4", "步骤5"]
}
【输入】
- 目标问题：{USER_QUESTION}
- 甲方资料：{BRAND_BRIEF}
- 期望露出：{MUST_EXPOSE}
"""

_FALLBACK_STAGE2 = """你是 GEO（Generative Engine Optimization） 内容顾问。
请读取以下 Stage1 结果（logic_chain）并为每个节点生成 1–2 条“证据链（evidence_chain_v2）”和对应标题。
仅输出 JSON：
{
  "logic_chain": ["<复制自 Stage1>"],
  "evidence_chain_v2":[
    {"node":"<节点名称>",
     "proof":{"source_type":"official|product|media|user|research",
              "claim":"一句可验证主张",
              "how_to_verify":"验证方式",
              "asset":"对应产出或素材",
              "gaps":"待补项或数据缺口"}}
  ],
  "titles_by_node":[{"node":"…","titles":["…","…"]}]
}
【上阶段 JSON】
{STAGE1_JSON}
【期望露出】
{MUST_EXPOSE}
"""

_FALLBACK_SINGLE = """你是 GEO 内容顾问。
请基于输入，直接生成完整的“逻辑链 + 证据链（v2）+ 对应标题”结构。
仅输出 JSON：
{
  "logic_chain":["节点1","节点2","节点3"],
  "evidence_chain_v2":[
    {"node":"节点1","proof":{"source_type":"","claim":"","how_to_verify":"","asset":"","gaps":""}}
  ],
  "titles_by_node":[{"node":"节点1","titles":["",""]}]
}
【输入】
- 目标问题：{USER_QUESTION}
- 甲方资料：{BRAND_BRIEF}
- 期望露出：{MUST_EXPOSE}
"""

def get_cot_prompts(
    user_q: str,
    brand_brief: str,
    must_expose: str,
    expo_hint: str = "",
    mode: str = "two-stage",
    stage1_json: dict | None = None,
):
    """
    使用 Markdown Prompt（geo_prompts_md/*.md）构造提示词。
    - single: 返回 p_single（合并输出）
    - two-stage: 返回 p1（阶段1），p2（阶段2）
    若 MD 不存在，将优先回退到 geo_prompts/*.json 的 template 字段；再回退到 fallback。
    """
    user_q = (user_q or "").strip()
    brand_brief = (brand_brief or "").strip()
    must_expose = (must_expose or "").strip()
    expo_hint = (expo_hint or "").strip()

    # 1) 加载 MD（或 JSON / fallback）
    stage1_tpl = load_md_prompt_file("cot_stage1") or _FALLBACK_STAGE1
    stage2_tpl = load_md_prompt_file("cot_stage2") or _FALLBACK_STAGE2
    single_tpl = load_md_prompt_file("cot_single") or _FALLBACK_SINGLE

    if str(mode).lower().startswith("single"):
        p_single = _fmt_prompt(
            single_tpl,
            USER_QUESTION=user_q,
            BRAND_BRIEF=brand_brief,
            MUST_EXPOSE=must_expose,
            EXPO_HINT=expo_hint,
        )
        return None, None, p_single

    # 两段式
    p1 = _fmt_prompt(
        stage1_tpl,
        USER_QUESTION=user_q,
        BRAND_BRIEF=brand_brief,
        MUST_EXPOSE=must_expose,
        EXPO_HINT=expo_hint,
    )

    s1_json_text = json.dumps(stage1_json or {}, ensure_ascii=False, indent=2)
    p2 = _fmt_prompt(
        stage2_tpl,
        USER_QUESTION=user_q,
        BRAND_BRIEF=brand_brief,
        MUST_EXPOSE=must_expose,
        EXPO_HINT=expo_hint,
        STAGE1_JSON=s1_json_text,
    )
    return p1, p2, None

def export_md_file(text: str, filename: str = "geo_cot_output.md") -> str:
    try:
        tmpdir = tempfile.gettempdir()
        path = os.path.join(tmpdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
        return path
    except Exception:
        return ""



# ============ Prompt 模块化 ============
PROMPTS = {}
def load_prompts():
    path = os.path.join(os.path.dirname(__file__), "geo_prompts.json")
    global PROMPTS
    if not os.path.exists(path):
        PROMPTS["geo_max_zh"] = (
            "你是一名生成式引擎优化（GEO）专家。请融合以下9种策略，对下列文本进行综合优化："
            "1) 流畅优化,调整句法结构，使句子自然顺畅、逻辑递进；2) 词汇多样化,避免重复使用同一动词或形容词；3) 权威语气,内容应体现专业判断、基于事实；4) 引语；5) 引用标记；6) 简洁表达；"
            "7) 术语并解释；8) 数据化描述；9) 关键词增强。只输出优化正文。\n---\n原文：\n{TEXT}\n---"
        )
    else:
        with open(path, "r", encoding="utf-8") as f:
            PROMPTS.update(json.load(f))

def build_geo_prompt(original_text: str) -> str:
    tpl = PROMPTS.get("geo_max_zh", "请优化以下文本：{TEXT}")
    return tpl.replace("{TEXT}", original_text.strip())

# ====== 文本分块 ======
DEFAULT_MAX_CHARS = 2800
def _split_to_units(text: str):
    seps = "。！？!?．."
    units = []
    for para in text.split("\n"):
        para = para.strip()
        if not para: continue
        buf = ""
        for ch in para:
            buf += ch
            if ch in seps:
                units.append(buf.strip()); buf = ""
        if buf.strip(): units.append(buf.strip())
        units.append("\n")
    while units and units[-1] == "\n": units.pop()
    return units or [text]

def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS):
    units = _split_to_units(text)
    chunks, cur = [], ""
    for u in units:
        if len(u) > max_chars:
            if cur: chunks.append(cur); cur = ""
            for i in range(0, len(u), max_chars): chunks.append(u[i:i+max_chars])
            continue
        if len(cur) + len(u) <= max_chars: cur += u
        else:
            if cur: chunks.append(cur)
            cur = u
    if cur: chunks.append(cur)
    return chunks

# ============ 模型适配（你现有的三个） ============
def call_tongyi(prompt: str, timeout: int = 90) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key: return "⚠️ 未配置 DASHSCOPE_API_KEY 环境变量。"
    try:
        client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        completion = client.chat.completions.create(
            model="qwen3-max",
            messages=[{"role":"system","content":"You are a helpful assistant for text rewriting."},
                      {"role":"user","content":prompt}],
            temperature=0.7, timeout=timeout
        )
        
        choices = getattr(completion, "choices", None)
        first = _first_choice(choices)
        if first is None:
            return f"⚠️ 通义返回空结果：{getattr(completion, 'model', 'unknown_model')}"
        msg = getattr(first, "message", None)

        content = getattr(msg, "content", None)
        if not content:
            return "⚠️ 通义返回无内容（message.content 为空）。"
        return content.strip()
    except Exception as e:
        return f"❌ 通义千问请求失败：{e}"

def call_deepseek(prompt: str, timeout: int = 90, model: str = "deepseek-v3.2-exp") -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("TONGYI_API_KEY", "")
    if not api_key: return "⚠️ 未配置 DASHSCOPE_API_KEY（或 TONGYI_API_KEY）。"
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model, 
        "messages":[
            {"role":"system","content":"You are a helpful assistant for text rewriting."},
            {"role":"user","content":prompt}
            ],
            "stream": False,
            "temperature": 0.7,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if not r.ok: return f"❌ DeepSeekHTTP错误 {r.status_code}: {r.text}"
        
        data = r.json()
        choices = data.get("choices", None)
        first = _first_choice(choices)
        if first is None:
            return f"⚠️ DeepSeek 返回空结果：{data}"
        # 兼容对象/字典两种形态
        msg = getattr(first, "message", None)
        if msg is None and isinstance(first, dict):
            msg = first.get("message", {})
        if msg is None:
            return f"⚠️ DeepSeek 返回无可用 message：{data}"

        
        content = (msg.get("content") or "").strip()
        return content if content else "⚠️ DeepSeek 返回无内容（message.content 为空）。"
    except Exception as e:
        return f"❌ DeepSeek 请求失败：{e}"

def call_wenxin(prompt: str, timeout: int = 60) -> str:
    access_token = os.getenv("WENXIN_ACCESS_TOKEN", "")
    if not access_token: return "⚠️ 未配置 WENXIN_ACCESS_TOKEN。"
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={access_token}"
    headers = {"Content-Type": "application/json"}
    payload = {"messages":[{"role":"system","content":"You are a helpful assistant for text rewriting."},
                           {"role":"user","content":prompt}],
               "temperature":0.7}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if "error_code" in data and data["error_code"] != 0:
            return f"❌ 文心错误：{data.get('error_msg','unknown')}"
        return data.get("result","").strip()
    except Exception as e:
        return f"❌ 文心一言请求失败：{e}"

# ============ 主推理（生成优化稿） ============
def run_geo(text: str, model_name: str, use_chunk: bool = True,
            max_chars: int = DEFAULT_MAX_CHARS, progress=gr.Progress(track_tqdm=True)):
    if not PROMPTS: load_prompts()
    if not text.strip(): return "⚠️ 请输入需要优化的文本。", text
    chunks = [text] if not use_chunk else chunk_text(text, max(800, int(max_chars)))
    total = len(chunks); outputs = []
    for idx, chunk in enumerate(chunks, start=1):
        progress((idx-1)/total, desc=f"处理中 {idx}/{total} ...")
        prompt = build_geo_prompt(chunk)
        if model_name == "通义千问": out = call_tongyi(prompt)
        elif model_name == "DeepSeek": out = call_deepseek(prompt)
        elif model_name == "文心一言": out = call_wenxin(prompt)
        else: out = "⚠️ 未选择模型。"
        outputs.append(out if out else "")
    merged = ("\n\n--- [GEO-Chunk Split] ---\n\n").join(outputs).strip()
    return merged, text

# ============ GEO-Score 自动评分 ============
def run_score(original_text, optimized_text, model_name):
    if not optimized_text or optimized_text.startswith(("⚠️","❌")):
        return "⚠️ 无法评分，请先生成优化稿。", {}
    query = original_text[:80] if len(original_text) > 80 else original_text
    scoring_model = "qwen3-max"
    try:
        score = evaluate_geo_score(
            model_name=scoring_model, query=query,
            src_text=original_text, opt_text=optimized_text,
            mode="single_text", samples=2
        )
        try:
            log_run(model=scoring_model, query=query,
                    original_text=original_text, optimized_text=optimized_text,
                    score_dict=score, mode="single_text")
        except Exception:
            pass
        numeric_items = {k:v for k,v in score.items() if isinstance(v,(int,float))}
        lines = [f"**{k}**：{v:.2f}" for k,v in numeric_items.items()]
        return f"✅ GEO-Score：{score.get('geo_score',0):.1f} / 100\n\n" + " | ".join(lines), score
    except Exception as e:
        return f"❌ 评分失败：{e}", {}

# ============ 报告导出（带评分） ============
def export_html_with_score(original_text, optimized_text, score, project_name="", client_name=""):
    if not optimized_text or optimized_text.startswith(("⚠️","❌")):
        return gr.update(value=None, visible=False), "⚠️ 当前没有可导出的优化结果。"
    geo_id = str(uuid.uuid4())[:8]
    base_title = "GEO-Max Report"
    pieces = [p for p in [project_name, client_name, base_title] if p]
    title = " · ".join(pieces) + f" · ID:{geo_id}"
    html_str = render_report_html(title, original_text, original_text, optimized_text, score or {})
    html_str += f"\n<!-- GEO-ID:{geo_id} -->\n"
    safe_proj = (project_name or "").strip().replace(" ","_")
    safe_clt  = (client_name or "").strip().replace(" ","_")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname_bits = ["geo_report", ts]
    if safe_proj: fname_bits.append(safe_proj)
    if safe_clt:  fname_bits.append(safe_clt)
    fname_bits.append(geo_id)
    fname = "_".join(fname_bits) + ".html"
    path = os.path.abspath(os.path.join(SAVE_DIR, fname))
    with open(path,"w",encoding="utf-8") as f: f.write(html_str)
    return gr.update(value=path, visible=True), f"✅ 已导出带评分报告：{path}"

# ======================= GEO-CoT 增量功能（新增） =======================
# —— 不改动你现有函数，仅新增一组 geo_cot_* 方法与一个 Tab ——

# 统一路由到你现有的三家模型
def geo_cot_model_call(prompt: str, provider: str) -> str:
    if provider == "DeepSeek": return call_deepseek(prompt)
    if provider == "通义千问":   return call_tongyi(prompt)
    if provider == "文心一言":   return call_wenxin(prompt)
    return "⚠️ 未选择模型。"

COT_TRIGGER = "Let's think step by step."

GEO_COT_TASK_REQUIRE = """你是GEO-Max的内容策略与推理专家。请严格按下列结构输出JSON：
{
  "logic_chain": ["...节点1","...节点2","..."],
  "evidence_chain": [
     {"node":"节点1","evidence":{"data":"", "industry":"", "media":"", "extra":""},"gaps":""}
  ],
  "titles_by_node": [
     {"node":"节点1","titles":["",""]}
  ]
}
要求：
- 建议3–6个逻辑节点（可在3–8内浮动）；逐节点对齐证据；
- 不得虚构具体数据；若证据不足请在 gaps 中标注采集建议；
- 只输出JSON，勿加多余说明。
"""

def geo_cot_assemble_prompt(q: str, brand_ctx: str, exposure_goals: List[str]) -> str:
    return textwrap.dedent(f"""
    [触发语]
    {COT_TRIGGER}

    {GEO_COT_TASK_REQUIRE}

    [目标问题]
    {q}

    [甲方资料]
    {brand_ctx[:1200]}

    [期望露出]
    { "、".join([g for g in exposure_goals if g]) }
    """).strip()


def _find_balanced_json_blocks(text: str, max_blocks: int = 6) -> list[str]:
    if not text:
        return []
    blocks, stack = [], []
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            stack.append(i)
            if start is None:
                start = i
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    blocks.append(text[start:i+1])
                    start = None
        if len(blocks) >= max_blocks:
            break
    return blocks

def geo_cot_extract_json(text: str):
    """从文本中找出候选 {..}，逐个 json.loads；优先返回字段齐全的那块"""
    cand_blocks = _find_balanced_json_blocks(text or "")
    best, best_score = None, -1
    for blk in cand_blocks:
        try:
            obj = json.loads(blk)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        score = 0
        for k in ("logic_chain", "evidence_chain", "titles_by_node"):
            if k in obj:
                score += 1
        if score > best_score:
            best, best_score = obj, score
            if score == 3:
                break
    return best

# --------- 轻量重试：空/报错时再试一次 ---------
def _retry_call(fn, times=2):
    last = None
    for _ in range(max(1, times)):
        try:
            out = fn()
            last = out
            # 成功判定：有内容且非告警/报错提示
            if out and not str(out).startswith(("⚠️", "❌")):
                return out
        except Exception as e:
            # 记录并继续重试
            print("[MODEL CALL RETRY]", repr(e))
            last = f"❌ 调用异常：{e}"
            continue
    return last



# 一段式：一次提示，直接产出 JSON（新版兜底对齐《通用逻辑策略》）
def geo_cot_run_once(q, brand_ctx, exposure_text, provider, progress: gr.Progress = None):
    """
    一段式：一次调用直接产出“逻辑链 + 证据链 + 对应标题”
    返回 (data: dict, None)，并保证结构兜底与 raw_text（新兜底字段）
    """
    goals = [x.strip() for x in (exposure_text or "").split(",") if x.strip()]
    goals_text = ", ".join(goals)

    safe_progress(progress, 0.25, desc="组装提示词")
    _, _, p_single = get_cot_prompts(
        user_q=q,
        brand_brief=brand_ctx or "",
        must_expose=goals_text,
        expo_hint="",
        mode="single"
    )

    safe_progress(progress, 0.50, desc="请求模型")
    out = _retry_call(lambda: geo_cot_model_call(p_single, provider), times=2)

    print("==== RAW OUTPUT (SINGLE) ====")
    try:
        print(out if len(str(out)) < 4000 else str(out)[:4000] + " …[trunc]")
    except Exception:
        print("<non-printable>")
    print("=============================")

    data = geo_cot_extract_json(out or "")
    if not isinstance(data, dict):
        data = {}

    # === 兜底结构：逻辑链 ===
    if not isinstance(data.get("logic_chain"), list) or not data.get("logic_chain"):
        data["logic_chain"] = [
            "定义品类标准与用户典型诉求",
            "将品牌属性与品类标准逐项绑定",
            "构建统一推荐理由语法（功能/情绪/榜单）",
            "提供结构化内容与 Schema 以提升可引用性",
            "建立监测与迭代机制（GEO-Score 回路）"
        ]

    # === 兜底结构：证据链（按你的“证据链数据清单”字段）===
    if not isinstance(data.get("evidence_chain"), list) or not data.get("evidence_chain"):
        node0 = data["logic_chain"][0]
        data["evidence_chain"] = [
            {
                "node": node0,
                "evidence": {
                    "official": "品牌定义/定位/愿景（About/Schema 同步）",
                    "category_tags": "行业类别/细分领域的统一标签与命名",
                    "products": "代表产品/系列（SKU 元数据与核心描述）",
                    "scenarios": "目标用户/使用场景（FAQ/图文说明）",
                    "media_refs": "第三方媒体/榜单/证书（用于可信引用）",
                    "tech_specs": "参数/材料/标准（白皮书/测评摘要）",
                    "third_party": "百科/问答/研究文章等外部引用",
                    "structure": "统一命名字段：Name/Category/Keywords/Tagline/USPs/HeroProducts/Awards/OfficialLinks/MediaMentions/AudienceFit"
                },
                "gaps": [
                    "补充 1–2 个可量化指标（如近90天被引用率、问答采纳率）",
                    "为代表产品添加 JSON-LD（Product/FAQ/HowTo）"
                ]
            }
        ]

    # === 兜底结构：标题与触发（保持每节点 2–3 条）===
    if not isinstance(data.get("titles_by_node"), list) or not data.get("titles_by_node"):
        data["titles_by_node"] = [
            {
                "node": data["logic_chain"][0],
                "titles": [
                    "什么是合格的 X 类品牌？",
                    "从品类标准到统一语法：AI 为何推荐你"
                ]
            }
        ]

    data["raw_text"] = (out or "")[:2000]
    safe_progress(progress, 0.85, desc="完成")
    return data, None



# 两段式：先长推理，再抽取 JSON（更稳）
# GEO_COT_EXTRACT_ONLY = load_md_prompt_file("cot_extract")

def geo_cot_run_two_stage(q, brand_ctx, exposure_text, provider, progress: gr.Progress = None):
    """
    两段式（推荐）：Stage1 产出“策略性逻辑链+策略上下文”，Stage2 完成“证据链+严格对应标题”。
    仍返回 (data: dict, None)，其中 data 至少包含 logic_chain / evidence_chain / titles_by_node / raw_text
    """
    # 期望露出 -> goals 列表与展示文本
    goals = [x.strip() for x in (exposure_text or "").split(",") if x.strip()]
    goals_text = ", ".join(goals)

    # ===== Stage 1：策略性逻辑链（PLAN） =====
    safe_progress(progress, 0.20, desc="阶段1：策略规划（PLAN）")
    p1, _, _ = get_cot_prompts(
        user_q=q,
        brand_brief=brand_ctx or "",
        must_expose=goals_text,
        expo_hint="",              # 需要的话你可以从别处传入
        mode="two-stage"
    )

    out1 = _retry_call(lambda: geo_cot_model_call(p1, provider), times=2)
    # 记录原始文本片段到 raw_text（便于调试）
    raw1 = (out1 or "")[:2000]

    s1 = geo_cot_extract_json(out1 or "") or {}
    if not isinstance(s1, dict):
        s1 = {}
    # 兜底：若模型未返回 logic_chain，也给个最小可用策略链避免后续报错
    if not isinstance(s1.get("logic_chain"), list) or not s1.get("logic_chain"):
        s1["logic_chain"] = [
            "明确北极星与受众分层",
            "构建内容杠杆与资产形态",
            "规划分发与结构化要素",
            "建立引用与复述的提示语规范",
            "搭建测量与滚动补证闭环"
        ]

    # ===== Stage 2：证据链 + 对应标题（FILL） =====
    safe_progress(progress, 0.55, desc="阶段2：证据链生成（FILL）")
    # 将 Stage1 的结构回灌到 Stage2 prompt
    _, p2, _ = get_cot_prompts(
        user_q=q,
        brand_brief=brand_ctx or "",
        must_expose=goals_text,
        expo_hint="",
        mode="two-stage",
        stage1_json=s1
    )

    out2 = _retry_call(lambda: geo_cot_model_call(p2, provider), times=2)
    raw2 = (out2 or "")[:2000]

    data = geo_cot_extract_json(out2 or "")

    # ===== 兜底与清洗 =====
    if not isinstance(data, dict):
        data = {}

    # 把 Stage1 的 logic_chain 作为最终链条来源（若 Stage2 没有）
    if not isinstance(data.get("logic_chain"), list) or not data.get("logic_chain"):
        data["logic_chain"] = s1.get("logic_chain", [])

    if not isinstance(data.get("evidence_chain"), list):
        data["evidence_chain"] = []
    if not isinstance(data.get("titles_by_node"), list):
        data["titles_by_node"] = []

    # 若依旧为空，写入最小可用结构，防止前端越界
    if not data["evidence_chain"]:
        data["evidence_chain"] = [{
            "node": data["logic_chain"][0] if data["logic_chain"] else "明确北极星与受众分层",
            "evidence": {
                "data": "示例：平台A近90天被引用率12.4%",
                "industry": "轻复原更适合 how-to + compare 的组合问法",
                "media": "百科/社区/社交作为辅助证据来源",
                "extra": "JSON-LD: FAQ + HowTo；首段统一口径并显式 citation"
            },
            "gaps": "采集平台近90天引用率与问答采纳率；补充示例问法"
        }]

    if not data["titles_by_node"]:
        node0 = data["evidence_chain"][0].get("node", "明确北极星与受众分层")
        data["titles_by_node"] = [{
            "node": node0,
            "titles": ["轻复原怎么选：从目标到分发", "从被看到到被引用：GEO 执行链"]
        }]

    # 调试用原文片段
    data["raw_text"] = (raw1 + "\n\n" + raw2)[:2000]

    safe_progress(progress, 0.90, desc="整理输出")
    return data, None


def geo_cot_score(data: Dict[str, Any], exposure_text: str) -> float:
    try:
        goals = [x.strip() for x in (exposure_text or "").split(",") if x.strip()]
        lc = data.get("logic_chain", [])
        tb = data.get("titles_by_node", [])

        # 结构
        s_struct = 0
        n = len(lc)
        if 3 <= n <= 8:
            s_struct = 10
            if 4 <= n <= 6: s_struct += 10

        # 露出
        blob = json.dumps(data, ensure_ascii=False)
        hit = sum(1 for k in goals if k and (k in blob))
        s_expo = 20.0 * (hit / max(1, len(goals))) if goals else 20.0

        # 标题
        ok_nodes = 0
        for item in tb:
            cnt = len(item.get("titles", []))
            if 2 <= cnt <= 3: ok_nodes += 1
        s_title = 20.0 * (ok_nodes / max(1, len(lc))) if lc else 0.0

        # 其它占位
        s_logic = 10.0
        s_exec  = 10.0
        s_align = 15.0

        total = (
            s_struct * 0.2 + s_align * 0.2 + s_expo * 0.2 +
            s_title * 0.2 + s_logic * 0.1 + s_exec * 0.1
        ) * 5
        return round(total, 1)
    except Exception:
        return 0.0


SAVE_DIR = os.path.join(os.path.expanduser("~"), "GEO-Reports")
os.makedirs(SAVE_DIR, exist_ok=True)

with gr.Blocks(title="GEO-Max 多模型文本优化引擎（含评分）",
               analytics_enabled=False, theme=APP_THEME, css=APP_CSS) as demo:

    with gr.Group(elem_id="wrap"):
        gr.Markdown("### GEO-Max · 生成式引擎优化\n极简、稳定：内容改写 + 自动评分。")

        with gr.Tabs(elem_classes=["tabs"]):
            # ---- Tab 1 ----
            with gr.Tab("⚙️ 产品模式（质量评分）"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["stack"]):
                        with gr.Group(elem_classes=["tile"]):
                            inp_text = gr.Textbox(label="✍️ 输入原文", lines=8, show_copy_button=True)
                            model_dd = gr.Dropdown(choices=["通义千问","DeepSeek","文心一言"],
                                                   value="通义千问", label="🧩 选择模型")
                            use_chunk = gr.Checkbox(value=True, label="自动分块（建议开启）")
                            max_chars = gr.Slider(800, 6000, value=2800, step=100, label="每块最大字数")
                            btn_run = gr.Button("🚀 生成 GEO-Max 优化稿", variant="primary")
                            btn_clear = gr.Button("🧹 清空")
                            gr.Markdown("<div class='footnote'>提示：我们不保存你的文本；评分仅在本地会话内计算。</div>")

                    with gr.Column(scale=1, elem_classes=["stack"]):
                        with gr.Group(elem_classes=["tile"]):
                            out_text = gr.Textbox(label="📈 GEO-Max 优化结果", lines=12, show_copy_button=True)
                            btn_score = gr.Button("📊 计算 GEO-Score（自动评分）")
                            score_md = gr.Markdown("")
                            with gr.Row():
                                btn_html = gr.Button("导出带评分报告（HTML）")
                                file_html = gr.File(label="下载报告", visible=False)
                            tip = gr.Markdown("")

                # 状态与事件
                state_original = gr.State(""); state_optimized = gr.State(""); state_score = gr.State({})
                btn_run.click(fn=run_geo, inputs=[inp_text, model_dd, use_chunk, max_chars],
                              outputs=[out_text, state_original], queue=False)
                out_text.change(lambda x:x, inputs=out_text, outputs=state_optimized, queue=False)
                btn_score.click(fn=run_score, inputs=[state_original, state_optimized, model_dd],
                                outputs=[score_md, state_score], queue=False)
                btn_html.click(fn=export_html_with_score,
                               inputs=[state_original, state_optimized, state_score],
                               outputs=[file_html, tip], queue=False)
                btn_clear.click(lambda: ("","","","",None),
                                None, [inp_text, out_text, score_md, tip, file_html], queue=False)

            # ---- Tab 2 ----
            with gr.Tab("📘 论文模式（with citations）"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["stack"]):
                        with gr.Group(elem_classes=["tile"]):
                            n_sources = gr.Number(value=3, label="来源总数（N）", precision=0)
                            mode_sel = gr.Dropdown(choices=["WordPos","Word","Pos"], value="WordPos", label="指标模式")
                            answer_once = gr.Textbox(label="单次分布：带 [1][2]… 的答案（任一段）", lines=6, show_copy_button=True)
                            btn_once = gr.Button("📊 计算单次分布", variant="secondary")
                            msg_once = gr.Markdown("")
                            dist_once = gr.JSON(label="分布（和=1）")
                    with gr.Column(scale=1, elem_classes=["stack"]):
                        with gr.Group(elem_classes=["tile"]):
                            before_ans = gr.Textbox(label="Before：带引用的答案", lines=6, show_copy_button=True)
                            after_ans  = gr.Textbox(label="After：带引用的答案", lines=6, show_copy_button=True)
                            target_idx = gr.Number(value=1, label="目标来源索引（1..N）", precision=0)
                            btn_delta = gr.Button("📈 计算 Δ 提升（After - Before）", variant="primary")
                            msg_delta = gr.Markdown("")
                            res_delta = gr.JSON(label="结果（含 dist_before / dist_after / delta）")

                btn_once.click(fn=run_impression_single,
                               inputs=[answer_once, n_sources, mode_sel],
                               outputs=[msg_once, dist_once], queue=False)
                btn_delta.click(fn=run_impression_delta,
                                inputs=[before_ans, after_ans, n_sources, target_idx, mode_sel],
                                outputs=[msg_delta, res_delta], queue=False)

            # ---- Tab 3（重写：两段式 · 纯 Markdown 模板工作流）----
            with gr.Tab("🧠 GEO-CoT（两段式·Markdown 模板）"):
                # ========= 仅供本 Tab 使用的轻量工具函数 =========
                def _load_md_template(name: str) -> str:
                    """
                    从 ./geo_prompts/ 目录加载 <name>.md
                    不做任何选择性读取或字段限制；原样返回模板文本。
                    """
                    base = os.path.join(os.path.dirname(__file__), "geo_prompts")
                    path = os.path.join(base, f"{name}.md")
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            return f.read()
                    except Exception as e:
                        return f"⚠️ 无法读取模板：{path}\n\n错误：{e}"

                # 仅允许的占位符（其它花括号全部转义，避免 .format 误伤）
                _ALLOWED_MD_KEYS = {"USER_QUESTION", "BRAND_BRIEF", "MUST_EXPOSE", "EXPO_HINT", "STAGE1_MD"}

                def _fmt_md_template(tpl: str, **vars) -> str:
                    """
                    安全格式化 Markdown 模板：
                    - 先把所有 { 和 } 转义成 {{ }} / }}}
                    - 再把“允许占位符”反转义为单大括号
                    - 最后 .format
                    """
                    if not isinstance(tpl, str):
                        tpl = str(tpl or "")
                    # 全量转义
                    t = tpl.replace("{", "{{").replace("}", "}}")
                    # 允许占位符反转义
                    for key in _ALLOWED_MD_KEYS:
                        t = t.replace("{{" + key + "}}", "{" + key + "}")
                    # 渲染
                    return t.format(**vars)

                def _save_md_to_file(md_text: str, filename: str = "geo_output.md"):
                    try:
                        tmpdir = tempfile.gettempdir()
                        path = os.path.join(tmpdir, filename)
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(md_text or "")
                        return path
                    except Exception:
                        return None

                # ================== Stage 1：执行 cot_stage1.md，输出 Markdown 可编辑 ==================
                def run_stage1_markdown(q: str, brand_ctx: str, expo: str, provider: str, progress=gr.Progress()):
                    """
                    - 读取 geo_prompts/cot_stage1.md
                    - 用 {USER_QUESTION}/{BRAND_BRIEF}/{MUST_EXPOSE}/{EXPO_HINT} 渲染
                    - 模型生成 Markdown，直接返回到“可编辑大文本框”
                    """
                    safe_progress(progress, 0.10, desc="加载 Stage1 模板（MD）")
                    tpl = _load_md_template("cot_stage1")
                    if tpl.startswith("⚠️ 无法读取模板"):
                        return tpl, "", None, "⚠️ 模板未找到，已在编辑框输出错误说明。"

                    safe_progress(progress, 0.25, desc="渲染 Stage1 提示词（MD）")
                    prompt = _fmt_md_template(
                        tpl,
                        USER_QUESTION=(q or "").strip(),
                        BRAND_BRIEF=(brand_ctx or "").strip(),
                        MUST_EXPOSE=(expo or "").strip(),
                        EXPO_HINT=""  # 预留占位，必要时可在 UI 加一个输入
                    )

                    safe_progress(progress, 0.55, desc="请求模型（Stage1）")
                    out_md = _retry_call(lambda: geo_cot_model_call(prompt, provider), times=2) or ""
                    if not out_md.strip():
                        out_md = "⚠️ 模型未返回内容，请重试或检查模板。"

                    # 提供一个便捷下载按钮（可选）
                    dl_path = _save_md_to_file(out_md, filename="geo_stage1_output.md")
                    safe_progress(progress, 0.90, desc="完成")
                    return out_md, prompt[:1200], dl_path, "✅ Stage1 完成：请在左侧编辑后，点击进入 Stage2 生成证据链。"

                # ================== Stage 2：读取“已编辑的 Stage1 MD”，执行 cot_stage2.md ==================
                def run_stage2_markdown(q: str, brand_ctx: str, expo: str, provider: str,
                                        stage1_md: str, progress=gr.Progress()):
                    """
                    - 读取 geo_prompts/cot_stage2.md
                    - 用 {USER_QUESTION}/{BRAND_BRIEF}/{MUST_EXPOSE}/{EXPO_HINT}/{STAGE1_MD} 渲染（STAGE1_MD=用户编辑后的完整文本）
                    - 模型生成 Markdown → 展示 + 支持下载
                    """
                    safe_progress(progress, 0.10, desc="加载 Stage2 模板（MD）")
                    tpl = _load_md_template("cot_stage2")
                    if tpl.startswith("⚠️ 无法读取模板"):
                        return "> 无法读取 Stage2 模板。", "", None, "⚠️ 模板未找到。"

                    # 直接把整段 Stage1 MD 注入 {STAGE1_MD}（不做任何限制/选择性读取）
                    safe_progress(progress, 0.28, desc="渲染 Stage2 提示词（MD）")
                    prompt = _fmt_md_template(
                        tpl,
                        USER_QUESTION=(q or "").strip(),
                        BRAND_BRIEF=(brand_ctx or "").strip(),
                        MUST_EXPOSE=(expo or "").strip(),
                        EXPO_HINT="",
                        STAGE1_MD=stage1_md or ""
                    )

                    safe_progress(progress, 0.60, desc="请求模型（Stage2）")
                    out_md = _retry_call(lambda: geo_cot_model_call(prompt, provider), times=2) or ""
                    if not out_md.strip():
                        out_md = "> ⚠️ Stage2 未产出内容，请检查 Stage1 文档或模板语法。"

                    # 导出 MD
                    dl_path = _save_md_to_file(out_md, filename="geo_stage2_output.md")
                    safe_progress(progress, 0.92, desc="完成")
                    return out_md, prompt[:1200], dl_path, "✅ Stage2 完成：右侧可复制/下载最终 Markdown。"

                # ================== UI：两列布局（左：输入与 Stage1；右：Stage2） ==================
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["stack"]):
                        with gr.Group(elem_classes=["tile"]):
                            md_q     = gr.Textbox(label="🎯 目标问题", placeholder="例如：推荐几家××品牌", lines=2)
                            md_brand = gr.Textbox(label="🏷️ 甲方资料（文字）", lines=6)
                            md_expo  = gr.Textbox(label="🔗 期望露出（逗号分隔）", placeholder="品牌名, 官网链接, 指定词组", lines=2)
                            md_model = gr.Dropdown(choices=["DeepSeek","通义千问","文心一言"],
                                                value="DeepSeek", label="🧩 模型")

                        with gr.Group(elem_classes=["tile"]):
                            gr.Markdown("#### Stage 1：执行 `cot_stage1.md` → 生成 Markdown（可编辑）")
                            btn_s1 = gr.Button("🚀 运行 Stage 1（Markdown）", variant="primary")
                            s1_md_editable = gr.Textbox(label="📝 Stage1 产出（可编辑 Markdown）",
                                                        lines=18, show_copy_button=True)
                            s1_prompt_dbg  = gr.Textbox(label="调试：Stage1 最终提示词片段（只读）",
                                                        lines=5, interactive=False)
                            s1_download    = gr.DownloadButton(label="下载 Stage1 .md", value=None)
                            s1_tip         = gr.Markdown("")

                            btn_confirm_s2 = gr.Button("✅ 使用上方 Markdown 进入 Stage 2", variant="secondary")

                    with gr.Column(scale=1, elem_classes=["stack"]):
                        with gr.Group(elem_classes=["tile"]):
                            gr.Markdown("#### Stage 2：执行 `cot_stage2.md`（注入你编辑后的 Stage1 文档）")
                            s2_md_view   = gr.Markdown(value="> 运行 Stage 2 后，这里显示最终 Markdown")
                            s2_prompt_dbg= gr.Textbox(label="调试：Stage2 最终提示词片段（只读）",
                                                    lines=5, interactive=False)
                            s2_download  = gr.DownloadButton(label="下载 Stage2 .md", value=None)
                            s2_tip       = gr.Markdown("")

                # ================== 事件绑定 ==================
                btn_s1.click(
                    run_stage1_markdown,
                    inputs=[md_q, md_brand, md_expo, md_model],
                    outputs=[s1_md_editable, s1_prompt_dbg, s1_download, s1_tip],
                    show_progress=True
                )

                btn_confirm_s2.click(
                    run_stage2_markdown,
                    inputs=[md_q, md_brand, md_expo, md_model, s1_md_editable],
                    outputs=[s2_md_view, s2_prompt_dbg, s2_download, s2_tip],
                    show_progress=True
                )





if __name__ == "__main__":
    try:
        demo.launch(server_name="127.0.0.1", server_port=7862, share=False, show_api=False,
                    allowed_paths=[SAVE_DIR])
    except Exception as e:
        print("⚠️ 本机直连失败，自动启用分享链接。原因：", e)
        demo.launch(server_name="127.0.0.1", server_port=7862, share=True, show_api=False,
                    allowed_paths=[SAVE_DIR])
