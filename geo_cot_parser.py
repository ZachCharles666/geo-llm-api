# geo_cot_parser.py
"""
Stage1 Markdown -> 结构化 Logic JSON 解析器（带 UUID & 兜底逻辑）

- 从 Stage1 Markdown 中提取：
  - MLC 节点（Human Logic Chain）
  - LLC 节点（Machine Logic Chain）
  - 收敛规则（Convergence）
  - Stage2 蓝图原文（Blueprint raw block）

- 为每个节点生成：
  - id: "MLC-01" / "LLC-01" 等
  - uuid: 全局唯一 ID（依赖 geo_ids.make_node_uuid）
  - run_id: 本次调用 run 的 ID（依赖 geo_ids.make_run_id）
"""
import re
from typing import Dict, Any, List

from geo_ids import make_run_id, make_node_uuid


# 兼容类似：
#   "## 行业共识（MLC-01）"
#   "### MLC-01（行业共识）"
MLC_HEADER_RE = re.compile(r"^#{1,6}[^\n]*MLC-(\d+)[^\n]*$", re.MULTILINE)
LLC_HEADER_RE = re.compile(r"^#{1,6}[^\n]*LLC-(\d+)[^\n]*$", re.MULTILINE)


def _extract_block(md: str, start_pos: int, header_positions: List[int]) -> str:
    """
    从某个标题位置开始，截取到“下一个标题”之前的内容。
    header_positions: 所有 MLC/LLC 标题在文中的起始位置（已排序）。
    """
    tail_candidates = [p for p in header_positions if p > start_pos]
    end_pos = tail_candidates[0] if tail_candidates else len(md)
    return md[start_pos:end_pos].strip()


def _parse_mlc_block(block: str) -> Dict[str, Any]:
    """
    解析一个 MLC markdown block，支持两种格式：

    1）严格格式（推荐）：
        ## 行业共识（MLC-01）
        - claim：xxx
        - rationale：yyy
        - evidence_need：zzz

    2）兜底格式（当前模型常见）：
        ## 行业共识（MLC-01）
        这里是一整段自然语言描述，没有 claim / rationale 标签

    兜底策略：
    - 若不存在显式 claim / rationale：
      - 将正文按句号/问号/感叹号/换行切分；
      - 第一句视为 claim，剩余拼成 rationale；
      - 若只有一句，则同时视为 claim & rationale。
    """
    lines = block.splitlines()

    # 1) 提取标题行与标题中的“名称”
    header_line = ""
    for ln in lines:
        if ln.strip().startswith("#"):
            header_line = ln.strip()
            break

    title = ""
    m_title = re.search(r"MLC-\d+（(.+?)）", header_line) or re.search(
        r"^#+\s*(.+?)（MLC-\d+）", header_line
    )
    if m_title:
        title = m_title.group(1).strip()

    # 2) 去掉标题之后的正文行
    body_lines: List[str] = []
    header_seen = False
    for ln in lines:
        if not header_seen:
            if ln is header_line:
                header_seen = True
            continue
        body_lines.append(ln)

    # 3) 先尝试解析“严格格式”的 bullet 写法
    claim = ""
    rationale = ""
    evidence_need: List[str] = []

    def _strip_label(s: str, label: str) -> str:
        return s.split(label, 1)[1].strip() if label in s else ""

    for raw in body_lines:
        s = raw.strip()
        if not s:
            continue
        # 兼容 "- claim：" / "• claim："等前缀
        s = s.lstrip("-•").strip()

        if s.startswith("claim：") and not claim:
            claim = _strip_label(s, "claim：")
        elif s.startswith("rationale：") and not rationale:
            rationale = _strip_label(s, "rationale：")
        elif s.startswith("evidence_need："):
            ev = _strip_label(s, "evidence_need：")
            if ev:
                evidence_need.append(ev)

    # 4) 若严格格式没解析出内容，则启动“兜底模式”
    plain_body = "\n".join(body_lines).strip()
    if (not claim and not rationale) and plain_body:
        # 用中文/英文标点粗暴切句
        parts = re.split(r"[。！？!?；;。\n]+", plain_body)
        parts = [p.strip() for p in parts if p.strip()]

        if parts:
            claim = parts[0]
            if len(parts) >= 2:
                rationale = "；".join(parts[1:])
            else:
                # 只有一句，就让 claim + rationale 共用
                rationale = parts[0]

    # 兜底：仍然为空时，至少把全文塞进 rationale
    if not claim and plain_body:
        claim = plain_body
    if not rationale and plain_body:
        rationale = plain_body

    return {
        "title": title,
        "claim": claim,
        "rationale": rationale,
        "evidence_need": evidence_need,
        "raw_block": block.strip(),
    }


def _parse_llc_block(block: str) -> Dict[str, Any]:
    """
    解析一个 LLC markdown block，目标提取：
    - title（如“判定变量定义”）
    - key / signal / mapping / landing
    """
    lines = block.splitlines()

    # 1) 提取标题行与标题中的“名称”
    header_line = ""
    for ln in lines:
        if ln.strip().startswith("#"):
            header_line = ln.strip()
            break

    title = ""
    m_title = re.search(r"LLC-\d+（(.+?)）", header_line) or re.search(
        r"^#+\s*(.+?)（LLC-\d+）", header_line
    )
    if m_title:
        title = m_title.group(1).strip()

    # 2) 标题之后的正文行
    body_lines: List[str] = []
    header_seen = False
    for ln in lines:
        if not header_seen:
            if ln is header_line:
                header_seen = True
            continue
        body_lines.append(ln)

    key = ""
    signal = ""
    mapping = ""
    landing = ""

    def _strip_label(s: str, label: str) -> str:
        return s.split(label, 1)[1].strip() if label in s else ""

    for raw in body_lines:
        s = raw.strip()
        if not s:
            continue
        s = s.lstrip("-•").strip()

        if s.startswith("key：") and not key:
            key = _strip_label(s, "key：")
        elif s.startswith("signal：") and not signal:
            signal = _strip_label(s, "signal：")
        elif s.startswith("mapping：") and not mapping:
            mapping = _strip_label(s, "mapping：")
        elif s.startswith("landing：") and not landing:
            landing = _strip_label(s, "landing：")

    # 兜底：若全部为空，则将正文整体作为 mapping
    plain_body = "\n".join(body_lines).strip()
    if not (key or signal or mapping or landing) and plain_body:
        mapping = plain_body

    return {
        "title": title,
        "key": key,
        "signal": signal,
        "mapping": mapping,
        "landing": landing,
        "raw_block": block.strip(),
    }


def stage1_md_to_json(
    md_text: str,
    user_question: str = "",
    brand_brief: str = "",
    must_expose: str = "",
    expo_hint: str = "",
) -> Dict[str, Any]:
    """
    将 Stage1 Markdown 文本解析为统一 JSON 结构：

    {
      "version": "geo-cot-v1",
      "run_id": "...",
      "meta": { ... },
      "mlc_nodes": [ ... ],
      "llc_nodes": [ ... ],
      "convergence": { "raw_block": "...", "uuid": "...", "run_id": "..." },
      "stage2_blueprint": { "raw_block": "...", "uuid": "...", "run_id": "..." }
    }
    """
    md = md_text or ""
    run_id = make_run_id()

    result: Dict[str, Any] = {
        "version": "geo-cot-v1",
        "run_id": run_id,
        "meta": {
            "run_id": run_id,
            "user_question": user_question or "",
            "brand_brief": brand_brief or "",
            "must_expose": must_expose or "",
            "expo_hint": expo_hint or "",
        },
        "mlc_nodes": [],
        "llc_nodes": [],
        "convergence": {},
        "stage2_blueprint": {},
    }

    # === 1) 收集所有 MLC / LLC 标题位置，用于截取 block ===
    mlc_matches = list(MLC_HEADER_RE.finditer(md))
    llc_matches = list(LLC_HEADER_RE.finditer(md))

    # 所有 header 的起始位置（用于 _extract_block）
    header_positions: List[int] = [m.start() for m in mlc_matches + llc_matches]
    header_positions.sort()

    # === 2) 解析 MLC 节点 ===
    mlc_nodes: List[Dict[str, Any]] = []
    for m in mlc_matches:
        num = m.group(1)
        node_id = f"MLC-{num}"
        block = _extract_block(md, m.start(), header_positions)
        parsed = _parse_mlc_block(block)

        node: Dict[str, Any] = {
            "id": node_id,
            "title": parsed["title"],
            "claim": parsed["claim"],
            "rationale": parsed["rationale"],
            "evidence_need": parsed["evidence_need"],
            "raw_block": parsed["raw_block"],
        }
        node["uuid"] = make_node_uuid("MLC", run_id, node_id)
        node["run_id"] = run_id
        mlc_nodes.append(node)

    # === 3) 解析 LLC 节点 ===
    llc_nodes: List[Dict[str, Any]] = []
    for m in llc_matches:
        num = m.group(1)
        node_id = f"LLC-{num}"
        block = _extract_block(md, m.start(), header_positions)
        parsed = _parse_llc_block(block)

        node = {
            "id": node_id,
            "title": parsed["title"],
            "key": parsed["key"],
            "signal": parsed["signal"],
            "mapping": parsed["mapping"],
            "landing": parsed["landing"],
            "raw_block": parsed["raw_block"],
        }
        node["uuid"] = make_node_uuid("LLC", run_id, node_id)
        node["run_id"] = run_id
        llc_nodes.append(node)

    result["mlc_nodes"] = mlc_nodes
    result["llc_nodes"] = llc_nodes

    # === 4) 收敛规则（Convergence）与 Stage2 蓝图 ===
    # 假设：
    #   "## III." 段为收敛规则
    #   "## IV." 段为 Stage2 Blueprint
    conv_block = ""
    stage2_block = ""

    if "## III." in md:
        conv_start = md.find("## III.")
        next_iv = md.find("## IV.", conv_start + 1)
        conv_block = md[conv_start : next_iv if next_iv != -1 else len(md)].strip()

    if "## IV." in md:
        iv_start = md.find("## IV.")
        stage2_block = md[iv_start:].strip()

    if conv_block:
        result["convergence"] = {
            "raw_block": conv_block,
            "uuid": make_node_uuid("CR", run_id, "CONVERGENCE"),
            "run_id": run_id,
        }

    if stage2_block:
        result["stage2_blueprint"] = {
            "raw_block": stage2_block,
            "uuid": make_node_uuid("CR", run_id, "STAGE2-BLUEPRINT"),
            "run_id": run_id,
        }

    return result
