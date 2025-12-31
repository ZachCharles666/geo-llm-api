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
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

try:
    from geo_ids import make_run_id, make_node_uuid
except ImportError:
    # 如果你之前已经在本文件顶部 import 过，可以删掉这个兜底
    from .geo_ids import make_run_id, make_node_uuid


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

def stage2_text_to_blueprint(
    stage2_md: str,
    user_question: str = "",
    brand_brief: str = "",
    must_expose: str = "",
    expo_hint: str = "",
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    将 Stage2 Markdown（内容矩阵）解析为 Blueprint JSON。

    解析目标：
    - I. 内容链总览  -> chains[]
    - II. 标题矩阵展开 -> titles[]
    - III. 标题级内容概述 -> abstracts[]
    - IV. 执行清单建议 -> routes{ execution_order, recommended_channels }

    说明：
    - 尽量避免改动原有 Blueprint 顶层结构；
    - 内部使用相对宽松的正则，能 parse 就 parse，parse 不了就留空。
    """
    md = stage2_md or ""
    if not run_id:
        run_id = make_run_id()

    blueprint_id = make_node_uuid("BP", run_id, "STAGE2-BLUEPRINT")

    # ---------- 0) 工具函数：按 "### I./II./III./IV." 切分大段 ----------

    def _get_section(md_text: str, label: str) -> str:
        """
        label: "I" / "II" / "III" / "IV"
        返回对应大段（含标题行），找不到则返回空串。
        """
        # 当前段落起点
        m_start = re.search(rf"^###\s+{label}\.\s.*$", md_text, flags=re.MULTILINE)
        if not m_start:
            return ""

        start = m_start.start()

        # 下一段落起点
        m_next = re.search(
            r"^###\s+(I|II|III|IV)\.\s.*$",
            md_text[m_start.end() :],
            flags=re.MULTILINE,
        )
        if m_next:
            end = m_start.end() + m_next.start()
        else:
            end = len(md_text)

        return md_text[start:end].strip()

    sec_I = _get_section(md, "I")  # 内容链总览
    sec_II = _get_section(md, "II")  # 标题矩阵展开
    sec_III = _get_section(md, "III")  # 标题级内容概述
    sec_IV = _get_section(md, "IV")  # 执行清单建议

    # ---------- 1) 解析 I. 内容链总览 -> chains[] ----------

    chains: List[Dict[str, Any]] = []
        
    if sec_I:
            # 思路：先按「Source_Node block」切片，再在每个 block 里分别匹配字段
        # 这样对缩进 / 空行的容忍度更高
        block_pattern = re.compile(
            r"(?:^\s*\d+\.\s*)?\*\*Source_Node\*\*.*?"
            r"(?=^\s*\d+\.\s*\*\*Source_Node\*\*|^####\s+MLC-|^###\s+II\.|\Z)",
            flags=re.MULTILINE | re.DOTALL,
        )

        for m_block in block_pattern.finditer(sec_I):
            chunk = m_block.group(0)

            # 分别匹配字段，缺少关键字段就跳过该 block
            m_source = re.search(
                r"\*\*Source_Node\*\*\s*:\s*(MLC-\d+)", chunk
            )
            m_cc = re.search(
                r"\*\*CC_ID\*\*\s*:\s*([A-Za-z0-9\-_]+)", chunk
            )
            m_target = re.search(
                r"\*\*目标受众\*\*\s*:\s*(.+)", chunk
            )
            m_belief = re.search(
                r"\*\*要改变的认知\*\*\s*:\s*(.+)", chunk
            )
            m_trigger = re.search(
                r"\*\*触发条件[（\(].*?[）\)]\*\*\s*:\s*(.+)", chunk
            )
            m_ev = re.search(
                r"\*\*可验证的论据方向\*\*\s*:\s*((?:\s*[-•]\s*.+\n?)*)",
                chunk,
            )

            # 至少要有 Source_Node / CC_ID / 目标受众 / 要改变的认知
            if not (m_source and m_cc and m_target and m_belief):
                continue

            source = m_source.group(1).strip()
            cc_id = m_cc.group(1).strip()
            target = m_target.group(1).strip()
            belief = m_belief.group(1).strip()
            trigger = m_trigger.group(1).strip() if m_trigger else ""

            evidence_list: List[str] = []
            if m_ev:
                evidence_raw = m_ev.group(1) or ""
                for ln in evidence_raw.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    ln = re.sub(r"^[-•]\s*", "", ln)
                    if ln:
                        evidence_list.append(ln)

            chains.append(
                {
                    "cc_id": cc_id,
                    "source_node": source,
                    "target_audience": target,
                    "belief_to_change": belief,
                    "trigger": trigger,
                    "evidence_directions": evidence_list,
                }
            )


    # ---------- 2) 解析 II. 标题矩阵展开 -> titles[] ----------

    titles: List[Dict[str, Any]] = []

    if sec_II:
        # 按 "#### MLC-01-CC1" 这样的子标题拆块
        block_pattern = re.compile(
            r"^####\s+(?P<cc_id>MLC-\d+-CC\d+).*?$"
            r"(?P<body>.*?)(?=^####\s+MLC-\d+-CC\d+|\Z)",
            flags=re.MULTILINE | re.DOTALL,
        )

        for m in block_pattern.finditer(sec_II):
            cc_id = m.group("cc_id").strip()
            body = m.group("body") or ""

            def _find_title(tag: str) -> Optional[str]:
                # 匹配 "- **H1**: xxx"
                mm = re.search(
                    rf"-\s*\*\*{tag}\*\*\s*[:：]\s*(.+)",
                    body,
                    flags=re.MULTILINE,
                )
                return mm.group(1).strip() if mm else None

            titles.append(
                {
                    "cc_id": cc_id,
                    "H1": _find_title("H1"),
                    "H2": _find_title("H2"),
                    "H3": _find_title("H3"),
                }
            )

    # ---------- 3) 解析 III. 标题级内容概述 -> abstracts[] ----------

    abstracts: List[Dict[str, Any]] = []

    if sec_III:
        block_pattern = re.compile(
            r"^####\s+(?P<cc_id>MLC-\d+-CC\d+).*?$"
            r"(?P<body>.*?)(?=^####\s+MLC-\d+-CC\d+|\Z)",
            flags=re.MULTILINE | re.DOTALL,
        )

        for m in block_pattern.finditer(sec_III):
            cc_id = m.group("cc_id").strip()
            body = m.group("body") or ""

            # 每个 ##cc_id 段中会有多组「标题/论点/论据方向/品牌露出逻辑」
            # 我们以 "- **标题**:" 为分块起点
            sub_pattern = re.compile(
                r"-\s*\*\*标题\*\*\s*[:：]\s*(?P<title>.+?)\n"
                r"\s*-\s*\*\*论点\*\*\s*[:：]\s*(?P<point>.+?)\n"
                r"\s*-\s*\*\*论据方向\*\*\s*[:：]\s*(?P<evidence>(?:\s*[-•]\s*.+\n?)*)"
                r"\s*-\s*\*\*品牌露出逻辑\*\*\s*[:：]\s*(?P<brand>.+?)(?=\n\s*-\s*\*\*标题\*\*|\Z)",
                flags=re.DOTALL,
            )

            for mm in sub_pattern.finditer(body):
                evidence_raw = mm.group("evidence") or ""
                evidence_list: List[str] = []
                for ln in evidence_raw.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    ln = re.sub(r"^[-•]\s*", "", ln)
                    if ln:
                        evidence_list.append(ln)

                abstracts.append(
                    {
                        "cc_id": cc_id,
                        "title": mm.group("title").strip(),
                        "point": mm.group("point").strip(),
                        "evidence_directions": evidence_list,
                        "brand_logic": mm.group("brand").strip(),
                    }
                )

    # ---------- 4) 解析 IV. 执行清单建议 -> routes{} ----------

    routes: Dict[str, Any] = {
        "execution_order": [],  # [{cc_id, priority}, ...]
        "recommended_channels": [],  # [{cc_id, channels}, ...]
    }

    if sec_IV:
        # 4.1 优先级建议（高/中/低）
        # 形如：- **高**: MLC-01-CC1，针对……
        priority_pattern = re.compile(
            r"-\s*(?:\*\*)?(?P<prio>高|中|低)(?:\*\*)?\s*[:：]\s*(?P<rest>.+)",
            flags=re.MULTILINE,
        )

        
        for m in priority_pattern.finditer(sec_IV):
            prio = m.group("prio")
            rest = m.group("rest")

            # 把 “MLC-01-CC1、MLC-01-CC2” 这种拆成数组
            cc_ids = re.findall(r"(MLC-\d+-CC\d+)", rest)
            if not cc_ids:
                # 如果实在没匹配到，至少保留一条“无 cc_id”的记录
                routes["execution_order"].append(
                    {"cc_id": "", "priority": prio, "text": rest.strip()}
                )
                continue

            for cc_id in cc_ids:
                routes["execution_order"].append(
                    {
                        "cc_id": cc_id,
                        "priority": prio,
                        "text": rest.strip(),
                    }
                )

        # 4.2 发布渠道建议
        # 形如：- **MLC-01-CC1**: xxx
        channel_block = sec_IV
        # 尝试只在“发布渠道建议”小节之后匹配
        m_pub = re.search(r"####\s*发布渠道建议.*", sec_IV)
        if m_pub:
            channel_block = sec_IV[m_pub.end() :]

        channel_pattern = re.compile(
            r"-\s*\*\*(?P<cc_id>MLC-\d+-CC\d+)\*\*\s*[:：]\s*(?P<channels>.+)",
            flags=re.MULTILINE,
        )
        
        for m in channel_pattern.finditer(channel_block):
            routes["recommended_channels"].append(
                {
                    "cc_id": m.group("cc_id").strip(),
                    "channels": m.group("channels").strip(),
                }
            )

        # 如果一个渠道都没匹配到，为了向后兼容，至少把 bullet 形式的渠道合并成一条
        if not routes["recommended_channels"]:
            # 简单从 sec_IV 中抓 “发布渠道建议” 后面的所有 "- xxx" 行
            # 1) 找到包含“发布渠道建议”的那一行（无论有没有 #### 或 **）
            m_title = re.search(r"发布渠道建议", sec_IV)
            if m_title:
                tail_block = sec_IV[m_title.end() :]
            else:
                tail_block = sec_IV

            # 2) 抓所有以 "- " 开头的行作为渠道
            ch_lines = []
            for ln in tail_block.splitlines():
                ln = ln.strip()
                if ln.startswith("- "):
                    ch_lines.append(re.sub(r"^-+\s*", "", ln))

            if ch_lines:
                routes["recommended_channels"].append(
                    {
                        "cc_id": "",
                        "channels": "；".join(ch_lines),
                    }
                )


    # ---------- 5) 组装 Blueprint ----------

    blueprint: Dict[str, Any] = {
        "run_id": run_id,
        "blueprint_id": blueprint_id,
        "brand": {
            "brief": brand_brief or "",
            "core_claim": must_expose or "",
            "user_question": user_question or "",
        },
        "chains": chains,
        "titles": titles,
        "abstracts": abstracts,
        "routes": routes,
        "meta": {
            "uuid": blueprint_id,
            "created_at": "",  # 如需可在外层补时间戳
            "stage": "stage2_blueprint_ready",
            "version": "2.0",
        },
    }

    return blueprint

# ============================
# Blueprint 一致性检查工具
# ============================

def check_blueprint_consistency(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """
    对 Blueprint 中的 CC_ID 进行对齐检查：
    - chains / titles / abstracts / routes.recommended_channels
    返回一个结构化的诊断结果，便于后续打 log 或前端展示。
    """
    chains_cc = {
        c.get("cc_id") for c in blueprint.get("chains", []) if c.get("cc_id")
    }
    titles_cc = {
        t.get("cc_id") for t in blueprint.get("titles", []) if t.get("cc_id")
    }
    abstracts_cc = {
        a.get("cc_id") for a in blueprint.get("abstracts", []) if a.get("cc_id")
    }
    routes_cc = {
        r.get("cc_id")
        for r in blueprint.get("routes", {}).get("recommended_channels", [])
        if r.get("cc_id")
    }

    all_cc = sorted(chains_cc | titles_cc | abstracts_cc | routes_cc)

    only_in_chains = sorted(chains_cc - titles_cc - abstracts_cc - routes_cc)
    only_in_titles = sorted(titles_cc - chains_cc - abstracts_cc - routes_cc)
    only_in_abstracts = sorted(abstracts_cc - chains_cc - titles_cc - routes_cc)
    only_in_routes = sorted(routes_cc - chains_cc - titles_cc - abstracts_cc)

    return {
        "all_cc_ids": all_cc,
        "chains_cc": sorted(chains_cc),
        "titles_cc": sorted(titles_cc),
        "abstracts_cc": sorted(abstracts_cc),
        "routes_cc": sorted(routes_cc),
        "only_in_chains": only_in_chains,
        "only_in_titles": only_in_titles,
        "only_in_abstracts": only_in_abstracts,
        "only_in_routes": only_in_routes,
        # 简单的健康度指标（0~1 之间，大致感知用）
        "alignment_score": _calc_alignment_score(
            chains_cc, titles_cc, abstracts_cc, routes_cc
        ),
    }


def _calc_alignment_score(
    chains_cc: set, titles_cc: set, abstracts_cc: set, routes_cc: set
) -> float:
    """
    非严谨打分，只是给一个 0~1 的 alignment 感知：
    - 所有集合完全重合时为 1.0
    - 其他情况下随重叠程度下降
    """
    sets = [chains_cc, titles_cc, abstracts_cc, routes_cc]
    non_empty = [s for s in sets if s]
    if len(non_empty) <= 1:
        return 1.0  # 只有一个集合时，不谈对齐，算满分

    # 交集 / 并集 作为简单指标
    union = set().union(*non_empty)
    inter = non_empty[0].intersection(*non_empty[1:])
    if not union:
        return 1.0

    return round(len(inter) / len(union), 3)
