# -*- coding: utf-8 -*-
# geo_impression.py
"""
GEO 论文模式（with_citations）用的“引用份额”指标：
- Word：按引用句子的词数加权（引用越长，份额越大）
- Pos：按引用出现的句子位置加权（越靠前，份额越大）
- WordPos：两者结合（简单可解释的乘积/归一）
支持单次分布与 before/after 的 Δ 提升计算。
"""

import re
from typing import Dict, List, Tuple

_SENT_SPLIT = re.compile(r'[。！？!?。\n]+')
_TOKEN_RE   = re.compile(r"\w+|[\u4e00-\u9fff]|[^\s]")  # 中英混合粗分词

def _split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return sents or [text.strip()]

def _tokens(sent: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(sent) if t.strip()]

def _find_citation_ids(sent: str) -> List[int]:
    """
    抓取句子中所有 [number] 引用编号，如 [1], [2]...
    若模型输出带空格如 [ 1 ] 也能兼容。
    """
    ids = []
    for m in re.finditer(r"\[\s*(\d+)\s*\]", sent):
        try:
            ids.append(int(m.group(1)))
        except:
            pass
    return ids

def _collect_citation_events(answer_with_citations: str) -> List[Tuple[int, int, int]]:
    """
    返回列表：[(citation_id, sent_idx, word_count), ...]
    - citation_id: 1,2,3...
    - sent_idx: 0-based 句子序号
    - word_count: 该句的“词（或字）数量”（粗分词）
    """
    events: List[Tuple[int, int, int]] = []
    sents = _split_sentences(answer_with_citations)
    for i, s in enumerate(sents):
        ids = _find_citation_ids(s)
        if not ids:
            continue
        wc = len(_tokens(s))
        for cid in ids:
            events.append((cid, i, wc))
    return events

def impression_word_count(answer_with_citations: str, n_sources: int) -> Dict[int, float]:
    """
    Word：按引用句子的词数累计到对应来源，最后归一化为份额分布（和为1）。
    """
    events = _collect_citation_events(answer_with_citations)
    if n_sources <= 0:
        n_sources = max([cid for cid,_,_ in events], default=0)
    raw = {i: 0.0 for i in range(1, n_sources+1)}
    for cid, _, wc in events:
        if cid in raw:
            raw[cid] += float(wc)
    s = sum(raw.values()) or 1.0
    return {k: v/s for k, v in raw.items()}

def impression_pos_count(answer_with_citations: str, n_sources: int) -> Dict[int, float]:
    """
    Pos：按句子位置加权。越靠前权重越高。这里用简单的 1/(idx+1)。
    """
    events = _collect_citation_events(answer_with_citations)
    if n_sources <= 0:
        n_sources = max([cid for cid,_,_ in events], default=0)
    raw = {i: 0.0 for i in range(1, n_sources+1)}
    for cid, idx, _ in events:
        if cid in raw:
            raw[cid] += 1.0 / float(idx + 1)  # 1, 1/2, 1/3, ...
    s = sum(raw.values()) or 1.0
    return {k: v/s for k, v in raw.items()}

def impression_wordpos_count(answer_with_citations: str, n_sources: int) -> Dict[int, float]:
    """
    WordPos：同时考虑词数与位置。使用 (wc * 1/(idx+1)) 累计，再归一化。
    """
    events = _collect_citation_events(answer_with_citations)
    if n_sources <= 0:
        n_sources = max([cid for cid,_,_ in events], default=0)
    raw = {i: 0.0 for i in range(1, n_sources+1)}
    for cid, idx, wc in events:
        if cid in raw:
            raw[cid] += float(wc) * (1.0 / float(idx + 1))
    s = sum(raw.values()) or 1.0
    return {k: v/s for k, v in raw.items()}

def compute_delta(
    before_answer: str,
    after_answer: str,
    n_sources: int,
    target_idx: int,
    mode: str = "WordPos"
) -> Dict[str, object]:
    """
    计算目标来源在优化前后份额的提升 Δ = after - before
    返回：
    {
      "mode": "WordPos"|"Word"|"Pos",
      "n_sources": int,
      "target_idx": int,
      "dist_before": {1: float, 2: float, ...},
      "dist_after":  {1: float, 2: float, ...},
      "delta": float
    }
    """

    # ---- 1) 选择模式（这里一定要定义 mode_name）----
    mode_in = (mode or "WordPos").strip().lower()
    if mode_in in ("wordpos", "word_pos", "word+pos"):
        f = impression_wordpos_count
        mode_name = "WordPos"
    elif mode_in == "word":
        f = impression_word_count
        mode_name = "Word"
    elif mode_in == "pos":
        f = impression_pos_count
        mode_name = "Pos"
    else:
        f = impression_wordpos_count
        mode_name = "WordPos"

    # ---- 2) 自动对齐 n_sources（取 before/after 内出现的最大 [x] 作为下限）----
    def _max_cid(txt: str) -> int:
        evts = _collect_citation_events(txt or "")
        return max([cid for cid, _, _ in evts], default=0)

    max_id_seen = max(_max_cid(before_answer), _max_cid(after_answer))
    try:
        n = int(n_sources or 1)
    except Exception:
        n = 1
    n = max(n, max_id_seen, 1)

    # ---- 3) 计算两个分布 ----
    dist_before = f(before_answer or "", n)
    dist_after  = f(after_answer  or "", n)

    # ---- 4) 目标来源越界保护 ----
    try:
        t_idx = int(target_idx or 1)
    except Exception:
        t_idx = 1
    if t_idx < 1 or t_idx > n:
        t_idx = 1

    # ---- 5) Δ 提升 ----
    delta = float(dist_after.get(t_idx, 0.0) - dist_before.get(t_idx, 0.0))

    return {
        "mode": mode_name,
        "n_sources": n,
        "target_idx": t_idx,
        "dist_before": dist_before,
        "dist_after": dist_after,
        "delta": round(delta, 4)
    }
