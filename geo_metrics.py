# -*- coding: utf-8 -*-
# geo_metrics.py

import math
import re
from collections import Counter
from typing import Tuple

_HAN_RE = re.compile(r'[\u4e00-\u9fff]')
_SENT_SPLIT = re.compile(r'[。！？!?\n]+')

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def compression_ratio(src: str, opt: str) -> float:
    """len(opt)/len(src)；<1 更精炼"""
    return _safe_div(len(opt), len(src), 1.0)

def _tokenize(text: str) -> Tuple[list, bool]:
    """粗分词：若包含大量汉字，则按汉字字符级统计；否则按空白分词。"""
    han_chars = _HAN_RE.findall(text)
    if len(han_chars) >= max(10, len(text) * 0.2):
        return han_chars, True
    tokens = re.findall(r"\w+|\S", text)
    return [t for t in tokens if t.strip()], False

def type_token_ratio(text: str) -> float:
    """TTR：unique_tokens / total_tokens （0~1）"""
    toks, _ = _tokenize(text)
    if not toks:
        return 0.0
    uniq = len(set(toks))
    return uniq / len(toks)

def reading_ease(text: str, lang: str = "auto") -> float:
    """
    简化的可读性：对中文以“平均句长(字符)”反算；对非中文以“平均句长(词)”反算。
    统一映射为 0~100 的启发式分值（越大越易读）。
    """
    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return 100.0

    toks, is_zh = _tokenize(text)
    if is_zh:
        total_chars = sum(len(s) for s in sentences)
        avg_chars_per_sent = _safe_div(total_chars, len(sentences), 1.0)
        # 以 20 字/句为舒适基准，句长每 +1，分值 -1；裁剪到 [0,100]
        score = 100.0 - max(0.0, (avg_chars_per_sent - 20.0))
        return max(0.0, min(100.0, score))
    else:
        # 粗略：平均每句词数越少越易读
        words_per_sent = _safe_div(len(toks), len(sentences), 1.0)
        # 以 20 词/句为舒适基准，句长每 +1，分值 -2
        score = 100.0 - max(0.0, (words_per_sent - 20.0)) * 2.0
        return max(0.0, min(100.0, score))
