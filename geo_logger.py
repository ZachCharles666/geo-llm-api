# -*- coding: utf-8 -*-
# geo_logger.py
"""
记录每次 GEO 优化与评分结果，便于后续可视化和分析
"""

import json
import os
import datetime
from typing import Dict, Any, List

LOG_PATH = "geo_log.json"

def _read_log() -> List[Dict[str, Any]]:
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _write_log(data: List[Dict[str, Any]]):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def log_run(
    model: str,
    query: str,
    original_text: str,
    optimized_text: str,
    score_dict: Dict[str, Any],
    mode: str = "single_text",
):
    """
    将一次 GEO 运行及评分结果写入 geo_log.json
    """
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "mode": mode,
        "query": query[:200],  # 截断防止太长
        "len_before": len(original_text or ""),
        "len_after": len(optimized_text or ""),
        "geo_score": round(score_dict.get("geo_score", 0), 2),
        "latency_ms": score_dict.get("latency_ms", 0),
        "details": score_dict,
    }

    data = _read_log()
    data.append(record)
    _write_log(data)
    return record

def get_recent_logs(limit: int = 10) -> List[Dict[str, Any]]:
    """
    获取最近 N 条记录（按时间倒序）
    """
    data = _read_log()
    return sorted(data, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
