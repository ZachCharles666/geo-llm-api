# main.py — Final aligned version for GEO-Max Engine
import traceback
import time
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from geo_evaluator import evaluate_geo_score
from geo_report import render_report_html
from geo_impression import (
    impression_word_count,
    impression_pos_count,
    impression_wordpos_count,
)

app = FastAPI(title="GEO-Max Engine API", version="1.0.0")


# -----------------------------------
# Request Models
# -----------------------------------
class ScoreRequest(BaseModel):
    # legacy: simplified input
    text: Optional[str] = None

    # aligned fields
    model_name: Optional[str] = Field(None, description="Model name, e.g. qwen-max / gpt-4o / deepseek")
    query: Optional[str] = None
    src_text: Optional[str] = None
    opt_text: Optional[str] = None
    mode: Optional[str] = "single_text"
    samples: Optional[int] = 1

    options: Optional[Dict[str, Any]] = None


class ReportRequest(BaseModel):
    model_name: Optional[str] = None
    query: Optional[str] = None
    src_text: Optional[str] = None
    opt_text: Optional[str] = None
    text: Optional[str] = None  # legacy fallback

    score_json: Optional[Dict[str, Any]] = None
    mode: Optional[str] = "single_text"
    samples: Optional[int] = 1
    options: Optional[Dict[str, Any]] = None


# -----------------------------------
# Helpers
# -----------------------------------
def safe_call(fn, *args, **kwargs):
    try:
        return True, fn(*args, **kwargs)
    except Exception as e:
        return False, {
            "error": str(e),
            "trace": traceback.format_exc()
        }


def normalize_to_evaluator(req: ScoreRequest):
    """
    把外部请求规范化为 evaluate_geo_score 真实参数顺序：
    model_name, query, src_text, opt_text, mode, samples
    """

    # 1) 模型默认值
    model_name = req.model_name or "qwen-max"   # 你可以改 gpt-4o / deepseek-coder 等

    # 2) 文本统一格式
    src_text = req.src_text or req.text or ""     # text fallback → src_text
    opt_text = req.opt_text or src_text           # 默认：opt_text = src_text（baseline）
    query = req.query or ""                       # query 允许为空字符串

    # 3) 模式 & 采样
    mode = req.mode or "single_text"
    samples = req.samples or 1

    return model_name, query, src_text, opt_text, mode, samples


# -----------------------------------
# 1) /score
# -----------------------------------
@app.post("/score")
def score(req: ScoreRequest):
    model_name, query, src_text, opt_text, mode, samples = normalize_to_evaluator(req)

    ok, data = safe_call(
        evaluate_geo_score,
        model_name,
        query,
        src_text,
        opt_text,
        mode,
        samples
    )

    if not ok:
        return {"ok": False, "where": "score", **data}

    return {
        "ok": True,
        "score": data
    }


# -----------------------------------
# 2) /report
# -----------------------------------
@app.post("/report")
def report(req: ReportRequest):
    # 对齐输入
    src_text = req.src_text or req.text or ""
    opt_text = req.opt_text or src_text
    query = req.query or ""
    model = req.model_name or "qwen-max"
    mode = req.mode or "single_text"
    samples = req.samples or 1

    # 如果前端没传 score → 重新算
    if req.score_json:
        score_json = req.score_json
    else:
        ok_s, sdata = safe_call(
            evaluate_geo_score,
            model,
            query,
            src_text,
            opt_text,
            mode,
            samples
        )
        if not ok_s:
            return {"ok": False, "where": "score_for_report", **sdata}
        score_json = sdata

    # 生成 report
    ok_r, rdata = safe_call(
        render_report_html,
        opt_text,
        score_json
    )
    if not ok_r:
        return {"ok": False, "where": "render_report_html", **rdata}

    return {
        "ok": True,
        "score": score_json,
        "html": rdata
    }


# -----------------------------------
# 3) /impression
# -----------------------------------
@app.post("/impression")
def impression(req: ScoreRequest):
    src_text = req.src_text or req.text or ""

    ok_i, idata = safe_call(
        lambda t: {
            "word_count": impression_word_count(t),
            "pos_count": impression_pos_count(t),
            "wordpos_count": impression_wordpos_count(t)
        },
        src_text
    )

    if not ok_i:
        return {"ok": False, "where": "impression", **idata}

    return {"ok": True, **idata}


@app.get("/ping")
def ping():
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
