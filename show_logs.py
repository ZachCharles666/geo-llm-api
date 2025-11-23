# -*- coding: utf-8 -*-
# show_logs.py
# 说明：可视化 geo_log.json（评分历史、筛选与折线趋势 + 导出 CSV）

import os
import json
import csv
import io
from datetime import datetime
from typing import List, Dict, Any, Tuple

import gradio as gr
import matplotlib.pyplot as plt

LOG_PATH = "geo_log.json"


# ---------------------------
# 数据读取与处理
# ---------------------------
def _read_logs() -> List[Dict[str, Any]]:
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 兜底：确保是列表
        if not isinstance(data, list):
            return []
        return data
    except Exception:
        return []


def _parse_ts(ts: str) -> datetime:
    # geo_logger 使用 datetime.isoformat(timespec="seconds")
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        # 其它格式兜底
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(ts, fmt)
            except Exception:
                pass
    # 都失败就给当前时间，避免排序崩
    return datetime.now()


def _filter_logs(
    logs: List[Dict[str, Any]],
    model: str = "全部",
    mode: str = "全部",
    keyword: str = "",
) -> List[Dict[str, Any]]:
    keyword = (keyword or "").strip().lower()
    def _ok(row: Dict[str, Any]) -> bool:
        if model != "全部" and str(row.get("model", "")).strip() != model:
            return False
        if mode != "全部" and str(row.get("mode", "")).strip() != mode:
            return False
        if keyword:
            hay = f"{row.get('query','')} {row.get('details',{})}".lower()
            return keyword in hay
        return True

    rows = [r for r in logs if _ok(r)]
    rows.sort(key=lambda r: _parse_ts(r.get("timestamp", "")), reverse=True)
    return rows


def _unique_values(logs: List[Dict[str, Any]], key: str) -> List[str]:
    vals = []
    for r in logs:
        v = str(r.get(key, "")).strip() or ""
        if v and v not in vals:
            vals.append(v)
    vals.sort()
    return vals


def _make_table_rows(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in logs:
        rows.append({
            "时间": r.get("timestamp", ""),
            "模型": r.get("model", ""),
            "模式": r.get("mode", ""),
            "GEO-Score": r.get("geo_score", 0),
            "前长度": r.get("len_before", 0),
            "后长度": r.get("len_after", 0),
            "耗时ms": r.get("latency_ms", 0),
            "查询摘要": (r.get("query", "") or "")[:80],
        })
    return rows


def _make_score_series(logs: List[Dict[str, Any]], top_n: int) -> Tuple[list, list]:
    # 取最近 top_n 条（过滤后已按时间倒序）
    pick = logs[:max(1, int(top_n))]
    # 反转为时间正序便于连线
    pick = list(reversed(pick))
    xs = [r.get("timestamp", "") for r in pick]
    ys = [float(r.get("geo_score", 0)) for r in pick]
    return xs, ys


def _plot_line(xs: List[str], ys: List[float]):
    fig = plt.figure(figsize=(7, 3.5), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, marker="o")
    ax.set_title("GEO-Score 趋势（最近N次）")
    ax.set_xlabel("时间")
    ax.set_ylabel("GEO-Score")
    ax.grid(True, linestyle="--", alpha=0.3)
    # x 轴刻度稀疏一些
    if len(xs) > 8:
        step = max(1, len(xs) // 8)
        for label in ax.xaxis.get_ticklabels():
            label.set_visible(False)
        for i, label in enumerate(ax.xaxis.get_ticklabels()):
            if i % step == 0:
                label.set_visible(True)
        fig.autofmt_xdate(rotation=20)
    else:
        plt.xticks(rotation=20)
    plt.ylim(0, 100)
    plt.tight_layout()
    return fig


def _to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


# ---------------------------
# Gradio 回调
# ---------------------------
def refresh_options():
    logs = _read_logs()
    models = ["全部"] + _unique_values(logs, "model")
    modes = ["全部"] + _unique_values(logs, "mode")
    return gr.update(choices=models, value=models[0]), gr.update(choices=modes, value=modes[0])


def run_query(model, mode, keyword, top_n):
    logs = _read_logs()
    flt = _filter_logs(logs, model, mode, keyword)
    table = _make_table_rows(flt)

    # 绘图
    if flt:
        xs, ys = _make_score_series(flt, top_n)
        fig = _plot_line(xs, ys)
    else:
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "暂无匹配记录", ha="center", va="center")
        plt.axis("off")

    # 导出文件
    csv_bytes = _to_csv_bytes(table)
    csv_path = None
    if csv_bytes:
        # 即时生成一个内存文件给 File 组件
        # Gradio 支持直接返回 (name, bytes) 的 tuple
        csv_path = ("geo_log_export.csv", csv_bytes)

    summary = f"共 {len(flt)} 条记录；显示最近 {min(len(flt), int(top_n))} 条用于趋势图。"
    return table, fig, csv_path, summary


# ---------------------------
# UI
# ---------------------------
with gr.Blocks(title="GEO-Max 日志浏览器", analytics_enabled=False) as demo:
    gr.Markdown("## 🗂️ GEO-Max 日志浏览器\n查看 `geo_log.json` 的历史记录、筛选与 GEO-Score 趋势，并可导出 CSV。")

    with gr.Row():
        btn_refresh = gr.Button("🔄 刷新选项", variant="secondary")
        dd_model = gr.Dropdown(choices=["全部"], value="全部", label="模型")
        dd_mode = gr.Dropdown(choices=["全部"], value="全部", label="模式")
        tb_keyword = gr.Textbox(label="关键词（查询/详情里模糊匹配）", placeholder="可留空")
        sl_topn = gr.Slider(5, 200, value=50, step=1, label="趋势图取最近 N 条")

    btn_query = gr.Button("📊 查询并绘图", variant="primary")

    with gr.Row():
        log_table = gr.Dataframe(
            headers=["时间","模型","模式","GEO-Score","前长度","后长度","耗时ms","查询摘要"],
            datatype=["str","str","str","number","number","number","number","str"],
            label="日志表格（按时间倒序）",
            interactive=False,
            wrap=True
        )

    with gr.Row():
        plot = gr.Plot(label="GEO-Score 趋势")
    with gr.Row():
        file_csv = gr.File(label="导出 CSV（点击下载）", interactive=False)
    note = gr.Markdown("")

    # 事件
    btn_refresh.click(fn=refresh_options, inputs=None, outputs=[dd_model, dd_mode])
    btn_query.click(fn=run_query, inputs=[dd_model, dd_mode, tb_keyword, sl_topn], outputs=[log_table, plot, file_csv, note])

if __name__ == "__main__":
    # 本地访问：127.0.0.1:7863
    demo.launch(server_name="127.0.0.1", server_port=7863, share=False)
