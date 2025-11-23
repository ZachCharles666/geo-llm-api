# -*- coding: utf-8 -*-
# geo_report.py

import html
import json
import time
from typing import Dict

def render_report_html(project_title: str,
                       query: str,
                       src_text: str,
                       opt_text: str,
                       score: Dict) -> str:
    """
    生成包含七维评分、客观指标与总分的简易 HTML 报告。
    """
    def esc(x): return html.escape(str(x))

    obj = score.get("objective", {})
    metrics_html = f"""
    <ul>
      <li>Relevance: {score['relevance']:.2f}</li>
      <li>Influence: {score['influence']:.2f}</li>
      <li>Uniqueness: {score['uniqueness']:.2f}</li>
      <li>Diversity: {score['diversity']:.2f}</li>
      <li>Subjective Position: {score['subjective_position']:.2f}</li>
      <li>Subjective Count: {score['subjective_count']:.2f}</li>
      <li>Follow-Up Likelihood: {score['follow_up']:.2f}</li>
    </ul>
    <h3>Objective Metrics</h3>
    <ul>
      <li>Compression Ratio: {obj.get('compression_ratio', 0):.3f}</li>
      <li>Type-Token Ratio (TTR): {obj.get('ttr', 0):.3f}</li>
      <li>Reading Ease: {obj.get('reading_ease', 0):.1f}</li>
    </ul>
    <h2>Total GEO-Score: {score['geo_score']:.1f} / 100</h2>
    <p><small>Model: {esc(score['model_used'])} | Samples: {score.get('samples',1)} | Latency: {score['latency_ms']} ms</small></p>
    """

    html_doc = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>{esc(project_title)} - GEO Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; margin: 24px; line-height: 1.6; }}
pre {{ background: #f7f7f7; padding: 12px; border-radius: 8px; white-space: pre-wrap; }}
.card {{ border: 1px solid #eee; border-radius: 12px; padding: 16px; margin: 16px 0; }}
h1,h2,h3 {{ margin: 12px 0; }}
</style>
</head>
<body>
<h1>{esc(project_title)}</h1>
<p><small>Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}</small></p>

<div class="card">
  <h2>Query</h2>
  <pre>{esc(query)}</pre>
</div>

<div class="card">
  <h2>Original Text</h2>
  <pre>{esc(src_text)}</pre>
</div>

<div class="card">
  <h2>Optimized Text</h2>
  <pre>{esc(opt_text)}</pre>
</div>

<div class="card">
  <h2>GEO-Score</h2>
  {metrics_html}
</div>

<div class="card">
  <h3>Raw JSON</h3>
  <pre>{esc(json.dumps(score, ensure_ascii=False, indent=2))}</pre>
</div>

</body>
</html>
"""
    return html_doc
