# pipeline/post_extract.py
from .inference_engine import call_model, safe_json_parse

EXTRACTOR = """请把上文非结构化推理内容，抽取为严格JSON：
{
  "logic_chain": ["..."],
  "evidence_chain": [{"node":"", "evidence": {"data":"", "industry":"", "media":"", "extra":""}, "gaps":""}],
  "titles_by_node": [{"node":"", "titles":["",""]}]
}
不得添加说明文字。"""

def two_stage_extract(coT_text, provider="deepseek"):
    prompt = coT_text + "\n\n" + EXTRACTOR
    out = call_model(prompt, provider=provider, temperature=0.0)
    return safe_json_parse(out)
