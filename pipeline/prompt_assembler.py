# pipeline/prompt_assembler.py
import json, textwrap, random

TEMPLATE_FEWSHOT = """【示例#{i}｜GEO-CoT Few-shot】
[目标问题]
{q}

[甲方资料摘录]
{brand}

[期望露出]
{expo}

[标准输出骨架]
1) 逻辑链（逐步列点）
2) 证据链（与逻辑链逐节点对齐；数据/行业/媒体为基本类别，可自定义扩展）
3) 每节点2–3个标题（自然嵌入露出要点）
——
"""

TEMPLATE_TASK = """你是GEO-Max的内容策略与推理专家。请严格按下列结构输出JSON：
{
  "logic_chain": ["...节点1", "...节点2", "..."],
  "evidence_chain": [
     {"node":"节点1","evidence":{"data":"...","industry":"...","media":"...","extra":""},"gaps":"..."},
     {"node":"节点2","evidence":{...},"gaps":"..."}
  ],
  "titles_by_node": [
     {"node":"节点1","titles":["...","..."]},
     {"node":"节点2","titles":["...","...","..."]}
  ]
}
注意：不得虚构具体数据；若证据不足请在gaps中标注采集建议。逻辑链建议3–6节点（可在3–8内浮动）。
当前输入：
[目标问题]
{q}

[甲方资料]
{brand}

[期望露出]
{expo}
"""

def assemble_prompt(demos_json_path, q, brand_ctx, exposure_goals, fewshot_k=3):
    demos = json.load(open(demos_json_path, "r", encoding="utf-8"))
    random.shuffle(demos)
    demos = demos[:fewshot_k]
    fewshot = []
    for i, d in enumerate(demos, 1):
        fewshot.append(TEMPLATE_FEWSHOT.format(
            i=i, q=d["question"],
            brand=(d.get("brand_ctx") or "")[:300],
            expo="、".join(d.get("exposure_goals") or [])
        ))
    task = TEMPLATE_TASK.format(q=q, brand=brand_ctx[:1200], expo="、".join(exposure_goals))
    return "\n".join(fewshot) + "\n" + task
