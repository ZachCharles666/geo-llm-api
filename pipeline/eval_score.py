# pipeline/eval_score.py
import re, json, yaml

def load_schema(path="config/eval_schema.yaml"):
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def structure_check(res, schema):
    ok = isinstance(res, dict) and all(k in res for k in ["logic_chain","evidence_chain","titles_by_node"])
    if not ok: return 0
    lc = res["logic_chain"]
    if not isinstance(lc, list): return 0
    n = len(lc)
    rng = schema["logic_chain"]["node_count"]
    base = 5 if rng["min"] <= n <= rng["max"] else 0
    opt_bonus = 5 if n in rng["optimal"] else 0
    # 证据链逐节点对齐（简单校验）
    ec = res["evidence_chain"]
    aligned = min(len(ec), n) / max(1, n)
    return base + opt_bonus if aligned>=0.8 else base

def exposure_coverage(res, exposure_goals, schema):
    text = json.dumps(res, ensure_ascii=False)
    hit = sum(1 for k in exposure_goals if k in text)
    auto_score = 10 * (hit / max(1, len(exposure_goals)))
    # 人工部分先留接口，这里默认给占位分
    manual_score = 10 * 0.7
    return auto_score + manual_score

def title_usability(res, schema):
    rng = schema["title"]["count_range"]
    ok_nodes = 0
    for item in res.get("titles_by_node", []):
        cnt = len(item.get("titles", []))
        if rng[0] <= cnt <= rng[1]: ok_nodes += 1
    ratio = ok_nodes / max(1, len(res.get("logic_chain", [])))
    return 20 * ratio

def overall_score(res, exposure_goals, schema):
    w = schema["weights"]
    s_struct = structure_check(res, schema)                # /20
    s_expo   = exposure_coverage(res, exposure_goals, schema)  # /20
    s_title  = title_usability(res, schema)                # /20
    # 其余两项先留接口（逻辑一致性/可执行性），默认给中位10/10（可后续人工复评替换）
    s_logic, s_exec = 10, 10
    s_evid = 15  # 证据对齐度占位（后续可做更细自动判定）
    total = (s_struct * w["structure_completeness"]
            + s_evid * w["evidence_alignment"]
            + s_expo * w["exposure_coverage"]
            + s_title * w["title_usability"]
            + s_logic * w["logic_consistency"]
            + s_exec * w["executability"]) * 5  # 权重总和=1，单项分*5映射到100
    return round(total, 1)
