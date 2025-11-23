# pipeline/demo_builder.py
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json, re, os, numpy as np

ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def parse_zero_shot_file(path):
    items, buf = [], {"Q":"", "A":"", "pred":""}
    with open(path, "r", encoding="utf-8") as f:
        block=""
        for line in f:
            if line.strip()=="" and block:
                items.append(block); block=""
            else:
                block += line
        if block: items.append(block)
    parsed=[]
    for blk in items:
        q = re.search(r"Q:\s*(.*)", blk)
        a = re.search(r"A:\s*(.*)", blk, re.S)
        p = re.search(r"Therefore, the answer is:\s*(.*)", blk)
        parsed.append({
            "question": q.group(1).strip() if q else "",
            "rationale": a.group(1).strip() if a else "",
            "pred_ans": p.group(1).strip() if p else ""
        })
    # 过滤：中文按字符数
    parsed = [x for x in parsed if 10 <= len(x["question"]) <= 2000 and len(x["rationale"])>=20]
    return parsed

def select_kmeans_centers(samples, k=4):
    encoder = SentenceTransformer(ENCODER_NAME)
    vecs = encoder.encode([s["question"] + " " + s["rationale"] for s in samples])
    k = min(k, len(samples))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(vecs)
    centers = []
    for cid in range(k):
        idxs = np.where(km.labels_==cid)[0]
        center_idx = idxs[np.argmin(np.linalg.norm(vecs[idxs]-km.cluster_centers_[cid], axis=1))]
        centers.append(samples[center_idx])
    return centers

def build_demos(raw_path, out_json, brand_ctx="", exposure_goals=None, k=4):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    items = parse_zero_shot_file(raw_path)
    chosen = select_kmeans_centers(items, k=k)
    for c in chosen:
        c["brand_ctx"] = brand_ctx
        c["exposure_goals"] = exposure_goals or []
    json.dump(chosen, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return out_json
