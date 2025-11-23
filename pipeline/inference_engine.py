# pipeline/inference_engine.py
import os, json, httpx

def call_model(prompt, provider="deepseek", temperature=0.2):
    if provider=="deepseek":
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
        payload = {"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],
                   "temperature":temperature}
    elif provider=="qwen":
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}"}
        payload = {"model":"qwen-turbo","messages":[{"role":"user","content":prompt}],
                   "temperature":temperature}
    else:
        raise NotImplementedError
    r = httpx.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    return txt

def safe_json_parse(text):
    # 从模型输出里提取JSON（容错）
    import re, json
    m = re.search(r"\{[\s\S]*\}", text)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
