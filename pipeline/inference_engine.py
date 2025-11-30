# pipeline/inference_engine.py
import os, json, httpx
from typing import Optional
from providers_groq_gemini import ModelHub

hub = ModelHub()

# 统一 provider 别名
ALIASES = {
    "groq": "groq",
    "gemini": "gemini",
    "grok": "grok",
    # 国内旧的也保留
    "deepseek": "deepseek",
    "qwen": "qwen",
    "dashscope": "qwen",
}

def call_model(prompt: str, provider: str = "deepseek", temperature: float = 0.2, model: Optional[str]=None):
    """
    手动优先 + 自动 fallback:
    - provider 指定 groq/gemini/grok/deepseek/qwen
    - 如果是 groq/gemini/grok → 走 hub.run
    - 如果是 deepseek/qwen → 走旧逻辑
    """
    p = ALIASES.get((provider or "").lower().strip(), provider)

    # 1) 走新 Provider（Groq / Gemini / Grok）
    if p in ("groq", "gemini", "grok"):
        kw = {"temperature": temperature}
        if model:
            kw["model"] = model
        return hub.run(p, prompt, **kw)

    # 2) 旧国内 Provider（DeepSeek / Qwen）
    if p == "deepseek":
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
        payload = {
            "model": model or "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
    elif p == "qwen":
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}"}
        payload = {
            "model": model or "qwen-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
    else:
        raise NotImplementedError(f"Unknown provider: {provider}")

    r = httpx.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def safe_json_parse(text: str):
    """从模型输出里提取JSON（容错）"""
    import re, json
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
