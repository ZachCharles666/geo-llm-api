# providers_groq_gemini.py
# 统一多 Provider 入口：Groq / Gemini / Grok
import os
import requests
import json
from typing import List, Dict, Any, Optional

# ==============================
#  G R O Q   (免费，超快推理)
# ==============================
class GroqProvider:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base = "https://api.groq.com/openai/v1"

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("缺少 GROQ_API_KEY")

        url = f"{self.base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            data["max_tokens"] = max_tokens

        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Groq error: {resp.status_code}, {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]


# ==============================
#  G E M I N I   (免费层)
# ==============================

class GeminiProvider:
    """
    Gemini 调用适配层（REST 版）

    - 优先从 GEMINI_API_KEY 读取，其次 GOOGLE_API_KEY
    - 使用 v1beta generateContent 接口
    - 与 google-genai SDK 底层一致，只是这里直接用 HTTP 方便集成
    """
    def __init__(self):
        # ✅ 同时兼容 GEMINI_API_KEY / GOOGLE_API_KEY，哪个有就用哪个
        api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        self.api_key = api_key
        self.base = os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/models",
        )

        if not self.api_key:
            raise RuntimeError(
                "缺少 Gemini API Key，请在环境变量 GEMINI_API_KEY 或 GOOGLE_API_KEY 中配置。"
            )

    def chat(
        self,
        text: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        统一的文本补全接口：
        - text: 纯文本 Prompt
        - model: 例如 'gemini-1.5-flash' / 'gemini-1.5-pro' / 'gemini-2.5-flash'
        """
        if not self.api_key:
            raise RuntimeError(
                "Gemini 未配置 API Key，请在 GEMINI_API_KEY 或 GOOGLE_API_KEY 中设置。"
            )

        url = f"{self.base}/{model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        # 对齐 Google 官方 generateContent 的请求结构
        data: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {"text": text}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if max_output_tokens is not None:
            data["generationConfig"]["maxOutputTokens"] = max_output_tokens

        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=120)

        if resp.status_code != 200:
            # 这里直接抛出错误，让上层（geo_core / app.py）可以在 UI 上显示到具体报错
            raise RuntimeError(f"Gemini error: {resp.status_code}, {resp.text}")

        j = resp.json()
        try:
            return j["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            # 如果响应结构不符合预期，给出更易懂的错误信息
            raise RuntimeError(f"Gemini 响应解析失败：{e}，原始返回：{j}")


# ==============================
#  G R O K  (xAI, OpenAI-compatible)
# ==============================

class GrokProvider:
    def __init__(self):
        self.api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        self.base = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "grok-2-latest",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("缺少 GROK_API_KEY 或 XAI_API_KEY")

        url = f"{self.base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            data["max_tokens"] = max_tokens

        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Grok error: {resp.status_code}, {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]


# =====================================
#   Provider Registry（统一模型入口）
# =====================================

class ModelHub:
    def __init__(self):
        self.groq = GroqProvider()
        self.gemini = GeminiProvider()
        self.grok = GrokProvider()

    def run(self, provider: str, prompt: str, **kw) -> str:
        """
        provider: "groq" | "gemini" | "grok"
        """
        p = (provider or "").lower().strip()

        if p == "groq":
            return self.groq.chat(
                messages=[{"role": "user", "content": prompt}],
                **kw
            )
        if p == "gemini":
            return self.gemini.chat(prompt, **kw)
        if p == "grok":
            return self.grok.chat(
                messages=[{"role": "user", "content": prompt}],
                **kw
            )

        raise RuntimeError(f"Unknown provider: {provider}")

# =====================================
#   兼容旧代码的适配层（norm_provider / DEFAULT_MODELS）
# =====================================

# 前端下拉框里的“可选项” → 内部 provider 标识
# 你现在 UI 里用的是：
# ["Groq", "Gemini", "Grok", "DeepSeek", "通义千问", "文心一言"]
UI_PROVIDER_MAP = {
    "Groq": "groq",
    "Gemini": "gemini",
    "Grok": "grok",
    # 暂时没有单独 provider，就都先走 groq 这条通道
    "DeepSeek": "groq",
    "通义千问": "groq",
    "文心一言": "groq",
}


def norm_provider(model_ui: str) -> str:
    """
    兼容 geo_core 里使用的 norm_provider：
    - 输入：前端下拉选择的文案，例如 "Groq" / "Gemini"
    - 输出：内部 provider 标识："groq" / "gemini" / "grok"
    """
    key = (model_ui or "").strip()
    return UI_PROVIDER_MAP.get(key, "groq")  # 默认走 groq


# 每个 provider 对应的默认模型名
# （如果你后面想区分，可以在这里细化）
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",   # 你在 Groq 上常用的模型
    "gemini": "gemini-2.5-pro",        # 对应上面 GeminiProvider 的默认 model
    "grok": "grok-2-latest",             # Grok 默认模型
}

