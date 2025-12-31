# providers_groq_gemini.py
# 说明：
# - 这是“可直接复制粘贴”的最终版本：ModelHub + KeyPool + Provider fallback（Groq/Gemini/Grok）
# - 兼容你现有调用：hub.run(p, prompt, **kw) / provider.chat(...)
# - 支持 3-5 个 Key：用逗号分隔的环境变量 *_API_KEYS
# - 遇到 429 会按 Groq 提示解析建议等待时间，并对“当前 key”做 cooldown，然后自动切换到下一个 key
# - 当某个 provider 的 key 全部都在 cooldown/失效，会自动 fallback 到下一个 provider（默认 groq -> gemini -> grok）
#
# 你需要配置的环境变量（至少一个即可）：
#   GROQ_API_KEYS="k1,k2,k3"      （推荐）
#   GROQ_API_KEY="k1"            （兼容旧版，单 key）
#
#   GEMINI_API_KEYS="g1,g2,g3"   （推荐）
#   GEMINI_API_KEY="g1"          （兼容旧版）
#   GOOGLE_API_KEY="g1"          （兼容旧版）
#
# 可选：
#   GEO_PROVIDER_CHAIN="groq,gemini,grok"   （默认就是这个顺序）
#   GEO_KEY_COOLDOWN_DEFAULT_SEC="600"      （默认 600 秒）
#   GEO_KEYPOOL_SHUFFLE="0|1"               （默认 0，不打乱；你也可以设 1 让 key 轮询更分散）

import os
import json
import time
import re
import threading
from typing import List, Dict, Any, Optional, Tuple

import requests


# =========================
#  KeyPool（多 Key + 冷却）
# =========================

class KeyPool:
    """
    目标：
    - 管理一个 provider 的多把 key
    - 429 / 临时失败时对当前 key 做 cooldown
    - 自动轮询可用 key
    """
    def __init__(self, name: str, keys: List[str], default_cooldown_sec: int = 600, shuffle: bool = False):
        self.name = name
        self.keys = [k.strip() for k in (keys or []) if k and k.strip()]
        self.default_cooldown_sec = int(default_cooldown_sec or 600)
        self.shuffle = bool(shuffle)
        self._lock = threading.Lock()
        self._idx = 0
        # cooldown_until[key] = epoch_seconds
        self._cooldown_until: Dict[str, float] = {}

        if not self.keys:
            raise RuntimeError(f"[KeyPool] Missing API keys for {name}")

        if self.shuffle and len(self.keys) > 1:
            # 为避免引入 random 造成不确定性，这里用“时间片”轻打乱（可选）
            # 你也可以改成 random.shuffle
            offset = int(time.time()) % len(self.keys)
            self.keys = self.keys[offset:] + self.keys[:offset]

    def _is_available(self, key: str, now: float) -> bool:
        until = self._cooldown_until.get(key, 0.0)
        return now >= until

    def mark_cooldown(self, key: str, seconds: Optional[int] = None):
        if not key:
            return
        sec = int(seconds or self.default_cooldown_sec)
        until = time.time() + max(1, sec)
        with self._lock:
            self._cooldown_until[key] = until

    def mark_bad(self, key: str, seconds: Optional[int] = None):
        # 与 cooldown 等价：你也可以按需区分“永久封禁”
        self.mark_cooldown(key, seconds=seconds)

    def pick(self) -> str:
        """
        取一个“当前可用”的 key。
        若所有 key 都在 cooldown，返回一个 cooldown 最短的 key（让上层决定是否等待/继续 fallback）。
        """
        now = time.time()
        with self._lock:
            n = len(self.keys)

            # 尝试最多 n 次找可用 key
            for _ in range(n):
                key = self.keys[self._idx % n]
                self._idx = (self._idx + 1) % n
                if self._is_available(key, now):
                    return key

            # 全部都在 cooldown：选一个最早可用的 key
            best_key = self.keys[0]
            best_until = self._cooldown_until.get(best_key, 0.0)
            for k in self.keys:
                u = self._cooldown_until.get(k, 0.0)
                if u < best_until:
                    best_key, best_until = k, u
            return best_key
    
    def pick_with_index(self) -> Tuple[str, int]:
        """
        原子地返回 (key, index)，避免 pick 之后再读 _idx 产生并发错位。
        """
        now = time.time()
        with self._lock:
            n = len(self.keys)
            if n <= 0:
                raise RuntimeError(f"[KeyPool] No keys for {self.name}")

            for _ in range(n):
                idx = self._idx % n
                key = self.keys[idx]
                self._idx = (self._idx + 1) % n
                if self._is_available(key, now):
                    return key, idx

            # 全部 cooldown：选最早可用的 key
            best_idx = 0
            best_key = self.keys[0]
            best_until = self._cooldown_until.get(best_key, 0.0)
            for i, k in enumerate(self.keys):
                u = self._cooldown_until.get(k, 0.0)
                if u < best_until:
                    best_idx, best_key, best_until = i, k, u
            return best_key, best_idx


    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            cds = {k: max(0, int(self._cooldown_until.get(k, 0.0) - now)) for k in self.keys}
        active = sum(1 for k, left in cds.items() if left == 0)
        return {
            "name": self.name,
            "keys_total": len(self.keys),
            "keys_active": active,
            "cooldowns_sec": cds,
        }


def _split_keys_from_env(primary: str, fallback_single: str) -> List[str]:
    """
    优先读 *_API_KEYS（逗号分隔），否则读 *_API_KEY（单 key）
    """
    v = (os.getenv(primary, "") or "").strip()
    if v:
        return [x.strip() for x in v.split(",") if x and x.strip()]
    single = (os.getenv(fallback_single, "") or "").strip()
    return [single] if single else []

def _is_rate_limit_error(err_msg: str) -> bool:
    s = (err_msg or "").lower()
    if "rate limit" in s:
        return True
    if "rate_limit" in s or "rate limit exceeded" in s:
        return True
    if "tpd" in s or "tpm" in s:
        return True
    if "error: 429" in s or " 429" in s:
        return True
    if '"code"' in s and "rate limit" in s:
        return True
    return False


def _is_retryable_transient(err_msg: str) -> bool:
    """
    除 429 以外的一些典型可重试错误：5xx、超时、连接失败等
    """
    s = (err_msg or "").lower()
    if "timeout" in s or "timed out" in s:
        return True
    if "connection" in s and ("reset" in s or "refused" in s or "aborted" in s):
        return True
    # 简单匹配 HTTP 5xx
    if "http 5" in s or " 500" in s or " 502" in s or " 503" in s or " 504" in s:
        return True
    return False

def _parse_groq_retry_after_seconds(msg: str) -> Optional[int]:
    """
    解析 Groq 429 message 中的 "Please try again in10m11.712s" -> seconds
    """
    if not msg:
        return None
    m = re.search(r"try again in\s*([0-9]+)m([0-9]+(?:\.[0-9]+)?)s", msg, re.IGNORECASE)
    if not m:
        return None
    minutes = int(m.group(1))
    seconds = float(m.group(2))
    sec_i = int(seconds) if float(int(seconds)) == seconds else int(seconds) + 1
    return minutes * 60 + sec_i


# =========================
#  Provider Runtime State (last ok/error)
# =========================
def _ensure_provider_state_fields(obj: Any):
    if not hasattr(obj, "last_ok_at"):
        obj.last_ok_at = None
    if not hasattr(obj, "last_error_at"):
        obj.last_error_at = None
    if not hasattr(obj, "last_error"):
        obj.last_error = None
    if not hasattr(obj, "last_status_code"):
        obj.last_status_code = None
    if not hasattr(obj, "last_retry_after_s"):
        obj.last_retry_after_s = None
    if not hasattr(obj, "active_key_index"):
        obj.active_key_index = None
    if not hasattr(obj, "key_count"):
        obj.key_count = None

def _mark_ok(obj: Any):
    _ensure_provider_state_fields(obj)
    obj.last_ok_at = time.time()
    obj.last_error = None
    obj.last_status_code = 200
    obj.last_retry_after_s = None

def _mark_error(obj: Any, status_code: Optional[int], err_text: str):
    _ensure_provider_state_fields(obj)
    obj.last_error_at = time.time()
    obj.last_error = (err_text or "")[:500]
    obj.last_status_code = status_code
    if status_code == 429:
        obj.last_retry_after_s = _parse_groq_retry_after_seconds(err_text or "")
    else:
        obj.last_retry_after_s = None


# =========================
#  Providers（支持 api_key 覆盖）
# =========================

class GroqProvider:
    def __init__(self):
        self.base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> str:
        key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("缺少 GROQ_API_KEY / GROQ_API_KEYS")

        url = f"{self.base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        data: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
            
            
        # --- state snapshot (minimal) ---
        try:
            # 如果你采用 GROQ_API_KEYS="k1,k2" 这种 KeyPool，建议你把 key_count/active_key_index 写成 KeyPool 的状态
            # 这里先用“可用字段就写”的防御方式
            self.key_count = getattr(self, "key_count", None) or (
                len([k for k in (os.getenv("GROQ_API_KEYS", "") or "").split(",") if k.strip()])
                or (1 if (os.getenv("GROQ_API_KEY") or "").strip() else 0)
            )
        except Exception:
            self.key_count = None
        # 如果你有 KeyPool，这里应写 self.active_key_index = self.pool.idx
        self.active_key_index = getattr(self, "active_key_index", None)


        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=120)
        if resp.status_code != 200:
            _mark_error(self, resp.status_code, resp.text)
            raise RuntimeError(f"Groq error: {resp.status_code}, {resp.text}")
        _mark_ok(self)
        return resp.json()["choices"][0]["message"]["content"]


class GeminiProvider:
    """
    - 同时兼容 GEMINI_API_KEYS / GEMINI_API_KEY / GOOGLE_API_KEY
    - 使用 v1beta generateContent
    """
    def __init__(self):
        self.base = os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/models",
        ).strip()

    def chat(
        self,
        text: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> str:
        key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("缺少 GEMINI_API_KEY / GEMINI_API_KEYS / GOOGLE_API_KEY")

        url = f"{self.base}/{model}:generateContent?key={key}"
        headers = {"Content-Type": "application/json"}

        data: Dict[str, Any] = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"temperature": temperature},
        }
        if max_output_tokens is not None:
            data["generationConfig"]["maxOutputTokens"] = max_output_tokens

        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=120)
        if resp.status_code != 200:
            _mark_error(self, resp.status_code, resp.text)
            raise RuntimeError(f"Gemini error: {resp.status_code}, {resp.text}")

        j = resp.json()
        try:
            out = j["candidates"][0]["content"]["parts"][0]["text"]
            _mark_ok(self)
            return out
        except Exception as e:
            _mark_error(self, 500, f"Gemini parse error: {e}; raw={j}")
            raise RuntimeError(f"Gemini 响应解析失败：{e}，原始返回：{j}")

class GrokProvider:
    """
    如果你当前没有 Grok 的实现/Key，这个 provider 也不会影响主流程：
    - 当 provider chain 里不含 grok，或 grok 没 key，会自动跳过
    - 这里保留接口占位，避免破坏你现有 ModelHub 结构
    """
    def __init__(self):
        self.base = (os.getenv("GROK_BASE_URL") or "").strip()

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "grok-2-latest",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> str:
        key = (api_key or os.getenv("GROK_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("缺少 GROK_API_KEY / GROK_API_KEYS")
        if not self.base:
            raise RuntimeError("缺少 GROK_BASE_URL（尚未配置）")

        # 你可按你自己的 Grok API 结构实现，这里先占位抛错，避免误用
        raise RuntimeError("GrokProvider 尚未实现具体调用，请先配置实现或从 provider chain 移除 grok。")


# =====================================
#   Provider Registry（统一模型入口）
# =====================================

class ModelHub:
    def __init__(self):
        self.groq = GroqProvider()
        self.gemini = GeminiProvider()
        self.grok = GrokProvider()

        default_cd = int((os.getenv("GEO_KEY_COOLDOWN_DEFAULT_SEC") or "600").strip() or 600)
        shuffle = (os.getenv("GEO_KEYPOOL_SHUFFLE") or "0").strip() == "1"

        groq_keys = _split_keys_from_env("GROQ_API_KEYS", "GROQ_API_KEY")
        gemini_keys = _split_keys_from_env("GEMINI_API_KEYS", "GEMINI_API_KEY")
        if not gemini_keys:
            # 兼容 GOOGLE_API_KEY（单 key）
            g = (os.getenv("GOOGLE_API_KEY") or "").strip()
            if g:
                gemini_keys = [g]

        grok_keys = _split_keys_from_env("GROK_API_KEYS", "GROK_API_KEY")

        # KeyPool：如果没有 keys，就不创建（run 时会跳过该 provider）
        self._pools: Dict[str, KeyPool] = {}
        if groq_keys:
            self._pools["groq"] = KeyPool("groq", groq_keys, default_cooldown_sec=default_cd, shuffle=shuffle)
        if gemini_keys:
            self._pools["gemini"] = KeyPool("gemini", gemini_keys, default_cooldown_sec=default_cd, shuffle=shuffle)
        if grok_keys:
            self._pools["grok"] = KeyPool("grok", grok_keys, default_cooldown_sec=default_cd, shuffle=shuffle)

        chain_env = (os.getenv("GEO_PROVIDER_CHAIN") or "groq,gemini").strip()
        self._chain = [x.strip().lower() for x in chain_env.split(",") if x and x.strip()]

    def health(self) -> Dict[str, Any]:
        """
        可选：用于你快速确认“可用 key/冷却状态”
        你可以在 FastAPI 里加一个 /api/health/providers 调这个方法
        """
        out = {}
        for p in ("groq", "gemini", "grok"):
            pool = self._pools.get(p)
            out[p] = pool.snapshot() if pool else {"name": p, "keys_total": 0, "keys_active": 0, "cooldowns_sec": {}}
        out["provider_chain"] = list(self._chain)
        return out
    
    def health_runtime(self) -> Dict[str, Any]:
        """
        runtime health：最近一次成功/失败摘要（不泄露 key）
        """
        def _view(p: Any, name: str) -> Dict[str, Any]:
            _ensure_provider_state_fields(p)
            return {
                "provider": name,
                "key_count": getattr(p, "key_count", None),
                "active_key_index": getattr(p, "active_key_index", None),
                "last_ok_at": getattr(p, "last_ok_at", None),
                "last_error_at": getattr(p, "last_error_at", None),
                "last_status_code": getattr(p, "last_status_code", None),
                "last_retry_after_s": getattr(p, "last_retry_after_s", None),
                "last_error": getattr(p, "last_error", None),
            }

        return {
            "provider_chain": list(self._chain),
            "providers": {
                "groq": _view(self.groq, "groq"),
                "gemini": _view(self.gemini, "gemini"),
                "grok": _view(self.grok, "grok"),
            }
        }


    def _try_provider_once(self, provider: str, prompt: str, **kw) -> str:
        """
        对某个 provider 执行“一次尝试”：会从 keypool 取 1 把 key 来调用。
        失败由上层判断是否 cooldown / 是否换 key / 是否 fallback provider。
        """
        p = (provider or "").lower().strip()
        pool = self._pools.get(p)
        if not pool:
            raise RuntimeError(f"No keypool for provider={p}")

        api_key = pool.pick()

        # 统一把 prompt 塞进去：
        if p == "groq":
            return self.groq.chat(
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                **kw,
            )
        if p == "gemini":
            return self.gemini.chat(
                text=prompt,
                api_key=api_key,
                **kw,
            )
        if p == "grok":
            return self.grok.chat(
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                **kw,
            )

        raise RuntimeError(f"Unknown provider: {provider}")

    def _run_with_key_rotation(self, provider: str, prompt: str, **kw) -> str:
        """
        在同一个 provider 内部做 key 轮询：
        - 最多尝试 keys_total 次
        - 遇到 429：解析等待时间 -> cooldown 当前 key -> 继续尝试下一把 key
        - 遇到 transient：短 cooldown 当前 key -> 继续尝试
        - 其他错误：也会尝试换 key（避免某把 key 被封/失效）
        """
        p = (provider or "").lower().strip()
        pool = self._pools.get(p)
        if not pool:
            raise RuntimeError(f"No keypool for provider={p}")

        last_err: Optional[Exception] = None
        attempts = max(1, len(pool.keys))

        for _ in range(attempts):
            api_key, picked_idx = pool.pick_with_index()

            # ✅ 写回 provider 的 keypool 状态（不泄露 key）
            kcnt = len(pool.keys) or None
            if p == "groq":
                self.groq.key_count = kcnt
                self.groq.active_key_index = picked_idx
            elif p == "gemini":
                self.gemini.key_count = kcnt
                self.gemini.active_key_index = picked_idx
            elif p == "grok":
                self.grok.key_count = kcnt
                self.grok.active_key_index = picked_idx

                
            try:
                if p == "groq":
                    return self.groq.chat(
                        messages=[{"role": "user", "content": prompt}],
                        api_key=api_key,
                        **kw,
                    )
                if p == "gemini":
                    return self.gemini.chat(
                        text=prompt,
                        api_key=api_key,
                        **kw,
                    )
                if p == "grok":
                    return self.grok.chat(
                        messages=[{"role": "user", "content": prompt}],
                        api_key=api_key,
                        **kw,
                    )
                raise RuntimeError(f"Unknown provider: {provider}")

            except Exception as e:
                last_err = e
                msg = str(e) or ""

                # 429：按 Groq 的提示精确 cooldown；Gemini 429 也走默认 cooldown
                if _is_rate_limit_error(msg):
                    wait_sec = _parse_groq_retry_after_seconds(msg) or pool.default_cooldown_sec
                    pool.mark_cooldown(api_key, seconds=wait_sec)
                    continue

                # transient：短冷却，避免马上重复撞同一 key
                if _is_retryable_transient(msg):
                    pool.mark_cooldown(api_key, seconds=min(60, pool.default_cooldown_sec))
                    continue

                # 其他错误（401/403/解析失败/Key失效等）：也先把 key 冷却一段时间并换 key 再试
                pool.mark_bad(api_key, seconds=min(300, pool.default_cooldown_sec))
                continue

        # key 都没跑通：抛出最后一个错误（让上层做 provider fallback）
        raise RuntimeError(f"{p} exhausted all keys. last_error={last_err}")

    def run(self, provider: str, prompt: str, **kw) -> str:
        """
        provider: "groq" | "gemini" | "grok"
        行为：
        - 优先使用 provider 参数指定的 provider
        - 若该 provider 在 keypool 内无法成功（429/失效/超时等），则按 GEO_PROVIDER_CHAIN fallback 到其他 provider
        """
        want = (provider or "").lower().strip()

        # 构造候选链：先尝试 want，再按 chain 补齐（去重）
        candidates: List[str] = []
        if want:
            candidates.append(want)
        for p in self._chain:
            if p not in candidates:
                candidates.append(p)

        last_err: Optional[Exception] = None

        for p in candidates:
            # 没有 keypool 的 provider 直接跳过（例如你未配置 grok）
            if p not in self._pools:
                continue

            try:
                return self._run_with_key_rotation(p, prompt, **kw)
            except Exception as e:
                last_err = e
                #  reminder：这里不直接 raise，继续 fallback
                continue

        # 全部 provider 都失败
        raise RuntimeError(f"All providers failed. want={want}, last_error={last_err}")

# =========================
#  Backward-compatible exports
#  - keep geo_core.py unchanged
# =========================

def norm_provider(model_ui: str) -> str:
    """
    兼容 geo_core.py 的旧接口：把 UI 传入的字符串归一化为 provider key。
    你项目里 model_ui 可能传 "Groq" / "groq" / "Gemini" / "deepseek" 等。
    """
    s = (model_ui or "").lower().strip()
    # 常见 UI 值兼容
    if s in ("groq", "groqprovider"):
        return "groq"
    if s in ("gemini", "google", "googleai", "google_ai"):
        return "gemini"
    if s in ("grok", "xai"):
        return "grok"
    if s in ("deepseek",):
        return "deepseek"
    if s in ("qwen", "dashscope"):
        return "qwen"
    # 默认：保持你项目过去的行为（更安全）
    return s or "groq"


# 兼容 geo_core.py：DEFAULT_MODELS 仍然存在
# 注意：这里的 model 名称只是“默认值”，最终是否使用由 call_model/hub/provider 决定
DEFAULT_MODELS: Dict[str, str] = {
    "groq": os.getenv("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile").strip(),
    "gemini": os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash").strip(),
    "grok": os.getenv("GROK_DEFAULT_MODEL", "grok-2-latest").strip(),
    # 旧国内 provider 也保留默认值（如果你后面还会走 inference_engine 的 old path）
    "deepseek": os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat").strip(),
    "qwen": os.getenv("QWEN_DEFAULT_MODEL", "qwen-turbo").strip(),
}


# 兼容旧用法：有些模块会直接 `from providers_groq_gemini import hub`
# 这样不需要你再改 pipeline/inference_engine.py
hub = ModelHub()