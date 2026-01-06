# api_brand.py
import os
import json
import time
import hmac
import hashlib
import requests
import stripe
import re

from app.domain.auth import service as auth_service
from app.domain.quota import service as quota_service
from app.infra.supabase import client as supabase_client
from app.services.payments import billing_core
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Literal, Tuple, cast
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv
from urllib.parse import quote

from cache_ttl import TTLCache
import logging

logger = logging.getLogger("uvicorn.error")
load_dotenv()

SUPABASE_URL = supabase_client.SUPABASE_URL
SUPABASE_ANON_KEY = supabase_client.SUPABASE_ANON_KEY  # ✅ 后端也需要，用于 /auth/v1/user
SUPABASE_SERVICE_ROLE_KEY = supabase_client.SUPABASE_SERVICE_ROLE_KEY

_require_env = supabase_client._require_env

# ✅ 仅在真正需要解析登录态时才强制检查 env，避免你本地开发某些路由不走鉴权时直接炸掉
def _ensure_supabase_env_for_auth():
    return supabase_client.ensure_supabase_env_for_auth()


def _ensure_supabase_env_for_db():
    return supabase_client.ensure_supabase_env_for_db()
    
    
# =========================
# Quota / Billing (Path 1)
# - monthly token quota for tier
# - addon_unlimited: bypass monthly quota and never resets
# - atomic consume via Supabase RPC: public.consume_tokens(...)
# =========================

# 月度 token 限额（可用 env 覆盖）
TIER_MONTHLY_TOKENS_FREE = quota_service.TIER_MONTHLY_TOKENS_FREE
TIER_MONTHLY_TOKENS_ALPHA_BASE = quota_service.TIER_MONTHLY_TOKENS_ALPHA_BASE
TIER_MONTHLY_TOKENS_ALPHA_PRO = quota_service.TIER_MONTHLY_TOKENS_ALPHA_PRO

# 是否对 COT 也计费（你可以先 false，只对 rewrite/score 计费）
CHARGE_COT = os.getenv("CHARGE_COT", "0").strip() == "1"

def _tier_monthly_limit(tier: str) -> int:
    return quota_service.tier_monthly_limit(tier)

def _yyyymm_utc_now() -> str:
    return quota_service.yyyymm_utc_now()

def _supabase_admin_headers():
    # 与 _supabase_rpc_consume_tokens 保持一致（service role）
    return supabase_client.supabase_admin_headers()

def _supabase_get_quota_row(user_id: str, yyyymm: str):
    return quota_service.supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)

def _utc_now_iso():
    return quota_service.utc_now_iso()

def _supabase_upsert_quota_row(
    *,
    user_id: str,
    tier: str,
    yyyymm: str,
    tokens_limit: int,
    reset_used: bool = False, # 支付成功建议设为 True，或者按需保持
):
    return quota_service.supabase_upsert_quota_row(
        user_id=user_id,
        tier=tier,
        yyyymm=yyyymm,
        tokens_limit=tokens_limit,
        reset_used=reset_used,
    )
    
def _supabase_patch_quota_row(
    *,
    user_id: str,
    yyyymm: str,
    tier: str,
    tokens_limit: int,
    reset_used: bool = False,
):
    """
    对 quota_monthly 做定向 PATCH（按 user_id + yyyymm）
    作用：
    - upsert 在某些 RLS / content-type / prefer 情况下可能“看似成功但未更新”
    - PATCH 能更明确地覆盖 tier / tokens_limit（必要时 tokens_used=0）
    """
    return quota_service.supabase_patch_quota_row(
        user_id=user_id,
        yyyymm=yyyymm,
        tier=tier,
        tokens_limit=tokens_limit,
        reset_used=reset_used,
    )


def _quota_sync_after_payment(
    *,
    user_id: str,
    tier: str,
    yyyymm: str | None = None,
    reset_used: bool = True,
):
    """
    支付成功后的“强一致性同步”：
    - 先 upsert（镜像写入）
    - 再读回 verify
    - 若不一致，再 PATCH 强制覆盖
    - 再读回 verify，仍不一致则 raise（让你在日志里看到明确错误）
    """
    if not user_id:
        return None

    return quota_service.quota_sync_after_payment(
        user_id=user_id, tier=tier, yyyymm=yyyymm, reset_used=reset_used
    )


def _quota_ensure_row(user_id: str, tier: str, yyyymm: str):
    """
    确保 quota_monthly 本月行存在，且 tokens_limit 与 tier 对齐。
    v0 策略：只做 upsert 初始化/修正 limit，不强行重置 tokens_used（避免误伤正在使用的用户）。
    你如果希望“支付成功时立刻重置本月额度”，可在这里把 tokens_used 置 0（见注释）。
    """
    return quota_service.quota_ensure_row(user_id=user_id, tier=tier, yyyymm=yyyymm)


def _supabase_rpc_consume_tokens(
    user_id: str,
    tier: str,
    kind: str,
    tokens_in: int,
    tokens_out: int,
    tokens_total: int,
) -> dict:
    """
    Call Supabase PostgREST RPC: /rest/v1/rpc/consume_tokens
    Atomic monthly quota consume with addon_unlimited bypass.
    """
    return quota_service.supabase_rpc_consume_tokens(
        user_id=user_id,
        tier=tier,
        kind=kind,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_total=tokens_total,
    )

def _quota_block_response(consumed: dict) -> JSONResponse:
    """
    Standardized quota exceeded response.
    """
    limit_ = consumed.get("tokens_limit")
    used_ = consumed.get("tokens_used")
    remaining_ = consumed.get("tokens_remaining")

    return JSONResponse(
        status_code=402,
        content={
            "ok": False,
            "error": "Monthly token quota exceeded.",
            "code": "MONTHLY_QUOTA_EXCEEDED",
            "quota": {
                "yyyymm": consumed.get("yyyymm"),
                "tokens_limit": limit_,
                "tokens_used": used_,
                "tokens_remaining": remaining_,
            },
        },
    )

    
# 简单内存缓存：减少每次请求都打 Supabase
# cache key: user_id -> (tier, expires_at)
_TIER_CACHE: dict[str, tuple[str, float]] = {}
_TIER_CACHE_TTL_SEC = 60.0


def _get_bearer_token(request: Request) -> str | None:
    return auth_service.get_bearer_token(request)

def _supabase_get_user_id(access_token: str) -> str | None:
    return auth_service.supabase_get_user_id(access_token)

def _supabase_auth_get_user(access_token: str) -> dict | None:
    return auth_service.supabase_auth_get_user(access_token)


def _supabase_get_profile_tier(user_id: str) -> str:
    return auth_service.supabase_get_profile_tier(user_id)

def resolve_user_and_tier(request: Request) -> dict:
    return auth_service.resolve_user_and_tier(request)


def require_authed_user(request: Request) -> dict:
    return auth_service.require_authed_user(request)


# =========================
# GEO Cache Debug Logging
# =========================
GEO_CACHE_DEBUG = os.getenv("GEO_CACHE_DEBUG", "0").strip() == "1"

def _cache_dbg(tag: str, hit: bool, cache_key: str, extra: dict | None = None):
    """
    仅在 GEO_CACHE_DEBUG=1 时打印 cache hit/miss
    """
    if not GEO_CACHE_DEBUG:
        return
    try:
        ck = (cache_key or "")[:24] + "..." if cache_key else ""
        payload = {"tag": tag, "hit": bool(hit), "key": ck}
        if extra:
            payload.update(extra)
        print("[GEO-CACHE]", json.dumps(payload, ensure_ascii=False))
    except Exception:
        # debug 不影响主流程
        print(f"[GEO-CACHE] tag={tag} hit={hit}")


# =========================
# Handler-level TTL Cache
# =========================
# 说明：
# - 进程内存缓存：uvicorn --reload / 多进程会导致缓存不共享，这是 1A 的预期行为
# - 通过 VERSION + TTL 双保险：改 prompt 时 bump 版本即可立即失效
GEO_PROMPTS_VERSION = os.getenv("GEO_PROMPTS_VERSION", "v1").strip()
GEO_SCORE_VERSION = os.getenv("GEO_SCORE_VERSION", "v1").strip()
GEO_COT_VERSION = os.getenv("GEO_COT_VERSION", "v1").strip()

# TTL 建议（调试期）
TTL_SCORE_SEC = int(os.getenv("GEO_CACHE_TTL_SCORE_SEC", "3600"))      # 60min
TTL_REWRITE_SEC = int(os.getenv("GEO_CACHE_TTL_REWRITE_SEC", "21600")) # 6h
TTL_COT_SEC = int(os.getenv("GEO_CACHE_TTL_COT_SEC", "43200"))         # 12h

# 容量上限（防止内存增长过快）
_SCORE_CACHE = TTLCache(max_items=int(os.getenv("GEO_CACHE_MAX_SCORE", "256")))
_REWRITE_CACHE = TTLCache(max_items=int(os.getenv("GEO_CACHE_MAX_REWRITE", "256")))
_COT_CACHE = TTLCache(max_items=int(os.getenv("GEO_CACHE_MAX_COT", "256")))

def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

def _stable_cache_key(prefix: str, payload: dict) -> str:
    """
    - payload 用 json dumps 固化字段顺序，避免 dict 顺序造成 key 漂移
    - 大文本用 hash，避免 key 过长
    """
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}:{_sha256_text(raw)}"


def parse_groq_retry_after_seconds(msg: str) -> Optional[int]:
    """
    从 Groq 429 message 中解析 'Please try again in10m11.712s' 这种片段
    返回整数秒（向上取整）。
    """
    if not msg:
        return None
    m = re.search(r"try again in\s*([0-9]+)m([0-9]+(?:\.[0-9]+)?)s", msg, re.IGNORECASE)
    if not m:
        return None
    minutes = int(m.group(1))
    seconds = float(m.group(2))
    return int(minutes * 60 + (seconds if seconds.is_integer() else int(seconds) + 1))

def _exc_status_code(e: Exception) -> Optional[int]:
    """
    尽量从不同异常对象中提取 status_code
    兼容 httpx / requests / 第三方 SDK 的常见属性结构
    """
    # 1) 直接属性：e.status_code / e.status
    for attr in ("status_code", "status"):
        if hasattr(e, attr):
            try:
                v = getattr(e, attr)
                if isinstance(v, int):
                    return v
            except Exception:
                pass

    # 2) httpx.HTTPStatusError: e.response.status_code
    if hasattr(e, "response"):
        resp = getattr(e, "response", None)
        if resp is not None and hasattr(resp, "status_code"):
            try:
                v = resp.status_code
                if isinstance(v, int):
                    return v
            except Exception:
                pass

    return None


def _is_groq_429(e: Exception) -> bool:
    """
    识别 Groq 429：优先看 status_code，其次用 message 兜底匹配
    """
    status = _exc_status_code(e)
    if status == 429:
        return True

    msg = (str(e) or "").lower()
    if "rate limit" in msg and ("429" in msg or "rate" in msg or "tpd" in msg):
        return True
    if msg.startswith("groq error: 429") or " code\": rate_limit_exceeded" in msg or "rate limit exceeded" in msg:
        return True

    return False


def _extract_groq_retry_after_sec(e: Exception) -> Optional[int]:
    """
    尽量获取建议等待秒数：
    - 优先从 response headers 的 Retry-After
    - 否则从报错 message 里的 "try again in10m11.712s" 解析
    """
    # 1) headers retry-after
    try:
        if hasattr(e, "response"):
            resp = getattr(e, "response", None)
            if resp is not None and hasattr(resp, "headers"):
                headers = resp.headers
                ra = None
                try:
                    ra = headers.get("retry-after") or headers.get("Retry-After")
                except Exception:
                    # 某些 headers 不是 dict
                    ra = None
                if ra:
                    try:
                        return int(float(ra))
                    except Exception:
                        pass
    except Exception:
        pass

    # 2) fallback: parse from message
    return parse_groq_retry_after_seconds(str(e) or "")


#  GEO-Rewrite 入口（Alpha 强化版）
#  - 权限：仅信任 Authorization: Bearer <supabase access_token> → profiles.tier
#  - Free：后端硬限制（token / 每日次数 / 总次数）
#  - Alpha：通过 profiles.tier 自动放开

FREE_TOKEN_LIMIT = 1000     # Free 单次约 1000 tokens
FREE_DAILY_LIMIT = 2        # Free 每日 2 次
FREE_TOTAL_LIMIT = 10       # Free 总计 10 次试用

# 进程级轻量计数器（Alpha 阶段先这样，未来可替换为 Redis/DB）
_rewrite_usage = {
    "total": 0,
    "today": 0,
    "date": None,
}

def _reset_daily_counter():
    """按 UTC 日期重置每日计数"""
    from datetime import datetime
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if _rewrite_usage.get("date") != today:
        _rewrite_usage["date"] = today
        _rewrite_usage["today"] = 0


def _estimate_tokens(text: str) -> int:
    """
    粗略 token 估算，尽量与前端 estimateTokens 对齐：
    - ASCII 字符按“单词”计数
    - 非 ASCII 字符按“字符”计数
    """
    if not text:
        return 0

    # 模拟前端：text.replace(/[^\x00-\x7F]/g, ' ')
    ascii_only_chars = [ch if ord(ch) < 128 else " " for ch in text]
    ascii_text = "".join(ascii_only_chars)

    ascii_words = len(
        [w for w in ascii_text.strip().split() if w]
    )
    non_ascii = sum(1 for ch in text if ord(ch) >= 128)

    return ascii_words + non_ascii

def _build_anchor_context(
    title: str | None,
    question: str | None,
    body: str
) -> str:
    """
    构造带锚点的 Rewrite 输入上下文（最小侵入）
    """
    t = (title or "").strip()
    q = (question or "").strip()
    b = (body or "").strip()

    parts = []

    if t:
        parts.append("[Article Title]\n" + t)

    if q:
        parts.append("[User Question]\n" + q)

    parts.append("[Article Body]\n" + b)

    return "\n\n".join(parts).strip()


def _strip_anchor_echo(text: str) -> str:
    """
    防御性处理：
    若模型把 [Article Title] / [User Question] 一并输出，
    尽量只保留 Article Body 对应的正文部分
    """
    if not text:
        return text

    marker = "[Article Body]"
    idx = text.rfind(marker)
    if idx >= 0:
        return text[idx + len(marker):].lstrip("\n ").strip()

    return text.strip()


# =========================
#  Stripe Billing (Alpha)
#  - Create Checkout Session
#  - Webhook: auto-upgrade profiles.tier
# =========================

# Stripe init
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()

STRIPE_PRICE_ALPHA_BASE = os.getenv("STRIPE_PRICE_ALPHA_BASE", "").strip()
STRIPE_PRICE_ALPHA_PRO = os.getenv("STRIPE_PRICE_ALPHA_PRO", "").strip()

# Frontend origin (optional)
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").strip()

# Configure Stripe SDK
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


def _must_env(name: str, value: str):
    if not value:
        raise RuntimeError(f"Missing required env: {name}")


def _supabase_patch_profile_tier(*, user_id: str, tier: str):
    """
    强制更新 public.profiles.tier（按 profiles.id 命中）
    重要：profiles 的主键字段是 id，不是 user_id
    """
    user_id = (user_id or "").strip()  # ✅ 关键：去掉空格/换行
    if not user_id:
        return None

    tier = (tier or "free").strip().lower()

    url = _sb_rest_url("profiles")  # e.g. https://xxx.supabase.co/rest/v1/profiles
    # ✅ 关键：用 id 过滤；并对值做 URL encode（虽然 uuid 一般不需要，但更稳）
    patch_url = f"{url}?id=eq.{quote(user_id, safe='')}"

    payload = {
        "tier": tier,
        "updated_at": _utc_now_iso(),
    }

    # prefer=return=representation 让响应返回更新后的行（便于验收）
    res = _sb_patch_json(patch_url, payload, prefer="return=representation")

    # ✅ 证据化：如果返回空/None，说明很可能“0 行更新”或被 RLS 拦
    if not res:
        print("[profiles][patch] empty result (0 row updated?)", {"patch_url": patch_url, "payload": payload})

    return res


def _profile_sync_tier_after_payment(*, user_id: str, tier: str):
    """
    支付成功后的 profiles.tier 强一致性更新：
    - PATCH 命中 profiles.id
    - 再读回验证
    - 不一致则 raise（至少在 webhook 日志里可见）
    """
    tier = (tier or "free").strip().lower()

    try:
        patched = _supabase_patch_profile_tier(user_id=user_id, tier=tier)
    except Exception as e:
        raise RuntimeError(f"[profile_sync] patch failed: {e}")

    # verify
    try:
        current = _supabase_get_profile_tier(user_id)
    except Exception as e:
        raise RuntimeError(f"[profile_sync] verify read failed: {e}")

    if (current or "").strip().lower() == tier:
        return {"tier": current, "patched": patched}

    raise RuntimeError(f"[profile_sync] verify failed: expect {tier}, got {current}, patched={patched}")


def _supabase_upsert_profile_row(user_id: str, email: str | None = None, tier: str | None = None) -> dict | None:
    return auth_service.supabase_upsert_profile_row(user_id=user_id, email=email, tier=tier)


def _tier_from_price_id(price_id: str) -> str:
    if price_id == STRIPE_PRICE_ALPHA_BASE:
        return "alpha_base"
    if price_id == STRIPE_PRICE_ALPHA_PRO:
        return "alpha_pro"
    # default safe fallback
    return "free"

# =========================
# Supabase Billing Tables Helpers (NEW)
# - billing_webhook_events: idempotency for webhook
# - billing_customers: user_id <-> stripe_customer_id
# - billing_subscriptions: subscription mapping + status/price/period_end
# =========================

BILLING_TABLE_WEBHOOK_EVENTS = "billing_webhook_events"
BILLING_TABLE_CUSTOMERS = "billing_customers"
BILLING_TABLE_SUBSCRIPTIONS = "billing_subscriptions"

def _sb_rest_url(table: str) -> str:
    _ensure_supabase_env_for_db()
    base = SUPABASE_URL.rstrip("/")
    return f"{base}/rest/v1/{table}"

def _sb_rpc_url(fn: str) -> str:
    _ensure_supabase_env_for_db()
    base = SUPABASE_URL.rstrip("/")
    return f"{base}/rest/v1/rpc/{fn}"

def _sb_get_json(url: str, params: dict) -> list:
    r = requests.get(url, headers=_supabase_admin_headers(), params=params, timeout=15)

    # 失败：把关键信息抛出来（避免 r.text 为空时你看不到原因）
    if r.status_code >= 300:
        ctype = (r.headers.get("content-type") or "").lower()
        body = (r.text or "")
        if len(body) > 800:
            body = body[:800] + "...(truncated)"
        raise RuntimeError(
            f"Supabase GET failed: {r.status_code} {r.reason} "
            f"(content-type={ctype}) body={body}"
        )

    # ✅ 成功但无 body：直接返回空列表（避免 JSONDecodeError）
    if r.status_code == 204 or not (r.content and r.content.strip()):
        return []

    # ✅ 非 JSON：也不要 r.json()，返回空列表并保留可诊断信息（不破坏调用方预期）
    ctype = (r.headers.get("content-type") or "").lower()
    if "application/json" not in ctype and not ctype.endswith("+json"):
        # 如果你希望更“硬”，可以改成 raise；但 Step1 验收阶段建议先不中断主流程
        return []

    # ✅ 正常 JSON
    try:
        return r.json() or []
    except Exception as e:
        body = (r.text or "")
        if len(body) > 800:
            body = body[:800] + "...(truncated)"
        raise RuntimeError(
            f"Supabase GET JSON decode failed: {e} "
            f"(status={r.status_code}, content-type={ctype}) body={body}"
        )


def _sb_post_json(url: str, payload: Any, prefer: str = "return=representation") -> list:
    headers = _supabase_admin_headers()
    headers["Prefer"] = prefer

    r = requests.post(url, headers=headers, json=payload, timeout=15)

    # 失败：同样增强诊断信息
    if r.status_code >= 300:
        ctype = (r.headers.get("content-type") or "").lower()
        body = (r.text or "")
        if len(body) > 800:
            body = body[:800] + "...(truncated)"
        raise RuntimeError(
            f"Supabase POST failed: {r.status_code} {r.reason} "
            f"(content-type={ctype}) body={body}"
        )

    # ✅ 成功但无 body：直接返回空列表（这是你当前 JSONDecodeError 的最常见来源）
    if r.status_code == 204 or not (r.content and r.content.strip()):
        return []

    # ✅ 非 JSON：避免 r.json()，返回空列表（保持函数返回类型一致）
    ctype = (r.headers.get("content-type") or "").lower()
    if "application/json" not in ctype and not ctype.endswith("+json"):
        return []

    # ✅ 正常 JSON
    try:
        return r.json() or []
    except Exception as e:
        body = (r.text or "")
        if len(body) > 800:
            body = body[:800] + "...(truncated)"
        raise RuntimeError(
            f"Supabase POST JSON decode failed: {e} "
            f"(status={r.status_code}, content-type={ctype}) body={body}"
        )

def _sb_patch_json(url: str, payload: dict, prefer: str | None = None):
    """
    PostgREST PATCH helper (service role)
    - 使用 service role headers，确保 webhook 有写权限
    - Prefer 可选：return=representation / return=minimal
    - 非 2xx 或返回空，打印 status_code + body，避免“静默失败”
    """
    headers = _supabase_admin_headers()  # ✅ 必须是 service role
    if prefer:
        headers = {**headers, "Prefer": prefer}

    r = requests.patch(url, headers=headers, json=payload, timeout=15)

    # ✅ 证据化：无论成功与否，开发期至少在异常时打印返回体
    if r.status_code < 200 or r.status_code >= 300:
        print("[sb][patch] FAILED", {"url": url, "status": r.status_code, "body": r.text})
        raise RuntimeError(f"Supabase PATCH failed: {r.status_code} {r.text}")

    # return=representation 时一般是 list；return=minimal 可能是空
    try:
        if r.text and r.text.strip():
            return r.json()
    except Exception:
        # JSON 解析失败也要打印
        print("[sb][patch] JSON decode failed", {"url": url, "status": r.status_code, "body": r.text})
        raise
    return None


def _supabase_upsert_profile_tier(*, user_id: str, new_tier: str):
    """
    profiles 镜像层：用 UPSERT 代替 PATCH
    - 解决 profiles 行不存在时 patch 失败的问题
    - 幂等：on_conflict=id
    """
    if not user_id:
        return None
    new_tier = (new_tier or "free").strip().lower()

    url = _sb_rest_url("profiles") + "?on_conflict=id"
    payload = {"id": user_id, "tier": new_tier}
    return _sb_post_json(
        url,
        payload,
        prefer="resolution=merge-duplicates,return=representation",
    )


# ---------- Webhook idempotency ----------
def _sb_webhook_event_exists(event_id: str) -> bool:
    """
    Check if billing_webhook_events has event_id.
    Non-blocking: any unexpected error -> treat as not exists (False).
    """
    if not event_id:
        return False
    try:
        url = _sb_rest_url(BILLING_TABLE_WEBHOOK_EVENTS)  # e.g. "billing_webhook_events"
        rows = _sb_get_json(
            url,
            params={
                "select": "event_id",
                "event_id": f"eq.{event_id}",
                "limit": "1",
            },
        )
        return bool(rows)
    except Exception as e:
        # Don't block webhook processing if dedup check fails
        print("[billing] webhook dedup exists check failed:", str(e))
        return False


def _sb_webhook_event_insert(
    event_id: str,
    event_type: str,
    user_id: str | None = None,
    session_id: str | None = None,
):
    """
    Insert into billing_webhook_events for dedup.
    Key fixes:
    - Prefer return=representation to avoid empty body -> JSON parse errors in _sb_post_json
    - Swallow duplicate insert (409 / unique violation) as OK to support replay/concurrency
    """
    if not event_id:
        return

    url = _sb_rest_url(BILLING_TABLE_WEBHOOK_EVENTS)

    payload = [{
        "event_id": event_id,
        "event_type": (event_type or ""),
        "user_id": user_id,
        "session_id": session_id,
    }]

    try:
        # ✅ Critical: force Supabase to return JSON so _sb_post_json().json() won't explode
        _sb_post_json(url, payload, prefer="return=representation")
        return
    except Exception as e:
        msg = (str(e) or "").lower()

        # ✅ Duplicate / already inserted: treat as success (webhook replay or concurrency)
        # Supabase/PostgREST commonly returns 409 Conflict
        if ("409" in msg) or ("duplicate" in msg) or ("unique" in msg) or ("already exists" in msg):
            return

        # Other errors should be visible for investigation
        raise


# ---------- Customer mapping ----------
def _sb_get_customer_id_by_user(user_id: str) -> str | None:
    if not user_id:
        return None
    url = _sb_rest_url(BILLING_TABLE_CUSTOMERS)
    rows = _sb_get_json(url, params={"select": "stripe_customer_id", "user_id": f"eq.{user_id}", "limit": "1"})
    if not rows:
        return None
    return (rows[0].get("stripe_customer_id") or "").strip() or None

def _sb_upsert_customer(user_id: str, stripe_customer_id: str, email: str | None = None):
    if not user_id or not stripe_customer_id:
        return
    url = _sb_rest_url(BILLING_TABLE_CUSTOMERS)
    payload = [{
        "user_id": user_id,
        "stripe_customer_id": stripe_customer_id,
        "email": (email or None),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }]
    headers = _supabase_admin_headers()
    headers["Prefer"] = "resolution=merge-duplicates,return=representation"
    r = requests.post(
        url,
        headers=headers,
        params={"on_conflict": "user_id"},
        json=payload,
        timeout=15,
    )
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase upsert customer failed: {r.status_code} {r.text}")
    return (r.json() or [None])[0]

# ---------- Subscription mapping ----------
def _sb_get_subscriptions_by_user(user_id: str, limit: int = 20) -> list[dict]:
    if not user_id:
        return []
    url = _sb_rest_url(BILLING_TABLE_SUBSCRIPTIONS)
    # billing_subscriptions has primary key stripe_subscription_id
    rows = _sb_get_json(
        url,
        params={
            "select": "stripe_subscription_id,user_id,stripe_customer_id,price_id,status,current_period_end,created_at,updated_at",
            "user_id": f"eq.{user_id}",
            "order": "created_at.desc",
            "limit": str(limit),
        },
    )
    return rows or []

def _sb_user_has_effective_subscription(user_id: str) -> tuple[bool, dict | None]:
    """
    v0/v1: define 'effective' subscription as any status that can grant/keep access
    - active, trialing are definitely effective
    - past_due / unpaid: still a 'do not create new subscription' guard (avoid double-pay)
    """
    subs = _sb_get_subscriptions_by_user(user_id=user_id, limit=50)
    if not subs:
        return (False, None)

    effective_status = {"active", "trialing", "past_due", "unpaid"}
    for s in subs:
        st = (s.get("status") or "").strip().lower()
        if st in effective_status:
            return (True, s)
    return (False, None)

def _sb_upsert_subscription(
    stripe_subscription_id: str,
    user_id: str,
    stripe_customer_id: str,
    price_id: str | None = None,
    status: str | None = None,
    current_period_end_iso: str | None = None,
):
    if not stripe_subscription_id or not user_id or not stripe_customer_id:
        return

    url = _sb_rest_url(BILLING_TABLE_SUBSCRIPTIONS)

    payload = [{
        "stripe_subscription_id": stripe_subscription_id,
        "user_id": user_id,
        "stripe_customer_id": stripe_customer_id,
        "price_id": price_id,
        "status": status,
        "current_period_end": current_period_end_iso,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }]

    headers = _supabase_admin_headers()
    headers["Prefer"] = "resolution=merge-duplicates,return=representation"
    r = requests.post(
        url,
        headers=headers,
        params={"on_conflict": "stripe_subscription_id"},
        json=payload,
        timeout=15,
    )
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase upsert subscription failed: {r.status_code} {r.text}")
    return (r.json() or [None])[0]

def _stripe_get_subscription_primary_price_id(sub: dict) -> str:
    """
    Returns first subscription item's price id (v0 assumes single plan item).
    """
    try:
        items = (sub.get("items") or {}).get("data") or []
        if not items:
            return ""
        price = (items[0].get("price") or {})
        return (price.get("id") or "").strip()
    except Exception:
        return ""

def _iso_from_unix_ts(ts: int | None) -> str | None:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None
    
def _inv_extract_user_id_and_tier(inv: dict) -> tuple[str | None, str | None]:
    """
    从 invoice 事件对象中提取 supabase_user_id / tier。

    你的证据显示：
    - invoice.metadata 通常为空
    - user_id/tier 出现在：
      1) invoice.lines.data[0].metadata
      2) invoice.parent.subscription_details.metadata

    返回: (user_id, tier)；取不到则 (None, None)
    """
    if not inv or not isinstance(inv, dict):
        return (None, None)

    def _pick(meta: dict) -> tuple[str | None, str | None]:
        if not meta or not isinstance(meta, dict):
            return (None, None)
        uid = (meta.get("supabase_user_id") or meta.get("user_id") or "").strip()
        tier = (meta.get("tier") or "").strip().lower()
        return (uid or None, tier or None)

    # 0) invoice.metadata（兼容未来）
    uid, tier = _pick(inv.get("metadata") or {})
    if uid or tier:
        return (uid, tier)

    # 1) invoice.lines.data[0].metadata
    try:
        lines = ((inv.get("lines") or {}).get("data") or [])
        if isinstance(lines, list) and len(lines) > 0:
            meta1 = lines[0].get("metadata") or {}
            uid, tier = _pick(meta1)
            if uid or tier:
                return (uid, tier)
    except Exception:
        pass

    # 2) invoice.parent.subscription_details.metadata（Stripe 会把订阅 metadata 镜像到这里）
    try:
        parent = inv.get("parent") or {}
        sub_details = parent.get("subscription_details") or {}
        meta2 = sub_details.get("metadata") or {}
        uid, tier = _pick(meta2)
        if uid or tier:
            return (uid, tier)
    except Exception:
        pass

    return (None, None)


# ---------- Billing helpers sourced from billing_core ----------
TIER_MONTHLY_TOKENS_FREE = billing_core.TIER_MONTHLY_TOKENS_FREE
TIER_MONTHLY_TOKENS_ALPHA_BASE = billing_core.TIER_MONTHLY_TOKENS_ALPHA_BASE
TIER_MONTHLY_TOKENS_ALPHA_PRO = billing_core.TIER_MONTHLY_TOKENS_ALPHA_PRO

STRIPE_SECRET_KEY = billing_core.STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET = billing_core.STRIPE_WEBHOOK_SECRET
STRIPE_PRICE_ALPHA_BASE = billing_core.STRIPE_PRICE_ALPHA_BASE
STRIPE_PRICE_ALPHA_PRO = billing_core.STRIPE_PRICE_ALPHA_PRO
FRONTEND_ORIGIN = billing_core.FRONTEND_ORIGIN

BILLING_TABLE_WEBHOOK_EVENTS = billing_core.BILLING_TABLE_WEBHOOK_EVENTS
BILLING_TABLE_CUSTOMERS = billing_core.BILLING_TABLE_CUSTOMERS
BILLING_TABLE_SUBSCRIPTIONS = billing_core.BILLING_TABLE_SUBSCRIPTIONS

_must_env = billing_core._must_env
_tier_monthly_limit = quota_service.tier_monthly_limit
_yyyymm_utc_now = quota_service.yyyymm_utc_now
_supabase_admin_headers = supabase_client.supabase_admin_headers
_supabase_get_quota_row = quota_service.supabase_get_quota_row
_utc_now_iso = quota_service.utc_now_iso
_supabase_upsert_quota_row = quota_service.supabase_upsert_quota_row
_supabase_patch_quota_row = quota_service.supabase_patch_quota_row
_quota_sync_after_payment = quota_service.quota_sync_after_payment
_quota_ensure_row = quota_service.quota_ensure_row

_sb_rest_url = supabase_client.sb_rest_url
_sb_rpc_url = supabase_client.sb_rpc_url
_sb_get_json = supabase_client.sb_get_json
_sb_post_json = supabase_client.sb_post_json
_sb_patch_json = supabase_client.sb_patch_json
_tier_from_price_id = billing_core._tier_from_price_id

_supabase_get_profile_tier = auth_service.supabase_get_profile_tier
_supabase_patch_profile_tier = billing_core._supabase_patch_profile_tier
_profile_sync_tier_after_payment = billing_core._profile_sync_tier_after_payment
_supabase_upsert_profile_row = auth_service.supabase_upsert_profile_row

_sb_webhook_event_exists = billing_core._sb_webhook_event_exists
_sb_webhook_event_insert = billing_core._sb_webhook_event_insert
_sb_get_customer_id_by_user = billing_core._sb_get_customer_id_by_user
_sb_upsert_customer = billing_core._sb_upsert_customer
_sb_get_subscriptions_by_user = billing_core._sb_get_subscriptions_by_user
_sb_user_has_effective_subscription = billing_core._sb_user_has_effective_subscription
_sb_upsert_subscription = billing_core._sb_upsert_subscription
_stripe_get_subscription_primary_price_id = billing_core._stripe_get_subscription_primary_price_id
_iso_from_unix_ts = billing_core._iso_from_unix_ts
_inv_extract_user_id_and_tier = billing_core._inv_extract_user_id_and_tier



