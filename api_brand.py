# api_brand.py
import os
import json
import time
import hmac
import hashlib
import requests
import stripe
import re

from app.services.payments import billing_core
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Literal, Tuple, cast
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv
from urllib.parse import quote

from cache_ttl import TTLCache
import logging

logger = logging.getLogger("uvicorn.error")
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()  # ✅ 后端也需要，用于 /auth/v1/user
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

def _require_env(name: str, value: str):
    if not value:
        raise RuntimeError(f"Missing required env: {name}")

# ✅ 仅在真正需要解析登录态时才强制检查 env，避免你本地开发某些路由不走鉴权时直接炸掉
def _ensure_supabase_env_for_auth():
    _require_env("SUPABASE_URL", SUPABASE_URL)
    _require_env("SUPABASE_ANON_KEY", SUPABASE_ANON_KEY)

def _ensure_supabase_env_for_db():
    _require_env("SUPABASE_URL", SUPABASE_URL)
    _require_env("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY)
    
    
# =========================
# Quota / Billing (Path 1)
# - monthly token quota for tier
# - addon_unlimited: bypass monthly quota and never resets
# - atomic consume via Supabase RPC: public.consume_tokens(...)
# =========================

# 月度 token 限额（可用 env 覆盖）
TIER_MONTHLY_TOKENS_FREE = int(os.getenv("TIER_MONTHLY_TOKENS_FREE", "20000"))
TIER_MONTHLY_TOKENS_ALPHA_BASE = int(os.getenv("TIER_MONTHLY_TOKENS_ALPHA_BASE", "200000"))
TIER_MONTHLY_TOKENS_ALPHA_PRO = int(os.getenv("TIER_MONTHLY_TOKENS_ALPHA_PRO", "500000"))

# 是否对 COT 也计费（你可以先 false，只对 rewrite/score 计费）
CHARGE_COT = os.getenv("CHARGE_COT", "0").strip() == "1"

def _tier_monthly_limit(tier: str) -> int:
    t = (tier or "free").lower().strip()
    if t == "alpha_base":
        return TIER_MONTHLY_TOKENS_ALPHA_BASE
    if t == "alpha_pro":
        return TIER_MONTHLY_TOKENS_ALPHA_PRO
    return TIER_MONTHLY_TOKENS_FREE

def _yyyymm_utc_now() -> str:
    return datetime.utcnow().strftime("%Y%m")

def _supabase_admin_headers():
    # 与 _supabase_rpc_consume_tokens 保持一致（service role）
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    return {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
    }

def _supabase_get_quota_row(user_id: str, yyyymm: str):
    supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    url = f"{supabase_url}/rest/v1/quota_monthly"
    params = {
        "select": "user_id,tier,yyyymm,tokens_limit,tokens_used,updated_at",
        "user_id": f"eq.{user_id}",
        "yyyymm": f"eq.{yyyymm}",
        "limit": "1",
    }
    r = requests.get(url, headers=_supabase_admin_headers(), params=params, timeout=15)
    r.raise_for_status()
    rows = r.json() or []
    return rows[0] if rows else None

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def _supabase_upsert_quota_row(
    *,
    user_id: str,
    tier: str,
    yyyymm: str,
    tokens_limit: int,
    reset_used: bool = False, # 支付成功建议设为 True，或者按需保持
):
    # 这里的 ?on_conflict=user_id,yyyymm 是关键
    url = _sb_rest_url("quota_monthly") + "?on_conflict=user_id,yyyymm"

    payload = {
        "user_id": user_id,
        "yyyymm": yyyymm,
        "tier": (tier or "free").strip().lower(),
        "tokens_limit": int(tokens_limit),
        "updated_at": _utc_now_iso(),
    }
    # 如果是支付成功同步，通常要把已用额度清零（或者不传，保持现状）
    if reset_used:
        payload["tokens_used"] = 0

    return _sb_post_json(
        url,
        payload,
        prefer="resolution=merge-duplicates,return=representation",
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
    if not user_id or not yyyymm:
        return None

    url = _sb_rest_url("quota_monthly")
    # PostgREST: PATCH /rest/v1/quota_monthly?user_id=eq.xxx&yyyymm=eq.202512
    patch_url = f"{url}?user_id=eq.{user_id}&yyyymm=eq.{yyyymm}"

    payload = {
        "tier": (tier or "free").strip().lower(),
        "tokens_limit": int(tokens_limit),
        "updated_at": _utc_now_iso(),
    }
    if reset_used:
        payload["tokens_used"] = 0

    return _sb_patch_json(
        patch_url,
        payload,
        prefer="return=representation",
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

    tier = (tier or "free").strip().lower()
    yyyymm = yyyymm or _yyyymm_utc_now()
    tokens_limit = int(_tier_monthly_limit(tier))

    # 1) upsert
    try:
        _supabase_upsert_quota_row(
            user_id=user_id,
            tier=tier,
            yyyymm=yyyymm,
            tokens_limit=tokens_limit,
            reset_used=reset_used,
        )
    except Exception as e:
        # upsert 失败要可见
        raise RuntimeError(f"[quota_sync] upsert failed: {e}")

    # 2) verify read
    row = _supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)
    cur_tier = ((row or {}).get("tier") or "").strip().lower()
    cur_limit = int((row or {}).get("tokens_limit") or 0)

    if row and cur_tier == tier and cur_limit == tokens_limit:
        return row

    # 3) fallback PATCH
    try:
        _supabase_patch_quota_row(
            user_id=user_id,
            yyyymm=yyyymm,
            tier=tier,
            tokens_limit=tokens_limit,
            reset_used=reset_used,
        )
    except Exception as e:
        raise RuntimeError(f"[quota_sync] patch failed: {e}")

    # 4) verify again
    row2 = _supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)
    cur_tier2 = ((row2 or {}).get("tier") or "").strip().lower()
    cur_limit2 = int((row2 or {}).get("tokens_limit") or 0)

    if row2 and cur_tier2 == tier and cur_limit2 == tokens_limit:
        return row2

    raise RuntimeError(
        f"[quota_sync] verify failed after patch. "
        f"expect(tier={tier},limit={tokens_limit}) "
        f"got(row={row2})"
    )


def _quota_ensure_row(user_id: str, tier: str, yyyymm: str):
    """
    确保 quota_monthly 本月行存在，且 tokens_limit 与 tier 对齐。
    v0 策略：只做 upsert 初始化/修正 limit，不强行重置 tokens_used（避免误伤正在使用的用户）。
    你如果希望“支付成功时立刻重置本月额度”，可在这里把 tokens_used 置 0（见注释）。
    """
    if not user_id:
        return None

    tier = (tier or "free").lower().strip()
    yyyymm = yyyymm or _yyyymm_utc_now()
    default_limit = int(_tier_monthly_limit(tier))

    row = _supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)
    if row is None:
        # 初始化
        ret =  _supabase_upsert_quota_row(
            user_id=user_id,
            tier=tier,
            yyyymm=yyyymm,
            tokens_limit=default_limit,
        )
        try:
            verify = _supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)
            print("[billing] quota verify:", {"user_id": user_id, "yyyymm": yyyymm, "row": verify})
        except Exception as e:
            print("[billing] quota verify failed:", e)
        return ret

    # 如果 tier/limit 不匹配，做一次修正性 upsert（merge-duplicates）
    try:
        cur_tier = (row.get("tier") or "").lower().strip() or "free"
        cur_limit = int(row.get("tokens_limit") or 0)
    except Exception:
        cur_tier, cur_limit = "free", 0

    if cur_tier != tier or cur_limit != default_limit:
        # ⚠️ 这里复用 upsert：会把 tokens_used 重置为 0（因为你的 _supabase_upsert_quota_row 写死 tokens_used=0）
        # 如果你“不想重置 tokens_used”，需要单独实现 PATCH；但你当前的问题就是要“支付后同步镜像”，
        # 对 alpha-pro 用户来说，支付后把 tokens_used 清零通常是可接受且更符合直觉的。
        ret =  _supabase_upsert_quota_row(
            user_id=user_id,
            tier=tier,
            yyyymm=yyyymm,
            tokens_limit=default_limit,
        )
        try:
            verify = _supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)
            print("[billing] quota verify:", {"user_id": user_id, "yyyymm": yyyymm, "row": verify})
        except Exception as e:
            print("[billing] quota verify failed:", e)
        return ret

    return row


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
    _ensure_supabase_env_for_db()

    url = f"{SUPABASE_URL}/rest/v1/rpc/consume_tokens"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }

    default_limit = _tier_monthly_limit(tier)

    payload = {
        "p_user_id": user_id,
        "p_kind": kind,
        "p_tokens_in": int(tokens_in or 0),
        "p_tokens_out": int(tokens_out or 0),
        "p_tokens_total": int(tokens_total or 0),
        "p_tokens_limit_default": int(default_limit),
    }

    r = requests.post(url, headers=headers, json=payload, timeout=15)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase RPC consume_tokens failed: {r.status_code} {r.text}")

    return r.json() or {}

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
    """
    从 Authorization: Bearer <token> 取 Supabase access_token
    """
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None

def _supabase_get_user_id(access_token: str) -> str | None:
    """
    用 Supabase Auth endpoint 校验 JWT 并拿到 user.id
    """
    _ensure_supabase_env_for_auth()
    url = f"{SUPABASE_URL}/auth/v1/user"
    headers = {
        "apikey": SUPABASE_ANON_KEY,                 # 必须
        "Authorization": f"Bearer {access_token}",   # 必须
    }
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json() or {}
    user = data.get("user") or data  # 兼容不同返回
    return user.get("id")

def _supabase_auth_get_user(access_token: str) -> dict | None:
    """
    用 Supabase Auth endpoint 校验 JWT 并拿到 user（id/email）
    返回：{"id": "...", "email": "..."} 或 None
    """
    _ensure_supabase_env_for_auth()
    url = f"{SUPABASE_URL}/auth/v1/user"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
    }
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json() or {}
    user = data.get("user") or data
    if not isinstance(user, dict):
        return None
    uid = (user.get("id") or "").strip()
    if not uid:
        return None
    return {"id": uid, "email": user.get("email")}


def _supabase_get_profile_tier(user_id: str) -> str:
    """
    用 service role 直接查 profiles.tier（安全、稳定）
    """
    _ensure_supabase_env_for_db()

    # PostgREST: /rest/v1/<table>
    url = f"{SUPABASE_URL}/rest/v1/profiles"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    params = {
        "select": "tier",
        "id": f"eq.{user_id}",
        "limit": "1",
    }
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code != 200:
        # 查不到/异常都按 free 兜底
        return "free"
    rows = r.json() or []
    if not rows:
        return "free"
    tier = (rows[0].get("tier") or "free").strip().lower()
    # ✅ 允许的 tier 白名单，避免脏数据
    if tier not in ("free", "alpha_base", "alpha_pro"):
        return "free"
    return tier

def resolve_user_and_tier(request: Request) -> dict:
    """
    用于“可匿名访问”的路由（比如 /api/quota/me）：
    - 无 Authorization：匿名 -> is_authed False
    - 有 Authorization 但 token 无效：不抛 401，按匿名处理（避免前端循环 401）
    - token 有效：
        1) 解析 user_id/email
        2) profiles upsert 兜底（确保行存在）
        3) 读取 profiles.tier（无则 free）
    返回统一结构：
      {
        "ok": True,
        "is_authed": bool,
        "user_id": str|None,
        "tier": "free|alpha_base|alpha_pro"
      }
    """
    auth = (request.headers.get("authorization") or request.headers.get("Authorization") or "").strip()
    if not auth:
        return {"ok": True, "is_authed": False, "user_id": None, "tier": "free"}

    parts = auth.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        # 不抛 401，按匿名处理
        return {"ok": True, "is_authed": False, "user_id": None, "tier": "free"}

    jwt = parts[1].strip()
    if not jwt:
        return {"ok": True, "is_authed": False, "user_id": None, "tier": "free"}

    # 1) 验证 token -> user
    try:
        user = _supabase_auth_get_user(jwt)
    except Exception as e:
        print("[auth] supabase auth get user failed:", e)
        return {"ok": True, "is_authed": False, "user_id": None, "tier": "free"}

    if not user:
        return {"ok": True, "is_authed": False, "user_id": None, "tier": "free"}

    user_id = (user.get("id") or "").strip()
    email = user.get("email") or None
    if not user_id:
        return {"ok": True, "is_authed": False, "user_id": None, "tier": "free"}

    # 2) profiles upsert：保证 profiles 有行（止血用）
    try:
        _supabase_upsert_profile_row(user_id=user_id, email=email, tier="free")
    except Exception as e:
        print("[auth] profiles upsert failed:", e)

    # 3) 读取 profiles.tier（service role）
    tier = "free"
    try:
        tier = _supabase_get_profile_tier(user_id)
    except Exception as e:
        print("[auth] profiles tier read failed:", e)
        tier = "free"

    return {"ok": True, "is_authed": True, "user_id": user_id, "tier": tier}


def require_authed_user(request: Request) -> dict:
    """
    Blocking auth guard (required auth):
    - Must have Authorization: Bearer <token>
    - Token must resolve to user_id via Supabase Auth
    - Ensure profiles row exists (id=user_id) to avoid "profiles empty but quota exists" inversion
    - Tier is resolved from profiles.tier (service role), with in-memory cache

    Returns:
      { "user_id": "...", "tier": "free|alpha_base|alpha_pro" }
    """
    token = _get_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")

    # 1) Validate token -> user_id
    try:
        user_id = _supabase_get_user_id(token)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Supabase token")

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired Supabase token")

    # 2) Ensure profiles row exists (best-effort)
    #    说明：这里拿不到 email 也没关系，先把 id 行建起来，避免 profiles 空表导致的割裂
    try:
        _supabase_upsert_profile_row(user_id=user_id, email=None, tier="free")
    except Exception as e:
        print("[auth] profiles ensure row failed:", e)

    # 3) Cache tier by user_id
    now = time.time()
    cached = _TIER_CACHE.get(user_id)
    if cached and cached[1] > now:
        return {"user_id": user_id, "tier": cached[0]}

    # 4) Query profiles.tier
    tier = _supabase_get_profile_tier(user_id)

    # 5) Store cache
    _TIER_CACHE[user_id] = (tier, now + _TIER_CACHE_TTL_SEC)
    return {"user_id": user_id, "tier": tier}


# =========================
# FastAPI 基础配置
# =========================

app = FastAPI(
    title="GEO-Max Alpha API",
    description="Brand brief 校验 + COT Stage1 运行 & 解析 等内部能力接口",
    version="0.2.0",
)

# Alpha 阶段 CORS 先放开，后续再收紧域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """
    profiles 的兜底 upsert：确保 profiles 至少存在一行（id=user_id）
    - 用 POST + on_conflict=id + Prefer: resolution=merge-duplicates
    - 允许 email 为空（你现在看到 email=null 是正常结果：说明当时拿不到 email）
    """
    user_id = (user_id or "").strip()
    if not user_id:
        return None

    payload = {"id": user_id}
    if email is not None:
        payload["email"] = email
    if tier is not None:
        payload["tier"] = tier

    url = _sb_rest_url("profiles")  # 如果你有常量，如 PROFILES_TABLE，就替换成常量
    params = "on_conflict=id"

    # PostgREST: /rest/v1/profiles?on_conflict=id
    full_url = f"{url}?{params}"

    # 关键：Prefer resolution=merge-duplicates 才是 upsert
    headers_prefer = "resolution=merge-duplicates,return=representation"
    rows = _sb_post_json(full_url, payload, prefer=headers_prefer)
    if rows and isinstance(rows, list):
        return rows[0]
    return None


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
_tier_monthly_limit = billing_core._tier_monthly_limit
_yyyymm_utc_now = billing_core._yyyymm_utc_now
_supabase_admin_headers = billing_core._supabase_admin_headers
_supabase_get_quota_row = billing_core._supabase_get_quota_row
_utc_now_iso = billing_core._utc_now_iso
_supabase_upsert_quota_row = billing_core._supabase_upsert_quota_row
_supabase_patch_quota_row = billing_core._supabase_patch_quota_row
_quota_sync_after_payment = billing_core._quota_sync_after_payment
_quota_ensure_row = billing_core._quota_ensure_row

_sb_rest_url = billing_core._sb_rest_url
_sb_rpc_url = billing_core._sb_rpc_url
_sb_get_json = billing_core._sb_get_json
_sb_post_json = billing_core._sb_post_json
_sb_patch_json = billing_core._sb_patch_json
_tier_from_price_id = billing_core._tier_from_price_id

_supabase_get_profile_tier = billing_core._supabase_get_profile_tier
_supabase_patch_profile_tier = billing_core._supabase_patch_profile_tier
_profile_sync_tier_after_payment = billing_core._profile_sync_tier_after_payment
_supabase_upsert_profile_row = billing_core._supabase_upsert_profile_row

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





@app.get("/api/quota/me")
def api_quota_me(request: Request):
    """
    返回当前登录用户本月 token quota（不消费 token）
    设计：可匿名访问；token 无效时也不返回 401，避免前端出现 401 循环。
    """
    info = resolve_user_and_tier(request)
    if not info.get("is_authed"):
        return {
            "ok": True,
            "is_authed": False,
            "tier": "free",
            "yyyymm": _yyyymm_utc_now(),
            "tokens_limit": 0,
            "tokens_used": 0,
            "tokens_remaining": 0,
        }

    user_id = info["user_id"]
    tier = info.get("tier") or "free"
    yyyymm = _yyyymm_utc_now()
    default_limit = int(_tier_monthly_limit(tier))

    row = _supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)
    if row is None:
        _supabase_upsert_quota_row(user_id=user_id, tier=tier, yyyymm=yyyymm, tokens_limit=default_limit)
        row = _supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)

    tokens_limit = int((row or {}).get("tokens_limit") or default_limit)
    tokens_used = int((row or {}).get("tokens_used") or 0)
    remaining = max(0, tokens_limit - tokens_used)

    return {
        "ok": True,
        "is_authed": True,
        "tier": tier,
        "yyyymm": yyyymm,
        "tokens_limit": tokens_limit,
        "tokens_used": tokens_used,
        "tokens_remaining": remaining,
    }



# Router registrations (keep GEO/COT ordering and ensure billing/webhooks are exposed)
from app.api.routers.billing import router as billing_router
from app.api.routers.webhooks import router as webhooks_router
from routers.cot_router import router as cot_router
from routers.geo_router import router as geo_router


app.include_router(cot_router)
app.include_router(geo_router)
app.include_router(billing_router)
app.include_router(webhooks_router)



