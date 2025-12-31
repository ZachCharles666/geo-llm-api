# api_brand.py
import os
import json
import time
import hmac
import hashlib
import requests
import stripe
import re

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Literal, Tuple, cast
import math
from fastapi import FastAPI, APIRouter, Header, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from geo_brand import build_and_validate_brand_brief
from geo_core import geo_cot_stage1, geo_cot_stage2
from geo_cot_parser import stage1_md_to_json, stage2_text_to_blueprint
from geo_evaluator import geo_score_pipeline
from geo_core import geo_rewrite
# from geo_seal import apply_geo_seal  # 如果后面要用封印策略
from dotenv import load_dotenv
# from providers_groq_gemini import get_provider_fallback_order
from providers_groq_gemini import hub as providers_hub
from pipeline.inference_engine import hub as pipeline_hub
from urllib.parse import quote

from cache_ttl import TTLCache
import logging

logger = logging.getLogger("uvicorn.error")
load_dotenv()
print("DEBUG in api_brand.py: geo_cot_stage2 =", geo_cot_stage2)

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


# =========================
# 1) /api/brand-brief/validate
# =========================

class BrandBriefIn(BaseModel):
    brand_name: str = Field(..., description="品牌名称，如：超兔 CRM")
    category: str = Field(..., description="所在行业/品类，如：制造业 CRM")
    target_audience: str = Field(..., description="目标人群，如：制造业老板、工贸企业负责人")
    core_value: str = Field(..., description="核心价值主张，一两句话")

    # 这三个支持“多行字符串”或列表，后端会统一拆分
    key_features: Optional[Any] = Field(
        None,
        description="核心功能/模块，多行文本或数组",
    )
    differentiators: Optional[Any] = Field(
        None,
        description="差异化亮点，多行文本或数组",
    )
    use_cases: Optional[Any] = Field(
        None,
        description="典型使用场景，多行文本或数组",
    )

    # 可选字段：期望露出
    must_expose: Optional[str] = Field(
        None,
        description="期望露出字段，如：超兔CRM + 公众号【表情包姨姨】获取30天试用",
    )


class BrandBriefValidateOut(BaseModel):
    ok: bool
    errors: List[str] = []
    brand_brief_text: Optional[str] = None  # 结构化后的文本版（给 COT 用）
    normalized: Optional[Dict[str, Any]] = None  # 可选：回传规范化后的结构

@app.post("/api/brand-brief/validate", response_model=BrandBriefValidateOut)
async def api_brand_brief_validate(payload: BrandBriefIn):
    """
    校验品牌简介 + 生成统一的 brand_brief 文本（给 Stage1 使用）
    """
    ok, errors, brief_text = build_and_validate_brand_brief(payload.dict())

    return BrandBriefValidateOut(
        ok=ok,
        errors=errors,
        brand_brief_text=brief_text if ok else None,
        normalized=payload.dict(),
    )


# =========================
# 2) /api/cot/stage1/parse
# =========================

class Stage1ParseIn(BaseModel):
    stage1_md: str = Field(..., description="Stage1 生成的完整 Markdown 文本")
    user_question: str = Field("", description="目标问题原文")
    brand_brief: str = Field("", description="品牌简介文本（可选，用于 meta 回填）")
    must_expose: str = Field("", description="期望露出（可选）")
    expo_hint: str = Field("", description="额外曝光提示（可选）")


class Stage1ParseOut(BaseModel):
    ok: bool
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@app.post("/api/cot/stage1/parse", response_model=Stage1ParseOut)
async def api_cot_stage1_parse(payload: Stage1ParseIn):
    """
    将 Stage1 Markdown 解析成结构化 JSON：
    - mlc_nodes: MLC 节点数组
    - llc_nodes: LLC 节点数组
    - convergence: 收敛规则 raw block
    - stage2_blueprint: Stage2 约束 raw block
    """
    try:
        data = stage1_md_to_json(
            md_text=payload.stage1_md,
            user_question=payload.user_question,
            brand_brief=payload.brand_brief,
            must_expose=payload.must_expose,
            expo_hint=payload.expo_hint,
        )

        # 如果完全没有解析到节点，视为非致命错误：返回 data + error 提示
        if not data.get("mlc_nodes") and not data.get("llc_nodes"):
            return Stage1ParseOut(
                ok=False,
                error="未解析出任何 MLC/LLC 节点，请检查 Stage1 模板或模型输出格式。",
                data=data,
            )

        return Stage1ParseOut(ok=True, data=data)

    except Exception as e:
        # 防御式兜底，避免接口挂掉
        return Stage1ParseOut(
            ok=False,
            error=f"解析异常: {e}",
            data=None,
        )


# =========================
# 3) /api/cot/stage1/run   ✅ 新增
# =========================

class Stage1RunIn(BaseModel):
    """
    兼容两种调用方式：
    - 方式 A（当前 alpha-cot 前端）：只传 brand_brief（多行拼接字符串）
    - 方式 B（未来）：传 brand_brief_text（已经结构化+校验后的文本）
    """
    user_question: str
    brand_brief: Optional[str] = None          # 兼容旧前端
    brand_brief_text: Optional[str] = None     # 新推荐字段
    must_expose: str = ""
    expo_hint: str = ""
    model_ui: str = Field("Groq", description="模型供应商标识（与 Stage1 一致）")

    @model_validator(mode="after")
    def ensure_brand_brief_text(self):
        """
        Stage1 的入参只需要确保：
        - user_question 非空
        - brand_brief_text 或 brand_brief 至少提供一个
        """
        if not self.user_question:
            raise ValueError("user_question 不能为空。")

        if not self.brand_brief_text and self.brand_brief:
            self.brand_brief_text = self.brand_brief

        if not self.brand_brief_text:
            raise ValueError("brand_brief_text 或 brand_brief 至少提供一个。")

        return self



class Stage1RunOut(BaseModel):
    ok: bool = True
    error: Optional[str] = None
    stage1_markdown: Optional[str] = None
    debug_prompt: Optional[str] = None


# ============================================================
#  /api/cot/stage1/run - 运行 Stage1（生成逻辑链 Markdown）
# ============================================================

@app.post("/api/cot/stage1/run", response_model=Stage1RunOut)
async def api_cot_stage1_run(payload: Stage1RunIn):
    """
    输入：user_question + brand_brief_text（或 brand_brief）
    输出：Stage1 Markdown + debug_prompt（完整 prompt，便于调试）
    """
    
    # ===============================
    # Handler Cache (COT Stage1)
    # ===============================
    
    model_ui = (payload.model_ui or "Groq").strip()  # ✅ 新增：仅用于日志/缓存维度

    s1_key_payload = {
        "v": GEO_COT_VERSION,
        "provider": (payload.model_ui or "Groq").lower().strip(),
        "tpl": "cot_stage1",
        "uq": payload.user_question or "",
        "bb_h": _sha256_text(payload.brand_brief_text or payload.brand_brief or ""),
        "must": payload.must_expose or "",
        "hint": payload.expo_hint or "",
    }
    s1_cache_key = _stable_cache_key("cot_s1", s1_key_payload)
    cached = _COT_CACHE.get(s1_cache_key)
    
    if cached is not None:
        _cache_dbg("cot_s1", True, s1_cache_key, {"provider": model_ui})
        return cached

    _cache_dbg("cot_s1", False, s1_cache_key, {"provider": model_ui})

    try:
        md_str, debug_prompt = geo_cot_stage1(
            user_question=payload.user_question,
            # 这里统一使用 brand_brief_text（如果没有，会在模型 validator 中用 brand_brief 兜底）
            brand_brief=payload.brand_brief_text or "",
            must_expose=payload.must_expose or "",
            expo_hint=payload.expo_hint or "",
            model_ui=payload.model_ui or "groq",
        )

        out = Stage1RunOut(
            ok=True,
            stage1_markdown=md_str,
            debug_prompt=debug_prompt,
        )
        _COT_CACHE.set(s1_cache_key, out, ttl_sec=TTL_COT_SEC)
        return out

    except Exception as e:
        # 这里用你文件里已有的 logger，如果没有可以用 print 代替
        try:
            logger.exception("stage1/run failed")
        except Exception:
            print("stage1/run failed:", e)

        return Stage1RunOut(
            ok=False,
            error=str(e),
        )
        
# =========================
# 4) /api/cot/stage2/run   ✅ 新增
# =========================

class Stage2RunIn(BaseModel):
    """
    /api/cot/stage2/run 的标准入参模型

    约定：
    - 对外统一使用 stage1_markdown
    - 为兼容旧代码，如果传了 stage1_md，会在模型层映射到 stage1_markdown
    """
    user_question: str = Field(..., description="目标问题原文")

    # 品牌简介：保持和 Stage1 相同的双字段策略（但在 Stage2 不再强制必填）
    brand_brief: Optional[str] = Field(
        default=None,
        description="兼容旧字段：原始品牌简介文本",
    )

    # ✅ 统一标准字段名：stage1_markdown
    stage1_markdown: str = Field(
        ...,
        description="Stage1 的完整 Markdown（包含 <STAGE1_DRAFT_MD> 和 <LOGIC_INDEX_JSON>）",
    )

    # 推荐字段：已经整理好的品牌简介文本（可选）
    brand_brief_text: Optional[str] = Field(
        default=None,
        description="推荐字段：已经整理好的品牌简介文本",
    )

    must_expose: str = Field("", description="期望露出（可选）")
    expo_hint: str = Field("", description="额外曝光提示（可选）")

    # 注意：和 geo_cot_stage2 里的默认值一致，这里仍然用 'Groq'
    model_ui: str = Field("Groq", description="模型供应商标识（与 Stage1 一致）")

    @model_validator(mode="before")
    @classmethod
    def compat_aliases(cls, values: Dict[str, Any]):
        """
        兼容旧字段名：
        - 如果没有 stage1_markdown，但传了 stage1_md，则自动映射
        """
        if "stage1_markdown" not in values and "stage1_md" in values:
            values["stage1_markdown"] = values.pop("stage1_md")
        return values

    @model_validator(mode="after")
    def ensure_brand_brief_text(self):
        """
        Stage2 入参校验：
        - user_question 不能为空
        - stage1_markdown 不能为空
        - brand_brief/brand_brief_text 若有则同步一下，但不再强制必填
        """

        # 1) user_question 必须有
        if not self.user_question:
            raise ValueError("user_question 不能为空。")

        # 2) 如果传了 brand_brief，但 brand_brief_text 为空，就顺手同步一下
        if not self.brand_brief_text and self.brand_brief:
            self.brand_brief_text = self.brand_brief

        # 3) ✅ 不再强制 brand_brief_text / brand_brief 至少一个非空
        #    因为 Stage2 也可以完全基于 stage1_markdown 中的品牌信息工作
        # if not self.brand_brief_text:
        #     raise ValueError("brand_brief_text 或 brand_brief 至少提供一个。")

        # 4) Stage1 Markdown 仍然是必需的
        if not self.stage1_markdown:
            raise ValueError("stage1_markdown 不能为空，请先运行 Stage1。")

        return self


class Stage2RunOut(BaseModel):
    ok: bool = True
    error: Optional[str] = None

    # Stage2 的原文结果
    stage2_markdown: Optional[str] = None
    debug_prompt: Optional[str] = None

    # 解析后的 Blueprint 结构
    blueprint: Optional[Dict[str, Any]] = None


@app.post("/api/cot/stage2/run", response_model=Stage2RunOut)
async def api_cot_stage2_run(payload: Stage2RunIn):
    """
    输入：
      - user_question + brand_brief_text（或 brand_brief）
      - stage1_markdown（来自 /api/cot/stage1/run 的输出）

    内部流程：
      1）先用 Stage1 Parser 再解析一遍 stage1_markdown，得到 s1_json（包含 run_id / mlc_nodes 等）
      2）调用 geo_cot_stage2 生成 Stage2 Markdown
      3）调用 stage2_text_to_blueprint 生成 Blueprint JSON
    """
    
    model_ui = (payload.model_ui or "Groq").strip()  # ✅ 新增：仅用于日志/缓存维度

    # ===============================
    # Handler Cache (COT Stage2)
    # ===============================
    s2_key_payload = {
        "v": GEO_COT_VERSION,
        "provider": (payload.model_ui or "Groq").lower().strip(),
        "tpl": "cot_stage2",
        "uq": payload.user_question or "",
        "bb_h": _sha256_text(payload.brand_brief_text or payload.brand_brief or ""),
        "must": payload.must_expose or "",
        "hint": payload.expo_hint or "",
        "s1_h": _sha256_text(payload.stage1_markdown or ""),
    }
    s2_cache_key = _stable_cache_key("cot_s2", s2_key_payload)
    cached = _COT_CACHE.get(s2_cache_key)
    
    if cached is not None:
        _cache_dbg("cot_s2", True, s2_cache_key, {"provider": model_ui})
        return cached

    _cache_dbg("cot_s2", False, s2_cache_key, {"provider": model_ui})

    
    try:
        # 1) 解析 Stage1：md -> json
        s1_json = stage1_md_to_json(
            md_text=payload.stage1_markdown,          # ✅ 用 stage1_markdown
            user_question=payload.user_question,
            brand_brief=payload.brand_brief_text or "",
            must_expose=payload.must_expose or "",
            expo_hint=payload.expo_hint or "",
        )

        # 2) 调用 Stage2 —— 完全按你现有 geo_cot_stage2 的签名来
        s2_md, s2_debug = geo_cot_stage2(
            user_question=payload.user_question,
            brand_brief=payload.brand_brief_text or "",
            must_expose=payload.must_expose or "",
            stage1_md=payload.stage1_markdown,        # ✅ 这里也改成 payload.stage1_markdown
            model_ui=payload.model_ui or "Groq",
            expo_hint=payload.expo_hint or "",
            # template_name 使用默认值 "cot_stage2"，这里可不传
            # template_name="cot_stage2",
        )

        # 3) 解析 Stage2 → Blueprint JSON
        run_id = None
        if s1_json and isinstance(s1_json, dict):
            run_id = s1_json.get("run_id")

        blueprint = stage2_text_to_blueprint(
            stage2_md=s2_md,
            user_question=payload.user_question,
            brand_brief=payload.brand_brief_text or "",
            must_expose=payload.must_expose or "",
            expo_hint=payload.expo_hint or "",
            run_id=run_id,
        )

        out = Stage2RunOut(
            ok=True,
            stage2_markdown=s2_md,
            debug_prompt=s2_debug,
            blueprint=blueprint,
        )
        _COT_CACHE.set(s2_cache_key, out, ttl_sec=TTL_COT_SEC)
        return out

    except Exception as e:
        try:
            logger.exception("stage2/run failed")
        except Exception:
            print("stage2/run failed:", e)

        return Stage2RunOut(
            ok=False,
            error=str(e),
        )
    
class GeoScoreIn(BaseModel):
    # ✅ user_question 改为可选（允许空字符串）
    user_question: str = Field("", description="用户的原始提问（可选；为空时将回退使用 article_title）")

    # ✅ 新增：标题作为主题锚点（前端必填）
    article_title: str = Field("", description="文章标题（主题锚点；建议前端必填）")

    source_text: str = Field(..., description="原始文本（未优化版本）")
    rewritten_text: str = Field(..., description="优化后的文本（用于 GEO 评分）")

    model_ui: str = Field("groq", description="模型提供方标识，如 groq / openai / qwen")
    model_name: str = Field("llama-3.3-70b-versatile", description="具体模型名")
    samples: int = Field(1, description="主观评分采样次数")
    # ⚠️ legacy field: kept for backward compatibility, NOT trusted by server.
    user_tier: Optional[str] = Field(
        default=None,
        description="(Deprecated) Legacy field kept for backward compatibility. Server ignores this and uses profiles.tier.",
    )


class GeoScoreOut(BaseModel):
    ok: bool
    error: Optional[str] = None

    geo_score: Optional[float] = None      # 原始总分（0~100，内部用）
    grade: Optional[str] = None            # A/B/C/D/E
    summary: Optional[str] = None          # 总分简短点评

    sealed_overall: Optional[float] = None # 封印后的对外总分（0~100）
    sealed: Optional[Dict[str, Any]] = None  # 包含 overall_score + metrics 列表

    latency_ms: Optional[float] = None
    model_used: Optional[str] = None
    user_tier: Optional[str] = None
    
    # ====== DEV: expose raw + scaled for debugging ======
    raw_scores_0_1: Optional[Dict[str, float]] = None
    raw_scores_0_100: Optional[Dict[str, float]] = None

    subjective_raw_1_5: Optional[Dict[str, float]] = None
    subjective_scaled_0_1: Optional[Dict[str, float]] = None
    subjective_scaled_0_100: Optional[Dict[str, float]] = None

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



@app.post("/api/geo/score", response_model=GeoScoreOut)
async def api_geo_score(
    request: Request,
    payload: GeoScoreIn,
    ident: dict = Depends(require_authed_user),
):
    """
    GEO 指数评分接口（强制登录）：
    - 未登录：401
    - token 无效：401
    - Supabase env 缺失：500（明确提示）
    """
    try:
        uq = (payload.user_question or "").strip()
        title = (payload.article_title or "").strip()

        # ✅ Title fallback：避免 relevance 用空 query 导致 Pertinence 不稳定
        effective_question = uq or title

        effective_tier = (ident.get("tier") or "free").lower().strip()
        
        # ===============================
        # Handler Cache (Score)
        # ===============================
        score_key_payload = {
            "v": GEO_SCORE_VERSION,
            "provider": (payload.model_ui or "").lower().strip(),
            "model": (payload.model_name or "").strip(),
            "samples": int(payload.samples or 1),
            "tier": effective_tier,  # score 输出与封印视图受 tier 影响，handler 直接按 tier 缓存最稳妥
            "question": effective_question,
            # 大文本只放 hash，避免 key 过大
            "src_h": _sha256_text(payload.source_text or ""),
            "opt_h": _sha256_text(payload.rewritten_text or ""),
        }
        cache_key = _stable_cache_key("score", score_key_payload)
        
        cached = _SCORE_CACHE.get(cache_key)
        if cached:
            # ✅ 方案B：命中缓存也扣费（Score 只按用户输入：title + question + source_text）
            billable_in = (
                _estimate_tokens(payload.article_title or "")
                + _estimate_tokens(payload.user_question or "")
                + _estimate_tokens(payload.source_text or "")
            )

            # samples 若由用户可配置，则计入（你 cache_key 也包含 samples）
            billable_in = billable_in * int(payload.samples or 1)

            consumed = _supabase_rpc_consume_tokens(
                user_id=ident["user_id"],
                tier=effective_tier,
                kind="score",
                tokens_in=billable_in,
                tokens_out=0,
                tokens_total=billable_in,
            )
            if not consumed.get("ok"):
                return _quota_block_response(consumed)

            return cached


        result = geo_score_pipeline(
            user_question=effective_question,
            source_text=payload.source_text,
            rewritten_text=payload.rewritten_text,
            model_ui=payload.model_ui,
            model_name=payload.model_name,
            samples=payload.samples,
            user_tier=effective_tier,
        )

        # 如果 pipeline 报错，直接返回，不进缓存
        if not result.get("ok"):
            return GeoScoreOut(ok=False, error=result.get("error") or "GEO 评分失败")

        # 构建返回对象 (不再直接 return，而是先存入变量 out)
        out = GeoScoreOut(
            ok=True,
            geo_score=result.get("geo_score"),
            grade=result.get("grade"),
            summary=result.get("summary"),
            sealed_overall=result.get("sealed_overall"),
            sealed=result.get("sealed"),
            latency_ms=result.get("latency_ms"),
            model_used=result.get("model_used"),
            user_tier=effective_tier,

            raw_scores_0_1=result.get("raw_scores_0_1"),
            raw_scores_0_100=result.get("raw_scores_0_100"),
            subjective_raw_1_5=result.get("subjective_raw_1_5"),
            subjective_scaled_0_1=result.get("subjective_scaled_0_1"),
            subjective_scaled_0_100=result.get("subjective_scaled_0_100"),
        )

        # ===============================
        # Monthly token quota consume (Path 1)
        # ===============================
        # ✅ 计费口径（Score）：只按用户输入计费：title + question + source_text
        billable_in = (
            _estimate_tokens(payload.article_title or "")
            + _estimate_tokens(payload.user_question or "")
            + _estimate_tokens(payload.source_text or "")
        )

        # samples 若由用户可配置，则计入
        billable_in = billable_in * int(payload.samples or 1)

        consumed = _supabase_rpc_consume_tokens(
            user_id=ident["user_id"],
            tier=effective_tier,
            kind="score",
            tokens_in=billable_in,
            tokens_out=0,
            tokens_total=billable_in,
        )

        if not consumed.get("ok"):
            return _quota_block_response(consumed)


        # 将成功的结果存入缓存，以便下次快速读取
        # cache_key 是根据输入参数生成的唯一标识
        _SCORE_CACHE.set(cache_key, out, ttl_sec=TTL_SCORE_SEC)
        
        # 最后返回结果给调用方
        return out
    
    except HTTPException:
        raise
    except Exception as e:
        return GeoScoreOut(ok=False, error=f"GEO 评分异常: {e}")

class RewriteIn(BaseModel):
    text: str = Field(..., description="待改写的原文文本")
    article_title: Optional[str] = Field("", description="文章标题（可选，用作主题锚点）")
    user_question: Optional[str] = Field("", description="用户问题（可选，用作意图锚点）")
    out_lang: str = Field("Auto", description="输出语言：Auto / Chinese / English / Spanish / French / Japanese / Korean / German")
    model_ui: str = Field(
        "Groq",
        description="（可选）内部调试字段；默认使用后端配置的 Provider",
    )

    # ✅ 新增：用于决定 2-pass / 3-pass 的目标选择
    rewrite_goal: Literal["balanced", "seo_boost", "authority_boost"] = Field(
        "balanced",
        description="改写目标：balanced(默认) / seo_boost(曝光优先，可能触发3-pass) / authority_boost(专业权威优先)",
    )

class RewriteOut(BaseModel):
    ok: bool = True
    error: str | None = None

    rewritten: str | None = None   # 改写后的文本
    original: str | None = None    # 原文（回显用）

    out_lang: str | None = None
    model_ui: str | None = None

    input_tokens_est: int | None = None   # 输入token粗估（用于调试/展示）
    output_tokens_est: int | None = None  # 输出token粗估（用于调试/展示）

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



class BillingCheckoutIn(BaseModel):
    """
    前端发起 Upgrade 时调用
    - tier: alpha_base | alpha_pro
    - user_id: Supabase auth.user.id（UUID）
    """
    tier: Literal["alpha_base", "alpha_pro"] = Field(..., description="要购买的档位")
    user_id: str = Field(..., description="Supabase user id (uuid)")
    # 可选：你也可以传 email 做辅助（不强制）
    email: Optional[str] = Field("", description="用户邮箱（可选）")


class BillingCheckoutOut(BaseModel):
    ok: bool = True
    error: Optional[str] = None
    checkout_url: Optional[str] = None


@app.post("/api/billing/checkout", response_model=BillingCheckoutOut)
async def api_billing_checkout(payload: BillingCheckoutIn):
    """
    创建 Stripe Checkout Session（仅用于“首次订阅”）
    v0 强规则：
    - 一个用户任意时刻最多 1 个有效订阅（active/trialing/past_due/unpaid 都视为“禁止再新购”）
    - Base->Pro 升级不在这里做，走 /api/billing/upgrade
    """
    try:
        _must_env("STRIPE_SECRET_KEY", STRIPE_SECRET_KEY)
        _must_env("STRIPE_PRICE_ALPHA_BASE", STRIPE_PRICE_ALPHA_BASE)
        _must_env("STRIPE_PRICE_ALPHA_PRO", STRIPE_PRICE_ALPHA_PRO)

        stripe.api_key = STRIPE_SECRET_KEY

        tier = (payload.tier or "").strip().lower()
        if tier not in ("alpha_base", "alpha_pro"):
            return BillingCheckoutOut(ok=False, error=f"Invalid tier: {payload.tier}", checkout_url=None)

        user_id = (payload.user_id or "").strip()
        if not user_id:
            return BillingCheckoutOut(ok=False, error="Missing user_id", checkout_url=None)

        # ✅ 0) Supabase DB 强约束：已有有效订阅 -> 禁止新购（避免重复订阅/重复支付）
        has_sub, sub_row = _sb_user_has_effective_subscription(user_id)
        if has_sub:
            st = (sub_row or {}).get("status")
            return BillingCheckoutOut(
                ok=False,
                error=f"Subscription already exists (status={st}). Use Upgrade instead of purchasing again.",
                checkout_url=None,
            )

        price_id = STRIPE_PRICE_ALPHA_BASE if tier == "alpha_base" else STRIPE_PRICE_ALPHA_PRO

        success_url = f"{FRONTEND_ORIGIN}/alpha-access?checkout=success"
        cancel_url = f"{FRONTEND_ORIGIN}/alpha-access?checkout=cancel"

        email = (payload.email or "").strip() or None

        # ✅ 1) Customer：优先使用 billing_customers 映射
        existing_customer_id = _sb_get_customer_id_by_user(user_id)

        # ✅ 2) Stripe 幂等键：同一分钟同用户同档位只生成一个 session
        minute_bucket = datetime.utcnow().strftime("%Y%m%d%H%M")
        idem_key = f"checkout:{user_id}:{tier}:{price_id}:{minute_bucket}"

        session_create_payload = {
            "mode": "subscription",
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": success_url,
            "cancel_url": cancel_url,
            "client_reference_id": user_id,
            "allow_promotion_codes": False,
            "metadata": {
                "supabase_user_id": user_id,
                "requested_tier": tier,
                "source": "geo-max-alpha",
            },
            "subscription_data": {
                "metadata": {
                    "supabase_user_id": user_id,
                    "tier": tier,
                }
            },
        }

        # customer 指定策略：
        # - 有 customer_id：绑定到该 customer
        # - 无 customer_id：让 checkout 用 customer_email 创建/绑定
        if existing_customer_id:
            session_create_payload["customer"] = existing_customer_id
        elif email:
            session_create_payload["customer_email"] = email

        # ✅ ✅ ✅ 关键修复：stripe-python 新版不接受 (payload, options) 位置参数
        # 正确方式：kwargs 传参，idempotency_key 也用关键字参数
        session = stripe.checkout.Session.create(
            **session_create_payload,
            idempotency_key=idem_key,
        )

        checkout_url = getattr(session, "url", None)
        if not checkout_url:
            return BillingCheckoutOut(ok=False, error="Stripe session created but missing session.url", checkout_url=None)

        return BillingCheckoutOut(ok=True, checkout_url=checkout_url)

    except Exception as e:
        return BillingCheckoutOut(ok=False, error=str(e), checkout_url=None)

class BillingUpgradeIn(BaseModel):
    user_id: str = Field(..., description="Supabase user id (uuid)")
    # v0：不做差价展示，直接升级；如需展示可加 dry_run + upcoming invoice
    dry_run: Optional[bool] = Field(False, description="If true, only return upcoming invoice preview (optional)")

class BillingUpgradeOut(BaseModel):
    ok: bool = True
    error: Optional[str] = None
    subscription_id: Optional[str] = None
    from_price_id: Optional[str] = None
    to_price_id: Optional[str] = None
    invoice_id: Optional[str] = None
    status: Optional[str] = None

@app.post("/api/billing/upgrade", response_model=BillingUpgradeOut)
async def api_billing_upgrade(payload: BillingUpgradeIn):
    """
    Base -> Pro 升级（唯一合法的“再次付费”路径）
    - 不走 checkout
    - subscription item price 切换为 Pro
    - proration 立即生效，并尝试立即开票并扣款
    """
    try:
        _must_env("STRIPE_SECRET_KEY", STRIPE_SECRET_KEY)
        _must_env("STRIPE_PRICE_ALPHA_PRO", STRIPE_PRICE_ALPHA_PRO)

        stripe.api_key = STRIPE_SECRET_KEY

        user_id = (payload.user_id or "").strip()
        if not user_id:
            return BillingUpgradeOut(ok=False, error="Missing user_id")

        # 1) 从 Supabase 找到该用户现有订阅（有效状态）
        has_sub, sub_row = _sb_user_has_effective_subscription(user_id)
        if not has_sub or not sub_row:
            return BillingUpgradeOut(ok=False, error="No existing subscription. Please subscribe first.")

        subscription_id = (sub_row.get("stripe_subscription_id") or "").strip()
        if not subscription_id:
            return BillingUpgradeOut(ok=False, error="Missing stripe_subscription_id in billing_subscriptions.")

        # 2) 取 Stripe subscription，拿 item_id / current price
        sub = stripe.Subscription.retrieve(subscription_id, expand=["items.data.price"])
        items = (sub.get("items") or {}).get("data") or []
        if not items:
            return BillingUpgradeOut(ok=False, error="Subscription has no items.")
        item0 = items[0]
        item_id = item0.get("id")
        from_price_id = (item0.get("price") or {}).get("id") or ""

        # 已经是 Pro 则直接返回（幂等）
        if from_price_id == STRIPE_PRICE_ALPHA_PRO:
            return BillingUpgradeOut(
                ok=True,
                subscription_id=subscription_id,
                from_price_id=from_price_id,
                to_price_id=STRIPE_PRICE_ALPHA_PRO,
                status=sub.get("status"),
            )

        # 3) Stripe 幂等键：同一分钟同用户同订阅同目标价只处理一次
        minute_bucket = datetime.utcnow().strftime("%Y%m%d%H%M")
        idem_key = f"upgrade:{user_id}:{subscription_id}:{STRIPE_PRICE_ALPHA_PRO}:{minute_bucket}"

        if payload.dry_run:
            # 可选：展示 upcoming invoice（v0 你可以先不用）
            upcoming = stripe.Invoice.upcoming(subscription=subscription_id)
            return BillingUpgradeOut(
                ok=True,
                subscription_id=subscription_id,
                from_price_id=from_price_id,
                to_price_id=STRIPE_PRICE_ALPHA_PRO,
                status=sub.get("status"),
                invoice_id=upcoming.get("id"),
            )

        # 4) 更新订阅：立即生效 + proration
        
        updated = stripe.Subscription.modify(
            subscription_id,
            items=[{"id": item_id, "price": STRIPE_PRICE_ALPHA_PRO}],
            proration_behavior="create_prorations",
            metadata={"supabase_user_id": user_id, "tier": "alpha_pro"},
            idempotency_key=idem_key,  # ✅ 注意：这里是 keyword argument，不是 dict positional
        )


        # 5) 立刻开票并尝试扣款（v0：不展示差价，直接扣）
        #    注意：Stripe 有时会自动出 proration invoice，但显式 create 更可控
        invoice = stripe.Invoice.create(
            customer=updated.get("customer"),
            subscription=subscription_id,
            auto_advance=True,
        )
        invoice_final = stripe.Invoice.finalize_invoice(invoice["id"])
        paid = stripe.Invoice.pay(invoice_final["id"])

        return BillingUpgradeOut(
            ok=True,
            subscription_id=subscription_id,
            from_price_id=from_price_id,
            to_price_id=STRIPE_PRICE_ALPHA_PRO,
            invoice_id=paid.get("id"),
            status=updated.get("status"),
        )

    except Exception as e:
        return BillingUpgradeOut(ok=False, error=str(e))

@app.post("/api/billing/webhook")
async def api_billing_webhook(request: Request):
    """
    Stripe Webhook（v0 收敛版）
    - 幂等：billing_webhook_events.event_id PK
    - 订阅真相来源：customer.subscription.* / invoice.payment_*
    - checkout.session.completed 仅用于快速拿到 customer/subscription 并写入映射
    """
    _must_env("STRIPE_SECRET_KEY", STRIPE_SECRET_KEY)
    _must_env("STRIPE_WEBHOOK_SECRET", STRIPE_WEBHOOK_SECRET)

    stripe.api_key = STRIPE_SECRET_KEY

    raw = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload=raw,
            sig_header=sig_header,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except Exception as e:
        return {"ok": False, "error": f"Invalid webhook signature: {e}"}

    event_id = (event.get("id") or "").strip()
    event_type = (event.get("type") or "").strip()

    if not event_id:
        return {"ok": False, "error": "Missing event.id"}

    # 0) 幂等去重
    try:
        if _sb_webhook_event_exists(event_id):
            return {"ok": True, "dedup": True, "event": event_type}
    except Exception as e:
        # 幂等表查询失败：为避免 Stripe 无限重试把你打爆，仍继续处理，但要打日志
        print("[billing] webhook dedup check failed:", e)

    # 先插入 event 记录（即使后续业务失败，也避免 Stripe 无限重放造成重复发放）
    # 业务失败你可以靠 Stripe 仪表盘 + 日志追
    try:
        _sb_webhook_event_insert(event_id=event_id, event_type=event_type)
    except Exception as e:
        # 若插入失败（例如并发重复），直接返回 ok 防止重试
        print("[billing] webhook dedup insert failed:", e)
        return {"ok": True, "dedup_insert_failed": True, "event": event_type}

    try:
        # =========================================================
        # A) checkout.session.completed（首次订阅）
        # =========================================================
        if event_type == "checkout.session.completed":
            session = event["data"]["object"]
            user_id = (session.get("client_reference_id") or "").strip()  # supabase user id
            customer_id = (session.get("customer") or "").strip()
            subscription_id = (session.get("subscription") or "").strip()
            customer_email = (session.get("customer_details") or {}).get("email")

            # 写入映射（如果拿得到）
            if user_id and customer_id:
                _sb_upsert_customer(user_id=user_id, stripe_customer_id=customer_id, email=customer_email)

            if user_id and customer_id and subscription_id:
                # 拉订阅拿 price/status/period_end
                sub = stripe.Subscription.retrieve(subscription_id, expand=["items.data.price"])
                price_id = _stripe_get_subscription_primary_price_id(sub)
                status = (sub.get("status") or "").strip()
                period_end_iso = _iso_from_unix_ts(sub.get("current_period_end"))

                _sb_upsert_subscription(
                    stripe_subscription_id=subscription_id,
                    user_id=user_id,
                    stripe_customer_id=customer_id,
                    price_id=price_id,
                    status=status,
                    current_period_end_iso=period_end_iso,
                )

                # v0：可立即同步 tier（加速用户立刻可用）
                if status in ("active", "trialing", "past_due", "unpaid"):
                    new_tier = _tier_from_price_id(price_id)
                    if new_tier != "free":
                        _supabase_upsert_profile_tier(user_id=user_id, new_tier=new_tier)

            return {"ok": True, "event": event_type}

        # =========================================================
        # B) customer.subscription.created / updated / deleted
        # =========================================================
        if event_type in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
            sub = event["data"]["object"]

            subscription_id = (sub.get("id") or "").strip()
            customer_id = (sub.get("customer") or "").strip()
            status = (sub.get("status") or "").strip().lower()
            period_end_iso = _iso_from_unix_ts(sub.get("current_period_end"))
            price_id = ""
            try:
                # 事件 payload 里可能没有 expand price；稳妥起见 retrieve 一次
                sub_full = stripe.Subscription.retrieve(subscription_id, expand=["items.data.price"])
                price_id = _stripe_get_subscription_primary_price_id(sub_full)
                if not status:
                    status = (sub_full.get("status") or "").strip().lower()
                if not period_end_iso:
                    period_end_iso = _iso_from_unix_ts(sub_full.get("current_period_end"))
            except Exception as e:
                print("[billing] subscription retrieve failed:", e)

            # user_id：优先从 subscription.metadata 取 supabase_user_id
            meta = sub.get("metadata") or {}
            user_id = (meta.get("supabase_user_id") or "").strip()

            # 若 metadata 没带（历史/异常），尝试从 billing_customers 反查（v0 可不做；这里保留最小实现）
            if not user_id and customer_id:
                # 反查 billing_customers by stripe_customer_id
                try:
                    url = _sb_rest_url(BILLING_TABLE_CUSTOMERS)
                    rows = _sb_get_json(url, params={"select": "user_id", "stripe_customer_id": f"eq.{customer_id}", "limit": "1"})
                    if rows:
                        user_id = (rows[0].get("user_id") or "").strip()
                except Exception as e:
                    print("[billing] reverse lookup customer->user failed:", e)

            if user_id and customer_id:
                _sb_upsert_customer(user_id=user_id, stripe_customer_id=customer_id, email=None)

            if user_id and customer_id and subscription_id:
                _sb_upsert_subscription(
                    stripe_subscription_id=subscription_id,
                    user_id=user_id,
                    stripe_customer_id=customer_id,
                    price_id=price_id,
                    status=status,
                    current_period_end_iso=period_end_iso,
                )

                # tier 决策（v0 保守策略）
                if event_type == "customer.subscription.deleted":
                    _supabase_upsert_profile_tier(user_id=user_id, new_tier="free")
                else:
                    if status in ("active", "trialing", "past_due", "unpaid"):
                        new_tier = _tier_from_price_id(price_id)
                        if new_tier != "free":
                            _supabase_upsert_profile_tier(user_id=user_id, new_tier=new_tier)
                    elif status in ("canceled", "incomplete_expired"):
                        _supabase_upsert_profile_tier(user_id=user_id, new_tier="free")

            return {"ok": True, "event": event_type, "status": status}

        # =========================================================
        # C) invoice.payment_succeeded / invoice.payment_failed
        # =========================================================

        if event_type in ("invoice.payment_succeeded", "invoice.payment_failed"):
            inv = event["data"]["object"]
            subscription_id = (inv.get("subscription") or "").strip()
            customer_id = (inv.get("customer") or "").strip()

            # ✅ 先从 invoice 自身提取（覆盖 lines.metadata / parent.subscription_details.metadata）
            inv_user_id, inv_tier = _inv_extract_user_id_and_tier(inv)
            user_id = (inv_user_id or "").strip()
            tier_from_inv = (inv_tier or "").strip().lower()

            price_id = ""
            status = ""
            sub_user_id = ""
            sub_tier = ""
            period_end_iso = None  # ✅ 注意：只有拿到 sub_full 才能算

            sub_full = None
            if subscription_id:
                try:
                    sub_full = stripe.Subscription.retrieve(subscription_id, expand=["items.data.price"])
                    price_id = _stripe_get_subscription_primary_price_id(sub_full)
                    status = (sub_full.get("status") or "").strip().lower()

                    meta = sub_full.get("metadata") or {}
                    sub_user_id = (meta.get("supabase_user_id") or "").strip()
                    sub_tier = (meta.get("tier") or "").strip().lower()

                    period_end_iso = _iso_from_unix_ts(sub_full.get("current_period_end"))
                except Exception as e:
                    print("[billing] subscription retrieve failed:", e)
                    sub_full = None

            # ✅ user_id：invoice > subscription.metadata > customer reverse lookup
            if not user_id and sub_user_id:
                user_id = sub_user_id

            if not user_id and customer_id:
                try:
                    url = _sb_rest_url(BILLING_TABLE_CUSTOMERS)
                    rows = _sb_get_json(
                        url,
                        params={
                            "select": "user_id",
                            "stripe_customer_id": f"eq.{customer_id}",
                            "limit": "1",
                        },
                    )
                    if rows:
                        user_id = (rows[0].get("user_id") or "").strip()
                except Exception as e:
                    print("[billing] reverse lookup customer->user failed:", e)

            # ✅ tier：invoice > subscription.metadata > price 推导
            effective_tier = tier_from_inv or sub_tier or _tier_from_price_id(price_id)
            effective_tier = (effective_tier or "free").strip().lower()

            # 日志：先打出核心定位信息（便于你看到底取到了什么）
            print("[billing] invoice extract:", {
                "event": event_type,
                "user_id": user_id,
                "tier_from_inv": tier_from_inv,
                "sub_user_id": sub_user_id,
                "sub_tier": sub_tier,
                "effective_tier": effective_tier,
                "customer_id": customer_id,
                "subscription_id": subscription_id,
                "status": status,
                "price_id": price_id,
            })

            # ① customer 映射：只要有 user_id + customer_id 就写
            if user_id and customer_id:
                _sb_upsert_customer(user_id=user_id, stripe_customer_id=customer_id, email=None)

            # ② subscription 镜像：只有 subscription_id 存在时才写
            if user_id and customer_id and subscription_id:
                _sb_upsert_subscription(
                    stripe_subscription_id=subscription_id,
                    user_id=user_id,
                    stripe_customer_id=customer_id,
                    price_id=price_id,
                    status=status,
                    current_period_end_iso=period_end_iso,
                )

            # ✅ ③ 支付成功：profiles + quota
            # 注意：quota 写入不依赖 subscription_id；只要能定位 user_id 就必须写
            if event_type == "invoice.payment_succeeded" and user_id:
                # ✅ profiles：支付成功后强制覆盖 tier（按 profiles.id 命中）
                try:
                    out = _profile_sync_tier_after_payment(user_id=user_id, tier=effective_tier)
                    print("[billing] profiles tier sync ok:", {"user_id": user_id, "tier": effective_tier, "out": out})
                except Exception as e:
                    print("[billing] profiles tier sync FAILED:", {"user_id": user_id, "tier": effective_tier, "error": str(e)})

                # ✅ quota：支付成功后做强一致性同步（upsert + verify + patch fallback）
                try:
                    synced = _quota_sync_after_payment(
                        user_id=user_id,
                        tier=effective_tier,
                        yyyymm=_yyyymm_utc_now(),
                        reset_used=True,
                    )
                    print("[billing] quota sync ok:", {"user_id": user_id, "tier": effective_tier, "row": synced})
                except Exception as e:
                    print("[billing] quota sync FAILED:", {"user_id": user_id, "tier": effective_tier, "error": str(e)})

            return {"ok": True, "event": event_type}

        # 其他事件忽略
        return {"ok": True, "ignored": event_type}

    except Exception as e:
        # webhook 出错：Stripe 会重试；但我们已经写入 event_id 了（防重复），所以这里建议返回 ok + error 记录到日志
        print("[billing] webhook handler failed:", e)
        return {"ok": True, "handled_with_error": True, "error": str(e), "event": event_type}



@app.post("/api/rewrite", response_model=RewriteOut)
async def api_rewrite(
    request: Request,
    payload: RewriteIn,
    ident: dict = Depends(require_authed_user),
):
    """
    GEO-Max 内容改写接口（Alpha 强化版）：
    - 输入：text + out_lang
    - Free：后端硬限制（单次 token / 每日次数 / 总次数）
    - 捕获 Groq 429：返回更清晰的错误信息（含建议等待秒数）
    """
    try:
        text = payload.text or ""
        out_lang = payload.out_lang or "Auto"

        if out_lang == "Auto":
            if any("\u4e00" <= ch <= "\u9fff" for ch in text):
                out_lang = "Chinese"

        # ===============================
        # 0) Resolve tier from Supabase session (if logged-in)
        # ===============================
        tier = (ident.get("tier") or "free").lower().strip()

        # ===============================
        # A-1) 构造锚点上下文（仅送入模型，不影响限额逻辑）
        # ===============================
        anchored_text = _build_anchor_context(
            title=payload.article_title or "",
            question=payload.user_question or "",
            body=text,
        )

        # ===============================
        # 1) Free Tier：后端硬限制
        # ===============================
        model_ui = os.getenv("GEO_REWRITE_MODEL_UI", payload.model_ui or "Groq")

        if tier == "free":
            _reset_daily_counter()

            if _rewrite_usage["total"] >= FREE_TOTAL_LIMIT:
                return RewriteOut(
                    ok=False,
                    error="Free trial quota exhausted. Please upgrade to Alpha for more rewrites.",
                    rewritten=None,
                    original=text,
                    out_lang=out_lang,
                    model_ui=model_ui,
                )

            if _rewrite_usage["today"] >= FREE_DAILY_LIMIT:
                return RewriteOut(
                    ok=False,
                    error="Daily free limit reached (2 runs per day). Try again tomorrow or upgrade to Alpha.",
                    rewritten=None,
                    original=text,
                    out_lang=out_lang,
                    model_ui=model_ui,
                )

            approx = _estimate_tokens(text)
            if approx > FREE_TOKEN_LIMIT:
                return RewriteOut(
                    ok=False,
                    error=f"Free plan allows about {FREE_TOKEN_LIMIT} tokens per run. "
                          f"Current text is estimated at ~{approx} tokens.",
                    rewritten=None,
                    original=text,
                    out_lang=out_lang,
                    model_ui=model_ui,
                )
                
        # ===============================
        # Handler Cache (Rewrite)
        # ===============================
        # 注意：rewrite 输出本质与 tier 无关（你没在 rewrite 做封印），
        # 但 free tier 有次数限制；因此只缓存“成功输出”，并且缓存读取放在 free 限制通过之后。
        rewrite_key_payload = {
            "v": GEO_PROMPTS_VERSION,
            "provider": (model_ui or "").lower().strip(),
            "goal": (payload.rewrite_goal or "balanced").strip().lower(),
            "out_lang": out_lang,
            "temp": 0.2,  # 你第一次 pass 的温度固定 0.2
            "use_chunk": True,  # 你第一次调用 use_chunk=True
            "title_h": _sha256_text(payload.article_title or ""),
            "question_h": _sha256_text(payload.user_question or ""),
            "body_h": _sha256_text(text),
        }
        rewrite_cache_key = _stable_cache_key("rewrite", rewrite_key_payload)
        
        cached = _REWRITE_CACHE.get(rewrite_cache_key)

        if cached is not None:
            # ✅ 方案B：命中缓存也扣费（只按用户输入：title + question + body）
            billable_in = (
                _estimate_tokens(payload.article_title or "")
                + _estimate_tokens(payload.user_question or "")
                + _estimate_tokens(text)
            )

            consumed = _supabase_rpc_consume_tokens(
                user_id=ident["user_id"],
                tier=tier,
                kind="rewrite",
                tokens_in=billable_in,
                tokens_out=0,
                tokens_total=billable_in,
            )
            if not consumed.get("ok"):
                return _quota_block_response(consumed)

            _cache_dbg("rewrite", True, rewrite_cache_key, {"provider": model_ui})
            return cached

        _cache_dbg("rewrite", False, rewrite_cache_key, {"provider": model_ui})

        # ===============================
        # 2) Length control（仍以原文 text 为基准）
        # ===============================
        a = _estimate_tokens(text)
        min_out = int(math.ceil(0.8 * a)) if a > 0 else 0

        dynamic_max_chars = max(2400, int(len(text) * 1.6))
        dynamic_max_chars = min(dynamic_max_chars, 8000)

        # ===============================
        # 3) 第一次改写（关键：使用 anchored_text）
        #    ✅ 捕获 Groq 429
        # ===============================
        try:
            optimized, _ = geo_rewrite(
                text=anchored_text,
                model_ui=model_ui,
                out_lang=out_lang,
                rewrite_goal=payload.rewrite_goal,
                use_chunk=True,
                max_chars=dynamic_max_chars,
                temperature=0.2,
            )
        except Exception as e:
            if _is_groq_429(e):
                wait_sec = _extract_groq_retry_after_sec(e)
                human = "Groq rate limit reached (429)."
                if wait_sec:
                    human += f" Please retry after ~{wait_sec} seconds."
                else:
                    human += " Please retry later."
                # 保留原始信息，便于你排查（不建议全量返回给用户太长；这里只拼在尾部）
                human += f" (details: {str(e)})"
                return RewriteOut(
                    ok=False,
                    error=human,
                    rewritten=None,
                    original=text,
                    out_lang=out_lang,
                    model_ui=model_ui,
                )
            raise

        optimized = _strip_anchor_echo(optimized or "")
        b = _estimate_tokens(optimized)

        # ===============================
        # 4) 兜底重试（仍使用 anchored_text）
        #    ✅ 捕获 Groq 429（注意：这里也可能触发第二次请求）
        # ===============================
        if a > 0 and b < min_out:
            try:
                optimized2, _ = geo_rewrite(
                    text=anchored_text,
                    model_ui=model_ui,
                    out_lang=out_lang,
                    rewrite_goal=payload.rewrite_goal,
                    use_chunk=False,
                    max_chars=max(dynamic_max_chars, 3600),
                    temperature=0.35,
                )
            except Exception as e:
                if _is_groq_429(e):
                    wait_sec = _extract_groq_retry_after_sec(e)
                    human = "Groq rate limit reached (429) during fallback rewrite."
                    if wait_sec:
                        human += f" Please retry after ~{wait_sec} seconds."
                    else:
                        human += " Please retry later."
                    human += f" (details: {str(e)})"
                    return RewriteOut(
                        ok=False,
                        error=human,
                        rewritten=None,
                        original=text,
                        out_lang=out_lang,
                        model_ui=model_ui,
                    )
                raise

            optimized2 = _strip_anchor_echo(optimized2 or "")
            b2 = _estimate_tokens(optimized2)

            if b2 >= min_out or b2 > b:
                optimized, b = optimized2, b2

        # ===============================
        # 5) Free 用户：成功后计数
        # ===============================
        if tier == "free":
            _rewrite_usage["total"] += 1
            _rewrite_usage["today"] += 1
        
        # ===============================
        # 5.5) Monthly token quota consume (Path 1)
        # ===============================
        # ✅ 计费口径（Rewrite）：只按用户输入计费：title + question + body
        billable_in = (
            _estimate_tokens(payload.article_title or "")
            + _estimate_tokens(payload.user_question or "")
            + _estimate_tokens(text)
        )

        consumed = _supabase_rpc_consume_tokens(
            user_id=ident["user_id"],
            tier=tier,
            kind="rewrite",
            tokens_in=billable_in,
            tokens_out=0,
            tokens_total=billable_in,
        )

        if not consumed.get("ok"):
            return _quota_block_response(consumed)


        # ===============================
        # 6) 返回（original 仍是用户正文）
        # ===============================
        
        out = RewriteOut(
            ok=True,
            rewritten=optimized,
            original=text,
            out_lang=out_lang,
            model_ui=model_ui,
            input_tokens_est=billable_in,
            output_tokens_est=_estimate_tokens(optimized),
        )

        _REWRITE_CACHE.set(rewrite_cache_key, out, ttl_sec=TTL_REWRITE_SEC)
        return out

    except Exception as e:
        # ✅ 最外层兜底也顺手识别一次 429，避免漏网
        if _is_groq_429(e):
            wait_sec = _extract_groq_retry_after_sec(e)
            human = "Groq rate limit reached (429)."
            if wait_sec:
                human += f" Please retry after ~{wait_sec} seconds."
            else:
                human += " Please retry later."
            human += f" (details: {str(e)})"
            print("api_rewrite failed (429):", e)
            return RewriteOut(
                ok=False,
                error=human,
                rewritten=None,
                original=payload.text,
                out_lang=payload.out_lang,
                model_ui=payload.model_ui,
            )

        print("api_rewrite failed:", e)
        return RewriteOut(
            ok=False,
            error=str(e),
            rewritten=None,
            original=payload.text,
            out_lang=payload.out_lang,
            model_ui=payload.model_ui,
        )

@app.get("/api/health/providers")
async def api_health_providers():
    """
    Provider / KeyPool 健康检查（不泄露 key）：
    - pools：每个 provider 的 key 数、可用数、cooldown 秒数
    - runtime：最近一次 ok/error、status_code、retry_after（如可解析）
    - chain：当前 fallback 链
    """
    started_at = time.time()

    # 1) KeyPool 视角（冷却/可用 key）
    pools = {}
    try:
        # 你 providers_groq_gemini.py 里 health() 返回的是 keypool snapshot + provider_chain
        pools = providers_hub.health()
    except Exception as e:
        pools = {"error": str(e)}

    # 2) Runtime 视角（最近一次成功/失败）
    runtime = {}
    try:
        # 你现在实现了 health_runtime()
        runtime = providers_hub.health_runtime()
    except Exception as e:
        runtime = {"error": str(e)}

    # 3) 关键环境变量（非敏感）
    env_view = {
        "GEO_PROVIDER_CHAIN": (os.getenv("GEO_PROVIDER_CHAIN") or "").strip(),
        "GEO_KEY_COOLDOWN_DEFAULT_SEC": (os.getenv("GEO_KEY_COOLDOWN_DEFAULT_SEC") or "").strip(),
        "GEO_KEYPOOL_SHUFFLE": (os.getenv("GEO_KEYPOOL_SHUFFLE") or "").strip(),
    }

    return {
        "ok": True,
        "env": env_view,
        "pools": pools,
        "runtime": runtime,
        "latency_ms": int((time.time() - started_at) * 1000),
    }

