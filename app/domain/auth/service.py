import time
from typing import Optional

from fastapi import HTTPException, Request

from app.domain.auth.models import AuthContext, Identity
from app.infra.supabase import auth_repo


_TIER_CACHE: dict[str, tuple[str, float]] = {}
_TIER_CACHE_TTL_SEC = 60.0


def get_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def supabase_get_user_id(access_token: str) -> str | None:
    return auth_repo._supabase_get_user_id(access_token)


def supabase_auth_get_user(access_token: str) -> dict | None:
    return auth_repo._supabase_auth_get_user(access_token)


def supabase_get_profile_tier(user_id: str) -> str:
    return auth_repo._supabase_get_profile_tier(user_id)


def supabase_upsert_profile_row(user_id: str, email: str | None = None, tier: str | None = None):
    return auth_repo._supabase_upsert_profile_row(user_id=user_id, email=email, tier=tier)


def resolve_user_and_tier(request: Request) -> dict:
    auth_header = (request.headers.get("authorization") or request.headers.get("Authorization") or "").strip()
    if not auth_header:
        return {"ok": True, "is_authed": False, "user_id": None, "tier": "free"}

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return {"ok": True, "is_authed": False, "user_id": None, "tier": "free"}

    jwt = parts[1].strip()
    if not jwt:
        return {"ok": True, "is_authed": False, "user_id": None, "tier": "free"}

    try:
        user = supabase_auth_get_user(jwt)
    except Exception as e:
        print("[auth] supabase auth get user failed:", e)
        return {"ok": True, "is_authed": False, "user_id": None, "tier": "free"}

    if not user:
        return {"ok": True, "is_authed": False, "user_id": None, "tier": "free"}

    user_id = (user.get("id") or "").strip()
    email = user.get("email") or None
    if not user_id:
        return {"ok": True, "is_authed": False, "user_id": None, "tier": "free"}

    try:
        supabase_upsert_profile_row(user_id=user_id, email=email, tier="free")
    except Exception as e:
        print("[auth] profiles upsert failed:", e)

    tier = "free"
    try:
        tier = supabase_get_profile_tier(user_id)
    except Exception as e:
        print("[auth] profiles tier read failed:", e)
        tier = "free"

    return {"ok": True, "is_authed": True, "user_id": user_id, "tier": tier}


def require_authed_user(request: Request) -> dict:
    token = get_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")

    try:
        user_id = supabase_get_user_id(token)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Supabase token")

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired Supabase token")

    try:
        supabase_upsert_profile_row(user_id=user_id, email=None, tier="free")
    except Exception as e:
        print("[auth] profiles ensure row failed:", e)

    now = time.time()
    cached: Optional[tuple[str, float]] = _TIER_CACHE.get(user_id)
    if cached and cached[1] > now:
        return {"user_id": user_id, "tier": cached[0]}

    tier = supabase_get_profile_tier(user_id)
    _TIER_CACHE[user_id] = (tier, now + _TIER_CACHE_TTL_SEC)
    return {"user_id": user_id, "tier": tier}
