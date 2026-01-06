import os
from datetime import datetime, timezone

from app.infra.supabase import quota_repo

TIER_MONTHLY_TOKENS_FREE = int(os.getenv("TIER_MONTHLY_TOKENS_FREE", "20000"))
TIER_MONTHLY_TOKENS_ALPHA_BASE = int(os.getenv("TIER_MONTHLY_TOKENS_ALPHA_BASE", "200000"))
TIER_MONTHLY_TOKENS_ALPHA_PRO = int(os.getenv("TIER_MONTHLY_TOKENS_ALPHA_PRO", "500000"))


def tier_monthly_limit(tier: str) -> int:
    t = (tier or "free").lower().strip()
    if t == "alpha_base":
        return TIER_MONTHLY_TOKENS_ALPHA_BASE
    if t == "alpha_pro":
        return TIER_MONTHLY_TOKENS_ALPHA_PRO
    return TIER_MONTHLY_TOKENS_FREE


def yyyymm_utc_now() -> str:
    return datetime.utcnow().strftime("%Y%m")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def supabase_get_quota_row(user_id: str, yyyymm: str):
    return quota_repo._supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)


def supabase_upsert_quota_row(
    *, user_id: str, tier: str, yyyymm: str, tokens_limit: int, reset_used: bool = False
):
    return quota_repo._supabase_upsert_quota_row(
        user_id=user_id,
        tier=tier,
        yyyymm=yyyymm,
        tokens_limit=tokens_limit,
        reset_used=reset_used,
    )


def supabase_patch_quota_row(
    *, user_id: str, yyyymm: str, tier: str, tokens_limit: int, reset_used: bool = False
):
    return quota_repo._supabase_patch_quota_row(
        user_id=user_id,
        yyyymm=yyyymm,
        tier=tier,
        tokens_limit=tokens_limit,
        reset_used=reset_used,
    )


def quota_sync_after_payment(*, user_id: str, tier: str, yyyymm: str | None = None, reset_used: bool = True):
    if not user_id:
        return None

    tier = (tier or "free").strip().lower()
    yyyymm = yyyymm or yyyymm_utc_now()
    tokens_limit = int(tier_monthly_limit(tier))

    try:
        supabase_upsert_quota_row(
            user_id=user_id,
            tier=tier,
            yyyymm=yyyymm,
            tokens_limit=tokens_limit,
            reset_used=reset_used,
        )
    except Exception as e:
        raise RuntimeError(f"[quota_sync] upsert failed: {e}")

    row = supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)
    cur_tier = ((row or {}).get("tier") or "").strip().lower()
    cur_limit = int((row or {}).get("tokens_limit") or 0)

    if row and cur_tier == tier and cur_limit == tokens_limit:
        return row

    try:
        supabase_patch_quota_row(
            user_id=user_id,
            yyyymm=yyyymm,
            tier=tier,
            tokens_limit=tokens_limit,
            reset_used=reset_used,
        )
    except Exception as e:
        raise RuntimeError(f"[quota_sync] patch failed: {e}")

    row2 = supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)
    cur_tier2 = ((row2 or {}).get("tier") or "").strip().lower()
    cur_limit2 = int((row2 or {}).get("tokens_limit") or 0)

    if row2 and cur_tier2 == tier and cur_limit2 == tokens_limit:
        return row2

    raise RuntimeError(
        f"[quota_sync] verify failed after patch. "
        f"expect(tier={tier},limit={tokens_limit}) "
        f"got(row={row2})"
    )


def quota_ensure_row(user_id: str, tier: str, yyyymm: str):
    if not user_id:
        return None

    tier = (tier or "free").lower().strip()
    yyyymm = yyyymm or yyyymm_utc_now()
    default_limit = int(tier_monthly_limit(tier))

    row = supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)
    if row is None:
        ret = supabase_upsert_quota_row(
            user_id=user_id,
            tier=tier,
            yyyymm=yyyymm,
            tokens_limit=default_limit,
        )
        try:
            verify = supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)
            print("[billing] quota verify:", {"user_id": user_id, "yyyymm": yyyymm, "row": verify})
        except Exception as e:
            print("[billing] quota verify failed:", e)
        return ret

    try:
        cur_tier = (row.get("tier") or "").lower().strip() or "free"
        cur_limit = int(row.get("tokens_limit") or 0)
    except Exception:
        cur_tier, cur_limit = "free", 0

    if cur_tier != tier or cur_limit != default_limit:
        ret = supabase_upsert_quota_row(
            user_id=user_id,
            tier=tier,
            yyyymm=yyyymm,
            tokens_limit=default_limit,
        )
        try:
            verify = supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)
            print("[billing] quota verify:", {"user_id": user_id, "yyyymm": yyyymm, "row": verify})
        except Exception as e:
            print("[billing] quota verify failed:", e)
        return ret

    return row


def supabase_rpc_consume_tokens(
    user_id: str,
    tier: str,
    kind: str,
    tokens_in: int,
    tokens_out: int,
    tokens_total: int,
) -> dict:
    default_limit = tier_monthly_limit(tier)
    payload = quota_repo._supabase_rpc_consume_tokens(
        user_id=user_id,
        tier=tier,
        kind=kind,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_total=tokens_total,
        tokens_limit_default=default_limit,
    )
    if isinstance(payload, dict) and "p_tokens_limit_default" not in payload:
        payload["p_tokens_limit_default"] = default_limit
    return payload
