from typing import Any

import requests

from app.infra.supabase import client


def _supabase_get_quota_row(user_id: str, yyyymm: str):
    supabase_url = client.SUPABASE_URL.rstrip("/")
    url = f"{supabase_url}/rest/v1/quota_monthly"
    params = {
        "select": "user_id,tier,yyyymm,tokens_limit,tokens_used,updated_at",
        "user_id": f"eq.{user_id}",
        "yyyymm": f"eq.{yyyymm}",
        "limit": "1",
    }
    r = requests.get(url, headers=client.supabase_admin_headers(), params=params, timeout=15)
    r.raise_for_status()
    rows = r.json() or []
    return rows[0] if rows else None


def _supabase_upsert_quota_row(
    *,
    user_id: str,
    tier: str,
    yyyymm: str,
    tokens_limit: int,
    reset_used: bool = False,
):
    url = client.sb_rest_url("quota_monthly") + "?on_conflict=user_id,yyyymm"

    payload = {
        "user_id": user_id,
        "yyyymm": yyyymm,
        "tier": (tier or "free").strip().lower(),
        "tokens_limit": int(tokens_limit),
        "updated_at": client.utc_now_iso(),
    }
    if reset_used:
        payload["tokens_used"] = 0

    return client.sb_post_json(
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
    if not user_id or not yyyymm:
        return None

    url = client.sb_rest_url("quota_monthly")
    patch_url = f"{url}?user_id=eq.{user_id}&yyyymm=eq.{yyyymm}"

    payload = {
        "tier": (tier or "free").strip().lower(),
        "tokens_limit": int(tokens_limit),
        "updated_at": client.utc_now_iso(),
    }
    if reset_used:
        payload["tokens_used"] = 0

    return client.sb_patch_json(
        patch_url,
        payload,
        prefer="return=representation",
    )


def _supabase_rpc_consume_tokens(
    user_id: str,
    tier: str,
    kind: str,
    tokens_in: int,
    tokens_out: int,
    tokens_total: int,
    tokens_limit_default: int | None = None,
) -> dict:
    client.ensure_supabase_env_for_db()

    url = f"{client.SUPABASE_URL}/rest/v1/rpc/consume_tokens"
    headers = {
        "apikey": client.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {client.SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "p_user_id": user_id,
        "p_kind": kind,
        "p_tokens_in": int(tokens_in or 0),
        "p_tokens_out": int(tokens_out or 0),
        "p_tokens_total": int(tokens_total or 0),
    }

    if tokens_limit_default is not None:
        payload["p_tokens_limit_default"] = int(tokens_limit_default)

    r = requests.post(url, headers=headers, json=payload, timeout=15)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase RPC consume_tokens failed: {r.status_code} {r.text}")

    return r.json() or {}
