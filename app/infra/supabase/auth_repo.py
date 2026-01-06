from typing import Optional

import requests

from app.infra.supabase import client


def _supabase_get_user_id(access_token: str) -> Optional[str]:
    client.ensure_supabase_env_for_auth()
    url = f"{client.SUPABASE_URL}/auth/v1/user"
    headers = {
        "apikey": client.SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
    }
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json() or {}
    user = data.get("user") or data
    return (user.get("id") or "").strip() or None


def _supabase_auth_get_user(access_token: str) -> dict | None:
    client.ensure_supabase_env_for_auth()
    url = f"{client.SUPABASE_URL}/auth/v1/user"
    headers = {
        "apikey": client.SUPABASE_ANON_KEY,
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
    client.ensure_supabase_env_for_db()
    url = f"{client.SUPABASE_URL}/rest/v1/profiles"
    headers = {
        "apikey": client.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {client.SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    params = {
        "select": "tier",
        "id": f"eq.{user_id}",
        "limit": "1",
    }
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code != 200:
        return "free"
    rows = r.json() or []
    if not rows:
        return "free"
    tier = (rows[0].get("tier") or "free").strip().lower()
    if tier not in ("free", "alpha_base", "alpha_pro"):
        return "free"
    return tier


def _supabase_upsert_profile_row(user_id: str, email: str | None = None, tier: str | None = None) -> dict | None:
    user_id = (user_id or "").strip()
    if not user_id:
        return None

    payload = {"id": user_id}
    if email is not None:
        payload["email"] = email
    if tier is not None:
        payload["tier"] = tier

    url = client.sb_rest_url("profiles")
    params = "on_conflict=id"
    full_url = f"{url}?{params}"
    headers_prefer = "resolution=merge-duplicates,return=representation"
    rows = client.sb_post_json(full_url, payload, prefer=headers_prefer)
    if rows and isinstance(rows, list):
        return rows[0]
    return None
