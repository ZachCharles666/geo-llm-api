import os
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()


def _require_env(name: str, value: str):
    if not value:
        raise RuntimeError(f"Missing required env: {name}")


def ensure_supabase_env_for_auth():
    _require_env("SUPABASE_URL", SUPABASE_URL)
    _require_env("SUPABASE_ANON_KEY", SUPABASE_ANON_KEY)


def ensure_supabase_env_for_db():
    _require_env("SUPABASE_URL", SUPABASE_URL)
    _require_env("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY)


def supabase_admin_headers():
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    return {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
    }


def supabase_anon_headers():
    supabase_key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    return {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }


def sb_rest_url(table: str) -> str:
    ensure_supabase_env_for_db()
    base = SUPABASE_URL.rstrip("/")
    return f"{base}/rest/v1/{table}"


def sb_rpc_url(fn: str) -> str:
    ensure_supabase_env_for_db()
    base = SUPABASE_URL.rstrip("/")
    return f"{base}/rest/v1/rpc/{fn}"


def sb_get_json(url: str, params: dict) -> list:
    r = requests.get(url, headers=supabase_admin_headers(), params=params, timeout=15)

    if r.status_code >= 300:
        ctype = (r.headers.get("content-type") or "").lower()
        body = (r.text or "")
        if len(body) > 800:
            body = body[:800] + "...(truncated)"
        raise RuntimeError(
            f"Supabase GET failed: {r.status_code} {r.reason} "
            f"(content-type={ctype}) body={body}"
        )

    if r.status_code == 204 or not (r.content and r.content.strip()):
        return []

    ctype = (r.headers.get("content-type") or "").lower()
    if "application/json" not in ctype and not ctype.endswith("+json"):
        return []

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


def sb_post_json(url: str, payload: Any, prefer: str = "return=representation") -> list:
    headers = supabase_admin_headers()
    headers["Prefer"] = prefer

    r = requests.post(url, headers=headers, json=payload, timeout=15)

    if r.status_code >= 300:
        ctype = (r.headers.get("content-type") or "").lower()
        body = (r.text or "")
        if len(body) > 800:
            body = body[:800] + "...(truncated)"
        raise RuntimeError(
            f"Supabase POST failed: {r.status_code} {r.reason} "
            f"(content-type={ctype}) body={body}"
        )

    if r.status_code == 204 or not (r.content and r.content.strip()):
        return []

    ctype = (r.headers.get("content-type") or "").lower()
    if "application/json" not in ctype and not ctype.endswith("+json"):
        return []

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


def sb_patch_json(url: str, payload: dict, prefer: str | None = None):
    headers = supabase_admin_headers()
    if prefer:
        headers = {**headers, "Prefer": prefer}

    r = requests.patch(url, headers=headers, json=payload, timeout=15)

    if r.status_code < 200 or r.status_code >= 300:
        print("[sb][patch] FAILED", {"url": url, "status": r.status_code, "body": r.text})
        raise RuntimeError(f"Supabase PATCH failed: {r.status_code} {r.text}")

    try:
        if r.text and r.text.strip():
            return r.json()
    except Exception:
        print("[sb][patch] JSON decode failed", {"url": url, "status": r.status_code, "body": r.text})
        raise
    return None


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()
