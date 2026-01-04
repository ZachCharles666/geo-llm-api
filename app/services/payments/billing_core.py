import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()


# 月度 token 限额（可用 env 覆盖）
TIER_MONTHLY_TOKENS_FREE = int(os.getenv("TIER_MONTHLY_TOKENS_FREE", "20000"))
TIER_MONTHLY_TOKENS_ALPHA_BASE = int(os.getenv("TIER_MONTHLY_TOKENS_ALPHA_BASE", "200000"))
TIER_MONTHLY_TOKENS_ALPHA_PRO = int(os.getenv("TIER_MONTHLY_TOKENS_ALPHA_PRO", "500000"))


def _require_env(name: str, value: str):
    if not value:
        raise RuntimeError(f"Missing required env: {name}")


def _ensure_supabase_env_for_auth():
    _require_env("SUPABASE_URL", SUPABASE_URL)
    _require_env("SUPABASE_ANON_KEY", SUPABASE_ANON_KEY)


def _ensure_supabase_env_for_db():
    _require_env("SUPABASE_URL", SUPABASE_URL)
    _require_env("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY)


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
    reset_used: bool = False,  # 支付成功建议设为 True，或者按需保持
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
        ret = _supabase_upsert_quota_row(
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
        cur_limit = int((row.get("tokens_limit") or 0))
        if cur_tier != tier or cur_limit != default_limit:
            # 这里不强制 tokens_used=0；如果你希望同步清零，再打开注释
            _supabase_upsert_quota_row(
                user_id=user_id,
                tier=tier,
                yyyymm=yyyymm,
                tokens_limit=default_limit,
                # reset_used=True,
            )
    except Exception as e:
        print("[billing] quota correction failed:", e)

    return row


STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()

STRIPE_PRICE_ALPHA_BASE = os.getenv("STRIPE_PRICE_ALPHA_BASE", "").strip()
STRIPE_PRICE_ALPHA_PRO = os.getenv("STRIPE_PRICE_ALPHA_PRO", "").strip()

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").strip()


def _must_env(name: str, value: str):
    if not value:
        raise RuntimeError(f"Missing required env: {name}")


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


def _tier_from_price_id(price_id: str) -> str:
    if price_id == STRIPE_PRICE_ALPHA_BASE:
        return "alpha_base"
    if price_id == STRIPE_PRICE_ALPHA_PRO:
        return "alpha_pro"
    # default safe fallback
    return "free"


def _supabase_get_profile_tier(user_id: str) -> str:
    user_id = (user_id or "").strip()
    if not user_id:
        return ""

    _ensure_supabase_env_for_db()

    url = _sb_rest_url("profiles")
    params = {
        "select": "id,tier",
        "id": f"eq.{user_id}",
        "limit": 1,
    }

    r = requests.get(
        url,
        headers={
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        },
        params=params,
        timeout=15,
    )
    r.raise_for_status()
    rows = r.json() or []

    if not rows:
        return ""

    # 若 profiles 里 tier 为空，也允许后续逻辑走到“默认 free”
    return (rows[0].get("tier") or "").lower().strip()


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
    subs = _sb_get_subscriptions_by_user(user_id, limit=10)
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
