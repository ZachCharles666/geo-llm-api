from datetime import datetime

import stripe

from app.api.schemas.billing import (
    BillingCheckoutIn,
    BillingCheckoutOut,
    BillingUpgradeIn,
    BillingUpgradeOut,
)
from app.services.payments import billing_core

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


class StripeBillingProvider:
    async def checkout(self, payload: BillingCheckoutIn) -> BillingCheckoutOut:
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

    async def upgrade(self, payload: BillingUpgradeIn) -> BillingUpgradeOut:
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
                upcoming = stripe.Invoice.upcoming(subscription=subscription_id)
                return BillingUpgradeOut(
                    ok=True,
                    subscription_id=subscription_id,
                    from_price_id=from_price_id,
                    to_price_id=STRIPE_PRICE_ALPHA_PRO,
                    status=sub.get("status"),
                    invoice_id=upcoming.get("id"),
                )

            updated = stripe.Subscription.modify(
                subscription_id,
                items=[{"id": item_id, "price": STRIPE_PRICE_ALPHA_PRO}],
                proration_behavior="create_prorations",
                metadata={"supabase_user_id": user_id, "tier": "alpha_pro"},
                idempotency_key=idem_key,
            )

            invoice = stripe.Invoice.create(
                customer=updated.get("customer"),
                subscription=subscription_id,
                auto_advance=True,
            )
            invoice_final = stripe.Invoice.finalize_invoice(invoice["id"])
            stripe.Invoice.pay(invoice_final["id"])

            return BillingUpgradeOut(
                ok=True,
                subscription_id=subscription_id,
                from_price_id=from_price_id,
                to_price_id=STRIPE_PRICE_ALPHA_PRO,
                invoice_id=invoice_final.get("id"),
                status=updated.get("status"),
            )

        except Exception as e:
            return BillingUpgradeOut(ok=False, error=str(e))

    async def handle_webhook(self, payload: bytes, signature: str):
        """
        Stripe Webhook（v0 收敛版）
        - 幂等：billing_webhook_events.event_id PK
        - 订阅真相来源：customer.subscription.* / invoice.payment_*
        - checkout.session.completed 仅用于快速拿到 customer/subscription 并写入映射
        """
        _must_env("STRIPE_SECRET_KEY", STRIPE_SECRET_KEY)
        _must_env("STRIPE_WEBHOOK_SECRET", STRIPE_WEBHOOK_SECRET)

        stripe.api_key = STRIPE_SECRET_KEY

        try:
            event = stripe.Webhook.construct_event(
                payload=payload,
                sig_header=signature,
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
            print("[billing] webhook dedup check failed:", e)

        try:
            _sb_webhook_event_insert(event_id=event_id, event_type=event_type)
        except Exception as e:
            print("[billing] webhook dedup insert failed:", e)
            return {"ok": True, "dedup_insert_failed": True, "event": event_type}

        try:
            # =========================================================
            # A) checkout.session.completed（首次订阅）
            # =========================================================
            if event_type == "checkout.session.completed":
                session = event["data"]["object"]
                user_id = (session.get("client_reference_id") or "").strip()
                customer_id = (session.get("customer") or "").strip()
                subscription_id = (session.get("subscription") or "").strip()
                customer_email = (session.get("customer_details") or {}).get("email")

                if user_id and customer_id:
                    _sb_upsert_customer(user_id=user_id, stripe_customer_id=customer_id, email=customer_email)

                if user_id and customer_id and subscription_id:
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
                    sub_full = stripe.Subscription.retrieve(subscription_id, expand=["items.data.price"])
                    price_id = _stripe_get_subscription_primary_price_id(sub_full)
                    if not status:
                        status = (sub_full.get("status") or "").strip().lower()
                    if not period_end_iso:
                        period_end_iso = _iso_from_unix_ts(sub_full.get("current_period_end"))
                except Exception as e:
                    print("[billing] subscription retrieve failed:", e)

                meta = sub.get("metadata") or {}
                user_id = (meta.get("supabase_user_id") or "").strip()

                if not user_id and customer_id:
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

                return {"ok": True, "event": event_type, "user_id": user_id, "status": status, "period_end": period_end_iso}

            # =========================================================
            # C) invoice.payment_succeeded / failed
            # =========================================================
            if event_type in ("invoice.payment_succeeded", "invoice.payment_failed"):
                inv = event["data"]["object"]

                inv_user_id, inv_tier = _inv_extract_user_id_and_tier(inv)
                inv_status = (inv.get("status") or "").strip().lower()
                inv_price_id = _tier_from_price_id(inv_tier) if inv_tier else ""
                paid = inv_status == "paid"

                try:
                    if inv_price_id and inv_user_id and paid:
                        _profile_sync_tier_after_payment(user_id=inv_user_id, tier=inv_tier)
                        _quota_sync_after_payment(user_id=inv_user_id, tier=inv_tier)
                except Exception as e:
                    print("[billing] invoice post-process failed:", e)

                return {"ok": True, "event": event_type, "paid": paid, "tier": inv_tier, "user_id": inv_user_id}

            return {"ok": True, "event": event_type, "ignored": True}

        except Exception as e:
            print("[billing] webhook handler failed:", e)
            return {"ok": False, "error": str(e)}
