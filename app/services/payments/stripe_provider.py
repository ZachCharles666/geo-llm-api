from datetime import datetime
from typing import Any

import stripe
from fastapi import Request

from app.services.payments import billing_core
from app.services.payments.provider import PaymentProvider


class StripePaymentProvider(PaymentProvider):
    """Stripe-backed payment provider."""

    def _get_payload_value(self, payload: Any, key: str) -> Any:
        if isinstance(payload, dict):
            return payload.get(key)
        return getattr(payload, key, None)

    async def create_checkout_session(self, payload: Any) -> dict:
        try:
            billing_core._must_env("STRIPE_SECRET_KEY", billing_core.STRIPE_SECRET_KEY)
            billing_core._must_env("STRIPE_PRICE_ALPHA_BASE", billing_core.STRIPE_PRICE_ALPHA_BASE)
            billing_core._must_env("STRIPE_PRICE_ALPHA_PRO", billing_core.STRIPE_PRICE_ALPHA_PRO)

            stripe.api_key = billing_core.STRIPE_SECRET_KEY

            tier = (self._get_payload_value(payload, "tier") or "").strip().lower()
            if tier not in ("alpha_base", "alpha_pro"):
                return {"ok": False, "error": f"Invalid tier: {tier}", "checkout_url": None}

            user_id = (self._get_payload_value(payload, "user_id") or "").strip()
            if not user_id:
                return {"ok": False, "error": "Missing user_id", "checkout_url": None}

            has_sub, sub_row = billing_core._sb_user_has_effective_subscription(user_id)
            if has_sub:
                status = (sub_row or {}).get("status")
                return {
                    "ok": False,
                    "error": f"Subscription already exists (status={status}). Use Upgrade instead of purchasing again.",
                    "checkout_url": None,
                }

            price_id = billing_core.STRIPE_PRICE_ALPHA_BASE if tier == "alpha_base" else billing_core.STRIPE_PRICE_ALPHA_PRO

            success_url = f"{billing_core.FRONTEND_ORIGIN}/alpha-access?checkout=success"
            cancel_url = f"{billing_core.FRONTEND_ORIGIN}/alpha-access?checkout=cancel"

            email = (self._get_payload_value(payload, "email") or "").strip() or None
            existing_customer_id = billing_core._sb_get_customer_id_by_user(user_id)

            minute_bucket = datetime.utcnow().strftime("%Y%m%d%H%M")
            idem_key = f"checkout:{user_id}:{tier}:{price_id}:{minute_bucket}"

            session_create_payload: dict[str, Any] = {
                "mode": "subscription",
                "line_items": [{"price": price_id, "quantity": 1}],
                "success_url": success_url,
                "cancel_url": cancel_url,
                "client_reference_id": user_id,
                "allow_promotion_codes": False,
                "metadata": {
                    "supabase_user_id": user_id,
                    "tier": tier,
                },
                "subscription_data": {
                    "metadata": {
                        "supabase_user_id": user_id,
                        "tier": tier,
                    }
                },
                "customer_email": email,
                "customer": existing_customer_id,
            }

            session = stripe.checkout.Session.create(idempotency_key=idem_key, **session_create_payload)

            try:
                if session and session.get("customer"):
                    billing_core._sb_upsert_customer(
                        user_id=user_id,
                        stripe_customer_id=session["customer"],
                        email=email,
                    )
            except Exception:
                pass

            return {
                "ok": True,
                "error": None,
                "checkout_url": session.get("url") if session else None,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "checkout_url": None}

    async def upgrade_subscription(self, payload: Any) -> dict:
        try:
            billing_core._must_env("STRIPE_SECRET_KEY", billing_core.STRIPE_SECRET_KEY)
            billing_core._must_env("STRIPE_PRICE_ALPHA_PRO", billing_core.STRIPE_PRICE_ALPHA_PRO)

            stripe.api_key = billing_core.STRIPE_SECRET_KEY

            user_id = (self._get_payload_value(payload, "user_id") or "").strip()
            if not user_id:
                return {"ok": False, "error": "Missing user_id"}

            has_sub, sub_row = billing_core._sb_user_has_effective_subscription(user_id)
            if not has_sub or not sub_row:
                return {"ok": False, "error": "No existing subscription. Please subscribe first."}

            subscription_id = (sub_row.get("stripe_subscription_id") or "").strip()
            if not subscription_id:
                return {"ok": False, "error": "Missing stripe_subscription_id in billing_subscriptions."}

            sub = stripe.Subscription.retrieve(subscription_id, expand=["items.data.price"])
            items = (sub.get("items") or {}).get("data") or []
            if not items:
                return {"ok": False, "error": "Subscription has no items."}

            item_id = (items[0].get("id") or "").strip()
            from_price_id = billing_core._stripe_get_subscription_primary_price_id(sub)
            to_price_id = billing_core.STRIPE_PRICE_ALPHA_PRO

            if not item_id:
                return {"ok": False, "error": "Subscription item id missing."}

            upgrade = stripe.Subscription.modify(
                subscription_id,
                items=[{"id": item_id, "price": to_price_id}],
                payment_behavior="create_invoice",
                proration_behavior="create_prorations",
                payment_settings={"save_default_payment_method": "on_subscription"},
            )

            invoice_preview = None
            if getattr(payload, "dry_run", False) or (isinstance(payload, dict) and payload.get("dry_run")):
                invoice_preview = stripe.Invoice.upcoming(subscription=subscription_id)
            else:
                try:
                    invoice_preview = stripe.Invoice.create(customer=upgrade.get("customer"))
                except Exception:
                    invoice_preview = None

            invoice_id = ""
            status = ""
            if invoice_preview:
                invoice_id = (invoice_preview.get("id") or "").strip()
                status = (invoice_preview.get("status") or "").strip().lower()

            customer_id = (upgrade.get("customer") or "").strip()
            if user_id and customer_id:
                try:
                    billing_core._sb_upsert_customer(user_id=user_id, stripe_customer_id=customer_id, email=None)
                except Exception:
                    pass

            if upgrade:
                try:
                    billing_core._sb_upsert_subscription(
                        stripe_subscription_id=subscription_id,
                        user_id=user_id,
                        stripe_customer_id=customer_id,
                        price_id=to_price_id,
                        status=(upgrade.get("status") or "").strip().lower(),
                        current_period_end_iso=billing_core._iso_from_unix_ts(upgrade.get("current_period_end")),
                    )
                except Exception:
                    pass

            return {
                "ok": True,
                "subscription_id": subscription_id,
                "from_price_id": from_price_id,
                "to_price_id": to_price_id,
                "invoice_id": invoice_id or None,
                "status": status or None,
                "error": None,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def handle_webhook(self, request: Request) -> dict:
        billing_core._must_env("STRIPE_SECRET_KEY", billing_core.STRIPE_SECRET_KEY)
        billing_core._must_env("STRIPE_WEBHOOK_SECRET", billing_core.STRIPE_WEBHOOK_SECRET)

        stripe.api_key = billing_core.STRIPE_SECRET_KEY

        payload = await request.body()
        sig_header = request.headers.get("stripe-signature", "")
        try:
            event = stripe.Webhook.construct_event(
                payload=payload,
                sig_header=sig_header,
                secret=billing_core.STRIPE_WEBHOOK_SECRET,
            )
        except Exception as exc:
            return {"ok": False, "error": f"Invalid payload or signature: {exc}"}

        event_type = (event.get("type") or "").strip()
        event_id = (event.get("id") or "").strip()
        if not event_type or not event_id:
            return {"ok": False, "error": "Invalid event"}

        try:
            if billing_core._sb_webhook_event_exists(event_id):
                return {"ok": True, "ignored": "duplicate"}
        except Exception as exc:
            return {"ok": False, "error": f"Webhook dedup check failed: {exc}"}

        try:
            billing_core._sb_webhook_event_insert(
                event_id=event_id,
                event_type=event_type,
                user_id=None,
                session_id=None,
            )
        except Exception:
            pass

        if event_type == "checkout.session.completed":
            session = event["data"]["object"]
            user_id = (session.get("client_reference_id") or "").strip()
            customer_id = (session.get("customer") or "").strip()
            subscription_id = (session.get("subscription") or "").strip()
            price_id = ""
            status = ""
            current_period_end_iso = None

            sub_full = None
            if subscription_id:
                try:
                    sub_full = stripe.Subscription.retrieve(subscription_id, expand=["items.data.price"])
                    price_id = billing_core._stripe_get_subscription_primary_price_id(sub_full)
                    status = (sub_full.get("status") or "").strip().lower()
                    current_period_end_iso = billing_core._iso_from_unix_ts(sub_full.get("current_period_end"))
                except Exception as exc:
                    print("[billing] subscription retrieve failed:", exc)
                    sub_full = None

            tier = billing_core._tier_from_price_id(price_id)

            if user_id and customer_id:
                try:
                    billing_core._sb_upsert_customer(user_id=user_id, stripe_customer_id=customer_id, email=None)
                except Exception:
                    pass

            if user_id and customer_id and subscription_id:
                try:
                    billing_core._sb_upsert_subscription(
                        stripe_subscription_id=subscription_id,
                        user_id=user_id,
                        stripe_customer_id=customer_id,
                        price_id=price_id,
                        status=status,
                        current_period_end_iso=current_period_end_iso,
                    )
                except Exception:
                    pass

            if user_id:
                try:
                    billing_core._supabase_upsert_profile_tier(user_id=user_id, new_tier=tier)
                except Exception as exc:
                    print("[billing] profiles tier sync failed:", exc)

                try:
                    billing_core._quota_ensure_row(user_id=user_id, tier=tier, yyyymm=billing_core._yyyymm_utc_now())
                except Exception as exc:
                    print("[billing] quota ensure failed:", exc)

            return {
                "ok": True,
                "event": event_type,
                "status": status,
            }

        if event_type in ("invoice.payment_succeeded", "invoice.payment_failed"):
            inv = event["data"]["object"]
            subscription_id = (inv.get("subscription") or "").strip()
            customer_id = (inv.get("customer") or "").strip()

            inv_user_id, inv_tier = billing_core._inv_extract_user_id_and_tier(inv)
            user_id = (inv_user_id or "").strip()
            tier_from_inv = (inv_tier or "").strip().lower()

            price_id = ""
            status = ""
            sub_user_id = ""
            sub_tier = ""
            period_end_iso = None

            sub_full = None
            if subscription_id:
                try:
                    sub_full = stripe.Subscription.retrieve(subscription_id, expand=["items.data.price"])
                    price_id = billing_core._stripe_get_subscription_primary_price_id(sub_full)
                    status = (sub_full.get("status") or "").strip().lower()

                    meta = sub_full.get("metadata") or {}
                    sub_user_id = (meta.get("supabase_user_id") or "").strip()
                    sub_tier = (meta.get("tier") or "").strip().lower()

                    period_end_iso = billing_core._iso_from_unix_ts(sub_full.get("current_period_end"))
                except Exception as exc:
                    print("[billing] subscription retrieve failed:", exc)
                    sub_full = None

            if not user_id and sub_user_id:
                user_id = sub_user_id

            if not user_id and customer_id:
                try:
                    url = billing_core._sb_rest_url(billing_core.BILLING_TABLE_CUSTOMERS)
                    rows = billing_core._sb_get_json(
                        url,
                        params={
                            "select": "user_id",
                            "stripe_customer_id": f"eq.{customer_id}",
                            "limit": "1",
                        },
                    )
                    if rows:
                        user_id = (rows[0].get("user_id") or "").strip()
                except Exception as exc:
                    print("[billing] reverse lookup customer->user failed:", exc)

            effective_tier = tier_from_inv or sub_tier or billing_core._tier_from_price_id(price_id)
            effective_tier = (effective_tier or "free").strip().lower()

            if user_id and customer_id:
                try:
                    billing_core._sb_upsert_customer(user_id=user_id, stripe_customer_id=customer_id, email=None)
                except Exception:
                    pass

            if user_id and customer_id and subscription_id:
                try:
                    billing_core._sb_upsert_subscription(
                        stripe_subscription_id=subscription_id,
                        user_id=user_id,
                        stripe_customer_id=customer_id,
                        price_id=price_id,
                        status=status,
                        current_period_end_iso=period_end_iso,
                    )
                except Exception:
                    pass

            if event_type == "invoice.payment_succeeded" and user_id:
                try:
                    billing_core._profile_sync_tier_after_payment(user_id=user_id, tier=effective_tier)
                except Exception as exc:
                    print("[billing] profiles tier sync FAILED:", exc)

                try:
                    billing_core._quota_sync_after_payment(
                        user_id=user_id,
                        tier=effective_tier,
                        yyyymm=billing_core._yyyymm_utc_now(),
                        reset_used=True,
                    )
                except Exception as exc:
                    print("[billing] quota sync FAILED:", exc)

            return {"ok": True, "event": event_type}

        return {"ok": True, "ignored": event_type}
