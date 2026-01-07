from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping

from fastapi import Request

from app.services.payments.models import NormalizedBillingEvent
from app.services.payments.paddle_mapping import map_paddle_event_to_action
from app.services.payments.provider import PaymentProvider


def _verify_paddle_signature(headers: Mapping[str, str], raw_body: bytes) -> tuple[bool, str | None]:
    secret = os.getenv("PADDLE_WEBHOOK_SECRET", "").strip()
    if not secret:
        return False, "missing paddle webhook secret"

    signature = headers.get("paddle-signature") or headers.get("Paddle-Signature")
    if not signature:
        return False, "missing paddle-signature header"

    # TODO(Phase 2C): implement HMAC verification using PADDLE_WEBHOOK_SECRET and raw_body
    _ = (raw_body, signature, secret)
    return True, None


def _parse_paddle_event(raw_body: bytes) -> NormalizedBillingEvent:
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    # Best-effort: accept id/event_id and type/event_type
    event_id = str(payload.get("event_id") or payload.get("id") or "").strip()
    if not event_id:
        raise ValueError("Missing event_id (expected event_id or id)")

    event_type = str(payload.get("event_type") or payload.get("type") or "").strip()
    if not event_type:
        raise ValueError("Missing event_type (expected event_type or type)")

    occurred_at = payload.get("occurred_at") or payload.get("event_time")
    if not occurred_at:
        # timezone-aware UTC "now" as fallback (no deprecation warning)
        occurred_at = datetime.now(timezone.utc).isoformat()

    # TODO(Phase 2C): extract customer_id/subscription_id/price_id/tier/status once Paddle payload shape is finalized.
    return NormalizedBillingEvent(
        provider="paddle",
        event_id=event_id,
        event_type=event_type,
        occurred_at=occurred_at,
        customer_id=None,
        subscription_id=None,
        price_id=None,
        tier=None,
        status=None,
        raw=payload,
    )


def _paddle_event_exists(event_id: str) -> bool:
    _ = event_id
    # TODO(Phase 2C): implement webhook dedup storage (Supabase table)
    return False


def _paddle_event_insert(event: NormalizedBillingEvent) -> None:
    _ = event
    # TODO(Phase 2C): implement webhook dedup insert storage
    return None


def _dispatch_billing_action(action_payload: dict[str, Any]) -> None:
    _ = action_payload
    # TODO(Phase 2C): call billing_core sync entrypoint for Paddle events
    return None


class PaddlePaymentProvider(PaymentProvider):
    """Controlled Paddle implementation for Phase 2A/2B smoke + scaffolding."""

    def _not_configured(self) -> dict[str, Any]:
        return {"ok": False, "error": "Paddle provider not configured"}

    async def create_checkout_session(self, payload: Any) -> dict[str, Any]:
        response = self._not_configured()
        response["checkout_url"] = None
        return response

    async def upgrade_subscription(self, payload: Any) -> dict[str, Any]:
        return self._not_configured()

    async def handle_webhook(self, request: Request) -> dict[str, Any]:
        raw_body = await request.body()

        ok, error = _verify_paddle_signature(request.headers, raw_body)
        if not ok:
            return {"ok": False, "error": error or "invalid paddle webhook signature"}

        try:
            event = _parse_paddle_event(raw_body)
        except Exception as exc:
            return {"ok": False, "error": f"Invalid event payload: {exc}"}

        # Idempotency scaffold (Phase 2C)
        try:
            if _paddle_event_exists(event.event_id):
                return {"ok": True, "status": "duplicate_ignored"}
        except Exception as exc:
            return {"ok": False, "error": f"Webhook dedup check failed: {exc}"}

        try:
            _paddle_event_insert(event)
        except Exception:
            # Do not fail the webhook if we cannot persist yet (Phase 2C will harden)
            pass

        action_payload = map_paddle_event_to_action(event) or {}

        # If we don't recognize the event yet, accept but mark as unhandled.
        if not action_payload:
            return {"ok": True, "status": "unhandled_event", "event": event.event_type}

        _dispatch_billing_action(action_payload)

        # Phase 2B: accept but indicate not fully processed until billing_core is wired in Phase 2C.
        return {"ok": True, "status": "accepted_unprocessed", "event": event.event_type, "action": action_payload.get("action")}
