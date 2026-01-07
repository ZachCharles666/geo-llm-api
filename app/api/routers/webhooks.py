import os

from fastapi import APIRouter, Request

from app.services.payments import get_payment_provider
from app.services.payments.providers import get_billing_provider


router = APIRouter()


@router.post("/api/billing/webhook")
async def api_billing_webhook(request: Request):
    provider_name = (os.getenv("PAYMENT_PROVIDER") or "").strip().lower()
    if provider_name == "paddle":
        provider = get_payment_provider()
        return await provider.handle_webhook(request)

    payload_bytes = await request.body()
    signature = request.headers.get("stripe-signature", "")

    provider = get_billing_provider()
    return await provider.handle_webhook(payload_bytes, signature)