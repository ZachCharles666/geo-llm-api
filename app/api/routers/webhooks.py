from fastapi import APIRouter, Request

from app.services.payments.providers import get_billing_provider

router = APIRouter()


@router.post("/api/billing/webhook")
async def api_billing_webhook(request: Request):
    payload_bytes = await request.body()
    signature = request.headers.get("stripe-signature", "")

    provider = get_billing_provider()
    return await provider.handle_webhook(payload_bytes, signature)
