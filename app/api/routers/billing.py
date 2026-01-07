from fastapi import APIRouter

from app.api.schemas.billing import (
    BillingCheckoutIn,
    BillingCheckoutOut,
    BillingUpgradeIn,
    BillingUpgradeOut,
)
from app.services.payments import get_payment_provider

router = APIRouter()


@router.post("/api/billing/checkout", response_model=BillingCheckoutOut)
async def api_billing_checkout(payload: BillingCheckoutIn):
    provider = get_payment_provider()
    return await provider.create_checkout_session(payload)


@router.post("/api/billing/upgrade", response_model=BillingUpgradeOut)
async def api_billing_upgrade(payload: BillingUpgradeIn):
    provider = get_payment_provider()
    return await provider.upgrade_subscription(payload)
