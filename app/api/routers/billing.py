from fastapi import APIRouter

from app.api.schemas.billing import (
    BillingCheckoutIn,
    BillingCheckoutOut,
    BillingUpgradeIn,
    BillingUpgradeOut,
)
from app.services.payments.providers import get_billing_provider

router = APIRouter()


@router.post("/api/billing/checkout", response_model=BillingCheckoutOut)
async def api_billing_checkout(payload: BillingCheckoutIn):
    provider = get_billing_provider()
    return await provider.checkout(payload)


@router.post("/api/billing/upgrade", response_model=BillingUpgradeOut)
async def api_billing_upgrade(payload: BillingUpgradeIn):
    provider = get_billing_provider()
    return await provider.upgrade(payload)
