import os

from app.services.payments.providers.stripe import StripeBillingProvider

_SELECTED_PROVIDER = os.getenv("BILLING_PROVIDER", "stripe").strip().lower() or "stripe"
_stripe_provider = StripeBillingProvider()


def get_billing_provider():
    if _SELECTED_PROVIDER == "stripe":
        return _stripe_provider
    raise ValueError(f"Unsupported billing provider: {_SELECTED_PROVIDER}")
