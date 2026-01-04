import os

from app.services.payments.paddle_provider import PaddlePaymentProvider
from app.services.payments.provider import PaymentProvider
from app.services.payments.stripe_provider import StripePaymentProvider


def get_payment_provider() -> PaymentProvider:
    provider_name = (os.getenv("PAYMENT_PROVIDER") or "stripe").strip().lower()
    if provider_name == "paddle":
        return PaddlePaymentProvider()
    return StripePaymentProvider()
