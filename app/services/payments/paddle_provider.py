from typing import Any

from fastapi import Request

from app.services.payments.provider import PaymentProvider


class PaddlePaymentProvider(PaymentProvider):
    """Placeholder for future Paddle implementation."""

    async def create_checkout_session(self, payload: Any) -> dict:
        raise NotImplementedError("TODO: implement Paddle checkout session creation")

    async def upgrade_subscription(self, payload: Any) -> dict:
        raise NotImplementedError("TODO: implement Paddle subscription upgrade")

    async def handle_webhook(self, request: Request) -> dict:
        raise NotImplementedError("TODO: implement Paddle webhook handling")
