from abc import ABC, abstractmethod
from typing import Any

from fastapi import Request


class PaymentProvider(ABC):
    """Payment provider interface used by billing endpoints."""

    @abstractmethod
    async def create_checkout_session(self, payload: Any) -> dict:
        """Create a checkout session for the given payload."""
        raise NotImplementedError

    @abstractmethod
    async def upgrade_subscription(self, payload: Any) -> dict:
        """Upgrade an existing subscription for the given payload."""
        raise NotImplementedError

    @abstractmethod
    async def handle_webhook(self, request: Request) -> dict:
        """Handle an incoming webhook request."""
        raise NotImplementedError
