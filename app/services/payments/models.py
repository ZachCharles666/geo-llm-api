from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class NormalizedBillingEvent(BaseModel):
    provider: str
    event_id: str
    event_type: str
    occurred_at: datetime | str
    customer_id: Optional[str] = None
    subscription_id: Optional[str] = None
    price_id: Optional[str] = None
    tier: Optional[str] = None
    status: Optional[str] = None
    raw: dict[str, Any]
