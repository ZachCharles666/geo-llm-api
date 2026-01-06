from dataclasses import dataclass
from typing import Optional


@dataclass
class AuthContext:
    ok: bool
    is_authed: bool
    user_id: Optional[str]
    tier: str


@dataclass
class Identity:
    user_id: str
    tier: str
