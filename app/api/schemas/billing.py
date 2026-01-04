from typing import Literal, Optional

from pydantic import BaseModel, Field


class BillingCheckoutIn(BaseModel):
    """
    前端发起 Upgrade 时调用
    - tier: alpha_base | alpha_pro
    - user_id: Supabase auth.user.id（UUID）
    """

    tier: Literal["alpha_base", "alpha_pro"] = Field(..., description="要购买的档位")
    user_id: str = Field(..., description="Supabase user id (uuid)")
    # 可选：你也可以传 email 做辅助（不强制）
    email: Optional[str] = Field("", description="用户邮箱（可选）")


class BillingCheckoutOut(BaseModel):
    ok: bool = True
    error: Optional[str] = None
    checkout_url: Optional[str] = None


class BillingUpgradeIn(BaseModel):
    user_id: str = Field(..., description="Supabase user id (uuid)")
    # v0：不做差价展示，直接升级；如需展示可加 dry_run + upcoming invoice
    dry_run: Optional[bool] = Field(False, description="If true, only return upcoming invoice preview (optional)")


class BillingUpgradeOut(BaseModel):
    ok: bool = True
    error: Optional[str] = None
    subscription_id: Optional[str] = None
    from_price_id: Optional[str] = None
    to_price_id: Optional[str] = None
    invoice_id: Optional[str] = None
    status: Optional[str] = None
