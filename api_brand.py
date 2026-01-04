from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import deps
from app.api.routers.billing import router as billing_router
from app.api.routers.webhooks import router as webhooks_router
from routers.cot_router import router as cot_router
from routers.geo_router import router as geo_router

app = FastAPI(
    title="GEO-Max Alpha API",
    description="Brand brief 校验 + COT Stage1 运行 & 解析 等内部能力接口",
    version="0.2.0",
)

# Alpha 阶段 CORS 先放开，后续再收紧域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/quota/me")
def api_quota_me(request: Request):
    """
    返回当前登录用户本月 token quota（不消费 token）
    设计：可匿名访问；token 无效时也不返回 401，避免前端出现 401 循环。
    """
    info = deps.resolve_user_and_tier(request)
    if not info.get("is_authed"):
        return {
            "ok": True,
            "is_authed": False,
            "tier": "free",
            "yyyymm": deps._yyyymm_utc_now(),
            "tokens_limit": 0,
            "tokens_used": 0,
            "tokens_remaining": 0,
        }

    user_id = info["user_id"]
    tier = info.get("tier") or "free"
    yyyymm = deps._yyyymm_utc_now()
    default_limit = int(deps._tier_monthly_limit(tier))

    row = deps._supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)
    if row is None:
        deps._supabase_upsert_quota_row(user_id=user_id, tier=tier, yyyymm=yyyymm, tokens_limit=default_limit)
        row = deps._supabase_get_quota_row(user_id=user_id, yyyymm=yyyymm)

    tokens_limit = int((row or {}).get("tokens_limit") or default_limit)
    tokens_used = int((row or {}).get("tokens_used") or 0)
    remaining = max(0, tokens_limit - tokens_used)

    return {
        "ok": True,
        "is_authed": True,
        "tier": tier,
        "yyyymm": yyyymm,
        "tokens_limit": tokens_limit,
        "tokens_used": tokens_used,
        "tokens_remaining": remaining,
    }

app.include_router(cot_router)
app.include_router(geo_router)
app.include_router(billing_router)
app.include_router(webhooks_router)
