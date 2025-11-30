# api_brand.py
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from geo_brand import build_and_validate_brand_brief
from geo_cot_parser import stage1_md_to_json
from geo_core import geo_cot_stage1  # ✅ 新增：调用 Stage1 引擎


# =========================
# FastAPI 基础配置
# =========================

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


# =========================
# 1) /api/brand-brief/validate
# =========================

class BrandBriefIn(BaseModel):
    brand_name: str = Field(..., description="品牌名称，如：超兔 CRM")
    category: str = Field(..., description="所在行业/品类，如：制造业 CRM")
    target_audience: str = Field(..., description="目标人群，如：制造业老板、工贸企业负责人")
    core_value: str = Field(..., description="核心价值主张，一两句话")

    # 这三个支持“多行字符串”或列表，后端会统一拆分
    key_features: Optional[Any] = Field(
        None,
        description="核心功能/模块，多行文本或数组",
    )
    differentiators: Optional[Any] = Field(
        None,
        description="差异化亮点，多行文本或数组",
    )
    use_cases: Optional[Any] = Field(
        None,
        description="典型使用场景，多行文本或数组",
    )

    # 可选字段：期望露出
    must_expose: Optional[str] = Field(
        None,
        description="期望露出字段，如：超兔CRM + 公众号【表情包姨姨】获取30天试用",
    )


class BrandBriefValidateOut(BaseModel):
    ok: bool
    errors: List[str] = []
    brand_brief_text: Optional[str] = None  # 结构化后的文本版（给 COT 用）
    normalized: Optional[Dict[str, Any]] = None  # 可选：回传规范化后的结构


@app.post("/api/brand-brief/validate", response_model=BrandBriefValidateOut)
async def api_brand_brief_validate(payload: BrandBriefIn):
    """
    校验品牌简介 + 生成统一的 brand_brief 文本（给 Stage1 使用）
    """
    ok, errors, brief_text = build_and_validate_brand_brief(payload.dict())

    return BrandBriefValidateOut(
        ok=ok,
        errors=errors,
        brand_brief_text=brief_text if ok else None,
        normalized=payload.dict(),
    )


# =========================
# 2) /api/cot/stage1/parse
# =========================

class Stage1ParseIn(BaseModel):
    stage1_md: str = Field(..., description="Stage1 生成的完整 Markdown 文本")
    user_question: str = Field("", description="目标问题原文")
    brand_brief: str = Field("", description="品牌简介文本（可选，用于 meta 回填）")
    must_expose: str = Field("", description="期望露出（可选）")
    expo_hint: str = Field("", description="额外曝光提示（可选）")


class Stage1ParseOut(BaseModel):
    ok: bool
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@app.post("/api/cot/stage1/parse", response_model=Stage1ParseOut)
async def api_cot_stage1_parse(payload: Stage1ParseIn):
    """
    将 Stage1 Markdown 解析成结构化 JSON：
    - mlc_nodes: MLC 节点数组
    - llc_nodes: LLC 节点数组
    - convergence: 收敛规则 raw block
    - stage2_blueprint: Stage2 约束 raw block
    """
    try:
        data = stage1_md_to_json(
            md_text=payload.stage1_md,
            user_question=payload.user_question,
            brand_brief=payload.brand_brief,
            must_expose=payload.must_expose,
            expo_hint=payload.expo_hint,
        )

        # 如果完全没有解析到节点，视为非致命错误：返回 data + error 提示
        if not data.get("mlc_nodes") and not data.get("llc_nodes"):
            return Stage1ParseOut(
                ok=False,
                error="未解析出任何 MLC/LLC 节点，请检查 Stage1 模板或模型输出格式。",
                data=data,
            )

        return Stage1ParseOut(ok=True, data=data)

    except Exception as e:
        # 防御式兜底，避免接口挂掉
        return Stage1ParseOut(
            ok=False,
            error=f"解析异常: {e}",
            data=None,
        )


# =========================
# 3) /api/cot/stage1/run   ✅ 新增
# =========================

class Stage1RunIn(BaseModel):
    """
    兼容两种调用方式：
    - 方式 A（当前 alpha-cot 前端）：只传 brand_brief（多行拼接字符串）
    - 方式 B（未来）：传 brand_brief_text（已经结构化+校验后的文本）
    """
    user_question: str
    brand_brief: Optional[str] = None          # 兼容旧前端
    brand_brief_text: Optional[str] = None     # 新推荐字段
    must_expose: str = ""
    expo_hint: str = ""
    model_ui: str = "groq"

    @model_validator(mode="after")
    def ensure_brand_brief_text(self):
        # 如果没传 brand_brief_text，但传了 brand_brief，就用 brand_brief 兜底
        if not self.brand_brief_text and self.brand_brief:
            self.brand_brief_text = self.brand_brief

        # 两个都为空才算真正错误
        if not self.user_question:
            raise ValueError("user_question 不能为空。")

        if not self.brand_brief_text:
            raise ValueError("brand_brief_text 或 brand_brief 至少提供一个。")

        return self


class Stage1RunOut(BaseModel):
    ok: bool = True
    error: Optional[str] = None
    stage1_markdown: Optional[str] = None
    debug_prompt: Optional[str] = None


# ============================================================
#  /api/cot/stage1/run - 运行 Stage1（生成逻辑链 Markdown）
# ============================================================

@app.post("/api/cot/stage1/run", response_model=Stage1RunOut)
async def api_cot_stage1_run(payload: Stage1RunIn):
    """
    输入：user_question + brand_brief_text（或 brand_brief）
    输出：Stage1 Markdown + debug_prompt（完整 prompt，便于调试）
    """
    try:
        md_str, debug_prompt = geo_cot_stage1(
            user_question=payload.user_question,
            # 这里统一使用 brand_brief_text（如果没有，会在模型 validator 中用 brand_brief 兜底）
            brand_brief=payload.brand_brief_text or "",
            must_expose=payload.must_expose or "",
            expo_hint=payload.expo_hint or "",
            model_ui=payload.model_ui or "groq",
        )

        return Stage1RunOut(
            ok=True,
            stage1_markdown=md_str,
            debug_prompt=debug_prompt,
        )

    except Exception as e:
        # 这里用你文件里已有的 logger，如果没有可以用 print 代替
        try:
            logger.exception("stage1/run failed")
        except Exception:
            print("stage1/run failed:", e)

        return Stage1RunOut(
            ok=False,
            error=str(e),
        )
