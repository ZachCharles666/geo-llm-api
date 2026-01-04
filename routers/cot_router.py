from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from geo_brand import build_and_validate_brand_brief
from geo_core import geo_cot_stage1, geo_cot_stage2
from geo_cot_parser import stage1_md_to_json, stage2_text_to_blueprint

from app.api.deps import (
    GEO_COT_VERSION,
    TTL_COT_SEC,
    _COT_CACHE,
    _cache_dbg,
    _sha256_text,
    _stable_cache_key,
    logger,
)

router = APIRouter()


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


@router.post("/api/brand-brief/validate", response_model=BrandBriefValidateOut)
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


@router.post("/api/cot/stage1/parse", response_model=Stage1ParseOut)
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
    model_ui: str = Field("Groq", description="模型供应商标识（与 Stage1 一致）")

    @model_validator(mode="after")
    def ensure_brand_brief_text(self):
        """
        Stage1 的入参只需要确保：
        - user_question 非空
        - brand_brief_text 或 brand_brief 至少提供一个
        """
        if not self.user_question:
            raise ValueError("user_question 不能为空。")

        if not self.brand_brief_text and self.brand_brief:
            self.brand_brief_text = self.brand_brief

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


@router.post("/api/cot/stage1/run", response_model=Stage1RunOut)
async def api_cot_stage1_run(payload: Stage1RunIn):
    """
    输入：user_question + brand_brief_text（或 brand_brief）
    输出：Stage1 Markdown + debug_prompt（完整 prompt，便于调试）
    """

    # ===============================
    # Handler Cache (COT Stage1)
    # ===============================

    model_ui = (payload.model_ui or "Groq").strip()  # ✅ 新增：仅用于日志/缓存维度

    s1_key_payload = {
        "v": GEO_COT_VERSION,
        "provider": (payload.model_ui or "Groq").lower().strip(),
        "tpl": "cot_stage1",
        "uq": payload.user_question or "",
        "bb_h": _sha256_text(payload.brand_brief_text or payload.brand_brief or ""),
        "must": payload.must_expose or "",
        "hint": payload.expo_hint or "",
    }
    s1_cache_key = _stable_cache_key("cot_s1", s1_key_payload)
    cached = _COT_CACHE.get(s1_cache_key)

    if cached is not None:
        _cache_dbg("cot_s1", True, s1_cache_key, {"provider": model_ui})
        return cached

    _cache_dbg("cot_s1", False, s1_cache_key, {"provider": model_ui})

    try:
        md_str, debug_prompt = geo_cot_stage1(
            user_question=payload.user_question,
            # 这里统一使用 brand_brief_text（如果没有，会在模型 validator 中用 brand_brief 兜底）
            brand_brief=payload.brand_brief_text or "",
            must_expose=payload.must_expose or "",
            expo_hint=payload.expo_hint or "",
            model_ui=payload.model_ui or "groq",
        )

        out = Stage1RunOut(
            ok=True,
            stage1_markdown=md_str,
            debug_prompt=debug_prompt,
        )
        _COT_CACHE.set(s1_cache_key, out, ttl_sec=TTL_COT_SEC)
        return out

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


# =========================
# 4) /api/cot/stage2/run   ✅ 新增
# =========================


class Stage2RunIn(BaseModel):
    """
    /api/cot/stage2/run 的标准入参模型

    约定：
    - 对外统一使用 stage1_markdown
    - 为兼容旧代码，如果传了 stage1_md，会在模型层映射到 stage1_markdown
    """

    user_question: str = Field(..., description="目标问题原文")

    # 品牌简介：保持和 Stage1 相同的双字段策略（但在 Stage2 不再强制必填）
    brand_brief: Optional[str] = Field(
        default=None,
        description="兼容旧字段：原始品牌简介文本",
    )

    # ✅ 统一标准字段名：stage1_markdown
    stage1_markdown: str = Field(
        ...,
        description="Stage1 的完整 Markdown（包含 <STAGE1_DRAFT_MD> 和 <LOGIC_INDEX_JSON>）",
    )

    # 推荐字段：已经整理好的品牌简介文本（可选）
    brand_brief_text: Optional[str] = Field(
        default=None,
        description="推荐字段：已经整理好的品牌简介文本",
    )

    must_expose: str = Field("", description="期望露出（可选）")
    expo_hint: str = Field("", description="额外曝光提示（可选）")

    # 注意：和 geo_cot_stage2 里的默认值一致，这里仍然用 'Groq'
    model_ui: str = Field("Groq", description="模型供应商标识（与 Stage1 一致）")

    @model_validator(mode="before")
    @classmethod
    def compat_aliases(cls, values: Dict[str, Any]):
        """
        兼容旧字段名：
        - 如果没有 stage1_markdown，但传了 stage1_md，则自动映射
        """
        if "stage1_markdown" not in values and "stage1_md" in values:
            values["stage1_markdown"] = values.pop("stage1_md")
        return values

    @model_validator(mode="after")
    def ensure_brand_brief_text(self):
        """
        Stage2 入参校验：
        - user_question 不能为空
        - stage1_markdown 不能为空
        - brand_brief/brand_brief_text 若有则同步一下，但不再强制必填
        """

        # 1) user_question 必须有
        if not self.user_question:
            raise ValueError("user_question 不能为空。")

        # 2) 如果传了 brand_brief，但 brand_brief_text 为空，就顺手同步一下
        if not self.brand_brief_text and self.brand_brief:
            self.brand_brief_text = self.brand_brief

        # 3) ✅ 不再强制 brand_brief_text / brand_brief 至少一个非空
        #    因为 Stage2 也可以完全基于 stage1_markdown 中的品牌信息工作
        # if not self.brand_brief_text:
        #     raise ValueError("brand_brief_text 或 brand_brief 至少提供一个。")

        # 4) Stage1 Markdown 仍然是必需的
        if not self.stage1_markdown:
            raise ValueError("stage1_markdown 不能为空，请先运行 Stage1。")

        return self


class Stage2RunOut(BaseModel):
    ok: bool = True
    error: Optional[str] = None

    # Stage2 的原文结果
    stage2_markdown: Optional[str] = None
    debug_prompt: Optional[str] = None

    # 解析后的 Blueprint 结构
    blueprint: Optional[Dict[str, Any]] = None


@router.post("/api/cot/stage2/run", response_model=Stage2RunOut)
async def api_cot_stage2_run(payload: Stage2RunIn):
    """
    输入：
      - user_question + brand_brief_text（或 brand_brief）
      - stage1_markdown（来自 /api/cot/stage1/run 的输出）

    内部流程：
      1）先用 Stage1 Parser 再解析一遍 stage1_markdown，得到 s1_json（包含 run_id / mlc_nodes 等）
      2）调用 geo_cot_stage2 生成 Stage2 Markdown
      3）调用 stage2_text_to_blueprint 生成 Blueprint JSON
    """

    model_ui = (payload.model_ui or "Groq").strip()  # ✅ 新增：仅用于日志/缓存维度

    # ===============================
    # Handler Cache (COT Stage2)
    # ===============================
    s2_key_payload = {
        "v": GEO_COT_VERSION,
        "provider": (payload.model_ui or "Groq").lower().strip(),
        "tpl": "cot_stage2",
        "uq": payload.user_question or "",
        "bb_h": _sha256_text(payload.brand_brief_text or payload.brand_brief or ""),
        "must": payload.must_expose or "",
        "hint": payload.expo_hint or "",
        "s1_h": _sha256_text(payload.stage1_markdown or ""),
    }
    s2_cache_key = _stable_cache_key("cot_s2", s2_key_payload)
    cached = _COT_CACHE.get(s2_cache_key)

    if cached is not None:
        _cache_dbg("cot_s2", True, s2_cache_key, {"provider": model_ui})
        return cached

    _cache_dbg("cot_s2", False, s2_cache_key, {"provider": model_ui})

    try:
        # 1) 解析 Stage1：md -> json
        s1_json = stage1_md_to_json(
            md_text=payload.stage1_markdown,          # ✅ 用 stage1_markdown
            user_question=payload.user_question,
            brand_brief=payload.brand_brief_text or "",
            must_expose=payload.must_expose or "",
            expo_hint=payload.expo_hint or "",
        )

        # 2) 调用 Stage2 —— 完全按你现有 geo_cot_stage2 的签名来
        s2_md, s2_debug = geo_cot_stage2(
            user_question=payload.user_question,
            brand_brief=payload.brand_brief_text or "",
            must_expose=payload.must_expose or "",
            stage1_md=payload.stage1_markdown,        # ✅ 这里也改成 payload.stage1_markdown
            model_ui=payload.model_ui or "Groq",
            expo_hint=payload.expo_hint or "",
            # template_name 使用默认值 "cot_stage2"，这里可不传
            # template_name="cot_stage2",
        )

        # 3) 解析 Stage2 → Blueprint JSON
        run_id = None
        if s1_json and isinstance(s1_json, dict):
            run_id = s1_json.get("run_id")

        blueprint = stage2_text_to_blueprint(
            stage2_md=s2_md,
            user_question=payload.user_question,
            brand_brief=payload.brand_brief_text or "",
            must_expose=payload.must_expose or "",
            expo_hint=payload.expo_hint or "",
            run_id=run_id,
        )

        out = Stage2RunOut(
            ok=True,
            stage2_markdown=s2_md,
            debug_prompt=s2_debug,
            blueprint=blueprint,
        )
        _COT_CACHE.set(s2_cache_key, out, ttl_sec=TTL_COT_SEC)
        return out

    except Exception as e:
        try:
            logger.exception("stage2/run failed")
        except Exception:
            print("stage2/run failed:", e)

        return Stage2RunOut(
            ok=False,
            error=str(e),
        )
