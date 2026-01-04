import os
import time
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from geo_core import geo_rewrite
from geo_evaluator import geo_score_pipeline
from providers_groq_gemini import hub as providers_hub

from app.api.deps import (
    GEO_PROMPTS_VERSION,
    GEO_SCORE_VERSION,
    TTL_REWRITE_SEC,
    TTL_SCORE_SEC,
    _REWRITE_CACHE,
    _SCORE_CACHE,
    _build_anchor_context,
    _cache_dbg,
    _estimate_tokens,
    _extract_groq_retry_after_sec,
    _is_groq_429,
    _reset_daily_counter,
    _rewrite_usage,
    _sha256_text,
    _stable_cache_key,
    _supabase_rpc_consume_tokens,
    require_authed_user,
    _quota_block_response,
    _strip_anchor_echo,
)

router = APIRouter()


class GeoScoreIn(BaseModel):
    # ✅ user_question 改为可选（允许空字符串）
    user_question: str = Field("", description="用户的原始提问（可选；为空时将回退使用 article_title）")

    # ✅ 新增：标题作为主题锚点（前端必填）
    article_title: str = Field("", description="文章标题（主题锚点；建议前端必填）")

    source_text: str = Field(..., description="原始文本（未优化版本）")
    rewritten_text: str = Field(..., description="优化后的文本（用于 GEO 评分）")

    model_ui: str = Field("groq", description="模型提供方标识，如 groq / openai / qwen")
    model_name: str = Field("llama-3.3-70b-versatile", description="具体模型名")
    samples: int = Field(1, description="主观评分采样次数")
    # ⚠️ legacy field: kept for backward compatibility, NOT trusted by server.
    user_tier: Optional[str] = Field(
        default=None,
        description="(Deprecated) Legacy field kept for backward compatibility. Server ignores this and uses profiles.tier.",
    )


class GeoScoreOut(BaseModel):
    ok: bool
    error: Optional[str] = None

    geo_score: Optional[float] = None      # 原始总分（0~100，内部用）
    grade: Optional[str] = None            # A/B/C/D/E
    summary: Optional[str] = None          # 总分简短点评

    sealed_overall: Optional[float] = None # 封印后的对外总分（0~100）
    sealed: Optional[Dict[str, Any]] = None  # 包含 overall_score + metrics 列表

    latency_ms: Optional[int] = None
    model_used: Optional[str] = None
    user_tier: Optional[str] = None

    raw_scores_0_1: Optional[Dict[str, float]] = None
    raw_scores_0_100: Optional[Dict[str, float]] = None
    subjective_raw_1_5: Optional[float] = None
    subjective_scaled_0_1: Optional[float] = None
    subjective_scaled_0_100: Optional[float] = None


@router.post("/api/geo/score", response_model=GeoScoreOut)
async def api_geo_score(
    request: Request,
    payload: GeoScoreIn,
    ident: dict = Depends(require_authed_user),
):
    """
    GEO 指数评分接口（强制登录）：
    - 未登录：401
    - token 无效：401
    - Supabase env 缺失：500（明确提示）
    """
    try:
        uq = (payload.user_question or "").strip()
        title = (payload.article_title or "").strip()

        # ✅ Title fallback：避免 relevance 用空 query 导致 Pertinence 不稳定
        effective_question = uq or title

        effective_tier = (ident.get("tier") or "free").lower().strip()

        # ===============================
        # Handler Cache (Score)
        # ===============================
        score_key_payload = {
            "v": GEO_SCORE_VERSION,
            "provider": (payload.model_ui or "").lower().strip(),
            "model": (payload.model_name or "").strip(),
            "samples": int(payload.samples or 1),
            "tier": effective_tier,  # score 输出与封印视图受 tier 影响，handler 直接按 tier 缓存最稳妥
            "question": effective_question,
            # 大文本只放 hash，避免 key 过大
            "src_h": _sha256_text(payload.source_text or ""),
            "opt_h": _sha256_text(payload.rewritten_text or ""),
        }
        cache_key = _stable_cache_key("score", score_key_payload)

        cached = _SCORE_CACHE.get(cache_key)
        if cached:
            # ✅ 方案B：命中缓存也扣费（Score 只按用户输入：title + question + source_text）
            billable_in = (
                _estimate_tokens(payload.article_title or "")
                + _estimate_tokens(payload.user_question or "")
                + _estimate_tokens(payload.source_text or "")
            )

            # samples 若由用户可配置，则计入（你 cache_key 也包含 samples）
            billable_in = billable_in * int(payload.samples or 1)

            consumed = _supabase_rpc_consume_tokens(
                user_id=ident["user_id"],
                tier=effective_tier,
                kind="score",
                tokens_in=billable_in,
                tokens_out=0,
                tokens_total=billable_in,
            )
            if not consumed.get("ok"):
                return _quota_block_response(consumed)

            return cached

        result = geo_score_pipeline(
            user_question=effective_question,
            source_text=payload.source_text,
            rewritten_text=payload.rewritten_text,
            model_ui=payload.model_ui,
            model_name=payload.model_name,
            samples=payload.samples,
            user_tier=effective_tier,
        )

        # 如果 pipeline 报错，直接返回，不进缓存
        if not result.get("ok"):
            return GeoScoreOut(ok=False, error=result.get("error") or "GEO 评分失败")

        # 构建返回对象 (不再直接 return，而是先存入变量 out)
        out = GeoScoreOut(
            ok=True,
            geo_score=result.get("geo_score"),
            grade=result.get("grade"),
            summary=result.get("summary"),
            sealed_overall=result.get("sealed_overall"),
            sealed=result.get("sealed"),
            latency_ms=result.get("latency_ms"),
            model_used=result.get("model_used"),
            user_tier=effective_tier,
            raw_scores_0_1=result.get("raw_scores_0_1"),
            raw_scores_0_100=result.get("raw_scores_0_100"),
            subjective_raw_1_5=result.get("subjective_raw_1_5"),
            subjective_scaled_0_1=result.get("subjective_scaled_0_1"),
            subjective_scaled_0_100=result.get("subjective_scaled_0_100"),
        )

        # ===============================
        # Monthly token quota consume (Path 1)
        # ===============================
        # ✅ 计费口径（Score）：只按用户输入计费：title + question + source_text
        billable_in = (
            _estimate_tokens(payload.article_title or "")
            + _estimate_tokens(payload.user_question or "")
            + _estimate_tokens(payload.source_text or "")
        )

        # samples 若由用户可配置，则计入
        billable_in = billable_in * int(payload.samples or 1)

        consumed = _supabase_rpc_consume_tokens(
            user_id=ident["user_id"],
            tier=effective_tier,
            kind="score",
            tokens_in=billable_in,
            tokens_out=0,
            tokens_total=billable_in,
        )

        if not consumed.get("ok"):
            return _quota_block_response(consumed)

        # 将成功的结果存入缓存，以便下次快速读取
        # cache_key 是根据输入参数生成的唯一标识
        _SCORE_CACHE.set(cache_key, out, ttl_sec=TTL_SCORE_SEC)

        # 最后返回结果给调用方
        return out

    except HTTPException:
        raise
    except Exception as e:
        return GeoScoreOut(ok=False, error=f"GEO 评分异常: {e}")


class RewriteIn(BaseModel):
    text: str = Field(..., description="待改写的原文文本")
    article_title: Optional[str] = Field("", description="文章标题（可选，用作主题锚点）")
    user_question: Optional[str] = Field("", description="用户问题（可选，用作意图锚点）")
    out_lang: str = Field("Auto", description="输出语言：Auto / Chinese / English / Spanish / French / Japanese / Korean / German")
    model_ui: str = Field(
        "Groq",
        description="（可选）内部调试字段；默认使用后端配置的 Provider",
    )

    # ✅ 新增：用于决定 2-pass / 3-pass 的目标选择
    rewrite_goal: Literal["balanced", "seo_boost", "authority_boost"] = Field(
        "balanced",
        description="改写目标：balanced(默认) / seo_boost(曝光优先，可能触发3-pass) / authority_boost(专业权威优先)",
    )


class RewriteOut(BaseModel):
    ok: bool = True
    error: str | None = None

    rewritten: str | None = None   # 改写后的文本
    original: str | None = None    # 原文（回显用）

    out_lang: str | None = None
    model_ui: str | None = None

    input_tokens_est: int | None = None   # 输入token粗估（用于调试/展示）
    output_tokens_est: int | None = None  # 输出token粗估（用于调试/展示）


#  GEO-Rewrite 入口（Alpha 强化版）
#  - 权限：仅信任 Authorization: Bearer <supabase access_token> → profiles.tier
#  - Free：后端硬限制（token / 每日次数 / 总次数）
#  - Alpha：通过 profiles.tier 自动放开

FREE_TOKEN_LIMIT = 1000     # Free 单次约 1000 tokens
FREE_DAILY_LIMIT = 2        # Free 每日 2 次
FREE_TOTAL_LIMIT = 10       # Free 总计 10 次试用


@router.post("/api/rewrite", response_model=RewriteOut)
async def api_rewrite(
    request: Request,
    payload: RewriteIn,
    ident: dict = Depends(require_authed_user),
):
    """
    GEO-Max 内容改写接口（Alpha 强化版）：
    - 输入：text + out_lang
    - Free：后端硬限制（单次 token / 每日次数 / 总次数）
    - 捕获 Groq 429：返回更清晰的错误信息（含建议等待秒数）
    """
    try:
        text = payload.text or ""
        out_lang = payload.out_lang or "Auto"

        if out_lang == "Auto":
            if any("\u4e00" <= ch <= "\u9fff" for ch in text):
                out_lang = "Chinese"

        # ===============================
        # 0) Resolve tier from Supabase session (if logged-in)
        # ===============================
        tier = (ident.get("tier") or "free").lower().strip()

        # ===============================
        # A-1) 构造锚点上下文（仅送入模型，不影响限额逻辑）
        # ===============================
        anchored_text = _build_anchor_context(
            title=payload.article_title or "",
            question=payload.user_question or "",
            body=text,
        )

        # ===============================
        # 1) Free Tier：后端硬限制
        # ===============================
        model_ui = os.getenv("GEO_REWRITE_MODEL_UI", payload.model_ui or "Groq")

        if tier == "free":
            _reset_daily_counter()

            if _rewrite_usage["total"] >= FREE_TOTAL_LIMIT:
                return RewriteOut(
                    ok=False,
                    error="Free trial quota exhausted. Please upgrade to Alpha for more rewrites.",
                    rewritten=None,
                    original=text,
                    out_lang=out_lang,
                    model_ui=model_ui,
                )

            if _rewrite_usage["today"] >= FREE_DAILY_LIMIT:
                return RewriteOut(
                    ok=False,
                    error="Daily free limit reached (2 runs per day). Try again tomorrow or upgrade to Alpha.",
                    rewritten=None,
                    original=text,
                    out_lang=out_lang,
                    model_ui=model_ui,
                )

            approx = _estimate_tokens(text)
            if approx > FREE_TOKEN_LIMIT:
                return RewriteOut(
                    ok=False,
                    error=f"Free plan allows about {FREE_TOKEN_LIMIT} tokens per run. "
                          f"Current text is estimated at ~{approx} tokens.",
                    rewritten=None,
                    original=text,
                    out_lang=out_lang,
                    model_ui=model_ui,
                )

        # ===============================
        # Handler Cache (Rewrite)
        # ===============================
        # 注意：rewrite 输出本质与 tier 无关（你没在 rewrite 做封印），
        # 但 free tier 有次数限制；因此只缓存“成功输出”，并且缓存读取放在 free 限制通过之后。
        rewrite_key_payload = {
            "v": GEO_PROMPTS_VERSION,
            "provider": (model_ui or "").lower().strip(),
            "goal": (payload.rewrite_goal or "balanced").strip().lower(),
            "out_lang": out_lang,
            "temp": 0.2,  # 你第一次 pass 的温度固定 0.2
            "use_chunk": True,  # 你第一次调用 use_chunk=True
            "title_h": _sha256_text(payload.article_title or ""),
            "question_h": _sha256_text(payload.user_question or ""),
            "body_h": _sha256_text(text),
        }
        rewrite_cache_key = _stable_cache_key("rewrite", rewrite_key_payload)

        cached = _REWRITE_CACHE.get(rewrite_cache_key)

        if cached is not None:
            # ✅ 方案B：命中缓存也扣费（只按用户输入：title + question + body）
            billable_in = (
                _estimate_tokens(payload.article_title or "")
                + _estimate_tokens(payload.user_question or "")
                + _estimate_tokens(text)
            )

            consumed = _supabase_rpc_consume_tokens(
                user_id=ident["user_id"],
                tier=tier,
                kind="rewrite",
                tokens_in=billable_in,
                tokens_out=0,
                tokens_total=billable_in,
            )

            if not consumed.get("ok"):
                return _quota_block_response(consumed)

            _cache_dbg("rewrite", True, rewrite_cache_key, {"provider": model_ui})
            return cached

        _cache_dbg("rewrite", False, rewrite_cache_key, {"provider": model_ui})

        # ===============================
        # 2) 动态 max_chars：根据 rewrite_goal 调整（alpha 兜底）
        # ===============================
        dynamic_max_chars = max(len(text) * 3 // 2, 2400)
        if payload.rewrite_goal == "seo_boost":
            dynamic_max_chars = max(dynamic_max_chars, 4000)
        elif payload.rewrite_goal == "authority_boost":
            dynamic_max_chars = max(dynamic_max_chars, 3200)

        # ===============================
        # 3) 双 pass 改写（anchored_text / use_chunk=True / max_chars 动态）
        # ===============================
        optimized, _ = geo_rewrite(
            text=anchored_text,
            model_ui=model_ui,
            out_lang=out_lang,
            rewrite_goal=payload.rewrite_goal,
            use_chunk=True,
            max_chars=dynamic_max_chars,
            temperature=0.2,
        )

        optimized = _strip_anchor_echo(optimized or "")
        b = _estimate_tokens(optimized)

        # ===============================
        # 4) 兜底重试（仍使用 anchored_text）
        #    ✅ 捕获 Groq 429（注意：这里也可能触发第二次请求）
        # ===============================
        min_out = max(_estimate_tokens(text) * 0.6, 120)
        a = _estimate_tokens(text)

        if a > 0 and b < min_out:
            try:
                optimized2, _ = geo_rewrite(
                    text=anchored_text,
                    model_ui=model_ui,
                    out_lang=out_lang,
                    rewrite_goal=payload.rewrite_goal,
                    use_chunk=False,
                    max_chars=max(dynamic_max_chars, 3600),
                    temperature=0.35,
                )
            except Exception as e:
                if _is_groq_429(e):
                    wait_sec = _extract_groq_retry_after_sec(e)
                    human = "Groq rate limit reached (429) during fallback rewrite."
                    if wait_sec:
                        human += f" Please retry after ~{wait_sec} seconds."
                    else:
                        human += " Please retry later."
                    human += f" (details: {str(e)})"
                    return RewriteOut(
                        ok=False,
                        error=human,
                        rewritten=None,
                        original=text,
                        out_lang=out_lang,
                        model_ui=model_ui,
                    )
                raise

            optimized2 = _strip_anchor_echo(optimized2 or "")
            b2 = _estimate_tokens(optimized2)

            if b2 >= min_out or b2 > b:
                optimized, b = optimized2, b2

        # ===============================
        # 5) Free 用户：成功后计数
        # ===============================
        if tier == "free":
            _rewrite_usage["total"] += 1
            _rewrite_usage["today"] += 1

        # ===============================
        # 5.5) Monthly token quota consume (Path 1)
        # ===============================
        # ✅ 计费口径（Rewrite）：只按用户输入计费：title + question + body
        billable_in = (
            _estimate_tokens(payload.article_title or "")
            + _estimate_tokens(payload.user_question or "")
            + _estimate_tokens(text)
        )

        consumed = _supabase_rpc_consume_tokens(
            user_id=ident["user_id"],
            tier=tier,
            kind="rewrite",
            tokens_in=billable_in,
            tokens_out=0,
            tokens_total=billable_in,
        )

        if not consumed.get("ok"):
            return _quota_block_response(consumed)

        # ===============================
        # 6) 返回（original 仍是用户正文）
        # ===============================

        out = RewriteOut(
            ok=True,
            rewritten=optimized,
            original=text,
            out_lang=out_lang,
            model_ui=model_ui,
            input_tokens_est=billable_in,
            output_tokens_est=_estimate_tokens(optimized),
        )

        _REWRITE_CACHE.set(rewrite_cache_key, out, ttl_sec=TTL_REWRITE_SEC)
        return out

    except Exception as e:
        # ✅ 最外层兜底也顺手识别一次 429，避免漏网
        if _is_groq_429(e):
            wait_sec = _extract_groq_retry_after_sec(e)
            human = "Groq rate limit reached (429)."
            if wait_sec:
                human += f" Please retry after ~{wait_sec} seconds."
            else:
                human += " Please retry later."
            human += f" (details: {str(e)})"
            print("api_rewrite failed (429):", e)
            return RewriteOut(
                ok=False,
                error=human,
                rewritten=None,
                original=payload.text,
                out_lang=payload.out_lang,
                model_ui=payload.model_ui,
            )

        print("api_rewrite failed:", e)
        return RewriteOut(
            ok=False,
            error=str(e),
            rewritten=None,
            original=payload.text,
            out_lang=payload.out_lang,
            model_ui=payload.model_ui,
        )


@router.get("/api/health/providers")
async def api_health_providers():
    """
    Provider / KeyPool 健康检查（不泄露 key）：
    - pools：每个 provider 的 key 数、可用数、cooldown 秒数
    - runtime：最近一次 ok/error、status_code、retry_after（如可解析）
    - chain：当前 fallback 链
    """
    started_at = time.time()

    # 1) KeyPool 视角（冷却/可用 key）
    pools = {}
    try:
        # 你 providers_groq_gemini.py 里 health() 返回的是 keypool snapshot + provider_chain
        pools = providers_hub.health()
    except Exception as e:
        pools = {"error": str(e)}

    # 2) Runtime 视角（最近一次成功/失败）
    runtime = {}
    try:
        # 你现在实现了 health_runtime()
        runtime = providers_hub.health_runtime()
    except Exception as e:
        runtime = {"error": str(e)}

    # 3) 关键环境变量（非敏感）
    env_view = {
        "GEO_PROVIDER_CHAIN": (os.getenv("GEO_PROVIDER_CHAIN") or "").strip(),
        "GEO_KEY_COOLDOWN_DEFAULT_SEC": (os.getenv("GEO_KEY_COOLDOWN_DEFAULT_SEC") or "").strip(),
        "GEO_KEYPOOL_SHUFFLE": (os.getenv("GEO_KEYPOOL_SHUFFLE") or "").strip(),
    }

    return {
        "ok": True,
        "env": env_view,
        "pools": pools,
        "runtime": runtime,
        "latency_ms": int((time.time() - started_at) * 1000),
    }
