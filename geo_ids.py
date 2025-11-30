# geo_ids.py — GEO-Max COT 节点 UUID 生成工具
#
# 设计目标：
# 1）为每次 Stage1 运行生成统一的 run_id（便于分组与追踪）
# 2）为每个节点（MLC / LLC / CRV / BP 等）生成稳定的 uuid
# 3）uuid 内编码：节点类型 + 版本号 + run_id + 内容指纹 hash
#
# 典型用法（在 geo_cot_parser.stage1_md_to_json 中）：
#
#   from geo_ids import make_run_id, make_node_uuid
#
#   run_id = make_run_id()
#   for node in mlc_nodes:
#       node["uuid"] = make_node_uuid(
#           node_type="MLC",
#           claim=node.get("claim", ""),
#           rationale=node.get("rationale", ""),
#           extra="\n".join(node.get("evidence_need", [])),
#           run_id=run_id,
#       )
#       node["run_id"] = run_id
#
# 后续 Stage2 / GEO 指数 / 封印策略等，都应以 uuid 为主键引用节点。

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Optional


def _normalize_text(text: str) -> str:
    """
    对节点文本做轻量归一化，保证同一逻辑在小改动时 hash 尽量稳定：
    - 转成字符串
    - 去掉首尾空白
    - 按空白切分再用单空格拼接
    - 统一为小写
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    parts = s.split()
    return " ".join(parts).lower()


def _short_hash(content: str, length: int = 10) -> str:
    """
    基于内容生成短哈希：
    - 使用 sha256
    - 统一小写十六进制
    - 取前 length 位
    """
    norm = _normalize_text(content)
    h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return h[:length]


def make_run_id() -> str:
    """
    生成一次 Stage1 运行的统一 run_id。

    格式示例：
        20251129T160301Z-8f3a

    组成：
    - UTC 时间戳到秒：YYYYMMDDTHHMMSSZ
    - 基于当前时间字符串的 4 位短 hash，降低并发冲突概率
    """
    now_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    # 将时间戳 + 当前时间（带小数）再 hash 一次，避免多进程/多线程冲突
    ts_full = f"{now_str}-{time.time():.6f}"
    suffix = _short_hash(ts_full, length=4)
    return f"{now_str}-{suffix}"


def make_node_uuid(
    node_type: str,
    claim: str,
    rationale: Optional[str] = "",
    extra: Optional[str] = "",
    run_id: Optional[str] = None,
    version: str = "v1",
) -> str:
    """
    生成单个 COT 节点的 uuid。

    参数：
    - node_type:
        节点类型字符串，如：
        "MLC"（Machine Logic Chain）
        "LLC"（Human Logic Chain）
        "CRV"（Convergence / 收敛规则）
        "BP" （Stage2 Blueprint）
        也可以扩展为你需要的其他类型。
    - claim:
        节点的主陈述内容（建议必填）
    - rationale:
        节点的主要推理/理由文本
    - extra:
        额外信息（如 evidence_need / section_title 等）
    - run_id:
        整条 Stage1 运行的统一 ID；若未提供则内部自动生成
    - version:
        UUID 规范版本号，默认 "v1"；未来规则升级时可切到 "v2"

    返回：
        像这样的字符串：
        COT-MLC-v1-20251129T160301Z-8f3a-a9c3f21b4d
    """
    if run_id is None:
        run_id = make_run_id()

    # 将节点核心信息拼接后做 hash，保证“节点本身”改变时 uuid 也会变化
    base = "|".join(
        [
            _normalize_text(node_type),
            _normalize_text(claim),
            _normalize_text(rationale or ""),
            _normalize_text(extra or ""),
        ]
    )
    content_hash = _short_hash(base, length=10)

    node_type_norm = (node_type or "NODE").upper()
    version_norm = version.strip() or "v1"

    return f"COT-{node_type_norm}-{version_norm}-{run_id}-{content_hash}"


__all__ = [
    "make_run_id",
    "make_node_uuid",
]
