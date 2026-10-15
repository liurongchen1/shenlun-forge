"""Core data models for shenlun-forge."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class InfoType(str, Enum):
    """Information type extracted from context."""
    FACT = "fact"               # 客观事实
    ACTION = "action"           # 动作/做法
    PROBLEM = "problem"         # 问题/缺陷
    CAUSE = "cause"             # 原因
    EFFECT_POS = "effect_pos"   # 积极影响
    EFFECT_NEG = "effect_neg"   # 消极影响
    OPINION = "opinion"         # 观点/判断
    DATA = "data"               # 数据/统计
    OTHER = "other"


# ── Compression guard rails (shared by every strategy) ────────────────────

# Below this size, compression is not worth the risk of dropping something:
# the token saving is a handful of tokens, and a lossy result is a real cost.
MIN_TOKENS_TO_COMPRESS = 200

# Compression must save at least this fraction of the original tokens to be
# worth returning. Otherwise the original context is passed through untouched,
# which also guarantees a result never grows relative to its input.
MIN_COMPRESSION_GAIN = 0.05

# Share of the token budget the verbatim recent-message window may occupy.
RECENT_BUDGET_SHARE = 0.4

# How to handle the case where the protected recent messages do not fit in the
# remaining budget:
#   "keep"     — keep them whole, even if the result exceeds max_tokens
#                (information completeness wins; stats["over_budget"] is set)
#   "truncate" — trim them so the result honours max_tokens
BUDGET_OVERFLOW_POLICIES = ("keep", "truncate")
RECENT_HEADER = "\n\n## Recent Messages (Uncompressed)\n"


class TaskType(str, Enum):
    """Task types that drive compression strategy."""
    QA = "qa"                   # 问答
    SUMMARY = "summary"         # 摘要
    CODE_GEN = "code_gen"       # 代码生成
    ANALYSIS = "analysis"       # 分析推理
    CREATIVE = "creative"       # 创意写作
    EXTRACTION = "extraction"   # 信息提取
    TOOL_USE = "tool_use"       # 工具调用
    CONVERSATION = "conversation"  # 多轮对话
    AUTO = "auto"               # 自动判断


@dataclass
class ContextBlock:
    """A single block of context with metadata."""
    content: str
    role: str = "user"          # user/assistant/system/tool
    msg_index: int = 0          # 在对话中的位置
    token_count: int = 0        # token数量
    info_type: Optional[InfoType] = None
    relevance_score: float = 0.0  # 与当前任务的关联度
    summary_label: Optional[str] = None  # 总括词
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressedBlock:
    """A compressed context block with provenance."""
    content: str
    original_indices: list[int] = field(default_factory=list)
    summary_label: Optional[str] = None
    info_type: Optional[InfoType] = None
    preserved_token_count: int = 0
    compression_ratio: float = 0.0  # 压缩比
    # Relevance inherited from the source blocks. The retriever scores every
    # message against the query; the formatter must honour that score when it
    # decides what fits in the budget, otherwise the ranking work is wasted.
    relevance_score: float = 0.0


@dataclass
class TaskProfile:
    """Task analysis result that drives compression."""
    task_type: TaskType = TaskType.AUTO
    key_elements: list[str] = field(default_factory=list)  # 需要保留的关键要素
    output_format: str = "sequential"  # sequential / structured / bullet
    token_budget: int = 4000
    priority_rules: dict[str, float] = field(default_factory=dict)  # 各要素权重
    # v0.3 P0: 分类维度与输出逻辑（供 Retriever/Formatter 使用）
    classification_dimension: str = "auto"  # auto / temporal / spatial / causal / layer / element
    output_logic: str = "auto"  # auto / parallel / causal / progressive / temporal
    # v0.3 Retriever 增强: 原始查询（供 Retriever 直接提取英文术语）
    current_query: str = ""


@dataclass
class CompressionResult:
    """Final compressed context output."""
    compressed_text: str
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    blocks: list[CompressedBlock] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    steps_log: dict[str, Any] = field(default_factory=dict)  # 每步处理日志


@dataclass
class CompressionPlan:
    """What to do about a context, decided before compressing it.

    Compression is lossy, so the first useful question is not "which strategy"
    but "is this worth compressing at all, and with how much budget". Answering
    that needs no LLM call and no guessing.
    """

    history_tokens: int
    model_window: int
    fits_in_window: bool
    recommended_max_tokens: Optional[int] = None
    advice: str = ""

_cached_encoder = None


def split_recent_window(
    blocks: list,
    max_tokens: int,
    recent_window: int = 0,
):
    """Split ``blocks`` into (early, recent), protecting the newest messages.

    The recent messages are kept verbatim, but they must not swallow the whole
    budget: the window shrinks — always keeping the newest message — until it
    fits within :data:`RECENT_BUDGET_SHARE` of ``max_tokens``. Without this the
    compressed part is squeezed to nothing and the result blows past the budget
    it was asked to respect.

    ``recent_window <= 0`` selects the adaptive default, ``max(3, n // 8)``.
    """
    n = len(blocks)
    if n == 0:
        return [], []

    if recent_window <= 0:
        recent_window = max(3, n // 8)

    keep = min(recent_window, n)
    recent_budget = int(max_tokens * RECENT_BUDGET_SHARE)
    while keep > 1 and sum(b.token_count for b in blocks[-keep:]) > recent_budget:
        keep -= 1

    if keep > 0 and n > keep:
        return blocks[:-keep], blocks[-keep:]
    return blocks, []

def truncate_to_tokens(text: str, target_tokens: int, suffix: str = "...") -> str:
    """Truncate ``text`` to at most ``target_tokens`` (tokenizer-accurate)."""
    if target_tokens <= 0:
        return ""
    if count_tokens(text) <= target_tokens:
        return text

    low, high = 0, len(text)
    while low < high - 10:
        mid = (low + high) // 2
        if count_tokens(text[:mid]) <= target_tokens:
            low = mid
        else:
            high = mid
    return text[:low] + suffix


def render_recent_blocks(
    recent_blocks: list,
    remaining_tokens: int,
    policy: str = "keep",
) -> tuple[str, bool]:
    """Render the protected recent messages for appending to a result.

    ``policy="keep"`` (the default) appends every recent message whole, even
    when that pushes the result past the caller's ``max_tokens``. That is
    deliberate: losing the newest turns is usually worse than overshooting a
    budget, and ``stats["over_budget"]`` reports the overshoot honestly.

    ``policy="truncate"`` treats ``max_tokens`` as a hard limit — it appends
    only what fits and trims the final message.

    Returns ``(text, truncated)``; ``text`` includes the section header and is
    empty when nothing could be kept.
    """
    if not recent_blocks:
        return "", False

    if policy != "truncate":
        return RECENT_HEADER + "\n".join(b.content for b in recent_blocks), False

    used = 0
    kept: list[str] = []
    truncated = False
    for block in recent_blocks:
        cost = count_tokens(block.content)
        if used + cost <= remaining_tokens:
            kept.append(block.content)
            used += cost
            continue
        left = remaining_tokens - used
        if left > 20:
            kept.append(truncate_to_tokens(block.content, left))
        truncated = True
        break

    if not kept:
        return "", truncated
    return RECENT_HEADER + "\n".join(kept), truncated


def validate_overflow_policy(policy: str) -> str:
    """Reject typo'd policies instead of silently falling back to a default."""
    if policy not in BUDGET_OVERFLOW_POLICIES:
        allowed = ", ".join(repr(p) for p in BUDGET_OVERFLOW_POLICIES)
        raise ValueError(f"budget_overflow must be one of {allowed}, got {policy!r}")
    return policy


def plan_compression(
    messages: list,
    model_window: int = 128_000,
    safety_margin: float = 0.1,
) -> CompressionPlan:
    """Decide whether a conversation needs compressing before spending anything.

    Needs no LLM, no API key and no model instance, so it can run at the top of
    an agent loop. Compression is lossy: if the history already fits the window,
    running it can only lose information.

    Args:
        messages: The conversation you are about to send.
        model_window: The model's context size in tokens.
        safety_margin: Share of the window to leave for the system prompt, the
            current question and the reply. Default ``0.1``.
    """
    history_tokens = sum(count_tokens((m or {}).get("content") or "") for m in messages)
    usable = max(1, int(model_window * (1.0 - safety_margin)))

    if history_tokens <= usable:
        return CompressionPlan(
            history_tokens=history_tokens,
            model_window=model_window,
            fits_in_window=True,
            recommended_max_tokens=None,
            advice=(
                f"{history_tokens:,} tokens fit the usable {usable:,} of a "
                f"{model_window:,}-token window. Send the history unchanged — "
                "compressing it can only drop information."
            ),
        )

    return CompressionPlan(
        history_tokens=history_tokens,
        model_window=model_window,
        fits_in_window=False,
        recommended_max_tokens=usable,
        advice=(
            f"{history_tokens:,} tokens exceed the usable {usable:,}. Compression "
            f"is required; pass max_tokens={usable:,} rather than something "
            "tighter. Measured answer quality improves with budget, and a tight "
            "budget costs a large share of questions outright."
        ),
    )


def count_tokens(text: str) -> int:
    """Standalone token counter using tiktoken.
    
    Uses cl100k_base encoding (GPT-4/Claude-compatible).
    Encoder is cached at module level after first call.
    Falls back to len(text)//3 if tiktoken is unavailable.
    """
    global _cached_encoder
    try:
        if _cached_encoder is None:
            import tiktoken
            _cached_encoder = tiktoken.get_encoding("cl100k_base")
        return len(_cached_encoder.encode(text))
    except ImportError:
        return len(text) // 3
