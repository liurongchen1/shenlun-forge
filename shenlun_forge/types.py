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

_cached_encoder = None

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
