"""
ShenlunForge — Task-Driven Context Compression for LLM Agents
=============================================================

Built on a four-stage analytical compression strategy (ACS):
  Analyze → Retrieve → Process → Format

A lightweight, framework-agnostic library for intelligently compressing
conversation context before sending it to LLMs. Reduces token usage while
preserving task-relevant information.

Key Innovation:
  Unlike generic summarization or sliding window approaches, ShenlunForge uses
  a task-aware four-step pipeline that mimics how expert analysts process
  information: first understand what's needed, then extract relevant facts,
  merge duplicates with descriptive labels, and format within budget.

Quick Start:
    from shenlun_forge import ShenlunForge

    forge = ShenlunForge(model="deepseek-chat", api_key="your-key")
    result = await forge.compress(
        messages=[{"role": "user", "content": "..."}, ...],
        current_query="What did we decide?",
        max_tokens=4000,
    )
    print(result.compressed_text)
"""

from ._version import __version__

__author__ = "Liu Rongchen"

from .types import (
    ContextBlock,
    CompressedBlock,
    TaskProfile,
    TaskType,
    CompressionResult,
    CompressionPlan,
    InfoType,
    count_tokens,
    plan_compression,
)
from .compressor import ShenlunForge
from .llm import OpenAIProvider
from .strategies.simple import SimpleStrategy
from .strategies.hybrid import HCSCompressor
from .registry import get_strategy_class, list_strategies, register_strategy
from .router import recommend_retriever, RetrieverChoice
from .evaluation import EvalPoint, EvalReport, build_eval_set, evaluate

from .analyzer import ACSAnalyzer
from .retriever import ACSRetriever
from .processor import ACSProcessor
from .formatter import ACSFormatter

# EmbeddingRetriever requires optional deps (sentence-transformers, torch)
# Import lazily — don't fail if not installed
try:
    from .embedding_retriever import EmbeddingRetriever
    _HAS_EMBEDDING = True
except ImportError:
    EmbeddingRetriever = None  # type: ignore
    _HAS_EMBEDDING = False

__all__ = [
    "ShenlunForge",
    "OpenAIProvider",
    "HCSCompressor",
    "ACSAnalyzer",
    "ACSRetriever",
    "ACSProcessor",
    "ACSFormatter",
    "EmbeddingRetriever",
    "SimpleStrategy",
    "ContextBlock",
    "CompressedBlock",
    "TaskProfile",
    "TaskType",
    "CompressionResult",
    "CompressionPlan",
    "plan_compression",
    "InfoType",
    "count_tokens",
    "get_strategy_class",
    "list_strategies",
    "register_strategy",
    "recommend_retriever",
    "RetrieverChoice",
    "EvalPoint",
    "EvalReport",
    "build_eval_set",
    "evaluate",
]
