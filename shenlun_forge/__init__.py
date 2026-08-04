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

__version__ = "0.2.0"
__author__ = "Liu Rongchen"

from .types import (
    ContextBlock,
    CompressedBlock,
    TaskProfile,
    TaskType,
    CompressionResult,
    InfoType,
    count_tokens,
)
from .compressor import ShenlunForge
from .llm import OpenAIProvider
from .strategies.simple import SimpleStrategy
from .strategies.hybrid import HCSCompressor
from .registry import get_strategy_class, list_strategies

from .analyzer import ACSAnalyzer
from .retriever import ACSRetriever
from .processor import ACSProcessor
from .formatter import ACSFormatter

__all__ = [
    "ShenlunForge",
    "OpenAIProvider",
    "HCSCompressor",
    "ACSAnalyzer",
    "ACSRetriever",
    "ACSProcessor",
    "ACSFormatter",
    "SimpleStrategy",
    "ContextBlock",
    "CompressedBlock",
    "TaskProfile",
    "TaskType",
    "CompressionResult",
    "InfoType",
    "count_tokens",
    "get_strategy_class",
    "list_strategies",
]
