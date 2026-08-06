"""ShenlunForge — Main orchestrator for task-driven context compression.

Built on a four-stage analytical compression strategy (ACS):
  Analyze → Retrieve → Process → Format

This is the main entry point. It orchestrates the four-step compression pipeline.
"""

from __future__ import annotations
import asyncio
import time
from typing import Any, Optional

from .types import (
    ContextBlock, CompressedBlock, TaskProfile, TaskType,
    CompressionResult, InfoType,
)
from .base import LLMProvider, BaseAnalyzer, BaseRetriever, BaseProcessor, BaseFormatter
from .llm import OpenAIProvider
from .analyzer import ACSAnalyzer
from .retriever import ACSRetriever
from .processor import ACSProcessor
from .formatter import ACSFormatter


class ShenlunForge:
    """
    Task-driven context compression engine.

    Usage:
        forge = ShenlunForge(llm=OpenAIProvider(model="deepseek-chat", ...))
        result = await forge.compress(
            messages=[{"role": "user", "content": "..."}, ...],
            current_query="What did we decide about the database?",
            max_tokens=4000,
        )
        print(result.compressed_text)
        print(f"Compression ratio: {result.compression_ratio:.1%}")
    """

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        analyzer: Optional[BaseAnalyzer] = None,
        retriever: Optional[BaseRetriever] = None,
        processor: Optional[BaseProcessor] = None,
        formatter: Optional[BaseFormatter] = None,
        # Convenience params
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # Retriever type: "rule" (default) or "embedding"
        retriever_type: str = "rule",
    ):
        # Set up LLM provider
        if llm is not None:
            self.llm = llm
        else:
            self.llm = OpenAIProvider(
                model=model,
                api_key=api_key,
                base_url=base_url,
            )

        # Set up pipeline components (default to ACS strategy)
        self.analyzer = analyzer or ACSAnalyzer(llm=self.llm)
        self.processor = processor or ACSProcessor(llm=self.llm)
        self.formatter = formatter or ACSFormatter(llm=self.llm)

        # Retriever: use provided instance, or create based on type
        if retriever is not None:
            self.retriever = retriever
        elif retriever_type == "embedding":
            from .embedding_retriever import EmbeddingRetriever
            self.retriever = EmbeddingRetriever()
        else:
            self.retriever = ACSRetriever(llm=self.llm)

    async def compress(
        self,
        messages: list[dict[str, str]],
        current_query: str = "",
        max_tokens: int = 4000,
        task_type: Optional[str] = None,
        verbose: bool = False,
        lightweight: bool = False,
        recent_window: int = 6,
    ) -> CompressionResult:
        """
        Compress conversation context using the ACS 4-step pipeline.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            current_query: The current user query that drives compression.
            max_tokens: Maximum tokens for the compressed output.
            task_type: Force a specific task type (skip auto-detection).
            verbose: If True, include detailed step logs.
            lightweight: If True, skip LLM-dependent steps (Analyze, Process).
                Only Retrieve + Format runs. Requires EmbeddingRetriever.
                Zero LLM API calls, but no compression — just smart selection.

        Returns:
            CompressionResult with compressed text and statistics.
        """
        start_time = time.time()
        steps_log = {}

        # Convert raw messages to ContextBlocks
        context_blocks = self._parse_messages(messages)
        original_tokens = sum(b.token_count for b in context_blocks)

        # Recent window protection: keep last N messages uncompressed
        recent_window_size = min(recent_window, len(context_blocks))
        if recent_window_size > 0 and len(context_blocks) > recent_window_size:
            early_blocks = context_blocks[:-recent_window_size]
            recent_blocks = context_blocks[-recent_window_size:]
        else:
            early_blocks = context_blocks
            recent_blocks = []

        if lightweight:
            # ── Lightweight mode: no LLM calls ──────────────────────────
            # Skip Analyze: create minimal TaskProfile with just the query
            task_profile = TaskProfile(
                task_type=TaskType.CONVERSATION,
                token_budget=max_tokens,
                current_query=current_query,
            )
            steps_log["step1_analyze"] = {"skipped": "lightweight mode"}

            # Step 2: Retrieve (embedding-based, no LLM, from early blocks only)
            step_start = time.time()
            relevant_blocks = await self.retriever.retrieve(early_blocks, task_profile)
            steps_log["step2_retrieve"] = {
                "duration_ms": round((time.time() - step_start) * 1000),
                "input_blocks": len(early_blocks),
                "output_blocks": len(relevant_blocks),
                "filtered_out": len(early_blocks) - len(relevant_blocks),
                "recent_protected": len(recent_blocks),
            }

            # Skip Process: convert ContextBlocks to CompressedBlocks directly
            compressed_blocks = [
                CompressedBlock(
                    content=b.content,
                    original_indices=[b.msg_index],
                    summary_label=b.role,
                    info_type=b.info_type,
                    preserved_token_count=b.token_count,
                    compression_ratio=1.0,
                )
                for b in relevant_blocks
            ]
            # Append recent window blocks as uncompressed
            if recent_blocks:
                for b in recent_blocks:
                    compressed_blocks.append(CompressedBlock(
                        content=b.content,
                        original_indices=[b.msg_index],
                        summary_label="近期消息",
                        info_type=b.info_type,
                        preserved_token_count=b.token_count,
                        compression_ratio=1.0,
                    ))
            steps_log["step3_process"] = {"skipped": "lightweight mode"}

            # Step 4: Format (concatenate + truncate to budget)
            step_start = time.time()
            compressed_text = await self.formatter.format(compressed_blocks, task_profile)
            compressed_tokens = self.llm.count_tokens(compressed_text)
            steps_log["step4_format"] = {
                "duration_ms": round((time.time() - step_start) * 1000),
                "output_tokens": compressed_tokens,
            }

        else:
            # ── Full mode: ACS 4-step pipeline with LLM ─────────────────
            # Step 1: Analyze the task
            step_start = time.time()
            if task_type:
                task_profile = TaskProfile(
                    task_type=TaskType(task_type),
                    token_budget=max_tokens,
                )
            else:
                task_profile = await self.analyzer.analyze(context_blocks, current_query)
                task_profile.token_budget = max_tokens
            steps_log["step1_analyze"] = {
                "duration_ms": round((time.time() - step_start) * 1000),
                "task_type": task_profile.task_type.value,
                "key_elements_count": len(task_profile.key_elements),
            }

            # Step 2: Retrieve and score relevant blocks (from early blocks only)
            step_start = time.time()
            relevant_blocks = await self.retriever.retrieve(early_blocks, task_profile)
            steps_log["step2_retrieve"] = {
                "duration_ms": round((time.time() - step_start) * 1000),
                "input_blocks": len(early_blocks),
                "output_blocks": len(relevant_blocks),
                "filtered_out": len(early_blocks) - len(relevant_blocks),
                "recent_protected": len(recent_blocks),
            }

            # Step 3: Process — Merge, label, normalize
            step_start = time.time()
            compressed_blocks = await self.processor.process(relevant_blocks, task_profile)
            steps_log["step3_process"] = {
                "duration_ms": round((time.time() - step_start) * 1000),
                "input_blocks": len(relevant_blocks),
                "output_groups": len(compressed_blocks),
                "merged_count": len(relevant_blocks) - len(compressed_blocks),
            }

            # Step 3b: Append recent window blocks as uncompressed
            if recent_blocks:
                for b in recent_blocks:
                    compressed_blocks.append(CompressedBlock(
                        content=b.content,
                        original_indices=[b.msg_index],
                        summary_label="近期消息",
                        info_type=b.info_type,
                        preserved_token_count=b.token_count,
                        compression_ratio=1.0,
                    ))

            # Step 4: Format within token budget
            step_start = time.time()
            compressed_text = await self.formatter.format(compressed_blocks, task_profile)
            compressed_tokens = self.llm.count_tokens(compressed_text)
            steps_log["step4_format"] = {
                "duration_ms": round((time.time() - step_start) * 1000),
                "output_tokens": compressed_tokens,
            }

        # Build result
        result = CompressionResult(
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / max(original_tokens, 1),
            blocks=compressed_blocks,
            stats={
                "total_duration_ms": round((time.time() - start_time) * 1000),
                "original_messages": len(messages),
                "original_blocks": len(context_blocks),
                "relevant_blocks": len(relevant_blocks),
                "compressed_groups": len(compressed_blocks),
            },
            steps_log=steps_log if verbose else {},
        )

        return result

    def compress_sync(
        self,
        messages: list[dict[str, str]],
        current_query: str = "",
        max_tokens: int = 4000,
        **kwargs,
    ) -> CompressionResult:
        """Synchronous wrapper for compress()."""
        return asyncio.run(self.compress(messages, current_query, max_tokens, **kwargs))

    def _parse_messages(self, messages: list[dict[str, str]]) -> list[ContextBlock]:
        """Convert raw message dicts to ContextBlocks."""
        blocks = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            role = msg.get("role", "user")
            token_count = self.llm.count_tokens(content)

            # Extract metadata if present
            metadata = {}
            if "name" in msg:
                metadata["name"] = msg["name"]
            if "tool_call_id" in msg:
                metadata["tool_call_id"] = msg["tool_call_id"]

            blocks.append(ContextBlock(
                content=content,
                role=role,
                msg_index=i,
                token_count=token_count,
                metadata=metadata,
            ))

        return blocks

    # ─── Convenience class methods ───────────────────────────────────────────

    @classmethod
    def create(
        cls,
        strategy: str = "acs",
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Factory method to create a compressor with a specific strategy.

        Uses the strategy registry for extensibility. To add a new strategy,
        register it in shenlun_forge.registry.

        Args:
            strategy: Compression strategy name. Use list_strategies() to see available options.
                - "acs": Full 4-step ACS pipeline (default)
                - "simple": Simple truncation (no LLM needed)
                - "hcs": ACS + verbatim code augmentation
            model: LLM model name.
            api_key: API key for the LLM.
            base_url: Base URL for the LLM API.
        """
        from .registry import get_strategy_class
        
        strategy_cls = get_strategy_class(strategy)
        
        # ShenlunForge (acs) takes model/api_key/base_url directly
        if strategy_cls is cls:
            return cls(model=model, api_key=api_key, base_url=base_url)
        else:
            # SimpleStrategy, HCSCompressor, etc.
            try:
                return strategy_cls(model=model, api_key=api_key, base_url=base_url)
            except TypeError:
                # SimpleStrategy takes no args
                return strategy_cls()
