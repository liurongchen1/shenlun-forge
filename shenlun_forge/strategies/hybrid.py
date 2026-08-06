"""Hybrid Compression Strategy v2 — ACS + Verbatim Code Augmentation.

v1 (removed): Split messages by content type → route separately → merge.
  Problem: Lost cross-cutting context (security facts discussed in narrative
  but implemented in code). Result: 0/15 on Execution phase.

v2 (this): Run ACS on ALL messages for full semantic understanding,
  then augment with verbatim code blocks. Both layers complement each other.
  - ACS captures narrative reasoning, decisions, and high-level facts
  - Verbatim code preserves exact implementation details
  - Together they cover facts that either alone would miss
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

from ..types import (
    ContextBlock, CompressedBlock, TaskProfile,
    CompressionResult,
)
from ..base import LLMProvider
from ..llm import OpenAIProvider
from ..analyzer import ACSAnalyzer
from ..retriever import ACSRetriever
from ..processor import ACSProcessor
from ..formatter import ACSFormatter


class HCSCompressor:
    """
    ACS semantic summary + verbatim code augmentation.

    Pipeline:
      1. Extract code blocks from all messages (verbatim, no LLM)
      2. Calculate remaining token budget for narrative
      3. Run full ACS 4-step pipeline on ALL messages
      4. Merge: ACS summary + verbatim code blocks
      5. Quality gate: if over-compressed, fall back to Simple

    Usage:
        hcs = HCSCompressor(model="deepseek-chat", api_key="...", base_url="...")
        result = await hcs.compress(
            messages=[{"role": "user", "content": "..."}, ...],
            current_query="What security measures did we implement?",
            max_tokens=4000,
        )
    """

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        code_budget_ratio: float = 0.4,
        min_narrative_budget: int = 200,
        fallback_savings_threshold: float = 80.0,
        retriever_type: str = "rule",
    ):
        if llm is not None:
            self.llm = llm
        else:
            self.llm = OpenAIProvider(
                model=model,
                api_key=api_key,
                base_url=base_url,
            )

        # ACS pipeline
        self.analyzer = ACSAnalyzer(llm=self.llm)
        self.processor = ACSProcessor(llm=self.llm)
        self.formatter = ACSFormatter(llm=self.llm)

        # Retriever: rule-based or embedding
        if retriever_type == "embedding":
            from ..embedding_retriever import EmbeddingRetriever
            self.retriever = EmbeddingRetriever()
        else:
            self.retriever = ACSRetriever(llm=self.llm)

        self.code_budget_ratio = code_budget_ratio
        self.min_narrative_budget = min_narrative_budget
        self.fallback_savings_threshold = fallback_savings_threshold

    async def compress(
        self,
        messages: list[dict[str, str]],
        current_query: str = "",
        max_tokens: int = 4000,
        verbose: bool = False,
        recent_window: int = 6,
    ) -> CompressionResult:
        start_time = time.time()
        steps_log = {}

        # Parse messages
        blocks = self._parse_messages(messages)
        original_tokens = sum(b.token_count for b in blocks)

        # Recent window protection: keep last N messages uncompressed
        recent_window_size = min(recent_window, len(blocks))
        if recent_window_size > 0 and len(blocks) > recent_window_size:
            early_blocks = blocks[:-recent_window_size]
            recent_blocks = blocks[-recent_window_size:]
        else:
            early_blocks = blocks
            recent_blocks = []

        # ─── Step 1: Extract code blocks verbatim (from early blocks) ───
        t0 = time.time()
        code_blocks_text, code_token_count = self._extract_code_blocks(early_blocks)
        steps_log["code_extraction"] = {
            "duration_ms": round((time.time() - t0) * 1000),
            "code_tokens": code_token_count,
            "has_code": code_token_count > 0,
        }

        # ─── Step 2: Budget allocation ─────────────────────────────────
        # Reserve space for code (capped at code_budget_ratio of total)
        actual_code_tokens = min(
            code_token_count,
            int(max_tokens * self.code_budget_ratio),
        )
        narrative_budget = max(
            self.min_narrative_budget,
            max_tokens - actual_code_tokens,
        )

        steps_log["budget"] = {
            "total_budget": max_tokens,
            "code_budget": actual_code_tokens,
            "narrative_budget": narrative_budget,
        }

        # ─── Step 3: Run ACS on early messages ───────────────────────
        t0 = time.time()
        acs_text, acs_steps = await self._run_acs_pipeline(
            early_blocks, current_query, narrative_budget,
        )
        steps_log["acs"] = acs_steps
        acs_tokens = self.llm.count_tokens(acs_text) if acs_text else 0

        # ─── Step 4: Merge ACS + verbatim code + recent window ─────
        parts = []
        if acs_text.strip():
            parts.append(acs_text)
        if code_blocks_text.strip():
            # Truncate code if it exceeds budget
            truncated_code = code_blocks_text
            if actual_code_tokens < code_token_count:
                truncated_code = self._truncate_text(code_blocks_text, actual_code_tokens)
            parts.append("## Preserved Code Blocks")
            parts.append(truncated_code)
        if recent_blocks:
            recent_text = "\n".join(b.content for b in recent_blocks)
            parts.append("## Recent Messages (Uncompressed)")
            parts.append(recent_text)

        compressed_text = "\n\n".join(parts) if parts else ""
        compressed_tokens = self.llm.count_tokens(compressed_text)

        # ─── Step 5: Quality gate ──────────────────────────────────────
        savings_pct = (1 - compressed_tokens / max(original_tokens, 1)) * 100
        if savings_pct > self.fallback_savings_threshold:
            simple_result = await self._fallback_simple(messages, current_query, max_tokens)
            steps_log["fallback"] = {
                "triggered": True,
                "reason": f"savings {savings_pct:.1f}% > {self.fallback_savings_threshold}%",
            }
            simple_result.stats["hybrid_fallback"] = True
            simple_result.steps_log = steps_log if verbose else {}
            return simple_result

        # Build compressed blocks for result
        compressed_blocks = []
        if acs_text:
            compressed_blocks.append(CompressedBlock(
                content=acs_text,
                original_indices=list(range(len(early_blocks))),
                summary_label="acs_summary",
                preserved_token_count=sum(b.token_count for b in early_blocks),
                compression_ratio=acs_tokens / max(original_tokens, 1),
            ))
        if code_blocks_text.strip():
            compressed_blocks.append(CompressedBlock(
                content=code_blocks_text,
                original_indices=[],
                summary_label="verbatim_code",
                preserved_token_count=code_token_count,
                compression_ratio=1.0,
            ))
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

        return CompressionResult(
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / max(original_tokens, 1),
            blocks=compressed_blocks,
            stats={
                "strategy": "hcs",
                "total_duration_ms": round((time.time() - start_time) * 1000),
                "original_messages": len(messages),
                "acs_tokens": acs_tokens,
                "code_tokens": actual_code_tokens,
                "code_blocks_found": code_token_count > 0,
                "fallback_triggered": False,
            },
            steps_log=steps_log if verbose else {},
        )

    def compress_sync(
        self,
        messages: list[dict[str, str]],
        current_query: str = "",
        max_tokens: int = 4000,
        **kwargs,
    ) -> CompressionResult:
        return asyncio.run(self.compress(messages, current_query, max_tokens, **kwargs))

    # ─── Internal Methods ──────────────────────────────────────────────

    async def _run_acs_pipeline(
        self,
        blocks: list[ContextBlock],
        query: str,
        budget: int,
    ) -> tuple[str, dict]:
        """Run full ACS 4-step pipeline on ALL blocks."""
        steps = {}

        # Step 1: Analyze
        t0 = time.time()
        task_profile = await self.analyzer.analyze(blocks, query)
        task_profile.token_budget = budget
        steps["analyze"] = {
            "duration_ms": round((time.time() - t0) * 1000),
            "task_type": task_profile.task_type.value,
        }

        # Step 2: Retrieve
        t0 = time.time()
        relevant = await self.retriever.retrieve(blocks, task_profile)
        steps["retrieve"] = {
            "duration_ms": round((time.time() - t0) * 1000),
            "input": len(blocks),
            "output": len(relevant),
        }

        # Step 3: Process
        t0 = time.time()
        processed = await self.processor.process(relevant, task_profile)
        steps["process"] = {
            "duration_ms": round((time.time() - t0) * 1000),
            "groups": len(processed),
        }

        # Step 4: Format
        t0 = time.time()
        formatted = await self.formatter.format(processed, task_profile)
        steps["format"] = {
            "duration_ms": round((time.time() - t0) * 1000),
        }

        return formatted, steps

    def _extract_code_blocks(self, blocks: list[ContextBlock]) -> tuple[str, int]:
        """Extract all code blocks (```...```) from messages verbatim."""
        code_parts = []
        for block in blocks:
            # Find fenced code blocks
            fences = re.findall(r'(```[\s\S]*?```)', block.content)
            if fences:
                for fence in fences:
                    code_parts.append(fence)

        if not code_parts:
            return "", 0

        code_text = "\n".join(code_parts)
        code_tokens = self.llm.count_tokens(code_text)
        return code_text, code_tokens

    def _truncate_text(self, text: str, target_tokens: int) -> str:
        """Truncate text to approximately target_tokens."""
        # Rough: 1 token ≈ 4 chars for English, ~2 chars for Chinese
        # Use conservative estimate
        max_chars = target_tokens * 3
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"

    async def _fallback_simple(
        self,
        messages: list[dict[str, str]],
        query: str,
        max_tokens: int,
    ) -> CompressionResult:
        from .simple import SimpleStrategy
        simple = SimpleStrategy()
        return await simple.compress(
            messages=messages,
            current_query=query,
            max_tokens=max_tokens,
        )

    def _parse_messages(self, messages: list[dict[str, str]]) -> list[ContextBlock]:
        blocks = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            role = msg.get("role", "user")
            token_count = self.llm.count_tokens(content)
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
