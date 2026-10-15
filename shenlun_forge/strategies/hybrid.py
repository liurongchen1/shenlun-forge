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
    CompressionResult, MIN_TOKENS_TO_COMPRESS, MIN_COMPRESSION_GAIN,
    split_recent_window, render_recent_blocks, validate_overflow_policy,
    render_dropped_notice, DROPPED_NOTICE_RESERVE_TOKENS,
)
from ..base import LLMProvider
from ..llm import OpenAIProvider, InstrumentedProvider, probe_llm
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

        # Probe once so callers can be told when the pipeline degrades.
        self.llm_ready, self.llm_unavailable_reason = probe_llm(self.llm)
        self._llm_stats = {"calls": 0, "failures": 0}
        active_llm = (
            InstrumentedProvider(self.llm, self._llm_stats)
            if self.llm_ready
            else None
        )

        # ACS pipeline
        self.analyzer = ACSAnalyzer(llm=active_llm)
        self.processor = ACSProcessor(llm=active_llm)
        self.formatter = ACSFormatter(llm=active_llm)

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
        recent_window: int = 0,
        budget_overflow: str = "keep",
        dropped_notice: bool = True,
    ) -> CompressionResult:
        """Compress. See ShenlunForge.compress for the argument semantics."""
        validate_overflow_policy(budget_overflow)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(
                f"max_tokens must be a positive integer, got {max_tokens!r}. If it "
                "came from plan_compression(), a None recommended_max_tokens means "
                "the history already fits — send it as is instead."
            )
        start_time = time.time()
        steps_log = {}
        recent_truncated = False

        # Per-call LLM instrumentation (reset so stats describe this run only)
        self._llm_stats["calls"] = 0
        self._llm_stats["failures"] = 0

        # Parse messages
        blocks = self._parse_messages(messages)
        original_tokens = sum(b.token_count for b in blocks)

        # Recent-message protection, budget-aware (see split_recent_window).
        early_blocks, recent_blocks = split_recent_window(blocks, max_tokens, recent_window)
        recent_tokens = sum(b.token_count for b in recent_blocks)

        # ─── Step 1: Extract code blocks verbatim (from early blocks) ───
        t0 = time.time()
        code_blocks_text, code_token_count = self._extract_code_blocks(early_blocks)
        steps_log["code_extraction"] = {
            "duration_ms": round((time.time() - t0) * 1000),
            "code_tokens": code_token_count,
            "has_code": code_token_count > 0,
        }

        # Hold back a few tokens so the "what was left out" note appended at
        # the end cannot push the result past max_tokens.
        notice_reserve = DROPPED_NOTICE_RESERVE_TOKENS if dropped_notice else 0

        # ─── Step 2: Budget allocation ─────────────────────────────────
        # Reserve space for code (capped at code_budget_ratio of total)
        actual_code_tokens = min(
            code_token_count,
            int(max_tokens * self.code_budget_ratio),
        )
        # The protected recent messages are appended verbatim, so their cost
        # has to come out of the budget too — otherwise the result exceeds it.
        narrative_budget = max(
            self.min_narrative_budget,
            max_tokens - actual_code_tokens - recent_tokens - notice_reserve,
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

        compressed_text = "\n\n".join(parts) if parts else ""

        # Append the protected recent messages, honouring the overflow policy.
        if recent_blocks:
            remaining = (
                max_tokens - notice_reserve - self.llm.count_tokens(compressed_text)
            )
            appended, truncated = render_recent_blocks(
                recent_blocks, remaining, budget_overflow
            )
            if appended:
                compressed_text = compressed_text + appended
            recent_truncated = recent_truncated or truncated

        compressed_tokens = self.llm.count_tokens(compressed_text)

        # ─── Step 5: Quality gate ──────────────────────────────────────
        savings_pct = (1 - compressed_tokens / max(original_tokens, 1)) * 100
        if savings_pct > self.fallback_savings_threshold:
            simple_result = await self._fallback_simple(
                messages, current_query, max_tokens, dropped_notice
            )
            steps_log["fallback"] = {
                "triggered": True,
                "reason": f"savings {savings_pct:.1f}% > {self.fallback_savings_threshold}%",
            }
            simple_result.stats["hybrid_fallback"] = True
            simple_result.stats["llm_available"] = self.llm_ready
            simple_result.stats["degraded"] = (
                (not self.llm_ready) or self._llm_stats["failures"] > 0
            )
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

        # ── Passthrough guard ────────────────────────────────────────
        # Hand back the original context when compression saves nothing
        # meaningful, so the caller never receives a longer "compressed" text.
        passthrough = False
        if original_tokens == 0 or (
            original_tokens > 0
            and (
                original_tokens < MIN_TOKENS_TO_COMPRESS
                or original_tokens <= max_tokens
                or compressed_tokens >= original_tokens * (1 - MIN_COMPRESSION_GAIN)
            )
        ):
            compressed_text = "\n".join(b.content for b in blocks)
            # Output is the original context — report the original token count
            # so a no-op run never shows a ratio above 1.0.
            compressed_tokens = original_tokens
            compressed_blocks = [
                CompressedBlock(
                    content=b.content,
                    original_indices=[b.msg_index],
                    summary_label=b.role,
                    info_type=b.info_type,
                    preserved_token_count=b.token_count,
                    compression_ratio=1.0,
                )
                for b in blocks
            ]
            passthrough = True

        # ── Say what was left out ─────────────────────────────────────────
        # The ACS summary is what covers the early messages; when it came back
        # empty those messages are genuinely gone, and the model needs to know
        # rather than answer as if it had read the whole conversation.
        if passthrough:
            kept_indices = {b.msg_index for b in blocks}
        else:
            kept_indices = {b.msg_index for b in recent_blocks}
            if acs_text.strip():
                kept_indices |= {b.msg_index for b in early_blocks}

        dropped_blocks = [b for b in blocks if b.msg_index not in kept_indices]
        dropped_messages = len(dropped_blocks)
        dropped_tokens = sum(b.token_count for b in dropped_blocks)

        if dropped_notice and not passthrough:
            language_hint = current_query or (
                messages[-1].get("content", "") if messages else ""
            )
            notice = render_dropped_notice(
                dropped_messages, dropped_tokens, language_hint
            )
            if notice:
                compressed_text += notice
                compressed_tokens = self.llm.count_tokens(compressed_text)

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
                "llm_available": self.llm_ready,
                "llm_calls": self._llm_stats["calls"],
                "llm_failures": self._llm_stats["failures"],
                "degraded": (not self.llm_ready) or self._llm_stats["failures"] > 0,
                "degraded_reason": (
                    self.llm_unavailable_reason
                    if not self.llm_ready
                    else (
                        f"{self._llm_stats['failures']} LLM call(s) failed at runtime; "
                        "rule-based fallback was used"
                        if self._llm_stats["failures"]
                        else ""
                    )
                ),
                "passthrough": passthrough,
                "over_budget": (not passthrough)
                and compressed_tokens > max_tokens * 1.02,
                "recent_truncated": recent_truncated,
                "dropped_messages": dropped_messages,
                "dropped_tokens": dropped_tokens,
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
        """Synchronous wrapper that also closes the LLM client afterwards."""

        async def _run() -> CompressionResult:
            try:
                return await self.compress(messages, current_query, max_tokens, **kwargs)
            finally:
                closer = getattr(self.llm, "aclose", None)
                if closer is not None:
                    await closer()

        return asyncio.run(_run())

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
        """Truncate text to at most target_tokens.

        This used to slice by ``target_tokens * 3`` characters, which is only
        roughly right for English prose. On Chinese text three characters is
        closer to 1.5-3 tokens, so the "truncated" code block could come out at
        twice the budget it was given and push the whole result over
        ``max_tokens``.
        """
        count = self.llm.count_tokens
        if count(text) <= target_tokens:
            return text

        low, high = 0, len(text)
        while low < high - 10:
            mid = (low + high) // 2
            if count(text[:mid]) <= target_tokens:
                low = mid
            else:
                high = mid
        return text[:low] + "\n...[truncated]"

    async def _fallback_simple(
        self,
        messages: list[dict[str, str]],
        query: str,
        max_tokens: int,
        dropped_notice: bool = True,
    ) -> CompressionResult:
        from .simple import SimpleStrategy
        simple = SimpleStrategy()
        return await simple.compress(
            messages=messages,
            current_query=query,
            max_tokens=max_tokens,
            dropped_notice=dropped_notice,
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
