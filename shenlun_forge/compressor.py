"""ShenlunForge — Main orchestrator for task-driven context compression.

Built on a four-stage analytical compression strategy (ACS):
  Analyze → Retrieve → Process → Format

This is the main entry point. It orchestrates the four-step compression pipeline.
"""

from __future__ import annotations
import asyncio
import time
import warnings
from typing import Any, Optional

from .types import (
    ContextBlock, CompressedBlock, TaskProfile, TaskType,
    CompressionResult, InfoType, MIN_TOKENS_TO_COMPRESS, MIN_COMPRESSION_GAIN,
    split_recent_window, render_recent_blocks, validate_overflow_policy,
    CompressionPlan, plan_compression,
)
from .base import LLMProvider, BaseAnalyzer, BaseRetriever, BaseProcessor, BaseFormatter
from .llm import OpenAIProvider, InstrumentedProvider, probe_llm
from .analyzer import ACSAnalyzer
from .retriever import ACSRetriever
from .processor import ACSProcessor
from .formatter import ACSFormatter


def _validate_max_tokens(max_tokens) -> int:
    """Reject a missing or nonsensical budget with a usable message.

    ``plan_compression()`` returns ``None`` when the history already fits, and
    passing that straight through used to fail deep inside as
    "unsupported operand type(s) for *: 'NoneType' and 'float'".
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError(
            f"max_tokens must be a positive integer, got {max_tokens!r}. If it "
            "came from plan_compression(), a None recommended_max_tokens means "
            "the history already fits — send it as is instead."
        )
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens!r}")
    return max_tokens


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

        # Probe once so we can tell the caller when the pipeline is about to
        # silently degrade to rule-based compression.
        self.llm_ready, self.llm_unavailable_reason = probe_llm(self.llm)
        self._llm_stats = {"calls": 0, "failures": 0}

        # Only hand a provider to the LLM-driven steps when it can actually be
        # used. Otherwise let them take their rule-based paths directly rather
        # than paying for a doomed request on every step.
        active_llm = (
            InstrumentedProvider(self.llm, self._llm_stats)
            if self.llm_ready
            else None
        )

        # Set up pipeline components (default to ACS strategy)
        self.analyzer = analyzer or ACSAnalyzer(llm=active_llm)
        self.processor = processor or ACSProcessor(llm=active_llm)
        self.formatter = formatter or ACSFormatter(llm=active_llm)

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
        recent_window: int = 0,
        budget_overflow: str = "keep",
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
            recent_window: Number of trailing messages kept verbatim. Defaults
                to ``max(3, len(messages) // 8)``, capped so it cannot claim
                more than 40% of the budget.
            budget_overflow: What to do when the protected recent messages do
                not fit in the remaining budget.

                - ``"keep"`` (default): keep them whole and let the result
                  exceed ``max_tokens``. The newest turns carry the live task
                  state, so losing them is usually worse than overshooting a
                  budget. The overshoot is reported as
                  ``stats["over_budget"]``.
                - ``"truncate"``: treat ``max_tokens`` as a hard limit, trimming
                  the newest message if necessary. Cheaper to reason about when
                  you are metering cost or the target model's window; costs you
                  the tail of the most recent turn. Reported as
                  ``stats["recent_truncated"]``.

        Returns:
            CompressionResult with compressed text and statistics.
        """
        validate_overflow_policy(budget_overflow)
        max_tokens = _validate_max_tokens(max_tokens)
        start_time = time.time()
        steps_log = {}
        recent_truncated = False

        # Per-call LLM instrumentation (reset so stats describe this run only)
        self._llm_stats["calls"] = 0
        self._llm_stats["failures"] = 0

        # Convert raw messages to ContextBlocks
        context_blocks = self._parse_messages(messages)
        original_tokens = sum(b.token_count for b in context_blocks)

        # Recent-message protection: the newest turns are kept verbatim, but
        # the window shrinks so it cannot eat the whole budget (see
        # split_recent_window).
        early_blocks, recent_blocks = split_recent_window(
            context_blocks, max_tokens, recent_window
        )

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
            steps_log["step3_process"] = {"skipped": "lightweight mode"}

            # Step 4: Format (early blocks only — recent blocks bypass formatter)
            recent_tokens = sum(b.token_count for b in recent_blocks) if recent_blocks else 0
            task_profile.token_budget = max(max_tokens - recent_tokens, max_tokens // 4)
            step_start = time.time()
            compressed_text = await self.formatter.format(compressed_blocks, task_profile)

            # Step 4b: Append the protected recent messages
            if recent_blocks:
                remaining = max_tokens - self.llm.count_tokens(compressed_text)
                appended, truncated = render_recent_blocks(
                    recent_blocks, remaining, budget_overflow
                )
                if appended:
                    compressed_text = compressed_text + appended
                recent_truncated = recent_truncated or truncated
                for b in recent_blocks:
                    compressed_blocks.append(CompressedBlock(
                        content=b.content,
                        original_indices=[b.msg_index],
                        summary_label="近期消息",
                        info_type=b.info_type,
                        preserved_token_count=b.token_count,
                        compression_ratio=1.0,
                    ))

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

            # Step 3b: Reserve token budget for recent window blocks
            recent_tokens = sum(b.token_count for b in recent_blocks) if recent_blocks else 0
            task_profile.token_budget = max(max_tokens - recent_tokens, max_tokens // 4)

            # Step 4: Format within token budget (early blocks only — recent bypass formatter)
            step_start = time.time()
            compressed_text = await self.formatter.format(compressed_blocks, task_profile)

            # Step 4b: Append the protected recent messages
            if recent_blocks:
                remaining = max_tokens - self.llm.count_tokens(compressed_text)
                appended, truncated = render_recent_blocks(
                    recent_blocks, remaining, budget_overflow
                )
                if appended:
                    compressed_text = compressed_text + appended
                recent_truncated = recent_truncated or truncated
                for b in recent_blocks:
                    compressed_blocks.append(CompressedBlock(
                        content=b.content,
                        original_indices=[b.msg_index],
                        summary_label="近期消息",
                        info_type=b.info_type,
                        preserved_token_count=b.token_count,
                        compression_ratio=1.0,
                    ))

            compressed_tokens = self.llm.count_tokens(compressed_text)
            steps_log["step4_format"] = {
                "duration_ms": round((time.time() - step_start) * 1000),
                "output_tokens": compressed_tokens,
            }

        # ── Passthrough guard ────────────────────────────────────────
        # Only return a compressed payload when it actually saves tokens.
        # Short conversations can otherwise come out longer than they went
        # in, so fall back to the original text untouched.
        passthrough = False
        if original_tokens == 0 or (
            original_tokens > 0
            and (
                original_tokens < MIN_TOKENS_TO_COMPRESS
                # It already fits: compressing can only lose information and
                # save nothing the caller asked for.
                or original_tokens <= max_tokens
                or compressed_tokens >= original_tokens * (1 - MIN_COMPRESSION_GAIN)
            )
        ):
            compressed_text = "\n".join(b.content for b in context_blocks)
            # The output is the original context, so report its token count
            # rather than re-measuring: re-tokenizing can shift boundaries
            # and push the ratio above 1.0 for a no-op compression.
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
                for b in context_blocks
            ]
            passthrough = True

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
                "llm_available": self.llm_ready,
                "llm_calls": self._llm_stats["calls"],
                "llm_failures": self._llm_stats["failures"],
                "degraded": self._is_degraded(),
                "degraded_reason": self._degradation_reason(),
                "passthrough": passthrough,
                # True when a real compression still came out over budget —
                # surfaced rather than hidden, since max_tokens is a promise.
                # A couple of tokens of slack: labels, separators and
                # re-tokenisation boundaries always cost something.
                "over_budget": (not passthrough)
                and compressed_tokens > max_tokens * 1.02,
                # Only ever True under budget_overflow="truncate": the newest
                # message could not be kept whole.
                "recent_truncated": recent_truncated,
            },
            steps_log=steps_log if verbose else {},
        )

        return result

    def _is_degraded(self) -> bool:
        """True when this run did not actually use the LLM as configured."""
        return (not self.llm_ready) or self._llm_stats["failures"] > 0

    def _degradation_reason(self) -> str:
        """Human-readable explanation for ``stats['degraded']``."""
        if not self.llm_ready:
            return self.llm_unavailable_reason
        if self._llm_stats["failures"]:
            return (
                f"{self._llm_stats['failures']} LLM call(s) failed at runtime; "
                "rule-based fallback was used"
            )
        return ""

    def plan(
        self,
        messages: list[dict[str, str]],
        model_window: int = 128_000,
        safety_margin: float = 0.1,
    ) -> CompressionPlan:
        """Decide whether to compress at all, and with how much budget.

        Convenience wrapper around :func:`shenlun_forge.types.plan_compression`,
        which you can also call directly (it needs no instance and no LLM).
        """
        return plan_compression(messages, model_window, safety_margin)

    def compress_sync(
        self,
        messages: list[dict[str, str]],
        current_query: str = "",
        max_tokens: int = 4000,
        **kwargs,
    ) -> CompressionResult:
        """Synchronous wrapper for compress().

        Each call runs in its own event loop, so the LLM client is closed
        before that loop goes away — otherwise httpx reports a closed loop on
        every subsequent call.
        """

        async def _run() -> CompressionResult:
            try:
                return await self.compress(messages, current_query, max_tokens, **kwargs)
            finally:
                await self._aclose_llm()

        return asyncio.run(_run())

    async def _aclose_llm(self) -> None:
        """Close the LLM client, if the provider supports it."""
        closer = getattr(self.llm, "aclose", None)
        if closer is not None:
            await closer()

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
        strategy: str = "simple",
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
                - "simple": Rule-based selection (default). No LLM, no API key,
                  runs in milliseconds.
                - "acs": Full 4-step ACS pipeline. Requires an LLM.
                - "hcs": ACS + verbatim code augmentation. Requires an LLM.
            model: LLM model name.
            api_key: API key for the LLM.
            base_url: Base URL for the LLM API.

        Note:
            Strategies that need an LLM ("acs", "hcs") emit a warning when no
            usable LLM is configured, because they would otherwise degrade to
            rule-based compression without telling you.
        """
        from .registry import get_strategy_class

        strategy_cls = get_strategy_class(strategy)

        # ShenlunForge (acs) takes model/api_key/base_url directly
        if strategy_cls is cls:
            obj = cls(model=model, api_key=api_key, base_url=base_url, **kwargs)
        else:
            # SimpleStrategy, HCSCompressor, etc.
            try:
                obj = strategy_cls(
                    model=model, api_key=api_key, base_url=base_url, **kwargs
                )
            except TypeError:
                # SimpleStrategy takes no args
                if kwargs:
                    warnings.warn(
                        f"strategy='{strategy}' ignores the extra options "
                        f"{sorted(kwargs)}; they only apply to the LLM "
                        "strategies ('acs', 'hcs').",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                obj = strategy_cls()

        if getattr(obj, "llm_ready", True) is False:
            warnings.warn(
                f"strategy='{strategy}' needs an LLM, but "
                f"{getattr(obj, 'llm_unavailable_reason', 'no usable LLM is available')}. "
                "Compression will fall back to rule-based mode. "
                'Install it with pip install "shenlun-forge[openai]" and set an '
                "API key, or use strategy='simple' to avoid calling an LLM.",
                RuntimeWarning,
                stacklevel=2,
            )

        return obj
