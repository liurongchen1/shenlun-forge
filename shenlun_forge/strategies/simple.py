"""Simple compression strategy — no LLM required.

Uses basic heuristics: dedup, truncation, and priority-based selection.
Good for environments where no LLM is available for compression.
"""

from __future__ import annotations
import re
from typing import Optional

from ..types import (
    ContextBlock, CompressedBlock, TaskProfile, TaskType,
    CompressionResult, InfoType, count_tokens,
    MIN_TOKENS_TO_COMPRESS, MIN_COMPRESSION_GAIN,
    split_recent_window, truncate_to_tokens, validate_overflow_policy,
)


# A block that shares vocabulary with the query is worth much more than one
# that merely sits near the end of the history. The retriever's position
# signals are intentionally weak, so without this bonus the tail of a long
# conversation can crowd out the messages that actually answer the question.
_QUERY_MATCH_BONUS = 0.5


class SimpleStrategy:
    """Simple compression without LLM — uses rules and heuristics only.

    Implements the same compress/compress_sync interface as ShenlunForge
    and HCSCompressor, so it can be used as a drop-in replacement.

    Scoring is delegated to :class:`~shenlun_forge.retriever.ACSRetriever`,
    which is itself LLM-free: it matches Chinese text with 2-gram overlap,
    English terms with IDF weighting, and follows terms that bridge the two.
    That gives the rule-based strategy the same relevance signal as the full
    pipeline without any API call.
    """

    def __init__(self):
        from ..retriever import ACSRetriever
        # top_k is effectively disabled: the token budget decides what fits,
        # not a fixed candidate count.
        self._retriever = ACSRetriever(llm=None, top_k=10_000)

    @staticmethod
    def _terms(text: str) -> set:
        """Cheap bag of terms: English words (stop words dropped) + Chinese 2-grams."""
        from ..retriever import ACSRetriever

        stop = ACSRetriever._TECH_STOP_WORDS
        lowered = text.lower()
        terms = {
            word for word in re.findall(r"[a-z][a-z0-9_]{2,}", lowered)
            if word not in stop
        }
        for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
            for i in range(len(segment) - 1):
                terms.add(segment[i:i + 2])
        return terms

    def compress_sync(
        self,
        messages: list[dict[str, str]],
        current_query: str = "",
        max_tokens: int = 4000,
        **kwargs,
    ) -> CompressionResult:
        """Synchronous wrapper for compress()."""
        import asyncio
        return asyncio.run(self.compress(messages, current_query, max_tokens, **kwargs))

    async def compress(
        self,
        messages: list[dict[str, str]],
        current_query: str = "",
        max_tokens: int = 4000,
        budget_overflow: str = "keep",
        **kwargs,
    ) -> CompressionResult:
        """Compress using simple rules: dedup + priority + truncation.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            current_query: The current user query that drives compression.
            max_tokens: Maximum tokens for the compressed output.
            recent_window: Number of trailing messages kept verbatim. Defaults
                to ``max(3, len(messages) // 8)``. The most recent turns carry
                the live task state, so dropping them is almost always wrong.
            budget_overflow: ``"keep"`` (default) keeps the recent messages
                whole even if the result exceeds ``max_tokens``;
                ``"truncate"`` trims them to stay within it.

        Returns:
            CompressionResult with compressed text and statistics.
        """
        validate_overflow_policy(budget_overflow)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(
                f"max_tokens must be a positive integer, got {max_tokens!r}. If it "
                "came from plan_compression(), a None recommended_max_tokens means "
                "the history already fits — send it as is instead."
            )

        # Convert to blocks
        blocks = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            role = msg.get("role", "user")
            blocks.append(ContextBlock(
                content=content,
                role=role,
                msg_index=i,
                token_count=count_tokens(content),
            ))

        original_tokens = sum(b.token_count for b in blocks)

        # Step 0: Protect the most recent messages — they are never dropped or
        # truncated, only budgeted for (see split_recent_window).
        early_blocks, recent_blocks = split_recent_window(
            blocks, max_tokens, kwargs.get("recent_window") or 0
        )
        recent_tokens = sum(b.token_count for b in recent_blocks)
        budget = max(max_tokens - recent_tokens, max_tokens // 4)

        # Step 1: Deduplicate
        seen = set()
        unique_blocks = []
        for block in early_blocks:
            key = block.content[:200]  # Simple dedup key
            if key not in seen:
                seen.add(key)
                unique_blocks.append(block)

        duplicates_removed = len(early_blocks) - len(unique_blocks)

        # Step 2: Score by relevance to the query
        if unique_blocks:
            profile = TaskProfile(
                task_type=TaskType.AUTO,
                current_query=current_query,
            )
            await self._retriever.retrieve(unique_blocks, profile)

        query_terms = self._terms(current_query) if current_query else set()
        scored = []
        for block in unique_blocks:
            score = block.relevance_score
            shares_query_terms = bool(query_terms and (self._terms(block.content) & query_terms))
            if shares_query_terms:
                score += _QUERY_MATCH_BONUS
            scored.append((score, block, shares_query_terms))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # Step 3: Fill the budget, query-matching blocks first.
        #
        # Spending the whole budget on whatever sits near the end of the
        # history is how a "compressed" context ends up full of unrelated
        # chatter. Non-matching blocks only top up when the matches leave
        # more than half the budget unused.
        result_blocks = []
        used_tokens = 0

        def take(pairs, allow_truncate: bool) -> None:
            nonlocal used_tokens
            for _score, block, _shares in pairs:
                block_tokens = block.token_count
                if used_tokens + block_tokens <= budget:
                    result_blocks.append(CompressedBlock(
                        content=block.content,
                        original_indices=[block.msg_index],
                        summary_label=None,
                        info_type=None,
                        preserved_token_count=block_tokens,
                        compression_ratio=1.0,
                    ))
                    used_tokens += block_tokens
                elif allow_truncate:
                    remaining = budget - used_tokens
                    if remaining > 20:
                        truncated = self._truncate_to_tokens(block.content, remaining)
                        result_blocks.append(CompressedBlock(
                            content=truncated,
                            original_indices=[block.msg_index],
                            summary_label=None,
                            info_type=None,
                            preserved_token_count=block_tokens,
                            compression_ratio=remaining / max(block_tokens, 1),
                        ))
                        used_tokens += remaining
                    break
                else:
                    break

        matched_pairs = [pair for pair in scored if pair[2]]
        other_pairs = [pair for pair in scored if not pair[2]]

        take(matched_pairs, allow_truncate=True)
        # Only fall back to position-based selection when nothing matched the
        # query at all; padding a good result with unrelated chatter is worse
        # than returning a short one.
        if not matched_pairs:
            take(other_pairs, allow_truncate=False)

        # Step 4: Append the protected recent messages verbatim.
        # With budget_overflow="truncate" only what fits is kept, and the
        # newest message is trimmed rather than pushing past max_tokens.
        recent_truncated = False
        for block in recent_blocks:
            content = block.content
            if budget_overflow == "truncate":
                left = max_tokens - used_tokens
                if count_tokens(content) > left:
                    recent_truncated = True
                    content = truncate_to_tokens(content, left) if left > 20 else ""
                    if not content:
                        break
            result_blocks.append(CompressedBlock(
                content=content,
                original_indices=[block.msg_index],
                summary_label="recent",
                info_type=None,
                preserved_token_count=block.token_count,
                compression_ratio=1.0,
            ))
            used_tokens += count_tokens(content)

        # Sort by original position
        result_blocks.sort(key=lambda x: x.original_indices[0])

        compressed_text = "\n".join(b.content for b in result_blocks)

        # Same contract as the other strategies: if nothing meaningful was
        # saved (or the input is too small to be worth touching), hand the
        # original context back untouched.
        passthrough = (
            original_tokens == 0
            or original_tokens < MIN_TOKENS_TO_COMPRESS
            or original_tokens <= max_tokens
            or used_tokens >= original_tokens * (1 - MIN_COMPRESSION_GAIN)
        )

        if passthrough:
            result_blocks = [
                CompressedBlock(
                    content=b.content,
                    original_indices=[b.msg_index],
                    summary_label=b.role,
                    info_type=None,
                    preserved_token_count=b.token_count,
                    compression_ratio=1.0,
                )
                for b in blocks
            ]
            compressed_text = "\n".join(b.content for b in blocks)
            used_tokens = original_tokens
            recent_truncated = False

        return CompressionResult(
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=used_tokens,
            compression_ratio=used_tokens / max(original_tokens, 1),
            blocks=result_blocks,
            stats={
                "strategy": "simple",
                "original_messages": len(messages),
                "deduped_count": duplicates_removed,
                "recent_protected": len(recent_blocks),
                # Rule-based by design: it never needs an LLM, so it is never
                # "degraded" — the flags exist so callers can treat every
                # strategy result the same way.
                "llm_available": False,
                "degraded": False,
                "passthrough": passthrough,
                "over_budget": (not passthrough) and used_tokens > max_tokens * 1.02,
                "recent_truncated": recent_truncated,
            },
        )

    def _truncate_to_tokens(self, text: str, target_tokens: int) -> str:
        """Truncate text to approximately target_tokens.
        
        Encodes once, slices tokens directly, then decodes. 
        Much faster than binary-search with repeated encoding.
        """
        from ..types import _cached_encoder
        enc = _cached_encoder
        if enc is None:
            # Fallback: rough estimate ~3 chars per token
            cutoff = target_tokens * 3
            return text[:cutoff] + "..." if len(text) > cutoff else text
        
        token_ids = enc.encode(text)
        if len(token_ids) <= target_tokens:
            return text
        # Slice tokens directly and decode
        truncated = enc.decode(token_ids[:target_tokens])
        return truncated + "..."

