"""Simple compression strategy — no LLM required.

Uses basic heuristics: dedup, truncation, and priority-based selection.
Good for environments where no LLM is available for compression.
"""

from __future__ import annotations
from typing import Optional

from ..types import (
    ContextBlock, CompressedBlock, TaskProfile, TaskType,
    CompressionResult, InfoType, count_tokens,
)


class SimpleStrategy:
    """Simple compression without LLM — uses rules and heuristics only.
    
    Implements the same compress/compress_sync interface as ShenlunForge
    and HCSCompressor, so it can be used as a drop-in replacement.
    """

    def __init__(self):
        pass

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
        **kwargs,
    ) -> CompressionResult:
        """Compress using simple rules: dedup + priority + truncation.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            current_query: The current user query that drives compression.
            max_tokens: Maximum tokens for the compressed output.
        
        Returns:
            CompressionResult with compressed text and statistics.
        """

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

        # Step 1: Deduplicate
        seen = set()
        unique_blocks = []
        for block in blocks:
            key = block.content[:200]  # Simple dedup key
            if key not in seen:
                seen.add(key)
                unique_blocks.append(block)

        # Step 2: Score by priority
        scored = []
        for i, block in enumerate(unique_blocks):
            score = self._score(block, current_query, i, len(unique_blocks))
            scored.append((score, block))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Step 3: Greedily fill budget
        result_blocks = []
        used_tokens = 0
        for score, block in scored:
            block_tokens = block.token_count
            if used_tokens + block_tokens <= max_tokens:
                result_blocks.append(CompressedBlock(
                    content=block.content,
                    original_indices=[block.msg_index],
                    summary_label=None,
                    info_type=None,
                    preserved_token_count=block_tokens,
                    compression_ratio=1.0,
                ))
                used_tokens += block_tokens
            else:
                # Partial inclusion
                remaining = max_tokens - used_tokens
                if remaining > 20:
                    # Truncate using token-aware estimation
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

        # Sort by original position
        result_blocks.sort(key=lambda x: x.original_indices[0])

        compressed_text = "\n".join(b.content for b in result_blocks)

        return CompressionResult(
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=used_tokens,
            compression_ratio=used_tokens / max(original_tokens, 1),
            blocks=result_blocks,
            stats={
                "strategy": "simple",
                "original_messages": len(messages),
                "deduped_count": len(blocks) - len(unique_blocks),
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

    def _score(
        self,
        block: ContextBlock,
        query: str,
        position: int,
        total: int,
    ) -> float:
        """Score a block for relevance."""
        score = 0.0

        # Keyword overlap
        if query:
            query_words = set(query.lower().split())
            content_words = set(block.content.lower().split())
            overlap = len(query_words & content_words) / max(len(query_words), 1)
            score += overlap * 0.4

        # Recency bonus
        score += (position / max(total - 1, 1)) * 0.3

        # Role bonus
        role_bonus = {"user": 0.8, "assistant": 0.6, "tool": 0.9, "system": 0.2}
        score += role_bonus.get(block.role, 0.4) * 0.3

        # Length penalty (prefer concise blocks)
        if block.token_count > 500:
            score -= 0.1

        return score
