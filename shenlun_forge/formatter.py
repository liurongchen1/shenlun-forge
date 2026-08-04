"""Step 4: Formatter — Final output formatting within token budget.

Maps to the Format step: format the final answer within word limits.
Ensures output fits within the token budget while preserving key information.
"""

from __future__ import annotations
from typing import Optional

from .base import BaseFormatter
from .types import CompressedBlock, TaskProfile, TaskType, InfoType, count_tokens


class ACSFormatter(BaseFormatter):
    """Formatter that produces final compressed context within token budget."""

    def __init__(self, llm=None):
        self.llm = llm

    async def format(
        self,
        blocks: list[CompressedBlock],
        task_profile: TaskProfile,
    ) -> str:
        """Format compressed blocks into final output within token budget."""

        if not blocks:
            return ""

        token_budget = task_profile.token_budget
        output_format = task_profile.output_format

        # Sort blocks by priority (relevance score from original blocks)
        blocks = sorted(blocks, key=lambda x: self._block_priority(x), reverse=True)

        if output_format == "structured":
            result = self._format_structured(blocks, token_budget)
        elif output_format == "bullet":
            result = self._format_bullet(blocks, token_budget)
        else:  # sequential
            result = self._format_sequential(blocks, token_budget)

        return result

    def _format_sequential(
        self,
        blocks: list[CompressedBlock],
        token_budget: int,
    ) -> str:
        """Format blocks in sequential order, fitting within budget."""
        parts = []
        used_tokens = 0

        for block in blocks:
            content = block.content
            label_prefix = f"【{block.summary_label}】" if block.summary_label else ""
            entry = f"{label_prefix}{content}" if label_prefix else content

            entry_tokens = count_tokens(entry)

            if used_tokens + entry_tokens > token_budget:
                # Truncate this block to fit remaining budget
                remaining = token_budget - used_tokens
                if remaining > 20:  # At least 20 tokens worth
                    entry = self._truncate_to_budget(entry, remaining)
                    parts.append(entry)
                break

            parts.append(entry)
            used_tokens += entry_tokens

        return "\n".join(parts)

    def _format_structured(
        self,
        blocks: list[CompressedBlock],
        token_budget: int,
    ) -> str:
        """Format blocks grouped by info type (structured output)."""
        from collections import defaultdict

        groups = defaultdict(list)
        for block in blocks:
            type_label = block.summary_label or "其他"
            groups[type_label].append(block)

        parts = []
        used_tokens = 0

        for label, group_blocks in groups.items():
            header = f"## {label}"
            header_tokens = count_tokens(header)

            if used_tokens + header_tokens > token_budget:
                break

            parts.append(header)
            used_tokens += header_tokens

            for block in group_blocks:
                entry = f"- {block.content}"
                entry_tokens = count_tokens(entry)

                if used_tokens + entry_tokens > token_budget:
                    remaining = token_budget - used_tokens
                    if remaining > 10:
                        entry = self._truncate_to_budget(entry, remaining)
                        parts.append(entry)
                    break

                parts.append(entry)
                used_tokens += entry_tokens

        return "\n".join(parts)

    def _format_bullet(
        self,
        blocks: list[CompressedBlock],
        token_budget: int,
    ) -> str:
        """Format blocks as bullet points."""
        parts = []
        used_tokens = 0

        for block in blocks:
            label = block.summary_label or ""
            content = block.content

            if label:
                entry = f"• **{label}**: {content}"
            else:
                entry = f"• {content}"

            entry_tokens = count_tokens(entry)

            if used_tokens + entry_tokens > token_budget:
                remaining = token_budget - used_tokens
                if remaining > 10:
                    entry = self._truncate_to_budget(entry, remaining)
                    parts.append(entry)
                break

            parts.append(entry)
            used_tokens += entry_tokens

        return "\n".join(parts)

    def _truncate_to_budget(self, text: str, target_tokens: int) -> str:
        """Truncate text to approximately target_tokens using binary search."""
        if count_tokens(text) <= target_tokens:
            return text
        low, high = 0, len(text)
        while low < high - 10:
            mid = (low + high) // 2
            if count_tokens(text[:mid]) <= target_tokens:
                low = mid
            else:
                high = mid
        return text[:low] + "..."

    def _block_priority(self, block: CompressedBlock) -> float:
        """Calculate priority score for block ordering.
        
        Uses InfoType enum values as keys to correctly map block.info_type
        to priority weights.
        """
        priority = 0.0

        # Higher compression ratio = was more bloated, lower priority
        # (we already extracted the essence, so less compressed = denser info)
        if block.compression_ratio > 0:
            priority += (1 - block.compression_ratio) * 0.3

        # Info type priority — keyed by InfoType enum values
        type_priority = {
            InfoType.FACT: 0.95,        # Factual context is highest priority
            InfoType.DATA: 0.90,        # Data/statistics very important
            InfoType.PROBLEM: 0.85,     # Problems/issues critical to preserve
            InfoType.ACTION: 0.80,      # Actions/decisions taken
            InfoType.CAUSE: 0.75,       # Causal reasoning
            InfoType.EFFECT_POS: 0.65,  # Positive outcomes
            InfoType.EFFECT_NEG: 0.65,  # Negative outcomes
            InfoType.OPINION: 0.55,     # Opinions/judgments
            InfoType.OTHER: 0.20,       # Other info is low priority
        }

        if block.info_type:
            info_type = block.info_type if isinstance(block.info_type, InfoType) else InfoType(block.info_type)
            priority += type_priority.get(info_type, 0.30) * 0.4

        # Original position bonus (earlier = slightly higher for context)
        if block.original_indices:
            position_bonus = 1.0 / (1 + block.original_indices[0] * 0.1)
            priority += position_bonus * 0.3

        return priority
