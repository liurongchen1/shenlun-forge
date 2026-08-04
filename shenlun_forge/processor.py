"""Step 3: Processor — Merge, label, and normalize extracted information.

Maps to the Process step: merge similar items, extract summary labels, normalize expressions.
This is the core innovation of the ACS approach.
"""

from __future__ import annotations
import json
import re
from typing import Optional
from collections import defaultdict

from .base import BaseProcessor
from .types import ContextBlock, CompressedBlock, TaskProfile, TaskType, InfoType, count_tokens


MERGE_PROMPT = """You are a context compression processor. Your job is to merge similar/related context blocks and extract summary labels.

Given the following context blocks, group them by semantic similarity, then for each group:
1. Merge the content into a concise representation
2. Extract a "summary label" (总括词) - a short 2-6 character label that captures the essence
3. Normalize informal expressions to formal/technical language

Context blocks:
{blocks_text}

Respond in JSON:
{{
  "groups": [
    {{
      "block_indices": [0, 2, 5],
      "summary_label": "简要标签",
      "merged_content": "合并后的精简内容",
      "info_type": "fact|action|problem|cause|effect_pos|effect_neg|opinion|data|other"
    }}
  ]
}}

Rules:
- Each block index must appear in exactly one group
- Summary labels should be concise (2-6 chars) and descriptive
- Merged content should preserve all critical information while being more concise
- Normalize colloquial language to technical/formal expressions
"""

NORMALIZE_MAP = {
    # Common informal → formal mappings (Chinese)
    "不行": "不可用",
    "有问题": "存在异常",
    "出错了": "执行失败",
    "太慢": "性能不足",
    "搞不定": "无法完成",
    "弄好了": "已完成",
    "改一下": "已修改",
    "没问题": "验证通过",
    # English informal → formal
    "doesn't work": "non-functional",
    "too slow": "performance degraded",
    "fixed it": "resolved",
    "broken": "non-functional",
    "kind of": "partially",
    "a lot of": "multiple",
}


class ACSProcessor(BaseProcessor):
    """Processor that merges, labels, and normalizes context (Process step)."""

    def __init__(self, llm=None, enable_merge: bool = True, enable_normalize: bool = True):
        self.llm = llm
        self.enable_merge = enable_merge
        self.enable_normalize = enable_normalize

    async def process(
        self,
        blocks: list[ContextBlock],
        task_profile: TaskProfile,
    ) -> list[CompressedBlock]:
        """Process blocks: merge similar, add labels, normalize."""

        if not blocks:
            return []

        # Step 3a: Merge similar content (合并同类项)
        if self.enable_merge:
            merged_groups = await self._merge_blocks(blocks, task_profile)
        else:
            merged_groups = [[(i, b)] for i, b in enumerate(blocks)]

        # Step 3b: Build compressed blocks with labels
        compressed = []
        for group in merged_groups:
            indices = [i for i, _ in group]
            group_blocks = [b for _, b in group]

            # Determine summary label (总括词)
            label = self._extract_summary_label(group_blocks, task_profile)

            # Merge content
            merged_content = self._merge_content(group_blocks, task_profile)

            # Normalize expressions (口语转书面语)
            if self.enable_normalize:
                merged_content = self._normalize_expressions(merged_content)

            # Calculate compression metrics
            original_tokens = sum(b.token_count for b in group_blocks)
            compressed_tokens = count_tokens(merged_content)

            compressed.append(CompressedBlock(
                content=merged_content,
                original_indices=indices,
                summary_label=label,
                info_type=group_blocks[0].info_type if group_blocks else None,
                preserved_token_count=original_tokens,
                compression_ratio=compressed_tokens / max(original_tokens, 1),
            ))

        # Sort by original first index to maintain flow
        compressed.sort(key=lambda x: x.original_indices[0] if x.original_indices else 0)

        return compressed

    async def _merge_blocks(
        self,
        blocks: list[ContextBlock],
        task_profile: TaskProfile,
    ) -> list[list[tuple[int, ContextBlock]]]:
        """Merge semantically similar blocks into groups."""

        # If LLM available, use intelligent merging
        if self.llm is not None and len(blocks) > 1:
            try:
                return await self._llm_merge(blocks, task_profile)
            except Exception:
                pass

        # Fallback: rule-based merging
        return self._rule_based_merge(blocks, task_profile)

    async def _llm_merge(
        self,
        blocks: list[ContextBlock],
        task_profile: TaskProfile,
    ) -> list[list[tuple[int, ContextBlock]]]:
        """Use LLM to intelligently merge similar blocks."""
        blocks_text = ""
        for i, block in enumerate(blocks):
            preview = block.content[:300] + "..." if len(block.content) > 300 else block.content
            blocks_text += f"\n[Block {i}] (role={block.role}, type={block.info_type}):\n{preview}\n"

        prompt = MERGE_PROMPT.format(blocks_text=blocks_text)
        result = await self.llm.complete_json(prompt, {})

        groups_raw = result.get("groups", [])

        # Convert to indexed groups
        merged = []
        assigned = set()
        for group in groups_raw:
            indices = group.get("block_indices", [])
            valid_indices = [i for i in indices if 0 <= i < len(blocks) and i not in assigned]
            if valid_indices:
                merged.append([(i, blocks[i]) for i in valid_indices])
                assigned.update(valid_indices)

        # Add unassigned blocks as individual groups
        for i in range(len(blocks)):
            if i not in assigned:
                merged.append([(i, blocks[i])])

        return merged if merged else [[(i, b)] for i, b in enumerate(blocks)]

    def _rule_based_merge(
        self,
        blocks: list[ContextBlock],
        task_profile: TaskProfile,
    ) -> list[list[tuple[int, ContextBlock]]]:
        """Rule-based merging by info type and proximity."""
        groups = []
        current_group = []
        current_type = None

        for i, block in enumerate(blocks):
            # Start new group if type changes significantly or gap is large
            if current_group and (
                block.info_type != current_type or
                (i - current_group[-1][0]) > 3  # Gap of more than 3 messages
            ):
                groups.append(current_group)
                current_group = []

            current_group.append((i, block))
            current_type = block.info_type

        if current_group:
            groups.append(current_group)

        return groups

    def _extract_summary_label(
        self,
        blocks: list[ContextBlock],
        task_profile: TaskProfile,
    ) -> Optional[str]:
        """Extract a summary label (总括词) for a group of blocks.

        This is the key innovation — extracting a concise "hat" label
        that summarizes the essence of the content.
        """
        # Try to extract from content patterns
        all_content = " ".join(b.content for b in blocks)

        # Look for existing section headers or labels
        header_patterns = [
            r'(?:#{1,3}\s+)(.+)',          # Markdown headers
            r'(?:\*\*|__)(.+?)(?:\*\*|__)', # Bold text
            r'(?:^|\n)([^\n]{2,20}?)[:：]', # Lines ending with colon
        ]

        for pattern in header_patterns:
            match = re.search(pattern, all_content)
            if match:
                label = match.group(1).strip()
                if 2 <= len(label) <= 20:
                    return label

        # Generate label from info type
        type_labels = {
            InfoType.FACT: "事实信息",
            InfoType.ACTION: "操作记录",
            InfoType.PROBLEM: "问题描述",
            InfoType.CAUSE: "原因分析",
            InfoType.EFFECT_POS: "积极影响",
            InfoType.EFFECT_NEG: "消极影响",
            InfoType.OPINION: "观点建议",
            InfoType.DATA: "数据指标",
            InfoType.OTHER: "其他信息",
        }
        info_type = blocks[0].info_type if blocks else InfoType.OTHER
        return type_labels.get(info_type, "上下文信息")

    def _merge_content(
        self,
        blocks: list[ContextBlock],
        task_profile: TaskProfile,
    ) -> str:
        """Merge multiple blocks into a concise representation."""
        if len(blocks) == 1:
            return blocks[0].content

        # For multiple blocks, create a structured merge
        parts = []
        seen_content = set()

        for block in blocks:
            # Deduplicate similar content
            content_hash = block.content[:100]  # Simple hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                # Prefix with role context
                prefix = f"[{block.role}]" if block.role != "assistant" else ""
                parts.append(f"{prefix} {block.content}".strip())

        return "\n".join(parts)

    def _normalize_expressions(self, content: str) -> str:
        """Normalize informal expressions to formal/technical language."""
        normalized = content

        # Apply normalization map
        for informal, formal in NORMALIZE_MAP.items():
            normalized = re.sub(
                re.escape(informal),
                formal,
                normalized,
                flags=re.IGNORECASE
            )

        return normalized
