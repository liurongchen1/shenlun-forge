"""Step 2: Retriever — Information extraction and relevance scoring.

Maps to the Retrieve step: find all relevant points from the material.
"""

from __future__ import annotations
import re
import math
from typing import Optional

from .base import BaseRetriever
from .types import ContextBlock, TaskProfile, TaskType, InfoType


# Keyword patterns for information type classification
INFO_TYPE_PATTERNS = {
    InfoType.ACTION: [
        r'\b(implement|create|build|add|remove|change|fix|update|deploy|configure)\b',
        r'搭建|完善|出台|整合|增设|实现|创建|修改|删除|部署|配置',
    ],
    InfoType.PROBLEM: [
        r'\b(error|bug|fail|issue|crash|broken|missing|lag|slow|wrong)\b',
        r'错误|失败|问题|缺陷|滞后|短缺|薄弱|缺位|闲置|异常|崩溃',
    ],
    InfoType.CAUSE: [
        r'\b(due to|because|caused by|reason|根源|since|as a result)\b',
        r'由于|根源|受制于|缺少|导致|原因是|因为',
    ],
    InfoType.EFFECT_POS: [
        r'\b(improve|increase|boost|optimize|enhance|success|benefit)\b',
        r'提升|改善|增加|优化|成功|收益|增收|化解',
    ],
    InfoType.EFFECT_NEG: [
        r'\b(decrease|reduce|worse|damage|loss|risk|threat)\b',
        r'下降|减少|恶化|损害|损失|风险|制约|加重|引发',
    ],
    InfoType.OPINION: [
        r'\b(think|believe|should|recommend|suggest|prefer|opinion)\b',
        r'认为|建议|应该|觉得|偏好|观点|意见',
    ],
    InfoType.DATA: [
        r'\b\d+[\.\d]*%?\s*(ms|s|MB|GB|TB|tokens|users|requests)\b',
        r'增长|下降|达到|占比|统计|数据|百分比',
    ],
    InfoType.FACT: [
        r'\b(is|are|was|were|has|have|defined|specif|version|name)\b',
        r'是|为|定义|版本|名称|配置|参数',
    ],
}

# Transition words that signal important content
IMPORTANT_SIGNALS = [
    "but", "however", "therefore", "thus", "consequently",
    "in conclusion", "importantly", "notably", "关键", "但是", "然而",
    "所以", "因此", "总之", "重要的是", "需要注意",
]


class ACSRetriever(BaseRetriever):
    """Retriever that scores context blocks based on task relevance (Retrieve step)."""

    def __init__(self, llm=None, min_score_threshold: float = 0.1):
        self.llm = llm
        self.min_score_threshold = min_score_threshold

    async def retrieve(
        self,
        messages: list[ContextBlock],
        task_profile: TaskProfile,
    ) -> list[ContextBlock]:
        """Score and filter context blocks based on task relevance."""

        scored_blocks = []

        for i, block in enumerate(messages):
            # Calculate composite relevance score
            score = self._calculate_relevance(block, task_profile, position=i, total=len(messages))
            block.relevance_score = score

            # Classify information type
            block.info_type = self._classify_info_type(block.content)

            scored_blocks.append(block)

        # Sort by relevance score (descending)
        scored_blocks.sort(key=lambda x: x.relevance_score, reverse=True)

        # Filter by threshold, but keep top-k minimum
        min_keep = min(5, len(scored_blocks))
        filtered = [b for b in scored_blocks if b.relevance_score >= self.min_score_threshold]

        # Always keep at least min_keep blocks
        if len(filtered) < min_keep:
            filtered = scored_blocks[:min_keep]

        # Re-sort by original position to maintain context flow
        filtered.sort(key=lambda x: x.msg_index)

        return filtered

    def _calculate_relevance(
        self,
        block: ContextBlock,
        task_profile: TaskProfile,
        position: int,
        total: int,
    ) -> float:
        """Calculate relevance score for a context block."""
        score = 0.0

        # 1. Keyword overlap with current query
        query_keywords = set(self._tokenize(task_profile.key_elements))
        content_keywords = set(self._tokenize([block.content]))
        if query_keywords:
            overlap = len(query_keywords & content_keywords) / len(query_keywords)
            score += overlap * 0.3

        # 2. Recency bonus (more recent = higher score)
        recency = (position + 1) / max(total, 1)
        score += recency * 0.2

        # 3. Role-based scoring
        role_weights = {"user": 0.8, "assistant": 0.6, "system": 0.3, "tool": 0.9}
        score += role_weights.get(block.role, 0.4) * 0.15

        # 4. Signal word bonus
        signal_count = sum(1 for s in IMPORTANT_SIGNALS if s in block.content.lower())
        score += min(signal_count * 0.05, 0.15)

        # 5. Content density (information per character)
        # Higher density = more meaningful content
        content = block.content.strip()
        if content:
            # Code blocks, tables, and structured data have higher density
            has_code = "```" in content or "def " in content
            has_table = "|" in content and "---" in content
            has_list = bool(re.search(r'^[\s]*[-*•]\s', content, re.MULTILINE))

            density_bonus = 0.0
            if has_code:
                density_bonus += 0.1
            if has_table:
                density_bonus += 0.08
            if has_list:
                density_bonus += 0.05
            score += density_bonus

        # 6. Priority rule matching
        for rule_key, weight in task_profile.priority_rules.items():
            if self._matches_priority_rule(block, rule_key):
                score += weight * 0.15

        return min(score, 1.0)

    def _classify_info_type(self, content: str) -> InfoType:
        """Classify the type of information in a content block."""
        content_lower = content.lower()
        scores = {}

        for info_type, patterns in INFO_TYPE_PATTERNS.items():
            match_count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content_lower)
                match_count += len(matches)
            scores[info_type] = match_count

        if not scores or max(scores.values()) == 0:
            return InfoType.OTHER

        return max(scores, key=scores.get)

    def _matches_priority_rule(self, block: ContextBlock, rule_key: str) -> bool:
        """Check if a block matches a priority rule."""
        content = block.content.lower()

        if rule_key == "user_explicit_preferences":
            return block.role == "user" and any(
                kw in content for kw in ["prefer", "want", "need", "必须", "要", "希望", "喜欢"]
            )
        elif rule_key == "factual_context":
            return block.info_type in (InfoType.FACT, InfoType.DATA)
        elif rule_key == "tool_results":
            return block.role == "tool" or "tool" in content.lower()[:50]
        elif rule_key == "reasoning_chain":
            return block.role == "assistant" and len(block.content) > 200
        elif rule_key == "conversation_flow":
            return block.role in ("user", "assistant") and len(block.content) < 100
        elif rule_key == "duplicated_info":
            return False  # Handled in processor step
        elif rule_key == "code_snippets":
            return "```" in block.content or "def " in block.content or "class " in block.content
        return False

    def _tokenize(self, texts: list[str]) -> list[str]:
        """Simple tokenization for keyword matching."""
        tokens = []
        for text in texts:
            # Split on whitespace and punctuation, keep Chinese chars
            words = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
            tokens.extend(words)
        return tokens
