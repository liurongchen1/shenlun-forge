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

    def __init__(self, llm=None, min_score_threshold: float = 0.1, top_k: int = 20):
        self.llm = llm
        self.min_score_threshold = min_score_threshold  # kept for backward compat
        self.top_k = top_k

    # Common English words that are NOT technical terms — filter them out
    _TECH_STOP_WORDS = {
        "the", "and", "for", "not", "with", "this", "that", "from", "are", "was",
        "has", "have", "can", "but", "all", "you", "your", "will", "would", "could",
        "should", "may", "might", "must", "shall", "each", "which", "their", "there",
        "these", "those", "some", "any", "more", "other", "into", "about", "after",
        "before", "between", "through", "during", "above", "below", "until", "while",
        "also", "very", "just", "only", "then", "than", "when", "what", "where",
        "how", "why", "who", "does", "done", "using", "used", "use", "based",
        "need", "needs", "want", "like", "well", "sure", "ok", "yes", "no",
        "here", "now", "got", "get", "gets", "let", "say", "said", "tell",
        "come", "take", "make", "know", "think", "see", "look", "find", "give",
        "going", "been", "being", "does", "did", "done", "such", "both", "same",
        "most", "much", "many", "few", "every", "another", "over", "down", "up",
        "off", "out", "back", "still", "already", "always", "never", "often",
        "really", "quite", "rather", "even", "though", "whether", "either",
        "per", "via", "etc", "vs", "ie", "eg", "aka",
    }

    def _extract_technical_terms(self, texts: list[str]) -> list[str]:
        """Extract English technical terms from text.
        
        Crucial for 中英混杂 conversations where the query/key_elements may be
        in Chinese but the actual technical content (function names, config keys,
        architecture terms) is in English.
        """
        terms = []
        for text in texts:
            # Extract English words/identifiers >= 3 chars (short ones are noise)
            found = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]{2,}\b', text)
            terms.extend([t.lower() for t in found if t.lower() not in self._TECH_STOP_WORDS])
        # Deduplicate
        return list(set(terms))

    def _tokenize_zh(self, text: str) -> set[str]:
        """Tokenize Chinese text into 2-char sliding window n-grams.
        
        Chinese has no word boundaries, so we use 2-character n-grams
        to enable approximate keyword matching.
        """
        result = set()
        segments = re.findall(r'[\u4e00-\u9fff]+', text.lower())
        for seg in segments:
            if len(seg) >= 2:
                for i in range(len(seg) - 1):
                    result.add(seg[i:i + 2])
        return result

    async def retrieve(
        self,
        messages: list[ContextBlock],
        task_profile: TaskProfile,
    ) -> list[ContextBlock]:
        """Score and filter context blocks based on task relevance."""

        # ── Cross-message statistics ──────────────────────────────────
        # Per-message term sets (for IDF and bridge computation)
        msg_term_sets: list[set[str]] = []
        tech_term_msg_freq: dict[str, int] = {}
        for m in messages:
            terms = set(self._extract_technical_terms([m.content]))
            msg_term_sets.append(terms)
            for t in terms:
                tech_term_msg_freq[t] = tech_term_msg_freq.get(t, 0) + 1

        n_msgs = max(len(messages), 1)

        # IDF weights: rarer English terms across messages → higher weight
        idf_weights: dict[str, float] = {}
        for term, freq in tech_term_msg_freq.items():
            idf_weights[term] = math.log((n_msgs + 1) / (freq + 1)) + 1.0

        # Shared tech terms (appear in 2+ messages)
        shared_tech_terms = {t for t, freq in tech_term_msg_freq.items() if freq >= 2}

        # ── English query terms (from current_query + key_elements) ──
        # Combine current_query and key_elements for maximum coverage
        query_text = task_profile.current_query + ' ' + ' '.join(task_profile.key_elements)
        en_query_terms = set()
        for t in re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]{2,}\b', query_text):
            t_lower = t.lower()
            if t_lower not in self._TECH_STOP_WORDS:
                en_query_terms.add(t_lower)

        # ── Bridge terms: Chinese→English bridge ─────────────────────
        # When query is mostly Chinese, find messages with high Chinese keyword
        # overlap and extract their English terms as additional query signals.
        # Uses 2-gram tokenization for Chinese (fixed from whole-phrase matching).
        bridge_terms: set[str] = set()
        query_zh_bigrams = self._tokenize_zh(query_text)
        if query_zh_bigrams:
            bridge_candidates: dict[str, int] = {}
            for msg_terms, m in zip(msg_term_sets, messages):
                msg_zh = self._tokenize_zh(m.content)
                zh_overlap = len(query_zh_bigrams & msg_zh)
                if zh_overlap >= 2:
                    for en_t in msg_terms:
                        if en_t not in en_query_terms:
                            bridge_candidates[en_t] = bridge_candidates.get(en_t, 0) + zh_overlap
            # Keep top 15 bridge terms weighted by Chinese overlap strength
            sorted_bridge = sorted(bridge_candidates.items(), key=lambda x: -x[1])
            bridge_terms = {t for t, _ in sorted_bridge[:15]}

        # ── Score all blocks ─────────────────────────────────────────
        scored_blocks = []
        for i, block in enumerate(messages):
            score = self._calculate_relevance(
                block, task_profile, position=i, total=len(messages),
                shared_tech_terms=shared_tech_terms,
                idf_weights=idf_weights,
                en_query_terms=en_query_terms,
                bridge_terms=bridge_terms,
                query_zh_bigrams=query_zh_bigrams,
            )
            block.relevance_score = score
            block.info_type = self._classify_info_type(block.content)
            scored_blocks.append(block)

        # Sort by relevance score (descending)
        scored_blocks.sort(key=lambda x: x.relevance_score, reverse=True)

        # Select top-k blocks (fixed top_k, no adaptive cap)
        k = min(self.top_k, len(scored_blocks))
        filtered = scored_blocks[:k]

        # Re-sort by original position to maintain context flow
        filtered.sort(key=lambda x: x.msg_index)

        return filtered

    def _calculate_relevance(
        self,
        block: ContextBlock,
        task_profile: TaskProfile,
        position: int,
        total: int,
        shared_tech_terms: set = None,
        idf_weights: dict = None,
        en_query_terms: set = None,
        bridge_terms: set = None,
        query_zh_bigrams: set = None,
    ) -> float:
        """Calculate relevance score for a context block.
        
        Weight rebalanced: query-dependent factors dominate (max ~1.0),
        query-independent factors are secondary (max ~0.33).
        """
        score = 0.0
        content_lower = block.content.lower()

        # ── Query-dependent factors (dominant) ──────────────────────

        # 1. Chinese keyword overlap via 2-gram matching
        # Now uses 2-gram sliding window for Chinese (fixed from whole-phrase).
        query_zh = query_zh_bigrams or set()
        content_zh = self._tokenize_zh(block.content)
        if query_zh:
            overlap = len(query_zh & content_zh) / len(query_zh)
            score += overlap * 0.40

        # 1b. English query term matching with IDF weighting
        if en_query_terms and idf_weights:
            block_en = set(self._extract_technical_terms([block.content]))
            matched_en = block_en & en_query_terms
            if matched_en:
                idf_score = sum(idf_weights.get(t, 1.0) for t in matched_en)
                score += min(idf_score * 0.10, 0.25)

        # 1c. Bridge terms (Chinese→English bridging)
        if bridge_terms and idf_weights:
            block_en = set(self._extract_technical_terms([block.content]))
            matched_bridge = block_en & bridge_terms
            if matched_bridge:
                bridge_score = sum(idf_weights.get(t, 1.0) for t in matched_bridge)
                score += min(bridge_score * 0.05, 0.15)

        # 1d. Cross-message technical term bonus (IDF-weighted, low weight)
        if shared_tech_terms and idf_weights:
            block_tech = set(self._extract_technical_terms([block.content]))
            shared_in_block = block_tech & shared_tech_terms
            if shared_in_block:
                shared_score = sum(idf_weights.get(t, 1.0) for t in shared_in_block)
                score += min(shared_score * 0.02, 0.10)

        # ── Query-independent factors (secondary) ───────────────────

        # 2. Recency bonus (reduced — was 0.20, now 0.05)
        recency = (position + 1) / max(total, 1)
        score += recency * 0.05

        # 3. Role-based scoring (reduced — was 0.15, now 0.03)
        role_weights = {"user": 0.8, "assistant": 0.6, "system": 0.3, "tool": 0.9}
        score += role_weights.get(block.role, 0.4) * 0.03

        # 4. Signal word bonus (reduced — was 0.15, now 0.05)
        signal_count = sum(1 for s in IMPORTANT_SIGNALS if s in content_lower)
        score += min(signal_count * 0.02, 0.05)

        # 5. Content density (reduced — was 0.23, now max ~0.07)
        content = block.content.strip()
        if content:
            has_code = "```" in content or "def " in content
            has_table = "|" in content and "---" in content
            has_list = bool(re.search(r'^[\s]*[-*•]\s', content, re.MULTILINE))

            density_bonus = 0.0
            if has_code:
                density_bonus += 0.03
            if has_table:
                density_bonus += 0.02
            if has_list:
                density_bonus += 0.02
            score += density_bonus

        # 6. Priority rule matching (reduced — was 0.15, now 0.03)
        for rule_key, weight in task_profile.priority_rules.items():
            if self._matches_priority_rule(block, rule_key):
                score += weight * 0.03

        # 7. P2: 首尾位置加权 (reduced — was 0.05, now 0.02)
        if total > 0:
            position_ratio = position / total
            if position_ratio < 0.2 or position_ratio > 0.8:
                score += 0.02

        # 8. P3: 词频特征加权 — 2-gram freq bonus (keep at 0.10)
        if query_zh and content_zh:
            overlapping_bigrams = query_zh & content_zh
            freq_bonus = 0.0
            for bigram in overlapping_bigrams:
                count = content_lower.count(bigram)
                if count >= 2:
                    freq_bonus += 0.03
            score += min(freq_bonus, 0.10)

        # 9. P4: 来源权威性加权 (reduced — was 0.15, now 0.05)
        authority_signals = [
            "确认", "决策", "必须", "规定", "决定", "要求", "标准",
            "confirmed", "decided", "must", "required", "standard", "mandatory",
        ]
        authority_count = sum(1 for s in authority_signals if s in content_lower)
        score += min(authority_count * 0.02, 0.05)

        # 10. P5: 结构层次检测 (reduced — was 0.08/-0.03, now 0.03/-0.01)
        deep_analysis_words = [
            "根本原因", "底层逻辑", "原理", "机制", "架构", "设计",
            "root cause", "underlying", "mechanism", "architecture", "design",
        ]
        surface_words = [
            "看起来", "表面上", "好像", "似乎",
            "looks like", "appears", "seems", "surface",
        ]
        deep_count = sum(1 for w in deep_analysis_words if w in content_lower)
        surface_count = sum(1 for w in surface_words if w in content_lower)
        score += deep_count * 0.03
        score -= surface_count * 0.01

        return min(max(score, 0.0), 1.0)

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
        """Tokenize text for keyword matching.
        
        For Chinese text, creates 2-character sliding window n-grams
        since Chinese has no word boundaries.
        For English/numeric text, keeps tokens as whole words.
        """
        tokens = []
        for text in texts:
            text_lower = text.lower()
            # Extract English words and Chinese character runs separately
            segments = re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]+', text_lower)
            for seg in segments:
                if not seg:
                    continue
                if seg.isascii():
                    # English/number token: keep as whole word (min 2 chars)
                    if len(seg) >= 2:
                        tokens.append(seg)
                else:
                    # Chinese segment: create 2-char sliding window n-grams
                    if len(seg) >= 2:
                        for i in range(len(seg) - 1):
                            tokens.append(seg[i:i + 2])
        return tokens
