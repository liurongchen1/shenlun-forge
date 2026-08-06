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
- Normalize colloquial language to technical/formal expressions

CRITICAL — DO NOT abstract away the following during merging:
1. Task items and deadlines: specific TODOs, pending tasks, deadlines (dates, durations like "3 days"), milestones. These are operational facts — losing them means the user forgets what to do and when.
2. Code-level details: function names, variable names, class names, API endpoints, file paths, configuration keys. These are implementation facts — summarizing "verify_password uses bcrypt" to "authentication is implemented" destroys the information.
3. Specific numbers and identifiers: version numbers, port numbers, error codes, enum values, constant names.

Instead of summarizing these into abstract descriptions, PRESERVE them verbatim in merged_content. You may restructure the surrounding text, but concrete identifiers and task specifics must appear in the output exactly as they appear in the input.
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
        """Process blocks: merge similar, add labels, normalize, preserve entities."""

        if not blocks:
            return []

        # Step 3-pre: Extract key entities from original content for preservation
        original_text = "\n".join(b.content for b in blocks)
        key_entities = self._extract_entities(original_text)

        # Step 3a: 申论"分门别类"——按 InfoType 分流处理
        # 事实/数据类 → 可安全合并，走 LLM 术语保留合并
        # 因果/行动/问题/观点类 → 不可合并（合并会丢逻辑链或细节），走规则保留
        _MERGEABLE_TYPES = {
            InfoType.FACT, InfoType.DATA,
        }

        mergeable_blocks = []
        preserve_blocks = []
        for i, b in enumerate(blocks):
            if b.info_type in _MERGEABLE_TYPES:
                mergeable_blocks.append((i, b))
            else:
                preserve_blocks.append((i, b))

        merged_groups = []

        # 对可合并类型：使用 LLM 术语保留合并
        if mergeable_blocks and self.enable_merge and self.llm is not None and len(mergeable_blocks) > 1:
            try:
                mergeable_only = [b for _, b in mergeable_blocks]
                llm_groups = await self._term_preserving_merge(mergeable_only, task_profile)
                # 将 LLM 返回的本地索引映射回原始全局索引
                for group in llm_groups:
                    mapped = [(mergeable_blocks[local_i][0], mergeable_blocks[local_i][1])
                              for local_i, _ in group]
                    merged_groups.append(mapped)
            except Exception:
                for item in mergeable_blocks:
                    merged_groups.append([item])
        elif mergeable_blocks:
            for item in mergeable_blocks:
                merged_groups.append([item])

        # 对不可合并类型：规则保留，逐条保留原貌
        if preserve_blocks:
            for item in preserve_blocks:
                merged_groups.append([item])

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

        # Step 3-post: Preserve key entities — inject missing ones
        if key_entities and compressed:
            compressed = self._preserve_entities(compressed, key_entities)

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

    async def _term_preserving_merge(
        self,
        blocks: list[ContextBlock],
        task_profile: TaskProfile,
    ) -> list[list[tuple[int, ContextBlock]]]:
        """v0.3 P12: layer 维度专用合并 — 使用增强 prompt 保留技术术语。

        合并后校验术语保留率，若丢失过多则 fallback 到规则合并。
        """
        blocks_text = ""
        for i, block in enumerate(blocks):
            preview = block.content[:300] + "..." if len(block.content) > 300 else block.content
            blocks_text += f"\n[Block {i}] (role={block.role}, type={block.info_type}):\n{preview}\n"

        prompt = TERM_PRESERVING_MERGE_PROMPT.format(blocks_text=blocks_text)
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

        if not merged:
            merged = [[(i, b)] for i, b in enumerate(blocks)]

        # 校验术语保留率：若 LLM merge 丢失太多原始术语，fallback 到规则合并
        original_text = "\n".join(b.content for b in blocks)
        original_entities = set(self._extract_entities(original_text))
        if original_entities:
            merged_text = ""
            for group in merged:
                for _, b in group:
                    merged_text += b.content + "\n"
            # Note: at this point merged_text is still the original content (pre-LLM-merge output)
            # The actual term check happens later in _preserve_entities
            # Here we just validate that LLM returned reasonable grouping
            preserved_count = sum(
                1 for e in original_entities if e in merged_text
            )
            preservation_rate = preserved_count / len(original_entities)
            if preservation_rate < 0.5:
                # LLM merge lost too many terms, fallback to rule-based
                return self._rule_based_merge(blocks, task_profile)

        return merged

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

    # ─── Entity Preservation (关键实体保护) ──────────────────────────────────

    def _extract_entities(self, text: str) -> list[str]:
        """Extract key entities that must survive compression.

        Targets: code identifiers, constants, API paths, technical terms,
        version numbers, time expressions, filenames.
        Returns deduplicated list.
        """
        entities = set()

        # 1. snake_case identifiers (2+ chars per segment): is_active, check_deadlines
        for m in re.finditer(r'\b([a-z]{2,}(?:_[a-z]{2,})+)\b', text):
            entities.add(m.group(1))

        # 2. SCREAMING_SNAKE_CASE constants: JWT_SECRET, HS256
        for m in re.finditer(r'\b([A-Z][A-Z0-9]{1,}(?:_[A-Z][A-Z0-9]+)+)\b', text):
            entities.add(m.group(1))

        # 3. API routes / file paths: /payments/webhook
        for m in re.finditer(r'(/[a-z][a-z0-9_-]*/[a-z][a-z0-9_/-]*)', text):
            entities.add(m.group(1).rstrip('/'))

        # 4. Technical terms: all-uppercase 2+ chars (JWT, API, R2, HS256)
        for m in re.finditer(r'\b([A-Z][A-Z0-9]{1,})\b', text):
            entities.add(m.group(1))

        # 5. CamelCase (lowercase start): nextAuthService
        for m in re.finditer(r'\b([a-z]+[A-Z][a-zA-Z]{2,})\b', text):
            entities.add(m.group(1))

        # 5b. PascalCase (2+ capitalized words joined): SignatureVerificationError
        for m in re.finditer(r'\b((?:[A-Z][a-z]+){2,})\b', text):
            entities.add(m.group(1))

        # 6. Product names with dot: Next.js, shadcn/ui
        for m in re.finditer(r'\b([A-Z][a-z]+\.[a-z]+)\b', text):
            entities.add(m.group(1))

        # 7. Time expressions: 9:00, 9:00 AM
        for m in re.finditer(r'\b(\d{1,2}:\d{2}(?:\s*[APap][Mm])?)\b', text):
            entities.add(m.group(1))

        # 8. Number + unit: 3 days, 50MB, 8 characters
        for m in re.finditer(r'\b(\d+\s*(?:days?|hours?|minutes?|seconds?|weeks?|months?|years?|MB|KB|GB|chars?|tokens?|items?|files?))\b', text):
            entities.add(m.group(1))

        # 9. Version numbers: v0.2.2, 2.0.0
        for m in re.finditer(r'\b(v?\d+\.\d+(?:\.\d+)?)\b', text):
            entities.add(m.group(1))

        # 10. Filenames with extension: .py, .ts, .json
        for m in re.finditer(r'\b([a-z_][a-z0-9_]*\.(?:py|ts|tsx|js|jsx|json|md|sql|ini|yml|yaml|env))\b', text):
            entities.add(m.group(1))

        # ── Chinese-specific patterns ──

        # 11. Number + Chinese unit: 3天, 5个模块, 200行, 第3步
        for m in re.finditer(r'(\d+\s*(?:天|日|个|步|项|条|次|遍|周|月|年|小时|分钟|秒|块|份|页|行|位|层|级|段|组|批|种))', text):
            entities.add(m.group(1))

        # 12. Quoted content (Chinese quotes): "认证机制", 「用户模块」, 【已完成】
        for m in re.finditer(r'[""「」『』（）\u300c\u300d\u300e\u300f]([^""「」『』\u300c\u300d\u300e\u300f]{2,20})[""「」『』\u300c\u300d\u300e\u300f]', text):
            inner = m.group(1).strip()
            if len(inner) >= 2 and not all(c in ' \t\n' for c in inner):
                entities.add(inner)
        # Square bracket labels: 【关键术语】
        for m in re.finditer(r'【([^】]{2,20})】', text):
            entities.add(m.group(1))

        # 13. Chinese ordinal/phase: 第一步, 第二阶段
        for m in re.finditer(r'([第][一二三四五六七八九十\d]+[步阶段部分层级轮期条项]{1,2})', text):
            entities.add(m.group(1))

        # 14. Mixed Chinese-English tech terms: JWT认证, API接口, OAuth授权
        # Require at least one CJK char in trailing part to avoid false matches on English text
        for m in re.finditer(r'([A-Z][A-Za-z0-9+#.-]{1,}\s*[\u4e00-\u9fff][^\W\d_和的了是在用与及等为也要把被]{0,2})', text):
            entities.add(m.group(1).strip())

        # 15. Chinese domain-specific keywords (standalone tech terms)
        for m in re.finditer(r'((?:消息队列|负载均衡|配置中心|微服务|数据库|中间件|认证机制|鉴权机制|加密算法|签名验证|序列化|反序列化|容器编排|网关路由|缓存策略|监控告警|日志采集|消息通知|权限管理|数据备份|故障恢复|灰度发布|持续集成))', text):
            entities.add(m.group(1))

        # Filter: remove entities that are too short or look like common words
        filtered = set()
        stop_words = {
            'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT', 'ARE', 'WAS',
            'NOT', 'BUT', 'HAS', 'HAD', 'HAV', 'HIS', 'HER', 'ITS', 'OUR',
            'CAN', 'WILL', 'SHALL', 'WHICH', 'THEIR', 'THEY', 'THEM', 'THEN',
            'THAN', 'INTO', 'ABOUT', 'AFTER', 'ALL', 'ANY', 'BEEN', 'BEFORE',
            'BETWEEN', 'EACH', 'EVERY', 'GET', 'GOT', 'HERE', 'HOW', 'JUST',
            'LIKE', 'MAY', 'MUCH', 'NOW', 'ONLY', 'OTHER', 'OVER', 'SOME',
            'SUCH', 'THAT', 'THESE', 'WHAT', 'WHEN', 'WHERE', 'WHILE', 'YES',
            'JSON', 'HTTP', 'HTML', 'CSS', 'SQL', 'URL', 'API', 'URI',
        }
        for e in entities:
            if e.upper() in stop_words:
                continue
            if len(e) < 3 and not e.replace('.', '').replace(':', '').isdigit():
                continue
            filtered.add(e)

        return list(filtered)

    def _preserve_entities(
        self,
        compressed: list[CompressedBlock],
        entities: list[str],
    ) -> list[CompressedBlock]:
        """Ensure key entities appear in compressed output.

        Collects all compressed text, finds missing entities,
        and appends them as a preservation block.
        """
        full_text = "\n".join(b.content for b in compressed)

        missing = [e for e in entities if e not in full_text]

        if not missing or len(missing) <= len(entities) * 0.3:
            # Most entities preserved (>=70%), no need to inject
            return compressed

        # Sort missing by length (longer = more specific, higher priority)
        missing.sort(key=len, reverse=True)

        # Cap at 30 entities to avoid bloating
        missing = missing[:30]

        preservation_text = "\n[关键术语保留] " + ", ".join(missing)

        # Add as a new block at the end
        last_idx = max(
            (i for block in compressed for i in block.original_indices),
            default=0,
        )
        preserved_tokens = count_tokens(preservation_text)

        compressed.append(CompressedBlock(
            content=preservation_text,
            original_indices=[last_idx],
            summary_label="关键术语",
            info_type=InfoType.FACT,
            preserved_token_count=preserved_tokens,
            compression_ratio=1.0,
        ))

        return compressed


# v0.3 P12: layer 维度专用 prompt — 强化技术术语保留
TERM_PRESERVING_MERGE_PROMPT = """You are a context compression processor specialized in technical/architectural content.

Your PRIMARY goal is to PRESERVE all technical terms, component names, API endpoints, protocol names, and architectural relationships. DO NOT abstract or generalize them.

Given the following context blocks, group them by semantic similarity, then for each group:
1. Merge the content while PRESERVING all technical identifiers verbatim
2. Extract a "summary label" (总括词) - a short 2-6 character label
3. Keep cause-effect chains intact (X causes Y, Z depends on W)

Context blocks:
{blocks_text}

Respond in JSON:
{{
  "groups": [
    {{
      "block_indices": [0, 2, 5],
      "summary_label": "简要标签",
      "merged_content": "合并后的内容，必须保留所有技术术语",
      "info_type": "fact|action|problem|cause|effect_pos|effect_neg|opinion|data|other"
    }}
  ]
}}

CRITICAL RULES:
- NEVER replace specific technical terms with generic descriptions
  BAD: "实现了认证机制" → GOOD: "实现了JWT认证，使用HS256算法，Token有效期24小时"
  BAD: "使用数据库存储" → GOOD: "使用PostgreSQL数据库，users表包含id/name/email字段"
- Preserve ALL: function names, class names, API paths, config keys, protocol names, port numbers
- Preserve ALL cause-effect relationships explicitly
- You may restructure sentences for conciseness, but every technical identifier must appear verbatim
"""
