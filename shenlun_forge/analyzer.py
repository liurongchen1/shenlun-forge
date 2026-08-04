"""Step 1: Analyzer — Task analysis and compression strategy determination.

Maps to the Analyze step: understand what's being asked before touching the context.
"""

from __future__ import annotations
import json
from typing import Optional

from .base import BaseAnalyzer
from .types import ContextBlock, TaskProfile, TaskType


# v0.3 P1: 分类维度关键词（Signal 2: query → classification_dimension）
DIMENSION_KEYWORDS = {
    "temporal": ["流程", "步骤", "阶段", "时间线", "顺序", "历史", "发展", "演进",
                 "process", "step", "stage", "timeline", "sequence", "history", "evolution",
                 "计划", "排期", "milestone", "阶段"],
    "causal": ["原因", "因果", "为什么", "导致", "根源", "归因", "影响",
               "cause", "reason", "why", "lead to", "root cause", "attribution"],
    "layer": ["架构", "层次", "分层", "模块", "组件", "结构", "系统",
              "architecture", "layer", "module", "component", "structure", "system"],
    "spatial": ["区域", "分布", "位置", "范围", "领域", "地理",
                "region", "distribution", "location", "scope", "domain", "geographic"],
    "element": ["要素", "维度", "方面", "因素", "属性", "特征",
                "element", "dimension", "aspect", "factor", "attribute", "feature"],
}

# v0.3 P1: 输出逻辑关键词（query → output_logic）
OUTPUT_LOGIC_KEYWORDS = {
    "temporal": ["按时间", "时间线", "先后", "顺序", " chronological", "按阶段", "演进"],
    "causal": ["原因", "因果", "归因", "为什么", "分析原因", "causal", "attribution"],
    "progressive": ["递进", "深入", "逐步", "层层", "progressive", "step by step", "渐进"],
    "parallel": ["并列", "对比", "比较", "同时", "parallel", "compare", "side by side", "分别"],
}


def _detect_dimension(query: str) -> str:
    """v0.3 P1: 根据 query 关键词检测分类维度。"""
    query_lower = query.lower()
    scores = {}
    for dim, keywords in DIMENSION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[dim] = score
    if scores:
        return max(scores, key=scores.get)
    return "auto"


def _detect_output_logic(query: str) -> str:
    """v0.3 P1: 根据 query 关键词检测输出逻辑。"""
    query_lower = query.lower()
    scores = {}
    for logic, keywords in OUTPUT_LOGIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[logic] = score
    if scores:
        return max(scores, key=scores.get)
    return "auto"


ANALYZE_PROMPT = """You are a task analysis engine. Given a conversation history and the current user query, determine:

1. **task_type**: What kind of task is this? One of:
   - "qa": User asks a specific question
   - "summary": User wants a summary of prior discussion
   - "code_gen": User wants code generated
   - "analysis": User wants reasoning/analysis
   - "creative": User wants creative writing
   - "extraction": User wants specific information extracted
   - "tool_use": User is calling tools/functions
   - "conversation": General multi-turn chat

2. **key_elements**: What elements from the conversation history are MOST important for answering the current query? List 3-8 specific elements (e.g., "user's stated preference for Python", "database schema discussed earlier", "error message from tool output").

3. **priority_rules**: For each type of information, assign a weight from 0.0 to 1.0 indicating how important it is to preserve. Consider:
   - "user_explicit_preferences": Things the user explicitly stated
   - "factual_context": Facts, data, configurations mentioned
   - "tool_results": Results from tool calls
   - "reasoning_chain": Prior reasoning steps
   - "conversation_flow": Greetings, transitions, small talk
   - "duplicated_info": Information repeated multiple times
   - "code_snippets": Code blocks and technical implementations

4. **output_format**: How should the compressed context be structured?
   - "sequential": Maintain chronological order
   - "structured": Group by topic/category
   - "bullet": Key points as bullet list

5. **classification_dimension**: What is the primary organizing dimension?
   - "auto": No clear dimension
   - "temporal": Time/process/stage sequence
   - "causal": Cause-effect relationships
   - "layer": Architectural/structural layers
   - "spatial": Geographic/domain distribution
   - "element": Attributes/factors/aspects

6. **output_logic**: How should the output flow logically?
   - "auto": No clear logic preference
   - "parallel": Side-by-side comparison
   - "causal": Cause → effect chain
   - "progressive": Step-by-step deepening
   - "temporal": Chronological order

Current query: {query}

Conversation history (last {n} messages):
{history_summary}

Respond in JSON format:
{{
  "task_type": "...",
  "key_elements": ["...", "..."],
  "priority_rules": {{
    "user_explicit_preferences": 0.0,
    "factual_context": 0.0,
    "tool_results": 0.0,
    "reasoning_chain": 0.0,
    "conversation_flow": 0.0,
    "duplicated_info": 0.0,
    "code_snippets": 0.0
  }},
  "output_format": "...",
  "classification_dimension": "...",
  "output_logic": "..."
}}"""


# Fast path: rule-based classification when LLM is not available
TASK_KEYWORDS = {
    TaskType.QA: ["what", "how", "why", "when", "where", "who", "which", "什么", "怎么", "为什么", "如何", "哪"],
    TaskType.SUMMARY: ["summarize", "summary", "tldr", "总结", "概述", "概括", "归纳"],
    TaskType.CODE_GEN: ["code", "function", "class", "implement", "write a", "debug", "fix", "代码", "函数", "实现", "编写", "修复"],
    TaskType.ANALYSIS: ["analyze", "compare", "evaluate", "pros and cons", "分析", "比较", "评估", "优缺点"],
    TaskType.CREATIVE: ["write a story", "poem", "creative", "imagine", "故事", "诗", "创意", "想象"],
    TaskType.EXTRACTION: ["extract", "find", "list all", "identify", "提取", "找到", "列出", "识别"],
    TaskType.TOOL_USE: ["call", "use tool", "invoke", "execute", "调用", "执行"],
}

# Default priority weights per task type
DEFAULT_PRIORITIES = {
    TaskType.QA: {
        "user_explicit_preferences": 0.95,
        "factual_context": 0.85,
        "tool_results": 0.80,
        "reasoning_chain": 0.60,
        "conversation_flow": 0.20,
        "duplicated_info": 0.10,
        "code_snippets": 0.70,
    },
    TaskType.SUMMARY: {
        "user_explicit_preferences": 0.70,
        "factual_context": 0.90,
        "tool_results": 0.60,
        "reasoning_chain": 0.85,
        "conversation_flow": 0.10,
        "duplicated_info": 0.05,
        "code_snippets": 0.50,
    },
    TaskType.CODE_GEN: {
        "user_explicit_preferences": 0.90,
        "factual_context": 0.70,
        "tool_results": 0.90,
        "reasoning_chain": 0.50,
        "conversation_flow": 0.10,
        "duplicated_info": 0.05,
        "code_snippets": 0.95,
    },
    TaskType.ANALYSIS: {
        "user_explicit_preferences": 0.80,
        "factual_context": 0.90,
        "tool_results": 0.75,
        "reasoning_chain": 0.95,
        "conversation_flow": 0.15,
        "duplicated_info": 0.05,
        "code_snippets": 0.60,
    },
    TaskType.CREATIVE: {
        "user_explicit_preferences": 0.85,
        "factual_context": 0.50,
        "tool_results": 0.30,
        "reasoning_chain": 0.40,
        "conversation_flow": 0.40,
        "duplicated_info": 0.10,
        "code_snippets": 0.20,
    },
    TaskType.EXTRACTION: {
        "user_explicit_preferences": 0.90,
        "factual_context": 0.95,
        "tool_results": 0.85,
        "reasoning_chain": 0.40,
        "conversation_flow": 0.10,
        "duplicated_info": 0.05,
        "code_snippets": 0.70,
    },
    TaskType.TOOL_USE: {
        "user_explicit_preferences": 0.90,
        "factual_context": 0.80,
        "tool_results": 0.95,
        "reasoning_chain": 0.70,
        "conversation_flow": 0.10,
        "duplicated_info": 0.05,
        "code_snippets": 0.80,
    },
    TaskType.CONVERSATION: {
        "user_explicit_preferences": 0.80,
        "factual_context": 0.60,
        "tool_results": 0.50,
        "reasoning_chain": 0.50,
        "conversation_flow": 0.70,
        "duplicated_info": 0.10,
        "code_snippets": 0.30,
    },
}


class ACSAnalyzer(BaseAnalyzer):
    """Analyzer that uses task-aware strategy (Analyze step)."""

    def __init__(self, llm=None, use_fast_path: bool = True):
        """
        Args:
            llm: LLM provider for intelligent analysis. If None, uses rule-based only.
            use_fast_path: If True, use rule-based classification as fallback when LLM fails.
        """
        self.llm = llm
        self.use_fast_path = use_fast_path

    async def analyze(
        self,
        messages: list[ContextBlock],
        current_query: str,
    ) -> TaskProfile:
        """Analyze task and determine compression strategy."""

        # Try LLM-based analysis first
        if self.llm is not None:
            try:
                return await self._llm_analyze(messages, current_query)
            except Exception:
                if not self.use_fast_path:
                    raise

        # Fallback to rule-based
        return self._rule_based_analyze(messages, current_query)

    async def _llm_analyze(
        self,
        messages: list[ContextBlock],
        current_query: str,
    ) -> TaskProfile:
        """Use LLM for intelligent task analysis."""
        # Build a compact history summary for the prompt
        history_lines = []
        for msg in messages[-10:]:  # Last 10 messages for analysis
            preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            history_lines.append(f"[{msg.role}] {preview}")
        history_summary = "\n".join(history_lines)

        prompt = ANALYZE_PROMPT.format(
            query=current_query,
            n=len(messages),
            history_summary=history_summary,
        )

        result = await self.llm.complete_json(prompt, {})

        # Parse LLM response into TaskProfile
        task_type = TaskType(result.get("task_type", "conversation"))
        key_elements = result.get("key_elements", [])
        priority_rules = result.get("priority_rules", DEFAULT_PRIORITIES[task_type])
        output_format = result.get("output_format", "sequential")
        # v0.3 P1: 解析分类维度和输出逻辑
        classification_dimension = result.get("classification_dimension", "auto")
        output_logic = result.get("output_logic", "auto")

        return TaskProfile(
            task_type=task_type,
            key_elements=key_elements,
            output_format=output_format,
            priority_rules=priority_rules,
            classification_dimension=classification_dimension,
            output_logic=output_logic,
        )

    def _rule_based_analyze(
        self,
        messages: list[ContextBlock],
        current_query: str,
    ) -> TaskProfile:
        """Fast rule-based task classification (no LLM needed)."""
        query_lower = current_query.lower()

        # Keyword matching for task type
        scores = {}
        for task_type, keywords in TASK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[task_type] = score

        if scores and max(scores.values()) > 0:
            task_type = max(scores, key=scores.get)
        else:
            task_type = TaskType.CONVERSATION

        # Determine output format based on task type
        format_map = {
            TaskType.QA: "sequential",
            TaskType.SUMMARY: "bullet",
            TaskType.CODE_GEN: "sequential",
            TaskType.ANALYSIS: "structured",
            TaskType.EXTRACTION: "bullet",
            TaskType.CONVERSATION: "sequential",
        }

        # v0.3 P1: 规则路径也走分类维度/输出逻辑检测
        classification_dimension = _detect_dimension(current_query)
        output_logic = _detect_output_logic(current_query)

        return TaskProfile(
            task_type=task_type,
            key_elements=self._extract_key_elements_heuristic(messages, current_query),
            output_format=format_map.get(task_type, "sequential"),
            priority_rules=DEFAULT_PRIORITIES.get(task_type, DEFAULT_PRIORITIES[TaskType.CONVERSATION]),
            classification_dimension=classification_dimension,
            output_logic=output_logic,
        )

    def _extract_key_elements_heuristic(
        self,
        messages: list[ContextBlock],
        current_query: str,
    ) -> list[str]:
        """Heuristic key element extraction without LLM."""
        elements = []

        # Look for explicit user instructions in recent messages
        for msg in messages[-5:]:
            if msg.role == "user":
                content = msg.content.lower()
                if any(kw in content for kw in ["记住", "注意", "要求", "必须", "prefer", "remember", "must"]):
                    elements.append(f"User instruction: {msg.content[:100]}")

        # Look for code blocks
        for msg in messages:
            if "```" in msg.content or "def " in msg.content or "class " in msg.content:
                elements.append("Code implementation in context")
                break

        # Look for tool results
        for msg in messages:
            if msg.role == "tool" or "tool_output" in msg.metadata.get("source", ""):
                elements.append("Tool call results available")
                break

        return elements[:5]  # Cap at 5 elements
