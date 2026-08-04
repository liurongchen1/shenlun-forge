"""Step 1: Analyzer — Task analysis and compression strategy determination.

Maps to the Analyze step: understand what's being asked before touching the context.
"""

from __future__ import annotations
import json
from typing import Optional

from .base import BaseAnalyzer
from .types import ContextBlock, TaskProfile, TaskType


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
  "output_format": "..."
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

        return TaskProfile(
            task_type=task_type,
            key_elements=key_elements,
            output_format=output_format,
            priority_rules=priority_rules,
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

        return TaskProfile(
            task_type=task_type,
            key_elements=self._extract_key_elements_heuristic(messages, current_query),
            output_format=format_map.get(task_type, "sequential"),
            priority_rules=DEFAULT_PRIORITIES.get(task_type, DEFAULT_PRIORITIES[TaskType.CONVERSATION]),
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
