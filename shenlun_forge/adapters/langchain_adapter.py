"""LangChain adapter for ShenlunForge.

Usage:
    from shenlun_forge.adapters.langchain_adapter import LangChainShenlunForge

    # As a Runnable component in a LangChain pipeline
    forge = LangChainShenlunForge(model="deepseek-chat", api_key="...")
    compressed = forge.invoke({
        "messages": chat_history,
        "query": current_question,
    })
"""

from __future__ import annotations
from typing import Any, Optional

from ..compressor import ShenlunForge
from ..llm import OpenAIProvider


class LangChainShenlunForge:
    """
    LangChain-compatible wrapper for ShenlunForge.

    Can be used as a Runnable-like component in LangChain pipelines,
    or as a callback/middleware for automatic context compression.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4000,
        strategy: str = "acs",
    ):
        self.forge = ShenlunForge.create(
            strategy=strategy,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self.max_tokens = max_tokens

    def invoke(self, input: dict, **kwargs) -> dict:
        """
        LangChain-compatible invoke interface.

        Args:
            input: Dict with keys:
                - messages: List of message dicts (role/content)
                - query: Current user query
                - max_tokens: Optional override for token budget

        Returns:
            Dict with compressed_text and metadata.
        """
        import asyncio

        messages = input.get("messages", [])
        query = input.get("query", "")
        max_tokens = input.get("max_tokens", self.max_tokens)

        # Handle LangChain message objects
        parsed_messages = self._parse_langchain_messages(messages)

        result = asyncio.run(self.forge.compress(
            messages=parsed_messages,
            current_query=query,
            max_tokens=max_tokens,
        ))

        return {
            "compressed_text": result.compressed_text,
            "compression_ratio": result.compression_ratio,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "stats": result.stats,
        }

    async def ainvoke(self, input: dict, **kwargs) -> dict:
        """Async version of invoke."""
        messages = input.get("messages", [])
        query = input.get("query", "")
        max_tokens = input.get("max_tokens", self.max_tokens)

        parsed_messages = self._parse_langchain_messages(messages)

        result = await self.forge.compress(
            messages=parsed_messages,
            current_query=query,
            max_tokens=max_tokens,
        )

        return {
            "compressed_text": result.compressed_text,
            "compression_ratio": result.compression_ratio,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "stats": result.stats,
        }

    def _parse_langchain_messages(self, messages: list) -> list[dict]:
        """Convert LangChain message objects to plain dicts."""
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                result.append(msg)
            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                # LangChain BaseMessage subclass
                result.append({
                    "role": self._map_role(msg.type),
                    "content": msg.content,
                })
            else:
                # Try generic conversion
                result.append({
                    "role": "user",
                    "content": str(msg),
                })
        return result

    def _map_role(self, msg_type: str) -> str:
        """Map LangChain message type to standard role."""
        mapping = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "function": "tool",
            "tool": "tool",
        }
        return mapping.get(msg_type, msg_type)

    def as_callback_handler(self):
        """Return a callback handler that auto-compresses context."""
        return _LangChainCallbackHandler(self)


class _LangChainCallbackHandler:
    """Callback handler for automatic context compression in LangChain."""

    def __init__(self, forge: LangChainShenlunForge):
        self.forge = forge
        self._compression_count = 0

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        """Called before LLM call — could compress prompts here."""
        pass  # Hook for future use

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> dict:
        """Called before chain execution — compress input if needed."""
        if "messages" in inputs:
            messages = inputs["messages"]
            if len(messages) > 10:  # Only compress if history is long
                result = self.forge.invoke(inputs)
                self._compression_count += 1
                # Replace messages with compressed version
                inputs["compressed_context"] = result["compressed_text"]
        return inputs
