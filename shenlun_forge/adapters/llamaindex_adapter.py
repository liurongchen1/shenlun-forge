"""LlamaIndex adapter for ShenlunForge.

LlamaIndex chat engines keep a growing list of ``ChatMessage`` objects. This
adapter trims that list (or flattens it to text) before it is sent to the model:

    from shenlun_forge.adapters.llamaindex_adapter import LlamaIndexShenlunForge

    compressor = LlamaIndexShenlunForge()          # rule-based by default

    # Drop the messages that no longer matter, keep ChatMessage objects:
    trimmed = compressor.trim(chat_history, query="What did we decide?")

    # Or hand the model a compact block of text instead of a message list:
    text = compressor.compress_text(chat_history, query="What did we decide?")

``llama-index-core`` is not a hard dependency: message objects are read by
attribute (``.role`` / ``.content``) and dicts work too. ``trim`` returns the
same objects you passed in, so it slots into a chat engine unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from ..compressor import ShenlunForge


# LlamaIndex MessageRole values vary by version ("human"/"ai" in older ones).
_TO_STANDARD_ROLE = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "tool_call": "tool",
    "chatbot": "assistant",
}


class LlamaIndexShenlunForge:
    """Trim a LlamaIndex chat history down to what the current query needs."""

    def __init__(
        self,
        strategy: str = "simple",
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4000,
        **kwargs: Any,
    ):
        self.forge = ShenlunForge.create(
            strategy=strategy,
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
        self.max_tokens = max_tokens

    # ── Public API ─────────────────────────────────────────────────────────

    def trim(self, messages: list, query: str = "", max_tokens: Optional[int] = None) -> list:
        """Return the surviving messages as the same kind of object you passed in."""
        result = self.forge.compress_sync(
            messages=self._to_dicts(messages),
            current_query=query,
            max_tokens=max_tokens or self.max_tokens,
        )
        kept = self._kept_indices(result)
        return self._rebuild(messages, kept)

    async def atrim(
        self, messages: list, query: str = "", max_tokens: Optional[int] = None
    ) -> list:
        """Async version of :meth:`trim`."""
        result = await self.forge.compress(
            messages=self._to_dicts(messages),
            current_query=query,
            max_tokens=max_tokens or self.max_tokens,
        )
        return self._rebuild(messages, self._kept_indices(result))

    def compress_text(
        self, messages: list, query: str = "", max_tokens: Optional[int] = None
    ) -> str:
        """Return the compressed history as one string, ready to drop into a prompt."""
        result = self.forge.compress_sync(
            messages=self._to_dicts(messages),
            current_query=query,
            max_tokens=max_tokens or self.max_tokens,
        )
        return result.compressed_text

    async def acompress_text(
        self, messages: list, query: str = "", max_tokens: Optional[int] = None
    ) -> str:
        result = await self.forge.compress(
            messages=self._to_dicts(messages),
            current_query=query,
            max_tokens=max_tokens or self.max_tokens,
        )
        return result.compressed_text

    # ── Internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _kept_indices(result) -> list:
        indices = {
            index for block in result.blocks for index in block.original_indices
        }
        return sorted(indices)

    @staticmethod
    def _to_dicts(messages: list) -> list:
        """Normalise LlamaIndex messages (or dicts) into role/content dicts."""
        normalised = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role", "user")).lower()
                content = message.get("content", "")
            else:
                raw_role = getattr(message, "role", "user")
                # MessageRole is an enum in recent versions; .value gives "user"
                role = str(getattr(raw_role, "value", raw_role)).lower()
                content = getattr(message, "content", None) or str(message)

            normalised.append({
                "role": _TO_STANDARD_ROLE.get(role, "user"),
                "content": content,
            })
        return normalised

    @staticmethod
    def _rebuild(original: list, kept_indices: list) -> list:
        """Map surviving indices back onto the caller's own message objects."""
        return [original[i] for i in kept_indices if 0 <= i < len(original)]
