"""OpenAI-compatible LLM provider implementation."""

from __future__ import annotations
import json
import os
import re
from typing import Any, Optional

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider (works with DeepSeek, Groq, etc.)."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: float = 600.0,
        max_retries: int = 2,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Long-context requests can take a while; the SDK defaults are fine for
        # chat but were hitting read timeouts on ~100k-token histories.
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None
        self._tokenizer = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install shenlun-forge[openai]"
                )
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                import tiktoken
                try:
                    self._tokenizer = tiktoken.encoding_for_model(self.model)
                except KeyError:
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                # Fallback: rough estimation
                self._tokenizer = None
        return self._tokenizer

    async def complete(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        # Providers occasionally return a well-formed response with no choices
        # (content filtering, truncation, an upstream hiccup). Indexing into it
        # raises "NoneType is not subscriptable", which says nothing about what
        # happened; fail with something a caller can act on.
        if not getattr(response, "choices", None):
            raise RuntimeError(
                f"{self.model} returned no choices (prompt was {len(prompt)} chars)"
            )
        return response.choices[0].message.content or ""

    async def complete_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        """Get structured JSON output from LLM with fallback parsing."""
        response_text = await self.complete(prompt, **kwargs)

        # Try to extract JSON from response
        try:
            # Try direct parse
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown code fences
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Try to find any JSON object/array
        for pattern in [r'\{[^{}]*\}', r'\[.*\]']:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue

        # Last resort: wrap in dict
        return {"raw_response": response_text}

    def count_tokens(self, text: str) -> int:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            # Rough estimation: ~4 chars per token for English, ~2 for Chinese
            return len(text) // 3
        return len(tokenizer.encode(text))

    async def aclose(self) -> None:
        """Close the underlying HTTP client.

        ``compress_sync`` runs each call in its own event loop. Without this,
        the cached client is garbage-collected after that loop is gone and
        httpx dumps a "RuntimeError: Event loop is closed" traceback to stderr
        on every call — noisy enough to look like a failure.
        """
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                # Closing is best-effort; a stale client is not worth raising.
                pass
            self._client = None


class InstrumentedProvider(LLMProvider):
    """Wraps a provider to count LLM calls and failures.

    The compressors use this so that a failing LLM (missing key, no credit,
    network error) shows up in ``result.stats`` as ``llm_failures`` and
    ``degraded`` instead of looking like a successful compression run.
    """

    def __init__(self, inner: LLMProvider, stats: Optional[dict] = None):
        self.inner = inner
        self.stats = stats if stats is not None else {"calls": 0, "failures": 0}

    async def complete(self, prompt: str, **kwargs) -> str:
        self.stats["calls"] += 1
        try:
            return await self.inner.complete(prompt, **kwargs)
        except Exception:
            self.stats["failures"] += 1
            raise

    async def complete_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        self.stats["calls"] += 1
        try:
            return await self.inner.complete_json(prompt, schema, **kwargs)
        except Exception:
            self.stats["failures"] += 1
            raise

    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)


def probe_llm(provider: Optional[LLMProvider]) -> tuple:
    """Best-effort check for whether LLM calls can actually succeed.

    Returns ``(ready, reason)``. ``ready=False`` means the compression
    pipeline silently falls back to rule-based mode, so callers should
    surface ``reason`` instead of pretending everything worked.
    """
    if provider is None:
        return False, "no LLM provider configured"

    # Only vets providers that actually carry an api_key attribute
    # (i.e. the built-in OpenAIProvider); custom providers are trusted.
    api_key = getattr(provider, "api_key", None)
    if api_key is not None and not api_key:
        return False, (
            "no API key configured; pass api_key=... or set "
            "SHENLUN_FORGE_API_KEY / OPENAI_API_KEY"
        )

    if isinstance(provider, OpenAIProvider):
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, (
                'the "openai" package is not installed; run '
                'pip install "shenlun-forge[openai]"'
            )

    return True, ""
