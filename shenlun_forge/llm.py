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
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.temperature = temperature
        self.max_tokens = max_tokens
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
