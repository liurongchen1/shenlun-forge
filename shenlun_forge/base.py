"""Abstract base classes for LLM providers and components."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional

from .types import ContextBlock, CompressedBlock, TaskProfile


class LLMProvider(ABC):
    """Abstract LLM provider interface. All LLM backends must implement this."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Send a completion request and return the response text."""
        ...

    @abstractmethod
    async def complete_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        """Send a completion request expecting structured JSON output."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the provider's tokenizer."""
        ...


class BaseAnalyzer(ABC):
    """Step 1: Analyze the task and determine compression strategy."""

    @abstractmethod
    async def analyze(
        self,
        messages: list[ContextBlock],
        current_query: str,
    ) -> TaskProfile:
        ...


class BaseRetriever(ABC):
    """Step 2: Extract and score relevant information from context."""

    @abstractmethod
    async def retrieve(
        self,
        messages: list[ContextBlock],
        task_profile: TaskProfile,
    ) -> list[ContextBlock]:
        ...


class BaseProcessor(ABC):
    """Step 3: Process extracted info - merge, label, normalize."""

    @abstractmethod
    async def process(
        self,
        blocks: list[ContextBlock],
        task_profile: TaskProfile,
    ) -> list[CompressedBlock]:
        ...


class BaseFormatter(ABC):
    """Step 4: Format output within token budget."""

    @abstractmethod
    async def format(
        self,
        blocks: list[CompressedBlock],
        task_profile: TaskProfile,
    ) -> str:
        ...
