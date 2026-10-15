"""Tests for ShenlunForge core functionality."""

import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shenlun_forge.types import (
    ContextBlock, TaskProfile, TaskType, InfoType, CompressionResult,
)
from shenlun_forge.analyzer import ACSAnalyzer
from shenlun_forge.retriever import ACSRetriever
from shenlun_forge.processor import ACSProcessor
from shenlun_forge.formatter import ACSFormatter
from shenlun_forge.compressor import ShenlunForge


class TestAnalyzer:
    """Test Step 1: Task Analysis."""

    def test_rule_based_qa_detection(self):
        analyzer = ACSAnalyzer(llm=None, use_fast_path=True)
        messages = [ContextBlock(content="Hello", role="user")]
        profile = asyncio.run(analyzer.analyze(messages, "What is the database schema?"))
        assert profile.task_type == TaskType.QA

    def test_rule_based_summary_detection(self):
        analyzer = ACSAnalyzer(llm=None, use_fast_path=True)
        messages = [ContextBlock(content="We discussed many things", role="user")]
        profile = asyncio.run(analyzer.analyze(messages, "Summarize our conversation"))
        assert profile.task_type == TaskType.SUMMARY

    def test_rule_based_code_detection(self):
        analyzer = ACSAnalyzer(llm=None, use_fast_path=True)
        messages = [ContextBlock(content="Let me show you code", role="assistant")]
        profile = asyncio.run(analyzer.analyze(messages, "Write a function to sort"))
        assert profile.task_type == TaskType.CODE_GEN

    def test_fallback_to_conversation(self):
        analyzer = ACSAnalyzer(llm=None, use_fast_path=True)
        messages = [ContextBlock(content="Hello", role="user")]
        profile = asyncio.run(analyzer.analyze(messages, "Hey there"))
        assert profile.task_type == TaskType.CONVERSATION


class TestRetriever:
    """Test Step 2: Information Retrieval."""

    def test_relevance_scoring(self):
        retriever = ACSRetriever(llm=None)
        blocks = [
            ContextBlock(content="Database schema with PostgreSQL tables", role="assistant", msg_index=0),
            ContextBlock(content="Hello how are you", role="user", msg_index=1),
            ContextBlock(content="The database uses SQLAlchemy ORM", role="assistant", msg_index=2),
        ]
        profile = TaskProfile(
            task_type=TaskType.QA,
            key_elements=["database", "schema"],
            priority_rules={"factual_context": 0.9},
        )

        results = asyncio.run(retriever.retrieve(blocks, profile))
        assert len(results) > 0
        # Database-related blocks should score higher
        db_blocks = [b for b in results if "database" in b.content.lower() or "schema" in b.content.lower()]
        assert len(db_blocks) > 0

    def test_info_type_classification(self):
        retriever = ACSRetriever(llm=None)
        block = ContextBlock(
            content="The error was caused by a missing database connection",
            role="assistant",
            msg_index=0,
        )
        info_type = retriever._classify_info_type(block.content)
        assert info_type in (InfoType.PROBLEM, InfoType.CAUSE)


class TestProcessor:
    """Test Step 3: Information Processing."""

    def test_rule_based_merge(self):
        processor = ACSProcessor(llm=None)
        blocks = [
            ContextBlock(content="We chose PostgreSQL", role="user", msg_index=0, info_type=InfoType.FACT),
            ContextBlock(content="PostgreSQL is good for relational data", role="assistant", msg_index=1, info_type=InfoType.FACT),
        ]
        profile = TaskProfile(task_type=TaskType.QA)

        groups = processor._rule_based_merge(blocks, profile)
        # Both blocks have the same info_type, should be merged
        assert len(groups) == 1

    def test_summary_label_extraction(self):
        processor = ACSProcessor(llm=None)
        blocks = [
            ContextBlock(
                content="## Database Setup\nWe configured PostgreSQL with SQLAlchemy",
                role="assistant",
                msg_index=0,
                info_type=InfoType.FACT,
            ),
        ]
        profile = TaskProfile(task_type=TaskType.QA)
        label = processor._extract_summary_label(blocks, profile)
        assert label == "Database Setup"

    def test_expression_normalization(self):
        processor = ACSProcessor(llm=None)
        result = processor._normalize_expressions("The app doesn't work, it's too slow")
        assert "non-functional" in result
        assert "performance degraded" in result


class TestFormatter:
    """Test Step 4: Output Formatting."""

    def test_sequential_format(self):
        from shenlun_forge.types import CompressedBlock
        formatter = ACSFormatter()
        blocks = [
            CompressedBlock(
                content="Database: PostgreSQL with SQLAlchemy",
                original_indices=[0],
                summary_label="数据库方案",
            ),
            CompressedBlock(
                content="Auth: JWT with bcrypt hashing",
                original_indices=[1],
                summary_label="认证方案",
            ),
        ]
        profile = TaskProfile(
            task_type=TaskType.QA,
            output_format="sequential",
            token_budget=4000,
        )

        result = asyncio.run(formatter.format(blocks, profile))
        assert "数据库方案" in result
        assert "认证方案" in result

    def test_budget_truncation(self):
        from shenlun_forge.types import CompressedBlock
        formatter = ACSFormatter()
        blocks = [
            CompressedBlock(
                content="A" * 10000,  # Very long block
                original_indices=[0],
                summary_label="Long content",
            ),
        ]
        profile = TaskProfile(
            task_type=TaskType.QA,
            output_format="sequential",
            token_budget=100,  # Very small budget
        )

        result = asyncio.run(formatter.format(blocks, profile))
        assert len(result) < 10000  # Should be truncated


class TestShenlunForge:
    """Integration tests for the full pipeline."""

    def test_create_factory(self):
        forge = ShenlunForge.create(strategy="simple")
        assert forge is not None

    def test_simple_compression(self):
        forge = ShenlunForge.create(strategy="simple")
        messages = [
            {"role": "user", "content": "I need a database for my app"},
            {"role": "assistant", "content": "I recommend PostgreSQL for your use case"},
            {"role": "user", "content": "What about authentication?"},
            {"role": "assistant", "content": "Use JWT tokens with bcrypt password hashing"},
        ]

        result = forge.compress_sync(
            messages=messages,
            current_query="What database did we choose?",
            max_tokens=4000,
        )

        assert result.compressed_text
        assert result.original_tokens > 0
        assert result.compression_ratio <= 1.0

    def test_empty_messages(self):
        forge = ShenlunForge.create(strategy="simple")
        result = forge.compress_sync(messages=[], current_query="Hello")
        assert result.compressed_text == ""
        assert result.original_tokens == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
