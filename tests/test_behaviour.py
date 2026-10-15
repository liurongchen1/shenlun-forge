"""Behavioural guarantees that a first-time user relies on.

These cover the contract described in the README: what the default strategy is,
what the result flags mean, and what happens when no LLM is available.
"""

import re
import warnings

import pytest

from shenlun_forge import ShenlunForge, __version__, list_strategies, register_strategy


def _make_history(n: int, prefix: str = "filler") -> list:
    return [
        {"role": "user" if i % 2 else "assistant", "content": f"{prefix} {i}"}
        for i in range(n)
    ]


def _without_notice(text: str) -> str:
    """Drop the trailing "what was left out" note the compressor appends."""
    return text.split("\n\n【压缩说明】")[0].split("\n\n[compression note]")[0]


def test_version_is_read_from_a_single_source():
    # Regression: __version__ used to be hard-coded and drifted from the
    # published distribution version.
    assert re.match(r"^\d+\.\d+", __version__)


def test_default_strategy_needs_no_llm():
    forge = ShenlunForge.create()
    assert type(forge).__name__ == "SimpleStrategy"


def test_llm_strategy_without_key_warns_at_construction():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ShenlunForge.create(strategy="acs", api_key="")

    assert any(issubclass(w.category, RuntimeWarning) for w in caught), (
        "an acs compressor with no API key must warn instead of failing silently"
    )


def test_run_without_llm_is_flagged_in_stats():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forge = ShenlunForge.create(strategy="acs", api_key="")

    result = forge.compress_sync(
        messages=_make_history(30, "turn"), current_query="turn", max_tokens=100
    )

    assert result.stats["llm_available"] is False
    assert result.stats["degraded"] is True
    assert result.stats["degraded_reason"]


def test_short_conversation_passes_through_unchanged():
    forge = ShenlunForge.create(strategy="simple")
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    result = forge.compress_sync(
        messages=messages, current_query="hello", max_tokens=4000
    )

    assert result.stats["passthrough"] is True
    assert "hello" in result.compressed_text
    # The whole point: never hand back something bigger than the input.
    assert result.compression_ratio <= 1.0


def test_compression_never_grows_the_payload():
    forge = ShenlunForge.create(strategy="simple")
    messages = [{"role": "user", "content": "x" * 400} for _ in range(10)]

    result = forge.compress_sync(
        messages=messages, current_query="x", max_tokens=100
    )

    assert 0 < result.compression_ratio <= 1.0
    assert result.compressed_tokens <= result.original_tokens


def test_recent_messages_are_never_dropped():
    forge = ShenlunForge.create(strategy="simple")
    messages = _make_history(40)
    messages.append({"role": "user", "content": "THE MOST RECENT TURN"})

    result = forge.compress_sync(
        messages=messages, current_query="something unrelated", max_tokens=100
    )

    assert "THE MOST RECENT TURN" in result.compressed_text
    assert result.stats["recent_protected"] > 0


def test_query_relevant_message_beats_recency():
    forge = ShenlunForge.create(strategy="simple")
    messages = [{"role": "assistant", "content": "The cache TTL is 300 seconds."}]
    messages += _make_history(20, "unrelated chatter")
    messages.append({"role": "user", "content": "done"})

    result = forge.compress_sync(
        messages=messages, current_query="what is the cache TTL?", max_tokens=60
    )

    assert "300 seconds" in result.compressed_text


def test_register_strategy_is_usable_from_create():
    class DummyStrategy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def compress(
            self, messages, current_query="", max_tokens=4000, **kwargs
        ):
            from shenlun_forge.types import CompressionResult

            return CompressionResult(compressed_text="dummy", stats={})

    register_strategy("test_dummy", DummyStrategy)
    try:
        assert "test_dummy" in list_strategies()
        forge = ShenlunForge.create(strategy="test_dummy", model="m")
        assert isinstance(forge, DummyStrategy)
    finally:
        from shenlun_forge.registry import _EXTRA_STRATEGIES

        _EXTRA_STRATEGIES.pop("test_dummy", None)


def test_empty_input_returns_empty_result():
    forge = ShenlunForge.create(strategy="simple")
    result = forge.compress_sync(messages=[], current_query="hello")

    assert result.compressed_text == ""
    assert result.original_tokens == 0


class TestPlanCompression:
    """`plan()` answers "should I compress at all" before anything is spent."""

    @staticmethod
    def _history(target_tokens: int) -> list:
        from shenlun_forge.types import count_tokens

        chunk = "这是关于数据库选型的讨论内容。" * 6
        per_chunk = count_tokens(chunk)
        count = max(1, target_tokens // per_chunk)
        return [
            {"role": "user" if i % 2 else "assistant", "content": chunk}
            for i in range(count)
        ]

    def test_history_that_fits_is_left_alone(self):
        from shenlun_forge import plan_compression

        plan = plan_compression(self._history(2000), model_window=128_000)
        assert plan.fits_in_window is True
        assert plan.recommended_max_tokens is None
        assert "unchanged" in plan.advice

    def test_history_that_does_not_fit_gets_a_budget(self):
        from shenlun_forge import plan_compression

        plan = plan_compression(self._history(20000), model_window=8000)
        assert plan.fits_in_window is False
        assert plan.recommended_max_tokens == int(8000 * 0.9)
        assert "max_tokens=" in plan.advice

    def test_safety_margin_shrinks_the_usable_window(self):
        from shenlun_forge import plan_compression

        history = self._history(4000)
        generous = plan_compression(history, model_window=10_000, safety_margin=0.1)
        tight = plan_compression(history, model_window=10_000, safety_margin=0.8)

        assert generous.fits_in_window is True    # usable 9000
        assert tight.fits_in_window is False      # usable 2000

    def test_plan_needs_no_api_key_or_instance(self):
        from shenlun_forge import plan_compression

        plan = plan_compression([{"role": "user", "content": "hi"}], model_window=1000)
        assert plan.history_tokens >= 0

    def test_forge_plan_matches_the_module_function(self):
        from shenlun_forge import ShenlunForge, plan_compression

        history = self._history(5000)
        forge = ShenlunForge(model="m", api_key="")
        assert forge.plan(history, model_window=4000).fits_in_window == \
            plan_compression(history, model_window=4000).fits_in_window


class TestOutputRespectsBudget:
    """Every output format must honour the token budget it was given.

    Regression: ``structured`` used to break only the inner loop after a
    partial truncation, so every remaining group appended another truncated
    entry and the "budgeted" result could be several times the budget.
    """

    def _blocks(self, n: int = 10):
        from shenlun_forge.types import CompressedBlock

        return [
            CompressedBlock(
                content="中文内容" * 200,
                original_indices=[i],
                summary_label=f"标签{i}",
                relevance_score=0.5,
            )
            for i in range(n)
        ]

    @pytest.mark.parametrize("output_format", ["sequential", "structured", "bullet"])
    def test_format_within_budget(self, output_format):
        import asyncio

        from shenlun_forge.formatter import ACSFormatter
        from shenlun_forge.types import TaskProfile, TaskType, count_tokens

        profile = TaskProfile(
            task_type=TaskType.ANALYSIS,
            output_format=output_format,
            token_budget=300,
        )
        out = asyncio.run(ACSFormatter().format(self._blocks(), profile))

        # A small overshoot is allowed (the "..." marker, token boundaries),
        # but nothing like the 9x blow-up this test exists to catch.
        assert count_tokens(out) <= 340, (
            f"{output_format} produced {count_tokens(out)} tokens for a "
            "300-token budget"
        )

    @pytest.mark.parametrize("output_format", ["sequential", "structured", "bullet"])
    def test_format_within_budget_when_all_blocks_share_one_group(self, output_format):
        import asyncio

        from shenlun_forge.formatter import ACSFormatter
        from shenlun_forge.types import TaskProfile, TaskType, count_tokens

        blocks = self._blocks()
        for block in blocks:
            block.summary_label = "同一个标签"

        profile = TaskProfile(
            task_type=TaskType.ANALYSIS,
            output_format=output_format,
            token_budget=300,
        )
        out = asyncio.run(ACSFormatter().format(blocks, profile))

        assert count_tokens(out) <= 340


class TestBudgetOverflowPolicy:
    """`budget_overflow` decides who wins when the newest turn is too big."""

    @staticmethod
    def _messages_with_a_huge_last_turn():
        history = [
            {"role": "user" if i % 2 else "assistant", "content": f"历史第 {i} 条 " * 40}
            for i in range(20)
        ]
        history.append({"role": "assistant", "content": "最新一轮的完整回复内容 " * 400})
        return history

    def test_keep_is_the_default_and_may_exceed_the_budget(self):
        forge = ShenlunForge.create(strategy="simple")
        result = forge.compress_sync(
            messages=self._messages_with_a_huge_last_turn(),
            current_query="总结一下",
            max_tokens=500,
        )

        assert result.stats["recent_truncated"] is False
        assert result.compressed_tokens > 500, (
            "the default keeps the newest turn whole, even past the budget"
        )
        assert result.stats["over_budget"] is True

    def test_truncate_enforces_the_budget(self):
        forge = ShenlunForge.create(strategy="simple")
        result = forge.compress_sync(
            messages=self._messages_with_a_huge_last_turn(),
            current_query="总结一下",
            max_tokens=500,
            budget_overflow="truncate",
        )

        assert result.stats["recent_truncated"] is True
        assert result.compressed_tokens <= 510
        assert result.stats["over_budget"] is False

    def test_truncate_only_cuts_the_tail(self):
        messages = self._messages_with_a_huge_last_turn()
        forge = ShenlunForge.create(strategy="simple")
        kept = forge.compress_sync(
            messages=messages, current_query="总结一下", max_tokens=500)
        cut = forge.compress_sync(
            messages=messages, current_query="总结一下", max_tokens=500,
            budget_overflow="truncate")

        # Compared without the closing note: under "truncate" the note is sized
        # out of the budget, so only the body should match as a prefix.
        kept_body = _without_notice(kept.compressed_text)
        cut_body = _without_notice(cut.compressed_text)
        assert cut_body.startswith(kept_body[: len(cut_body) - 3])

    def test_unknown_policy_is_rejected(self):
        forge = ShenlunForge.create(strategy="simple")
        with pytest.raises(ValueError, match="budget_overflow"):
            forge.compress_sync(
                messages=self._messages_with_a_huge_last_turn(),
                current_query="x",
                budget_overflow="typo",
            )


class TestDroppedNotice:
    """The summary has to say what it does not contain.

    A model handed a lossy summary without being told it is lossy answers "that
    is not in the context" — even when the answer was simply never shown. The
    note costs one line and removes that failure mode.
    """

    @staticmethod
    def _long_history(n: int = 20) -> list:
        return [
            {
                "role": "user" if i % 2 else "assistant",
                "content": f"第 {i} 条历史记录，包含若干需要记住的细节。" * 20,
            }
            for i in range(n)
        ]

    def test_notice_appears_and_counts_what_was_dropped(self):
        forge = ShenlunForge.create(strategy="simple")
        result = forge.compress_sync(
            messages=self._long_history(),
            current_query="总结一下这些历史",
            max_tokens=400,
        )

        assert result.stats["dropped_messages"] > 0
        assert "【压缩说明】" in result.compressed_text
        assert f"{result.stats['dropped_messages']} 条" in result.compressed_text

    def test_notice_language_follows_the_question(self):
        forge = ShenlunForge.create(strategy="simple")
        result = forge.compress_sync(
            messages=self._long_history(),
            current_query="What did we decide about the retry policy?",
            max_tokens=400,
        )

        assert "[compression note]" in result.compressed_text
        assert "【压缩说明】" not in result.compressed_text

    def test_nothing_is_claimed_when_nothing_was_dropped(self):
        forge = ShenlunForge.create(strategy="simple")
        result = forge.compress_sync(
            messages=_make_history(4), current_query="总结", max_tokens=4000,
        )

        assert result.stats["passthrough"] is True
        assert result.stats["dropped_messages"] == 0
        assert "压缩说明" not in result.compressed_text
        assert "compression note" not in result.compressed_text

    def test_opt_out_keeps_the_summary_clean_but_still_reports_the_gap(self):
        forge = ShenlunForge.create(strategy="simple")
        messages = self._long_history()
        announced = forge.compress_sync(
            messages=messages, current_query="总结一下", max_tokens=400)
        silent = forge.compress_sync(
            messages=messages, current_query="总结一下", max_tokens=400,
            dropped_notice=False)

        assert "压缩说明" not in silent.compressed_text
        assert silent.stats["dropped_messages"] == announced.stats["dropped_messages"]
        assert silent.stats["dropped_tokens"] > 0
        assert silent.compressed_tokens < announced.compressed_tokens

    def test_truncate_still_honours_the_budget_with_the_notice(self):
        messages = self._long_history() + [
            {"role": "assistant", "content": "最新一轮的完整回复内容 " * 400}
        ]
        forge = ShenlunForge.create(strategy="simple")
        result = forge.compress_sync(
            messages=messages, current_query="总结一下", max_tokens=500,
            budget_overflow="truncate")

        # The note is part of the output, so it has to fit inside the budget the
        # caller asked for — not be appended on top of it.
        assert result.compressed_tokens <= 510
        assert "【压缩说明】" in result.compressed_text

    def test_full_pipeline_also_says_what_is_missing(self):
        # No API key, so ACS degrades to its rule-based path. The note is
        # emitted by the compressor itself, so it must appear on both paths.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forge = ShenlunForge.create(strategy="acs", api_key="")
        result = forge.compress_sync(
            messages=self._long_history(),
            current_query="总结一下这些历史",
            max_tokens=400,
        )

        assert result.stats["dropped_messages"] > 0
        assert "【压缩说明】" in result.compressed_text


class TestMaxTokensValidation:
    """A missing or nonsense budget must fail readably, not deep inside.

    `plan_compression()` returns `recommended_max_tokens=None` when the history
    already fits. Passing that straight through used to die in the pipeline
    with "unsupported operand type(s) for *: 'NoneType' and 'float'".
    """

    MESSAGES = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好"}]

    def test_none_budget_is_rejected(self):
        forge = ShenlunForge.create(strategy="simple")
        with pytest.raises(ValueError, match="positive integer"):
            forge.compress_sync(messages=self.MESSAGES, current_query="x", max_tokens=None)

    def test_zero_and_negative_are_rejected(self):
        forge = ShenlunForge.create(strategy="simple")
        for bad in (0, -5):
            with pytest.raises(ValueError, match="positive"):
                forge.compress_sync(messages=self.MESSAGES, current_query="x", max_tokens=bad)

    def test_error_points_at_the_plan_compression_case(self):
        forge = ShenlunForge.create(strategy="simple")
        with pytest.raises(ValueError, match="plan_compression"):
            forge.compress_sync(messages=self.MESSAGES, current_query="x", max_tokens=None)
