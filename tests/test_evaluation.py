"""Tests for the self-evaluation helpers that do not need an LLM."""

from shenlun_forge import EvalPoint, EvalReport
from shenlun_forge.evaluation import _conversations_to_message_lists


def test_accepts_both_conversation_shapes():
    messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]
    assert _conversations_to_message_lists([messages]) == [messages]
    assert _conversations_to_message_lists([{"messages": messages}]) == [messages]


def test_skips_conversations_too_short_to_test():
    short = [{"role": "user", "content": "hi"}]
    assert _conversations_to_message_lists([short]) == []
    assert _conversations_to_message_lists([{"messages": []}]) == []


def test_skips_malformed_entries():
    good = [{"role": "user", "content": f"m{i}"} for i in range(8)]
    assert _conversations_to_message_lists([good, {"messages": None}, None]) == [good]


def test_report_summary_ranks_strategies():
    report = EvalReport(
        budget=16000,
        counts={"simple": 10, "truncate": 10},
        accuracy={"simple": 0.42, "truncate": 0.31},
    )
    text = report.summary()
    assert text.index("simple") < text.index("truncate"), "best strategy should rank first"
    assert "42.0%" in text
    assert "noise" in text, "the summary should warn about the metric's own swing"


def test_report_summary_handles_empty():
    assert "no strategies" in EvalReport(budget=1000).summary()


def test_eval_point_defaults():
    point = EvalPoint(history=[], query="q", reference="r")
    assert point.key_points == []
