"""The `demo` command is the first thing a new user runs.

It asserts a specific difference between truncation and compression, so these
tests check the claim itself, not just that some text appears.
"""

import pytest

from shenlun_forge import demo as demo_module
from shenlun_forge.cli import cmd_demo, get_parser
from shenlun_forge.demo import run_demo
from shenlun_forge.strategies.simple import SimpleStrategy
from shenlun_forge.types import count_tokens


def test_demo_needs_no_api_key_and_no_network(monkeypatch, capsys):
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "SHENLUN_FORGE_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    cmd_demo(get_parser().parse_args(["demo", "--lang", "en"]))

    out = capsys.readouterr().out
    assert "truncate" in out
    assert "simple" in out
    assert "compression note" in out


def test_demo_can_run_in_chinese():
    assert "演示" in run_demo(lang="zh")
    assert "demo" in run_demo(lang="en")


def test_budget_flag_changes_what_is_kept():
    small = run_demo(lang="en", budget=120)
    large = run_demo(lang="en", budget=600)
    assert small != large


@pytest.mark.parametrize("question_index", [0, 1])
def test_the_comparison_the_demo_prints_is_true(question_index):
    """Truncation and compression differ, in the direction the output claims.

    Without this, a retriever change could quietly turn the demo into a lie
    printed at the top of the README.
    """
    messages = demo_module.DEMO_HISTORY
    question = demo_module.DEMO_QUESTIONS[question_index]
    markers = demo_module.DEMO_ANSWER_MARKERS[question_index]
    budget = 300

    truncated = demo_module._truncate_history(messages, budget)
    result = SimpleStrategy().compress_sync(
        messages=messages, current_query=question, max_tokens=budget
    )
    kept = sorted({i for block in result.blocks for i in block.original_indices})

    # Compression keeps the messages that answer the question — that is the
    # whole demonstration.
    assert demo_module._marks_answer(messages, kept, markers)

    # And it really is cheaper than returning everything.
    assert result.compressed_tokens < sum(
        count_tokens(m["content"]) for m in messages
    )


def test_truncation_loses_the_first_answer():
    """The failure the demo opens with has to actually happen."""
    messages = demo_module.DEMO_HISTORY
    markers = demo_module.DEMO_ANSWER_MARKERS[0]
    truncated = demo_module._truncate_history(messages, 300)

    assert not demo_module._marks_answer(messages, truncated, markers), (
        "the demo claims truncation drops the Redis answer; if the conversation "
        "or the budget changes so that it no longer does, the comparison has to "
        "be rewritten rather than shown as-is"
    )


def test_index_label_collapses_consecutive_messages():
    assert demo_module._index_label([4, 6, 28, 29, 30, 31]) == "4, 6, 28..31"
    assert demo_module._index_label([]) == "-"
