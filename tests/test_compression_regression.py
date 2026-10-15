"""Compression behaviour regression on a fixed conversation — no API key needed.

The end-to-end question ("can a model still answer from the compressed text?")
needs an LLM and your own conversations, so it stays a manual step
(``shenlun-forge eval``). This covers the layer underneath, and runs in CI on
every commit: given a fixed conversation, the compressors should keep the same
kind of content, respect the budget, protect the newest turns, and adapt to the
question.

When one of these fails, compression behaviour changed. Whether that is a
regression or an improvement is a judgement call — look at what moved, then
update the expectation deliberately instead of loosening the assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shenlun_forge import ShenlunForge, plan_compression


_DATASET = json.loads(
    (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "data"
        / "real_dev_conversation.json"
    ).read_text(encoding="utf-8")
)
CONVERSATION = _DATASET["conversation"]
SCENARIOS = _DATASET["scenarios"]


def _retention(result, key_facts) -> float:
    text = result.compressed_text.lower()
    return sum(1 for fact in key_facts if fact.lower() in text) / len(key_facts)


@pytest.mark.parametrize("budget", [2000, 4000])
def test_stays_within_budget(budget):
    forge = ShenlunForge.create()
    for scenario in SCENARIOS[:4]:
        result = forge.compress_sync(
            messages=CONVERSATION, current_query=scenario["query"], max_tokens=budget
        )
        assert result.stats["over_budget"] is False
        assert result.compressed_tokens <= budget * 1.02


@pytest.mark.parametrize("scenario_index", [0, 1, 2])
def test_keeps_most_key_facts(scenario_index):
    """The compressors used to keep 14/14, 18/18 and 16/16 on these scenarios.

    0.75 is deliberately loose: this asserts "still useful", not "unchanged".
    """
    scenario = SCENARIOS[scenario_index]
    forge = ShenlunForge.create()
    result = forge.compress_sync(
        messages=CONVERSATION, current_query=scenario["query"], max_tokens=2000
    )
    assert _retention(result, scenario["key_facts"]) >= 0.75


def test_history_that_fits_is_passed_through():
    """The whole conversation is ~5.4k tokens, so an 8k budget needs no work."""
    forge = ShenlunForge.create()
    result = forge.compress_sync(
        messages=CONVERSATION, current_query=SCENARIOS[0]["query"], max_tokens=8000
    )
    assert result.stats["passthrough"] is True
    assert result.compressed_tokens == result.original_tokens


def test_newest_turns_survive_a_tight_budget():
    forge = ShenlunForge.create()
    result = forge.compress_sync(
        messages=CONVERSATION, current_query=SCENARIOS[3]["query"], max_tokens=800
    )
    tail = CONVERSATION[-1]["content"][:30]
    assert tail in result.compressed_text


def test_plan_agrees_with_what_compression_does():
    small = plan_compression(CONVERSATION, model_window=2000)
    large = plan_compression(CONVERSATION, model_window=128_000)

    assert small.fits_in_window is False
    assert small.recommended_max_tokens
    assert large.fits_in_window is True
    assert large.recommended_max_tokens is None


def test_compression_is_query_aware():
    """Two different questions must not produce identical output.

    If this ever passes trivially (both outputs equal), the retriever stopped
    using the query and every question would get the same context.
    """
    forge = ShenlunForge.create()
    version = forge.compress_sync(
        messages=CONVERSATION, current_query="版本号是多少？发布在哪些平台？", max_tokens=800
    )
    entities = forge.compress_sync(
        messages=CONVERSATION, current_query="实体保护机制是怎么实现的？", max_tokens=800
    )
    assert version.compressed_text != entities.compressed_text
