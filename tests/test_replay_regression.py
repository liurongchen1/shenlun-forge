"""The replay metric: what counts as hard information, and what a drop means.

Uses synthetic conversations — the real ones these tools were built against are
private, and the rules are what need testing, not the data.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "benchmarks" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


replay = _load("replay_regression", "replay_regression.py")


MESSAGES = [
    {"id": 1, "role": "user", "content": "The goal is to fix /srv/api/monitor.py today."},
    {"id": 2, "role": "assistant", "content": "pm2 restart monitor\nIt came back listening on 127.0.0.1:8643."},
    {"id": 3, "role": "user", "content": "Also keep 500 MB free under /data/backups/."},
]


def test_hard_information_is_extracted_by_category():
    facts = replay.extract_facts(MESSAGES)
    assert "/srv/api/monitor.py" in facts
    assert facts["/srv/api/monitor.py"]["category"] == "路径"
    assert ":8643" in facts
    assert "127.0.0.1" in facts
    assert "500 MB" in facts


def test_command_lines_are_captured_whole():
    facts = replay.extract_facts([MESSAGES[1]])
    command = next(v for v in facts if v.startswith("pm2 restart"))
    assert command.startswith("pm2 restart monitor")


def test_common_acronyms_are_not_mistaken_for_constants():
    """JSON and HTTP are words, not identifiers — counting them adds noise."""
    facts = replay.extract_facts([{"id": 1, "role": "user", "content": "send it as JSON over HTTP"}])
    assert "JSON" not in facts
    assert "HTTP" not in facts


def test_nothing_is_lost_when_everything_is_visible():
    series = [{
        "turn": 1,
        "uptoId": 3,
        "contextText": "\n".join(m["content"] for m in MESSAGES),
        "currentUserContent": "",
    }]
    turns = replay.score_series(MESSAGES, series)
    assert turns[0]["loss"] == 0.0
    assert turns[0]["total"] > 0


def test_losing_a_message_shows_up_as_loss():
    visible = MESSAGES[0]["content"]  # only the first message survives
    series = [{"turn": 1, "uptoId": 3, "contextText": visible, "currentUserContent": ""}]
    turns = replay.score_series(MESSAGES, series)
    assert turns[0]["loss"] > 0
    # The user turn that was dropped is counted on the user side too.
    assert turns[0]["user_loss"] > 0


def test_empty_history_scores_zero_rather_than_being_skipped():
    """Turn counts have to line up with the other implementation's report."""
    series = [{"turn": 1, "uptoId": 1, "contextText": "", "currentUserContent": "hi"}]
    turns = replay.score_series([{"id": 1, "role": "user", "content": "hi"}], series)
    assert len(turns) == 1
    assert turns[0]["loss"] == 0.0


def test_summary_averages_per_conversation_not_per_turn():
    """A long conversation must not outvote a short one."""
    conversations = {
        "a": {"turns": [{"loss": 0.0, "user_loss": None, "early_loss": None}] * 100},
        "b": {"turns": [{"loss": 0.5, "user_loss": None, "early_loss": None}]},
    }
    summary = replay.summarise(conversations)
    assert summary["overall"] == 0.25


def test_paired_comparison_reports_no_change_for_identical_runs():
    turns = [{"turn": i, "total": 10, "loss": i / 20} for i in range(1, 6)]
    conversations = {"1": turns}
    report = replay.compare(conversations, conversations)
    assert report["n"] == 5
    assert report["mean"] == 0.0
    assert report["worse"] == 0 and report["better"] == 0


def test_paired_comparison_detects_a_consistent_regression():
    before = {"1": [{"turn": i, "total": 10, "loss": 0.1} for i in range(1, 6)]}
    after = {"1": [{"turn": i, "total": 10, "loss": 0.4} for i in range(1, 6)]}
    report = replay.compare(before, after)
    assert report["mean"] > 0.29
    assert report["worse"] == 5
