"""Hard information, and the decision that hangs off where it sits."""

from shenlun_forge import ShenlunForge
from shenlun_forge.facts import extract_facts, user_fact_share


def _conversation(user_turns, assistant_turns):
    messages = []
    for user, assistant in zip(user_turns, assistant_turns):
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return messages


USER_HEAVY = _conversation(
    [
        "The goal is to fix /srv/api/monitor.py and keep it on :8643.",
        "Do not touch /srv/api/billing.py, the freeze still applies.",
        "Ship it to /data/backups/ before Friday.",
    ],
    [
        "Understood.",
        "Noted, I will leave that file alone.",
        "Okay, will do.",
    ],
)

ASSISTANT_HEAVY = _conversation(
    [
        "What is the pipeline?",
        "How does entity protection work?",
        "And the threshold?",
    ],
    [
        "It is Analyzer -> Classifier -> Processor -> Formatter, see /src/pipeline.ts "
        "and /src/analyzer.ts.",
        "It calls _extract_entities and _preserve_entities, defined in "
        "/src/preserve.py and /src/retriever.ts.",
        "The threshold is 70%, configured via ENTITY_THRESHOLD in /etc/forge.conf "
        "and ENTITY_MIN_LENGTH in /etc/forge.d/base.toml.",
    ],
)


def test_user_share_is_high_when_the_user_carries_the_literals():
    share = user_fact_share(USER_HEAVY)
    assert share is not None and share > 0.5


def test_user_share_is_low_when_the_assistant_carries_the_literals():
    share = user_fact_share(ASSISTANT_HEAVY)
    assert share is not None and share < 0.3


def test_too_little_hard_information_means_no_decision():
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    assert user_fact_share(messages) is None


def test_auto_protects_user_turns_in_a_user_heavy_conversation():
    forge = ShenlunForge.create()
    result = forge.compress_sync(messages=USER_HEAVY, current_query="what now?", max_tokens=200)
    assert result.stats["user_protection"] is True


def test_auto_leaves_an_assistant_heavy_conversation_alone():
    forge = ShenlunForge.create()
    result = forge.compress_sync(
        messages=ASSISTANT_HEAVY, current_query="how does it work?", max_tokens=200
    )
    assert result.stats["user_protection"] is False


def test_auto_and_a_forced_setting_agree_when_forced():
    forge = ShenlunForge.create()
    for setting in (True, False):
        result = forge.compress_sync(
            messages=USER_HEAVY, current_query="q", max_tokens=200,
            protect_user_messages=setting,
        )
        assert result.stats["user_protection"] is setting


def test_a_typo_is_rejected_rather_than_guessed():
    forge = ShenlunForge.create()
    try:
        forge.compress_sync(
            messages=USER_HEAVY, current_query="q", max_tokens=200,
            protect_user_messages="yes please",
        )
    except ValueError as exc:
        assert "protect_user_messages" in str(exc)
    else:
        raise AssertionError("a bad setting should raise, not pick a default")
