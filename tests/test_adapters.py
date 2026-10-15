"""Adapter tests — run without langchain-core or llama-index-core installed."""

from shenlun_forge.adapters.langchain_adapter import LangChainShenlunForge
from shenlun_forge.adapters.llamaindex_adapter import LlamaIndexShenlunForge


HISTORY = [
    {"role": "user", "content": "Which database should we use for bookings?"},
    {"role": "assistant", "content": "For the database: PostgreSQL. We ruled out MongoDB."},
] + [
    {"role": "assistant" if i % 2 else "user",
     "content": f"unrelated chatter number {i} about the design of widget {i}"}
    for i in range(20)
] + [
    {"role": "user", "content": "So which database did we pick?"},
]


class _FakeMessage:
    """Stands in for llama_index.core.llms.ChatMessage."""

    def __init__(self, role, content):
        self.role = role
        self.content = content

    def __repr__(self):
        return f"_FakeMessage({self.role!r}, {self.content[:24]!r})"


class _FakeRole:
    """Stands in for the MessageRole enum (has .value)."""

    def __init__(self, value):
        self.value = value


def test_llamaindex_adapter_accepts_dicts():
    compressor = LlamaIndexShenlunForge(max_tokens=80)
    trimmed = compressor.trim(HISTORY, query="which database did we pick?")

    assert trimmed, "expected at least one surviving message"
    assert all(isinstance(m, dict) for m in trimmed)
    joined = " ".join(m["content"] for m in trimmed)
    assert "PostgreSQL" in joined
    assert len(trimmed) < len(HISTORY)


def test_llamaindex_adapter_returns_the_callers_own_objects():
    compressor = LlamaIndexShenlunForge(max_tokens=80)
    messages = [_FakeMessage(_FakeRole(m["role"]), m["content"]) for m in HISTORY]

    trimmed = compressor.trim(messages, query="which database did we pick?")

    assert trimmed, "expected at least one surviving message"
    assert all(isinstance(m, _FakeMessage) for m in trimmed)
    assert any("PostgreSQL" in m.content for m in trimmed)


def test_llamaindex_adapter_compress_text():
    compressor = LlamaIndexShenlunForge(max_tokens=80)
    text = compressor.compress_text(HISTORY, query="which database did we pick?")

    assert isinstance(text, str)
    assert "PostgreSQL" in text


def test_langchain_adapter_invoke():
    adapter = LangChainShenlunForge(strategy="simple", max_tokens=80)
    out = adapter.invoke({"messages": HISTORY, "query": "which database did we pick?"})

    assert "PostgreSQL" in out["compressed_text"]
    assert out["compressed_tokens"] <= out["original_tokens"]


def test_langchain_adapter_maps_message_objects():
    class _Msg:
        def __init__(self, type_, content):
            self.type = type_
            self.content = content

    adapter = LangChainShenlunForge(strategy="simple", max_tokens=80)
    parsed = adapter._parse_langchain_messages([
        _Msg("human", "which database did we pick?"),
        _Msg("ai", "PostgreSQL, after ruling out MongoDB."),
    ])

    assert [m["role"] for m in parsed] == ["user", "assistant"]
