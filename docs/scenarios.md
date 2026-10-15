# When to use it (and when not to)

ShenlunForge sits between your agent and the model: it takes the message list
you were about to send and returns a smaller one that keeps what the current
query needs. What it saves and what it keeps depends on the strategy you pick.

## Pick a strategy

| Your situation | Use | Why |
|---|---|---|
| Long agent history, query needs reasoning | `acs` | Reads the query, keeps what answers it |
| Code-heavy history, exact identifiers matter | `hcs` | ACS summary **plus** verbatim code blocks |
| Want token savings with no API key | `simple` | Pure rules, milliseconds, no network |
| Structured data: timelines, task lists, configs | `simple` | Rules keep discrete items intact; summarising tends to blur them |
| Question and answer do not share vocabulary | `acs` | Lexical matching cannot bridge "battery drains" → "BMS bug" |
| You only need the last N turns | nothing | Truncation is simpler and free |
| You need semantic recall over a document set | RAG, not this | Different problem: retrieval over a corpus, not compression of one conversation |

The honest summary: `simple` is a fast filter, `acs` is a reader. If your query
and your conversation share vocabulary, `simple` is often enough. If the answer
is spread across messages and phrased differently from the question, you want
`acs`.

## Scenario 1 — Long agent history (main use case)

An agent loop appends turns until the history no longer fits. Compress on every
call, for the query you are about to answer.

```python
from shenlun_forge import ShenlunForge

forge = ShenlunForge.create(strategy="acs", model="deepseek-chat", api_key="...")

async def ask(messages, question):
    compressed = await forge.compress(
        messages=messages,
        current_query=question,
        max_tokens=4000,
    )
    return await call_model(compressed.compressed_text, question)
```

Notes:

- Compress against the *current* question. The same history yields different
  output for different queries — that is the point.
- The most recent turns are always kept verbatim (adaptive window,
  `max(3, n // 8)`), because they carry the live task state. Override with
  `recent_window=N`.
- One LLM call per compression step. Budget for the latency.

## Scenario 2 — Zero-cost local compression

No key, no network, no model. Useful for trimming logs, preparing a summary
input, or running in an offline/air-gapped environment.

```bash
shenlun-forge compress --messages chat.json --query "What did we decide?" -o out.json
```

```python
from shenlun_forge import ShenlunForge

forge = ShenlunForge.create()          # 'simple' is the default
result = forge.compress_sync(messages=messages, current_query="...", max_tokens=800)
```

On a 40-message Chinese conversation this typically lands around 0.4× original
tokens in a few milliseconds.

## Scenario 3 — Code-heavy conversations

Summaries lose identifiers. `hcs` runs the ACS pipeline *and* re-attaches the
code blocks verbatim, so function names, config keys and error strings survive.

```python
forge = ShenlunForge.create(strategy="hcs", model="deepseek-chat", api_key="...")
```

Tuning knobs: `code_budget_ratio` (default `0.4`, share of the budget reserved
for verbatim code) and `min_narrative_budget` (default `200`).

## Scenario 4 — Chinese and mixed Chinese/English

Chinese has no word boundaries and technical vocabulary is often English even
when the discussion is Chinese. The retriever handles both: Chinese is matched
with 2-character n-grams, English terms are weighted by how rare they are
across the conversation (IDF), and terms appearing in messages that overlap a
Chinese query are used as additional signals.

Nothing special is required — but it is worth knowing this is why `simple` does
better on mixed-language histories than a plain keyword overlap would.

## Scenario 5 — Embedding retrieval

Swap the scorer for a local embedding model when lexical overlap is not enough
but you still do not want an LLM call:

```bash
pip install "shenlun-forge[embedding]"
```

The model comes from `huggingface.co` and is cached after the first fetch. The
common `HF_ENDPOINT=https://hf-mirror.com` mirror does **not** host fastembed's
ONNX repositories, so it fails — use a proxy that reaches huggingface.co.

```python
forge = ShenlunForge(retriever_type="embedding")   # construct directly
result = await forge.compress(messages=messages, current_query="...", lightweight=True)
```

`lightweight=True` skips analyze and process entirely: no LLM call at all, just
semantic selection plus formatting. Note that `retriever_type` belongs to the
`ShenlunForge` class itself, not to `create()` — and `simple` has its own
built-in rule-based scorer.

## What it is not

- Not a memory system. It does not persist anything; it compresses the list you
  hand it. See MemGPT-style designs if you need paging across sessions.
- Not retrieval. If the knowledge lives outside the conversation, that is RAG.
- Not lossless. It is designed to drop things. Verify what it kept against your
  own evaluation queries before trusting it in production.
