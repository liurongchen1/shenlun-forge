# Quickstart

Five minutes, two levels: compress a conversation with no API key, then turn
on the LLM-powered pipeline when you want deeper reasoning.

## 1. Install

```bash
pip install shenlun-forge
```

That is enough for the rule-based strategy and the CLI. Add extras only when
you need them:

| Extra | Command | What it unlocks |
|---|---|---|
| `openai` | `pip install "shenlun-forge[openai]"` | `acs` and `hcs` strategies |
| `langchain` | `pip install "shenlun-forge[langchain]"` | LangChain adapter |
| `embedding` | `pip install "shenlun-forge[embedding]"` | Semantic retrieval (~95MB ONNX model, downloaded on first use) |
| `full` | `pip install "shenlun-forge[full]"` | Everything above |

## 2. Compress something, no API key

Put a conversation in a JSON file:

```json
[
  {"role": "user", "content": "Which database did we pick?"},
  {"role": "assistant", "content": "For the database: PostgreSQL. We ruled out MongoDB."}
]
```

```bash
shenlun-forge compress --messages messages.json \
    --query "Which database did we pick?" \
    --max-tokens 500
```

Output is JSON:

```json
{
  "compressed_text": "...",
  "original_tokens": 258,
  "compressed_tokens": 80,
  "compression_ratio": 0.31,
  "stats": {
    "strategy": "simple",
    "llm_available": false,
    "degraded": false,
    "passthrough": false
  }
}
```

Same thing from Python:

```python
from shenlun_forge import ShenlunForge

forge = ShenlunForge.create(strategy="simple")   # the default
result = forge.compress_sync(
    messages=messages,                  # list of {"role", "content"}
    current_query="Which database did we pick?",
    max_tokens=500,
)
print(result.compressed_text)
```

Working scripts: [`examples/quickstart_no_llm.py`](../examples/quickstart_no_llm.py).

## 3. Turn on the LLM pipeline

The `acs` strategy reads the conversation and decides what matters, which finds
things keyword matching cannot — for example that "my battery drains fast" and
"BMS firmware bug" belong together.

```bash
export SHENLUN_FORGE_API_KEY="sk-..."
export SHENLUN_FORGE_BASE_URL="https://api.deepseek.com/v1"
export SHENLUN_FORGE_MODEL="deepseek-chat"

shenlun-forge compress --messages messages.json --query "..." --strategy acs
```

```python
from shenlun_forge import ShenlunForge

forge = ShenlunForge.create(
    strategy="acs",
    model="deepseek-chat",
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",   # any OpenAI-compatible endpoint
)
result = forge.compress_sync(messages=messages, current_query="...", max_tokens=4000)
```

Async is the native interface:

```python
import asyncio

async def main():
    result = await forge.compress(messages=messages, current_query="...", max_tokens=4000)

asyncio.run(main())
```

See [`examples/compress_before_llm.py`](../examples/compress_before_llm.py).

## 4. Decide whether you need to compress at all

Cheap to check, and it needs no API key. If the history already fits the model's
window, compressing it can only lose information:

```python
from shenlun_forge import plan_compression

plan = plan_compression(messages=history, model_window=128_000)
print(plan.fits_in_window)     # True -> just send the history
print(plan.advice)             # one sentence explaining the decision
```

When `fits_in_window` is False, `plan.recommended_max_tokens` is the budget to
hand to `compress()` — the whole usable window, because quality keeps improving
with budget while a tight budget costs a large share of questions outright.

## 5. Read the result

| Field | Meaning |
|---|---|
| `compressed_text` | What you send to the model instead of the full history |
| `original_tokens` / `compressed_tokens` | Token counts using the same tokenizer |
| `compression_ratio` | `compressed / original`. Lower is more compression |
| `blocks` | The messages that survived, with source indices |
| `stats["passthrough"]` | `True` means compression saved nothing, so the original text came back untouched |
| `stats["degraded"]` | `True` means the run did not use the LLM as configured |
| `stats["degraded_reason"]` | Why: no API key, missing package, or runtime failures |
| `stats["llm_calls"]` / `stats["llm_failures"]` | How many LLM requests were made and how many failed |
| `stats["over_budget"]` | `True` means the result exceeded `max_tokens` (see below) |
| `stats["recent_truncated"]` | `True` means the newest message was trimmed to fit |
| `stats["dropped_messages"]` / `stats["dropped_tokens"]` | How much history is not in the summary (see below) |

## Five things that surprise people

**`max_tokens` is a ceiling, not a target.** The rule-based strategy only keeps
messages that relate to your query. If that is four messages, you get four
messages — it will not pad the context with unrelated chatter to fill the
budget.

**Compression can be a no-op.** If the history is under 200 tokens, or nothing
meaningful can be dropped, the original text comes back unchanged with
`passthrough: true` and `compression_ratio: 1.0`. That is deliberate: silently
returning something *longer* than the input would be worse.

**`acs` and `hcs` need an LLM.** Without a key they warn and fall back to
rules. If you see that warning, either set a key or use `strategy="simple"` —
see the [FAQ](faq.md).

**The newest turn may push past `max_tokens`.** When the last message alone
does not fit, the default keeps it whole and reports `over_budget: true`. That
is a choice, not a bug — losing the live turn is usually the worse outcome.
Make the budget hard instead:

```python
result = await forge.compress(
    messages=messages,
    current_query=query,
    max_tokens=4000,
    budget_overflow="truncate",   # trim the newest message to fit
)
assert result.compressed_tokens <= 4000
```

From the CLI: `--budget-overflow truncate`. Either way the result tells you
which happened.

**The summary says what it left out.** You do not have to guess how much history
the model is missing, and neither does the model:

```
[compression note] 37 earlier messages (~12,480 tokens) are not included above.
If the answer is not here, say what is missing rather than guessing.
```

This is on by default. It costs one line, is written in the language of your
question, and is reserved inside `max_tokens` rather than appended on top of it.
Without it, a model that never saw the answer has no way to know that — it
assumes the summary *is* the conversation, and answers "not in the context" or
simply makes something up. Turn it off with `dropped_notice=False` (CLI:
`--no-dropped-notice`); `stats["dropped_messages"]` still tells you the size of
the gap.
