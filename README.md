# ShenlunForge

**Task-driven context compression for LLM agents.**

Your agent's history keeps growing. ShenlunForge takes the message list you were
about to send to the model and returns a smaller one that keeps what the
*current question* needs — not a blind truncation, and not a generic summary.

```python
from shenlun_forge import ShenlunForge

forge = ShenlunForge.create()                 # rule-based, no API key
result = forge.compress_sync(messages=history, current_query=question, max_tokens=800)

print(result.compressed_text)                 # send this instead
```

[![PyPI](https://img.shields.io/pypi/v/shenlun-forge.svg)](https://pypi.org/project/shenlun-forge/)
[![Python](https://img.shields.io/pypi/pyversions/shenlun-forge.svg)](https://pypi.org/project/shenlun-forge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/liurongchen1/shenlun-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/liurongchen1/shenlun-forge/actions/workflows/ci.yml)

English · [简体中文](README.zh-CN.md)

---

## Who this is for

- **Agent developers** whose `messages` list no longer fits the context window,
  or costs more than it should. Compress before each model call, for the
  question you are about to ask.
- **Anyone with a long conversation to shrink** and no interest in paying for
  an API to do it. The rule-based strategy is local, free and runs in
  milliseconds.
- **Chinese and mixed Chinese/English conversations**, where word-boundary
  tokenizers and English-tuned compressors tend to do badly.

## Install

```bash
pip install shenlun-forge
```

That covers the rule-based strategy and the CLI. Extras:

```bash
pip install "shenlun-forge[openai]"      # 'acs' / 'hcs' strategies
pip install "shenlun-forge[langchain]"   # LangChain adapter
pip install "shenlun-forge[llamaindex]"  # llama-index message compatibility
pip install "shenlun-forge[embedding]"   # local embedding retrieval
pip install "shenlun-forge[full]"        # everything
```

Python 3.9+.

## 30 seconds, no API key

```python
from shenlun_forge import ShenlunForge

history = [
    {"role": "user", "content": "Which database should we use for bookings?"},
    {"role": "assistant", "content": "For the database: PostgreSQL. We ruled out MongoDB."},
    {"role": "user", "content": "And how do we send the confirmation email?"},
    {"role": "assistant", "content": "Resend, with a template per language."},
    # ... 40 more turns ...
]

forge = ShenlunForge.create()          # 'simple' is the default
result = forge.compress_sync(
    messages=history,
    current_query="Which database did we settle on?",
    max_tokens=500,
)

print(result.compression_ratio)        # 0.31 — 69% of the tokens gone
print(result.compressed_text)
```

Or from the shell:

```bash
shenlun-forge compress --messages chat.json --query "Which database did we settle on?"
```

More: [`docs/quickstart.md`](docs/quickstart.md) ·
[`examples/quickstart_no_llm.py`](examples/quickstart_no_llm.py)

## When to use which strategy

| Your situation | Strategy | Why |
|---|---|---|
| Long history, the answer needs reasoning across messages | `acs` | Reads the query, keeps what answers it |
| Code-heavy history, exact identifiers matter | `hcs` ⚠️ | ACS summary **plus** verbatim code blocks — but measured as no better than the free options, see below |
| Want savings with no API key | `simple` (default) | Pure rules, local, milliseconds |
| Timelines, task lists, configs, structured data | `simple` | Rules keep discrete items intact |
| Question and answer share no vocabulary | `acs` | Keyword matching cannot bridge "app drains fast" → "BMS bug" |
| You only need the last N turns | nothing | Truncation is simpler and free |
| You need retrieval over many documents | RAG, not this | Different problem |

The short version: **`simple` is a fast filter, `acs` is a reader.** If your
query and your history share words, `simple` is usually enough. If the answer is
scattered and phrased differently from the question, you want `acs`.

Full reasoning in [`docs/scenarios.md`](docs/scenarios.md).

## With an LLM

`acs` and `hcs` need an OpenAI-compatible endpoint — DeepSeek, OpenAI, Groq,
vLLM, Ollama's OpenAI shim, anything.

```python
forge = ShenlunForge.create(
    strategy="acs",
    model="deepseek-chat",
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
)

result = await forge.compress(
    messages=history,
    current_query="What security measures did we implement?",
    max_tokens=4000,
)
```

`hcs` adds verbatim code retention on top, for histories where a summarised
`verify_password` is worse than no summary at all.

See [`examples/compress_before_llm.py`](examples/compress_before_llm.py) and
[`docs/integration.md`](docs/integration.md).

## Should you compress at all?

Compression is lossy, so the first question is whether it is needed. Answering
that is free — no API key, no LLM call, no instance:

```python
from shenlun_forge import plan_compression

plan = plan_compression(messages=history, model_window=128_000)

if plan.fits_in_window:
    send(history)                                  # nothing to do
else:
    result = await forge.compress(
        messages=history,
        current_query=query,
        max_tokens=plan.recommended_max_tokens,    # the whole usable window
    )
```

`plan.advice` states the decision in a sentence. The reason to ask first: in
our measurements, a tight budget costs a large share of questions outright and
quality keeps improving as the budget grows, so there is rarely a reason to
compress below what the window allows.

`ShenlunForge.plan()` is the same thing as a method, if you already have an
instance.

## Reading the result

```python
result.compressed_text          # the string you send to the model
result.original_tokens          # tokens in
result.compressed_tokens        # tokens out
result.compression_ratio        # compressed / original
result.blocks                   # survivors, with source indices

result.stats["passthrough"]     # True = nothing was gained, original returned
result.stats["degraded"]        # True = did not run as configured
result.stats["degraded_reason"] # why
result.stats["llm_calls"]       # LLM requests made
result.stats["llm_failures"]    # how many failed
```

Three behaviours are deliberate and worth knowing up front:

- **`max_tokens` is a ceiling, not a target.** If four messages relate to your
  query, you get four. Padding the context with unrelated chatter would make it
  worse, not better.
- **A no-op returns the original.** If the history is under 200 tokens, or
  compression would save less than 5%, you get your input back untouched with
  `passthrough: True`. Returning something *longer* than the input would be
  worse than returning it unchanged.
- **The newest turn wins ties.** If the final message alone would blow the
  budget, the default keeps it whole — losing the live turn is usually worse
  than overshooting — and says so via `over_budget: True`. When `max_tokens` is
  a hard limit (you meter cost by it, or the target model's window is exactly
  that size), pass `budget_overflow="truncate"`: it trims the newest message to
  fit and reports `recent_truncated: True`. Either way the choice is explicit
  and visible in the result.

## CLI

```bash
# Rule-based, no key (default)
shenlun-forge compress --messages chat.json --query "What did we decide?"

# Read from stdin, write structured output
cat chat.json | shenlun-forge compress --stdin --query "Recap" -o out.json

# LLM-powered, with credentials from the environment
export SHENLUN_FORGE_API_KEY="sk-..."
export SHENLUN_FORGE_BASE_URL="https://api.deepseek.com/v1"
shenlun-forge compress -m chat.json -q "What security measures did we implement?" -s acs -v

shenlun-forge strategies
shenlun-forge version

# Measure whether compression helps on your own conversations
shenlun-forge eval --conversations ./my-chats --strategies truncate,simple
```

## How it works

The LLM pipeline (`acs`) is a four-stage compression pass modeled on how an
analyst handles source material:

| Stage | What it does |
|---|---|
| **Analyze** | Reads the question, decides which kinds of information matter |
| **Retrieve** | Scores every message — query overlap, information type, recency, role, signal words, density |
| **Process** | Merges duplicates, attaches a short label to each group, preserves identifiers (function names, constants, paths, `3 days`-style constraints) |
| **Format** | Greedily fills the token budget, highest priority first |

The rule-based path (`simple`) uses the same retrieval scoring without any LLM
call, then fills the budget with the messages that relate to the query. Both
paths always keep the most recent turns verbatim — that is where the live task
state lives.

Chinese is matched with 2-character n-grams and English terms are weighted by
rarity (IDF) across the conversation, with cross-lingual bridging so a Chinese
question can find the English identifiers that answer it.

## Evaluation

Reproducible with [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py):
a mixed Chinese/English development conversation (18 messages, ~5.4k tokens),
8 question sets, 132 answer facts, 1200-token budget. Fact retention = share of
the answer key found in the compressed text.

| Strategy | Facts retained | Retention |
|----------|---------------|-----------|
| No compression (upper bound) | 129/132 | 97.7% |
| Truncate last 6 | 38/132 | 28.8% |
| Sliding window | 38/132 | 28.8% |
| `simple` (rules) | **111/132** | **84.1%** |
| `acs` (LLM) | 72/132 | 54.5% |
| `hcs` (LLM + code) | 68/132 | 51.5% |

Read that table carefully, because it says something inconvenient: **on literal
fact retention the free, rule-based strategy beats the LLM pipeline.** This is
not a bug. `simple` keeps messages verbatim, so every identifier it retains
matches the answer key exactly; `acs` merges and rewrites, which a substring
check cannot credit. The LLM pipeline earns its cost elsewhere — on questions
whose wording never appears in the conversation, where earlier internal
experiments showed the opposite ordering.

So: **start with `simple`,** and reach for `acs` when you can watch it fail on
paraphrase-heavy questions. Per-scenario breakdown and the caveats are in
[`benchmarks/README.md`](benchmarks/README.md).

## Does it actually help the model answer?

Fact retention measures wording, not usefulness. We also run a
question-answering evaluation: compress the history, ask a model to answer using
**only** the compressed text, and score the answer against what was actually
correct. `benchmarks/qa_eval.py`; details and caveats in
[`benchmarks/README.md`](benchmarks/README.md).

Whether you see a difference depends on whether your data contains one.

**On a real agent workload** (131 points, 12 conversations, 3k–69k tokens) the
strategies tie at a 4000-token budget — 39.8% for `simple` against 39.3% for
naive truncation. The answers in that data sit near the end of the history, so
keeping the recent tail is already close to optimal.

**On [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)**
(public, MIT licensed — ~100k tokens of history per question, the answer buried
in one of ~48 sessions, 96 independent questions) at a 16,000-token budget:

| Retriever | Correct |
|-----------|---------|
| Naive truncation | 19.8% |
| `simple` (keyword matching) | 30.2% |
| **`embedding` (semantic, `retriever_type="embedding"`)** | **44.8%** |

All three differences are significant (McNemar p = 0.021, 0.0094, < 0.001).
**Changing the retriever moves the result further than any tuning knob in this
library** — further than the token budget, the merge policy, or the
recent-message share, all of which we measured as noise. Semantic retrieval is
also the only configuration that scores at all on temporal-reasoning questions
(25%, against 0% for both alternatives).

By question type:

| Question type | truncate | keyword | embedding |
|---------------|----------|---------|-----------|
| Recall what the assistant said | 19% | 25% | **62%** |
| Temporal reasoning | 0% | 0% | **25%** |
| Knowledge update ("my current X?") | 75% | 62% | **81%** |
| Cross-session | 12% | 19% | **38%** |
| Recall what the **user** said | 12% | **69%** | 56% |

Which gives a more useful set of rules than "it compresses well":

- **Long history, need to find something stated earlier** → use
  `retriever_type="embedding"`. This is the case it wins big.
- **Questions quoting the user's own words** → keyword matching is still better
  (69% vs 56%); lexical overlap is a strong signal there.
- **Don't compress when the newest value is the answer.** For "what is my
  current X?" questions truncation is cheaper and better than keyword matching.
- **Short conversations** → the rule-based path is far faster and the
  difference disappears.

On 67 points from real Chinese work conversations the picture is flat
(truncation 51.0%, keyword 46.5%, embeddings 48.4%; nothing significant). That
is not a contradiction: in a normal work conversation the answer sits near the
end, where truncation is already optimal, so there is nothing for a retriever
to win. The rule that holds in both experiments: **retrieval pays off in
proportion to how far the answer is from the end of the history.**

## What it costs

Accuracy without a price tag is not a decision. Measured on a 100k-token
history with a 16k budget (`benchmarks/cost_benchmark.py`):

| Strategy | Median latency | Output tokens | LLM calls |
|---|---|---|---|
| truncate | **5 ms** | 15,900 | 0 |
| `simple` | **41 ms** | 6,519 | 0 |
| `embedding` | **109 ms** | 9,607 | 0 |
| `acs` | 3,298 ms | 9,579 | 1 |
| `hcs` | 3,801 ms | 6,519 | 0 |

The free paths are milliseconds; the LLM strategies are seconds. Note that
semantic retrieval costs about **60 ms** more than keyword matching —
negligible next to a single model call. "Embeddings are expensive" is not a good
reason to skip them.

## Measure it on your own conversations

Every number here comes from one dataset, and the main lesson of building them
was that the answer depends on your data: on ordinary work conversations
compression ties with truncation, while on long histories with the answer buried
early it wins by 25 points. Nobody can tell you which case you are in — so the
library measures it:

```bash
shenlun-forge eval --conversations ./my-chats --strategies truncate,simple,embedding
```

```
从 20 个对话构建测试（提取答案要点需要模型调用）...
得到 84 个测试点，开始评测...

Accuracy by strategy (budget 16000 tokens, n=84 points):
  embedding             44.1%
  simple                30.6%
  truncate              21.4%

Best: embedding. Treat differences under ~5 points as noise ...
```

Or from Python:

```python
from shenlun_forge import OpenAIProvider, build_eval_set, evaluate

llm = OpenAIProvider(model="deepseek-chat", api_key="sk-...")
points = await build_eval_set(my_conversations, llm=llm)
report = await evaluate(points, strategies=["truncate", "simple"], llm=llm)
print(report.summary())
```

It compresses each history, asks a model to answer using **only** the compressed
text, and scores the answer against key points taken from the reply that was
actually given. The keys are extracted once and frozen — extracting them at
scoring time would grade every run against a different ruler.

Two things to know before you trust a result: differences under ~5 points are
noise at this sample size, and the independent unit is the **conversation**, not
the question — adding turns to a conversation you already have sharpens nothing.
Also note this sends your transcripts to the model provider, since extraction
and judging both need an LLM.

## Status and limitations

- `acs` costs one to two LLM calls per compression. That is the point: it is
  the expensive path you take when the question needs reading.
- Degradation is reported, never hidden — but it is also never fatal. Check
  `stats["degraded"]` in production.
- The `v2` design notes (strategic reasoning, phase-aware retention,
  multi-dimensional quality scoring) are design documents, not shipped code.
- **`hcs` is not recommended.** It was tested where its one feature should
  matter — 45 points from Chinese sessions containing 26k command executions
  and file diffs — and landed level with truncation and keyword matching
  (45.9% vs 46.0% / 49.6%, p=1.00). Across six evaluations it has never been
  significantly better, while costing an extra LLM call. The code stays; the
  recommendation does not.
- **Automatic retriever routing does not work**, and this is measured rather
  than suspected. `recommend_retriever()` picks the better retriever 41% of the
  time. The deeper reason is in `benchmarks/README.md`: an answer is either
  randomly positioned (nothing predicts it, and "always retrieve" is already
  optimal) or near the end (where the two options tie, so there is nothing to
  win). Choose the retriever yourself.

## Documentation

- [`docs/quickstart.md`](docs/quickstart.md) — install to first result
- [`docs/scenarios.md`](docs/scenarios.md) — when to use it, and when not to
- [`docs/integration.md`](docs/integration.md) — OpenAI SDKs, LangChain, custom stages
- [`docs/faq.md`](docs/faq.md) — degraded runs, passthrough, cost, Chinese support
- [`benchmarks/README.md`](benchmarks/README.md) — how the numbers were measured
- [`RELEASING.md`](RELEASING.md) — release checklist
- [`CHANGELOG.md`](CHANGELOG.md)

## Contributing

Useful contributions, roughly in order of value: adapters for other frameworks
(AutoGen, CrewAI), a semantic scoring pass for `benchmarks/` that measures what
the current substring metric misses, better handling of code-level detail, and
expression-normalisation maps beyond Chinese/English.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{shenlunforge2026,
  title={ShenlunForge: Task-Driven Context Compression for LLM Agents},
  author={Liu, Rongchen},
  year={2026},
  url={https://github.com/liurongchen1/shenlun-forge}
}
```
