# FAQ

## I passed no API key and it did not fail. What happened?

Strategies that need an LLM (`acs`, `hcs`) check for a usable provider when the
compressor is constructed. If there is none, you get a warning:

```
RuntimeWarning: strategy='acs' needs an LLM, but no API key configured; ...
```

The run still completes, using the rule-based path, and the result says so:

```python
result.stats["degraded"]          # True
result.stats["degraded_reason"]   # 'no API key configured; ...'
```

If you would rather have it be quiet, pick the strategy explicitly:

```python
forge = ShenlunForge.create(strategy="simple")   # no LLM by design, never degraded
```

Two causes are common: the key is missing, or the `openai` package is not
installed (`pip install "shenlun-forge[openai]"`).

## `compression_ratio` is 1.0 and `passthrough` is `True`. Is it broken?

No. That means compression was not worth doing: either the history is under
200 tokens, or it would not have saved at least 5% of them. The original
context was returned unchanged.

It happens on short conversations and on histories where nothing can be
dropped. Returning the input untouched is the honest outcome; the alternative
is handing back something *longer* than what you passed in, because summarising
adds structure and labels.

If you want compression anyway, lower `max_tokens` so the budget actually
binds, or check that your `current_query` reflects what you are about to ask.

## Why did it not use my whole `max_tokens` budget?

`max_tokens` is a ceiling. The rule-based strategy keeps messages that relate
to your query; if that is four messages, you get four messages. Filling the
remaining budget with unrelated chatter would make the context worse, not
better.

If you want more context kept, the query is usually the lever — a vague query
matches little. Or use `acs`, which understands paraphrase instead of matching
words.

## The result is *over* `max_tokens`. Is that a bug?

No — it is the default trade-off, and it only happens one way: the newest
message does not fit in the budget on its own.

The recent turns are protected from compression because they carry the live
task state. When the last of them is larger than what is left of the budget,
the default (`budget_overflow="keep"`) keeps it whole and reports it:

```python
result.stats["over_budget"]       # True
result.stats["recent_truncated"]  # False — nothing was cut
```

That default assumes a slightly oversized context costs you less than losing
the turn you are in the middle of. If that assumption is wrong for you —
because you bill by token count, or the target model's window is exactly
`max_tokens` — flip it:

```python
result = await forge.compress(
    messages=messages, current_query=query, max_tokens=4000,
    budget_overflow="truncate",
)
result.stats["recent_truncated"]  # True — the newest message was trimmed
result.compressed_tokens <= 4000  # budget is now a hard limit
```

CLI equivalent: `--budget-overflow truncate`. Both modes keep everything
earlier in the context identical; `truncate` only shortens the tail. If you
want neither overshoot nor truncation, raise `max_tokens` so the newest message
fits.

## `simple` or `acs`?

Ask whether your question and your answer share vocabulary.

- **They share words** ("what did we decide about the cache?" / "we set the
  cache TTL to 300s") → `simple`. Milliseconds, no cost.
- **They do not** ("why is the app slow?" / "the N+1 query in the orders list
  needs an index") → `acs`. This is exactly what the LLM pass buys you.

`simple` is a fast filter; `acs` is a reader. Neither is strictly better.

## Does it work with Chinese? Mixed Chinese and English?

Yes, and this is one of the reasons the project exists. Chinese is tokenised
into 2-character n-grams (there are no word boundaries to split on) and English
technical terms are weighted by rarity across the conversation. Terms that show
up in messages overlapping a Chinese query are used as extra signals, which is
how "开门盘点" style paraphrases get matched to "公开评估" style wording.

Mixed histories are the target case: discussion in Chinese, identifiers in
English.

## The embedding model download is slow or fails.

`EmbeddingRetriever` uses fastembed with `BAAI/bge-small-zh-v1.5` (~95MB, ONNX,
CPU-friendly), fetched from Hugging Face on first use. It needs a network path
to `huggingface.co`; once fetched the model is cached locally and later runs
are offline.

Note for users in mainland China: `HF_ENDPOINT=https://hf-mirror.com` is the
usual Hugging Face mirror, but it does **not** carry the ONNX repositories that
fastembed pulls from — pointing at it makes the download fail with
`Could not load model ... from any source`. Use a proxy that reaches
`huggingface.co` directly instead.

You do not need it at all unless you set `retriever_type="embedding"`.

## Can I make it fail loudly instead of degrading?

Currently the degradation is always reported rather than raised, so a run never
dies mid-pipeline. To make it fatal in your own code:

```python
result = await forge.compress(messages, query, max_tokens=4000)
if result.stats["degraded"]:
    raise RuntimeError(result.stats["degraded_reason"])
```

Constructing an `acs` compressor with no key warns at construction time, which
is usually early enough to catch misconfiguration at startup.

## How many extra tokens/latency does `acs` cost?

One LLM call per compression for the task analysis, plus one for merging when
there are mergeable blocks. Measured on a 16-message conversation with
`deepseek-chat`, the pipeline adds on the order of seconds, not milliseconds —
roughly 8-10s end to end in our runs, dominated by the two calls. `simple` is
milliseconds and free.

That trade is the whole design: pay the LLM when the task needs reading, skip
it when it does not.

## `shenlun-forge version` prints an old version.

Fixed in 0.3.0: the version now has a single source of truth
(`shenlun_forge/_version.py`), read by both the package and `pyproject.toml`.
Before that, `__version__` was hard-coded and drifted from the published
version.

## Where do the benchmark numbers come from?

From [`benchmarks/run_benchmark.py`](../benchmarks/run_benchmark.py), which you
can run yourself:

```bash
python benchmarks/run_benchmark.py
```

It scores every strategy on a fixed conversation (18 messages, 8 question sets,
132 answer facts) by string containment against an answer key.

That metric rewards copying text verbatim, which is why the rule-based strategy
wins it — an LLM that merges and rewrites loses literal matches even when it
preserved the meaning. [`benchmarks/README.md`](../benchmarks/README.md)
explains why that ordering is expected rather than a regression, and why you
should still run it on your own data before trusting any published number.
