# Integration

ShenlunForge is a library, not a service: it takes `messages`, returns smaller
`messages`, and does not own your model calls. That keeps it usable from any
stack.

## The core pattern

```python
result = await forge.compress(messages=history, current_query=question, max_tokens=4000)
answer = await your_model_call(result.compressed_text, question)
```

`compressed_text` is a plain string, ready to drop into a prompt. If you need
the surviving messages individually, use `result.blocks` (each keeps its
`original_indices`).

## OpenAI-compatible SDKs

Only the compression step needs a key. Pass any OpenAI-compatible endpoint:

```python
forge = ShenlunForge.create(
    strategy="acs",
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)
```

Works with DeepSeek, Groq, Together, vLLM, Ollama (OpenAI shim), Azure OpenAI
and OpenAI itself. Temperature defaults to `0.1` so analysis stays stable.

## LangChain

```python
from shenlun_forge.adapters.langchain_adapter import LangChainShenlunForge

forge = LangChainShenlunForge(strategy="acs", model="deepseek-chat", api_key="...")

# Sync or async, both accept plain dicts or LangChain message objects
out = forge.invoke({"messages": history, "query": question})
out = await forge.ainvoke({"messages": history, "query": question})
```

As a chain step:

```python
from langchain_core.runnables import RunnableLambda

compress = RunnableLambda(forge.invoke)
chain = compress | prompt | model | parser
```

The adapter also exposes `as_callback_handler()` for automatic compression when
a chain receives a long message list.

## LlamaIndex

```python
from shenlun_forge.adapters.llamaindex_adapter import LlamaIndexShenlunForge

compressor = LlamaIndexShenlunForge(strategy="simple")   # rule-based default

# Keep the message objects, drop the ones that no longer matter:
trimmed = compressor.trim(chat_history, query=question)
response = chat_engine.chat(question, chat_history=trimmed)

# Or flatten to text and put it in the prompt yourself:
context = compressor.compress_text(chat_history, query=question)
```

Async variants are `atrim()` and `acompress_text()`. The adapter reads messages
by attribute (`.role` / `.content`) and returns the very objects you passed in
— `ChatMessage`, plain dicts, or your own class. `llama-index-core` is not
required at all; the extra exists only so the import path is declared for apps
that already depend on it.

## Custom components

Each pipeline stage is a replaceable class. Swap in your own:

```python
from shenlun_forge import ShenlunForge, ACSAnalyzer, ACSRetriever
from shenlun_forge.llm import OpenAIProvider

provider = OpenAIProvider(model="gpt-4o-mini", api_key="...")

forge = ShenlunForge(
    analyzer=ACSAnalyzer(llm=provider, use_fast_path=True),
    retriever=ACSRetriever(llm=provider, top_k=30),
    # processor and formatter default to the ACS implementations
)
```

Every stage implements an abstract base class in `shenlun_forge.base`
(`BaseAnalyzer`, `BaseRetriever`, `BaseProcessor`, `BaseFormatter`,
`LLMProvider`), so a custom stage only has to match that interface.

## Custom strategies

Register a strategy once and it becomes available to `create()`, the CLI and
`list_strategies()`:

```python
from shenlun_forge import ShenlunForge, register_strategy

class LastNTurns:
    """Keep the most recent messages, ignore the query."""

    async def compress(self, messages, current_query="", max_tokens=4000, **kwargs):
        ...

register_strategy("last_n", LastNTurns)

forge = ShenlunForge.create(strategy="last_n")
```

A custom strategy only has to provide
`async def compress(messages, current_query="", max_tokens=4000, **kwargs)`
returning a `CompressionResult`. If its `__init__` accepts
`model`/`api_key`/`base_url`, `create()` passes them through.

## Detecting a degraded run

Because a missing key must never look like a successful LLM compression, every
result reports it:

```python
result = await forge.compress(messages, query, max_tokens=4000)

if result.stats["degraded"]:
    logger.warning("compression degraded: %s", result.stats["degraded_reason"])
if result.stats["passthrough"]:
    logger.info("no compression possible; original context used")
```

Use it as a health signal: if `degraded` is `True` in production, your pipeline
is quietly costing more tokens than you think.

## Choosing a retriever

| `retriever_type` | Cost | Best for |
|---|---|---|
| `"rule"` (default) | No model, no network | Anything with shared vocabulary; Chinese/mixed text |
| `"embedding"` | ~95MB ONNX model, CPU | Lexically different wording; offline semantic matching |

```python
forge = ShenlunForge(retriever_type="embedding")
```

The model is fetched from `huggingface.co` on first use and cached afterwards.
The usual `HF_ENDPOINT=https://hf-mirror.com` mirror does not carry fastembed's
ONNX repositories — pointing at it fails with "Could not load model ... from any
source". Use a proxy that can reach huggingface.co.

## Concurrency and caching

- A `ShenlunForge` instance is safe to reuse; each `compress()` call resets its
  own LLM counters.
- `EmbeddingRetriever` caches embeddings in memory keyed by content hash, so
  repeated queries over the same conversation are near-free.
