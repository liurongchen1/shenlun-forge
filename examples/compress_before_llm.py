#!/usr/bin/env python3
"""Compress the conversation history before you send it to the model.

This is the main use case: your agent loop keeps appending to ``messages``
until the history no longer fits (or costs too much). Pass the history through
ShenlunForge first, then send ``result.compressed_text`` instead.

Needs an LLM for the ``acs`` strategy:

    export SHENLUN_FORGE_API_KEY="sk-..."
    export SHENLUN_FORGE_BASE_URL="https://api.deepseek.com/v1"
    python examples/compress_before_llm.py

Without a key it tells you why and falls back to the rule-based strategy.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shenlun_forge import ShenlunForge


def build_history() -> list:
    """Stand-in for whatever your agent accumulated over the last N turns."""
    history = [
        {"role": "system", "content": "You are a senior backend engineer."},
        {"role": "user", "content": "We're building a payments service. Where do we start?"},
        {"role": "assistant", "content": "Start with the data model. Payments need immutable ledger rows."},
        {"role": "user", "content": "Postgres or MySQL?"},
        {"role": "assistant", "content": "Postgres. We rejected MySQL because we need partial indexes and stronger JSON support."},
        {"role": "user", "content": "How do we handle webhooks from Stripe?"},
        {"role": "assistant", "content": (
            "Verify the signature first, then persist an idempotency key so retries are safe.\n\n"
            "```python\n"
            "def handle_webhook(payload: bytes, sig: str) -> None:\n"
            "    event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)\n"
            "    if seen(event['id']):\n"
            "        return  # idempotent replay\n"
            "    mark_seen(event['id'])\n"
            "```"
        )},
        {"role": "user", "content": "What did we rule out for the database?"},
    ]
    # Pad so the compression has something to do.
    for i in range(30):
        role = "user" if i % 2 else "assistant"
        history.append({
            "role": role,
            "content": f"Follow-up {i}: we also discussed retry policy {i} with "
                       f"backoff base {1 + i % 5}s, request timeout={10 + i}s, "
                       f"the p99 latency budget, and the metrics dashboard panel {i}.",
        })
    history.append({"role": "user", "content": "Summarise the database decision."})
    return history


async def main() -> None:
    history = build_history()
    query = "What database did we settle on, and what did we rule out?"

    has_key = bool(
        os.getenv("SHENLUN_FORGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    strategy = "acs" if has_key else "simple"
    if not has_key:
        print("No SHENLUN_FORGE_API_KEY / OPENAI_API_KEY found.")
        print("Falling back to the rule-based 'simple' strategy.\n")

    forge = ShenlunForge.create(
        strategy=strategy,
        model=os.getenv("SHENLUN_FORGE_MODEL", "deepseek-chat"),
        api_key=os.getenv("SHENLUN_FORGE_API_KEY"),
        base_url=os.getenv("SHENLUN_FORGE_BASE_URL") or None,
    )

    result = await forge.compress(
        messages=history,
        current_query=query,
        max_tokens=600,
        verbose=True,
    )

    print(f"strategy      : {strategy}")
    print(f"tokens        : {result.original_tokens} -> {result.compressed_tokens} "
          f"({result.compression_ratio:.0%} kept)")
    print(f"llm available : {result.stats.get('llm_available')}")
    if result.stats.get("degraded"):
        print(f"degraded      : {result.stats.get('degraded_reason')}")
    if result.stats.get("passthrough"):
        print("passthrough   : original context returned unchanged")

    # What you would actually send to the model:
    prompt = (
        "Conversation context (compressed):\n"
        f"{result.compressed_text}\n\n"
        f"Answer this question using the context above:\n{query}"
    )
    print(f"\n--- prompt you would send ({len(prompt)} chars) ---")
    print(prompt[:1200])


if __name__ == "__main__":
    asyncio.run(main())
