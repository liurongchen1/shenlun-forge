#!/usr/bin/env python3
"""Use ShenlunForge as a compression step inside a LangChain pipeline.

Two integration styles:

1. Call it yourself before the model call (explicit, easiest to reason about).
2. ``LangChainShenlunForge`` as a Runnable-like component that accepts
   ``{"messages": [...], "query": "..."}`` and returns the compressed text.

``langchain-core`` is not required for the conversion logic; the adapter
accepts plain dicts and LangChain ``BaseMessage`` objects alike.

    python examples/langchain_middleware.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shenlun_forge.adapters.langchain_adapter import LangChainShenlunForge


HISTORY = [
    {"role": "system", "content": "You are a support agent for an e-bike shop."},
    {"role": "user", "content": "My battery drains in 10 km."},
    {"role": "assistant", "content": "That's below spec. Which model and firmware?"},
    {"role": "user", "content": "City-3, firmware 2.4.1. It started after the last update."},
    {"role": "assistant", "content": "Firmware 2.4.1 had a BMS bug. We're rolling back to 2.3.9 for affected units."},
    {"role": "user", "content": "How do I roll back?"},
    {"role": "assistant", "content": "Hold the power button 8s, then use the app's Service menu. Warranty covers it."},
    {"role": "user", "content": "And what if it doesn't help?"},
    {"role": "assistant", "content": "Then we replace the BMS module free of charge within 24 months."},
    {"role": "user", "content": "So what's the actual fix for my battery?"},
]

# A handful of unrelated turns, the kind that bloat a real history.
for _i in range(14):
    HISTORY.insert(-1, {
        "role": "assistant" if _i % 2 else "user",
        "content": f"Unrelated ticket {_i}: release-notes wording, "
                   f"the {2020 + _i} model lineup, and shipping lead times.",
    })


def main() -> None:
    has_key = bool(os.getenv("SHENLUN_FORGE_API_KEY") or os.getenv("OPENAI_API_KEY"))
    forge = LangChainShenlunForge(
        strategy="acs" if has_key else "simple",
        model=os.getenv("SHENLUN_FORGE_MODEL", "deepseek-chat"),
        api_key=os.getenv("SHENLUN_FORGE_API_KEY"),
        base_url=os.getenv("SHENLUN_FORGE_BASE_URL") or None,
        max_tokens=300,
    )

    # Style 1: explicit call. `invoke` is sync; `ainvoke` is async.
    out = forge.invoke({
        "messages": HISTORY,
        "query": "Which firmware version do I roll back to, and how?",
    })

    print("compressed_text:")
    print(out["compressed_text"])
    print()
    print(f"ratio={out['compression_ratio']:.0%} "
          f"({out['original_tokens']} -> {out['compressed_tokens']} tokens)")

    # Style 2: drop-in for a chain step.
    #
    #   from langchain_core.runnables import RunnableLambda
    #   compress = RunnableLambda(forge.invoke)
    #   chain = compress | prompt | model | parser
    #
    # The adapter also exposes `as_callback_handler()` for automatic
    # compression when a chain receives a long message list.


if __name__ == "__main__":
    main()
