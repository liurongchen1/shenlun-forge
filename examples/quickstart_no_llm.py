#!/usr/bin/env python3
"""ShenlunForge in 30 seconds — no API key, no network, no LLM.

The ``simple`` strategy is pure rules: it scores each message against the
current query and fills the token budget with the most relevant ones. That is
usually all you need to stop shipping a 20k-token history to the model.

    python examples/quickstart_no_llm.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shenlun_forge import ShenlunForge


CONVERSATION = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "I'm building a restaurant booking app. Let's start with the backend."},
    {"role": "assistant", "content": "Sure. What stack are you comfortable with?"},
    {"role": "user", "content": "Python. I was thinking Django, but I've heard FastAPI is faster."},
    {"role": "assistant", "content": "FastAPI is a good fit for a JSON API, Django gives you an admin panel for free. Do you need the admin?"},
    {"role": "user", "content": "No admin needed. The staff will use our own React app."},
    {"role": "assistant", "content": "Then go with FastAPI. We rejected Django because its ORM and admin are dead weight here."},
    {"role": "user", "content": "OK. And for storage?"},
    {"role": "assistant", "content": "For the database: PostgreSQL. We ruled out MongoDB because bookings need multi-row transactions across tables."},
    {"role": "user", "content": "Makes sense. Auth?"},
    {"role": "assistant", "content": "For auth: JWT with bcrypt password hashing, 24h token expiry. Show the code?"},
    {"role": "user", "content": "Yes please."},
    {"role": "assistant", "content": "```python\nfrom fastapi import Depends, HTTPException\nfrom jose import jwt\n\ndef verify_token(token: str = Depends(oauth2_scheme)) -> dict:\n    try:\n        return jwt.decode(token, JWT_SECRET, algorithms=[\"HS256\"])\n    except JWTError:\n        raise HTTPException(status_code=401, detail=\"invalid token\")\n```"},
    {"role": "user", "content": "Nice. Remind me of the deadline?"},
    {"role": "assistant", "content": "Booking flow is due in 3 days, payment integration in 2 weeks."},
    {"role": "user", "content": "So what did we decide for the database and auth setup?"},
]

QUERY = "What did we decide for the database and auth setup?"


def main() -> None:
    # No api_key, no model, no network — rule-based selection only.
    forge = ShenlunForge.create(strategy="simple")

    result = forge.compress_sync(
        messages=CONVERSATION,
        current_query=QUERY,
        max_tokens=180,
    )

    print(f"query: {QUERY!r}\n")
    print(f"{'messages in:':<16} {len(CONVERSATION)}")
    print(f"{'tokens in:':<16} {result.original_tokens}")
    print(f"{'tokens out:':<16} {result.compressed_tokens}")
    print(f"{'kept:':<16} {result.compression_ratio:.0%} "
          f"({len(result.blocks)} of {len(CONVERSATION)} messages)")

    stats = result.stats
    if stats.get("passthrough"):
        print("\nnote: nothing was gained, so the original context was returned.")
    print(f"\nllm used: {stats.get('llm_available')}   degraded: {stats.get('degraded')}")

    print("\n" + "-" * 60)
    print(result.compressed_text)
    print("-" * 60)

    dropped = [m for m in CONVERSATION if m["content"] not in result.compressed_text]
    if dropped:
        print(f"\ndropped {len(dropped)} message(s), e.g.:")
        for msg in dropped[:3]:
            preview = msg["content"].replace("\n", " ")[:70]
            print(f"  [{msg['role']}] {preview}...")


if __name__ == "__main__":
    main()
