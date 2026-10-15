#!/usr/bin/env python3
"""Replay real conversations through ShenlunForge, in the harness's own format.

The point is to be able to run someone else's replay report against this
library without asking them to change anything. Input is a conversation export
(``[{id, role, content}]``, oldest first); output is the same
``{label}-sim-{id}.json`` shape a TypeScript harness writes for its ``simulate``
command, so a report script written for that harness can read both.

The timing that matters is reproduced exactly: before each user turn the model
is shown the compressed history of everything *before* that turn, and the turn
itself is appended afterwards. That is what makes "how much of what was said
early is still visible at turn N" a well-defined question.

    python benchmarks/replay_curve.py --label shenlun --budget 3600 22 138 50

Sanity check on the budget: a harness that estimates tokens as ``len/3``
reaches for 1800 "tokens" but is really sending ~3600, so comparing the two
means matching real tokens, not nominal ones.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shenlun_forge import ShenlunForge
from shenlun_forge.types import count_tokens


def harness_tokens(text: str) -> int:
    """The `len(text)/3` estimate a server-side harness uses, for like-for-like
    reporting of context size alongside its own numbers."""
    return math.ceil(len(text) / 3)


def load_conversation(directory: Path, conversation_id: int) -> list:
    path = directory / f"conv{conversation_id}.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path} — export it with:\n"
            f'  psql "$DATABASE_URL" -At -c \'select json_agg(row_to_json(t)) from '
            f'(select id, role, content from "Message" where "conversationId" = '
            f"{conversation_id} order by id) t;' > {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def replay(
    conversation_id: int,
    rows: list,
    forge,
    budget: int,
    recent: int,
    label: str,
    matched_budgets: list | None = None,
    protect_user_messages: bool = False,
):
    series = []
    history: list[dict[str, str]] = []

    for row in rows:
        if row.get("role") == "user":
            query = row.get("content") or ""
            turn_budget = budget
            if matched_budgets and len(series) < len(matched_budgets):
                # Same amount of context to work with, turn by turn. Anything
                # else measures who was allowed to send more, not who compresses
                # better — and a harness that keeps a verbatim tail alongside a
                # small summary sends far more than its own summary budget says.
                turn_budget = max(200, matched_budgets[len(series)])
            if history:
                result = forge.compress_sync(
                    messages=history,
                    current_query=query,
                    max_tokens=turn_budget,
                    recent_window=recent,
                    protect_user_messages=protect_user_messages,
                )
                context_text = result.compressed_text
            else:
                # Nothing before the first turn: nothing to compress.
                context_text = ""
            context_messages = [
                {"role": row.get("role") or "user", "content": row.get("content") or ""}
                for row in history
            ]
            series.append({
                "turn": len(series) + 1,
                "uptoId": row["id"],
                "historyBefore": len(history),
                "currentUserContent": query,
                "contextMessages": len(context_messages),
                "contextTokens": harness_tokens(context_text),
                "summaryLength": len(context_text),
                "contextText": context_text,
            })
        history.append({
            "role": row.get("role") or "user",
            "content": row.get("content") or "",
        })

    return {"realId": conversation_id, "label": label, "series": series}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="+", type=int, help="conversation ids to replay")
    parser.add_argument("--label", default="shenlun")
    parser.add_argument("--dir", type=Path,
                        default=Path(os.environ.get("HERMES_REPLAY_DIR", "/tmp/hermes-realdata")))
    parser.add_argument("--budget", type=int, default=3600,
                        help="real tokens for the compressed context (default 3600, "
                             "which matches a len/3 harness asking for 1800)")
    parser.add_argument("--recent", type=int, default=14,
                        help="most recent messages kept verbatim (default 14)")
    parser.add_argument("--strategy", default="simple")
    parser.add_argument("--match-label", default=None,
                        help="read <label>-sim-<id>.json and give this run the same "
                             "context size the other implementation sent, turn by turn")
    parser.add_argument("--protect-user", action="store_true",
                        help="rank user turns ahead of everything else (see "
                             "SimpleStrategy.protect_user_messages)")
    args = parser.parse_args()

    forge = ShenlunForge.create(strategy=args.strategy)

    for conversation_id in args.ids:
        rows = load_conversation(args.dir, conversation_id)
        matched = None
        if args.match_label:
            mate = args.dir / f"{args.match_label}-sim-{conversation_id}.json"
            if not mate.exists():
                raise SystemExit(f"missing {mate} — run the other side's replay first")
            matched = [
                count_tokens(item.get("contextText") or "")
                for item in json.loads(mate.read_text(encoding="utf-8"))["series"]
            ]
        started = time.time()
        payload = replay(
            conversation_id, rows, forge, args.budget, args.recent, args.label,
            matched, args.protect_user,
        )
        out = args.dir / f"{args.label}-sim-{conversation_id}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        last = payload["series"][-1] if payload["series"] else {}
        print(
            f"[sim:{args.label}] real={conversation_id} 轮数={len(payload['series'])} "
            f"末轮上下文tokens={last.get('contextTokens', 0)} "
            f"用时={int((time.time() - started) * 1000)}ms → {out.name}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
