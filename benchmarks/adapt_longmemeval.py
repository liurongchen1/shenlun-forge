#!/usr/bin/env python3
"""Convert LongMemEval into the qa-set format used by `qa_eval.py`.

LongMemEval (https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned,
MIT licensed) is a good stress test for a compression strategy, for a reason
that has nothing to do with how impressive the conversations are:

- each question comes with ~100k tokens of history across ~48 sessions
- the answer lives in one or two of those sessions, and the question gives no
  hint about where
- the expected answer is a short, checkable fact

That makes it a clean way to separate "smart selection" from "keep the most
recent messages": a strategy that just keeps the tail will miss the needle most
of the time, by construction.

    python benchmarks/adapt_longmemeval.py --input longmemeval_s.json \\
        --out longmemeval-qa.json --limit 100

The output embeds ~100k tokens of history per point, so keep it out of version
control; only this script belongs in the repo.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shenlun_forge.types import count_tokens


def flatten_sessions(entry: dict) -> list:
    """Flatten haystack_sessions into a single message list."""
    messages = []
    for session in entry.get("haystack_sessions") or []:
        for message in session:
            content = (message.get("content") or "").strip()
            if not content:
                continue
            messages.append({
                "role": message.get("role") or "user",
                "content": content,
            })
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100,
                        help="how many questions to convert (0 = all)")
    parser.add_argument("--stratify", action="store_true", default=True,
                        help="sample evenly across question_type")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    print(f"loaded {len(raw)} entries")

    if args.stratify:
        by_type: dict = {}
        for entry in raw:
            by_type.setdefault(entry.get("question_type", "unknown"), []).append(entry)
        rng = random.Random(args.seed)
        per_type = max(1, args.limit // max(len(by_type), 1))
        selected = []
        for name, items in sorted(by_type.items()):
            rng.shuffle(items)
            selected.extend(items[:per_type])
            print(f"  {name:<26} {min(per_type, len(items))} of {len(items)}")
        selected.sort(key=lambda e: e["question_id"])
    else:
        selected = raw[:args.limit] if args.limit else raw

    if args.limit:
        selected = selected[:args.limit]

    points = []
    for entry in selected:
        history = flatten_sessions(entry)
        if len(history) < 6:
            continue
        points.append({
            "id": entry["question_id"],
            "conversation": entry["question_id"],
            "turn": 0,
            "question_type": entry.get("question_type"),
            "history_tokens": sum(count_tokens(m["content"]) for m in history),
            "history": history,
            "query": entry["question"],
            # The whole expected answer is the answer key. Short and checkable,
            # but it means scoring is all-or-nothing per point.
            "reference": entry["answer"],
            "points": [str(entry["answer"])],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "name": "longmemeval-s",
        "source": str(args.input),
        "count": len(points),
        "points": points,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(points)} points to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
