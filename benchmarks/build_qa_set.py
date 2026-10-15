#!/usr/bin/env python3
"""Build a question-answering evaluation set from real conversations.

Fact-retention scoring (see `run_benchmark.py`) rewards copying text verbatim,
which is not what compression is for. This builder creates a set that measures
the thing that actually matters: **can a model still answer the question when
it only sees the compressed context?**

For every substantive user turn it records:

  history     - the conversation up to that turn (what gets compressed)
  query       - the user's message
  reference   - the answer that was actually given at the time
  points      - 3-6 key points that answer covers, extracted once and frozen

Freezing the key points is the whole point. Extracting them at scoring time
means every run is graded against a different ruler - enough variance to swamp
the differences between strategies you are trying to measure.

    export DEEPSEEK_API_KEY=sk-...
    python benchmarks/build_qa_set.py --conversations ./samples --out qa-set.json

The output contains raw conversation text. Keep it out of version control.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shenlun_forge.llm import OpenAIProvider
from shenlun_forge.types import count_tokens


# Turns that carry no question worth testing.
LOW_VALUE_QUERIES = {
    "继续", "好的", "嗯", "ok", "可以", "继续吧", "再来", "好", "是的", "对",
    "嗯嗯", "不用了", "go on", "continue", "okay", "yes", "thanks",
}
# Pasted terminal output and scheduled-job boilerplate are not questions.
NOISE_PREFIXES = ("[定时任务", "didimac@", "root@", "```", "$ ", "PS C:\\")

POINTS_PROMPT = """从下面的回答里提取 3-6 个关键要点，用于检验另一个回答是否覆盖了同样的信息。

要点应当是具体的事实、数字、结论、步骤或名称，不要写成"介绍了X"这种空泛概括。

回答：
{reference}

只输出 JSON：{{"points": ["...", "..."]}}"""


def _to_messages(raw) -> list:
    """Accept either {"messages": [...]} or a bare list."""
    if isinstance(raw, dict):
        return raw.get("messages") or []
    return raw if isinstance(raw, list) else []


def collect_points(
    conversations_dir: Path,
    min_history_tokens: int,
    min_reference_chars: int,
    min_query_chars: int,
    limit: Optional[int],
    glob: str = "*.json",
    max_history_tokens: int = 0,
) -> list:
    points = []
    for path in sorted(conversations_dir.glob(glob)):
        if path.name.startswith((".", "index")):
            continue
        try:
            messages = _to_messages(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        for i, message in enumerate(messages):
            if message.get("role") != "user" or i + 1 >= len(messages):
                continue
            nxt = messages[i + 1]
            if nxt.get("role") != "assistant":
                continue

            query = (message.get("content") or "").strip()
            reference = (nxt.get("content") or "").strip()

            if len(query) < min_query_chars or query.lower() in LOW_VALUE_QUERIES:
                continue
            if query.startswith(NOISE_PREFIXES):
                continue
            if len(reference) < min_reference_chars:
                continue

            history = messages[:i]
            if len(history) < 6:
                continue
            if max_history_tokens:
                # Keep the most recent N tokens of history. Real agent
                # transcripts grow without bound; capping keeps the compression
                # ratio in a range where the comparison means something, and
                # keeps encoding cost sane.
                budget, cut = 0, 0
                for k in range(len(history) - 1, -1, -1):
                    budget += count_tokens(history[k].get("content") or "")
                    if budget > max_history_tokens:
                        cut = k + 1
                        break
                history = history[cut:]
                if len(history) < 6:
                    continue
            history_tokens = sum(count_tokens(m.get("content") or "") for m in history)
            if history_tokens < min_history_tokens:
                continue

            points.append({
                "id": f"{path.stem}#{i}",
                "conversation": path.stem,
                "turn": i,
                "history_tokens": history_tokens,
                "history": history,
                "query": query,
                "reference": reference,
            })

    # Longest first, then keep a spread of sizes rather than only the giants.
    points.sort(key=lambda p: -p["history_tokens"])
    if limit:
        buckets: dict = {"big": [], "mid": [], "small": []}
        for p in points:
            key = ("big" if p["history_tokens"] >= 25000
                   else "mid" if p["history_tokens"] >= 10000 else "small")
            buckets[key].append(p)
        per = max(1, limit // 3)
        points = buckets["big"][:per] + buckets["mid"][:per] + buckets["small"][:per]
        points = sorted(points, key=lambda p: -p["history_tokens"])[:limit]

    return points


async def freeze_key_points(points: list, llm: OpenAIProvider) -> None:
    """Extract the key points once and store them on each point."""
    for index, point in enumerate(points, 1):
        result = await llm.complete_json(
            POINTS_PROMPT.format(reference=point["reference"]), {}
        )
        pts = [str(p).strip() for p in result.get("points", []) if str(p).strip()]
        point["points"] = pts[:6]
        print(f"  [{index}/{len(points)}] {point['id']}: {len(point['points'])} points",
              flush=True)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=Path, required=True,
                        help="folder of conversation JSON files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--min-history-tokens", type=int, default=3000)
    parser.add_argument("--min-reference-chars", type=int, default=400)
    parser.add_argument("--min-query-chars", type=int, default=15)
    parser.add_argument("--glob", default="*.json",
                        help="which files in --conversations to read (default: *.json)")
    parser.add_argument("--max-history-tokens", type=int, default=0,
                        help="cap history at the most recent N tokens (0 = no cap)")
    parser.add_argument("--limit", type=int, default=None,
                        help="keep at most N points, spread across size buckets")
    parser.add_argument("--api-key",
                        default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("SHENLUN_FORGE_API_KEY"))
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("set DEEPSEEK_API_KEY (or pass --api-key): extracting key points needs an LLM")

    points = collect_points(
        args.conversations, args.min_history_tokens,
        args.min_reference_chars, args.min_query_chars, args.limit, args.glob,
        args.max_history_tokens,
    )
    if not points:
        print("No usable evaluation points found - check the thresholds.")
        return 1

    print(f"Found {len(points)} candidate points; extracting key points...")
    llm = OpenAIProvider(model=args.model, api_key=args.api_key,
                         base_url=args.base_url, temperature=0.0)
    await freeze_key_points(points, llm)
    await llm.aclose()

    kept = [p for p in points if p["points"]]
    payload = {
        "name": args.name or args.conversations.name,
        "source": str(args.conversations),
        "filters": {
            "min_history_tokens": args.min_history_tokens,
            "min_reference_chars": args.min_reference_chars,
            "min_query_chars": args.min_query_chars,
            "glob": args.glob,
        },
        "count": len(kept),
        "points": kept,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    sizes = [p["history_tokens"] for p in kept]
    print(f"\nWrote {len(kept)} points to {args.out}")
    print(f"history tokens: min {min(sizes):,} / median {sorted(sizes)[len(sizes)//2]:,} / max {max(sizes):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
