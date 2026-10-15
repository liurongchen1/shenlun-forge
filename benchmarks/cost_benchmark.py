#!/usr/bin/env python3
"""Measure what each strategy costs in time and LLM calls.

An accuracy number without a price tag next to it is not a decision. The
retriever that wins by 25 points on long histories is also the one that loads a
95MB model, runs on the CPU and encodes every message.

    python benchmarks/cost_benchmark.py                      # rule-based only
    export DEEPSEEK_API_KEY=sk-...                           # adds acs / hcs

Reports median latency per compression, the resulting token count, and the LLM
calls each strategy makes. The first embedding call includes model load, which
is reported separately because it is a one-off.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shenlun_forge import ShenlunForge
from shenlun_forge.types import count_tokens

DATA = ROOT / "benchmarks" / "data" / "real_dev_conversation.json"


def synthesise(target_tokens: int) -> list:
    """Build a conversation of roughly `target_tokens` for scale testing."""
    chunk = ("关于架构设计的讨论：数据库选型、认证方案、部署方式、监控告警。"
             "我们考虑了多种方案，最终倾向于 PostgreSQL 加 JWT。") * 3
    per_chunk = count_tokens(chunk)
    count = max(2, target_tokens // per_chunk)
    return [{"role": "user" if i % 3 == 0 else "assistant", "content": chunk}
            for i in range(count)]


async def time_strategy(name: str, messages: list, query: str, budget: int,
                        repeats: int, api_key: str, base_url: str) -> dict:
    calls_before = None
    if name == "truncate":
        def run():
            kept, used = [], 0
            for m in reversed(messages):
                cost = count_tokens(m["content"])
                if used + cost > budget:
                    break
                kept.append(m)
                used += cost
            return used
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            used = run()
            times.append(time.perf_counter() - t0)
        return {"strategy": "truncate", "median_s": statistics.median(times),
                "tokens": used, "llm_calls": 0, "note": "no model, no key"}

    if name == "embedding":
        forge = ShenlunForge(retriever_type="embedding")
        kwargs = {"lightweight": True}
    else:
        forge = ShenlunForge.create(strategy=name, api_key=api_key or None,
                                    base_url=base_url or None)
        kwargs = {}

    times, tokens, calls = [], 0, 0
    for i in range(repeats):
        t0 = time.perf_counter()
        result = await forge.compress(messages=messages, current_query=query,
                                      max_tokens=budget, **kwargs)
        elapsed = time.perf_counter() - t0
        if i == 0:
            tokens = result.compressed_tokens
            calls = result.stats.get("llm_calls", 0)
        else:
            times.append(elapsed)
    if not times:
        times = [elapsed]
    return {"strategy": name, "median_s": statistics.median(times),
            "tokens": tokens, "llm_calls": calls, "note": ""}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=16000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scale", type=int, default=100_000,
                        help="history size to synthesise, in tokens")
    parser.add_argument("--strategies", default="truncate,simple,embedding,acs,hcs")
    parser.add_argument("--api-key",
                        default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("SHENLUN_FORGE_API_KEY"))
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()

    messages = synthesise(args.scale)
    total = sum(count_tokens(m["content"]) for m in messages)
    print(f"对话：{len(messages)} 条消息，{total:,} token；预算 {args.budget}；"
          f"每策略重复 {args.repeats} 次取中位\n")

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if not args.api_key:
        strategies = [s for s in strategies if s not in ("acs", "hcs")]
        print("（未设置 DEEPSEEK_API_KEY，跳过 acs / hcs）\n")

    rows = []
    for name in strategies:
        row = await time_strategy(name, messages, "数据库和认证是怎么定的？",
                                  args.budget, args.repeats, args.api_key, args.base_url)
        rows.append(row)
        print(f"  {name:<10} {row['median_s']*1000:>9.1f} ms  "
              f"{row['tokens']:>7,} tok  {row['llm_calls']} 次 LLM 调用", flush=True)

    base = next((r["median_s"] for r in rows if r["strategy"] == "truncate"), None)
    print(f"\n{'策略':<12}{'中位延迟':>10}{'输出token':>11}{'LLM调用':>9}{'相对截断':>10}")
    for r in rows:
        rel = f"{r['median_s']/base:>9.0f}x" if base else "-"
        print(f"{r['strategy']:<12}{r['median_s']*1000:>9.1f}ms{r['tokens']:>11,}"
              f"{r['llm_calls']:>9}{rel}")
    print("\n注：embedding 的首次调用包含模型加载（一次性），首次与中位可能差很多。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
