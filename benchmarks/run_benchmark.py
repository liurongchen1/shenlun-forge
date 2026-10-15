#!/usr/bin/env python3
"""Reproducible fact-retention benchmark for the compression strategies.

Compares every strategy against plain truncation and a sliding window on a
fixed conversation, and scores how many pre-written answer facts survive
compression (string containment, case-insensitive).

    python benchmarks/run_benchmark.py                    # rule-based only
    export DEEPSEEK_API_KEY=sk-...                        # adds acs / hcs
    python benchmarks/run_benchmark.py --json out.json

No API key is required: without one the LLM strategies are skipped and the
rule-based rows are still produced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shenlun_forge import ShenlunForge
from shenlun_forge.types import count_tokens


DATA = Path(__file__).resolve().parent / "data" / "real_dev_conversation.json"
MAX_TOKENS = 1200


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Baselines ──────────────────────────────────────────────────────────────


def baseline_none(messages: list) -> str:
    return "\n".join(f"[{m['role']}]: {m['content']}" for m in messages)


def baseline_truncate(messages: list, keep: int = 6) -> str:
    return "\n".join(f"[{m['role']}]: {m['content']}" for m in messages[-keep:])


def baseline_sliding_window(messages: list, max_tokens: int = MAX_TOKENS) -> str:
    """Keep whole messages from the end until the token budget is exhausted."""
    kept: list = []
    used = 0
    for message in reversed(messages):
        cost = count_tokens(message["content"])
        if used + cost > max_tokens:
            break
        kept.append(message)
        used += cost
    kept.reverse()
    return "\n".join(f"[{m['role']}]: {m['content']}" for m in kept)


# ── Scoring ────────────────────────────────────────────────────────────────


def score(text: str, key_facts: list) -> dict:
    haystack = text.lower()
    found = [fact for fact in key_facts if fact.lower() in haystack]
    return {
        "found": len(found),
        "total": len(key_facts),
        "retention": len(found) / len(key_facts) if key_facts else 0.0,
        "missing": [fact for fact in key_facts if fact not in found],
    }


# ── Runner ─────────────────────────────────────────────────────────────────


async def compress_with(strategy: str, messages: list, query: str, api_key: str | None):
    forge = ShenlunForge.create(
        strategy=strategy,
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    )
    return await forge.compress(messages=messages, current_query=query, max_tokens=MAX_TOKENS)


async def run(dataset: dict, api_key: str | None) -> dict:
    conversation = dataset["conversation"]
    strategies = ["truncate", "sliding_window", "simple"]
    if api_key:
        strategies += ["acs", "hcs"]

    rows: dict = {}
    for name in strategies:
        totals = {"found": 0, "total": 0, "degraded_runs": 0}
        per_scenario = []
        for scenario in dataset["scenarios"]:
            query = scenario["query"]
            degraded = False
            if name == "truncate":
                text = baseline_truncate(conversation)
            elif name == "sliding_window":
                text = baseline_sliding_window(conversation)
            else:
                result = await compress_with(name, conversation, query, api_key)
                text = result.compressed_text
                degraded = bool(result.stats.get("degraded"))

            scored = score(text, scenario["key_facts"])
            totals["found"] += scored["found"]
            totals["total"] += scored["total"]
            totals["degraded_runs"] += int(degraded)
            per_scenario.append({
                "scenario": scenario["name"],
                "found": scored["found"],
                "total": scored["total"],
                "degraded": degraded,
            })
            print(
                f"  [{name:14s}] {scenario['name']}: "
                f"{scored['found']}/{scored['total']}"
                + ("  (degraded: rule-based fallback)" if degraded else ""),
                flush=True,
            )

        rows[name] = {
            "found": totals["found"],
            "total": totals["total"],
            "retention": totals["found"] / totals["total"] if totals["total"] else 0.0,
            "degraded_runs": totals["degraded_runs"],
            "per_scenario": per_scenario,
        }

    # Reference points: the ceiling, and what truncation costs.
    full_text = baseline_none(conversation)
    no_compression = score(full_text, [
        fact for scenario in dataset["scenarios"] for fact in scenario["key_facts"]
    ])
    rows["no_compression"] = {
        "found": no_compression["found"],
        "total": no_compression["total"],
        "retention": no_compression["retention"],
        "degraded_runs": 0,
        "per_scenario": [],
    }
    return rows


def render(rows: dict, dataset: dict) -> str:
    order = ["no_compression", "truncate", "sliding_window", "simple", "acs", "hcs"]
    labels = {
        "no_compression": "No compression (upper bound)",
        "truncate": "Truncate last 6",
        "sliding_window": f"Sliding window ({MAX_TOKENS} tokens)",
        "simple": "shenlun-forge simple (rules)",
        "acs": "shenlun-forge acs (LLM)",
        "hcs": "shenlun-forge hcs (LLM + code)",
    }
    lines = [
        f"# Fact retention — {dataset['description']}",
        "",
        f"- Conversation: {len(dataset['conversation'])} messages, "
        f"{count_tokens(' '.join(m['content'] for m in dataset['conversation']))} tokens",
        f"- Questions: {len(dataset['scenarios'])}, "
        f"{sum(len(s['key_facts']) for s in dataset['scenarios'])} answer facts",
        f"- Token budget: {MAX_TOKENS}",
        "",
        "| Strategy | Facts retained | Retention |",
        "|---|---|---|",
    ]
    for name in order:
        if name not in rows:
            continue
        row = rows[name]
        note = ""
        if row.get("degraded_runs"):
            note = f" ⚠️ {row['degraded_runs']}/{len(dataset['scenarios'])} runs degraded"
        lines.append(
            f"| {labels[name]} | {row['found']}/{row['total']} | "
            f"{row['retention']:.1%}{note} |"
        )
    lines.append("")
    if any(rows[name].get("degraded_runs") for name in rows):
        lines.append(
            "> ⚠️ Rows marked *degraded* did not actually run with an LLM — "
            "they fell back to rules, so their numbers are not LLM results."
        )
        lines.append("")
    lines.append(
        "Scoring is string containment against a fixed answer key — it measures "
        "whether the identifying token survived, not whether the model could "
        "answer the question. Treat it as a directional signal."
    )

    llm_rows = [n for n in ("acs", "hcs") if n in rows]
    if llm_rows:
        lines.append("")
        lines.append(
            "Note: this key measures *literal* fact retention. The rule-based "
            "strategy copies surviving messages verbatim, so it does well here; "
            "the LLM pipeline merges and rewrites, which costs literal matches "
            "but is what lets it answer questions whose wording differs from "
            "the conversation's."
        )
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--json", type=Path, help="write raw results here")
    parser.add_argument(
        "--api-key",
        default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("SHENLUN_FORGE_API_KEY"),
        help="key for the LLM strategies; without one they are skipped",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    print(f"Dataset: {dataset['id']} — {dataset['description']}")
    if args.api_key:
        print("LLM strategies: enabled")
    else:
        print("LLM strategies: skipped (no DEEPSEEK_API_KEY / SHENLUN_FORGE_API_KEY)")
    print()

    rows = await run(dataset, args.api_key)
    report = render(rows, dataset)

    print()
    print(report)

    if args.json:
        args.json.write_text(
            json.dumps({"dataset": dataset["id"], "results": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nRaw results written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
