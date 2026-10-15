#!/usr/bin/env python3
"""Score compression strategies on "can the model still answer the question?".

Runs a frozen evaluation set (see `build_qa_set.py`) and, for each strategy,
compresses the history, asks a model to answer using *only* the compressed
context, then scores the answer against the frozen key points.

Scoring is blind: the judge is not told which strategy produced the answer.

    export DEEPSEEK_API_KEY=sk-...
    python benchmarks/qa_eval.py --set qa-set.json --strategies truncate,simple,acs,hcs

`truncate` is the naive baseline (keep the most recent messages that fit) and is
the number to beat: on real conversations it is strong, because the answer often
lives in the most recent turns.
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

from shenlun_forge import ShenlunForge
from shenlun_forge.llm import OpenAIProvider
from shenlun_forge.types import count_tokens


ANSWER_PROMPT = """以下是之前对话的上下文（可能已被压缩，部分内容缺失）：

{context}

---
只依据以上上下文回答用户的问题。上下文里没有相关信息时，直接说“上下文里没有相关信息”。

问题：{query}"""

JUDGE_PROMPT = """问题：
{query}

需要覆盖的要点：
{points}

待评估回答：
{candidate}

判断待评估回答覆盖了哪些要点（语义一致即可，不要求措辞相同；含糊带过不算覆盖）。
只输出 JSON：{{"covered": [0, 2]}}"""


def truncate_history(history: list, budget: int) -> str:
    """Keep the most recent messages that fit — the baseline to beat."""
    kept, used = [], 0
    for message in reversed(history):
        cost = count_tokens(message.get("content") or "")
        if used + cost > budget:
            break
        kept.append(message)
        used += cost
    kept.reverse()
    return "\n".join(f"[{m.get('role')}] {m.get('content')}" for m in kept)


async def compress_history(
    strategy: str,
    point: dict,
    budget: int,
    forge_cache: dict,
    api_key: str,
    base_url: str,
    model: str,
    dropped_notice: bool = True,
    external_cmd: Optional[str] = None,
    external_budget_scale: float = 1.0,
) -> tuple:
    """Return (context_text, compression_ratio) for one strategy."""
    if external_cmd:
        text, _ = await compress_external(
            external_cmd, point, budget, external_budget_scale
        )
        # Ratio measured here, with this file's tokenizer, rather than taken
        # from the compressor. Otherwise each strategy reports its own idea of
        # a token and the ratios cannot be lined up with each other.
        original = "\n".join(
            f"[{m.get('role')}] {m.get('content')}" for m in point["history"]
        )
        return text, count_tokens(text) / max(count_tokens(original), 1)
    if strategy == "truncate":
        return truncate_history(point["history"], budget), 1.0
    if strategy == "none":
        text = "\n".join(f"[{m.get('role')}] {m.get('content')}" for m in point["history"])
        return text, 1.0
    if strategy == "embedding":
        # Local semantic retrieval, no LLM call: ShenlunForge's lightweight mode.
        if strategy not in forge_cache:
            forge_cache[strategy] = ShenlunForge(retriever_type="embedding")
        result = await forge_cache[strategy].compress(
            messages=point["history"], current_query=point["query"],
            max_tokens=budget, lightweight=True,
            dropped_notice=dropped_notice,
        )
        return result.compressed_text, result.compression_ratio

    if strategy not in forge_cache:
        forge_cache[strategy] = ShenlunForge.create(
            strategy=strategy, api_key=api_key, base_url=base_url, model=model,
        )
    result = await forge_cache[strategy].compress(
        messages=point["history"], current_query=point["query"], max_tokens=budget,
        dropped_notice=dropped_notice,
    )
    return result.compressed_text, result.compression_ratio


async def compress_external(
    cmd: str, point: dict, budget: int, budget_scale: float = 1.0
) -> tuple:
    """Run someone else's compressor as a subprocess and use its output.

    The contract is deliberately tiny — JSON in on stdin, JSON out on stdout —
    so that any implementation can be measured on the same points, against the
    same frozen answer key, with the same judge. That is the whole point: a
    score only means something when the only thing that differs between two
    numbers is the compressor.

    stdin:  {"messages": [{"role": ..., "content": ...}], "query": ..., "maxTokens": N}
    stdout: {"text": "...", "ratio": 0.31}

    ``budget_scale`` exists because "how many tokens fit" is not a shared unit.
    A compressor that estimates tokens as ``len(text) / 3`` — common on servers
    with no tokenizer installed — sees Chinese text as ~2.2x cheaper than it
    really is, so the same ``maxTokens`` buys it far more context. Measuring
    that compressor against an equal budget means dividing the number you hand
    it by the ratio you measured. Pass it with ``--external-budget-scale``.
    """
    payload = json.dumps({
        "messages": point["history"],
        "query": point["query"],
        "maxTokens": max(1, int(budget * budget_scale)),
    }).encode()

    process = await asyncio.create_subprocess_shell(
        cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(payload)
    if process.returncode != 0:
        raise RuntimeError(
            f"external compressor failed ({process.returncode}): "
            f"{stderr.decode(errors='replace')[:500]}"
        )
    try:
        data = json.loads(stdout.decode())
    except json.JSONDecodeError:
        raise RuntimeError(
            "external compressor did not print JSON on stdout: "
            f"{stdout.decode(errors='replace')[:200]!r}"
        )
    text = data.get("text") or data.get("compressedText") or ""
    if not text:
        raise RuntimeError("external compressor returned no text")
    ratio = data.get("ratio") or data.get("compressionRatio") or 1.0
    return text, float(ratio)


async def evaluate_point(
    point: dict,
    strategy: str,
    budget: int,
    forge_cache: dict,
    answerer: OpenAIProvider,
    judge: OpenAIProvider,
    api_key: str,
    base_url: str,
    model: str,
    samples: int,
    dropped_notice: bool = True,
    external_cmd: Optional[str] = None,
    external_budget_scale: float = 1.0,
) -> dict:
    """Compress once, then answer+judge `samples` times.

    Compressing once is deliberate: recompressing would multiply the cost of
    the LLM strategies without telling us anything, since what we are trying to
    measure is how much the *scoring* moves between identical runs.
    """
    context, ratio = await compress_history(
        strategy, point, budget, forge_cache, api_key, base_url, model,
        dropped_notice, external_cmd, external_budget_scale)

    total = max(len(point["points"]), 1)
    scores, answers = [], []
    for _ in range(max(samples, 1)):
        # Long histories make for large requests, and the API occasionally times
        # out on them. Retry the answer+judge pair rather than losing the point
        # (the compression above is reused, so a retry is cheap).
        for attempt in range(3):
            try:
                answer = await answerer.complete(
                    ANSWER_PROMPT.format(context=context, query=point["query"]))
                verdict = await judge.complete_json(JUDGE_PROMPT.format(
                    query=point["query"],
                    points=json.dumps(point["points"], ensure_ascii=False),
                    candidate=answer,
                ), {})
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                print(f"  retry {point['id']} ({type(exc).__name__})", flush=True)
                await asyncio.sleep(2 * (attempt + 1))
        covered = [c for c in verdict.get("covered", [])
                   if isinstance(c, int) and 0 <= c < len(point["points"])]
        scores.append(len(covered) / total)
        answers.append(answer)

    return {
        "id": point["id"],
        "history_tokens": point["history_tokens"],
        "compression_ratio": round(ratio, 3),
        "total": total,
        "score": _mean(scores),
        # How much this one point moved between identical runs. Averaged across
        # points, this is the metric's own noise floor.
        "score_spread": (max(scores) - min(scores)) if len(scores) > 1 else 0.0,
        "scores": scores,
        "answer_chars": int(_mean([len(a) for a in answers])),
        "answer": answers[0],
    }


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _fmt(value) -> str:
    """None means "no points in this band", which is not the same as 0%."""
    return "-" if value is None else f"{value:.1%}"


def summarise(rows: list, samples: int = 1) -> dict:
    def band(low, high):
        return [r["score"] for r in rows if low <= r["history_tokens"] < high]

    small, mid, big = band(0, 10000), band(10000, 25000), band(25000, 10**9)
    scores = [r["score"] for r in rows]
    n = len(scores)

    # Two different uncertainties, and confusing them is how you talk yourself
    # into a difference that is not there:
    #   ci95      - how precisely we know the mean of *this* point set
    #               (shrinks as you add points)
    #   rerun     - how much the mean itself moves when the identical run is
    #               repeated (does NOT shrink with more points; it is the floor
    #               set by judge variance)
    se = 0.0
    if n > 1:
        mean = _mean(scores)
        variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
        se = (variance ** 0.5) / (n ** 0.5)

    rerun = None
    if samples > 1:
        run_means = []
        for i in range(samples):
            values = [r["scores"][i] for r in rows if len(r.get("scores", ())) > i]
            if values:
                run_means.append(_mean(values))
        if len(run_means) > 1:
            rerun = max(run_means) - min(run_means)

    return {
        "n": n,
        "all": _mean(scores),
        "stderr": se,
        "ci95": 1.96 * se,
        "rerun_swing": rerun,
        "run_means": run_means if samples > 1 else None,
        "small_3k_10k": _mean(small) if small else None,
        "mid_10k_25k": _mean(mid) if mid else None,
        "big_25k+": _mean(big) if big else None,
        "mean_ratio": _mean([r["compression_ratio"] for r in rows]),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, required=True, dest="set_path")
    parser.add_argument("--strategies", default="truncate,simple,acs,hcs")
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--samples", type=int, default=1,
                        help="repeat answering+judging N times; averages out judge variance")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N points")
    parser.add_argument("--rebuild-every", type=int, default=0,
                        help="drop cached compressor state every N points (0 = never). "
                             "The embedding retriever caches every message it encodes, "
                             "which grows without bound on long histories.")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="process at most N not-yet-scored points, then stop "
                             "(0 = all). Progress is written to --out as it goes, so "
                             "repeating the same command continues where it left off. "
                             "Use this to cap peak memory: a fresh process releases "
                             "everything the previous batch allocated.")
    parser.add_argument("--out", type=Path, default=None, help="write full results as JSON")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--api-key",
                        default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("SHENLUN_FORGE_API_KEY"))
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--external", action="append", default=[], metavar="NAME=COMMAND",
                        help="Score someone else's compressor as a strategy. The command "
                             "receives {messages, query, maxTokens} as JSON on stdin and "
                             "must print {\"text\": ..., \"ratio\": ...} on stdout. Repeat "
                             "for several. This is how a comparison stays honest: same "
                             "points, same frozen answer key, same judge, one variable.")
    parser.add_argument("--external-budget-scale", type=float, default=1.0,
                        help="Multiply maxTokens handed to --external commands. Use it "
                             "when the other compressor counts tokens differently: a "
                             "len/3 estimator treats Chinese as ~2.2x cheaper than a real "
                             "tokenizer, so passing 0.45 gives it the same real budget.")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("set DEEPSEEK_API_KEY (or pass --api-key)")

    data = json.loads(args.set_path.read_text(encoding="utf-8"))
    points = data["points"]
    if args.limit:
        points = points[:args.limit]

    # A strategy may be written "simple" or "simple-nodrop". The suffix runs the
    # same compressor with the closing "what was left out" note suppressed, which
    # is how the note's effect is measured: everything else has to stay identical.
    specs = [s.strip() for s in args.strategies.split(",") if s.strip()]
    strategies = specs
    dropped_notice_for = {
        name: not name.endswith("-nodrop") for name in specs
    }
    base_strategy_for = {
        name: (name[: -len("-nodrop")] if name.endswith("-nodrop") else name)
        for name in specs
    }

    external_for: dict = {}
    for spec in args.external:
        if "=" not in spec:
            parser.error(f"--external expects NAME=COMMAND, got {spec!r}")
        name, command = spec.split("=", 1)
        name = name.strip()
        if name in base_strategy_for:
            parser.error(f"--external name {name!r} collides with --strategies")
        external_for[name] = command.strip()
        base_strategy_for[name] = name
        dropped_notice_for[name] = True
    strategies = specs + [name for name in external_for if name not in specs]

    # ── Resume support ───────────────────────────────────────────────────────
    # Results are written after every point, so a batch can be stopped at any
    # time and the same command rerun to continue. That also bounds peak memory
    # on datasets where a single point is already large.
    previous: dict = {}
    if args.out and args.out.exists():
        try:
            previous = json.loads(args.out.read_text(encoding="utf-8")).get("results", {})
        except (json.JSONDecodeError, OSError):
            print(f"warning: could not read {args.out}, starting fresh")
            previous = {}

    restored = {s: {r["id"]: r for r in previous.get(s, {}).get("rows", [])}
                for s in strategies}
    scored_ids = set.intersection(*(set(v) for v in restored.values())) if restored else set()
    pending = [p for p in points if p["id"] not in scored_ids]
    if args.batch_size:
        pending = pending[:args.batch_size]

    answerer = OpenAIProvider(model=args.model, api_key=args.api_key,
                              base_url=args.base_url, temperature=0.0)
    judge = OpenAIProvider(model=args.model, api_key=args.api_key,
                           base_url=args.base_url, temperature=0.0)

    print(f"Set: {data.get('name')}  |  {len(points)} points  |  budget {args.budget}  "
          f"|  samples {args.samples}")
    print(f"Strategies: {', '.join(strategies)}\n")
    if scored_ids:
        print(f"resuming: {len(scored_ids)} points already scored, "
              f"{len(pending)} in this batch\n")
    if not pending:
        print("nothing left to score — every point already has a result\n")

    order = {p["id"]: i for i, p in enumerate(points)}

    def write_out(current_rows: dict) -> None:
        """Write progress, keeping any strategies this run did not touch."""
        if not args.out:
            return
        payload = {s: v for s, v in previous.items() if s not in strategies}
        for name in strategies:
            rows_for = current_rows.get(name) or list(restored[name].values())
            if not rows_for:
                continue
            rows_for = sorted(rows_for, key=lambda r: order.get(r["id"], 0))
            payload[name] = {
                "summary": summarise(rows_for, args.samples),
                "rows": rows_for,
            }
        args.out.write_text(json.dumps({
            "set": data.get("name"), "budget": args.budget,
            "samples": args.samples, "results": payload,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    results = {}
    for strategy in strategies:
        rows = [restored[strategy][pid] for pid in scored_ids if pid in restored[strategy]]
        forge_cache: dict = {}
        for index, point in enumerate(pending):
            if args.rebuild_every and index and index % args.rebuild_every == 0:
                forge_cache.clear()
            row = await evaluate_point(
                point, base_strategy_for[strategy], args.budget, forge_cache,
                answerer, judge, args.api_key, args.base_url, args.model,
                args.samples,
                dropped_notice_for[strategy],
                external_for.get(strategy),
                args.external_budget_scale,
            )
            rows.append(row)
            write_out({**{s: results[s]["rows"] for s in results}, strategy: rows})
            if not args.quiet:
                spread = (f" ±{row['score_spread']:.0%}"
                          if row.get("score_spread") else "")
                print(f"  {strategy:9s} {point['id']:<16} {point['history_tokens']:>7,}tok "
                      f"ratio={row['compression_ratio']:.2f} score={row['score']:.0%}{spread}",
                      flush=True)
        # keep point order stable across resumed runs
        rows.sort(key=lambda r: order.get(r["id"], 0))
        results[strategy] = {"summary": summarise(rows, args.samples), "rows": rows}
        write_out({s: results[s]["rows"] for s in results})
        s = results[strategy]["summary"]
        inner = (f", 重跑波动 ±{s['rerun_swing']:.1%}"
                 if s.get("rerun_swing") is not None else "")
        print(f"  ==> {strategy}: 全部 {s['all']:.1%} (±{s['ci95']:.1%} 95%CI){inner} "
              f"| small {_fmt(s['small_3k_10k'])} | mid {_fmt(s['mid_10k_25k'])} "
              f"| big {_fmt(s['big_25k+'])}\n", flush=True)

    print("=== summary ===")
    print(f"{'strategy':<10}{'all':>8}{'±95%CI':>9}{'noise':>8}{'small':>8}{'mid':>8}{'big':>8}{'ratio':>8}")
    for strategy in strategies:
        s = results[strategy]["summary"]
        noise = (f"±{s['rerun_swing']:.0%}"
                 if s.get("rerun_swing") is not None else "-")
        print(f"{strategy:<10}{s['all']:>7.1%}{s['ci95']:>8.1%}{noise:>8}"
              f"{_fmt(s['small_3k_10k']):>8}{_fmt(s['mid_10k_25k']):>8}"
              f"{_fmt(s['big_25k+']):>8}{s['mean_ratio']:>8.2f}")
    print('\n95%CI = 对「这批点的均值」的置信区间（加点数会收窄）')
    print('noise = 同一配置重跑时均分的波动幅度（加点数不会收窄，是这只尺子的地板）')

    await answerer.aclose()
    await judge.aclose()

    if args.out:
        args.out.write_text(json.dumps({
            "set": data.get("name"), "budget": args.budget, "samples": args.samples,
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nFull results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
