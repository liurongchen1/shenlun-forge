#!/usr/bin/env python3
"""ShenlunForge CLI — command-line interface for context compression.

Usage:
    shenlun-forge compress [OPTIONS]
    shenlun-forge strategies
    shenlun-forge version

Examples:
    # Compress a JSON messages file (default strategy: simple, no API key)
    shenlun-forge compress --messages messages.json --query "What did we decide?" --max-tokens 4000

    # Compress from stdin
    echo '[{"role":"user","content":"Hello"}]' | shenlun-forge compress --query "Summary" --stdin

    # List available strategies
    shenlun-forge strategies

    # LLM-powered strategies need a key (acs = full pipeline, hcs = keep code verbatim)
    shenlun-forge compress --messages messages.json --strategy acs \
        --api-key "sk-..." --base-url "https://api.deepseek.com/v1"
    shenlun-forge compress --messages messages.json --strategy hcs --max-tokens 2000

    # Rule-based strategy, explicit (no API key needed)
    shenlun-forge compress --messages messages.json --strategy simple --max-tokens 1000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


STAR_HINT_MARKER = Path.home() / ".shenlun_forge_star_hint"


def maybe_show_star_hint() -> None:
    """Mention the star request once per machine, then never again.

    Written to a local marker file so it does not nag: a library that repeats a
    request on every run is worse than one that never asks. Failures here are
    ignored — a missing tip is not worth breaking a command over.
    """
    try:
        if STAR_HINT_MARKER.exists():
            return
        STAR_HINT_MARKER.write_text("shown\n", encoding="utf-8")
    except OSError:
        return
    print(
        "tip: 如果 shenlun-forge 帮到了你，给颗星是最简单的支持\n"
        "     a star is the simplest way to support it:\n"
        "     https://github.com/liurongchen1/shenlun-forge",
        file=sys.stderr,
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shenlun-forge",
        description="ShenlunForge — Task-driven context compression for LLM agents",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- compress ---
    compress_parser = subparsers.add_parser("compress", help="Compress conversation context")
    compress_parser.add_argument(
        "--messages", "-m",
        help="Path to JSON file containing messages array. Each item: {\"role\": \"...\", \"content\": \"...\"}",
    )
    compress_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read messages JSON from stdin instead of a file",
    )
    compress_parser.add_argument(
        "--query", "-q",
        default="",
        help="Current user query that drives compression (default: empty)",
    )
    compress_parser.add_argument(
        "--strategy", "-s",
        default="simple",
        help=(
            "Compression strategy: simple, acs, hcs (default: simple). "
            "'simple' needs no API key; 'acs' and 'hcs' require an LLM."
        ),
    )
    compress_parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=4000,
        help="Maximum tokens for compressed output (default: 4000)",
    )
    compress_parser.add_argument(
        "--model",
        default=os.getenv("SHENLUN_FORGE_MODEL", "deepseek-chat"),
        help="LLM model name (default: deepseek-chat, or SHENLUN_FORGE_MODEL env var)",
    )
    compress_parser.add_argument(
        "--api-key",
        default=os.getenv("SHENLUN_FORGE_API_KEY",
                          os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", ""))),
        help="API key (or set SHENLUN_FORGE_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY)",
    )
    compress_parser.add_argument(
        "--base-url",
        default=os.getenv("SHENLUN_FORGE_BASE_URL", os.getenv("OPENAI_BASE_URL", "")),
        help="LLM API base URL (or set SHENLUN_FORGE_BASE_URL / OPENAI_BASE_URL env var)",
    )
    compress_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include detailed step-by-step processing log in output",
    )
    compress_parser.add_argument(
        "--budget-overflow",
        choices=["keep", "truncate"],
        default="keep",
        help=(
            "When the newest messages do not fit --max-tokens: 'keep' "
            "(default) preserves them whole and may exceed the budget; "
            "'truncate' trims them to stay within it."
        ),
    )
    compress_parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)",
    )

    # --- strategies ---
    subparsers.add_parser("strategies", help="List available compression strategies")

    # --- eval ---
    eval_parser = subparsers.add_parser(
        "eval", help="Measure whether compression helps on your own conversations")
    eval_parser.add_argument(
        "--conversations", "-c", required=True,
        help="Directory of JSON files, each containing {\"messages\": [...]}",
    )
    eval_parser.add_argument(
        "--strategies", "-s", default="truncate,simple",
        help="Comma-separated; 'truncate' is the naive baseline (default: truncate,simple)",
    )
    eval_parser.add_argument("--budget", "-t", type=int, default=16000,
                             help="Token budget per compression (default: 16000)")
    eval_parser.add_argument("--max-points", type=int, default=40,
                             help="Cap how many questions to test (default: 40)")
    eval_parser.add_argument("--samples", type=int, default=1,
                             help="Answer+judge repeats per point, averaged (default: 1)")
    eval_parser.add_argument(
        "--model", default=os.getenv("SHENLUN_FORGE_MODEL", "deepseek-chat"))
    eval_parser.add_argument(
        "--api-key",
        default=os.getenv("SHENLUN_FORGE_API_KEY",
                          os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", ""))))
    eval_parser.add_argument(
        "--base-url",
        default=os.getenv("SHENLUN_FORGE_BASE_URL", os.getenv("OPENAI_BASE_URL", "")))
    eval_parser.add_argument("--json", help="Also write the report as JSON here")

    # --- version ---
    subparsers.add_parser("version", help="Show version information")

    return parser


def load_messages(args) -> list[dict[str, str]]:
    """Load messages from file or stdin."""
    if args.stdin:
        raw = sys.stdin.read()
    elif args.messages:
        path = Path(args.messages)
        if not path.exists():
            print(f"Error: File not found: {args.messages}", file=sys.stderr)
            sys.exit(1)
        raw = path.read_text(encoding="utf-8")
    else:
        print("Error: Either --messages <file> or --stdin is required.", file=sys.stderr)
        sys.exit(1)

    try:
        messages = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(messages, list):
        print("Error: Messages must be a JSON array.", file=sys.stderr)
        sys.exit(1)

    return messages


def resolve_base_url(explicit: str) -> str:
    """Pick an endpoint that matches the API key we found.

    A `DEEPSEEK_API_KEY` with the default OpenAI endpoint returns a confusing
    401, so infer the matching base URL when the caller did not give one.
    """
    if explicit:
        return explicit
    if os.getenv("DEEPSEEK_BASE_URL"):
        return os.getenv("DEEPSEEK_BASE_URL")
    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        return "https://api.deepseek.com/v1"
    return ""


async def run_compress(args):
    """Execute the compress command."""
    from .compressor import ShenlunForge

    messages = load_messages(args)

    if args.strategy == "simple":
        from .strategies.simple import SimpleStrategy
        compressor = SimpleStrategy()
    else:
        compressor = ShenlunForge.create(
            strategy=args.strategy,
            model=args.model,
            api_key=args.api_key,
            base_url=resolve_base_url(args.base_url) or None,
        )

    result = await compressor.compress(
        messages=messages,
        current_query=args.query,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        budget_overflow=args.budget_overflow,
    )

    # Never let a degraded or no-op run look like a success.
    if result.stats.get("degraded"):
        reason = result.stats.get("degraded_reason") or "compression ran in degraded mode"
        print(f"Warning: {reason}", file=sys.stderr)
    if result.stats.get("over_budget"):
        print(
            f"Note: output is {result.compressed_tokens} tokens, over the "
            f"{args.max_tokens} budget, because --budget-overflow=keep preserved "
            "the newest messages whole. Use --budget-overflow=truncate to "
            "enforce the budget.",
            file=sys.stderr,
        )
    if result.stats.get("recent_truncated"):
        print(
            "Note: the newest message was truncated to stay within the budget.",
            file=sys.stderr,
        )
    if result.stats.get("passthrough"):
        print(
            "Note: compression would not have saved tokens, so the original "
            "context was returned unchanged.",
            file=sys.stderr,
        )

    output = {
        "compressed_text": result.compressed_text,
        "original_tokens": result.original_tokens,
        "compressed_tokens": result.compressed_tokens,
        "compression_ratio": round(result.compression_ratio, 4),
        "stats": result.stats,
    }
    if args.verbose:
        output["steps_log"] = result.steps_log

    output_json = json.dumps(output, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


def cmd_strategies(args):
    """List available strategies."""
    from .registry import list_strategies

    strategies = list_strategies()
    print("Available compression strategies:\n")
    descriptions = {
        "acs": "Full 4-step ACS pipeline (Analyze→Retrieve→Process→Format). Requires an LLM.",
        "simple": "Rule-based selection. No LLM, no API key, milliseconds. [default]",
        "hcs": "ACS semantic summary + verbatim code blocks. Requires an LLM.",
    }
    for name in strategies:
        desc = descriptions.get(name, "")
        print(f"  {name:12s} {desc}")


def cmd_version(args):
    """Show version."""
    from . import __version__
    print(f"shenlun-forge {__version__}")


async def run_eval(args):
    """Measure compression on the user's own conversations."""
    import json as _json
    from .evaluation import build_eval_set, evaluate
    from .llm import OpenAIProvider

    if not args.api_key:
        print("Error: eval needs an API key (extracting answer keys and judging "
              "answers both call a model).", file=sys.stderr)
        sys.exit(1)

    directory = Path(args.conversations)
    if not directory.is_dir():
        print(f"Error: not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    conversations = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            conversations.append(_json.loads(path.read_text(encoding="utf-8")))
        except _json.JSONDecodeError:
            continue
    if not conversations:
        print(f"Error: no readable conversations in {directory}", file=sys.stderr)
        sys.exit(1)

    llm = OpenAIProvider(model=args.model, api_key=args.api_key,
                         base_url=resolve_base_url(args.base_url) or None, temperature=0.0)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    print(f"从 {len(conversations)} 个对话构建测试（提取答案要点需要模型调用）...",
          file=sys.stderr)
    points = await build_eval_set(conversations, llm=llm, max_points=args.max_points)
    if not points:
        print("Error: no usable evaluation points found — check that the "
              "conversations contain substantial question/answer turns.",
              file=sys.stderr)
        sys.exit(1)
    print(f"得到 {len(points)} 个测试点，开始评测...\n", file=sys.stderr)

    report = await evaluate(points, strategies=strategies, budget=args.budget,
                            llm=llm, samples=args.samples,
                            progress=lambda s, i, n: print(f"  {s} {i}/{n}", file=sys.stderr))
    await llm.aclose()

    print()
    print(report.summary())
    if args.json:
        Path(args.json).write_text(_json.dumps({
            "budget": report.budget, "counts": report.counts,
            "accuracy": report.accuracy, "per_point": report.per_point,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入 {args.json}", file=sys.stderr)


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.command in ("compress", "eval"):
        maybe_show_star_hint()

    if args.command == "compress":
        asyncio.run(run_compress(args))
    elif args.command == "eval":
        asyncio.run(run_eval(args))
    elif args.command == "strategies":
        cmd_strategies(args)
    elif args.command == "version":
        cmd_version(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
