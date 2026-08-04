#!/usr/bin/env python3
"""ShenlunForge CLI — command-line interface for context compression.

Usage:
    shenlun-forge compress [OPTIONS]
    shenlun-forge strategies
    shenlun-forge version

Examples:
    # Compress a JSON messages file
    shenlun-forge compress --messages messages.json --query "What did we decide?" --max-tokens 4000

    # Compress from stdin
    echo '[{"role":"user","content":"Hello"}]' | shenlun-forge compress --query "Summary" --stdin

    # List available strategies
    shenlun-forge strategies

    # Use a specific strategy
    shenlun-forge compress --messages messages.json --strategy hcs --max-tokens 2000

    # Use SimpleStrategy (no API key needed)
    shenlun-forge compress --messages messages.json --strategy simple --max-tokens 1000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


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
        default="acs",
        help="Compression strategy: acs, simple, hcs (default: acs)",
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
        default=os.getenv("SHENLUN_FORGE_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        help="API key (or set SHENLUN_FORGE_API_KEY / OPENAI_API_KEY env var)",
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
        "--output", "-o",
        help="Output file path (default: stdout)",
    )

    # --- strategies ---
    subparsers.add_parser("strategies", help="List available compression strategies")

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
            base_url=args.base_url or None,
        )

    result = await compressor.compress(
        messages=messages,
        current_query=args.query,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
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
        "acs": "Full 4-step ACS pipeline (Analyze→Retrieve→Process→Format). Requires LLM.",
        "simple": "Rule-based compression. No LLM needed. Fast and lightweight.",
        "hcs": "ACS semantic summary + verbatim code augmentation. Requires LLM.",
    }
    for name in strategies:
        desc = descriptions.get(name, "")
        print(f"  {name:12s} {desc}")


def cmd_version(args):
    """Show version."""
    from . import __version__
    print(f"shenlun-forge {__version__}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.command == "compress":
        asyncio.run(run_compress(args))
    elif args.command == "strategies":
        cmd_strategies(args)
    elif args.command == "version":
        cmd_version(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
