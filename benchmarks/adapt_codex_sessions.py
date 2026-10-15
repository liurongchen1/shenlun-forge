#!/usr/bin/env python3
"""Turn Codex session history into conversation files for `build_qa_set.py`.

Codex records every event of a session in
``$CODEX_HOME/thread_history_1.sqlite``: user messages, agent messages,
reasoning, command executions, file changes — and a ``contextCompaction``
marker each time the context was compacted.

Two properties make this a useful evaluation source:

- **The record is complete.** Compaction is recorded as a marker with no
  payload, so the surrounding messages are untouched; the full conversation can
  be rebuilt from the items.
- **It is real.** Long, multi-topic, in whatever language the user actually
  works in — the opposite of a synthetic benchmark.

    python benchmarks/adapt_codex_sessions.py --out ./codex-conversations
    python benchmarks/build_qa_set.py --conversations ./codex-conversations --out qa.json

Privacy: the output is your own conversations. Keep it out of version control.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shenlun_forge.types import count_tokens


# Real credentials, replaced before anything is written or sent to an API.
# Passwords and addresses are left alone: they are ordinary conversation
# content, and redacting them would change what the evaluation measures.
REDACTIONS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "<REDACTED-KEY>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
     "<REDACTED-PRIVATE-KEY>"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9\-_.]{20,}"), r"\1 <REDACTED>"),
]


def redact(text: str) -> str:
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def message_text(item: dict) -> Optional[str]:
    """Extract plain conversation text, or None for non-conversation items."""
    kind = item.get("type")
    if kind == "userMessage":
        parts = [c.get("text", "") for c in item.get("content", []) if isinstance(c, dict)]
        return "".join(parts).strip()
    if kind == "agentMessage":
        return (item.get("text") or "").strip()
    return None


TOOL_OUTPUT_LIMIT = 2000


def tool_text(item: dict) -> Optional[str]:
    """Render a tool/command item as text — the code-heavy part of a session.

    This is what `hcs` exists to preserve, so an evaluation that filters these
    out cannot say anything about it.
    """
    kind = item.get("type")
    if kind == "commandExecution":
        command = item.get("command") or ""
        output = item.get("aggregatedOutput") or ""
        body = f"$ {command}"
        if output:
            body += f"\n{output[:TOOL_OUTPUT_LIMIT]}"
        return body.strip()
    if kind == "fileChange":
        parts = []
        for change in item.get("changes") or []:
            path = change.get("path", "")
            diff = (change.get("diff") or "")[:TOOL_OUTPUT_LIMIT]
            parts.append(f"[file] {path}\n{diff}".strip())
        return "\n".join(parts).strip()
    if kind == "mcpToolCall":
        result = item.get("result") or {}
        texts = [c.get("text", "") for c in (result.get("content") or []) if isinstance(c, dict)]
        joined = "".join(texts)[:TOOL_OUTPUT_LIMIT]
        return f"[tool:{item.get('tool')}] {joined}".strip()
    return None


def load_threads(db_path: Path, min_user_turns: int, do_redact: bool = True,
                 include_tools: bool = False):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    thread_ids = [r[0] for r in con.execute(
        "select thread_id, count(*) c from thread_items group by thread_id order by c desc")]

    for thread_id in thread_ids:
        messages = []
        for (raw,) in con.execute(
            "select item_json from thread_items where thread_id=? order by rollout_ordinal",
            (thread_id,),
        ):
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = message_text(item)
            if not text:
                if include_tools:
                    text = tool_text(item)
            if not text:
                continue
            if do_redact:
                text = redact(text)
            messages.append({
                "role": {"userMessage": "user", "agentMessage": "assistant"}.get(
                    item["type"], "tool"),
                "content": text,
            })
        user_turns = sum(1 for m in messages if m["role"] == "user")
        if user_turns >= min_user_turns:
            yield thread_id, messages
    con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path,
                        default=Path.home() / ".codex" / "thread_history_1.sqlite")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-user-turns", type=int, default=20,
                        help="skip conversations with fewer user turns than this")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--no-redact", action="store_true",
                        help="keep credential-looking strings (default: redact them)")
    parser.add_argument("--include-tools", action="store_true",
                        help="also include command output, file diffs and tool "
                             "results as 'tool' messages. Off by default: it makes "
                             "transcripts much longer, and it is what you want if "
                             "you are evaluating code-retention (hcs).")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"session database not found: {args.db}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    print(f"{'对话':<22}{'轮次':>6}{'消息':>7}{'token':>10}")
    for thread_id, messages in load_threads(args.db, args.min_user_turns,
                                             do_redact=not args.no_redact,
                                             include_tools=args.include_tools):
        tokens = sum(count_tokens(m["content"]) for m in messages)
        user_turns = sum(1 for m in messages if m["role"] == "user")
        (args.out / f"{thread_id}.json").write_text(
            json.dumps({"messages": messages}, ensure_ascii=False), encoding="utf-8")
        written += 1
        print(f"{thread_id[:20]:<22}{user_turns:>6}{len(messages):>7}{tokens:>10,}")
        if args.limit and written >= args.limit:
            break

    print(f"\n写出 {written} 个对话 -> {args.out}")
    print("提醒：这是你自己的对话内容，不要提交到版本库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
