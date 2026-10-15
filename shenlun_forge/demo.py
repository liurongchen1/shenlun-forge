"""A self-contained demonstration: one conversation, two questions, no setup.

The point of the demo is not "compression makes the text shorter" — plain
truncation does that too, and it is free. The point is *which* messages
survive: the ones the current question needs, wherever they sit in the
history. So the demo keeps everything else constant and shows the two
approaches side by side on the same conversation.

Runs in milliseconds, needs no API key, and makes no network calls.
"""

from __future__ import annotations

import textwrap

from .strategies.simple import SimpleStrategy
from .types import count_tokens


# Three unrelated topics in one conversation, which is how real histories look:
# the answer to a question asked at the end is usually somewhere in the middle,
# surrounded by discussions that have nothing to do with it.
DEMO_HISTORY: list[dict[str, str]] = [
    {"role": "user", "content": "We're starting work on the order service. It replaces the nightly CSV job that emails the finance team."},
    {"role": "assistant", "content": "Where does that data come from today?"},
    {"role": "user", "content": "Postgres, table orders. About 2M rows, growing by 30k a day."},
    {"role": "assistant", "content": "At that rate a read replica buys you a year, not forever. Monthly partitions on created_at would keep the queries flat."},
    {"role": "user", "content": "Let's do monthly partitions. Separately: should we use Redis for the menu cache?"},
    {"role": "assistant", "content": "You could, but the menu is under 40KB and changes twice a week. An in-process LRU with a 10-minute TTL does the same job and adds no operational dependency."},
    {"role": "user", "content": "Agreed, no Redis then. Nobody on the team has run Redis in production and we are not learning it during a migration."},
    {"role": "assistant", "content": "Reasonable. If the cache ever needs to be shared between machines, revisit it — not before."},
    {"role": "user", "content": "Now the refunds flow. Finance wants a manual approval step above 500 EUR."},
    {"role": "assistant", "content": "Then refunds need a state machine: requested, approved, paid. Keep the approver id on the row for the audit trail."},
    {"role": "user", "content": "Fine. Also we need to backfill 2019-2023 into the new schema before the cutover."},
    {"role": "assistant", "content": "Backfill in batches of 50k with a checkpoint table. A single transaction over four years of data will lock the table for hours."},
    {"role": "user", "content": "On staging the backfill ran 40 minutes for one month, so that plan sounds right."},
    {"role": "assistant", "content": "Extrapolating, that is about 8 hours of wall clock for the full range. Run it against the replica and cut over when it catches up."},
    {"role": "user", "content": "Alerting: we want a Slack ping if the lag goes over 5 minutes."},
    {"role": "assistant", "content": "Use the existing alerting stack rather than a new integration. A lag alert that nobody owns will be muted within a week."},
    {"role": "user", "content": "Permissions next: support agents should see orders but not refunds above 500 EUR."},
    {"role": "assistant", "content": "Then the approval step is also a permission boundary. Check the role on the endpoint, not in the template."},
    {"role": "user", "content": "Compliance also wants a year of history on who changed what."},
    {"role": "assistant", "content": "An audit table with actor, action, entity id and timestamp. Do not reuse the application logs for this — they roll over in two weeks."},
    {"role": "user", "content": "Migration plan is three phases: dual write, backfill, cutover."},
    {"role": "assistant", "content": "Dual write means the legacy job keeps running until the backfill finishes, otherwise finance sees a gap in the numbers."},
    {"role": "user", "content": "Fair. Different problem: production returns 500s on long conversations, but only when the thread is old."},
    {"role": "assistant", "content": "That is the shape of a request body exceeding the model's window. Are you trimming the history anywhere before sending it?"},
    {"role": "user", "content": "No, we send everything. Adding a trim felt like a behaviour change to sneak into the same sprint."},
    {"role": "assistant", "content": "It will keep getting worse as threads grow. Bound the history at the API boundary and leave the trimming to one place."},
    {"role": "user", "content": "Did that: cap the request at 4000 tokens and alert when the trimmed share goes over 90%."},
    {"role": "assistant", "content": "Good — the alert is the part that tells you when the cap starts costing you answers."},
    {"role": "user", "content": "Naming for the new service: ledger-svc or order-svc?"},
    {"role": "assistant", "content": "ledger-svc. It owns the refunds table too, so 'order' undersells what it does."},
    {"role": "user", "content": "ledger-svc it is. Deployment target is Fly.io, region sin, two machines."},
    {"role": "assistant", "content": "Two machines in sin is plenty for 30k writes a day. Make the health check hit the database, not just the process."},
]

# Asked at the end of the conversation. Both answers live in the middle.
DEMO_QUESTIONS = (
    "Why did we decide not to use Redis?",
    "What did we do about the 500s on long conversations?",
)

# Words that identify the answer in the history, used only to mark it in the
# output. Not part of the compression — it is how we show whether it survived.
DEMO_ANSWER_MARKERS = (
    ("Redis",),
    ("4000 tokens", "500s"),
)


TEXT = {
    "en": {
        "title": "ShenlunForge demo — no API key, no setup, no network",
        "intro": (
            "One conversation, {messages} messages, {tokens:,} tokens, three unrelated\n"
            "topics. Both questions are asked at the end, and neither answer is near\n"
            "the end. What matters is *which* messages survive."
        ),
        "budget": "budget",
        "history": "history",
        "history_line": "{count} messages",
        "budget_line": "{count} tokens",
        "question": "Q",
        "truncate": "truncate",
        "truncate_note": "keep the newest messages that fit — the baseline to beat",
        "simple": "simple",
        "simple_note": "shenlun-forge, rules only",
        "kept": "kept {kept} of {total} messages · {tokens:,} tokens · messages [{indices}]",
        "dropped_answer": "the answer is not here — it was dropped with the {topic} part",
        "kept_answer": "and the message that answers it survived:",
        "closing_note": "and the summary says what it left out:",
        "different": (
            "Same history, different question — a different set of messages survived.\n"
            "Plain truncation returns the same tail no matter what you asked."
        ),
        "next": (
            "Next:\n"
            "  shenlun-forge compress --messages your.json --query \"your question\"\n"
            "  shenlun-forge eval --conversations ./your-chats   # measure it on your data"
        ),
        "topics": ("Redis", "500s/4000-token cap"),
    },
    "zh": {
        "title": "ShenlunForge 演示 —— 不需要 key、不需要配置、不联网",
        "intro": (
            "一段对话，{messages} 条消息、{tokens:,} token，里面有三个互不相关的话题。\n"
            "两个问题都问在最后，而答案都不在末尾。重点不是「变短了」，而是**活下来\n"
            "的是哪几条**。"
        ),
        "budget": "预算",
        "history": "原始历史",
        "history_line": "{count} 条消息",
        "budget_line": "{count} token",
        "question": "问",
        "truncate": "粗暴截断",
        "truncate_note": "保留能装下的最近几条 —— 要打败的基线",
        "simple": "simple",
        "simple_note": "shenlun-forge，纯规则、不需要 key",
        "kept": "保留 {kept}/{total} 条 · {tokens:,} token · 消息 [{indices}]",
        "dropped_answer": "答案不在这里 —— 它和「{topic}」那段一起被丢了",
        "kept_answer": "而能回答问题的消息活了下来：",
        "closing_note": "而且摘要会自己交代漏了什么：",
        "different": (
            "同一段历史，换一个问题 —— 活下来的消息跟着变了。\n"
            "粗暴截断不管你问什么，返回的都是同一条尾巴。"
        ),
        "next": (
            "接下来：\n"
            "  shenlun-forge compress --messages your.json --query \"你的问题\"\n"
            "  shenlun-forge eval --conversations ./your-chats   # 用你自己的数据测一测"
        ),
        "topics": ("Redis", "500/4000 token 上限"),
    },
}


def _preview(text: str, limit: int = 58) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _truncate_history(messages: list[dict[str, str]], budget: int) -> list[int]:
    """Keep the newest messages that fit. Returns their indices."""
    kept, used = [], 0
    for index in range(len(messages) - 1, -1, -1):
        cost = count_tokens(messages[index].get("content") or "")
        if used + cost > budget:
            break
        kept.append(index)
        used += cost
    return sorted(kept)


def _marks_answer(messages: list[dict[str, str]], indices: list[int], markers: tuple) -> bool:
    """True when a kept message actually carries the answer."""
    for index in indices:
        content = messages[index].get("content") or ""
        if any(marker.lower() in content.lower() for marker in markers):
            return True
    return False


def _index_label(indices: list[int]) -> str:
    """Collapse a run of consecutive indices: [18,19,20,21] -> "18..21"."""
    if not indices:
        return "-"
    parts, start, previous = [], indices[0], indices[0]
    for index in indices[1:] + [None]:
        if index == (previous + 1 if previous is not None else None):
            previous = index
            continue
        parts.append(str(start) if start == previous else f"{start}..{previous}")
        start = previous = index
    return ", ".join(parts)


def _answer_lines(
    messages: list[dict[str, str]], indices: list[int], markers: tuple
) -> list[str]:
    """Show the surviving messages that actually carry the answer."""
    lines = []
    for index in indices:
        content = messages[index].get("content") or ""
        if any(marker.lower() in content.lower() for marker in markers):
            lines.append(
                f"          [{index:>2}] {messages[index]['role']:<9} {_preview(content, 62)}"
            )
    return lines


def run_demo(lang: str = "en", budget: int = 300) -> str:
    """Return the whole demo as one string, ready to print."""
    text = TEXT.get(lang, TEXT["en"])
    messages = DEMO_HISTORY
    total_tokens = sum(count_tokens(m["content"]) for m in messages)

    out: list[str] = []
    out.append(text["title"])
    out.append("=" * 72)
    out.append("")
    out.append(text["intro"].format(messages=len(messages), tokens=total_tokens))
    out.append("")
    out.append(
        f"  {text['history']:>10}  "
        + text["history_line"].format(count=len(messages))
    )
    out.append(
        f"  {text['budget']:>10}  " + text["budget_line"].format(count=budget)
    )
    out.append("")

    strategy = SimpleStrategy()
    results = []
    for question, markers in zip(DEMO_QUESTIONS, DEMO_ANSWER_MARKERS):
        results.append(
            strategy.compress_sync(
                messages=messages, current_query=question, max_tokens=budget
            )
        )

    for position, (question, markers, result) in enumerate(
        zip(DEMO_QUESTIONS, DEMO_ANSWER_MARKERS, results)
    ):
        kept_from_blocks = sorted({
            i for block in result.blocks for i in block.original_indices
        })
        truncated = _truncate_history(messages, budget)

        out.append("─" * 72)
        out.append(f"{text['question']}  {question}")
        out.append("─" * 72)

        approaches = (
            (
                text["truncate"],
                text["truncate_note"],
                truncated,
                sum(count_tokens(messages[i]["content"]) for i in truncated),
            ),
            (
                text["simple"],
                text["simple_note"],
                kept_from_blocks,
                result.compressed_tokens,
            ),
        )

        # The baseline first, so the comparison reads as "same history, and here
        # is what the cheap approach keeps".
        for position_in_list, (name, note, kept, tokens) in enumerate(approaches):
            out.append(f"\n  {name:<10} {note}")
            out.append("    " + text["kept"].format(
                kept=len(kept), total=len(messages), tokens=tokens,
                indices=_index_label(kept),
            ))
            if _marks_answer(messages, kept, markers):
                out.append(f"    ✓ {text['kept_answer']}")
                out.extend(_answer_lines(messages, kept, markers))
            else:
                out.append(
                    f"    ✗ {text['dropped_answer'].format(topic=text['topics'][position])}"
                )

        if position == 0:
            notice = result.compressed_text.split("\n\n")
            closing = next(
                (part for part in notice if part.startswith("[" ) or part.startswith("【")),
                "",
            )
            if closing:
                out.append(f"\n  {text['closing_note']}")
                for line in textwrap.wrap(closing.strip(), width=66):
                    out.append(f"    {line}")
        else:
            out.append("")
            out.append("  " + text["different"].replace("\n", "\n  "))

        out.append("")

    out.append("─" * 72)
    out.append(text["next"])
    return "\n".join(out)
