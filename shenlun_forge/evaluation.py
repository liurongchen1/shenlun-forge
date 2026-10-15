"""Measure whether compression actually helps on *your* conversations.

The numbers in the README describe what happened on one particular dataset, and
the main lesson of building them was that the answer depends entirely on the
data: on real work conversations compression ties with plain truncation; on
long histories with the answer buried early it wins by 25 points. Nobody can
tell you which case you are in — you have to measure it.

That is what this module is for. It turns your own transcripts into evaluation
points, runs each strategy against them, and reports what it finds:

    from shenlun_forge import OpenAIProvider, build_eval_set, evaluate

    llm = OpenAIProvider(model="deepseek-chat", api_key="sk-...")
    points = await build_eval_set(my_conversations, llm=llm)
    report = await evaluate(points, strategies=["simple", "embedding"], llm=llm)
    print(report.summary())

Both steps call an LLM: extracting the answer key needs one, and scoring
answers needs another. Nothing is uploaded beyond the text you pass in — which
is worth remembering, because your transcripts are yours.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .base import LLMProvider


# Turns that carry no question worth testing.
_LOW_VALUE = {
    "继续", "好的", "嗯", "ok", "可以", "继续吧", "再来", "好", "是的", "对",
    "嗯嗯", "不用了", "go on", "continue", "okay", "yes", "thanks",
}


@dataclass
class EvalPoint:
    """One question, the history it should be answered from, and the answer key."""

    history: list
    query: str
    reference: str
    key_points: list = field(default_factory=list)


@dataclass
class EvalReport:
    """Accuracy per strategy, with the sample size behind it."""

    budget: int
    counts: dict = field(default_factory=dict)      # strategy -> points scored
    accuracy: dict = field(default_factory=dict)    # strategy -> 0..1
    per_point: dict = field(default_factory=dict)   # strategy -> [scores]

    def summary(self) -> str:
        if not self.accuracy:
            return "no strategies were evaluated"
        lines = [f"Accuracy by strategy (budget {self.budget} tokens, "
                 f"n={max(self.counts.values(), default=0)} points):"]
        for name, value in sorted(self.accuracy.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:<18}{value:>7.1%}")
        if len(self.accuracy) > 1:
            best = max(self.accuracy, key=self.accuracy.get)
            lines.append("")
            lines.append(f"Best: {best}. Treat differences under ~5 points as noise — "
                         "that is the metric's own re-run swing at this sample size; "
                         "add more conversations, not more turns, to sharpen it.")
        return "\n".join(lines)


def _conversations_to_message_lists(conversations) -> list:
    """Accept a list of message lists, or of {"messages": [...]} dicts."""
    out = []
    for convo in conversations:
        if isinstance(convo, dict):
            messages = convo.get("messages") or []
        else:
            messages = convo
        if isinstance(messages, list) and len(messages) >= 6:
            out.append(messages)
    return out


_KEY_POINT_PROMPT = """从下面的回答里提取 3-6 个关键要点，用于检验另一个回答是否覆盖了同样的信息。

要点应当是具体的事实、数字、结论、步骤或名称，不要写成"介绍了X"这种空泛概括。

回答：
{reference}

只输出 JSON：{{"points": ["...", "..."]}}"""

# References can be very long in agent transcripts; the answer key does not need
# all of it, and an oversized prompt is what a provider is most likely to reject.
_MAX_REFERENCE_CHARS = 4000


async def build_eval_set(
    conversations,
    llm: LLMProvider,
    min_query_chars: int = 15,
    min_reference_chars: int = 400,
    min_history_tokens: int = 3000,
    max_points: Optional[int] = None,
) -> list:
    """Turn conversations into evaluation points with frozen answer keys.

    Freezing matters: extracting key points at scoring time would grade every
    run against a different ruler, and that variance alone is enough to hide
    the differences you are looking for.

    Args:
        conversations: A list of message lists, or of ``{"messages": [...]}``
            dicts. Each message needs ``role`` and ``content``.
        llm: Provider used to extract the key points.
        min_query_chars / min_reference_chars: Skip turns too thin to test.
        min_history_tokens: Skip questions with almost no history to compress.
        max_points: Stop after this many points (0 or None = all).
    """
    from .types import count_tokens

    points = []
    for messages in _conversations_to_message_lists(conversations):
        history_tokens = 0
        for i, message in enumerate(messages):
            if message.get("role") != "user":
                history_tokens += count_tokens(message.get("content") or "")
                continue
            query = (message.get("content") or "").strip()
            reference = ((messages[i + 1].get("content") or "").strip()
                         if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant"
                         else "")
            if (len(query) >= min_query_chars and query.lower() not in _LOW_VALUE
                    and len(reference) >= min_reference_chars
                    and history_tokens >= min_history_tokens and i >= 6):
                points.append(EvalPoint(history=messages[:i], query=query, reference=reference))
            history_tokens += count_tokens(query)
            if max_points and len(points) >= max_points:
                break
        if max_points and len(points) >= max_points:
            break

    for point in points:
        # A provider can refuse a single reference (filtering, an upstream
        # hiccup). Skip that point rather than losing the whole run.
        try:
            verdict = await llm.complete_json(
                _KEY_POINT_PROMPT.format(reference=point.reference[:_MAX_REFERENCE_CHARS]), {})
        except Exception:
            point.key_points = []
            continue
        point.key_points = [str(p).strip() for p in verdict.get("points", [])
                            if str(p).strip()][:6]

    return [p for p in points if p.key_points]


_ANSWER_PROMPT = """以下是之前对话的上下文（可能已被压缩，部分内容缺失）：

{context}

---
只依据以上上下文回答用户的问题。上下文里没有相关信息时，直接说"上下文里没有相关信息"。

问题：{query}"""

_JUDGE_PROMPT = """问题：
{query}

需要覆盖的要点：
{points}

待评估回答：
{candidate}

判断待评估回答覆盖了哪些要点（语义一致即可，不要求措辞相同；含糊带过不算覆盖）。
只输出 JSON：{{"covered": [0, 2]}}"""


async def evaluate(
    points: list,
    strategies=("simple", "embedding"),
    budget: int = 16000,
    llm: Optional[LLMProvider] = None,
    samples: int = 1,
    progress=None,
) -> EvalReport:
    """Score strategies on "can a model still answer from the compressed text?".

    Args:
        points: Output of :func:`build_eval_set`.
        strategies: Strategy names, plus the literal ``"truncate"`` for the
            naive baseline. ``"embedding"`` uses lightweight mode (no LLM).
        budget: Token budget handed to each compression.
        llm: Provider used to answer and to judge. Required.
        samples: How many times to answer+judge each point, averaged. Raises
            precision at proportional cost.
        progress: Optional callback ``(strategy, index, total)``.

    Returns:
        An :class:`EvalReport`.
    """
    if llm is None:
        raise ValueError("evaluate() needs an LLM provider for answering and judging")

    from .compressor import ShenlunForge
    from .types import count_tokens

    report = EvalReport(budget=budget)
    total = len(points)

    for strategy in strategies:
        scores, forge = [], None
        for index, point in enumerate(points):
            if strategy == "truncate":
                kept, used = [], 0
                for message in reversed(point.history):
                    cost = count_tokens(message.get("content") or "")
                    if used + cost > budget:
                        break
                    kept.append(message)
                    used += cost
                kept.reverse()
                context = "\n".join(f"[{m.get('role')}] {m.get('content')}" for m in kept)
            else:
                if forge is None:
                    forge = (ShenlunForge(retriever_type="embedding")
                             if strategy == "embedding"
                             else ShenlunForge.create(strategy=strategy, llm=llm))
                result = await forge.compress(
                    messages=point.history, current_query=point.query,
                    max_tokens=budget, **({"lightweight": True} if strategy == "embedding" else {}),
                )
                context = result.compressed_text

            best = 0.0
            for _ in range(max(samples, 1)):
                try:
                    answer = await llm.complete(_ANSWER_PROMPT.format(
                        context=context, query=point.query))
                    verdict = await llm.complete_json(_JUDGE_PROMPT.format(
                        query=point.query,
                        points=json.dumps(point.key_points, ensure_ascii=False),
                        candidate=answer,
                    ), {})
                except Exception:
                    continue          # count this attempt as zero, keep the run
                covered = [c for c in verdict.get("covered", [])
                           if isinstance(c, int) and 0 <= c < len(point.key_points)]
                best += len(covered) / max(len(point.key_points), 1)
            scores.append(best / max(samples, 1))
            if progress:
                progress(strategy, index + 1, total)

        report.counts[strategy] = len(scores)
        report.accuracy[strategy] = sum(scores) / max(len(scores), 1)
        report.per_point[strategy] = scores

    return report
