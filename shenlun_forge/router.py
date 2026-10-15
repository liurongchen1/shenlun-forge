"""Pick a retriever for the query at hand, instead of always using one.

**Experimental, and currently not recommended — it failed validation.**

Measured on 67 points from real Chinese coding sessions (via
`benchmarks/route_eval.py`), this router picked the better of the two
retrievers 41% of the time — barely above chance — and routed **64 of 67**
queries to embeddings in a dataset where plain truncation was actually the
better choice (51.0% vs 48.4%). Choosing by its advice scored 48.5%, *worse*
than always truncating.

The signal is the problem, not the plumbing: `recent_share` correlates -0.09
with "truncation would have won". A conversation stays on topic, so the query
resembles many messages throughout the history, which makes similarity look
"spread out" even when the answer sits in the last few turns. Similarity spread
is not a proxy for answer position.

The code is kept because the measurement harness is reusable and someone may
find a better signal; `benchmarks/README.md` records the attempt. Do not wire it
into anything by default.

Six evaluations across two datasets and two languages produced one rule that
held everywhere: **the value of retrieval is a function of how far the answer
sits from the end of the history.**

- Answer near the end (a normal work conversation: you ask about what you were
  just doing) — keeping the tail is already optimal, and keyword matching or
  embeddings add nothing. Both private datasets measured this as a tie.
- Answer buried early (information scattered across dozens of sessions) —
  semantic retrieval is worth 14-25 points over truncation, and it is the only
  difference in this project that survived a significance test.

So the useful question is not "which retriever is better" but "which one does
*this* query need". That is what this module answers, from the distribution of
semantic similarity between the query and the history — no LLM call, one pass of
embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Share of the most similar messages that have to sit in the recent tail before
# keeping the tail is judged sufficient.
RECENT_SHARE_THRESHOLD = 0.6

# How far back counts as "recent": the last quarter of the history.
RECENT_FRACTION = 0.25


@dataclass
class RetrieverChoice:
    """Which retriever to use for one query, and why."""

    retriever: str          # "truncate" | "embedding"
    confidence: float       # 0.0-1.0
    recent_share: float     # share of top matches sitting in the recent tail
    reason: str


def recommend_retriever(
    messages: list,
    query: str,
    top_k: int = 20,
    retriever=None,
) -> RetrieverChoice:
    """Decide whether this query needs a retriever or just the recent tail.

    Args:
        messages: The conversation history, oldest first.
        query: The question about to be asked.
        top_k: How many messages to consider as "the relevant ones".
        retriever: An :class:`~shenlun_forge.embedding_retriever.EmbeddingRetriever`
            to reuse (it caches embeddings, so passing one is much cheaper).

    Returns:
        A :class:`RetrieverChoice`. Falls back to ``"truncate"`` with low
        confidence when embeddings are unavailable, since that path always
        works and costs nothing.
    """
    if not messages or not query:
        return RetrieverChoice("truncate", 0.0, 1.0, "no query to match against")

    try:
        import numpy as np
        from .embedding_retriever import EmbeddingRetriever
    except ImportError:
        return RetrieverChoice(
            "truncate", 0.0, 1.0,
            "embeddings unavailable (pip install \"shenlun-forge[embedding]\"); "
            "defaulting to the free path",
        )

    helper = retriever or EmbeddingRetriever()
    try:
        query_vec = helper._embed_query(query)
        block_vecs = helper._embed_blocks([m.get("content", "") for m in messages])
    except Exception as exc:  # model download failed, no network, ...
        return RetrieverChoice(
            "truncate", 0.0, 1.0, f"embedding lookup failed ({type(exc).__name__}); using the free path",
        )

    similarities = np.dot(block_vecs, query_vec)
    k = min(top_k, len(messages))
    top_indices = np.argsort(similarities)[::-1][:k]

    cutoff = len(messages) * (1.0 - RECENT_FRACTION)
    recent_hits = int(sum(1 for i in top_indices if i >= cutoff))
    recent_share = recent_hits / max(k, 1)

    if recent_share >= RECENT_SHARE_THRESHOLD:
        confidence = min(1.0, (recent_share - RECENT_SHARE_THRESHOLD) / (1 - RECENT_SHARE_THRESHOLD) * 0.5 + 0.5)
        return RetrieverChoice(
            "truncate", round(confidence, 2), round(recent_share, 2),
            f"{recent_share:.0%} of the best-matching messages sit in the recent tail; "
            "keeping the tail is enough and free",
        )

    confidence = min(1.0, (RECENT_SHARE_THRESHOLD - recent_share) / RECENT_SHARE_THRESHOLD * 0.5 + 0.5)
    return RetrieverChoice(
        "embedding", round(confidence, 2), round(recent_share, 2),
        f"only {recent_share:.0%} of the best-matching messages are recent; the "
        "relevant material is spread out, which is where retrieval pays",
    )
