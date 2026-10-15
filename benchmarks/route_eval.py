#!/usr/bin/env python3
"""Check whether the retriever recommendation picks the right tool per query.

`recommend_retriever` claims a query that only needs the recent tail should skip
retrieval. This measures that claim against results already on disk: for each
point it asks for a recommendation, then compares the score that choice would
have produced with always-truncate, always-embedding, and the per-point best.

Only embeddings are computed — no LLM calls — so it is cheap to rerun. It does
load the embedding model, so run it in batches on long histories:

    python benchmarks/route_eval.py --set qa.json \\
        --truncate-results t.json --embedding-results e.json \\
        --offset 0 --samples 10 --out part0.json

Repeat with a rising --offset; merge the parts with --merge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shenlun_forge import recommend_retriever
from shenlun_forge.embedding_retriever import EmbeddingRetriever


def load_scores(path: Path, strategy: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["id"]: r["score"] for r in data["results"][strategy]["rows"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, required=True, dest="set_path")
    parser.add_argument("--truncate-results", type=Path, required=True)
    parser.add_argument("--embedding-results", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--merge", type=Path, nargs="*", default=None,
                        help="merge these part files and print the comparison")
    args = parser.parse_args()

    if args.merge:
        rows = []
        for part in args.merge:
            rows.extend(json.loads(part.read_text(encoding="utf-8"))["rows"])
        return report(rows, label=str(args.merge[0]))

    points = json.loads(args.set_path.read_text(encoding="utf-8"))["points"]
    truncate = load_scores(args.truncate_results, "truncate")
    embedding = load_scores(args.embedding_results, "embedding")
    retriever = EmbeddingRetriever()

    rows = []
    for point in points[args.offset:args.offset + args.samples]:
        choice = recommend_retriever(point["history"], point["query"], retriever=retriever)
        pid = point["id"]
        if pid not in truncate or pid not in embedding:
            continue
        rows.append({
            "id": pid,
            "choice": choice.retriever,
            "confidence": choice.confidence,
            "recent_share": choice.recent_share,
            "truncate": truncate[pid],
            "embedding": embedding[pid],
            "history_tokens": point.get("history_tokens", 0),
        })
        print(f"  {pid:<18} 推荐={choice.retriever:<10} recent_share={choice.recent_share:.2f} "
              f"截断={truncate[pid]:.0%} 语义={embedding[pid]:.0%}", flush=True)

    args.out.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    print(f"\n写出 {len(rows)} 条 -> {args.out}")
    return report(rows, label=args.set_path.stem)


def report(rows: list, label: str) -> int:
    if not rows:
        print("没有可评估的点")
        return 1
    n = len(rows)
    picked = [r["embedding"] if r["choice"] == "embedding" else r["truncate"] for r in rows]
    always_t = [r["truncate"] for r in rows]
    always_e = [r["embedding"] for r in rows]
    oracle = [max(r["truncate"], r["embedding"]) for r in rows]
    n_emb = sum(1 for r in rows if r["choice"] == "embedding")

    print(f"\n=== {label}（{n} 点）===")
    print(f"  推荐分布: truncate {n - n_emb} / embedding {n_emb}")
    print(f"  总是截断    {sum(always_t)/n:>7.1%}")
    print(f"  总是语义    {sum(always_e)/n:>7.1%}")
    print(f"  按推荐选择  {sum(picked)/n:>7.1%}   ← 本方案")
    print(f"  每点最优    {sum(oracle)/n:>7.1%}   (理论上界)")

    # how often did the recommendation land on the better of the two?
    better = [r for r in rows if r["truncate"] != r["embedding"]]
    if better:
        right = sum(1 for r in better
                    if (r["embedding"] > r["truncate"]) == (r["choice"] == "embedding"))
        print(f"  在有差异的 {len(better)} 个点上，推荐选中更优策略 {right} 次 "
              f"({right/len(better):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
