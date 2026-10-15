#!/usr/bin/env python3
"""Look for a runtime-visible signal that predicts which retriever wins.

The first attempt (`router.py`) used "how spread out is semantic similarity
across the history" and failed: it correlates -0.09 with truncation winning.
The reason came from LongMemEval's ground truth — there the answer is buried at
a *random* position (mean 0.53, 18.8% in the final quarter), so no runtime
signal can predict where it is, and the right move is simply "always retrieve".

The signal worth testing is narrower: **is the most relevant message already in
the recent turns?** If it is, keeping the tail captures it and retrieval has
nothing to add. Unlike spread, this asks about the one message that matters.

Emits the raw similarities so the correlation can be computed offline against
results already on disk. Run in batches; it loads the embedding model.

    python benchmarks/signal_probe.py --set qa.json --offset 0 --samples 15 --out p0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shenlun_forge.embedding_retriever import EmbeddingRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, required=True, dest="set_path")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--recent-turns", type=int, default=6,
                        help="how many trailing messages count as 'the recent turns'")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np

    points = json.loads(args.set_path.read_text(encoding="utf-8"))["points"]
    retriever = EmbeddingRetriever()
    rows = []

    for point in points[args.offset:args.offset + args.samples]:
        history = point["history"]
        query_vec = retriever._embed_query(point["query"])
        block_vecs = retriever._embed_blocks([m.get("content", "") for m in history])
        sims = np.dot(block_vecs, query_vec)

        k = min(args.recent_turns, len(history))
        recent = sims[-k:]
        top_index = int(np.argmax(sims))
        rows.append({
            "id": point["id"],
            "history_len": len(history),
            "recent_max": float(recent.max()),
            "recent_mean": float(recent.mean()),
            "global_max": float(sims.max()),
            # 1.0 means the single most relevant message is in the recent turns
            "peak_in_recent": 1.0 if top_index >= len(history) - k else 0.0,
            "peak_position": top_index / max(len(history) - 1, 1),
            "peak_margin": float(sims.max() - recent.max()),
        })
        print(f"  {point['id']:<18} 峰值位置={rows[-1]['peak_position']:.2f} "
              f"在最近={bool(rows[-1]['peak_in_recent'])}", flush=True)

    args.out.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    print(f"\n写出 {len(rows)} 条 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
