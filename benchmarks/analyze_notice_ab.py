#!/usr/bin/env python3
"""Compare "simple" against "simple-nodrop" from a qa_eval run.

The closing note is not supposed to raise the score on a question-answering
benchmark — when the answer was dropped, the model cannot recover it by being
told it is missing. What it should change is *behaviour*: the model should stop
answering as if it had read the whole conversation, and say what it cannot know.
So this reports two things per strategy:

  score           — did the note cost anything (it must not)
  stated_missing  — how often the answer says the context does not contain it
  invented        — gave a concrete answer anyway, with no points covered

Usage:
    python benchmarks/analyze_notice_ab.py results.json [more.json ...]
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


# Phrases that mean "I checked and it is not in what you gave me". Deliberately
# narrow: "I am not sure" is not the same claim as "this is not in the context".
MISSING = re.compile(
    r"上下文(里|中)?(并)?(没|未)有|没有相关信息|未(被)?包含|不在(提供|以上|给定)的?上下文"
    r"|无法从(以上|提供的)?上下文|not (in|included in|present in) the (provided )?context"
    r"|does not contain|no information (about|on)|not mentioned in the context"
    r"|is not available in the (provided )?context",
    re.IGNORECASE,
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ci95(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return 1.96 * (variance ** 0.5) / (n ** 0.5)


# How much of the original history the summary still shows. The note matters
# most in the middle of this range: at a 0.05 ratio the model can see the
# context is a fragment, and at 1.0 nothing was dropped at all.
BANDS = (
    ("ratio<0.2", lambda r: r < 0.2),
    ("0.2-0.5", lambda r: 0.2 <= r < 0.5),
    ("0.5-0.8", lambda r: 0.5 <= r < 0.8),
    ("ratio>=0.8", lambda r: r >= 0.8),
)


def _stats(rows: list) -> dict:
    scores = [r["score"] for r in rows]
    stated = [bool(MISSING.search(r.get("answer") or "")) for r in rows]
    # A concrete answer that covered none of the key points, on a question whose
    # answer we know exists somewhere in the history.
    invented = [s == 0.0 and not m for s, m in zip(scores, stated)]
    return {
        "n": len(rows),
        "score": _mean(scores),
        "score_ci95": _ci95(scores),
        "stated_missing": _mean([1.0 if s else 0.0 for s in stated]),
        "invented": _mean([1.0 if i else 0.0 for i in invented]),
        "zero_scores": _mean([1.0 if s == 0.0 else 0.0 for s in scores]),
    }


def _conversation(row_id: str) -> str:
    """Points from one conversation are not independent samples of each other."""
    return row_id.split("#")[0]


def _two_sided_p(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2))


def paired(a_rows: list, b_rows: list) -> dict:
    """Compare a against b on the *same* points.

    Pairing is a large gain here: the two runs answer identical questions, so
    the question's own difficulty cancels out and only the difference is left.
    The clustered figure is the honest one — points from one conversation share
    a speaker and a topic, so a 131-point set drawn from 12 conversations
    carries about 12 independent observations, not 131.
    """
    by_id = {r["id"]: r for r in b_rows}
    ids = [r["id"] for r in a_rows if r["id"] in by_id]
    diffs = [r["score"] - by_id[r["id"]]["score"]
             for r in a_rows if r["id"] in by_id]
    n = len(diffs)
    if n < 2:
        return {"n": n}

    mean = _mean(diffs)
    sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5
    t = mean / (sd / math.sqrt(n)) if sd else 0.0

    groups: dict[str, list] = defaultdict(list)
    for row_id, diff in zip(ids, diffs):
        groups[_conversation(row_id)].append(diff)
    gmeans = [_mean(v) for v in groups.values()]
    k = len(gmeans)
    out = {
        "n": n, "conversations": k, "mean": mean,
        "t": t, "p": _two_sided_p(t),
        "wins": sum(1 for d in diffs if d > 0),
        "losses": sum(1 for d in diffs if d < 0),
    }
    if k >= 2:
        gm = _mean(gmeans)
        gsd = (sum((g - gm) ** 2 for g in gmeans) / (k - 1)) ** 0.5
        t_cluster = gm / (gsd / math.sqrt(k)) if gsd else 0.0
        out.update({"cluster_mean": gm, "cluster_t": t_cluster,
                    "cluster_p": _two_sided_p(t_cluster)})
    return out


def analyse(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data["results"]
    out = {"set": data.get("set"), "budget": data.get("budget"), "strategies": {}}
    for name, payload in results.items():
        rows = payload["rows"]
        stats = _stats(rows)
        stats["bands"] = {
            label: _stats([r for r in rows if test(r.get("compression_ratio") or 1.0)])
            for label, test in BANDS
        }
        out["strategies"][name] = stats
    out["rows"] = {name: payload["rows"] for name, payload in results.items()}
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    for arg in argv:
        report = analyse(Path(arg))
        print(f"\n=== {report['set']}  (budget {report['budget']}) ===")
        print(f"{'strategy':<18}{'n':>5}{'score':>9}{'±95%CI':>9}"
              f"{'stated_missing':>16}{'invented':>10}{'zero':>8}")
        for name, s in report["strategies"].items():
            print(f"{name:<18}{s['n']:>5}{s['score']:>8.1%}{s['score_ci95']:>9.1%}"
                  f"{s['stated_missing']:>15.1%}{s['invented']:>10.1%}"
                  f"{s['zero_scores']:>8.1%}")

        names = list(report["strategies"])
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                 if a.replace("-nodrop", "") == b.replace("-nodrop", "")]
        for a, b in pairs:
            sa, sb = report["strategies"][a], report["strategies"][b]
            print(f"\n  {a} vs {b}")
            print(f"    score           {sa['score'] - sb['score']:+.1%}")
            print(f"    stated_missing  {sa['stated_missing'] - sb['stated_missing']:+.1%}")
            print(f"    invented        {sa['invented'] - sb['invented']:+.1%}")
            print("    (score must not drop; stated_missing is the intended change)")
            print(f"\n    by how much of the history survived:")
            print(f"      {'band':<12}{'n':>5}{'score diff':>12}"
                  f"{'stated_missing diff':>21}")
            for label, _ in BANDS:
                ba, bb = sa["bands"][label], sb["bands"][label]
                if not ba["n"] or not bb["n"]:
                    continue
                print(f"      {label:<12}{ba['n']:>5}"
                      f"{ba['score'] - bb['score']:>12.1%}"
                      f"{ba['stated_missing'] - bb['stated_missing']:>21.1%}")

            p = paired(report["rows"][a], report["rows"][b])
            print(f"\n    paired on the same {p['n']} points "
                  f"({p['conversations']} conversations):")
            print(f"      mean difference {p['mean']:+.1%}  "
                  f"(wins {p['wins']}, losses {p['losses']})")
            print(f"      t={p['t']:.2f}, p={p['p']:.3f} (point level)")
            if "cluster_t" in p:
                print(f"      t={p['cluster_t']:.2f}, p={p['cluster_p']:.3f} "
                      f"(grouped by conversation — the one to quote)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
