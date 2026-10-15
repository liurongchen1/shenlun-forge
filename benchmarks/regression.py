#!/usr/bin/env python3
"""Run the question-answering evaluation and compare it against a saved baseline.

The measured advantage of the retriever over naive truncation is the most
fragile thing in this project: it holds only when the answer sits early in a
long history, and it is easy to lose accidentally while changing something
unrelated. This script makes that loss visible.

    # first run: record what you have
    python benchmarks/regression.py --set qa-set.json --save-baseline

    # afterwards: fail if a strategy drifts past the tolerance
    python benchmarks/regression.py --set qa-set.json

The baseline holds scores only — no conversation text — so it is safe to commit.
Compare against a set built from public data if you want the numbers to be
reproducible by someone else.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = ROOT / "benchmarks" / "regression-baseline.json"


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def key_for(name: str, budget: int) -> str:
    return f"{name}@{budget}"


def run_eval(
    set_path: Path,
    strategies: list,
    budget: int,
    samples: int,
    limit,
    api_key: str,
    base_url: str,
    model: str,
) -> dict:
    """Shell out to qa_eval.py so there is exactly one implementation of scoring."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "results.json"
        cmd = [
            sys.executable, str(ROOT / "benchmarks" / "qa_eval.py"),
            "--set", str(set_path),
            "--strategies", ",".join(strategies),
            "--budget", str(budget),
            "--samples", str(samples),
            "--api-key", api_key,
            "--base-url", base_url,
            "--model", model,
            "--quiet",
            "--out", str(out),
        ]
        if limit:
            cmd += ["--limit", str(limit)]
        # Capture qa_eval's output: its per-point log and "results written"
        # line are noise here, but we want them if it fails.
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            sys.stderr.write(proc.stdout[-3000:])
            sys.stderr.write(proc.stderr[-3000:])
            raise SystemExit(f"qa_eval.py failed (exit {proc.returncode})")
        return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, required=True, dest="set_path")
    parser.add_argument("--name", default=None, help="defaults to the set file stem")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--strategies", default="simple,truncate")
    parser.add_argument("--budget", type=int, default=16000)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="fewer points = faster, noisier")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="allowed drop in score before this is a regression "
                             "(default 0.05; measured re-run swing is ~0.02)")
    parser.add_argument("--save-baseline", action="store_true",
                        help="record this run as the baseline")
    parser.add_argument("--api-key",
                        default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("SHENLUN_FORGE_API_KEY"))
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("set DEEPSEEK_API_KEY (or pass --api-key)")

    name = args.name or args.set_path.stem
    key = key_for(name, args.budget)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    print(f"Running {name} @ {args.budget} tokens: {', '.join(strategies)}")
    results = run_eval(args.set_path, strategies, args.budget, args.samples,
                       args.limit, args.api_key, args.base_url, args.model)
    got = {s: results["results"][s]["summary"] for s in strategies}
    n = next(iter(got.values()))["n"]

    baseline = load_baseline(args.baseline)
    expected = baseline.get(key)

    print()
    if args.save_baseline:
        baseline[key] = {
            "n": n,
            "budget": args.budget,
            "strategies": {s: round(got[s]["all"], 4) for s in strategies},
            "note": results.get("set", name),
        }
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(baseline, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"saved baseline for {key} (n={n}) -> {args.baseline}")
        for s in strategies:
            print(f"  {s:<10} {got[s]['all']:.1%}")
        return 0

    if not expected:
        print(f"no baseline for {key}; re-run with --save-baseline to record one")
        for s in strategies:
            print(f"  {s:<10} {got[s]['all']:.1%}")
        return 0

    if expected.get("n") != n:
        print(f"NOTE: baseline used n={expected.get('n')}, this run used n={n} — "
              "the comparison is not like-for-like")

    drifted = []
    print(f"{'strategy':<12}{'baseline':>10}{'now':>9}{'delta':>9}  status")
    for s in strategies:
        was = expected["strategies"].get(s)
        now = got[s]["all"]
        if was is None:
            print(f"{s:<12}{'-':>10}{now:>9.1%}{'-':>9}  new")
            continue
        delta = now - was
        ok = delta >= -args.tolerance
        if not ok:
            drifted.append(s)
        print(f"{s:<12}{was:>10.1%}{now:>9.1%}{delta:>+9.1%}  "
              + ("OK" if ok else f"REGRESSION (>{args.tolerance:.0%} drop)"))

    print()
    if drifted:
        print(f"FAIL: {', '.join(drifted)} dropped past the {args.tolerance:.0%} tolerance.")
        print("Before assuming a real regression, re-run once: the metric's own "
              "re-run swing is ~2 points.")
        return 1
    print(f"PASS: nothing dropped past the {args.tolerance:.0%} tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
