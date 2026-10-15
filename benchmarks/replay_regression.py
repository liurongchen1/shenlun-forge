#!/usr/bin/env python3
"""Score a replay by how much *hard information* the model can still see.

This is the metric a harness report tends to use, and it is a good one for
regressions: it counts only things that cannot be paraphrased — paths, URLs,
IPs, ports, command lines, filenames, numbers, config literals, constant names
— so it needs no LLM judge, costs nothing to run, and gives the same answer
twice. Facts are matched by substring against what the model was actually
shown (compressed history + the current turn).

Two modes:

    python benchmarks/replay_regression.py --label shenlun-user --save-baseline 22 138 50
    python benchmarks/replay_regression.py --label shenlun-user 22 138 50

The second compares against the recorded baseline and exits 1 on a regression,
so it drops into a shell script. Pairing is per turn, and the significance
figure is grouped by conversation — points inside one conversation share a
speaker and a topic, so counting turns as independent would overstate n by an
order of magnitude.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "benchmarks" / "replay-baseline.json"


# ── What counts as hard information ────────────────────────────────────────
# Mirrors the extraction rules a TypeScript harness report uses, so the two
# scores mean the same thing. Anything that survives paraphrase is excluded on
# purpose: this metric is about literals, not about meaning.

_STOP_CONSTANTS = {
    "HTTPS", "HTTP", "TODO", "README", "JSON", "HTML", "CSS", "VPN", "SSH",
    "PPTX", "API", "URL", "APP", "MBTI", "CPU", "DNS", "SSL", "TLS", "SQL",
    "UUID", "EXAMPLES", "WARNING", "ERROR", "DEBUG", "OPTIONS", "PATCH",
    "DELETE", "POST", "HEAD", "GET", "PUT",
}

_COMMANDS = re.compile(
    r"^\s*(?:ssh|ssh-keygen|scp|rsync|pm2|npm|npx|yarn|pnpm|systemctl|nginx|docker|"
    r"docker-compose|psql|curl|curl\.exe|wget|grep|rg|sed|awk|chmod|chown|sudo|cd|"
    r"ls|cat|vim|nano|kubectl|git|python3?|node|brew|apt|apt-get|yum|tar|unzip|kill|"
    r"tail|head|find|xargs|openssl|netstat|ipconfig|certbot|ufw|export|Get-[A-Za-z]+|"
    r"Set-[A-Za-z]+|New-[A-Za-z]+|Add-[A-Za-z]+|Test-[A-Za-z]+)\b[^\n]*",
    re.MULTILINE,
)

_FILE_EXTENSIONS = (
    "ts|tsx|js|jsx|mjs|json|md|conf|sql|yml|yaml|env|db|py|sh|ps1|bat|html|css|"
    "png|jpg|jpeg|zip|log|txt|prisma|toml|ini|pem|key"
)

_CATEGORIES = (
    ("路径", re.compile(r"(?:[A-Za-z]:\\[^\s\"'`,;，。)]+|/(?:[\w.@%+-]+/)+[\w.@%+-]+)"), None, 0),
    ("URL", re.compile(r"https?://[^\s\"'）)】,、，。]+"), None, 0),
    ("IP", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), None, 0),
    ("端口", re.compile(r"(?::\d{2,5}\b)|(?:端口\s*\d{2,5})"), None, 0),
    ("常量名", re.compile(r"\b[A-Z][A-Z0-9_]{4,}\b"),
     lambda t: t not in _STOP_CONSTANTS, 0),
    ("命令行", _COMMANDS, None, 0),
    ("文件名", re.compile(rf"\b[\w.-]+\.(?:{_FILE_EXTENSIONS})\b", re.IGNORECASE), None, 0),
    ("数值", re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|MB|GB|KB|TB|ms|分钟|小时|天|元|万|亿|次|条|张|页)\b"), None, 0),
    ("配置字面值", re.compile(r"[`\"“'【]?([A-Za-z0-9_.\/\\:@#%-]{3,60})[`\"”'】]?"),
     lambda t: bool(re.search(r"[./\\:@#_-]", t)) and bool(re.search(r"[A-Za-z]", t)) and len(t) >= 4, 1),
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_facts(messages: list) -> dict:
    """Every hard fact, mapped to where it first appeared."""
    facts: dict[str, dict] = {}
    for index, message in enumerate(messages):
        content = message.get("content") or ""
        for name, pattern, keep, group in _CATEGORIES:
            for hit in pattern.finditer(content):
                value = _norm(hit.group(group) or "")
                if len(value) < 3:
                    continue
                if keep is not None and not keep(value):
                    continue
                if value not in facts:
                    facts[value] = {
                        "category": name,
                        "msgIndex": index,
                        "role": message.get("role"),
                    }
    return facts


def _facts_upto(rows: list, upto_id: int, include_current: bool) -> tuple[dict, int]:
    """What had been said (and should be visible) by this turn."""
    if include_current:
        prefix = [m for m in rows if m["id"] <= upto_id]
    else:
        prefix = [m for m in rows if m["id"] < upto_id]
    return extract_facts(prefix), len(prefix)


def score_series(rows: list, series: list) -> list[dict]:
    """Lost-fact rate for every turn of one conversation."""
    per_turn = []
    for item in series:
        visible = _norm(f"{item.get('contextText') or ''}\n{item.get('currentUserContent') or ''}")
        facts, prefix_len = _facts_upto(rows, item["uptoId"], True)
        # A turn with nothing quotable in front of it is not a free pass — it
        # scores zero loss, which is what the harness report does, so the turn
        # counts stay aligned between the two implementations.
        lost = [v for v in facts if v not in visible]
        user_facts = [v for v, meta in facts.items() if meta["role"] == "user"]
        user_lost = [v for v in user_facts if v not in visible]
        # "Early" means the first 30% of everything said so far: the part that
        # has had the most chances to be dropped.
        early_count = max(1, math.floor(prefix_len * 0.3))
        early = [v for v, meta in facts.items() if meta["msgIndex"] < early_count]
        early_lost = [v for v in early if v not in visible]
        per_turn.append({
            "turn": item["turn"],
            "total": len(facts),
            "lost": len(lost),
            "loss": (len(lost) / len(facts)) if facts else 0.0,
            "user_total": len(user_facts),
            "user_loss": (len(user_lost) / len(user_facts)) if user_facts else None,
            "early_total": len(early),
            "early_loss": (len(early_lost) / len(early)) if early else None,
        })
    return per_turn
def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _two_sided_p(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2))


def summarise(conversations: dict) -> dict:
    """Mean loss per conversation, then across conversations."""
    per_conversation = {}
    for conversation_id, data in conversations.items():
        turns = data["turns"]
        losses = [t["loss"] for t in turns]
        user = [t["user_loss"] for t in turns if t["user_loss"] is not None]
        early = [t["early_loss"] for t in turns if t["early_loss"] is not None]
        per_conversation[conversation_id] = {
            "turns": len(turns),
            "loss": _mean(losses),
            "user_loss": _mean(user) if user else None,
            "early_loss": _mean(early) if early else None,
        }
    overall = _mean([c["loss"] for c in per_conversation.values()])
    return {"per_conversation": per_conversation, "overall": overall}


def load(directory: Path, label: str, conversation_id: int):
    rows = json.loads((directory / f"conv{conversation_id}.json").read_text(encoding="utf-8"))
    series = json.loads(
        (directory / f"{label}-sim-{conversation_id}.json").read_text(encoding="utf-8")
    )["series"]
    return rows, series


def measure(directory: Path, label: str, ids: list) -> dict:
    conversations = {}
    for conversation_id in ids:
        try:
            rows, series = load(directory, label, conversation_id)
        except FileNotFoundError as exc:
            raise SystemExit(f"missing replay data: {exc.filename}")
        conversations[str(conversation_id)] = {"turns": score_series(rows, series)}
    return conversations


def compare(baseline: dict, current: dict) -> dict:
    """Paired, per turn, grouped by conversation — the honest figure."""
    pairs = []
    for conversation_id, data in current.items():
        base = baseline.get(conversation_id)
        if not base:
            continue
        by_turn = {t["turn"]: t for t in base}
        for turn in data:
            other = by_turn.get(turn["turn"])
            if other and turn["total"] == other["total"]:
                pairs.append((conversation_id, turn["loss"] - other["loss"]))
    if not pairs:
        return {}
    diffs = [d for _, d in pairs]
    mean = _mean(diffs)
    spread = (sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5 if len(diffs) > 1 else 0.0
    t = mean / (spread / math.sqrt(len(diffs))) if spread else 0.0

    grouped: dict[str, list] = defaultdict(list)
    for conversation_id, diff in pairs:
        grouped[conversation_id].append(diff)
    group_means = [_mean(v) for v in grouped.values()]
    clustered = {}
    if len(group_means) > 1:
        gm = _mean(group_means)
        gs = (sum((g - gm) ** 2 for g in group_means) / (len(group_means) - 1)) ** 0.5
        ct = gm / (gs / math.sqrt(len(group_means))) if gs else 0.0
        clustered = {"mean": gm, "t": ct, "p": _two_sided_p(ct)}
    return {
        "n": len(diffs),
        "conversations": len(grouped),
        "mean": mean,
        "t": t,
        "p": _two_sided_p(t),
        "clustered": clustered,
        "worse": sum(1 for d in diffs if d > 0),
        "better": sum(1 for d in diffs if d < 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="+", type=int)
    parser.add_argument("--label", default="shenlun")
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--dir", type=Path,
                        default=Path(os.environ.get("HERMES_REPLAY_DIR", "/tmp/hermes-realdata")))
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="loss may rise by this much before it counts as a regression")
    args = parser.parse_args()

    conversations = measure(args.dir, args.label, args.ids)
    summary = summarise(conversations)

    print(f"replay label: {args.label}")
    print(f"{'conv':<8}{'turns':>7}{'loss':>9}{'user loss':>11}{'early loss':>12}")
    for conversation_id, data in summary["per_conversation"].items():
        user = "-" if data["user_loss"] is None else f"{data['user_loss']:.1%}"
        early = "-" if data["early_loss"] is None else f"{data['early_loss']:.1%}"
        print(f"{conversation_id:<8}{data['turns']:>7}{data['loss']:>9.1%}{user:>11}{early:>12}")
    print(f"{'all':<8}{'':>7}{summary['overall']:>9.1%}")

    if args.save_baseline:
        payload = {
            "label": args.label,
            "tolerance": args.tolerance,
            "conversations": {
                cid: data["turns"] for cid, data in conversations.items()
            },
            "summary": summary,
        }
        args.baseline.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nbaseline saved to {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"\nno baseline at {args.baseline} — run with --save-baseline first")
        return 1

    recorded = json.loads(args.baseline.read_text(encoding="utf-8"))
    base = recorded.get("conversations", {})
    report = compare(
        base, {cid: data["turns"] for cid, data in conversations.items()}
    )
    if not report:
        print("\nbaseline has no overlapping turns — nothing to compare")
        return 1

    print(f"\nvs baseline ({recorded.get('label')}):")
    print(f"  paired over {report['n']} turns in {report['conversations']} conversations")
    print(f"  loss change {report['mean']:+.1%} (worse in {report['worse']}, "
          f"better in {report['better']})")
    print(f"  point level   t={report['t']:.2f}, p={report['p']:.3f}")
    if report["clustered"]:
        c = report["clustered"]
        print(f"  by conversation t={c['t']:.2f}, p={c['p']:.3f}  (the one to quote)")

    regressed = [
        cid for cid, data in summary["per_conversation"].items()
        if cid in base and data["loss"] > _mean([t["loss"] for t in base[cid]]) + args.tolerance
    ]
    if regressed:
        print(f"\nREGRESSION: loss rose past {args.tolerance:.0%} in conversation(s) "
              f"{', '.join(sorted(regressed))}")
        return 1
    print(f"\nwithin tolerance ({args.tolerance:.0%} per conversation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
