"""Hard information: the parts of a conversation that cannot be paraphrased.

Paths, URLs, IPs, ports, command lines, filenames, numbers, config literals,
constant names. Two things use this:

* telling the model what the summary left out is one thing, but deciding *what
  to spend the budget on* needs to know where the irreplaceable content is —
  see ``SimpleStrategy.protect_user_messages="auto"``, which looks at whether
  the hard information sits in the user's turns or the assistant's;
* measuring a compression, where a substring check is the only judge that costs
  nothing, gives the same answer twice, and cannot be talked into a verdict.

Everything that survives paraphrase is deliberately excluded: this is about
literals, not meaning.
"""

from __future__ import annotations

import re


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

# (name, pattern, extra filter, group)
CATEGORIES = (
    ("path", re.compile(r"(?:[A-Za-z]:\\[^\s\"'`,;，。)]+|/(?:[\w.@%+-]+/)+[\w.@%+-]+)"), None, 0),
    ("url", re.compile(r"https?://[^\s\"'）)】,、，。]+"), None, 0),
    ("ip", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), None, 0),
    ("port", re.compile(r"(?::\d{2,5}\b)|(?:端口\s*\d{2,5})"), None, 0),
    ("constant", re.compile(r"\b[A-Z][A-Z0-9_]{4,}\b"),
     lambda t: t not in _STOP_CONSTANTS, 0),
    ("command", _COMMANDS, None, 0),
    ("filename", re.compile(rf"\b[\w.-]+\.(?:{_FILE_EXTENSIONS})\b", re.IGNORECASE), None, 0),
    ("number", re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|MB|GB|KB|TB|ms|分钟|小时|天|元|万|亿|次|条|张|页)\b"), None, 0),
    ("literal", re.compile(r"[`\"“'【]?([A-Za-z0-9_.\/\\:@#%-]{3,60})[`\"”'】]?"),
     lambda t: bool(re.search(r"[./\\:@#_-]", t)) and bool(re.search(r"[A-Za-z]", t)) and len(t) >= 4, 1),
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_facts(messages) -> dict:
    """Map every hard fact in ``messages`` to where it first appeared.

    ``messages`` is anything with ``role`` and ``content`` — dicts or objects.
    Each value is ``{"category": ..., "index": ..., "role": ...}``.
    """
    facts: dict = {}
    for index, message in enumerate(messages):
        if isinstance(message, dict):
            content = message.get("content") or ""
            role = message.get("role")
        else:
            content = getattr(message, "content", "") or ""
            role = getattr(message, "role", None)
        for name, pattern, keep, group in CATEGORIES:
            for hit in pattern.finditer(content):
                value = _normalise(hit.group(group) or "")
                if len(value) < 3:
                    continue
                if keep is not None and not keep(value):
                    continue
                if value not in facts:
                    facts[value] = {"category": name, "index": index, "role": role}
    return facts


def user_fact_share(messages, minimum: int = 10) -> float | None:
    """Fraction of the hard information that sits in user turns.

    ``None`` when there is too little of it to mean anything — a handful of
    facts is not a basis for a decision about where the budget should go.
    """
    facts = extract_facts(messages)
    if len(facts) < minimum:
        return None
    user = sum(1 for meta in facts.values() if meta["role"] == "user")
    return user / len(facts)
