"""The `--external` contract: score someone else's compressor on these points.

Needs no API key and no network — each case uses a tiny shell command instead
of a real compressor, so the failure modes are tested without one.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load_qa_eval():
    spec = importlib.util.spec_from_file_location(
        "qa_eval", ROOT / "benchmarks" / "qa_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["qa_eval"] = module
    spec.loader.exec_module(module)
    return module


qa_eval = _load_qa_eval()

POINT = {
    "history": [
        {"role": "user", "content": "Which cache did we pick?"},
        {"role": "assistant", "content": "An in-process LRU, not Redis."},
    ],
    "query": "Why not Redis?",
}


def _command(script: str) -> str:
    return f"python3 -c {json.dumps(script)}"


def test_external_compressor_output_is_used():
    text, ratio = asyncio.run(qa_eval.compress_external(
        _command("import json,sys; json.load(sys.stdin); print(json.dumps({'text': 'compressed', 'ratio': 0.25}))"),
        POINT, 100,
    ))
    assert text == "compressed"
    assert ratio == 0.25


def test_external_compressor_receives_the_point_on_stdin():
    text, _ = asyncio.run(qa_eval.compress_external(
        _command(
            "import json,sys;"
            "d=json.load(sys.stdin);"
            "print(json.dumps({'text': d['query']+'|'+str(len(d['messages']))+'|'+str(d['maxTokens'])}))"
        ),
        POINT, 250,
    ))
    assert text == "Why not Redis?|2|250"


def test_camel_case_field_names_are_accepted():
    """A JS implementation should not need to rename its own fields."""
    text, ratio = asyncio.run(qa_eval.compress_external(
        _command("import json,sys; json.load(sys.stdin); print(json.dumps({'compressedText': 'from ts', 'compressionRatio': 0.4}))"),
        POINT, 100,
    ))
    assert text == "from ts"
    assert ratio == 0.4


def test_failing_command_is_reported_not_ignored():
    """Silently scoring an empty string would look like a very bad compressor."""
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(qa_eval.compress_external(
            _command("import sys; sys.stderr.write('boom'); sys.exit(3)"),
            POINT, 100,
        ))


def test_non_json_output_is_reported():
    with pytest.raises(RuntimeError, match="JSON"):
        asyncio.run(qa_eval.compress_external(
            _command("import sys; sys.stdin.read(); print('not json')"),
            POINT, 100,
        ))


def test_empty_output_is_reported():
    with pytest.raises(RuntimeError, match="no text"):
        asyncio.run(qa_eval.compress_external(
            _command("import json,sys; json.load(sys.stdin); print(json.dumps({'text': ''}))"),
            POINT, 100,
        ))
