# Releasing

Everything here is local until you run `twine upload`. Building artifacts is
not publishing.

## Version numbers

There is exactly one place to change the version:

```
shenlun_forge/_version.py     ->  __version__ = "0.3.0"
```

`pyproject.toml` reads it through `[tool.setuptools.dynamic]`, and the package
imports it, so the CLI, the distribution metadata and the source cannot drift
apart. `scripts/check_version.py` enforces this and runs in CI.

During development the version carries a `.devN` suffix (e.g. `0.3.0.dev0`) so
it can never be confused with an already-published release. Strip the suffix
when you decide to publish.

## Checklist

```bash
# 1. Make sure the tree is clean and tests pass
pytest tests/ -q
python scripts/check_version.py

# 2. Update CHANGELOG.md ("Unreleased" -> the version you are about to cut)

# 3. Bump the version in shenlun_forge/_version.py (remove .devN)
# 4. Commit

# 5. Build and verify (refuses to run on a dirty tree)
./scripts/release.sh

# 6. Publish — only when you actually mean to
twine upload dist/*
```

The build step must produce **both** `*.tar.gz` (sdist) and `*.whl`. Releases
0.2.2 through 0.2.4.1 shipped wheels only, which means anyone installing from
source or auditing the package had nothing to fetch. `release.sh` fails if the
sdist is missing.

Use an API token, not your password:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmc...
```

## After publishing

```bash
# Tag the release
git tag -a v0.3.0 -m "v0.3.0"
git push origin v0.3.0

# GitHub repo topics (the repo currently has none)
gh repo edit liurongchen1/shenlun-forge \
  --add-topic llm \
  --add-topic context-compression \
  --add-topic prompt-compression \
  --add-topic agent \
  --add-topic langchain \
  --add-topic python \
  --add-topic chinese-nlp

# Verify what PyPI actually serves
pip index versions shenlun-forge
```

## Known follow-ups

- **Rotate the DeepSeek key** that is hard-coded as a fallback default in
  several files under `experiments/` (for example
  `experiments/run_batch_experiments_deepseek_embedding.py`). That directory is
  git-ignored and never shipped, but the key exists on disk and should be
  treated as compromised. The same scripts also call
  `recent_window`/pipeline internals directly, so do not copy them into
  `benchmarks/` — use the clean runner there instead.
- `docs/` is English-only. `README.zh-CN.md` covers the Chinese quickstart; if
  the Chinese audience grows, mirror the four `docs/*.md` files.
