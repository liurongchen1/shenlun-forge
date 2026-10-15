# Changelog

Notable changes to this project. Version numbers follow the published
distribution versions on PyPI.

## Unreleased

### Added

- **`protect_user_messages`** on `compress()` / `compress_sync()` — rank user
  turns ahead of the rest and never cut one in half. Default `"auto"`.

  Found by replaying 666 messages from six real production agent conversations
  turn by turn (`benchmarks/replay_curve.py`): the gap against the harness this
  was compared with was almost entirely in *user turns* — at turn 88 of the
  longest conversation, 56.6% of the user messages' hard facts were missing
  from the context, against 18.3% for an implementation that protects them.
  A user turn is usually a constraint the rest of the conversation depends on
  ("the goal is X", "do not touch Y"), so ranking it by relevance to the current
  question drops exactly the ones that matter most.

  The rule is not universal, and what separates the two cases is not whether the
  user is asking or instructing — it is where the hard information sits, which
  is visible before anything is compressed. `"auto"` measures that share and
  protects only when the user's turns carry it. Across eight conversations, the
  four where user turns held 46–72% of the hard information gained 0 to 8
  points; the four at 0–14% were unchanged or, in one case, 14 points worse
  when forced on. `stats["user_protection"]` reports what was decided.

- **`shenlun_forge.facts`** — extraction of hard information: paths, URLs, IPs,
  ports, command lines, filenames, numbers, config literals, constants. The
  parts of a conversation that cannot be paraphrased. Used by the `"auto"`
  decision above and by the replay metric below, so the two cannot drift apart.

- **`benchmarks/replay_curve.py`** and **`benchmarks/replay_regression.py`** —
  replay a whole conversation the way a live agent compresses it (compress the
  history, append the turn, repeat), score how much *hard information* is still
  visible at every turn, and fail a shell script when a change makes it worse.

  Hard information means paths, URLs, IPs, ports, command lines, filenames,
  numbers and config literals — no LLM judge, so it costs nothing, has no
  variance, and gives the same answer twice. `--match-label` hands this run the
  same context size the other implementation sent, turn by turn; without it the
  comparison measured who was allowed to send more rather than who compresses
  better.

- **`qa_eval.py --external NAME=COMMAND`** — enter any other compressor into the
  same evaluation: same points, same frozen answer key, same judge, same
  budget. The command reads `{messages, query, maxTokens}` as JSON on stdin and
  prints `{text, ratio}`; a non-zero exit, non-JSON stdout, or empty text is a
  hard error rather than a silent zero.

  This is the comparison that is actually worth quoting. Two compressors scored
  on different conversations, with different answer keys, at different budgets,
  produce two numbers that cannot be subtracted from each other. Also in
  `benchmarks/external_adapter_example.ts`, a template for a TypeScript project
  that imports the *deployed* module rather than a copy.

## 0.4.0 — 2026-10-15

### Added

- **`shenlun-forge demo`** — one command, no API key, no network, under a
  second. It runs a built-in 32-message conversation (three unrelated topics)
  through plain truncation and through this library, and prints which messages
  each one kept, whether the one that answers the question survived, and the
  closing note the summary appends. `--lang zh` for Chinese, `--budget N` to
  change the demonstrated budget.

  It exists because the difference this library makes is not "shorter text" —
  truncation does that for free — but *which* messages survive. A demo that
  only showed token counts would advertise the wrong thing. Two tests keep it
  honest: one asserts the answering message really does survive compression,
  and one asserts truncation really does lose it.

- **The summary now says what it does not contain.** Every result ends with a
  line like
  `[compression note] 37 earlier messages (~12,480 tokens) are not included above.`

  The failure this removes is specific: a model handed a lossy summary has no
  way to know it is lossy, so when the answer happens to sit in the dropped part
  it answers "that is not in the context" — or invents something. One line
  corrects "I read the whole conversation" to "I read part of it, and here is
  where the gap is".

  On by default, written in the language of the question, and sized inside
  `max_tokens` rather than appended on top of it (the budget is reserved up
  front, so `budget_overflow="truncate"` still holds as a hard limit). Turn it
  off with `dropped_notice=False`, or `--no-dropped-notice` on the CLI.
  `stats["dropped_messages"]` and `stats["dropped_tokens"]` are reported either
  way, so the gap can be logged without putting a line in the prompt.

  For the ACS pipeline the counts come from what the formatter actually wrote,
  not from what it was handed — a block dropped for budget reasons is a block
  the model cannot see.

  Measured three times (`simple` vs the same compressor with the note
  suppressed, paired by question, significance grouped by conversation):

  | Dataset | Score | 20–50% compression band |
  |---|---|---|
  | LongMemEval, 96 independent ~100k-token histories | −1.0pp (p=0.32) | — |
  | hermes production, 131 points / 12 conversations | +5.4pp (p=0.089) | +10.4pp (p=0.034) |
  | hermes chat logs, 163 points / 29 conversations | +4.1pp (p=0.026) | +11.6pp (p=0.005) |

  So: no measurable cost where it cannot help, and a gain where a compressed
  history still reads like a complete one. The original motivation — that a
  model invents answers when the context is incomplete — did **not** show up:
  concrete answers covering no key points are 0.0–3.1% of points either way,
  and the rate at which answers declare information missing barely moves. The
  note changes how the model uses the material it has, not how much it has.
  Full write-up and reproduction commands in `benchmarks/README.md`.

- `tests/test_compression_regression.py` — behaviour regression on a fixed
  conversation that runs in CI on every commit and needs no credentials. It
  asserts the compressors still keep most of the key facts, stay inside their
  budget, protect the newest turns, and produce different output for different
  questions. Deliberately loose: it catches behaviour changing, not numbers
  wobbling.

  Why this and not the full evaluation in CI: end-to-end answer quality needs
  an API key *and* your conversations, neither of which belongs in a public
  workflow. The split is documented in `benchmarks/README.md`.

## 0.3.1 — 2026-10-15

Metadata and a request, no behaviour change.

### Changed

- **The package summary now claims only what the library can prove.** The old
  one sold "analytical compression strategy", which six evaluations could not
  substantiate — on ordinary work conversations compression ties with plain
  truncation, and the one large effect comes from the retriever rather than the
  pipeline. The new summary points at the capability that survived measurement:
  *"Context compression for LLM agents that you can measure on your own
  conversations"*.

### Added

- A one-time request to star the project if it helps, in the README, the package
  docstring, and the CLI. The CLI version writes a local marker file, so it
  appears once per machine and never again.

  This is deliberately not a growth hack aimed at AI assistants. Whether an
  assistant mentions a project depends on whether its claims can be checked, and
  that is decided by the benchmarks, not by the wording of a request.

## 0.3.0 — 2026-10-15

### Fixed

- `shenlun_forge.__version__` and `shenlun-forge version` now report the real
  package version. Before this, `__version__` was hard-coded to `0.2.0` while
  the distribution was `0.2.4.1`, so the CLI printed a version that did not
  exist on PyPI.
- Compression no longer returns a payload larger than its input. When the
  history is under 200 tokens, or a run would save less than 5% of them, the
  original context is returned unchanged and `stats["passthrough"]` is set to
  `True`. Previously a short conversation could come back with
  `compression_ratio > 1.0`.
- Context that already fits is no longer compressed. If `original_tokens <=
  max_tokens` the text is returned unchanged (`passthrough: True`) — compressing
  in that case can only lose information and saves the caller nothing.
- The rule-based strategy no longer spends the whole budget on whatever sits
  near the end of the history. Messages that share vocabulary with the query go
  in first; unrelated messages only fill the budget when nothing matched, and
  recent turns are always preserved verbatim (previously `simple` had no
  recency protection at all, so the newest turns could be dropped).
- LLM-driven strategies no longer degrade silently. `acs` and `hcs` warn at
  construction time when no usable LLM is configured, and every result carries
  `stats["llm_available"]`, `stats["llm_calls"]`, `stats["llm_failures"]`,
  `stats["degraded"]` and `stats["degraded_reason"]`.
- Compressed output could blow past `max_tokens`: the verbatim recent-message
  window was appended *on top of* the budget rather than inside it, so a
  1200-token request could return 2000+ tokens and squeeze the compressed part
  down to a few hundred. The window now shrinks to at most 40% of the budget,
  and any remaining overshoot is reported as `stats["over_budget"]`.
- The formatter ignored the retriever's relevance scores and ranked blocks by
  information type and position alone, so the query-aware ranking the retriever
  had just computed was thrown away. `CompressedBlock` now carries
  `relevance_score` and the formatter weights it highest.
- The `structured` output format did not enforce its token budget. After a
  partial truncation it broke only the inner loop, so every remaining group
  appended another truncated entry — a 300-token budget produced 2766 tokens
  (9x), and a 1200-token budget produced 2000+. `acs` uses this format for
  analysis-shaped queries, so its outputs were both oversized and lower quality
  than the budget allowed. Reported benchmark numbers for `acs` were inflated
  by this and have been re-measured.
- `hcs` truncated code blocks by estimating three characters per token, which
  is roughly right for English prose and wrong for Chinese or code. On real
  conversations the "truncated" block came out at up to twice its allowance and
  pushed the whole result past `max_tokens` (2 of 18 production sessions at a
  4000-token budget). Truncation is now token-exact, the same way the formatter
  does it.
- `EmbeddingRetriever` cached every embedding it ever produced, forever. On one
  conversation that is invisible; scoring a few dozen 100k-token conversations
  drove resident memory to ~3GB. The cache is now bounded (`cache_size`,
  default 20000 entries, LRU eviction). Note that memory still grew on very long
  histories even with the cache bounded — see the known-issue note in the
  module docstring; the rule-based retriever is the safer choice there.
- `EmbeddingRetriever` used fastembed's default batch size of 256. A few hundred
  messages then activated `256 × 512 × 512 × 12` layers of intermediate tensors,
  measured at **4.1GB for a single ~100k-token history**. New `batch_size`
  parameter defaults to 32, which cuts that to ~1.2GB for a little speed.
- `hcs` was tested where its one feature should matter — 45 points from Chinese
  sessions that include 26k command executions and file diffs — and landed level
  with truncation and keyword matching (45.9% vs 46.0% / 49.6%, p=1.00). Across
  six evaluations it has never been significantly better while costing an extra
  LLM call per compression. It is no longer a recommended strategy; the code
  stays for anyone who wants it.
- `compress_sync` leaked the HTTP client across event loops: each call ran in a
  fresh loop, and the previous client was collected after that loop had closed,
  so httpx printed a `RuntimeError: Event loop is closed` traceback per call.
  The client is now closed at the end of each sync call. Also affects
  `LangChainShenlunForge.invoke`, which now delegates to `compress_sync`.

### Changed

- **The retriever matters more than any tuning knob, and that is now measured.**
  On public LongMemEval (96 independent 100k-token questions, 16k budget):
  truncation 19.8%, keyword matching 30.2%, semantic embeddings 44.8% — all three
  pairwise differences significant (McNemar p=0.021, 0.0094, <0.001). Earlier
  tuning attempts (recent-message share, merge policy, budget) were all noise by
  comparison. For long histories, `retriever_type="embedding"` is the
  configuration to reach for; for questions quoting the user's own words,
  keyword matching is still ahead (69% vs 56%).
- **The default strategy is now `simple`** — rule-based, no API key, runs in
  milliseconds. Both `ShenlunForge.create()` and `shenlun-forge compress`
  default to it. Pass `strategy="acs"` or `strategy="hcs"` for LLM-powered
  compression.
- The recent-message protection window is adaptive rather than a fixed six
  messages, and protected messages bypass the token-budget formatter so they
  are never truncated.
- Package version now has a single source of truth
  (`shenlun_forge/_version.py`), read by both `pyproject.toml` and the package.
- README restructured around use cases (English first, 中文版见
  `README.zh-CN.md`).

### Added

- Usage documentation: `docs/quickstart.md`, `docs/scenarios.md`,
  `docs/integration.md`, `docs/faq.md`.
- Runnable examples: `examples/quickstart_no_llm.py`,
  `examples/compress_before_llm.py`, `examples/langchain_middleware.py`.
- `shenlun_forge.llm.probe_llm()` and `shenlun_forge.llm.InstrumentedProvider`
  for detecting and reporting LLM availability.
- `shenlun_forge.register_strategy()` — register a custom strategy so it works
  with `create()`, `list_strategies()` and the CLI.
- `plan_compression()` (and `ShenlunForge.plan()`) — decide whether a context
  needs compressing at all before spending anything on it. Needs no LLM: if the
  history fits the window it says so and recommends not compressing, otherwise
  it returns the budget to use.
- `benchmarks/qa_eval.py --batch-size N` — score at most N unscored points per
  run and write results after every point, so the same command can be repeated
  to continue. Bounds peak memory, and makes an interrupted run resumable
  instead of restarting from zero.
- `shenlun_forge.router.recommend_retriever()` — **experimental and not
  recommended**: it fails validation, picking the better retriever only 41% of
  the time and routing 64 of 67 queries the wrong way on data where truncation
  won. Kept so the signal can be retried against `benchmarks/route_eval.py`;
  documented as a failed attempt rather than a feature.
- `benchmarks/adapt_codex_sessions.py` (with `--include-tools`) and
  `benchmarks/adapt_longmemeval.py` — turn local Codex session history or public
  LongMemEval into the conversation format `build_qa_set.py` consumes.
- `benchmarks/route_eval.py` — measures whether a retriever recommendation
  actually picks the better option, against results already on disk.
- `benchmarks/signal_probe.py` — tests a narrower routing signal (is the peak
  similarity in the recent turns) and shows it fails too. Together with
  `route_eval.py` they establish why automatic routing is a dead end here: an
  answer is either randomly positioned (nothing predicts it, and "always
  retrieve" is already optimal) or sits near the end (where the two options tie,
  so there is nothing to win).
- `budget_overflow` on every `compress()` / `compress_sync()` and `--budget-overflow`
  on the CLI. When the newest message alone exceeds the remaining budget,
  `"keep"` (default) preserves it whole — information completeness first — and
  reports `stats["over_budget"]`; `"truncate"` trims it so `max_tokens` stays a
  hard limit and reports `stats["recent_truncated"]`. Previously the overshoot
  was unavoidable and undecided.
- LlamaIndex adapter: `shenlun_forge.adapters.llamaindex_adapter`, with
  `trim()` / `atrim()` returning the caller's own message objects and
  `compress_text()` for prompt assembly.
- `benchmarks/` — a runnable fact-retention benchmark with a documented dataset
  and per-scenario results, plus the honest reading that the rule-based
  strategy wins on literal fact retention.
- GitHub Actions CI: tests on Python 3.9/3.11/3.12/3.13, examples smoke test, and
  a build job that checks metadata and installs the built wheel.
- `scripts/check_version.py` (fails if the version can drift again) and
  `scripts/release.sh` (builds sdist **and** wheel, refuses a dirty tree).
- `RELEASING.md`, and `py.typed` so type checkers use the inline annotations.
- `license-files` metadata so PyPI shows the MIT license.
- Tests are back in the repository (`tests/`), fixed to the current class names
  and extended with behavioural tests for the guarantees above.

## Earlier public releases

- `0.2.4.1`, `0.2.4`, `0.2.3`, `0.2.2`, `0.2.1`, `0.2.0` — iterative releases
  of the ACS pipeline (analyze, retrieve, process, format), the rule-based
  `simple` strategy, the hybrid `hcs` strategy, the embedding retriever, the
  CLI, and the LangChain adapter.
