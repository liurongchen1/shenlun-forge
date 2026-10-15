# Benchmarks

Reproducible fact-retention measurement for the compression strategies, so the
numbers in the README can be checked instead of taken on faith.

There are two evaluations here, measuring different things:

| Script | Measures | Use it to |
|---|---|---|
| `run_benchmark.py` | Fact retention against an answer key (substring match) | Check that identifiers survive verbatim |
| `qa_eval.py` | Whether a model can still *answer* from the compressed text | Check that compression is actually useful |

The second is the one to trust for "is this working". The first rewards copying
text, so it flatters whichever strategy copies the most.

```bash
python benchmarks/run_benchmark.py                  # rule-based rows only
export DEEPSEEK_API_KEY=sk-...                      # adds the LLM rows
python benchmarks/run_benchmark.py --json result.json
```

No API key is required. Without one the LLM strategies are skipped, and if a
run falls back to rules it is reported as `degraded` rather than silently
counted as an LLM result.

## The question-answering evaluation

```bash
# 1. Build a frozen evaluation set from your own conversations
python benchmarks/build_qa_set.py --conversations ./your-chats --out qa-set.json

# 2. Score strategies against it
python benchmarks/qa_eval.py --set qa-set.json --strategies truncate,simple,acs,hcs
```

`build_qa_set.py` turns each substantive user turn into an evaluation point:
the history before it, the question, the answer that was actually given, and
3-6 key points extracted from that answer **once and frozen**. Freezing matters:
re-extracting key points at scoring time means every run is graded against a
different ruler, which is enough variance to drown out the differences you are
trying to measure.

`qa_eval.py` then compresses the history, asks a model to answer using only the
compressed text (blind — the judge is not told which strategy produced it), and
scores coverage of the key points. `truncate` is included as the baseline to
beat, and it is a strong one.

### Running it in batches (memory)

On ~100k-token histories the embedding retriever is heavy: one point peaked at
**4.1GB**, because fastembed's own default batch size is 256 and a few hundred
messages then activate `256 × 512 × 512 × 12 layers` of intermediate tensors.
Two things fix it:

- `EmbeddingRetriever(batch_size=32)` cuts that to ~1.2GB. Lower it further if
  you are tighter on memory; it only costs speed.
- `qa_eval.py --batch-size N` scores at most N unscored points per run and
  writes results after every point, so the same command can be repeated to
  continue. A fresh process releases everything the previous batch allocated,
  which is the only reliable way to get memory back from Python/ONNX — clearing
  caches inside a long-lived process does not.

```bash
# repeat until it reports "nothing left to score"
python benchmarks/qa_eval.py --set qa-set.json --strategies embedding \
    --batch-size 3 --out embedding.json
```

Keep the batch small. With eight points per batch the process was killed partway
through and two points were silently skipped — only the resume log revealed it.
With three per batch it completes.

### Read the error bars, not the point estimate

`qa_eval.py` reports two different uncertainties, and mixing them up is how you
convince yourself of a difference that is not there:

| Column | Meaning | Shrinks with more… |
|---|---|---|
| `±95%CI` | How precisely the mean of *this point set* is known | points |
| `noise` | How much the mean moves when the identical run is repeated | nothing — it is the floor |

Set `--samples N` to measure `noise`: each point is compressed once and then
answered+judged N times, so the reported swing is judge variance, not
compression variance.

**More important: the independent unit is the conversation, not the point.**
Points drawn from the same conversation are heavily correlated — they share a
speaker, a topic and a writing style. Scoring 60 points from 10 conversations
does not give you n=60; it gives you roughly n=10, and every confidence interval
computed on the point count is far too narrow. Group by conversation before
believing any difference (the script's output includes `conversation` in each
row id for exactly this reason).

### What we measured, and what it means

On 131 points from 12 production conversations (3k-69k tokens):

- Point-level scoring suggests the rule-based strategy beats naive truncation by
  ~5 points (44.5% vs 39.3% on a balanced 60-point subset, two samples each).
  **Grouped by conversation that difference is t≈1.95 — at the edge of
  significance, not past it.** Treat "it helps" as the working hypothesis, not
  an established result.
- **The metric's own re-run swing is ~2 points** at the overall level (and up to
  ~5 points in a size band, where there are fewer points). Anything smaller than
  that is not a result.
- Two plausible improvements — raising the recent-message share (0.4 → 0.6/0.8)
  and replacing keyword retrieval with local embeddings — both landed inside
  that noise and were **not** adopted.
- **Budget is the real lever.** On the 20 conversations that must compress at
  every budget: 4000 → 49.3%, 8000 → 57.0%, 16000 → 63.2% (rule-based), while
  truncation went 47.8% → 55.2% → 53.7%. **Caveat:** a repeat run of the same
  measurement gave 53.7% → 53.8% → 57.2%, so only the direction (more budget is
  better) is safe to rely on, not the magnitude.

The honest summary: at tight budgets this library is roughly as good as cutting
off the oldest messages, and it costs more compute to get there. Its advantage
appears when the budget is generous. If that is not your situation, truncation
may genuinely be the better tool.

Run-to-run noise also means **do not tune parameters on a small point set** —
the first version of this experiment used 24 points and produced a
completely non-monotonic share curve that was pure noise.

If you want to detect a 5-point difference reliably, **add conversations, not
points**: a hundred more turns from the conversations you already have will not
help.

### LongMemEval: where the strategy actually wins

The conversations above are one user's workload, and they have a property that
hides the effect: the answer usually sits near the end of the history, so "keep
the most recent messages" is already close to optimal.

[LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
(MIT licensed, public) removes that property. Each question comes with ~100k
tokens of history across ~48 sessions, and the answer lives in one or two of
them with nothing in the question to say which. 96 questions, each an
independent history — the sample size the private data could not provide.

Convert it with `benchmarks/adapt_longmemeval.py`, then run `qa_eval.py` on the
result.

| Budget | `simple` | truncate | Difference |
|---|---|---|---|
| 4,000 | 13.5% | 11.5% | +2.1% (t=0.63, not significant) |
| 16,000 | 30.2% | 19.8% | **+10.4% (t=2.57, p=0.021)** |

At a workable budget the strategy beats truncation by a statistically
significant margin. That is the first result in this project that survives a
significance test.

By question type at a 16,000-token budget (n=16 each):

| Type | `simple` | truncate | Difference |
|---|---|---|---|
| single-session-user | 69% | 12% | **+56%** |
| single-session-preference | 6% | 0% | +6% |
| multi-session | 19% | 12% | +6% |
| single-session-assistant | 25% | 19% | +6% |
| temporal-reasoning | 0% | 0% | 0% |
| knowledge-update | 62% | 75% | **-12%** |

That breakdown is the actionable part:

- **Recalling a fact stated early** (`single-session-user`): +56 points. This is
  the case the library exists for — find the one session that matters out of
  forty-eight.
- **Knowledge updates** ("what is my *current* X?"): truncation wins. When the
  newest value is the answer, keeping the newest messages is exactly right.
- **Temporal reasoning**: both score 0. Neither approach can do it; do not claim
  otherwise.

It also explains the private-data result. Those conversations had their answers
near the end, so truncation was already near-optimal there — a benchmark can
only show a difference if its data contains one.

#### Does semantic retrieval beat keyword matching?

Yes, and it is the largest effect measured in this project. Same 96 questions,
same 16,000-token budget, same frozen answer key:

| Retriever | Correct | vs truncation | vs keyword matching |
|---|---|---|---|
| truncation (baseline) | 19.8% | — | — |
| keyword (`simple`) | 30.2% | +10.4pp (13:3, p=0.021) | — |
| `hcs` (ACS + verbatim code) | 32.3% | +12.5pp (13:1, p=0.002) | +2.1pp (p=0.50, n.s.) |
| **embeddings** | **44.8%** | **+25.0pp (26:2, p<0.001)** | **+14.6pp (20:6, p=0.0094)** |

All three pairwise comparisons are significant. Swapping the retriever moved the
result further than anything else we tried — further than the budget, the merge
policy, or the recent-window share. Those earlier experiments were all noise
because they were tuning the wrong part.

`hcs` is the strategy that keeps code blocks verbatim. On this set it lands
level with keyword matching (32.3% vs 30.2%, p=0.50) — expected, since
LongMemEval is everyday conversation with almost no code in it, so the feature
never gets a chance to matter.

So it was tested where it should matter: 45 points from Chinese coding-agent
sessions **including 26k command executions, file diffs and tool output** (the
adapter has `--include-tools` for exactly this):

| Strategy | Correct |
|---|---|
| truncation | 46.0% |
| keyword matching | 49.6% |
| `hcs` (verbatim code) | 45.9% |

No difference (p = 1.00). **`hcs` is not worth its keep**: across six
evaluations on two datasets and two content types it has never once been
significantly better than the much simpler alternatives, and it costs an extra
LLM call per compression. It stays in the code for anyone who wants it, but it
is no longer a recommended strategy and the auto-routing logic does not consider
it (see below).

By question type (embeddings vs keyword matching):

| Type | truncation | keyword | embeddings | diff |
|---|---|---|---|---|
| single-session-assistant | 19% | 25% | **62%** | +38pp |
| temporal-reasoning | 0% | 0% | **25%** | +25pp |
| knowledge-update | 75% | 62% | **81%** | +19pp |
| multi-session | 12% | 19% | **38%** | +19pp |
| single-session-preference | 0% | 6% | 6% | 0 |
| single-session-user | 12% | **69%** | 56% | -12pp |

Two findings worth more than the headline number:

- **Temporal reasoning goes from impossible to partially possible.** Neither
  truncation nor keyword matching scores a single point; embeddings reach 25%.
  It is the only configuration in this project that scores at all.
- **Keyword matching still wins one category** — recalling a fact the user
  stated themselves (69% vs 56%). When the question quotes the user's own
  words, lexical overlap is a strong signal, and semantic similarity blurs it.

Cost: ~95MB model, CPU inference, and much slower than the rule-based path (see
"Running it in batches" above). For long histories where you need to find things
stated earlier, that is a good trade. For short conversations it is not.

#### The same test in Chinese: no advantage at all

Repeating it on 67 points from 14 real Chinese coding-agent sessions
(3k–100k tokens of history, a user's own work conversations — not public),
gives a flat result:

| Retriever | Correct |
|---|---|
| truncation | 51.0% |
| keyword matching | 46.5% |
| embeddings | 48.4% |

No pairwise difference is significant (all p = 1.00; grouped by conversation,
embedding − truncation = +4.1%, t = 0.70). Keyword matching is marginally
*worse* than truncation.

The reason is visible in the absolute numbers: truncation scores **51% here
versus 19.8% on LongMemEval**. In a real work conversation the answer nearly
always sits near the end — you ask about what you were just doing — so keeping
the tail already captures most of it, and there is nothing left for a smarter
retriever to find. LongMemEval is built the other way round on purpose: the
answer is buried in one of ~48 sessions, so truncation fails by construction.

**The rule that survives both experiments:** the value of retrieval is a
function of how far the answer sits from the end of the history. Near the end,
truncation is free and already optimal. Far from it, the choice of retriever is
worth up to 25 points. Neither the language nor the domain drives this — the
answer's position does.

### A failed attempt worth recording

Once the rule above was clear — retrieval's value tracks how far the answer sits
from the end — the obvious next step was to detect that distance automatically
and send each query to the cheaper option. `shenlun_forge/router.py` tries to,
using the distribution of semantic similarity between query and history.

It does not work. On 67 real Chinese coding sessions it picked the better
retriever 41% of the time, and sent **64 of 67** queries to embeddings in a
dataset where truncation was the better choice:

| Approach | Correct |
|---|---|
| always truncate | **51.0%** |
| always embeddings | 48.4% |
| **by the router's advice** | 48.5% |

The signal is at fault, not the machinery: `recent_share` correlates **-0.09**
with "truncation would have won". A conversation stays on topic, so the query
resembles messages throughout the history — similarity looks spread out whether
or not the answer is in the last few turns. **Similarity spread is not a proxy
for answer position.**

Recorded because "add a smart router" is the obvious idea, and shipping
something that merely sounds sensible is easy. The module stays in the tree
marked experimental, and `route_eval.py` is there to test a better signal.

#### A second signal, and why the whole idea does not work

The first signal measured how *spread out* the query's similarity was across the
history. A narrower one is more plausible: not spread, but **is the single most
relevant message already in the recent turns?** If it is, keeping the tail
captures it. `benchmarks/signal_probe.py` measures that.

It fails too — `peak_in_recent` correlates -0.054 with truncation winning, and
the two groups it separates are indistinguishable (-0.005 vs +0.035). Routing on
it scores 2.9 points *worse* than always truncating.

Two failures in a row prompted a better question: is there anything to route
*for*? Comparing the three datasets answers it.

| Dataset | Answer position | Truncation | Retrieval | Room for a router? |
|---|---|---|---|---|
| LongMemEval | **random** (mean 0.53, 18.8% in the final quarter) | 19.8% | 44.8% | No — the position is random, so nothing predicts it, and the answer is simply "always retrieve" |
| Chinese Codex sessions | near the end | 51.0% | 48.4% | No — the two are tied, so there is no gain to capture |
| Private hermes sessions | near the end | 35.0% | — | No — same reason |

**A router only has value when the answer is systematically far from the end,
and in exactly that case the right choice is obvious.** Everywhere else it is
either unpredictable (random position) or pointless (the two options tie).
Two dead ends here are not a signal-tuning problem; there is no signal to find.

The practical consequence: do not route automatically. Default to the free
path, and switch on retrieval when you know your workload needs it — which is a
judgement about your data that you can make and the library cannot.

## Regression checking

That advantage is fragile: it holds only when the answer sits early in a long
history, and it is easy to lose while changing something unrelated — a
retriever tweak, a formatter reorder, a different merge policy.
`regression.py` makes the loss visible instead of silent.

```bash
# record what you have now
python benchmarks/regression.py --set qa-set.json --save-baseline

# afterwards: exit 1 if a strategy drops past the tolerance
python benchmarks/regression.py --set qa-set.json
```

- Exit code is 1 on a regression, 0 otherwise, so it drops into a shell script.
- Default tolerance is 5 points, deliberately above the metric's ~2-point
  re-run swing (see "Read the error bars" above). A failure means "look at
  this", not "the sky is falling" — re-run once before believing it.
- `benchmarks/regression-baseline.json` stores the recorded scores. It contains
  numbers only, no conversation text, so it is safe to commit.
- It is **not** wired into CI because it needs an API key. Run it before a
  release and after any change to the retriever, processor or formatter.

The committed baseline has two entries: one from public LongMemEval data (which
anyone can reproduce) and one from the private workload that produced the
opposite result, kept because a baseline that only contains the flattering case
is not worth much.

## Dataset

`data/real_dev_conversation.json` — 18 messages, ~5.4k tokens, a mixed
Chinese/English development session about building this library (architecture,
code, experiments, roadmap). Eight question sets with 132 pre-written answer
facts in total.

It is small and it is *one* conversation. It is not a standard benchmark, and
these numbers should not be quoted as one.

## Metric

**Fact retention** = fraction of the answer key found in the compressed text,
matched by case-insensitive substring containment.

This measures whether the *identifying token* survived. It does not check
whether a model could answer the question from the result. That distinction
matters enormously here — see below.

## Results (v0.3.0, token budget 1200)

| Strategy | Facts retained | Retention |
|----------|---------------|-----------|
| No compression (upper bound) | 129/132 | 97.7% |
| Truncate last 6 | 38/132 | 28.8% |
| Sliding window (1200 tokens) | 38/132 | 28.8% |
| `simple` (rules) | **111/132** | **84.1%** |
| `acs` (LLM) | 72/132 | 54.5% |
| `hcs` (LLM + verbatim code) | 68/132 | 51.5% |

Per scenario:

| Scenario | simple | acs | hcs |
|----------|--------|-----|-----|
| Version & config recall | 14/14 | 9/14 | 9/14 |
| Framework architecture | 18/18 | 5/18 | 15/18 |
| Entity preservation | 16/16 | 12/16 | 3/16 |
| Experiment data recall | 10/15 | 1/15 | 1/15 |
| Optimisation plan & decisions | 18/19 | 15/19 | 12/19 |
| Paper & roadmap | 15/17 | 15/17 | 15/17 |
| Code-level details | 13/18 | 13/18 | 13/18 |
| Causal reasoning | **7/15** | 2/15 | 0/15 |

> These numbers were re-measured after fixing a formatter bug that let the
> `structured` output format run to several times its token budget. Earlier
> figures for `acs` were inflated by that overshoot and are not comparable.

## How to read this

The rule-based strategy wins on this metric, and that is not an accident or a
regression — it follows from what the metric rewards:

- `simple` **copies surviving messages verbatim**, so every identifier it keeps
  is byte-identical to the original and substring matching finds it.
- `acs` **merges and rewrites** content through an LLM, and `hcs` spends up to
  40% of the budget on verbatim code blocks. Both trade literal matches for
  compression that a substring check cannot see.

So this table answers "which strategy preserves exact wording", and on that
question the cheap one wins. It does **not** measure what the LLM pipeline
exists for: answering a question whose vocabulary never appears in the
conversation, or reasoning across messages that individually look unrelated.
Earlier internal experiments on reasoning-heavy question sets (for example
"what did we consider and reject?") found the opposite ordering — the LLM
pipeline clearly ahead — because those facts are not present as literal strings
anywhere.

**Practical guidance:** start with `simple`. It is free, instant, and preserves
more literal detail than the LLM path. Move to `acs` when you can see it
failing on questions phrased differently from the source material.

## What to measure next

- The metric rewards verbatim copying. A model-based or semantic scoring pass
  would measure the LLM pipeline's actual value.
- `hcs` reserves 40% of the budget for code; on conversations that are not
  code-heavy it displaces narrative content. `code_budget_ratio` is the knob.
- `ACSProcessor` normalises informal expressions ("doesn't work" →
  "non-functional"), which can break literal matching of the original wording.
- **Merging is expensive at this budget.** Disabling the LLM merge step
  (`ACSProcessor(enable_merge=False)`) raised `acs` from 49.2% to 56.8% in a
  side-by-side run: merging collapses several messages into one large block, so
  a 1200-token budget then fits only two or three of them. Measured with the
  default merge enabled, `normalize` made no difference at all.
- One conversation, one language mix. English-only and code-heavy datasets
  would make the picture less anecdotal.
