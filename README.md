# AI Customer Support Agent — SpotifyCares

An agent that reads an incoming customer tweet, classifies its intent, retrieves how the
brand historically resolved similar issues, drafts a reply grounded in those precedents,
and decides **AUTO_HANDLE** or **ESCALATE** with a stated reason.

Built on the Kaggle *Customer Support on Twitter* dataset (2.8M tweets, 108 brands).
Brand selection was measured, not assumed — see [`DATA_ANALYSIS.md`](DATA_ANALYSIS.md).

```
customer message
      ↓  intent classifier      (embeddings + logistic regression, 12 intents)
      ↓  retrieval              (top-5 historical resolutions, 13,347 indexed)
      ↓  generator              (LLM, constrained to retrieved evidence)
      ↓  escalation             (7 ordered rules, first match decides + explains)
      ↓  {intent, reply, action, reason, evidence_ids}
```

---

## Chat app (customer-facing UI)

```bash
pip install fastapi uvicorn
python -m app.server            # open http://127.0.0.1:8000
```

A multi-turn chat on top of the same pipeline (`src/chat.py`, `app/`). It adds three
chat behaviours without touching the evaluated path: classification and retrieval use
the last 3 customer turns; a message that was not understood gets up to 2 clarifying
questions (`CLARIFY`) before escalating, while every risk rule still escalates
immediately; and a handoff, once made, is permanent for the conversation. Toggle
**Agent view** to see the intent, rule, reason and past cases behind each reply.

Needs the provider key in `.env`. Without one, every turn falls back to a question or a
handoff, never an ungrounded answer. Chat responses are cached in
`data/cache/chat_llm/` (gitignored). The chat flow is **not evaluated**: the numbers
below are for the single-message pipeline. No live agent queue is connected, so
"passing you to a specialist" is where a real helpdesk integration would go.

---

## What is and is not measured yet

Please read this before any number in this repo.

| Component | Status | Evidence |
|---|---|---|
| Intent classification | **Measured** on 200 hand-labelled examples | `evaluation/results/metrics.json` |
| Retrieval | **Measured** on 200 examples | same |
| Escalation decision | **Measured** on 200 examples | same |
| Failure analysis | **Measured**, derived from the above | `evaluation/results/failure_analysis.txt` |
| Reply quality (LLM judge) | **Measured** on all 200 examples × 3 reply systems — judge scores, **not yet validated against humans** | `metrics.json` → `reply_quality`, per-example in `predictions.jsonl` |
| Human-vs-LLM judge agreement | **NOT STARTED** — no human has scored anything | infrastructure ready, see below |

The offline half reproduces exactly and for free. The reply-quality half was produced with
`gpt-5.6-luna` (OpenAI) for both generation and judging; every response is cached in
`data/cache/llm/`, so a re-run reads them from disk instead of calling the API.

---

## Setup

```bash
pip install -r requirements.txt
```

`anthropic` and `google-genai` are optional — only needed if you switch provider.

---

## Reproduce the headline results from the cache — **no API key, no cost**

The repository ships the trained classifier, the retrieval index and the embedding cache
(`data/cache/`), the processed pairs (`data/processed/`), and **all 1,001 LLM responses
from the final run** (`data/cache/llm/`). Reproducing every reported number, including
reply quality, is one command:

```bash
python -m evaluation.run                                          # full evaluation, served from cache
python -m evaluation.failure_analysis --top 6 --examples 2        # top failure modes, offline
python -m pytest -q                                               # 28 tests, offline
```

Expected final line of `evaluation.run`:

```
HEADLINE  intent macro-F1 0.587 (acc 0.575) | escalation F1 0.809, false auto-handle 14.1% | reply quality 4.58/5
```

Results are written to `evaluation/results/` (`metrics.json`, `predictions.jsonl`,
`report.md`). Progress lines show `cache hits` and `live` counts as the run goes.

**Verified:** a cache-only re-run, with the API endpoint pointed at localhost so no call
could leave the machine, finished in **14.6 s** with **1,001 cache hits and 0 live
calls**. Every metric matched the final run, and `predictions.jsonl` was byte-identical.
The only differences were `llm_calls` and `runtime_seconds`.

**Run it without an API key.** Without a key, a cache miss cannot spend money: the run
stops with *"LLM unavailable"* (exit code 2) instead of calling the API. No `.env` is
needed — the defaults (`openai` / `gpt-5.6-luna`) are the ones the cache was built with.

Only intent, retrieval and escalation? Add `--no-llm`: it skips generation and judging
entirely and prints the same headline without the reply-quality term.

### Rebuilding from raw data (optional)

```bash
python scripts/download_data.py       # 516 MB twcs.csv (~2 min)
python scripts/prepare_data.py        # ~30s   raw tweets -> conversations -> pairs
python scripts/build_index.py         # ~2min  train classifier + build retrieval index
python -m evaluation.run --no-llm     # intent + retrieval + escalation metrics
```

The first `build_index.py` run downloads `all-MiniLM-L6-v2` (~90 MB). A rebuild
overwrites the shipped classifier and index. If that changes any LLM prompt, those calls
miss the cache — so reproduce reply quality from the shipped artefacts, as above.

---

## Headline results (200 hand-labelled examples)

| System | Intent acc | Intent macro-F1 | Retrieval ref-recall@5 | Escalation F1 | False auto-handle |
|---|---:|---:|---:|---:|---:|
| Majority baseline (trivial) | 0.070 | 0.011 | — | 0.693 | 0.0% (0/106) |
| TF-IDF + LogReg (simple) | **0.605** | **0.605** | 0.415 | 0.779 | **3.8%** (4/106) |
| Proposed (embeddings) | 0.575 | 0.587 | **0.470** | **0.809** | 14.1% (15/106) |

Two results are stated up front because they matter most:

- **The TF-IDF baseline beats the proposed classifier on intent macro-F1 (0.605 vs
  0.587).** Diagnosed in [`REPORT.md`](REPORT.md) §C.2 and
  [`DECISION_LOG.md`](DECISION_LOG.md) #7: both were trained on keyword-derived labels,
  which TF-IDF can memorise; on messages no keyword rule matched, the two are level.
- **The proposed system wrongly auto-handles 14.1% of cases that needed a human (15/106)**,
  versus 3.8% for the baseline, because it automates 40.5% of traffic instead of 22.0%.
  It is suited to drafting replies a human approves, not unsupervised auto-send.

Where it wins: retrieval finds answers closer to what Spotify actually said
(reference-recall@5 0.470 vs 0.415), escalation F1 is highest (0.809), and replies are
the best of the three reply systems (below).

### Reply quality (LLM judge, 1–5, all 200 examples)

| Reply system | Correctness | Groundedness | Consistency | Helpfulness | No unsupported claims | Overall |
|---|---:|---:|---:|---:|---:|---:|
| LLM, no evidence | 3.33 | 1.67 | 2.88 | 3.00 | 4.64 | 3.11 |
| Top precedent pasted verbatim | 3.39 | 4.82 | 4.16 | 3.06 | 4.86 | 4.06 |
| **Proposed** (LLM grounded in top-5 evidence) | **4.32** | **4.88** | **4.85** | **3.88** | **4.98** | **4.58** |

The grounded generator beats both reply baselines on every criterion. This doubles as a
**retrieval ablation**: removing retrieval collapses groundedness (4.88 → 1.67) and the
full system wins on 195/200 examples. Removing generation (pasting the top precedent
verbatim) keeps replies grounded but less correct and less helpful; the full system wins
on 110/200, ties on 71 and loses on 19. The verbatim baseline uses the proposed system's
own top precedent, so it isolates the value of the generation step. Full ablation in
[`REPORT.md`](REPORT.md) §C.6.

**Treat these as judge readings, not truth:** the judge is the same model as the
generator (`gpt-5.6-luna`), and no human has scored a reply yet (see below).

---

## Commands that require an API key (and cost money)

| Command | What it does | Cost |
|---|---|---|
| `python -m evaluation.run --limit 1` | one-example smoke test: generation + judge | ~5 calls |
| `python -m evaluation.run --limit 20` | 20-example pilot | ~100 calls |
| `python -m evaluation.run` | full 200-example evaluation incl. reply quality | 1,001 calls — **all cached, so free** unless prompts change |
| `python scripts/label_training_data.py` | re-label the training pool with an LLM | ~200 calls |

The reported full run made 1,001 LLM calls (400 generation, 601 judge): 760 served from
the cache and 241 live, in 735.5 s. Token usage and cost are not tracked by the harness.
All 1,001 responses are now cached, so these commands only cost money for prompts that
change.

Configure in `.env` (see `.env.example`). No code change is needed to switch provider.
The reported numbers used:

```
GENERATION_PROVIDER=openai
JUDGE_PROVIDER=openai
OPENAI_API_KEY=...
GENERATION_MODEL=gpt-5.6-luna
JUDGE_MODEL=gpt-5.6-luna
```

Supported providers: `openai`, `openrouter`, `gemini`, `anthropic`. Only the selected
provider's key is required. Changing provider or model changes the cache key, so it
triggers live calls.

---

## How results are cached

Every LLM response is written to `data/cache/llm/<hash>.json`, keyed by
**provider + model + system prompt + prompt + max_tokens + temperature**.

- Re-running an evaluation costs nothing for calls already made — a crashed run resumes
  where it stopped.
- Provider is in the key because the same model id is reachable both directly and via
  OpenRouter, and those are not interchangeable.
- `temperature=0` throughout, so sampling noise cannot be mistaken for improvement.
- Embeddings are cached separately in `data/cache/*.npy`.

Delete `data/cache/llm/` to force live re-runs.

---

## Golden evaluation set

**`evaluation/golden_set.jsonl`** — 200 examples, hand-labelled.

```json
{"id": "g001", "pair_id": "2948915", "customer_message": "...", "intent": "other",
 "expected_action": "ESCALATE", "difficulty": "hard", "reference_resolution": "...",
 "historical_handoff": true, "n_words": 21}
```

| Property | Value |
|---|---|
| Size | 200 |
| Intents covered | 12 of 12 (11–25 examples each) |
| Action balance | 106 ESCALATE / 94 AUTO_HANDLE |
| Difficulty | 80 easy / 78 medium / 42 hard |
| Source | held-out conversations only (start ≥ 2017-11-25) |

**Sampling and labelling protocol** (also in `scripts/sample_golden.py` and
`scripts/build_golden_set.py`):

- Stratified, not random: every intent, plus deliberate slices for messages no keyword
  rule matched (98/200), weak retrieval (30), very short (32), very long (48), and low
  classifier confidence (25).
- Labelled from the **customer message alone**. Model predictions and Spotify's actual
  reply were hidden during labelling to avoid anchoring.
- `expected_action` answers *"could a public reply grounded in precedent resolve this, or
  does it need account access?"* — it is **not** copied from what Spotify did. Spotify's
  real behaviour is stored separately as `historical_handoff`.
- Independent check: hand labels agree with Spotify's actual handoff behaviour on
  **160/200 (80%)**.
- Rebuild with `python scripts/build_golden_set.py` (validates schema, fails on bad labels).

Raw hand labels live in `evaluation/hand_labels.tsv` so they can be audited separately
from the generated file.

---

## Failure analysis

```bash
python -m evaluation.failure_analysis --top 6 --examples 2   # = evaluation/results/failure_analysis.txt
python -m evaluation.failure_analysis --json                 # machine-readable
```

Failure modes are **discovered from the data**, not hand-written: every wrong prediction
is bucketed by an observable signal, buckets are counted, and the largest are printed
with real examples. Output changes when the system changes. On the final run: intent
wrong on 85/200, action wrong on 43/200 (28 over-escalations + 15 false auto-handles).

The script counts routing failures only; it still lists reply-text modes (evidence
mismatch, invented policy) as UNMEASURED. Reply-level results from the final run are
summarised in [`REPORT.md`](REPORT.md) §D: no proposed reply scored ≤ 3 on
unsupported claims, and 19/200 scored below the verbatim precedent.

---

## Human-vs-LLM judge agreement — **pending**

The brief requires evidence that the LLM judge agrees with humans. **This has not been
done.** The judge has now scored all 200 golden examples, but no human has scored any
reply. The infrastructure is ready:

```bash
# 1. uses the judge scores already in evaluation/results/predictions.jsonl
python -m evaluation.human_judge_validation build --n 40

# 2. a human fills the human_* fields in evaluation/human_judge_validation.jsonl

# 3. offline, no API
python -m evaluation.human_judge_validation score
```

`score` reports exact agreement, within-1 agreement, and **quadratic-weighted Cohen's
kappa** per rubric criterion. Both subcommands refuse to run on missing or unfilled data
rather than estimate anything — verified.

Until this exists, treat every reply-quality number as an unvalidated instrument reading.
A judge that gives high scores is not thereby a reliable judge.

---

## Tests

```bash
python -m pytest -q     # 28 passed
```

Covers text cleaning, conversation reconstruction, the time/conversation split (including
a leakage check that the golden set never appears in training), the taxonomy, all seven
escalation rules, JSON parsing, and output schemas. Fully offline.

---

## Repository layout

```
src/            preprocessing, intents(+yaml), classifier, retrieval,
                generator, escalation, agent, llm (provider layer), embeddings
evaluation/     run.py, baselines.py, judge.py, failure_analysis.py,
                human_judge_validation.py, golden_set.jsonl, hand_labels.tsv
scripts/        download_data, prepare_data, build_index, sample_golden,
                build_golden_set, label_training_data
tests/          test_pipeline.py (28 tests)
```

`DATA_ANALYSIS.md` (brand selection) · `DECISION_LOG.md` (15 decisions) ·
`REPORT.md` (results, failure modes, limitations)

---

## Current limitations

1. **False auto-handle rate is 14.1%** (15/106; 95% CI 8.8–22.0%), versus 3.8% for the
   TF-IDF baseline. The proposed system automates more (40.5% vs 22.0%) and is riskier at
   current thresholds — suited to human-approved drafts, not unsupervised auto-send.
2. **The proposed classifier loses to the TF-IDF baseline on intent** (macro-F1 0.587 vs
   0.605). Diagnosed, not hidden: training labels came from keyword rules, so TF-IDF had
   memorised its own supervision. On the 98 examples no rule matched, the two are level
   (0.459 vs 0.469).
3. **Reply quality rests on an unvalidated LLM judge.** It has run on all 200 examples
   (proposed 4.58 vs verbatim 4.06 vs no-evidence 3.11), but no human-agreement
   measurement exists yet. See above.
4. **Generation and judge are the same model** (`gpt-5.6-luna`), so the judge may favour
   its own output. That would inflate the two LLM-written systems relative to the
   verbatim baseline, which is human-written.
5. **The system is frozen; nothing was tuned on the golden set.** Some changes score
   better on it (e.g. a 0.30 confidence floor: macro-F1 0.595) but were not applied,
   because the threshold would be read off the test set (`DECISION_LOG.md` #15).
6. **200 examples** means roughly ±3–7 percentage points of sampling uncertainty; small
   differences between systems are not significant.
7. **Data is from 2017.** The agent reproduces 2017 Spotify behaviour, not correct
   present-day advice.
8. **Conversation reconstruction is approximate** — 0.19% of reply edges are dangling and
   branching threads collapse into one component.
