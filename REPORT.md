# AI Customer Support Agent for SpotifyCares — Report

**Final results.** Every number below comes from the final 200-example run
(`evaluation/results/metrics.json`, per-example detail in `predictions.jsonl`). The system
was frozen before that run and nothing was tuned on its output. Reproduce with the
existing LLM cache — see the README.

| | |
|---|---|
| Golden set | 200 hand-labelled held-out messages · 12 intents · 106 ESCALATE / 94 AUTO_HANDLE · 80 easy / 78 medium / 42 hard |
| LLM | `gpt-5.6-luna` (OpenAI) for both reply generation and judging, temperature 0 |
| LLM calls | 1,001 (400 generation + 601 judge): 760 cache hits, 241 live · all now cached |
| Runtime | 735.5 s end-to-end · all 200 examples completed |
| Reproducibility | cache-only re-run: 14.6 s, 1,001 hits, 0 live calls, identical metrics and byte-identical predictions |
| Tokens / cost | not tracked by the harness |

### Summary in five lines

1. **Replies are the clear win.** Grounded replies score **4.58/5** vs 4.06 for pasting the
   closest past answer and 3.11 for the same LLM without retrieval. They beat the no-evidence
   reply on 195/200 examples.
2. **Intent classification is not a win.** The simple TF-IDF baseline **beats** the proposed
   classifier on macro-F1 (**0.605 vs 0.587**) and accuracy (0.605 vs 0.575).
3. **Escalation is better on F1 (0.809 vs 0.779) but riskier.** The proposed system wrongly
   auto-handles **14.1%** of cases that needed a human, versus **3.8%** for the baseline.
4. **Retrieval finds answers closer to what Spotify actually said** (reference-recall@5
   0.470 vs 0.415), and it is what makes replies grounded and escalation explainable.
5. **Reply scores are unvalidated judge readings.** The judge is the same model as the
   generator and has not been checked against human scores.

---

## A. Problem framing

Spotify's Twitter support handles two kinds of message that look alike and must be
treated differently:

- **Reusable problems** — "why isn't this album on Spotify?", "the app keeps crashing".
  The correct answer is the same for every customer and does not depend on their account.
- **Account-specific problems** — "I was charged twice", "I can't log in", "my account
  was hacked". These require opening the account.

The system separates these, drafts a reply for the first group grounded in what Spotify
actually said before, and routes the second group to a human with a stated reason.

**The costs are asymmetric.** Escalating an easy question wastes an agent's minute.
Auto-answering a billing dispute or a hacked account loses a customer or creates a legal
problem. So the metric that matters most is not accuracy — it is the **false auto-handle
rate**.

**Why Spotify** (from `DATA_ANALYSIS.md`): of 108 brands, most do not actually answer on
Twitter — TMobileHelp deflects to DM 82% of the time, comcastcares 72%, AppleSupport 53%.
Their history cannot ground anything. Amazon is the largest but is multilingual and mostly
redirects to a webform (41% of replies contain a link). Airlines answer well but their
answers expire ("running 11 minutes late"), so nothing is reusable. Spotify has 29,099
reusable pairs, 0.0% ephemeral content, 0.1% non-English — and its own DM-handoff
behaviour supplies an *observed* escalation signal (billing 90.3% handoff vs feature
requests 14.9%, overall 43.3%, so the target is near-balanced rather than degenerate).

---

## B. Baselines

Every component is compared against something simpler, run on exactly the same data.

| Component | Trivial baseline | Simple baseline | Proposed |
|---|---|---|---|
| Intent | Majority class | TF-IDF + logistic regression | MiniLM embeddings + logistic regression |
| Retrieval | — | TF-IDF cosine over the same index | Dense cosine, intent as soft re-rank |
| Escalation | Same 7 rules, fed by each classifier's predictions | ← | ← |
| Reply | LLM with **no** evidence | Closest precedent pasted **verbatim** (no LLM) | LLM grounded in top-5 precedents |

The two intent classifiers were trained on identical weakly-labelled data. The verbatim
reply baseline pastes the proposed system's own top precedent, so the gap between it and
the proposed reply measures the generation step alone.

---

## C. Results

### C.1 Headline comparison

| System | Intent acc | Intent macro-F1 | Retrieval ref-recall@5 | Escalation F1 | False auto-handle | Auto-handled |
|---|---:|---:|---:|---:|---:|---:|
| Trivial (majority class) | 0.070 | 0.011 | — | 0.693 | 0.0% (0/106) | 0.0% |
| Simple (TF-IDF + LogReg) | **0.605** | **0.605** | 0.415 | 0.779 | **3.8%** (4/106) | 22.0% |
| **Proposed** (embeddings + LogReg) | 0.575 | 0.587 | **0.470** | **0.809** | 14.1% (15/106) | 40.5% |

### C.2 Intent — the TF-IDF baseline wins

**The simple TF-IDF baseline beats the proposed classifier on intent macro-F1 (0.605 vs
0.587), accuracy (0.605 vs 0.575) and weighted F1 (0.589 vs 0.574).** Per class it wins 7
of 12 intents; the proposed classifier wins 4 and ties 1.

| Intent | Support | TF-IDF F1 | Proposed F1 |
|---|---:|---:|---:|
| family_plan | 14 | **0.966** | 0.933 |
| student_discount | 15 | **0.933** | 0.839 |
| login_access | 21 | 0.750 | **0.757** |
| account_compromised | 14 | 0.690 | **0.733** |
| content_unavailable | 19 | 0.667 | 0.667 |
| billing_charge | 12 | **0.720** | 0.643 |
| subscription_manage | 23 | **0.667** | 0.615 |
| app_crash_performance | 14 | **0.636** | 0.533 |
| playlist_library | 14 | 0.381 | **0.467** |
| playback_error | 11 | **0.519** | 0.429 |
| feature_request_feedback | 25 | **0.333** | 0.267 |
| other | 18 | 0.000 | **0.167** |

**Why the baseline wins — diagnosed, not excused.** Training labels came from keyword
rules. Splitting the golden set by whether a rule fires on the message:

| Slice | TF-IDF acc | Proposed acc |
|---|---:|---:|
| Rule fired (102 examples) | **0.745** | 0.676 |
| Rule never fired (98 examples) | 0.459 | **0.469** |

TF-IDF was not better at understanding messages — it had learned the keywords that
generated its own supervision. On messages the rules never saw, the two are level. The
rules fired on 51% of the golden set and agreed with the human label only **72.5%** of the
time; that is the ceiling both classifiers were trained against. The label source, not the
model, is the bottleneck (decision log #7).

### C.3 Retrieval

Two recall definitions, because neither alone is honest:

- **intent_recall@k** — does the top-k contain a precedent of the gold intent? Partly
  circular: 66% of index intent tags were assigned by the classifier being evaluated.
- **reference_recall@k** — does the top-k contain a past Spotify reply semantically close
  (cosine ≥ 0.50) to the reply Spotify actually sent for *this* message? Independent of the
  intent model, and closer to what the generator needs.

| Metric | TF-IDF retrieval | Proposed (dense) |
|---|---:|---:|
| reference_recall@1 | 0.190 | **0.265** |
| reference_recall@3 | 0.345 | **0.390** |
| reference_recall@5 | 0.415 | **0.470** |
| intent_recall@1 | 0.390 | **0.580** |
| intent_recall@3 | **0.635** | 0.595 |
| intent_recall@5 | **0.715** | 0.630 |

Dense retrieval finds historical answers closer to what Spotify actually said at every k.
By intent_recall@3/5 lexical retrieval is ahead, which says the two retrievers fail on
different messages (failure mode 3).

### C.4 Escalation — and the 14.1% problem

| System | Precision | Recall | F1 | False auto-handle rate | Auto-handle share |
|---|---:|---:|---:|---:|---:|
| Majority (escalate all) | 0.530 | 1.000 | 0.693 | 0.0% (0/106) | 0.0% |
| TF-IDF | 0.654 | 0.962 | 0.779 | **3.8% (4/106)** | 22.0% |
| Proposed | **0.765** | 0.859 | **0.809** | **14.1% (15/106)** | 40.5% |

> **Limitation, stated plainly: the proposed system auto-handles 14.1% of the messages
> that needed a human — 15 of 106, nearly four times the TF-IDF baseline's 3.8%.**
> 95% Wilson intervals are 8.8–22.0% vs 1.5–9.3%, so this is not sampling noise. It is
> the price of automating 40.5% of traffic instead of 22.0%. At this rate the system is
> suitable for drafting replies a human approves, **not for unsupervised auto-send.**

Which rule decided each case (proposed): `confident_and_grounded` 81 (the auto-handles),
`intent_requires_human` 78, `intent_abstained` 18, `weak_retrieval` 17,
`high_risk_language` 4, `conflicting_evidence` 2.

### C.5 Reply quality (LLM judge, 1–5, all 200 examples)

| Reply system | Correctness | Groundedness | Consistency | Helpfulness | No unsupported claims | **Overall** |
|---|---:|---:|---:|---:|---:|---:|
| LLM, no evidence | 3.33 | 1.67 | 2.88 | 3.00 | 4.64 | 3.11 |
| Closest precedent verbatim | 3.39 | 4.82 | 4.16 | 3.06 | 4.86 | 4.06 |
| **Proposed** (LLM + top-5 evidence) | **4.32** | **4.88** | **4.85** | **3.88** | **4.98** | **4.58** |

The judge does not see which system wrote a reply. Scores are aggregated over all 200
examples, including the 119 the system would escalate (for those the draft goes to a
human). Auto-handled cases alone score about the same (proposed 4.59 on the 81).

### C.6 Retrieval ablation

Removing retrieval, or removing generation, from the reply path:

| Configuration | Retrieval | Generation | Overall | Groundedness | Correctness | Helpfulness |
|---|:---:|:---:|---:|---:|---:|---:|
| Generation only | — | ✓ | 3.11 | 1.67 | 3.33 | 3.00 |
| Retrieval only (top-1 verbatim) | ✓ | — | 4.06 | 4.82 | 3.39 | 3.06 |
| **Retrieval + generation** | ✓ | ✓ | **4.58** | **4.88** | **4.32** | **3.88** |

Per example, the full system beats generation-only on **195/200** (4 ties, 1 loss) and
beats verbatim retrieval on **110/200** (71 ties, 19 losses).

Split by how similar the closest precedent was:

| Top-1 precedent similarity | n | Generation only | Retrieval only | Retrieval + generation |
|---|---:|---:|---:|---:|
| < 0.55 (weak) | 36 | 3.21 | 3.74 | 4.47 |
| 0.55–0.70 | 86 | 3.11 | 3.92 | 4.55 |
| ≥ 0.70 (strong) | 78 | 3.05 | 4.35 | 4.66 |

Pasting a precedent is only as good as the match: verbatim quality climbs from 3.74 to
4.35 as similarity rises, while generation-only is flat. Generation adds most where
retrieval is weakest (+0.73 over verbatim) and least where it is strongest (+0.31).
*Caveat:* 4.47 on weak retrieval is either the generator adapting loose precedents well
or the judge being lenient on fluent text — only human scores can separate the two.

*(Per-example counts and the similarity split are computed from `predictions.jsonl`
— `systems.<name>.judge.mean` and `evidence[0].similarity` — not stored in `metrics.json`.)*

### C.7 Why historical retrieval is still valuable

The intent classifier underperforms its baseline, but retrieval does not depend on it
being right — predicted intent is only a soft re-ranking signal (decision log #10). The
evidence that retrieval earns its place:

1. **It is what makes replies grounded.** Without it, the same model scores 1.67 on
   groundedness and 160/200 replies score ≤ 2; the judge flags an unsupported claim
   (score ≤ 3) in 20/200. With it, groundedness is 4.88 and **zero** replies score ≤ 3 on
   unsupported claims.
2. **It retrieves what Spotify actually said.** Reference-recall beats lexical retrieval at
   k = 1, 3 and 5 (0.470 vs 0.415 at k = 5).
3. **It is a safety signal independent of the classifier.** 19 cases were escalated
   because retrieval was weak (17) or the precedents disagreed (2). None of the 81
   auto-handles happened without at least 2 precedents, a closest match ≥ 0.55, ≥ 60%
   intent agreement, and a historical handoff share ≤ 50%.
4. **It makes every reply auditable.** 197/200 proposed replies cite at least one
   retrieved `evidence_id`, validated against what was actually retrieved.
5. **It supplies brand-specific facts the model has no other source for.** For `g023` the
   top precedent states that the Hulu plan comes with the student discount — 2017 Spotify
   policy that lives in the history, not in a general-purpose model.

---

## D. Top 5 failure modes

Discovered by `python -m evaluation.failure_analysis --top 6`, which buckets every wrong
prediction by an observable signal and counts. Not hand-picked; regenerated from the
final predictions (output identical to `evaluation/results/failure_analysis.txt`). Of 200
examples, intent is wrong on 85 and action on 43 (28 over-escalations + 15 false
auto-handles).

### 1. Confidently wrong intent despite strong, agreeing evidence — 49/200 (24.5%)

**Example (`g002`, medium):** *"I somehow managed to disable my spotify account how do i
fix this?"*
- **Labelled:** `login_access` / ESCALATE
- **System:** `account_compromised` (confidence 0.52) / ESCALATE
- **Closest precedent** (sim 0.74): *"My piece of shit account was hacked…"*

**Why:** The retriever found consistent neighbours and the classifier still picked an
adjacent intent. Keyword-derived labels cannot separate semantically adjacent pairs, and
the confusion matrix shows it is systematic: `feature_request_feedback → playlist_library`
9 times, `subscription_manage → billing_charge` 4, `content_unavailable →
feature_request_feedback` 4.
**Fix:** Re-label the training pool with an LLM instead of regexes; merge intent pairs the
confusion matrix shows are not separable.
*Mitigating:* here the wrong intent still produced the right action, because both intents
escalate.

### 2. Over-escalation — 28/200 (14.0%)

**Example (`g023`, medium):** *"I see that if you're a student you can get spotify and hulu
for 4.99 I'm not a student so i pay the full 9.99 can I also get hulu with spotify for the
one price, and if so how?"*
- **Labelled:** `subscription_manage` / **AUTO_HANDLE** — a general pricing question
- **System:** `student_discount` (confidence 0.98) / ESCALATE via `intent_requires_human`
- **Closest precedent** (sim 0.87) is a near-identical question Spotify answered publicly.

**Why:** The intent-level default overrode good evidence. The rule fires on the *topic*
(student) rather than on whether the question needs account access.
**Fix:** Let evidence quality decide for more intents, as `subscription_manage` already
does. `intent_requires_human` alone decided 78 of 200 cases.

### 3. Wrong intent where retrieval was also weak — 19/200 (9.5%)

**Example (`g017`, medium):** *"Do you all have an educational out reach? I have a project I
would like my students to do…"* → predicted `student_discount` (0.53), closest precedent
only 0.50.

**Why:** The message sits in a sparse region of embedding space — closest precedent below
the 0.55 floor — so neither classifier nor retriever had an anchor. Typically short,
slangy, or off-distribution messages.
**Fix:** Fall back to lexical retrieval when dense similarity is low. The data supports it:
TF-IDF scores higher `intent_recall@5` (0.715 vs 0.630), so the two fail on different
messages.

### 4. Abstained on an answerable message — 15/200 (7.5%)

**Example (`g177`, hard):** *"'The next 30 minutes are ad free…' \*5 minutes later\* 'The next
30 minutes are ad free'….go away"* → `other` (confidence 0.38 < 0.45) / ESCALATE. The
closest precedent (sim 0.78) is word-for-word the reply Spotify actually sent.

**Why:** No intent cleared the 0.45 confidence floor, so the classifier abstained. Safe,
but a miss — the retriever had the answer.
**Not fixed on purpose:** a 0.30 floor scores macro-F1 0.595 vs 0.587 on this golden set,
but that threshold was read off the test set, so it was not applied (decision log #15).

### 5. Unsafe auto-handle — 15/200 (12 general + 3 account-sensitive)

These are the 15 false auto-handles behind the 14.1% rate.

**Most serious (`g079`, easy):** *"I have a premium account that I can't listen to! My login
defaults to my facebook account which was a free account. I need to disassociate my FB
account from my Spotify account. Can you assist?"*
- **Labelled:** `login_access` / ESCALATE
- **System:** `subscription_manage` (confidence **0.48**, barely over the floor) /
  **AUTO_HANDLE** via `confident_and_grounded`

**Why:** A login problem was misread as a subscription question, so the escalation layer
never saw the rule that would have forced a human. **Escalation inherits every classifier
error on the sensitive intents** — the system must be right twice. The other 12 are
multi-issue or already-escalated messages whose precedents looked confidently answerable
(e.g. `g065`, a card-country problem that needed account access).
**Fix:** A message-level safety net independent of predicted intent: if the raw text
matches account-specific patterns (charged, refund, password, hacked, log in), escalate
regardless of intent.

**Reply-text failures are rare by comparison.** The proposed reply never scored ≤ 3 on
`no_unsupported_claims`, scored ≤ 2 on correctness in 7/200, and scored below the
verbatim precedent in 19/200. The typical loss is a generic re-ask — `g072`, where the
customer had already given the details the reply asks for. Even the 15 false
auto-handles got well-rated replies (mean 4.59): the danger there is the routing, not the
wording.

---

## E. What is misleading about my headline number?

The headline is **intent macro-F1 0.587, escalation F1 0.809, reply quality 4.58/5**.
Here is what it does not prove.

**1. It is not the best intent number available.** The simple TF-IDF baseline scores 0.605.
Quoting 0.587 without that comparison would be dishonest.

**2. Escalation F1 hides the mistake that matters.** F1 weighs both errors equally; they are
not equal. The proposed system has the best F1 *and* a **14.1%** false auto-handle rate,
nearly four times the baseline's 3.8%. It wins F1 by automating more (40.5% vs 22.0%).
A system that escalated everything would score F1 0.693 with a *perfect* 0% false
auto-handle rate. **F1 is the wrong headline for this problem.**

**3. 4.58/5 is a judge reading, not a measurement of correctness.**
- The judge is **the same model** as the generator (`gpt-5.6-luna`), so self-preference
  would favour the two LLM-written replies over the human-written verbatim precedent. The
  +0.52 gap over verbatim is the number most in need of human confirmation.
- The judge sees the precedents, so "groundedness" partly rewards restating the evidence,
  whether or not that evidence fit the customer.
- No human has scored a reply, so the judge's agreement with humans is unknown.
- The scale is compressed at the top (unsupported claims 4.98, groundedness 4.88), so the
  *ranking* of systems is more trustworthy than the absolute level.

**4. It is inflated by easy examples.** Difficulty was labelled by hand *before* seeing
predictions:

| Difficulty | n | Intent accuracy | Escalation accuracy |
|---|---:|---:|---:|
| Easy | 80 | 78.7% | 90.0% |
| Medium | 78 | 47.4% | 67.9% |
| Hard | 42 | 35.7% | 76.2% |

Intent accuracy more than halves from easy to hard. The aggregate describes the mix of the
sample as much as the system.

**5. Macro-F1 over 12 classes with 200 examples is noisy.** Classes hold 11–25 examples,
so one flipped prediction moves a per-class F1 by ~5 points. The 0.587-vs-0.605 gap is 6
examples of accuracy and should not be read as a ranking without a significance test —
in either direction.

**6. Retrieval recall is definition-dependent.** By `reference_recall@5` the proposed
system wins (0.470 vs 0.415); by `intent_recall@5` it **loses** (0.630 vs 0.715). Both are
reported for that reason.

**7. Labels have a known ceiling.** Weak-labelling rules were right 72.5% of the time where
they fired. Every trained model inherits that.

**8. It is 2017 Spotify.** High scores mean faithfully imitating 2017 policy, not giving
correct advice today.

---

## F. What I would do with one more week

Ordered by expected value per day, not novelty. None of these changes the architecture.

1. **Intent-independent safety net (1 day).** Escalate on account-specific phrases
   regardless of predicted intent. Directly targets the 3 account-sensitive false
   auto-handles, the most expensive failure. Then re-measure the false auto-handle rate.
2. **Human validation of the judge (1–2 days, no API cost).** Hand-score 40 replies with
   `human_judge_validation build/score` and report quadratic-weighted kappa. Until this
   exists, the reply-quality result is an instrument reading.
3. **Re-judge with a different model family (half a day, ~600 calls).** Removes the
   self-preference caveat; generation stays cached, so only judge calls are new.
4. **Re-label ~4,000 training messages with an LLM (1–2 days, ~200 calls).**
   `scripts/label_training_data.py` is written. Attacks the 72.5% label ceiling that
   explains the intent result.
5. **Evidence-based escalation for topic-level defaults (1–2 days).** Replace blanket
   `student_discount` / `family_plan` escalation with evidence checks, targeting the 28
   over-escalations. Tune thresholds on a dev split from the training period, **not** on the
   golden set, and accept a change only if the false auto-handle rate does not rise.
6. **Lexical fallback below the similarity floor (1 day).** Motivated by TF-IDF's higher
   `intent_recall@5`.
7. **Grow the golden set to ~500 with a second labeller**, for inter-annotator agreement
   and bootstrap confidence intervals on every headline number.

Deliberately *not* on this list: fine-tuning, a vector database, an agent framework, or a
redesign. The bottleneck is label quality and evaluation coverage, not architecture.

---

## G. Honest limitations

1. **14.1% false auto-handle rate** (15/106; 95% CI 8.8–22.0%), versus 3.8% for the TF-IDF
   baseline. Not safe for unsupervised auto-send at current thresholds.
2. **The proposed classifier is worse than a simple baseline on intent** (macro-F1 0.587 vs
   0.605), because both were trained on keyword-derived labels.
3. **Reply quality is measured only by an LLM judge** that is the same model as the
   generator, and whose agreement with humans has not been measured.
4. **200 examples, single labeller.** No inter-annotator agreement. Hand labels agree with
   Spotify's actual handoff behaviour 80% of the time — a sanity check, not a validation.
5. **Some thresholds were set by judgement, not tuned on held-out data** (0.45 confidence
   floor, 0.55 similarity floor, k = 5).
6. **Conversations are truncated at the DM boundary** — we observe *that* a human took over,
   never what they did. The corpus is biased toward publicly-resolvable cases.
7. **The evaluation window is one fortnight** of late 2017, so seasonal effects (a major
   album release, Christmas) are over-represented.
8. **Thread reconstruction is approximate** — 0.19% dangling reply edges; branching threads
   collapse into one conversation.
9. **Token usage and cost were not instrumented**; only call counts are recorded.
