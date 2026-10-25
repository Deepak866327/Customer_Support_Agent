# Decision Log

Fifteen non-obvious decisions, each one actually present in the code or the evaluation
protocol. Where a decision was later proved wrong by measurement, that is recorded rather
than quietly fixed. Numbers are from the final 200-example run (`REPORT.md`).

---

### 1. Selected SpotifyCares out of 108 brands, on answer *reusability* rather than volume

**Decision:** Score candidate brands on whether their historical replies could be reused
for a future customer, not on how many tweets they have.
**Why:** The brief requires replies grounded in how the brand historically resolved
similar issues. That is only possible if past answers stay true over time. Spotify's
answers are troubleshooting steps and policy statements ("log out, restart, log back
in"), measured at **0.0% ephemeral content** and 0.1% non-English, with 29,099 reusable
pairs — the largest of any finalist.
**Trade-off:** Gave up AmazonHelp's 144,905 substantive pairs (5× more data).

### 2. Rejected the largest brand (Amazon) and the cleanest ones (airlines and rail)

**Decision:** Eliminate AmazonHelp, AmericanAir, Delta, British_Airways, GWRHelp and
VirginTrains despite each looking attractive.
**Why:** Amazon's traffic is heavily multilingual and its dominant reply is a redirect,
not an answer (*"we'd not be able to check/access your account. Kindly drop in your
details through the link"*; 41.3% of replies contain a link). The airlines have clean
English and crisp intents, but their answers expire: *"That service is currently running
11 minutes late…"* — 11.0% of AmericanAir replies contain live-state tokens. No
historical corpus contains today's operations, so an honest system would escalate nearly
everything and the evaluation would go degenerate.
**Trade-off:** Lost the most data (Amazon) and the cleanest intent taxonomies (airlines).

### 3. Excluded "please DM us" replies from the retrieval index

**Decision:** Index only replies that are not a bare handoff and are ≥12 words
(`is_substantive` in `src/preprocessing.py`).
**Why:** A handoff is not a resolution. Indexing "please send us a DM" would let the
generator ground a reply in evidence containing no answer, producing confident,
well-cited, useless replies.
**Trade-off:** Cut the index from 43,092 pairs to 13,347. Also biases the corpus toward
publicly-resolvable cases, since what happened inside the DM is never observable.

### 4. Used Spotify's own DM-handoff behaviour as the escalation signal

**Decision:** Treat "Spotify moved this to a private message" as an observed record that
a human was required, and set each intent's escalation default from its measured handoff
rate.
**Why:** It is real behaviour rather than an invented rule, and it separates cleanly:
billing 90.3%, login 83.5%, student/family 82.7%, hacked 70.6% versus content-missing
19.6% and feature-request 14.9%. The overall rate of 43.3% means the escalation target is
near-balanced rather than degenerate.
**Trade-off:** It encodes Spotify's 2017 staffing policy, not an objective truth about
what *should* be automated. It is also the source of the largest over-escalation bucket:
`intent_requires_human` decided 78/200 cases, including answerable student-pricing
questions (failure mode 2).

### 5. Derived 12 intents from the brand's own messages instead of importing a taxonomy

**Decision:** TF-IDF + KMeans (k=14) over 26,481 opening customer messages, then merge
and name the clusters by hand into 11 intents plus `other`.
**Why:** An imported taxonomy (e.g. Banking77) describes someone else's business. The
clusters that emerged — login, billing, family plan, student discount, account
compromise, content unavailable, app crash, playlist/library, playback, feature request —
are what Spotify's customers actually write about.
**Trade-off:** 12 classes over 200 golden examples means 11–25 examples per class, so
per-class F1 carries wide error bars.

### 6. Treated `other` as an abstention, not a trainable class

**Decision:** Messages that no rule matched are dropped from training. `other` is emitted
at inference when no intent clears a 0.45 confidence floor.
**Why:** 57.5% of opening messages matched no keyword rule, but sampling them showed they
mostly *do* have a clear intent that keywords cannot see (*"I thought Liam's new song was
coming out today, can't see it on Spotify"*). Training a literal `other` class on them
would teach the model that most traffic is intentless and hand the trivial baseline a
fake 57.5% accuracy.
**Trade-off:** The classifier can never *positively* identify a genuinely intentless
message; it can only fail to be confident about one. Measured cost: 15/200 abstentions on
messages that had a nameable intent (failure mode 4).

### 7. Reported that the TF-IDF baseline beats the proposed classifier, with the diagnosis

**Decision:** Publish that TF-IDF **beats** the proposed system on intent (macro-F1 0.605
vs 0.587, accuracy 0.605 vs 0.575) rather than re-engineering until it didn't.
**Why:** Splitting the golden set by whether a keyword rule fired explains it: TF-IDF
scores 0.745 where rules fired versus 0.676 for embeddings, but on the 98 messages rules
never matched the two are level (0.459 vs 0.469). TF-IDF was not better — it had learned
the rules that generated its own training labels, which agree with humans only 72.5% of
the time. The labels, not the model, were the bottleneck.
**Trade-off:** The headline intent number is worse than a baseline. The real fix —
re-labelling the training pool with an LLM (`scripts/label_training_data.py`) — was left
for future work rather than rushed in before the final run.

### 8. Split by conversation and by time, not randomly by tweet

**Decision:** Conversations starting before 2017-11-25 train; later ones evaluate.
**Why:** Two tweets from one thread share near-identical wording, so a random split lets
the classifier and the retriever see the evaluation thread during training. A time
boundary on the conversation's first timestamp also stops a thread straddling the split.
A test asserts the golden set never appears in training.
**Trade-off:** The evaluation window is a single busy fortnight, so any seasonal effect
(Taylor Swift's *Reputation* release, Christmas) is over-represented.

### 9. Chose frozen embeddings plus logistic regression over fine-tuning or k-NN

**Decision:** `all-MiniLM-L6-v2` embeddings into logistic regression.
**Why:** Logistic regression gives probability estimates the escalation layer needs as a
confidence signal; k-NN over noisy weak labels would amplify seed noise. Fine-tuning was
rejected because there is no clean labelled set large enough to justify it — the labels
were the known weak point (#7), so training harder on them would not help. The model runs
on a laptop CPU with no GPU.
**Trade-off:** Leaves accuracy on the table if good labels ever become available.

### 10. Made intent a soft re-ranking signal in retrieval, not a hard filter

**Decision:** Retrieve top-60 by cosine, then sort so the predicted intent comes first,
and return 5.
**Why:** Retrieving only within a *wrongly* predicted intent is worse than retrieving
broadly: the escalation layer can detect inconsistent evidence, but it cannot detect
evidence that was filtered away before it ever saw it. This is why retrieval stays
valuable despite the weak classifier: dense retrieval still beats lexical on
reference-recall at every k (0.470 vs 0.415 at k=5).
**Trade-off:** Top-5 can contain off-intent neighbours, which lowers `intent_recall@k`
(0.630 vs TF-IDF's 0.715 at k=5).

### 11. Chose k=5 for retrieval

**Decision:** Five precedents per query.
**Why:** Enough for the agreement signal the escalation layer needs (a 3–2 split is
detectable; a 1–1 split is not), while short enough to stay inside a prompt without
crowding out the customer's actual message.
**Trade-off:** Not tuned by sweep — k was fixed by the needs of the escalation rule, and
`evidence_agreement` thresholds were then set against it.

### 12. Made escalation rule-based instead of a learned risk model

**Decision:** Seven ordered conditions; the first to fire decides and supplies the reason.
**Why:** The brief requires a *stated reason*, and an opaque score cannot give one. With
200 golden examples there is also nothing to train a calibrated risk model on. Rules are
ordered by cost of the mistake, not frequency, so safety checks run before convenience
ones.
**Trade-off:** Rules inherit every classifier error. Measured cost: a **14.1%** false
auto-handle rate (15/106), of which 3 are sensitive intents misread as safe ones, so the
intent rule never fired (`evaluation/failure_analysis.py`, mode 6). The cheapest fix — an
intent-independent phrase check — is the first item in the one-more-week plan.

### 13. Required the generator to cite `evidence_ids`, and validated them

**Decision:** The reply schema is `{reply, confidence, evidence_ids}`, and cited ids are
filtered against the ids actually retrieved (`src/generator.py`).
**Why:** Without citations, "grounded in historical resolutions" is unfalsifiable. With
them, any reply can be traced to the precedents it used, and a model that invents an id
is caught automatically rather than believed. In the final run, 197/200 replies cite at
least one valid precedent.
**Trade-off:** Citation presence proves the model *pointed at* evidence, not that the
reply is faithful to it. Faithfulness still needs the judge — and the judge needs humans.

### 14. Externalised provider/model config, cached every LLM call, and accepted a same-model judge

**Decision:** `GENERATION_PROVIDER` / `JUDGE_PROVIDER` / `*_MODEL` come from `.env`;
provider and model are part of the cache key; every response is cached to disk by
content hash. The final run used `gpt-5.6-luna` for both generation and judging.
**Why:** The project ran on several providers during development (Anthropic → Gemini →
OpenRouter free tier → OpenAI), each time without touching pipeline code. Free tiers
could not complete a 1,000-call run; the final run needed a paid model. Caching made that
run resumable after it was interrupted — 740 responses carried over, 241 were fetched
live — and makes it re-runnable by a reviewer with no API key. Both roles used the
OpenAI provider default rather than a second, separately-configured judge model.
**Trade-off:** A same-model judge risks self-preference, which would favour LLM-written
replies over the human-written verbatim baseline. This is reported as a limitation, and
re-judging with a different model family (~600 calls, generation stays cached) is on the
one-more-week list.

### 15. Froze the system before the final run and did not tune on the golden set

**Decision:** Treat the 200-example run as final. Thresholds (0.45 confidence floor, 0.55
similarity floor, k=5) were not changed after seeing golden-set results, even where a
change scored better.
**Why:** Lowering the confidence floor to 0.30 raises macro-F1 from 0.587 to 0.595 on the
golden set — but that threshold would be read off the test set, making the number
optimistic. The same applies to rebalancing escalation to cut the 14.1% false
auto-handle rate. A smaller honest number is worth more than a larger contaminated one.
**Trade-off:** The reported system is knowingly not the best-scoring configuration. The
improvements are listed in `REPORT.md` §F with the protocol they need (a dev split from
the training period).
