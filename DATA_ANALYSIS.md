# Data Analysis — Brand Selection

Everything below is measured from the raw Kaggle file `data/raw/twcs.csv`
(`thoughtvector/customer-support-on-twitter`). The measurements are reproduced by
`python scripts/analyze_brands.py`.

## 1. Dataset shape

| Property | Value |
|---|---|
| Rows (tweets) | 2,811,774 |
| Inbound (customer) tweets | 1,537,843 |
| Outbound (brand) tweets | 1,273,931 |
| Distinct brand accounts | 108 |
| Reply edges resolvable in-file | 2,013,577 |
| Dangling `in_response_to_tweet_id` (parent not in file) | 3,862 (0.19%) |
| Reconstructed conversations (connected components) | 798,197 |

**Schema:** `tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id`.

**Conversation reconstruction:** the dataset is a reply *forest*, not a table of threads.
I build an undirected graph over `(tweet_id, in_response_to_tweet_id)` pairs and take
connected components as conversations (union-find). This is more robust than walking
`response_tweet_id` forward, because that column is a comma-separated multi-value field and
threads branch. 0.19% of edges point at tweets absent from the dump; those tweets simply
start their own component. **This is an accepted, documented imperfection — I do not attempt
perfect thread reconstruction.**

## 2. Why brand choice is not arbitrary

The assignment needs replies *grounded in how the brand historically resolved similar issues*.
That requirement kills most brands, because a large share of "support" tweets are not
resolutions at all — they are deflections ("please DM us"). A brand can look great on volume
and be worthless as a grounding corpus.

So I scored the top 20 brands on four things, not just size:

- **`deflect_pct`** — share of brand replies that only ask the customer to move to DM/private
  message. High = no reusable resolution content.
- **`substantive_pairs`** — customer→brand reply pairs where the reply is not a deflection and
  is >= 12 words. This is the actual usable retrieval corpus size.
- **`foreign_cust_pct`** — share of customer messages that are not predominantly Latin script.
  A multilingual corpus breaks a single small English pipeline.
- **`ephemeral_pct`** — share of replies containing live-state references (record locator,
  "running 11 min late", flight numbers, order numbers). These replies **cannot be reused**
  for a future customer, so they are poison as retrieved evidence.

### Top brands by outbound volume

| Brand | Convs | Cust msgs | Brand msgs | Deflect % | Substantive pairs | Substantive % | Link % |
|---|---:|---:|---:|---:|---:|---:|---:|
| AmazonHelp | 82,556 | 203,598 | 169,840 | 0.6 | 144,905 | 85.8 | 41.3 |
| AppleSupport | 80,717 | 131,764 | 106,860 | **52.5** | 48,361 | 45.3 | 75.4 |
| Uber_Support | 41,923 | 72,154 | 56,270 | 35.9 | 34,785 | 61.9 | 51.3 |
| SpotifyCares | 28,280 | 48,543 | 43,265 | 30.8 | 29,100 | 67.5 | 50.5 |
| Tesco | 16,722 | 34,228 | 38,573 | 26.8 | 27,136 | 70.5 | 8.3 |
| AmericanAir | 26,386 | 50,054 | 36,764 | 16.8 | 26,576 | 72.7 | 6.5 |
| Delta | 26,168 | 45,296 | 42,253 | 16.5 | 25,747 | 61.1 | 15.3 |
| British_Airways | 16,452 | 31,187 | 29,361 | 14.0 | 22,596 | 77.1 | 6.8 |
| hulu_support | 14,955 | 26,807 | 21,872 | 0.5 | 21,531 | 99.3 | 54.8 |
| SouthwestAir | 21,636 | 35,370 | 28,977 | 17.0 | 20,987 | 72.8 | 16.2 |
| XboxSupport | 13,455 | 32,395 | 24,557 | 20.9 | 17,525 | 75.4 | 40.4 |
| VirginTrains | 14,853 | 37,832 | 27,817 | 2.6 | 17,132 | 62.5 | 13.2 |
| GWRHelp | 10,732 | 27,278 | 19,364 | 1.5 | 16,424 | 85.4 | 7.0 |
| sainsburys | 10,934 | 23,710 | 19,466 | 23.1 | 12,476 | 64.3 | 30.6 |
| Ask_Spectrum | 18,532 | 33,252 | 25,860 | **49.5** | 11,920 | 46.5 | 39.4 |
| AskPlayStation | 12,533 | 23,954 | 19,098 | 25.3 | 11,355 | 60.8 | 36.6 |
| sprintcare | 13,560 | 30,212 | 22,381 | **48.0** | 9,916 | 44.6 | 17.3 |
| ChipotleTweets | 14,392 | 23,084 | 18,749 | 1.0 | 9,133 | 49.1 | 27.8 |
| comcastcares | 24,063 | 39,579 | 33,031 | **71.5** | 8,279 | 25.1 | 4.0 |
| TMobileHelp | 22,820 | 47,158 | 34,317 | **82.1** | 5,194 | 15.2 | 48.2 |

**Immediately eliminated:** TMobileHelp (82% deflection), comcastcares (72%), AppleSupport
(53%), Ask_Spectrum (50%), sprintcare (48%). Their historical "resolutions" are overwhelmingly
"DM us", so there is nothing to ground a reply in.

### Finalist screen (the metrics that actually matter)

| Brand | Pairs | Foreign cust % | Deflect % | Ephemeral % | **Reusable evidence** | Reusable % |
|---|---:|---:|---:|---:|---:|---:|
| **SpotifyCares** | 43,092 | **0.1** | 30.8 | **0.0** | **29,099** | 67.5 |
| Tesco | 38,468 | 0.0 | 26.9 | 2.1 | 27,005 | 70.2 |
| AmericanAir | 36,531 | 0.0 | 16.8 | 11.0 | 26,035 | 71.3 |
| hulu_support | 21,681 | 0.0 | 0.5 | 0.0 | 21,528 | 99.3 |
| XboxSupport | 23,235 | 0.1 | 21.5 | 0.1 | 17,518 | 75.4 |
| GWRHelp | 19,237 | 0.0 | 1.5 | 5.5 | 15,444 | 80.3 |
| AskPlayStation | 18,675 | 0.0 | 27.1 | 0.0 | 11,333 | 60.7 |

### Why AmazonHelp was rejected despite being 5x larger

AmazonHelp has the most substantive pairs by far (144,905), and I still rejected it. Sampling
real pairs shows why:

- **It is multilingual.** Large volumes of Japanese and Indian-market traffic. Handling that
  properly needs multilingual embeddings and multilingual judging — real work that adds nothing
  to the thing being evaluated.
- **Its replies are soft deflections.** The dominant reply pattern is *"Since Twitter is a
  social media platform, we'd not be able to check/access your account. Kindly drop in your
  details through the link"* and *"reach out to us here: https://t.co/…"*. 41.3% of replies
  contain a link. The historical "resolution" is usually a redirect, not an answer.
- Its intent surface (retail orders, Prime, Kindle, Alexa, Fresh, AWS, seller disputes) is so
  broad that a 10–15 intent taxonomy would be arbitrary.

### Why the airlines and rail operators were rejected

AmericanAir, Delta, British_Airways, SouthwestAir, GWRHelp and VirginTrains have clean English
and low deflection, and their intent taxonomies are crisp. But their answers are **not
reusable**, which is fatal for a retrieval-grounded agent:

> C: *"@GWRHelp Was planning on getting 1110 to Portsmouth (bath spa)"*
> B: *"That service is currently running 11 minutes late but is expected to reach Filton at 11:18. Lewis"*

> C: *"…every day the service seems to get worse"*
> B: *"Please follow and DM your record locator and details so we can look into this for you."*

11.0% of AmericanAir replies and 5.5% of GWRHelp replies contain live-state tokens. The correct
answer depends on today's operational state, which no historical corpus contains. An honest
system would escalate nearly everything, and the evaluation would be degenerate.

### Why hulu_support was rejected

The cleanest corpus in the dataset (99.3% substantive, 0.5% deflection) — but the replies are
mostly empathy and acknowledgement rather than resolution (*"That's really a great idea, Aaron!
We'll be sure to pass this along to our developers"*), and at 21.5k pairs across a narrow
domain (billing, login, playback, catalog) the taxonomy struggles to reach 8 distinct intents
with adequate support. It was the runner-up.

## 3. Recommendation: **SpotifyCares**

Four reasons, each backed by a measurement above.

**1. Its answers are reusable.** Ephemeral content is 0.0%. Spotify support answers are
troubleshooting steps and policy statements that were as true in 2016 as they are for the next
customer — *"does logging out > restarting your phone > logging back in help at all?"*, *"we're
afraid it isn't possible to share Radio Stations you've created, but you can turn them into
playlists"*. This is precisely what "grounded in how the brand historically resolved similar
issues" requires. Airlines and rail fail here; Spotify is the strongest passer.

**2. Enough reusable evidence, and it is clean.** 29,099 reusable pairs — the largest of any
finalist — at 0.1% non-English. Retrieval density per intent is high enough that top-k
neighbours are genuinely similar rather than the nearest thing in a sparse space.

**3. A natural 10–12 intent taxonomy actually exists in the data.** TF-IDF + KMeans (k=14) over
27,600 opening customer messages produced clearly separable, nameable clusters without any
prompting: login/password (7.9%), billing & cancellation (8.1%), family plan (4.4%), student
discount (3.6%), account compromise (1.2%), content unavailable (2.0%), album requests (3.4%),
app/device bugs (7.2%), playlist & playback (6.6%), feature requests. That is a taxonomy
derived from the brand's data, not imported.

**4. — the decisive one — the brand's own behaviour gives a real escalation label.**
Spotify's 30.8% deflection rate is not a defect here; it is supervision. When Spotify's agents
could resolve a case in public they did, and when they needed account access they handed off to
DM. That handoff is an observable, non-invented record of *"a human was required"*:

| Customer message slice | n | Historical human-handoff rate |
|---|---:|---:|
| billing / charge / refund | 2,401 | **90.3%** |
| login / password / locked out | 1,594 | **83.5%** |
| student / family plan | 1,811 | **82.7%** |
| account hacked / unauthorised | 511 | **70.6%** |
| app crash / bug | 768 | 41.7% |
| content not available | 560 | 19.6% |
| feature request | 1,050 | 14.9% |
| **overall** | 28,277 | **43.3%** |

The overall rate of 43.3% means the AUTO_HANDLE / ESCALATE target is close to balanced — no
degenerate majority class — and the per-slice spread from 14.9% to 90.3% means the signal is
genuinely learnable rather than noise. Every other candidate either deflects on almost
everything (TMobile, Comcast) or almost nothing (hulu, GWR), and in both cases escalation
evaluation becomes meaningless.

**Selected brand: `SpotifyCares` — 28,280 conversations, 48,543 customer messages, 29,099
reusable historical resolutions, 43.3% historical human-handoff rate.**

## 4. Documented limitations of this data

1. **Thread reconstruction is approximate.** 0.19% of reply edges are dangling. Branching
   threads collapse into one component, so a "conversation" is occasionally two interleaved
   exchanges.
2. **Conversations are truncated at the DM boundary.** When Spotify moves to DM, the actual
   resolution happens off-dataset. We observe *that* a human took over, never *what they did*.
   The escalation label is therefore honest, but the resolution corpus is biased toward cases
   that were publicly resolvable.
3. **Customer IDs are anonymised integers** (`@115712`), so there is no customer history and no
   way to detect repeat contacts.
4. **The window is 2013-09-18 → 2017-12-03**, heavily concentrated in Oct–Dec 2017. Spotify's
   actual policies have changed since; the agent is grounded in 2017 behaviour, and every
   accuracy claim is about reproducing *2017 Spotify*, not correct 2026 advice.
5. **`inbound` is not "is a customer"**. Other brands and bots appear in threads. We filter to
   the two-party case.
6. **No resolution outcome is recorded.** Nothing in the data says whether the customer's
   problem was actually fixed. "Historical resolution" means "what the brand said", not "what
   worked".
