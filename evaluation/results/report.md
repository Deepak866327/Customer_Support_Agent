# Evaluation results

Golden set: 200 hand-labelled examples, k=5.

## Headline comparison

| System | Intent acc | Intent macro-F1 | Retrieval ref-recall@5 | Escalation F1 | False auto-handle | Reply quality |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.070 | 0.011 | n/a | 0.693 | 0.0% | n/a |
| TF-IDF baseline | 0.605 | 0.605 | 0.415 | 0.779 | 3.8% | 4.06 |
| Proposed system | 0.575 | 0.587 | 0.470 | 0.809 | 14.1% | 4.58 |

Weak-labelling rules fired on 51% of the golden set and agreed with the human label 72.5% of the time - that is the ceiling the classifier was trained against.

## Per-intent F1 (proposed)

| Intent | F1 | support |
|---|---:|---:|
| family_plan | 0.933 | 14 |
| student_discount | 0.839 | 15 |
| login_access | 0.757 | 21 |
| account_compromised | 0.733 | 14 |
| content_unavailable | 0.667 | 19 |
| billing_charge | 0.643 | 12 |
| subscription_manage | 0.615 | 23 |
| app_crash_performance | 0.533 | 14 |
| playlist_library | 0.467 | 14 |
| playback_error | 0.429 | 11 |
| feature_request_feedback | 0.267 | 25 |
| other | 0.167 | 18 |

## Retrieval

| Metric | TF-IDF | Proposed |
|---|---:|---:|
| intent_recall@1 | 0.390 | 0.580 |
| reference_recall@1 | 0.190 | 0.265 |
| intent_recall@3 | 0.635 | 0.595 |
| reference_recall@3 | 0.345 | 0.390 |
| intent_recall@5 | 0.715 | 0.630 |
| reference_recall@5 | 0.415 | 0.470 |
| mean_top1_similarity | 0.278 | 0.663 |

## Escalation (ESCALATE = positive class)

| System | P | R | F1 | False auto-handle rate | Auto-handle share |
|---|---:|---:|---:|---:|---:|
| Majority | 0.530 | 1.000 | 0.693 | 0.0% (0/106) | 0.0% |
| TF-IDF | 0.654 | 0.962 | 0.779 | 3.8% (4/106) | 22.0% |
| Proposed | 0.765 | 0.859 | 0.809 | 14.1% (15/106) | 40.5% |

Which rule decided each case (proposed):

- `confident_and_grounded`: 81
- `intent_requires_human`: 78
- `intent_abstained`: 18
- `weak_retrieval`: 17
- `high_risk_language`: 4
- `conflicting_evidence`: 2

## Reply quality (LLM judge, 1-5)

| System | correctness | groundedness | consistency | helpfulness | no unsupported claims | overall |
|---|---:|---:|---:|---:|---:|---:|
| retrieved_verbatim | 3.39 | 4.82 | 4.16 | 3.06 | 4.86 | 4.06 |
| llm_no_evidence | 3.33 | 1.67 | 2.88 | 3.00 | 4.64 | 3.11 |
| proposed | 4.32 | 4.88 | 4.85 | 3.88 | 4.98 | 4.58 |

These are judge scores, not truth. See JUDGE_VALIDATION.md for the human-agreement measurement before quoting them.

## Performance by labelling difficulty

| Difficulty | n | Intent accuracy | Escalation accuracy |
|---|---:|---:|---:|
| easy | 80 | 0.787 | 0.900 |
| medium | 78 | 0.474 | 0.679 |
| hard | 42 | 0.357 | 0.762 |

---

Full per-example output: `evaluation/results/predictions.jsonl`.
Runtime 735.5s.