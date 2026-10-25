"""Merge hand labels with the sampled candidates into the golden set.

    python scripts/build_golden_set.py

evaluation/golden_candidates.jsonl  (stratified sample, machine-generated)
  + evaluation/hand_labels.tsv      (intent / expected_action / difficulty, by hand)
  -> evaluation/golden_set.jsonl

Labelling protocol (see README):
  * Labels were assigned from the customer message ALONE. Model predictions and the
    reply Spotify actually sent were hidden during labelling to avoid anchoring.
  * `expected_action` answers: "could a public reply grounded in past cases resolve
    this, or does it need account access / a human judgement call?" It is NOT copied
    from what Spotify did - Spotify's actual behaviour is kept separately as
    `historical_handoff` so the two can be compared.
  * `difficulty` is the labeller's own confidence: easy = unambiguous, medium = needs
    thought, hard = genuinely ambiguous, multi-issue, non-English, or context-free.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
VALID_ACTIONS = {"AUTO_HANDLE", "ESCALATE"}
VALID_DIFFICULTY = {"easy", "medium", "hard"}


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from src.intents import load_taxonomy

    valid_intents = set(load_taxonomy().names)

    labels: dict[str, dict[str, str]] = {}
    with (EVAL / "hand_labels.tsv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            labels[row["id"]] = row

    out_path = EVAL / "golden_set.jsonl"
    written = 0
    problems: list[str] = []

    with (EVAL / "golden_candidates.jsonl").open(encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            cand = json.loads(line)
            lab = labels.get(cand["id"])
            if lab is None:
                problems.append(f"{cand['id']}: no hand label")
                continue
            if lab["intent"] not in valid_intents:
                problems.append(f"{cand['id']}: unknown intent {lab['intent']!r}")
            if lab["expected_action"] not in VALID_ACTIONS:
                problems.append(f"{cand['id']}: bad action {lab['expected_action']!r}")
            if lab["difficulty"] not in VALID_DIFFICULTY:
                problems.append(f"{cand['id']}: bad difficulty {lab['difficulty']!r}")

            dst.write(
                json.dumps(
                    {
                        "id": cand["id"],
                        "pair_id": cand["pair_id"],
                        "customer_message": cand["customer_message"],
                        "intent": lab["intent"],
                        "expected_action": lab["expected_action"],
                        "difficulty": lab["difficulty"],
                        "reference_resolution": cand["actual_spotify_reply"],
                        "historical_handoff": cand["historical_handoff"],
                        "n_words": cand["n_words"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    if problems:
        print("VALIDATION PROBLEMS:")
        for p in problems:
            print("  ", p)
        sys.exit(1)

    print(f"wrote {written} labelled examples -> {out_path}")

    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    from collections import Counter

    print("\nintent distribution:")
    for k, v in Counter(r["intent"] for r in rows).most_common():
        print(f"  {k:26s} {v}")
    print("\nexpected action:", dict(Counter(r["expected_action"] for r in rows)))
    print("difficulty     :", dict(Counter(r["difficulty"] for r in rows)))
    agree = sum(
        (r["expected_action"] == "ESCALATE") == r["historical_handoff"] for r in rows
    )
    print(
        f"\nmy ESCALATE label agrees with what Spotify actually did on "
        f"{agree}/{len(rows)} ({100 * agree / len(rows):.1f}%)"
    )


if __name__ == "__main__":
    main()
