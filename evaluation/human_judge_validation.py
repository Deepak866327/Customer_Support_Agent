"""Infrastructure for validating the LLM judge against human scores.

    # 1. build a blank scoring sheet (offline, no API)
    python -m evaluation.human_judge_validation build --n 40

    # 2. a human fills in the human_* fields in the generated JSONL, by hand

    # 3. compare humans against the LLM judge (offline, no API)
    python -m evaluation.human_judge_validation score

STATUS: **NOT YET RUN.** No human has scored anything, and the LLM judge has not been
run over the golden set. This module only prepares the measurement; it never invents
scores. `build` refuses to emit rows for examples that have no LLM judge output yet, and
`score` refuses to compute agreement over unfilled rows.

Why this matters: the brief requires evidence that the LLM judge agrees with humans. A
judge that hands out high scores is not thereby a good judge - it may simply be
generous, or biased toward replies produced by its own model family. Until this
measurement exists, every reply-quality number in the report is an unvalidated
instrument reading and is labelled as such.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "evaluation" / "results" / "predictions.jsonl"
SHEET = ROOT / "evaluation" / "human_judge_validation.jsonl"

# Human-facing field names mirror the judge rubric in evaluation/judge.py one-for-one,
# so the two are directly comparable without a mapping table.
HUMAN_FIELDS = {
    "human_correctness": "correctness",
    "human_evidence_support": "groundedness",
    "human_past_answer_match": "consistency",
    "human_helpfulness": "helpfulness",
    "human_no_false_promises": "no_unsupported_claims",
}

RUBRIC = """Score each field 1-5 (integers). Leave null if you cannot judge it.

  human_correctness        Is it true for Spotify, and does it answer what was asked?
  human_evidence_support   Is every claim supported by the precedents shown?
  human_past_answer_match  Does it handle this the way the precedents show Spotify did?
  human_helpfulness        Does it move the customer forward?
  human_no_false_promises  Does it avoid inventing policy or promising account actions?
  human_overall            Your own 1-5 summary (not an average - your judgement).

Score the reply in front of you. `reference_resolution` is what Spotify actually sent;
it is context, not a target. A different but equally good reply should score well."""


def _load_predictions() -> list[dict]:
    if not PREDICTIONS.exists():
        raise SystemExit(
            f"{PREDICTIONS} not found.\nRun the evaluation first: python -m evaluation.run"
        )
    return [
        json.loads(l)
        for l in PREDICTIONS.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def build(n: int, system: str) -> None:
    """Emit a blank scoring sheet, stratified over difficulty."""
    rows = _load_predictions()
    scored = [r for r in rows if r.get("systems", {}).get(system)]

    if not scored:
        raise SystemExit(
            "No LLM-judged replies found in predictions.jsonl.\n"
            "This sheet compares humans against the judge, so the judge must run first:\n"
            "    python -m evaluation.run          (requires an API key, costs calls)\n"
            "Then re-run this command.\n"
            "Nothing was written - this tool will not fabricate replies or scores."
        )

    # stratify so hard cases are not under-represented in the validation sample
    by_diff: dict[str, list[dict]] = {}
    for r in scored:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)
    per = max(1, n // max(1, len(by_diff)))
    picked: list[dict] = []
    for _, group in sorted(by_diff.items()):
        picked.extend(group[:per])
    for r in scored:
        if len(picked) >= n:
            break
        if r not in picked:
            picked.append(r)
    picked = picked[:n]

    with SHEET.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_rubric": RUBRIC, "_system_under_test": system}) + "\n")
        for r in picked:
            block = r["systems"][system]
            fh.write(
                json.dumps(
                    {
                        "example_id": r["id"],
                        "difficulty": r.get("difficulty"),
                        "customer_message": r["customer_message"],
                        "generated_reply": block["draft"]["reply"],
                        "evidence_ids": block["draft"]["evidence_ids"],
                        "evidence": [
                            {"id": e["evidence_id"], "past_customer": e["customer_message"],
                             "spotify_replied": e["resolution"]}
                            for e in r.get("evidence", [])
                        ],
                        "reference_resolution": r.get("reference_resolution", ""),
                        # to be filled in by a human - deliberately null
                        **{k: None for k in HUMAN_FIELDS},
                        "human_overall": None,
                        "human_notes": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(picked)} blank rows -> {SHEET}")
    print("All human_* fields are null. Fill them in by hand, then run:")
    print("    python -m evaluation.human_judge_validation score")


def _kappa_quadratic(a: list[int], b: list[int], lo: int = 1, hi: int = 5) -> float | None:
    """Quadratic-weighted Cohen's kappa - the right statistic for ordered 1-5 ratings."""
    n = len(a)
    if n == 0:
        return None
    cats = list(range(lo, hi + 1))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    ha = [sum(obs[i]) / n for i in range(k)]
    hb = [sum(obs[i][j] for i in range(k)) / n for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * obs[i][j] / n
            den += w * ha[i] * hb[j]
    if den == 0:
        return None
    return 1.0 - num / den


def score(system: str) -> None:
    if not SHEET.exists():
        raise SystemExit(
            f"{SHEET} not found.\nRun: python -m evaluation.human_judge_validation build"
        )
    lines = [l for l in SHEET.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in lines]
    rows = [r for r in rows if "example_id" in r]  # drop the rubric header line

    filled = [r for r in rows if any(r.get(k) is not None for k in HUMAN_FIELDS)]
    if not filled:
        raise SystemExit(
            f"{len(rows)} rows found, none scored by a human yet.\n"
            "Agreement cannot be computed and will NOT be estimated. "
            "Fill in the human_* fields first."
        )

    judged = {r["id"]: r for r in _load_predictions()}
    print(f"human-scored rows: {len(filled)} of {len(rows)}")
    if len(filled) < len(rows):
        print(f"  (note: {len(rows) - len(filled)} rows still blank and are excluded)")
    print()

    header = f"{'criterion':24s} {'n':>4s} {'exact':>7s} {'within1':>8s} {'kappa_qw':>9s} {'human':>6s} {'llm':>6s}"
    print(header)
    print("-" * len(header))

    for hkey, jkey in HUMAN_FIELDS.items():
        pairs: list[tuple[int, int]] = []
        for r in filled:
            hv = r.get(hkey)
            block = judged.get(r["example_id"], {}).get("systems", {}).get(system)
            if hv is None or not block:
                continue
            jv = block.get("judge", {}).get(jkey)
            if jv is None:
                continue
            pairs.append((int(hv), int(jv)))
        if not pairs:
            print(f"{hkey:24s} {'0':>4s}       -        -         -      -      -")
            continue
        h, j = [p[0] for p in pairs], [p[1] for p in pairs]
        exact = sum(x == y for x, y in pairs) / len(pairs)
        within1 = sum(abs(x - y) <= 1 for x, y in pairs) / len(pairs)
        kq = _kappa_quadratic(h, j)
        kqs = f"{kq:9.3f}" if kq is not None else f"{'n/a':>9s}"
        print(
            f"{hkey:24s} {len(pairs):4d} {exact:7.1%} {within1:8.1%} {kqs} "
            f"{sum(h)/len(h):6.2f} {sum(j)/len(j):6.2f}"
        )

    print()
    print("Interpretation guide (Landis & Koch, for kappa):")
    print("  < 0.20 poor | 0.21-0.40 fair | 0.41-0.60 moderate | 0.61-0.80 substantial")
    print()
    print("A high mean LLM score is NOT evidence the judge is reliable. Only agreement")
    print("with human scores is. Report kappa alongside every reply-quality number.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write a blank human scoring sheet")
    b.add_argument("--n", type=int, default=40)
    b.add_argument("--system", default="proposed")
    s = sub.add_parser("score", help="compare filled human scores against the LLM judge")
    s.add_argument("--system", default="proposed")
    args = ap.parse_args()

    if args.cmd == "build":
        build(args.n, args.system)
    else:
        score(args.system)


if __name__ == "__main__":
    main()
