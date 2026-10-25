"""Find and rank the system's real failure modes from evaluation output.

    python -m evaluation.failure_analysis            # top 5, human-readable
    python -m evaluation.failure_analysis --top 8    # more modes
    python -m evaluation.failure_analysis --json     # machine-readable

Runs entirely offline. It reads `evaluation/results/predictions.jsonl` (written by
`python -m evaluation.run --no-llm`) plus the golden set, and makes no API calls.

Failure modes are DISCOVERED, not assumed: every example the system got wrong is
bucketed by an observable signal, buckets are counted, and the biggest ones are
reported with real examples pulled straight from the data. Nothing here is written by
hand, so the output changes when the system changes.

What this cannot see: anything that depends on the drafted reply (reply/evidence
mismatch, hallucinated policy, tone). Those need `evaluation/run.py` without --no-llm,
which costs API calls. Modes that need reply text are reported as UNMEASURED rather
than guessed at - see `unmeasurable_modes()`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "evaluation" / "results" / "predictions.jsonl"
GOLDEN = ROOT / "evaluation" / "golden_set.jsonl"

# Intents whose taxonomy default is ESCALATE. Auto-handling one of these is the
# expensive mistake the escalation layer exists to prevent.
SENSITIVE = {
    "billing_charge",
    "login_access",
    "account_compromised",
    "student_discount",
    "family_plan",
}


@dataclass
class Mode:
    """One discovered failure mode."""

    key: str
    title: str
    why: str
    fix: str
    rows: list[dict] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.rows)


def load_rows() -> list[dict]:
    if not PREDICTIONS.exists():
        raise SystemExit(
            f"{PREDICTIONS} not found.\nRun:  python -m evaluation.run --no-llm"
        )
    rows = [json.loads(l) for l in PREDICTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    golden = {
        json.loads(l)["id"]: json.loads(l)
        for l in GOLDEN.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    for r in rows:
        g = golden.get(r["id"], {})
        r["reference_resolution"] = g.get("reference_resolution", "")
        r["historical_handoff"] = g.get("historical_handoff")
    return rows


def classify(rows: list[dict]) -> list[Mode]:
    """Bucket every wrong prediction by an observable cause."""
    modes: dict[str, Mode] = {}

    def add(key: str, title: str, why: str, fix: str, row: dict) -> None:
        if key not in modes:
            modes[key] = Mode(key, title, why, fix)
        modes[key].rows.append(row)

    for r in rows:
        intent_wrong = r["pred_intent"] != r["gold_intent"]
        action_wrong = r["pred_action"] != r["gold_action"]
        ev = r.get("evidence") or []
        top_sim = max((e["similarity"] for e in ev), default=0.0)
        agreement = (
            Counter(e["intent"] for e in ev).most_common(1)[0][1] / len(ev) if ev else 0.0
        )

        # --- the expensive mistake: auto-handled something that needed a person ----
        if action_wrong and r["pred_action"] == "AUTO_HANDLE":
            if r["gold_intent"] in SENSITIVE:
                add(
                    "unsafe_auto_handle_sensitive",
                    "Auto-handled an account-sensitive case",
                    "The intent classifier put a billing / login / account-security "
                    "message into a safe-looking intent, so the escalation layer never "
                    "saw the intent rule that would have forced a human. Escalation "
                    "inherits every classifier error on these five intents.",
                    "Add a message-level safety net independent of predicted intent: "
                    "if the raw text matches account-specific patterns (charged, "
                    "refund, password, hacked), escalate regardless of the predicted "
                    "intent. Cheap, and it removes the dependency on being right twice.",
                    r,
                )
            else:
                add(
                    "unsafe_auto_handle_other",
                    "Auto-handled a case a human labelled ESCALATE",
                    "Retrieval found confident, agreeing precedents, so every rule "
                    "passed - but the human judged the case as needing a person for "
                    "reasons the evidence does not encode (multi-issue message, "
                    "customer already escalated, tone).",
                    "Raise `min_top_similarity`, or add a rule that escalates when the "
                    "message contains more than one distinct request.",
                    r,
                )

        # --- classifier errors, split by whether retrieval could have saved it ------
        if intent_wrong:
            if r["pred_intent"] == "other":
                add(
                    "abstained_on_answerable",
                    "Abstained (`other`) on a message that had a real intent",
                    "No intent cleared the 0.45 confidence floor, so the classifier "
                    "abstained. The abstention is safe - it escalates - but it is a "
                    "miss: these messages did have a clear intent a human could name.",
                    "Lower the floor to ~0.30 (measured on this same golden set: macro-F1 "
                    "0.595 vs 0.587) or calibrate probabilities, so the floor means the "
                    "same thing across intents.",
                    r,
                )
            elif top_sim < 0.55:
                add(
                    "weak_retrieval_neighbourhood",
                    "Wrong intent where retrieval was also weak",
                    "The message sits in a sparse region of the embedding space: the "
                    "closest historical case is below the 0.55 similarity floor, so "
                    "neither the classifier nor the retriever had a good anchor. These "
                    "are usually short, slangy or off-distribution messages.",
                    "Expand the index beyond opening messages, or fall back to a "
                    "lexical retriever when dense similarity is low - the TF-IDF "
                    "baseline scored higher intent_recall@5 (0.715 vs 0.630).",
                    r,
                )
            elif agreement < 0.6:
                add(
                    "conflicting_precedents",
                    "Wrong intent where retrieved precedents disagreed",
                    "Top-5 neighbours span several intents, so the retrieved evidence "
                    "points in different directions. The message genuinely sits on a "
                    "boundary between two intents.",
                    "Use retrieval agreement as a feature for the classifier rather "
                    "than only as an escalation signal, so disagreement lowers "
                    "confidence instead of being ignored.",
                    r,
                )
            else:
                add(
                    "confident_intent_confusion",
                    "Confidently wrong intent despite strong, agreeing evidence",
                    "The retriever found similar, consistent cases and the classifier "
                    "still chose the wrong label - the two intents are genuinely close "
                    "in meaning (e.g. a billing complaint phrased as a subscription "
                    "question). Training labels came from keyword rules, which cannot "
                    "separate these pairs.",
                    "Replace the keyword-derived training labels for the confusable "
                    "pairs, and merge intents that the confusion matrix shows are not "
                    "separable in practice.",
                    r,
                )

        # --- over-escalation: cheap mistake, but it is what caps automation ---------
        if action_wrong and r["pred_action"] == "ESCALATE":
            add(
                "over_escalation",
                "Escalated a case that could have been answered",
                "Usually the intent rule fired (the predicted intent's default is "
                "ESCALATE) or retrieval fell below threshold. Cheap per case, but this "
                "is the mode that limits how much work the system can actually take on.",
                "Make `subscription_manage`-style conditional intents the norm rather "
                "than the exception: decide from evidence quality instead of an "
                "intent-level default.",
                r,
            )

    return sorted(modes.values(), key=lambda m: -m.count)


def unmeasurable_modes() -> list[dict]:
    """Failure modes that exist but cannot be observed from offline artifacts."""
    return [
        {
            "mode": "Reply not supported by cited evidence",
            "needs": "generated reply + evidence_ids",
            "how": "python -m evaluation.run  (costs API calls)",
        },
        {
            "mode": "Invented policy / unsupported promise in a reply",
            "needs": "generated reply + judge scores",
            "how": "python -m evaluation.run, then judge rubric criterion "
                   "`no_unsupported_claims`",
        },
        {
            "mode": "Reply quality worse than pasting the top precedent verbatim",
            "needs": "judge scores for `retrieved_verbatim` vs `proposed`",
            "how": "python -m evaluation.run (both systems are already wired up)",
        },
    ]


def example_block(r: dict, idx: int) -> str:
    ev = r.get("evidence") or []
    top = ev[0] if ev else None
    lines = [
        f"  Example {idx} - {r['id']} (difficulty: {r['difficulty']})",
        f"    Customer message : {r['customer_message'][:200]}",
        f"    Human label      : intent={r['gold_intent']}  action={r['gold_action']}",
        f"    System predicted : intent={r['pred_intent']} "
        f"(confidence {r['intent_confidence']:.2f})  action={r['pred_action']}",
        f"    Deciding rule    : {r['escalation_rule']}",
        f"    Stated reason    : {r['escalation_reason'][:160]}",
    ]
    if top:
        lines += [
            f"    Closest precedent (sim {top['similarity']:.2f}, intent {top['intent']}):",
            f"      customer said : {top['customer_message'][:150]}",
            f"      spotify replied: {top['resolution'][:150]}",
        ]
    if r.get("reference_resolution"):
        lines.append(f"    Spotify actually replied: {r['reference_resolution'][:150]}")
    return "\n".join(lines)


def render(modes: list[Mode], rows: list[dict], top: int, examples: int) -> str:
    out = [
        "=" * 78,
        "TOP FAILURE MODES - discovered from evaluation output, not hand-written",
        "=" * 78,
        f"Golden set: {len(rows)} examples.  "
        f"Intent wrong: {sum(r['pred_intent'] != r['gold_intent'] for r in rows)}.  "
        f"Action wrong: {sum(r['pred_action'] != r['gold_action'] for r in rows)}.",
        "",
    ]
    for i, m in enumerate(modes[:top], 1):
        share = 100 * m.count / len(rows)
        out += [
            "-" * 78,
            f"{i}. {m.title}",
            f"   Affected: {m.count} of {len(rows)} examples ({share:.1f}%)",
            "",
            f"   Why it fails: {m.why}",
            "",
            f"   Proposed fix: {m.fix}",
            "",
        ]
        for j, r in enumerate(sorted(m.rows, key=lambda x: x["id"])[:examples], 1):
            out += [example_block(r, j), ""]

    out += ["=" * 78, "NOT MEASURABLE OFFLINE (needs generated replies / judge scores)", "=" * 78]
    for u in unmeasurable_modes():
        out += [f"  - {u['mode']}", f"      needs: {u['needs']}", f"      run  : {u['how']}"]
    out += [
        "",
        "These are listed so the gap is explicit. They are NOT included in the ranked",
        "modes above, because no evidence for them exists in the offline artifacts.",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5, help="how many failure modes to report")
    ap.add_argument("--examples", type=int, default=2, help="real examples per mode")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--out", type=Path, default=None, help="also write the text report here")
    args = ap.parse_args()

    rows = load_rows()
    modes = classify(rows)

    if args.json:
        print(
            json.dumps(
                {
                    "n_examples": len(rows),
                    "modes": [
                        {
                            "rank": i,
                            "key": m.key,
                            "title": m.title,
                            "count": m.count,
                            "share": round(m.count / len(rows), 4),
                            "why": m.why,
                            "fix": m.fix,
                            "example_ids": [r["id"] for r in m.rows],
                        }
                        for i, m in enumerate(modes[: args.top], 1)
                    ],
                    "unmeasurable_offline": unmeasurable_modes(),
                },
                indent=2,
            )
        )
        return

    text = render(modes, rows, args.top, args.examples)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
