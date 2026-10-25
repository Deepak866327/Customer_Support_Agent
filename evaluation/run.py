"""Reproducible evaluation across intent, retrieval, escalation and reply quality.

    python -m evaluation.run                 # full run (uses the committed LLM cache)
    python -m evaluation.run --no-llm        # skip reply generation and judging
    python -m evaluation.run --limit 20      # smoke test / cost pilot

Writes evaluation/results/ : metrics.json, predictions.jsonl, report.md
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.baselines import TfidfRetriever, verbatim_reply  # noqa: E402
from evaluation.judge import Judge  # noqa: E402
from src.classifier import MajorityBaseline, TfidfBaseline  # noqa: E402
from src.embeddings import encode  # noqa: E402
from src.escalation import EscalationConfig, decide  # noqa: E402
from src.generator import generate  # noqa: E402
from src.intents import OTHER, load_taxonomy  # noqa: E402
from src.llm import LLMClient, LLMUnavailable  # noqa: E402
from src.retrieval import ResolutionIndex, evidence_agreement  # noqa: E402

GOLDEN = ROOT / "evaluation" / "golden_set.jsonl"
RESULTS = ROOT / "evaluation" / "results"
K = 5
REF_SIM_THRESHOLD = 0.50  # see DECISIONS.md #11


# --------------------------------------------------------------------------- helpers
def load_golden(limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def intent_metrics(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(gold, pred), 4),
        "macro_f1": round(f1_score(gold, pred, average="macro", labels=labels, zero_division=0), 4),
        "weighted_f1": round(
            f1_score(gold, pred, average="weighted", labels=labels, zero_division=0), 4
        ),
        "per_class_f1": {
            lab: round(v, 4)
            for lab, v in zip(
                labels,
                f1_score(gold, pred, average=None, labels=labels, zero_division=0).tolist(),
            )
        },
    }


def escalation_metrics(gold: list[str], pred: list[str]) -> dict:
    """ESCALATE is the positive class. False auto-handle is the metric that matters."""
    p, r, f, _ = precision_recall_fscore_support(
        gold, pred, labels=["ESCALATE"], average="binary", pos_label="ESCALATE", zero_division=0
    )
    should_escalate = [g == "ESCALATE" for g in gold]
    auto_handled = [p_ == "AUTO_HANDLE" for p_ in pred]
    n_should = sum(should_escalate)
    false_auto = sum(s and a for s, a in zip(should_escalate, auto_handled))
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "false_auto_handle_rate": round(false_auto / n_should, 4) if n_should else 0.0,
        "false_auto_handles": int(false_auto),
        "n_should_escalate": int(n_should),
        "auto_handle_share": round(sum(auto_handled) / len(pred), 4),
    }


def retrieval_metrics(hits_per_row: list[list], gold_intents: list[str], ref_sims: list[list[float]]) -> dict:
    """Two definitions, because neither alone is honest.

    intent_recall@k - does top-k contain a case of the same intent? Cheap, but the
      index's own intent tags are model-assigned for 66% of entries, so it partly
      measures the classifier agreeing with itself.
    reference_recall@k - does top-k contain a historical resolution semantically close
      to the reply Spotify actually sent for THIS message? Independent of the intent
      model, and closer to what the generator actually needs.
    """
    out = {}
    for k in (1, 3, 5):
        ir = np.mean(
            [any(h.intent == g for h in hits[:k]) for hits, g in zip(hits_per_row, gold_intents)]
        )
        rr = np.mean(
            [max(sims[:k], default=0.0) >= REF_SIM_THRESHOLD for sims in ref_sims]
        )
        out[f"intent_recall@{k}"] = round(float(ir), 4)
        out[f"reference_recall@{k}"] = round(float(rr), 4)
    out["mean_top1_similarity"] = round(
        float(np.mean([hits[0].similarity if hits else 0.0 for hits in hits_per_row])), 4
    )
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    ap.add_argument("--no-llm", action="store_true", help="skip reply generation and judging")
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()

    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    tx = load_taxonomy()
    golden = load_golden(args.limit)
    messages = [r["customer_message"] for r in golden]
    gold_intents = [r["intent"] for r in golden]
    gold_actions = [r["expected_action"] for r in golden]
    labels = tx.names
    print(f"golden set: {len(golden)} examples")

    with (ROOT / "data" / "cache" / "classifier.pkl").open("rb") as fh:
        embed_clf = pickle.load(fh)
    index = ResolutionIndex.load()

    # ---- retrain the two intent baselines on exactly the same seeded data ----------
    import pandas as pd

    train = pd.read_parquet(ROOT / "data" / "processed" / "train_pairs.parquet")
    pool = train[
        train["is_opening"] & (train["customer_text"].str.split().str.len() >= 4)
    ].drop_duplicates(subset="customer_text")
    pool = pool.assign(seed=tx.seed_label_many(pool["customer_text"]))
    seeded = pool[pool["seed"].notna()]
    Xtr, ytr = seeded["customer_text"].tolist(), seeded["seed"].tolist()

    print("training baselines ...")
    majority = MajorityBaseline().fit(Xtr, ytr)
    tfidf = TfidfBaseline().fit(Xtr, ytr)

    systems = {
        "majority": majority,
        "tfidf_lr": tfidf,
        "proposed": embed_clf,
    }

    # ---- intent ------------------------------------------------------------------
    print("evaluating intent ...")
    metrics: dict = {"n_examples": len(golden), "k": args.k, "intent": {}}
    preds: dict[str, tuple[list[str], np.ndarray]] = {}
    for name, clf in systems.items():
        if name == "proposed":
            p, c = clf.predict(messages, tag="golden")
        else:
            p, c = clf.predict(messages)
        preds[name] = (p, np.asarray(c))
        metrics["intent"][name] = intent_metrics(gold_intents, p, labels)

    metrics["intent"]["confusion_matrix"] = {
        "labels": labels,
        "proposed": confusion_matrix(gold_intents, preds["proposed"][0], labels=labels).tolist(),
    }
    metrics["weak_label_precision_vs_human"] = _weak_label_precision(tx, golden)

    # ---- retrieval ---------------------------------------------------------------
    print("evaluating retrieval ...")
    tfidf_ret = TfidfRetriever(index)
    ref_vecs = encode([r["reference_resolution"] for r in golden], tag="goldenref")

    dense_hits, dense_ref_sims = [], []
    lex_hits, lex_ref_sims = [], []
    for i, msg in enumerate(messages):
        d = index.search(msg, k=args.k, intent=preds["proposed"][0][i])
        l = tfidf_ret.search(msg, k=args.k)
        dense_hits.append(d)
        lex_hits.append(l)
        dense_ref_sims.append(_ref_sims(d, ref_vecs[i]))
        lex_ref_sims.append(_ref_sims(l, ref_vecs[i]))

    metrics["retrieval"] = {
        "tfidf_retrieval": retrieval_metrics(lex_hits, gold_intents, lex_ref_sims),
        "proposed": retrieval_metrics(dense_hits, gold_intents, dense_ref_sims),
    }

    # ---- escalation --------------------------------------------------------------
    print("evaluating escalation ...")
    cfg = EscalationConfig()
    metrics["escalation"] = {}
    decisions: dict[str, list] = {}
    for name in systems:
        p, c = preds[name]
        ds = [
            decide(m, it, float(cf), dense_hits[i], tx, cfg)
            for i, (m, it, cf) in enumerate(zip(messages, p, c))
        ]
        decisions[name] = ds
        metrics["escalation"][name] = escalation_metrics(gold_actions, [d.action for d in ds])
    metrics["escalation"]["proposed"]["rule_counts"] = dict(
        Counter(d.rule for d in decisions["proposed"])
    )

    # ---- reply quality -----------------------------------------------------------
    rows_out = []
    if args.no_llm:
        print("skipping reply generation and judging (--no-llm)")
        metrics["reply_quality"] = None
    else:
        print("generating and judging replies ...")
        client = LLMClient()
        judge = Judge()
        reply_systems = ("retrieved_verbatim", "llm_no_evidence", "proposed")
        collected: dict[str, list] = {s: [] for s in reply_systems}
        try:
            for i, row in enumerate(golden):
                msg = row["customer_message"]
                ev = dense_hits[i]
                intent = preds["proposed"][0][i]
                drafts = {
                    "retrieved_verbatim": verbatim_reply(ev),
                    "llm_no_evidence": generate(msg, intent, [], client),
                    "proposed": generate(msg, intent, ev, client),
                }
                scored = {}
                for sysname, draft in drafts.items():
                    ev_for_judge = ev if sysname != "llm_no_evidence" else []
                    js = judge.score(msg, draft.reply, ev_for_judge, row["reference_resolution"])
                    collected[sysname].append(js)
                    scored[sysname] = {"draft": draft.as_dict(), "judge": js.as_dict()}
                rows_out.append(
                    {
                        "id": row["id"],
                        "customer_message": msg,
                        "gold_intent": row["intent"],
                        "pred_intent": intent,
                        "intent_confidence": round(float(preds["proposed"][1][i]), 4),
                        "gold_action": row["expected_action"],
                        "pred_action": decisions["proposed"][i].action,
                        "escalation_reason": decisions["proposed"][i].reason,
                        "escalation_rule": decisions["proposed"][i].rule,
                        "difficulty": row["difficulty"],
                        "evidence": [e.as_dict() for e in ev],
                        "systems": scored,
                    }
                )
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}/{len(golden)}  (cache hits {client.hits}, live {client.misses})")
        except LLMUnavailable as exc:
            print(f"\nLLM unavailable: {exc}")
            print("Re-run with --no-llm, or set the provider API key to generate the cache.")
            sys.exit(2)

        metrics["reply_quality"] = {
            s: _aggregate(collected[s]) for s in reply_systems if collected[s]
        }
        metrics["llm_calls"] = {"cache_hits": client.hits + judge.client.hits,
                                "live_calls": client.misses + judge.client.misses}

    # ---- write -------------------------------------------------------------------
    if not rows_out:
        rows_out = [
            {
                "id": r["id"],
                "customer_message": r["customer_message"],
                "gold_intent": r["intent"],
                "pred_intent": preds["proposed"][0][i],
                "intent_confidence": round(float(preds["proposed"][1][i]), 4),
                "gold_action": r["expected_action"],
                "pred_action": decisions["proposed"][i].action,
                "escalation_reason": decisions["proposed"][i].reason,
                "escalation_rule": decisions["proposed"][i].rule,
                "difficulty": r["difficulty"],
                "evidence": [e.as_dict() for e in dense_hits[i]],
            }
            for i, r in enumerate(golden)
        ]

    metrics["by_difficulty"] = _by_difficulty(golden, preds["proposed"][0], decisions["proposed"])
    metrics["runtime_seconds"] = round(time.time() - t0, 1)

    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (RESULTS / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (RESULTS / "report.md").write_text(_render(metrics, tx, gold_intents, preds), encoding="utf-8")

    print(f"\ndone in {metrics['runtime_seconds']}s -> {RESULTS}")
    print(_headline(metrics))


# --------------------------------------------------------------------------- support
def _ref_sims(hits, ref_vec) -> list[float]:
    if not hits:
        return []
    vecs = encode([h.resolution for h in hits], tag="hitres")
    return (vecs @ ref_vec).tolist()


def _aggregate(scores) -> dict:
    from evaluation.judge import CRITERIA

    out = {c: round(float(np.mean([s.scores[c] for s in scores])), 3) for c in CRITERIA}
    out["overall"] = round(float(np.mean([s.mean for s in scores])), 3)
    return out


def _weak_label_precision(tx, golden) -> dict:
    """How often the weak-labelling rules agree with the human label, where they fire.

    This is the ceiling the classifier is trained against, and it belongs in the report.
    """
    fired = [(tx.seed_label(r["customer_message"]), r["intent"]) for r in golden]
    hits = [(s, g) for s, g in fired if s is not None]
    if not hits:
        return {"n_fired": 0}
    correct = sum(s == g for s, g in hits)
    return {
        "n_fired": len(hits),
        "coverage": round(len(hits) / len(golden), 4),
        "precision": round(correct / len(hits), 4),
    }


def _by_difficulty(golden, pred_intents, decisions) -> dict:
    out = {}
    for level in ("easy", "medium", "hard"):
        idx = [i for i, r in enumerate(golden) if r["difficulty"] == level]
        if not idx:
            continue
        out[level] = {
            "n": len(idx),
            "intent_accuracy": round(
                float(np.mean([pred_intents[i] == golden[i]["intent"] for i in idx])), 4
            ),
            "escalation_accuracy": round(
                float(np.mean([decisions[i].action == golden[i]["expected_action"] for i in idx])),
                4,
            ),
        }
    return out


def _headline(m: dict) -> str:
    p = m["intent"]["proposed"]
    e = m["escalation"]["proposed"]
    line = (
        f"\nHEADLINE  intent macro-F1 {p['macro_f1']:.3f} (acc {p['accuracy']:.3f})"
        f" | escalation F1 {e['f1']:.3f}, false auto-handle {e['false_auto_handle_rate']:.1%}"
    )
    if m.get("reply_quality"):
        line += f" | reply quality {m['reply_quality']['proposed']['overall']:.2f}/5"
    return line


def _render(m: dict, tx, gold_intents, preds) -> str:
    L = ["# Evaluation results", "", f"Golden set: {m['n_examples']} hand-labelled examples, k={m['k']}.", ""]

    L += ["## Headline comparison", "",
          "| System | Intent acc | Intent macro-F1 | Retrieval ref-recall@5 | Escalation F1 | False auto-handle | Reply quality |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    ret = m["retrieval"]
    rq = m.get("reply_quality") or {}
    rows = [
        ("Majority baseline", "majority", None, None),
        ("TF-IDF baseline", "tfidf_lr", "tfidf_retrieval", "retrieved_verbatim"),
        ("Proposed system", "proposed", "proposed", "proposed"),
    ]
    for label, ikey, rkey, qkey in rows:
        i = m["intent"][ikey]
        e = m["escalation"][ikey]
        r = f"{ret[rkey]['reference_recall@5']:.3f}" if rkey else "n/a"
        q = f"{rq[qkey]['overall']:.2f}" if qkey and qkey in rq else "n/a"
        L.append(
            f"| {label} | {i['accuracy']:.3f} | {i['macro_f1']:.3f} | {r} | "
            f"{e['f1']:.3f} | {e['false_auto_handle_rate']:.1%} | {q} |"
        )

    wl = m.get("weak_label_precision_vs_human", {})
    if wl.get("n_fired"):
        L += ["", f"Weak-labelling rules fired on {wl['coverage']:.0%} of the golden set and "
                  f"agreed with the human label {wl['precision']:.1%} of the time - that is the "
                  "ceiling the classifier was trained against."]

    L += ["", "## Per-intent F1 (proposed)", "", "| Intent | F1 | support |", "|---|---:|---:|"]
    support = Counter(gold_intents)
    for k, v in sorted(m["intent"]["proposed"]["per_class_f1"].items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v:.3f} | {support.get(k, 0)} |")

    L += ["", "## Retrieval", "", "| Metric | TF-IDF | Proposed |", "|---|---:|---:|"]
    for key in ret["proposed"]:
        L.append(f"| {key} | {ret['tfidf_retrieval'][key]:.3f} | {ret['proposed'][key]:.3f} |")

    L += ["", "## Escalation (ESCALATE = positive class)", "",
          "| System | P | R | F1 | False auto-handle rate | Auto-handle share |", "|---|---:|---:|---:|---:|---:|"]
    for label, key in (("Majority", "majority"), ("TF-IDF", "tfidf_lr"), ("Proposed", "proposed")):
        e = m["escalation"][key]
        L.append(f"| {label} | {e['precision']:.3f} | {e['recall']:.3f} | {e['f1']:.3f} | "
                 f"{e['false_auto_handle_rate']:.1%} ({e['false_auto_handles']}/{e['n_should_escalate']}) "
                 f"| {e['auto_handle_share']:.1%} |")

    if m["escalation"]["proposed"].get("rule_counts"):
        L += ["", "Which rule decided each case (proposed):", ""]
        for rule, n in sorted(m["escalation"]["proposed"]["rule_counts"].items(), key=lambda x: -x[1]):
            L.append(f"- `{rule}`: {n}")

    if rq:
        L += ["", "## Reply quality (LLM judge, 1-5)", "",
              "| System | correctness | groundedness | consistency | helpfulness | no unsupported claims | overall |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for s, v in rq.items():
            L.append(f"| {s} | {v['correctness']:.2f} | {v['groundedness']:.2f} | "
                     f"{v['consistency']:.2f} | {v['helpfulness']:.2f} | "
                     f"{v['no_unsupported_claims']:.2f} | {v['overall']:.2f} |")
        L += ["", "These are judge scores, not truth. See JUDGE_VALIDATION.md for the "
                  "human-agreement measurement before quoting them."]

    L += ["", "## Performance by labelling difficulty", "",
          "| Difficulty | n | Intent accuracy | Escalation accuracy |", "|---|---:|---:|---:|"]
    for lvl, v in m["by_difficulty"].items():
        L.append(f"| {lvl} | {v['n']} | {v['intent_accuracy']:.3f} | {v['escalation_accuracy']:.3f} |")

    L += ["", "---", "",
          "Full per-example output: `evaluation/results/predictions.jsonl`.",
          f"Runtime {m['runtime_seconds']}s."]
    return "\n".join(L)


if __name__ == "__main__":
    main()
