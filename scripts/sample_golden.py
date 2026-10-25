"""Draw a stratified candidate pool for hand-labelling into the golden set.

    python scripts/sample_golden.py

Samples ONLY from data/processed/eval_pairs.parquet (conversations that start on or
after 2017-11-25), so nothing in the golden set was seen by the classifier, the
retrieval index, or the weak labeller.

Stratification (see README "Golden set"):
  * every intent the classifier can predict, rare ones included
  * ~1/3 messages that NO weak-labelling rule matched  -> the hard slice
  * short (< 10 words) and long (> 30 words) messages
  * cases where retrieval is weak (top similarity < 0.55)

Writes evaluation/golden_candidates.jsonl. Labelling is then done by hand; the labelled
result is evaluation/golden_set.jsonl.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.intents import load_taxonomy  # noqa: E402
from src.retrieval import ResolutionIndex, evidence_agreement  # noqa: E402

SEED = 42
TARGET = 200
OUT = ROOT / "evaluation" / "golden_candidates.jsonl"


def main() -> None:
    rng = np.random.default_rng(SEED)
    tx = load_taxonomy()
    with (ROOT / "data" / "cache" / "classifier.pkl").open("rb") as fh:
        clf = pickle.load(fh)
    index = ResolutionIndex.load()

    ev = pd.read_parquet(ROOT / "data" / "processed" / "eval_pairs.parquet")
    pool = ev[ev["is_opening"] & (ev["customer_text"].str.split().str.len() >= 4)].copy()
    pool = pool.drop_duplicates(subset="customer_text").reset_index(drop=True)
    print(f"eval pool: {len(pool):,} opening messages")

    pred, conf = clf.predict(pool["customer_text"].tolist(), tag="goldenpool")
    pool["pred_intent"] = pred
    pool["pred_conf"] = conf
    pool["seed_intent"] = tx.seed_label_many(pool["customer_text"])
    pool["rule_matched"] = pool["seed_intent"].notna()
    pool["n_words"] = pool["customer_text"].str.split().str.len()

    tops, agrees = [], []
    for msg, it in zip(pool["customer_text"], pool["pred_intent"]):
        hits = index.search(msg, k=5, intent=it)
        tops.append(max((h.similarity for h in hits), default=0.0))
        agrees.append(evidence_agreement(hits))
    pool["top_sim"] = tops
    pool["evidence_agreement"] = agrees

    # ---- stratified draw -------------------------------------------------------
    per_intent = max(1, TARGET // (len(tx.trainable_names) + 1))
    picked: list[int] = []

    def take(frame: pd.DataFrame, n: int) -> None:
        avail = frame[~frame.index.isin(picked)]
        if len(avail) == 0 or n <= 0:
            return
        chosen = avail.sample(min(n, len(avail)), random_state=int(rng.integers(1e6)))
        picked.extend(chosen.index.tolist())

    # 1. even coverage across predicted intents (rare intents included)
    for name in tx.trainable_names + ["other"]:
        take(pool[pool["pred_intent"] == name], per_intent)

    # 2. hard slice: no rule fired
    take(pool[~pool["rule_matched"]], 40)
    # 3. weak retrieval
    take(pool[pool["top_sim"] < 0.55], 25)
    # 4. length extremes
    take(pool[pool["n_words"] < 10], 15)
    take(pool[pool["n_words"] > 30], 15)
    # 5. low classifier confidence (ambiguous)
    take(pool[pool["pred_conf"] < 0.5], 15)
    # 6. fill to target at random
    take(pool, TARGET - len(picked))

    sample = pool.loc[picked[:TARGET]].copy()

    with OUT.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(sample.itertuples(), start=1):
            fh.write(
                json.dumps(
                    {
                        "id": f"g{i:03d}",
                        "pair_id": row.pair_id,
                        "customer_message": row.customer_text,
                        "actual_spotify_reply": row.reply_text,
                        "historical_handoff": bool(row.conv_handoff),
                        "pred_intent": row.pred_intent,
                        "pred_conf": round(float(row.pred_conf), 3),
                        "rule_intent": row.seed_intent,
                        "n_words": int(row.n_words),
                        "top_sim": round(float(row.top_sim), 3),
                        "evidence_agreement": round(float(row.evidence_agreement), 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"wrote {len(sample)} candidates -> {OUT}")
    print("\nslice coverage:")
    print(f"  rule matched      : {sample['rule_matched'].sum()} / {len(sample)}")
    print(f"  weak retrieval    : {(sample['top_sim'] < 0.55).sum()}")
    print(f"  short (<10w)      : {(sample['n_words'] < 10).sum()}")
    print(f"  long (>30w)       : {(sample['n_words'] > 30).sum()}")
    print(f"  low conf (<0.5)   : {(sample['pred_conf'] < 0.5).sum()}")
    print(f"  historical handoff: {sample['conv_handoff'].sum()}")
    print("\npredicted intent distribution:")
    print(sample["pred_intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
