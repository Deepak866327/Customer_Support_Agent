"""Train the intent classifier and build the retrieval index.

    python scripts/build_index.py

Reads  data/processed/train_pairs.parquet
Writes data/cache/retrieval_index.pkl, data/cache/classifier.pkl, data/cache/*.npy

First run downloads all-MiniLM-L6-v2 (~90 MB) and encodes ~24k messages: ~2 min on CPU.
Later runs hit the embedding cache and take seconds.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.classifier import EmbeddingClassifier  # noqa: E402
from src.intents import load_taxonomy  # noqa: E402
from src.retrieval import ResolutionIndex  # noqa: E402

PROC = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
MIN_WORDS = 4


def load_training_pool() -> pd.DataFrame:
    train = pd.read_parquet(PROC / "train_pairs.parquet")
    pool = train[
        train["is_opening"] & (train["customer_text"].str.split().str.len() >= MIN_WORDS)
    ].copy()
    return pool.drop_duplicates(subset="customer_text")


def main() -> None:
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    tx = load_taxonomy()

    pool = load_training_pool()
    pool["seed_intent"] = tx.seed_label_many(pool["customer_text"])
    seeded = pool[pool["seed_intent"].notna()].copy()

    print(f"[1/3] weak labelling: {len(seeded):,} / {len(pool):,} messages seeded "
          f"({100 * len(seeded) / len(pool):.1f}%)")
    for name, n in seeded["seed_intent"].value_counts().items():
        print(f"      {name:26s} {n:5d}")

    print("[2/3] training embedding classifier ...", flush=True)
    clf = EmbeddingClassifier().fit(seeded["customer_text"].tolist(),
                                    seeded["seed_intent"].tolist())
    with (CACHE / "classifier.pkl").open("wb") as fh:
        pickle.dump(clf, fh, protocol=5)

    print("[3/3] building retrieval index ...", flush=True)
    # Index every reusable resolution, not just rule-seeded ones. Seed labels are
    # needed to *train*; the index only needs an intent tag per entry, and the trained
    # classifier supplies that for the ~58% of messages no rule matched. Restricting
    # the index to seeded rows would have thrown away two thirds of the evidence.
    idx_rows = pool[pool["is_substantive"]].copy()
    idx_intents = idx_rows["seed_intent"].tolist()
    unseeded = idx_rows["seed_intent"].isna().to_numpy()
    if unseeded.any():
        predicted, _ = clf.predict(
            idx_rows.loc[unseeded, "customer_text"].tolist(), tag="index_unseeded"
        )
        it = iter(predicted)
        idx_intents = [
            next(it) if lab is None else lab for lab in idx_rows["seed_intent"].tolist()
        ]
    index = ResolutionIndex.build(idx_rows, idx_intents)
    index.save()

    summary = {
        "training_pool": len(pool),
        "seeded": len(seeded),
        "seed_rate": round(len(seeded) / len(pool), 4),
        "index_size": len(index),
        "index_seeded": int(idx_rows["seed_intent"].notna().sum()),
        "index_classifier_labelled": int(idx_rows["seed_intent"].isna().sum()),
        "classes": sorted(seeded["seed_intent"].unique().tolist()),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }
    (CACHE / "index_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\ndone in {time.time() - t0:.0f}s")
    print(f"  classifier -> {CACHE / 'classifier.pkl'}")
    print(f"  index      -> {CACHE / 'retrieval_index.pkl'} ({len(index):,} resolutions)")


if __name__ == "__main__":
    main()
