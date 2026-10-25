"""Build the SpotifyCares corpus from the raw twcs dump.

    python scripts/prepare_data.py

Reads  data/raw/twcs.csv  (516 MB, ~2.8M rows)
Writes data/processed/pairs.parquet, train_pairs.parquet, eval_pairs.parquet,
       conversations.parquet, stats.json

Takes ~90s. Everything downstream reads only the processed files, so this runs once.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    BRAND,
    assign_conversations,
    build_brand_corpus,
    build_pairs,
    corpus_stats,
    load_raw,
    time_split,
)

RAW = ROOT / "data" / "raw" / "twcs.csv"
OUT = ROOT / "data" / "processed"


def main() -> None:
    if not RAW.exists():
        sys.exit(
            f"Missing {RAW}.\nDownload it with:  python scripts/download_data.py"
        )
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1/5] loading {RAW.name} ...", flush=True)
    df = load_raw(RAW)
    print(f"      {len(df):,} tweets  ({time.time() - t0:.0f}s)")

    print("[2/5] reconstructing conversations (union-find) ...", flush=True)
    df = assign_conversations(df)
    print(f"      {df['conv_id'].nunique():,} conversations dataset-wide")

    print(f"[3/5] filtering to {BRAND} ...", flush=True)
    corpus = build_brand_corpus(df, BRAND)
    del df

    print("[4/5] building customer -> reply pairs ...", flush=True)
    pairs = build_pairs(corpus, BRAND)
    stats = corpus_stats(corpus, pairs, BRAND)

    print("[5/5] time-based conversation split ...", flush=True)
    train, evalp = time_split(pairs)

    keep = [
        "conv_id",
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "clean_text",
    ]
    corpus[keep].to_parquet(OUT / "conversations.parquet", index=False)
    pairs.to_parquet(OUT / "pairs.parquet", index=False)
    train.to_parquet(OUT / "train_pairs.parquet", index=False)
    evalp.to_parquet(OUT / "eval_pairs.parquet", index=False)

    summary = {
        "brand": BRAND,
        "conversations": stats.conversations,
        "customer_messages": stats.customer_messages,
        "brand_messages": stats.brand_messages,
        "pairs": stats.pairs,
        "opening_pairs": int(pairs["is_opening"].sum()),
        "substantive_pairs": int(pairs["is_substantive"].sum()),
        "conversation_handoff_rate": round(stats.handoff_rate, 4),
        "train_pairs": len(train),
        "eval_pairs": len(evalp),
        "train_conversations": int(train["conv_id"].nunique()),
        "eval_conversations": int(evalp["conv_id"].nunique()),
        "date_min": str(pairs["reply_at"].min()),
        "date_max": str(pairs["reply_at"].max()),
    }
    (OUT / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT}")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
