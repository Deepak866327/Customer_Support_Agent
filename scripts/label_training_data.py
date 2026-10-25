"""Label the training pool with an LLM, replacing keyword weak supervision.

    python scripts/label_training_data.py [--n 4000]

Why this exists: the first version trained on regex weak labels. Measured against the
200 hand-labelled golden examples, those rules fired on 51% of messages and were right
72.5% of the time - so every model trained on them was capped at reproducing regexes.
The diagnostic that proved it (evaluation/results/ from the --no-llm run):

    slice              TF-IDF   embeddings
    rule-fired  (102)   0.745      0.676
    rule-missed  (98)   0.459      0.469

TF-IDF only won where the rules had fired, i.e. it had memorised them. On messages the
rules never matched, both sat at ~0.46. The label source was the bottleneck.

Messages are labelled 20 per request against the same taxonomy a human would read, and
every response is cached by content hash, so re-running costs nothing. The regex rules
survive as a cheap sanity check, not as ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.intents import load_taxonomy  # noqa: E402
from src.llm import LABEL_MODEL, LLMClient, parse_json  # noqa: E402

BATCH = 20
OUT = ROOT / "data" / "processed" / "train_labels.jsonl"
SEED = 42


def build_system(tx) -> str:
    lines = [
        "You label incoming customer messages sent to Spotify's Twitter support account "
        "(2017) with exactly one intent from this taxonomy.",
        "",
    ]
    for i in tx.intents:
        lines.append(f"- {i.name}: {i.description}")
        for ex in i.examples[:2]:
            lines.append(f'    e.g. "{ex[:130]}"')
    lines += [
        "",
        "Rules:",
        "- Choose the ONE intent matching the customer's primary need.",
        "- If a message mixes issues, label the one they most want solved.",
        "- Use `other` only when no intent genuinely fits: off-topic chatter, "
        "context-free fragments, or pure venting with no identifiable request.",
        "- Non-English messages get the same treatment; translate mentally first.",
        "",
        'Return ONLY a JSON array, one object per input, in order: '
        '[{"i": <index>, "intent": "<name>"}, ...]',
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="messages to label")
    ap.add_argument("--model", default=LABEL_MODEL)
    args = ap.parse_args()

    tx = load_taxonomy()
    valid = set(tx.names)
    system = build_system(tx)

    train = pd.read_parquet(ROOT / "data" / "processed" / "train_pairs.parquet")
    pool = train[
        train["is_opening"] & (train["customer_text"].str.split().str.len() >= 4)
    ].drop_duplicates(subset="customer_text")

    # Stratify so rare intents are represented: half drawn from rule-seeded messages
    # (balanced across seeds), half from messages no rule matched.
    pool = pool.assign(seed=tx.seed_label_many(pool["customer_text"]))
    seeded, unseeded = pool[pool["seed"].notna()], pool[pool["seed"].isna()]
    per = max(1, (args.n // 2) // max(1, seeded["seed"].nunique()))
    part_a = seeded.groupby("seed", group_keys=False).apply(
        lambda g: g.sample(min(per, len(g)), random_state=SEED), include_groups=False
    )
    part_b = unseeded.sample(min(args.n - len(part_a), len(unseeded)), random_state=SEED)
    sample = pd.concat([part_a, part_b]).sample(frac=1.0, random_state=SEED)
    texts = sample["customer_text"].tolist()
    print(f"labelling {len(texts):,} messages ({len(part_a):,} seeded + {len(part_b):,} unseeded)")

    client = LLMClient(model=args.model)
    labels: list[str | None] = [None] * len(texts)
    t0 = time.time()

    for start in range(0, len(texts), BATCH):
        chunk = texts[start : start + BATCH]
        prompt = "\n".join(f"[{i}] {t}" for i, t in enumerate(chunk))
        try:
            raw = client.complete(prompt, system=system, max_tokens=1200)
            parsed = json.loads(raw[raw.find("[") : raw.rfind("]") + 1])
            for item in parsed:
                i = int(item["i"])
                name = str(item["intent"]).strip()
                if 0 <= i < len(chunk) and name in valid:
                    labels[start + i] = name
        except Exception as exc:
            print(f"  batch {start}: {type(exc).__name__}: {exc}")
        done = min(start + BATCH, len(texts))
        if done % 400 == 0 or done == len(texts):
            got = sum(l is not None for l in labels)
            print(f"  {done}/{len(texts)}  labelled={got}  "
                  f"cache={client.hits} live={client.misses}  {time.time() - t0:.0f}s")

    with OUT.open("w", encoding="utf-8") as fh:
        for text, seed, lab in zip(texts, sample["seed"].tolist(), labels):
            if lab is None:
                continue
            fh.write(json.dumps(
                {"customer_text": text, "intent": lab, "rule_intent": seed}, ensure_ascii=False
            ) + "\n")

    kept = sum(l is not None for l in labels)
    print(f"\nwrote {kept:,} labels -> {OUT}")
    dist = pd.Series([l for l in labels if l]).value_counts()
    print("\nlabel distribution:")
    for k, v in dist.items():
        print(f"  {k:26s} {v:5d}")

    both = [(s, l) for s, l in zip(sample["seed"].tolist(), labels) if s and l]
    if both:
        agree = sum(s == l for s, l in both)
        print(f"\nagreement with regex rules where both exist: "
              f"{agree}/{len(both)} ({100 * agree / len(both):.1f}%)")


if __name__ == "__main__":
    main()
