"""Retrieve historical (customer message -> Spotify resolution) pairs.

Index contents are deliberately restricted to *substantive* replies: not a bare
"please DM us", and >= 12 words. A handoff is not a resolution, so indexing one would
let the generator ground a reply in evidence that contains no answer.

sklearn NearestNeighbors on 384-d vectors over a ~20k index, not FAISS: exact cosine
search at this size is ~5 ms and adds no dependency.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.embeddings import encode

INDEX_PATH = Path(__file__).resolve().parents[1] / "data" / "cache" / "retrieval_index.pkl"


@dataclass(frozen=True)
class Evidence:
    """One retrieved historical resolution."""

    evidence_id: str
    customer_message: str
    resolution: str
    intent: str
    similarity: float
    was_handoff: bool

    def as_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "customer_message": self.customer_message,
            "resolution": self.resolution,
            "intent": self.intent,
            "similarity": round(self.similarity, 4),
            "was_handoff": self.was_handoff,
        }


class ResolutionIndex:
    def __init__(
        self,
        ids: list[str],
        customer: list[str],
        resolution: list[str],
        intents: list[str],
        handoff: np.ndarray,
        vectors: np.ndarray,
    ) -> None:
        self.ids = ids
        self.customer = customer
        self.resolution = resolution
        self.intents = intents
        self.handoff = handoff
        self.vectors = vectors
        self._nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(vectors)

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def build(cls, pairs: pd.DataFrame, intents: list[str]) -> "ResolutionIndex":
        """`pairs` must already be filtered to substantive, opening-message rows."""
        vectors = encode(pairs["customer_text"].tolist(), tag="index")
        return cls(
            ids=pairs["pair_id"].astype(str).tolist(),
            customer=pairs["customer_text"].tolist(),
            resolution=pairs["reply_text"].tolist(),
            intents=list(intents),
            handoff=pairs["conv_handoff"].to_numpy(dtype=bool),
            vectors=vectors,
        )

    def search(
        self,
        query: str,
        k: int = 5,
        intent: str | None = None,
        pool: int = 60,
        query_vec: np.ndarray | None = None,
    ) -> list[Evidence]:
        """Top-k by cosine, optionally re-ranked to prefer the predicted intent.

        Intent is a soft filter, not a hard one: retrieving over-narrowly on a wrong
        predicted intent is worse than retrieving broadly, because the escalation layer
        can see inconsistent evidence but cannot see evidence that was filtered away.
        """
        vec = encode([query], tag="query")[0] if query_vec is None else query_vec
        pool = max(pool, k)
        dist, idx = self._nn.kneighbors(vec.reshape(1, -1), n_neighbors=min(pool, len(self)))
        dist, idx = dist[0], idx[0]

        hits = [
            Evidence(
                evidence_id=self.ids[i],
                customer_message=self.customer[i],
                resolution=self.resolution[i],
                intent=self.intents[i],
                similarity=float(1.0 - d),
                was_handoff=bool(self.handoff[i]),
            )
            for d, i in zip(dist, idx)
        ]
        if intent is not None:
            hits.sort(key=lambda e: (e.intent != intent, -e.similarity))
        return hits[:k]

    def save(self, path: Path = INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "ids": self.ids,
                    "customer": self.customer,
                    "resolution": self.resolution,
                    "intents": self.intents,
                    "handoff": self.handoff,
                    "vectors": self.vectors,
                },
                fh,
                protocol=5,
            )

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "ResolutionIndex":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing - run:  python scripts/build_index.py"
            )
        with path.open("rb") as fh:
            d = pickle.load(fh)
        return cls(**d)


def evidence_agreement(hits: list[Evidence]) -> float:
    """Share of retrieved evidence sharing the modal intent.

    The escalation layer's proxy for "do these historical cases actually agree?".
    1.0 means every neighbour was the same kind of problem; low values mean the
    retriever landed in a mixed neighbourhood and the evidence contradicts itself.
    """
    if not hits:
        return 0.0
    counts: dict[str, int] = {}
    for h in hits:
        counts[h.intent] = counts.get(h.intent, 0) + 1
    return max(counts.values()) / len(hits)
