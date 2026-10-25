"""Baseline systems for each stage of the pipeline.

Intent baselines live in src/classifier.py (MajorityBaseline, TfidfBaseline) because the
agent needs to be able to run with any of them. This module holds the baselines that
exist only for evaluation: retrieval and reply generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.generator import Draft
from src.retrieval import Evidence, ResolutionIndex


@dataclass
class TfidfRetriever:
    """Simple retrieval baseline: lexical TF-IDF cosine over the same index.

    Isolates how much the sentence encoder is actually worth. If dense retrieval does
    not beat this, the embedding model is not earning its dependency.
    """

    index: ResolutionIndex
    name: str = "tfidf_retrieval"

    def __post_init__(self) -> None:
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self.index.customer)

    def search(self, query: str, k: int = 5, intent: str | None = None) -> list[Evidence]:
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = np.argsort(sims)[::-1][:k]
        return [
            Evidence(
                evidence_id=self.index.ids[i],
                customer_message=self.index.customer[i],
                resolution=self.index.resolution[i],
                intent=self.index.intents[i],
                similarity=float(sims[i]),
                was_handoff=bool(self.index.handoff[i]),
            )
            for i in top
        ]


def verbatim_reply(evidence: list[Evidence]) -> Draft:
    """Trivial reply baseline: return the single closest historical reply, unchanged.

    No LLM at all. This is the bar the generator has to clear: if paraphrasing the top
    hit with a model is no better than pasting it, the generation step is decoration.
    """
    if not evidence:
        return Draft(reply="", confidence=0.0, evidence_ids=[])
    top = evidence[0]
    return Draft(reply=top.resolution, confidence=top.similarity, evidence_ids=[top.evidence_id])
