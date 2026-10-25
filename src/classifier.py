"""Intent classifiers: two baselines and the proposed system.

All three share one interface so `evaluation/run.py` can treat them identically:

    fit(texts, labels) -> self
    predict(texts)     -> (labels, confidences)

`other` is never a trained class. It is emitted as an abstention when no intent clears
`min_confidence`, which is also what routes a message to escalation. See
`src.intents.Taxonomy.seed_label` for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.embeddings import encode
from src.intents import OTHER

SEED = 42


@dataclass
class MajorityBaseline:
    """Trivial baseline: always predict the most frequent training intent."""

    name: str = "majority"
    label_: str = OTHER
    prior_: float = 0.0

    def fit(self, texts, labels):
        vals, counts = np.unique(np.asarray(labels), return_counts=True)
        self.label_ = str(vals[counts.argmax()])
        self.prior_ = float(counts.max() / counts.sum())
        return self

    def predict(self, texts) -> tuple[list[str], np.ndarray]:
        n = len(list(texts))
        return [self.label_] * n, np.full(n, self.prior_, dtype=np.float32)


@dataclass
class TfidfBaseline:
    """Simple ML baseline: word+char TF-IDF into multinomial logistic regression."""

    name: str = "tfidf_lr"
    min_confidence: float = 0.0
    vec_: TfidfVectorizer | None = None
    clf_: LogisticRegression | None = None

    def fit(self, texts, labels):
        self.vec_ = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode"
        )
        X = self.vec_.fit_transform(texts)
        self.clf_ = LogisticRegression(
            max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED
        )
        self.clf_.fit(X, labels)
        return self

    def predict(self, texts) -> tuple[list[str], np.ndarray]:
        proba = self.clf_.predict_proba(self.vec_.transform(texts))
        return _decode(proba, self.clf_.classes_, self.min_confidence)


@dataclass
class EmbeddingClassifier:
    """Proposed system: frozen sentence embeddings + logistic regression.

    Logistic regression on embeddings rather than k-NN: it gives calibrated-ish
    probabilities, which the escalation layer needs as a confidence signal, and k-NN
    over weak labels amplifies seed noise. No fine-tuning - there is no clean labelled
    set large enough to justify it, and the frozen encoder already separates these
    intents.
    """

    name: str = "embed_lr"
    min_confidence: float = 0.45
    clf_: LogisticRegression | None = None
    classes_: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, texts, labels):
        X = encode(list(texts), tag="train")
        self.clf_ = LogisticRegression(
            max_iter=3000, C=8.0, class_weight="balanced", random_state=SEED
        )
        self.clf_.fit(X, labels)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, texts, tag: str = "pred") -> np.ndarray:
        return self.clf_.predict_proba(encode(list(texts), tag=tag))

    def predict(self, texts, tag: str = "pred") -> tuple[list[str], np.ndarray]:
        return _decode(self.predict_proba(texts, tag), self.classes_, self.min_confidence)

    def predict_vectors(self, X: np.ndarray) -> tuple[list[str], np.ndarray]:
        """`predict` on pre-computed embeddings, so live chat need not hit the disk cache."""
        return _decode(self.clf_.predict_proba(X), self.classes_, self.min_confidence)


def _decode(
    proba: np.ndarray, classes: np.ndarray, min_confidence: float
) -> tuple[list[str], np.ndarray]:
    """Argmax, with abstention to `other` below the confidence floor."""
    best = proba.argmax(axis=1)
    conf = proba[np.arange(len(proba)), best].astype(np.float32)
    labels = [
        OTHER if c < min_confidence else str(classes[b]) for b, c in zip(best, conf)
    ]
    return labels, conf


def build(name: str):
    return {
        "majority": MajorityBaseline,
        "tfidf_lr": TfidfBaseline,
        "embed_lr": EmbeddingClassifier,
    }[name]()
