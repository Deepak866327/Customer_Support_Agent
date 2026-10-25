"""Sentence embeddings with an on-disk cache.

One small model (`all-MiniLM-L6-v2`, 384-d, ~90 MB) serves both the classifier and the
retriever. Encoding the 27k-message corpus takes ~60s on CPU and is cached, so every
later run is instant and evaluation costs nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"

_model = None


def _get_model():
    global _model
    if _model is None:
        import os

        # Once downloaded, load fully offline: otherwise the hub is re-checked with ~20
        # HTTP requests on every load, which took ~35s on a slow connection. The flag is
        # read when huggingface_hub is imported, so the cache is checked by path.
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        if any((hub / f"models--{MODEL_NAME.replace('/', '--')}" / "snapshots").glob("*/config.json")):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _key(texts: list[str], tag: str) -> str:
    h = hashlib.sha256()
    h.update(MODEL_NAME.encode())
    h.update(tag.encode())
    h.update(str(len(texts)).encode())
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def encode(texts, tag: str = "", use_cache: bool = True) -> np.ndarray:
    """L2-normalised embeddings, shape (n, 384). Cosine similarity == dot product."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"emb_{tag}_{_key(texts, tag)}.npy"
    if use_cache and path.exists():
        return np.load(path)

    vecs = _get_model().encode(
        texts,
        batch_size=256,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 2000,
    ).astype(np.float32)

    if use_cache:
        np.save(path, vecs)
    return vecs
