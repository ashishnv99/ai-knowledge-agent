from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Loads once per process (fast after first load)
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Returns normalized embeddings for a batch of texts.
    all-MiniLM-L6-v2 -> 384 dims.
    """
    m = _model()
    vectors = m.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]
