# rag_api/chunking.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    """
    Naive char-based chunking with overlap.
    - chunk_size: number of characters per chunk
    - overlap: number of characters to overlap between chunks
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    n = len(cleaned)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(Chunk(index=idx, text=chunk))
            idx += 1

        if end == n:
            break
        start = end - overlap

    return chunks
