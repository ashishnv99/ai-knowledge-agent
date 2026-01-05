from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db import (
    EMBEDDING_DIM,
    Chunk,
    Document,
    get_db_session,
    init_db,
)

from embeddings import embed_text, embed_texts


# pgvector distance helpers (SQLAlchemy)
from pgvector.sqlalchemy import Vector
from sqlalchemy import select

app = FastAPI(title="AI Knowledge Agent")


@app.on_event("startup")
def on_startup() -> None:
    # Creates pgvector extension + tables
    init_db()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# -------------------------
# Debug models
# -------------------------
class SeedRequest(BaseModel):
    title: str = Field(default="Debug Doc", description="Document title")
    num_chunks: int = Field(default=8, ge=1, le=200)
    chunk_text_prefix: str = Field(default="This is chunk", description="Prefix for seeded chunk text")
    # If you want deterministic vectors for repeatable testing, set a seed.
    rng_seed: Optional[int] = Field(default=42)


class SeedResponse(BaseModel):
    document_id: int
    chunk_ids: List[int]
    embedding_dim: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: Optional[int] = None


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    chunk_index: int
    score: float
    chunk_text: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    embedding_dim: int
    top_k: int
    hits: List[SearchHit]


# -------------------------
# Helpers
# -------------------------
def _random_vector(dim: int) -> List[float]:
    # Using uniform random floats. You can switch to normal distribution too.
    return [random.random() for _ in range(dim)]


# -------------------------
# Step 4: Debug endpoints
# -------------------------

@app.post("/debug/seed", response_model=SeedResponse)
def debug_seed(req: SeedRequest, db: Session = Depends(get_db_session)) -> SeedResponse:
    """
    Inserts a single Document and N chunks with REAL local embeddings (SentenceTransformers).
    This proves: ingestion-style embedding + DB insert works.
    """
    # Create document
    doc = Document(title=req.title, source="debug", content_type="text/plain")
    db.add(doc)
    db.flush()  # get doc.id

    # Build chunk texts
    texts: List[str] = []
    for i in range(req.num_chunks):
        texts.append(f"{req.chunk_text_prefix} #{i}. It mentions topic_{i % 3}.")

    # Create embeddings in one batch (fast)
    embs = embed_texts(texts)  # List[List[float]]

    # Insert chunks
    chunk_ids: List[int] = []
    for i, (text, emb) in enumerate(zip(texts, embs)):
        chunk = Chunk(
            document_id=doc.id,
            chunk_index=i,
            chunk_text=text,
            embedding=emb,
            chunk_metadata={"source": "debug", "topic": f"topic_{i % 3}"},
        )
        db.add(chunk)
        db.flush()
        chunk_ids.append(chunk.id)

    db.commit()

    return SeedResponse(document_id=doc.id, chunk_ids=chunk_ids, embedding_dim=EMBEDDING_DIM)


@app.post("/debug/search", response_model=SearchResponse)
def debug_search(req: SearchRequest, db: Session = Depends(get_db_session)) -> SearchResponse:
    """
    Runs a vector similarity search using a REAL embedded query (SentenceTransformers).
    This proves: query->embedding->pgvector search works.
    """
    query_vec = embed_text(req.query)

    # cosine distance (lower is better)
    distance = Chunk.embedding.cosine_distance(query_vec).label("distance")

    stmt = select(Chunk, distance)

    if req.document_id is not None:
        stmt = stmt.where(Chunk.document_id == req.document_id)

    stmt = stmt.order_by(distance.asc()).limit(req.top_k)

    rows = db.execute(stmt).all()

    hits: List[SearchHit] = []
    for chunk, dist in rows:
        # Display-friendly score. Not a strict cosine similarity, but higher is better.
        score = float(1.0 - dist)

        hits.append(
            SearchHit(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                score=score,
                chunk_text=chunk.chunk_text,
                metadata=chunk.chunk_metadata or {},
            )
        )

    return SearchResponse(embedding_dim=EMBEDDING_DIM, top_k=req.top_k, hits=hits)
