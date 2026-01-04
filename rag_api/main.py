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
    # Provide your own query vector if you want. Otherwise we generate a random one.
    query_vector: Optional[List[float]] = None
    top_k: int = Field(default=5, ge=1, le=50)
    # Optional: restrict search to a document
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
    Inserts a single Document and N chunks with random embeddings.
    This proves: DB is writable + pgvector column works.
    """
    if req.rng_seed is not None:
        random.seed(req.rng_seed)

    doc = Document(title=req.title, source="debug", content_type="text/plain")
    db.add(doc)
    db.flush()  # get doc.id without committing

    chunk_ids: List[int] = []
    for i in range(req.num_chunks):
        text = f"{req.chunk_text_prefix} #{i}. It mentions topic {i % 3}."
        emb = _random_vector(EMBEDDING_DIM)

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
    Runs a vector similarity search over chunks.
    This proves: retrieval works (ORDER BY vector distance).
    """
    # Validate / generate query vector
    if req.query_vector is None:
        query_vec = _random_vector(EMBEDDING_DIM)
    else:
        query_vec = req.query_vector
        if len(query_vec) != EMBEDDING_DIM:
            raise ValueError(f"query_vector must have length {EMBEDDING_DIM}, got {len(query_vec)}")

    # Build query:
    # Using cosine distance (<=>) for similarity. Lower is better.
    # We'll convert to a "score" as (1 - distance) for display, but note it's not a strict cosine similarity.
    distance = Chunk.embedding.cosine_distance(query_vec).label("distance")

    stmt = select(Chunk, distance)

    if req.document_id is not None:
        stmt = stmt.where(Chunk.document_id == req.document_id)

    stmt = stmt.order_by(distance.asc()).limit(req.top_k)

    rows = db.execute(stmt).all()

    hits: List[SearchHit] = []
    for chunk, dist in rows:
        score = float(1.0 - dist)  # display-friendly; closer => higher score
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
