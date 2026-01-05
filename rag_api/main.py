# rag_api/main.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from ollama_client import generate as ollama_generate

from db import (
    EMBEDDING_DIM,
    Chunk as DbChunk,
    Document,
    get_db_session,
    init_db,
)
from embeddings import embed_text, embed_texts
from chunking import chunk_text

app = FastAPI(title="AI Knowledge Agent")


# -------------------------
# App lifecycle
# -------------------------
@app.on_event("startup")
def on_startup() -> None:
    # Creates pgvector extension + tables (safe to run multiple times)
    init_db()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ============================================================
# DEBUG ENDPOINTS (seed + search) — proves vector DB works
# ============================================================

class SeedRequest(BaseModel):
    title: str = Field(default="Debug Doc", description="Document title")
    num_chunks: int = Field(default=8, ge=1, le=200)
    chunk_text_prefix: str = Field(default="This is chunk", description="Prefix for seeded chunk text")


class SeedResponse(BaseModel):
    document_id: int
    chunk_ids: List[int]
    embedding_dim: int


class DebugSearchRequest(BaseModel):
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


@app.post("/debug/seed", response_model=SeedResponse)
def debug_seed(req: SeedRequest, db: Session = Depends(get_db_session)) -> SeedResponse:
    """
    Inserts a single Document and N chunks with REAL local embeddings (SentenceTransformers).
    This is just a test harness; it does not read uploaded documents.
    """
    # Create document
    doc = Document(title=req.title, source="debug", content_type="text/plain")
    db.add(doc)
    db.flush()  # get doc.id

    # Create deterministic-ish chunk text
    texts: List[str] = []
    for i in range(req.num_chunks):
        # NOTE: keep topic tokens stable for debugging
        texts.append(f"{req.chunk_text_prefix} #{i}. It mentions topic_{i % 3}.")

    # Batch embed
    embs = embed_texts(texts)  # List[List[float]]

    # Insert chunks
    chunk_ids: List[int] = []
    for i, (text, emb) in enumerate(zip(texts, embs)):
        row = DbChunk(
            document_id=doc.id,
            chunk_index=i,
            chunk_text=text,
            embedding=emb,
            chunk_metadata={"source": "debug", "topic": f"topic_{i % 3}"},
        )
        db.add(row)
        db.flush()
        chunk_ids.append(row.id)

    db.commit()
    return SeedResponse(document_id=doc.id, chunk_ids=chunk_ids, embedding_dim=EMBEDDING_DIM)


@app.post("/debug/search", response_model=SearchResponse)
def debug_search(req: DebugSearchRequest, db: Session = Depends(get_db_session)) -> SearchResponse:
    """
    Vector similarity search using a REAL embedded query (SentenceTransformers).
    """
    query_vec = embed_text(req.query)

    # cosine distance (lower is better)
    distance = DbChunk.embedding.cosine_distance(query_vec).label("distance")

    stmt = select(DbChunk, distance)

    if req.document_id is not None:
        stmt = stmt.where(DbChunk.document_id == req.document_id)

    stmt = stmt.order_by(distance.asc()).limit(req.top_k)
    rows = db.execute(stmt).all()

    hits: List[SearchHit] = []
    for chunk, dist in rows:
        # Display-friendly score (higher is better)
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


# ============================================================
# REAL INGEST + REAL SEARCH (Milestone: semantic search on your data)
# ============================================================

class IngestRequest(BaseModel):
    title: str = Field(default="Untitled")
    text: str = Field(min_length=1)
    source: str = Field(default="raw_text")
    chunk_size: int = Field(default=800, ge=200, le=4000)
    overlap: int = Field(default=150, ge=0, le=2000)


class IngestResponse(BaseModel):
    document_id: int
    num_chunks: int
    embedding_dim: int


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, db: Session = Depends(get_db_session)) -> IngestResponse:
    """
    Real ingestion pipeline:
    raw text -> chunks -> embeddings -> store (documents + chunks)
    """
    # 1) Create Document row
    doc = Document(title=req.title, source=req.source, content_type="text/plain")
    db.add(doc)
    db.flush()

    # 2) Chunk text
    chunks = chunk_text(req.text, chunk_size=req.chunk_size, overlap=req.overlap)
    if not chunks:
        db.commit()
        return IngestResponse(document_id=doc.id, num_chunks=0, embedding_dim=EMBEDDING_DIM)

    texts = [c.text for c in chunks]

    # 3) Embed chunks (batch)
    embs = embed_texts(texts)

    # 4) Store chunks
    for c, emb in zip(chunks, embs):
        row = DbChunk(
            document_id=doc.id,
            chunk_index=c.index,
            chunk_text=c.text,
            embedding=emb,
            chunk_metadata={
                "source": req.source,
                "title": req.title,
                "chunk_size": req.chunk_size,
                "overlap": req.overlap,
            },
        )
        db.add(row)

    db.commit()
    return IngestResponse(document_id=doc.id, num_chunks=len(chunks), embedding_dim=EMBEDDING_DIM)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: Optional[int] = None


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, db: Session = Depends(get_db_session)) -> SearchResponse:
    """
    Real semantic search:
    query text -> query embedding -> pgvector similarity -> top-k chunks
    """
    query_vec = embed_text(req.query)
    distance = DbChunk.embedding.cosine_distance(query_vec).label("distance")

    stmt = select(DbChunk, distance)

    if req.document_id is not None:
        stmt = stmt.where(DbChunk.document_id == req.document_id)

    stmt = stmt.order_by(distance.asc()).limit(req.top_k)
    rows = db.execute(stmt).all()

    hits: List[SearchHit] = []
    for chunk, dist in rows:
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

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: Optional[int] = None


class Citation(BaseModel):
    chunk_id: int
    document_id: int
    chunk_index: int
    snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]


def _build_rag_prompt(question: str, hits: List[SearchHit]) -> str:
    """
    Builds a grounded prompt with explicit sources.
    The model should answer using ONLY sources.
    """
    sources_lines = []
    for i, h in enumerate(hits, start=1):
        # Keep the source text bounded
        text = h.chunk_text.strip()
        sources_lines.append(
            f"[S{i}] (doc={h.document_id}, chunk={h.chunk_id}, idx={h.chunk_index}, score={h.score:.3f})\n"
            f"{text}\n"
        )

    sources_block = "\n".join(sources_lines)

    return f"""You are a helpful assistant. Answer the user's question using ONLY the sources provided.
If the sources do not contain enough information, say "I don't know based on the provided documents."
Cite sources inline using [S1], [S2], etc.

Question:
{question}

Sources:
{sources_block}

Answer:"""


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db_session)) -> ChatResponse:
    """
    RAG chat:
    question -> embed -> retrieve top-k -> call Ollama -> return answer + citations.
    """
    # Reuse your existing search logic (inline here)
    query_vec = embed_text(req.question)
    distance = DbChunk.embedding.cosine_distance(query_vec).label("distance")

    stmt = select(DbChunk, distance)
    if req.document_id is not None:
        stmt = stmt.where(DbChunk.document_id == req.document_id)

    stmt = stmt.order_by(distance.asc()).limit(req.top_k)
    rows = db.execute(stmt).all()

    hits: List[SearchHit] = []
    citations: List[Citation] = []

    for chunk, dist in rows:
        score = float(1.0 - dist)
        hit = SearchHit(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            score=score,
            chunk_text=chunk.chunk_text,
            metadata=chunk.chunk_metadata or {},
        )
        hits.append(hit)

        citations.append(
            Citation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                snippet=chunk.chunk_text[:180],
                score=score,
            )
        )

    prompt = _build_rag_prompt(req.question, hits)
    answer = ollama_generate(prompt)

    return ChatResponse(answer=answer, citations=citations)
