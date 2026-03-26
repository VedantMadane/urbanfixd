from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from agents.qa_agent import QAAgent
from ingestion.loader import DocumentIngestor
from retrieval.vector_store import LocalVectorStore

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_agent() -> QAAgent:
    root = _project_root()
    docs_dir = root / "data" / "docs"
    index_dir = root / "data" / "index"

    ingestor = DocumentIngestor()
    store = LocalVectorStore(index_dir=index_dir)

    loaded = store.load()
    if not loaded:
        chunks = ingestor.ingest(docs_dir, source="local_docs")
        store.add_documents(chunks)
        store.save()
    return QAAgent(store)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, agent: QAAgent = Depends(get_agent)) -> QueryResponse:
    result = agent.answer(payload.question)
    return QueryResponse(**result)
