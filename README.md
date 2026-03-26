# UrbanFixd - AI Agent Systems Take-Home

Small production-style AI knowledge assistant that answers questions from a local document dataset.
The implementation focuses on clean architecture, local execution, and testability.

## Tech Stack

- Python + FastAPI
- Local vector index with FAISS
- `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Pytest for automated testing

## Project Structure

```text
urbanfixd/
  api/routes.py
  ingestion/loader.py
  retrieval/vector_store.py
  agents/qa_agent.py
  tests/
  data/docs/
  main.py
  README.md
```

## Architecture Decisions

1. **Pipeline separation by responsibility**
   - `ingestion/loader.py`: load, clean, chunk, and attach metadata.
   - `retrieval/vector_store.py`: embed chunks, persist local FAISS index, retrieve top-k contexts.
   - `agents/qa_agent.py`: reasoning layer that builds a prompt, ranks context sentences, and returns answer + citations.
   - `api/routes.py`: minimal HTTP interface with dependency injection.

2. **Local-first operation**
   - All data is stored in `data/docs`.
   - FAISS index and metadata are persisted to `data/index`.
   - No paid APIs or external hosted services are required.

3. **Deterministic tests**
   - Retrieval tests use a fake embedder to avoid network/model download during CI.
   - API tests override the QA dependency to keep tests fast and stable.

## Data and Metadata

The dataset includes 15 local markdown knowledge pages in `data/docs`.
Each chunk carries metadata:

```json
{ "source": "local_docs", "file": "01_fastapi_async.md", "chunk_id": 0, "checksum": "..." }
```

## API

### `GET /health`

Returns service status:

```json
{ "status": "ok" }
```

### `POST /query`

Request:

```json
{ "question": "How does FastAPI handle async requests?" }
```

Response:

```json
{
  "answer": "FastAPI runs on ASGI servers and async handlers await non-blocking I/O...",
  "sources": [
    { "file": "01_fastapi_async.md", "chunk_id": 0, "score": 0.82 }
  ]
}
```

## Setup and Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Tests

```bash
pytest -q
```

Test coverage includes:
- retriever returns relevant document
- API endpoint returns response
- duplicate documents are ignored

## Tradeoffs Due to Time Limits

- The answer generator uses lightweight extractive reasoning instead of a full local LLM pipeline.
- FAISS index is exact (`IndexFlatIP`) for simplicity over advanced ANN configurations.
- No async background reindexing yet; index is built lazily on first query if absent.

## How This Would Scale for Production

- Split ingestion/indexing into asynchronous worker jobs.
- Move vector storage to a distributed vector database (or FAISS sharded service).
- Add a model-serving layer for embeddings with batching and GPU acceleration.
- Add caching for repeated queries and top-k retrieval results.
- Add observability: tracing, retrieval metrics, answer quality monitoring, and alerting.
- Add auth, rate limiting, request validation hardening, and deployment orchestration.

## Final Question: 1M Documents and 500 RPS Redesign

To support 1 million documents and 500 requests/second, redesign into separate online and offline systems:

1. **Offline indexing plane**
   - Stream documents into a durable queue.
   - Distributed workers perform cleaning/chunking/embedding.
   - Store chunk metadata in object storage + relational metadata store.
   - Build sharded ANN indexes (IVF/HNSW/PQ) and publish versioned snapshots.

2. **Online query plane**
   - Stateless API pods behind a load balancer.
   - Dedicated retrieval service that fans out queries to vector shards.
   - Query understanding/reranking stage for better precision.
   - Caching layer for hot queries and precomputed embeddings.

3. **Reliability and operations**
   - Blue/green index rollout with rollback.
   - Circuit breakers and fallback behavior when retrieval is degraded.
   - End-to-end tracing and SLOs on latency, retrieval recall, and answer quality.
