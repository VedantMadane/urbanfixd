from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
from hashlib import md5
import json
import re

import numpy as np
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback for unsupported runtimes
    faiss = None

from ingestion.loader import DocumentChunk


class LocalFallbackEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True):
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
            for token in tokens:
                h = int(md5(token.encode("utf-8")).hexdigest(), 16)
                vectors[i, h % self.dim] += 1.0
        return vectors


class LocalVectorStore:
    def __init__(
        self,
        index_dir: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedder: Any | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "docs.index"
        self.meta_path = self.index_dir / "metadata.json"
        self.model_name = model_name
        self.embedder = embedder
        self.index: Any | None = None
        self.metadata: list[dict] = []
        self._ids: set[str] = set()
        self._memory_vectors: np.ndarray | None = None

    def _ensure_embedder(self) -> Any:
        if self.embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.embedder = SentenceTransformer(self.model_name)
            except Exception:
                # Keep the service fully local and runnable even when heavy deps are unavailable.
                self.embedder = LocalFallbackEmbedder()
        return self.embedder

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vectors, dtype=np.float32)
        if faiss is not None:
            faiss.normalize_L2(vecs)
        else:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs

    def _new_index(self, dim: int) -> Any:
        if faiss is None:
            return {"dim": dim}
        return faiss.IndexFlatIP(dim)

    def _chunk_uid(self, metadata: dict) -> str:
        return f"{metadata['checksum']}::{metadata['chunk_id']}"

    def load(self) -> bool:
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        if faiss is not None:
            self.index = faiss.read_index(str(self.index_path))
        else:
            vectors_path = self.index_dir / "vectors.npy"
            if not vectors_path.exists():
                return False
            self._memory_vectors = np.load(vectors_path)
            self.index = self._new_index(self._memory_vectors.shape[1])
        self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._ids = {self._chunk_uid(item) for item in self.metadata}
        return True

    def save(self) -> None:
        if self.index is None:
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if faiss is not None:
            faiss.write_index(self.index, str(self.index_path))
        elif self._memory_vectors is not None:
            np.save(self.index_dir / "vectors.npy", self._memory_vectors)
        self.meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def add_documents(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        embedder = self._ensure_embedder()

        new_chunks: list[DocumentChunk] = []
        for chunk in chunks:
            uid = self._chunk_uid(chunk.metadata)
            if uid in self._ids:
                continue
            new_chunks.append(chunk)
            self._ids.add(uid)
        if not new_chunks:
            return 0

        texts = [chunk.text for chunk in new_chunks]
        vectors = embedder.encode(texts, convert_to_numpy=True)
        vectors = self._normalize(vectors)

        if self.index is None:
            self.index = self._new_index(vectors.shape[1])
        if faiss is not None:
            self.index.add(vectors)
        else:
            if self._memory_vectors is None:
                self._memory_vectors = vectors
            else:
                self._memory_vectors = np.vstack([self._memory_vectors, vectors])
        self.metadata.extend([asdict(chunk) for chunk in new_chunks])
        return len(new_chunks)

    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.index is None:
            return []
        embedder = self._ensure_embedder()
        query_vec = embedder.encode([query], convert_to_numpy=True)
        query_vec = self._normalize(query_vec)
        if faiss is not None:
            if self.index.ntotal == 0:
                return []
            scores, ids = self.index.search(query_vec, k=min(k, self.index.ntotal))
        else:
            if self._memory_vectors is None or len(self._memory_vectors) == 0:
                return []
            scores_1d = np.dot(self._memory_vectors, query_vec[0])
            order = np.argsort(-scores_1d)[: min(k, len(scores_1d))]
            scores = np.array([scores_1d[order]], dtype=np.float32)
            ids = np.array([order], dtype=np.int32)
        results: list[dict] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            payload = dict(self.metadata[idx])
            payload["score"] = float(score)
            results.append(payload)
        return results
