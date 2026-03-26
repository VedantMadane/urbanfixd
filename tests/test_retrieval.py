from pathlib import Path
import re

import numpy as np

from ingestion.loader import DocumentChunk
from retrieval.vector_store import LocalVectorStore


class FakeEmbedder:
    def __init__(self) -> None:
        self.vocab = [
            "fastapi",
            "async",
            "request",
            "python",
            "typing",
            "dependency",
            "injection",
        ]

    def encode(self, texts, convert_to_numpy=True):
        vectors = []
        for text in texts:
            tokens = re.findall(r"[a-zA-Z]+", text.lower())
            vectors.append([tokens.count(word) for word in self.vocab])
        return np.array(vectors, dtype=np.float32)


def test_retriever_returns_relevant_chunk(tmp_path: Path) -> None:
    store = LocalVectorStore(index_dir=tmp_path / "index", embedder=FakeEmbedder())
    chunks = [
        DocumentChunk(
            text="FastAPI handles async request functions with the ASGI event loop.",
            metadata={"source": "test", "file": "fastapi.md", "chunk_id": 0, "checksum": "x1"},
        ),
        DocumentChunk(
            text="Python typing provides Optional, Union, and generic containers.",
            metadata={"source": "test", "file": "python.md", "chunk_id": 0, "checksum": "x2"},
        ),
    ]

    added = store.add_documents(chunks)
    assert added == 2

    results = store.search("How do async requests work in FastAPI?", k=1)

    assert len(results) == 1
    assert results[0]["metadata"]["file"] == "fastapi.md"
