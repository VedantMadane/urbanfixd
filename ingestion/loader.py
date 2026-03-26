from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable
import re


@dataclass(frozen=True)
class RawDocument:
    text: str
    source: str
    file: str
    checksum: str


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    metadata: dict


class DocumentIngestor:
    def __init__(self, chunk_size: int = 120, chunk_overlap: int = 24) -> None:
        if chunk_size <= chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk_overlap")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, data_dir: str | Path, source: str = "local_docs") -> list[RawDocument]:
        root = Path(data_dir)
        if not root.exists():
            return []

        dedup_checksums: set[str] = set()
        docs: list[RawDocument] = []
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            cleaned = self.clean_text(raw_text)
            if not cleaned:
                continue
            digest = sha256(cleaned.encode("utf-8")).hexdigest()
            if digest in dedup_checksums:
                continue
            dedup_checksums.add(digest)
            docs.append(
                RawDocument(
                    text=cleaned,
                    source=source,
                    file=path.name,
                    checksum=digest,
                )
            )
        return docs

    def clean_text(self, text: str) -> str:
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk_document(self, document: RawDocument) -> Iterable[DocumentChunk]:
        words = document.text.split()
        step = self.chunk_size - self.chunk_overlap
        chunk_id = 0
        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size]
            if not window:
                continue
            yield DocumentChunk(
                text=" ".join(window),
                metadata={
                    "source": document.source,
                    "file": document.file,
                    "chunk_id": chunk_id,
                    "checksum": document.checksum,
                },
            )
            chunk_id += 1

    def ingest(self, data_dir: str | Path, source: str = "local_docs") -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for doc in self.load_documents(data_dir=data_dir, source=source):
            chunks.extend(self.chunk_document(doc))
        return chunks
