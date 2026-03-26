from pathlib import Path

from ingestion.loader import DocumentIngestor


def test_duplicate_documents_are_ignored(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("FastAPI supports async request handlers.", encoding="utf-8")
    (docs / "b.txt").write_text("FastAPI supports async request handlers.", encoding="utf-8")
    (docs / "c.txt").write_text("Dependency injection makes endpoint composition easier.", encoding="utf-8")

    ingestor = DocumentIngestor(chunk_size=10, chunk_overlap=2)
    loaded = ingestor.load_documents(docs, source="test")

    assert len(loaded) == 2
    filenames = {doc.file for doc in loaded}
    assert filenames == {"a.txt", "c.txt"}
