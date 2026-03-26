from __future__ import annotations

import re
from collections import Counter

from retrieval.vector_store import LocalVectorStore


class QAAgent:
    def __init__(self, vector_store: LocalVectorStore, top_k: int = 4) -> None:
        self.vector_store = vector_store
        self.top_k = top_k

    def _tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z]{3,}", text.lower()))

    def _rank_sentences(self, question: str, contexts: list[str]) -> list[str]:
        q_tokens = self._tokens(question)
        sentences: list[str] = []
        for context in contexts:
            sentences.extend(re.split(r"(?<=[.!?])\s+", context))
        ranked = sorted(
            (s.strip() for s in sentences if s.strip()),
            key=lambda s: (len(self._tokens(s) & q_tokens), len(s)),
            reverse=True,
        )
        deduped: list[str] = []
        seen = Counter()
        for sentence in ranked:
            key = sentence.lower()
            if seen[key]:
                continue
            deduped.append(sentence)
            seen[key] += 1
            if len(deduped) == 3:
                break
        return deduped

    def _build_prompt(self, question: str, contexts: list[str]) -> str:
        joined_context = "\n\n".join(f"- {ctx}" for ctx in contexts)
        return (
            "You are a grounded knowledge assistant.\n"
            "Use only the context below.\n"
            "If context is insufficient, say you do not know.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{joined_context}\n\n"
            "Answer:"
        )

    def answer(self, question: str) -> dict:
        retrieved = self.vector_store.search(question, k=self.top_k)
        if not retrieved:
            return {
                "answer": "I could not find relevant information in the local knowledge base.",
                "sources": [],
            }

        contexts = [item["text"] for item in retrieved]
        _prompt = self._build_prompt(question, contexts)
        best_sentences = self._rank_sentences(question, contexts)
        if not best_sentences:
            best_sentences = contexts[:2]

        sources = [
            {
                "file": item["metadata"]["file"],
                "chunk_id": item["metadata"]["chunk_id"],
                "score": round(item["score"], 4),
            }
            for item in retrieved
        ]

        return {
            "answer": " ".join(best_sentences),
            "sources": sources,
        }
