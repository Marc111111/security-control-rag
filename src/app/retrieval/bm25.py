from __future__ import annotations

import re
from collections import Counter

from app.schemas import DocumentChunk, RetrievedEvidence


class KeywordIndex:
    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: object | None = None

    def add(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        self._tokenized.extend(_tokens(chunk.text) for chunk in chunks)
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, *, top_k: int = 12) -> list[RetrievedEvidence]:
        if not self.chunks:
            return []
        query_tokens = _tokens(query)
        if self._bm25 is not None:
            scores = list(self._bm25.get_scores(query_tokens))
        else:
            query_counts = Counter(query_tokens)
            scores = [
                sum(min(query_counts[token], Counter(tokens)[token]) for token in query_counts)
                for tokens in self._tokenized
            ]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        max_score = max((score for _, score in ranked), default=1.0) or 1.0
        return [
            RetrievedEvidence(
                chunk=self.chunks[index],
                score=float(score) / float(max_score),
                source=str(self.chunks[index].metadata.get("source_path", "")),
                retrieval_method="keyword",
            )
            for index, score in ranked
            if score > 0
        ]


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if token not in {"the", "and", "for", "with", "that", "this"}
    ]

