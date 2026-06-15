from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from app.schemas import DocumentChunk, RetrievedEvidence


class KeywordIndex:
    def __init__(self, *, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.chunks: list[DocumentChunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: object | None = None
        if self.path and self.path.exists():
            self._load()

    def add(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        self._tokenized.extend(_tokens(chunk.text) for chunk in chunks)
        if self.path:
            self._append(chunks)
        self._rebuild()

    def _rebuild(self) -> None:
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

    def _append(self, chunks: list[DocumentChunk]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")

    def _load(self) -> None:
        if self.path is None:
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.chunks.append(DocumentChunk.model_validate(json.loads(line)))
        self._tokenized = [_tokens(chunk.text) for chunk in self.chunks]
        self._rebuild()


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if token not in {"the", "and", "for", "with", "that", "this"}
    ]
