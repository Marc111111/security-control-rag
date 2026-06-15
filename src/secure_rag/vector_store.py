from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

from secure_rag.schema import Chunk, Metadata, RetrievalHit


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ...

    def search(self, embedding: list[float], *, top_k: int = 8) -> list[RetrievalHit]:
        ...


class MemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self._items.extend(zip(chunks, embeddings, strict=True))

    def search(self, embedding: list[float], *, top_k: int = 8) -> list[RetrievalHit]:
        hits = [
            RetrievalHit(
                chunk=chunk,
                score=_cosine_similarity(embedding, stored_embedding),
                vector_score=_cosine_similarity(embedding, stored_embedding),
            )
            for chunk, stored_embedding in self._items
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


class ChromaVectorStore:
    def __init__(self, *, path: str | Path, collection: str = "security_controls") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb to use ChromaVectorStore") from exc

        client = chromadb.PersistentClient(path=str(path))
        self.collection = client.get_or_create_collection(name=collection)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[_chroma_metadata(chunk.metadata) for chunk in chunks],
            embeddings=embeddings,
        )

    def search(self, embedding: list[float], *, top_k: int = 8) -> list[RetrievalHit]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[RetrievalHit] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            score = 1.0 / (1.0 + float(distance))
            hits.append(
                RetrievalHit(
                    chunk=Chunk(id=chunk_id, text=text, metadata=metadata),
                    score=score,
                    vector_score=score,
                )
            )
        return hits


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    denominator = left_norm * right_norm
    return numerator / denominator if denominator else 0.0


def _chroma_metadata(metadata: Metadata) -> Metadata:
    clean: Metadata = {}
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool) or value is None:
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean
