from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.schemas import DocumentChunk, RetrievedEvidence
from secure_rag.vector_store import _cosine_similarity


class DenseStore(Protocol):
    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        ...

    def search(self, embedding: list[float], *, top_k: int) -> list[RetrievedEvidence]:
        ...


class MemoryDenseStore:
    def __init__(self) -> None:
        self.items: list[tuple[DocumentChunk, list[float]]] = []

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        self.items.extend(zip(chunks, embeddings, strict=True))

    def search(self, embedding: list[float], *, top_k: int) -> list[RetrievedEvidence]:
        hits = [
            RetrievedEvidence(
                chunk=chunk,
                score=_cosine_similarity(embedding, stored),
                source=str(chunk.metadata.get("source_path", chunk.metadata.get("filename", ""))),
                retrieval_method="dense",
            )
            for chunk, stored in self.items
        ]
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]


class QdrantDenseStore:
    def __init__(self, *, url: str, collection: str, vector_size: int) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
        except ImportError as exc:
            raise RuntimeError("Install qdrant-client to use QdrantDenseStore") from exc
        self.collection = collection
        self.client = QdrantClient(url=url)
        collections = {item.name for item in self.client.get_collections().collections}
        if collection not in collections:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        try:
            from qdrant_client.http.models import PointStruct
        except ImportError as exc:
            raise RuntimeError("Install qdrant-client to use QdrantDenseStore") from exc
        points = [
            PointStruct(
                id=_qdrant_point_id(chunk.id),
                vector=embedding,
                payload={"chunk_id": chunk.id, "text": chunk.text, "metadata": chunk.metadata},
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, embedding: list[float], *, top_k: int) -> list[RetrievedEvidence]:
        result = self.client.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
        )
        hits: list[RetrievedEvidence] = []
        for point in result:
            payload = point.payload or {}
            metadata = dict(payload.get("metadata") or {})
            chunk = DocumentChunk(
                id=str(payload.get("chunk_id") or point.id),
                text=str(payload.get("text", "")),
                metadata=metadata,
            )
            hits.append(
                RetrievedEvidence(
                    chunk=chunk,
                    score=float(point.score),
                    source=str(metadata.get("source_path", metadata.get("filename", ""))),
                    retrieval_method="dense",
                )
            )
        return hits


def _qdrant_point_id(chunk_id: str) -> str:
    try:
        return str(UUID(chunk_id[:32]))
    except ValueError:
        return str(UUID(bytes=chunk_id.encode("utf-8")[:16].ljust(16, b"0")))
