from __future__ import annotations

from secure_rag.embeddings import EmbeddingClient
from secure_rag.schema import RetrievalHit
from secure_rag.vector_store import VectorStore


class Retriever:
    def __init__(self, *, embedding_client: EmbeddingClient, vector_store: VectorStore) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def retrieve(self, query: str, *, top_k: int = 8) -> list[RetrievalHit]:
        embedding = self.embedding_client.embed([query])[0]
        return self.vector_store.search(embedding, top_k=top_k)

