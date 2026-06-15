from __future__ import annotations

from pathlib import Path

from secure_rag.chunking import chunk_documents
from secure_rag.embeddings import EmbeddingClient
from secure_rag.loaders import load_path
from secure_rag.schema import Chunk
from secure_rag.vector_store import VectorStore


class RagIndexer:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        chunk_size: int = 1_500,
        overlap: int = 200,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size

    def index_path(self, path: str | Path) -> list[Chunk]:
        documents = load_path(path)
        chunks = chunk_documents(documents, chunk_size=self.chunk_size, overlap=self.overlap)
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            embeddings = self.embedding_client.embed([chunk.text for chunk in batch])
            self.vector_store.add(batch, embeddings)
        return chunks
