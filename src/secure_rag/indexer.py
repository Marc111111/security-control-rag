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
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.overlap = overlap

    def index_path(self, path: str | Path) -> list[Chunk]:
        documents = load_path(path)
        chunks = chunk_documents(documents, chunk_size=self.chunk_size, overlap=self.overlap)
        embeddings = self.embedding_client.embed([chunk.text for chunk in chunks])
        self.vector_store.add(chunks, embeddings)
        return chunks

