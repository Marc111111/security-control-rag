from pathlib import Path

from secure_rag.chunking import chunk_document
from secure_rag.embeddings import HashEmbeddingClient
from secure_rag.retriever import Retriever
from secure_rag.schema import SourceDocument
from secure_rag.vector_store import MemoryVectorStore


def test_memory_retrieval_returns_most_relevant_chunk() -> None:
    embedding_client = HashEmbeddingClient()
    store = MemoryVectorStore()
    chunks = []
    for name, text in [
        ("backup.md", "offline backups restoration ransomware recovery"),
        ("access.md", "admin access multi factor authentication"),
    ]:
        chunks.extend(chunk_document(SourceDocument(path=Path(name), text=text)))
    store.add(chunks, embedding_client.embed([chunk.text for chunk in chunks]))
    retriever = Retriever(embedding_client=embedding_client, vector_store=store)

    hits = retriever.retrieve("ransomware backup", top_k=1)

    assert len(hits) == 1
    assert hits[0].chunk.metadata["source_path"] == "backup.md"
    assert hits[0].score > 0

