from pathlib import Path

from secure_rag.indexer import RagIndexer
from secure_rag.vector_store import MemoryVectorStore


class CountingEmbeddingClient:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(len(text))] for text in texts]


def test_indexer_batches_embedding_requests(tmp_path: Path) -> None:
    source = tmp_path / "controls.txt"
    source.write_text(" ".join(f"control-{index}" for index in range(120)), encoding="utf-8")
    embedding_client = CountingEmbeddingClient()
    store = MemoryVectorStore()
    indexer = RagIndexer(
        embedding_client=embedding_client,
        vector_store=store,
        chunk_size=30,
        overlap=0,
        batch_size=3,
    )

    chunks = indexer.index_path(source)

    assert len(chunks) > 3
    assert embedding_client.batch_sizes[0] == 3
    assert max(embedding_client.batch_sizes) <= 3

