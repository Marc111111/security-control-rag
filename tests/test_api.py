from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from secure_rag.api import create_app
from secure_rag.runtime import RagRuntimeConfig
from secure_rag.schema import Chunk, ControlAnswer, RetrievalHit


class FakeService:
    def __init__(self) -> None:
        self.config = RagRuntimeConfig(db_path=Path("storage/test"))
        self.last_query: dict[str, Any] | None = None

    def ingest(
        self,
        source: str | Path,
        *,
        chunk_size: int = 1_500,
        overlap: int = 200,
        batch_size: int = 64,
    ) -> int:
        self.last_query = {
            "source": str(source),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "batch_size": batch_size,
        }
        return 3

    def query(
        self,
        *,
        message: str,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> ControlAnswer:
        self.last_query = {
            "message": message,
            "context": context,
            "history": history,
            "top_k": top_k,
        }
        return ControlAnswer(
            answer="Maintain offline backups. Sources: [S1]",
            insufficient_evidence=False,
            sources=[{"source_path": "controls.csv", "score": 0.91}],
            raw={"hit_count": 1},
        )

    def retrieve(
        self,
        *,
        message: str,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        self.last_query = {
            "message": message,
            "context": context,
            "history": history,
            "top_k": top_k,
        }
        return [
            RetrievalHit(
                chunk=Chunk(
                    id="chunk-1",
                    text="Maintain offline backups.",
                    metadata={"source_path": "controls.csv"},
                ),
                score=0.91,
            )
        ]


def test_health_returns_runtime_configuration() -> None:
    service = FakeService()
    client = TestClient(create_app(service))

    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "strict_corpus_first"
    assert data["db_path"] == "storage/test"
    assert data["generation_model"] == "qwen3:14b"


def test_query_accepts_natural_language_and_context() -> None:
    service = FakeService()
    client = TestClient(create_app(service))

    response = client.post(
        "/api/query",
        json={
            "message": "recommend controls",
            "context": {"risk": "ransomware", "tier": 2},
            "history": [{"role": "user", "content": "previous question"}],
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"].startswith("Maintain offline backups")
    assert response.json()["sources"][0]["source_path"] == "controls.csv"
    assert service.last_query == {
        "message": "recommend controls",
        "context": {"risk": "ransomware", "tier": 2},
        "history": [{"role": "user", "content": "previous question"}],
        "top_k": 5,
    }


def test_retrieve_returns_source_chunks_without_generation() -> None:
    client = TestClient(create_app(FakeService()))

    response = client.post("/api/retrieve", json={"message": "ransomware backup", "top_k": 1})

    assert response.status_code == 200
    data = response.json()
    assert data["hits"][0]["score"] == 0.91
    assert data["hits"][0]["chunk"]["metadata"]["source_path"] == "controls.csv"


def test_ingest_returns_indexed_chunk_count() -> None:
    service = FakeService()
    client = TestClient(create_app(service))

    response = client.post("/api/ingest", json={"source": "data/raw"})

    assert response.status_code == 200
    assert response.json() == {"indexed_chunks": 3}


def test_ui_is_served_at_root() -> None:
    client = TestClient(create_app(FakeService()))

    response = client.get("/")

    assert response.status_code == 200
    assert "Security Control RAG" in response.text
