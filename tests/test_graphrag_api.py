from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app.graph.store import MemoryGraphStore
from app.main import create_app
from app.pipeline import GraphRagPipeline
from app.retrieval.vector_store import MemoryDenseStore
from secure_rag.embeddings import HashEmbeddingClient


class FakeChatModel:
    def chat(self, messages: list[dict[str, str]]) -> str:
        return """
        {
          "executive_summary": "Missing anti-malware increases malware exposure.",
          "assumptions": ["The company has endpoints in scope."],
          "threats": ["Malware"],
          "vulnerabilities": ["Unprotected endpoints"],
          "risks": ["Business disruption"],
          "recommended_controls": ["CIS Controls 10.1"],
          "risk_control_matrix": [
            {
              "gap": "No anti-malware",
              "threat": "Malware",
              "vulnerability": "Unprotected endpoints",
              "risk": "Business disruption",
              "likelihood": "medium",
              "impact": "medium",
              "controls": ["CIS Controls 10.1"],
              "evidence": ["S1"]
            }
          ],
          "missing_information": [],
          "source_citations": [{"id": "S1", "source": "fixture"}],
          "from_retrieved_evidence": "Control and risk wording came from retrieved evidence.",
          "general_model_reasoning": ""
        }
        """


def test_graphrag_api_ingest_retrieve_and_query(tmp_path: Path) -> None:
    source = tmp_path / "controls.md"
    source.write_text(
        "CIS Controls 10.1: Deploy and maintain anti-malware protection. "
        "Anti-malware mitigates malware threats, unprotected endpoint vulnerabilities, "
        "and business disruption risk.",
        encoding="utf-8",
    )
    settings = Settings(vector_backend="memory", graph_backend="memory", debug=True)
    pipeline = GraphRagPipeline(
        settings,
        embedding_client=HashEmbeddingClient(dimensions=32),
        dense_store=MemoryDenseStore(),
        graph_store=MemoryGraphStore(),
        chat_model=FakeChatModel(),
    )
    client = TestClient(create_app(pipeline))

    ingest_response = client.post("/api/ingest", json={"source": str(source), "chunk_size": 300})
    assert ingest_response.status_code == 200
    assert ingest_response.json()["indexed_chunks"] == 1

    retrieve_response = client.post(
        "/api/retrieve",
        json={"question": "No anti-malware solution is in place.", "top_k": 4},
    )
    assert retrieve_response.status_code == 200
    assert retrieve_response.json()["retrieved_chunks"]

    query_response = client.post(
        "/api/query",
        json={"question": "No anti-malware solution is in place.", "debug": True},
    )
    assert query_response.status_code == 200
    body: dict[str, Any] = query_response.json()
    assert body["answer"]["threats"] == ["Malware"]
    assert body["answer"]["risk_control_matrix"][0]["controls"] == ["CIS Controls 10.1"]
    assert body["sources"]
    assert body["debug"]["retrieved_chunks"]
