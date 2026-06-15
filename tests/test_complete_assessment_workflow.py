from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import app.workflows.complete_assessment as complete_assessment
import app.workflows.job_manager as job_manager
from app.assessment.mock_data import sample_foundation_packet
from app.config import Settings
from app.graph.store import MemoryGraphStore
from app.main import create_app
from app.pipeline import GraphRagPipeline
from app.retrieval.vector_store import MemoryDenseStore
from secure_rag.embeddings import HashEmbeddingClient


class WorkflowFakeChatModel:
    def chat(self, messages: list[dict[str, str]]) -> str:
        joined = "\n".join(message["content"] for message in messages)
        if "Write only paragraph text" in joined:
            return """
            {
              "management_summary": "Acme Hosting has malware and recovery gaps that need review.",
              "introduction": "This draft summarizes the vendor assessment for a business owner.",
              "objective": "The objective is to explain risk from tier and questionnaire evidence.",
              "risk_exposure": "Exposure is driven by endpoint and recovery gaps.",
              "conclusion": "The assessment should remain a draft until analyst review."
            }
            """
        return """
        {
          "executive_summary": "The weakness increases malware and ransomware exposure.",
          "assumptions": ["The vendor operates endpoints or workloads in scope."],
          "threats": ["Malware", "Ransomware"],
          "vulnerabilities": ["Missing endpoint protection", "Untested recovery procedures"],
          "risks": ["Business disruption", "Data loss"],
          "recommended_controls": ["CIS 10.1", "NIST CSF RC.RP"],
          "risk_control_matrix": [
            {
              "gap": "Weak protection or recovery",
              "threat": "Malware",
              "vulnerability": "Missing endpoint protection",
              "risk": "Business disruption",
              "likelihood": "medium",
              "impact": "high",
              "controls": ["CIS 10.1"],
              "evidence": ["S1"]
            }
          ],
          "missing_information": ["Confirm current endpoint scope."],
          "source_citations": [{"id": "S1", "source": "test-standards.md"}],
          "from_retrieved_evidence": "Controls and risk language came from retrieved evidence.",
          "general_model_reasoning": ""
        }
        """


class SlowWorkflowFakeChatModel(WorkflowFakeChatModel):
    def chat(self, messages: list[dict[str, str]]) -> str:
        time.sleep(0.25)
        return super().chat(messages)


class HugeWorkflowFakeChatModel(WorkflowFakeChatModel):
    def chat(self, messages: list[dict[str, str]]) -> str:
        return "x" * 120_000


def test_complete_assessment_workflow_uses_adapter_rag_and_persists_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        complete_assessment,
        "_chat_model",
        lambda *args, **kwargs: WorkflowFakeChatModel(),
    )
    source = tmp_path / "test-standards.md"
    source.write_text(
        "CIS 10.1 requires deploying and maintaining anti-malware protection. "
        "NIST CSF recovery planning supports ransomware recovery and restoration.",
        encoding="utf-8",
    )
    settings = Settings(
        vector_backend="memory",
        graph_backend="memory",
        debug=True,
        keyword_index_path=str(tmp_path / "keyword" / "chunks.jsonl"),
        run_store_path=str(tmp_path / "runs"),
    )
    pipeline = GraphRagPipeline(
        settings,
        embedding_client=HashEmbeddingClient(dimensions=32),
        dense_store=MemoryDenseStore(),
        graph_store=MemoryGraphStore(),
        chat_model=WorkflowFakeChatModel(),
    )
    pipeline.ingest(source, chunk_size=300)
    client = TestClient(create_app(pipeline))
    packet = sample_foundation_packet().model_dump(mode="json")

    response = client.post(
        "/api/workflows/complete-assessment/run",
        json={
            "input_source": {"adapter": "foundation_packet_v1", "payload": packet},
            "model": {"provider": "ollama", "model": "qwen3:14b"},
            "top_k": 5,
            "debug": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    step_names = [step["name"] for step in body["steps"]]
    assert step_names[:3] == [
        "Read assessment input",
        "Clean and sort questionnaire answers",
        "Create risk questions from weak answers",
    ]
    assert any(name.startswith("Plan searches for ") for name in step_names)
    assert any(name.startswith("Search standards evidence for ") for name in step_names)
    assert any(name.startswith("Prepare model prompt for ") for name in step_names)
    assert any(name.startswith("Ask model to write risk answer for ") for name in step_names)
    assert any(name.startswith("Store risk answer for ") for name in step_names)
    assert step_names[-1] == "Prepare final result for the application"
    assert body["steps"][1]["input"] == body["steps"][0]["output"]
    assert body["steps"][2]["input"] == body["steps"][1]["output"]
    assert "risk_questions" in body["steps"][2]["output"]
    selected_index = next(
        index
        for index, name in enumerate(step_names)
        if name.startswith("Select next weak answer to evaluate")
    )
    assert body["steps"][selected_index + 1]["input"] == body["steps"][selected_index][
        "output"
    ]
    assert body["steps"][selected_index + 2]["input"] == body["steps"][selected_index + 1][
        "output"
    ]
    assert body["steps"][selected_index + 3]["input"] == body["steps"][selected_index + 2][
        "output"
    ]
    assert body["steps"][selected_index + 4]["input"] == body["steps"][selected_index + 3][
        "output"
    ]
    assert body["steps"][selected_index + 5]["input"] == body["steps"][selected_index + 4][
        "output"
    ]
    assert body["final_result"]["assessment_id"] == packet["assessment_id"]
    assert body["final_result"]["risk_evaluations"]
    assert body["cost_estimate"]["llm_call_count"] >= 2
    assert body["cost_estimate"]["estimate_policy"] == "conservative_workflow_reserve"
    assert body["cost_estimate"]["estimated_cost_usd"] == 0
    assert body["cost_estimate"]["estimated_cost_eur"] == 0
    assert body["preflight"]["token_budget_tolerance_percent"] == 10
    assert body["token_budget"]["preflight_estimated_total_tokens"] == body["preflight"][
        "estimated_total_tokens"
    ]
    assert body["token_budget"]["allowed_total_tokens"] == body["preflight"][
        "allowed_total_tokens"
    ]
    assert body["token_budget"]["calls"]
    assert body["token_budget"]["calls"][0]["source"] == "rough_estimate"
    assert body["duration_seconds"] >= 0
    assert body["completed_at"]
    assert Path(body["run_path"]).exists()

    runs_response = client.get("/api/workflows/complete-assessment/runs")
    assert runs_response.status_code == 200
    assert runs_response.json()[0]["run_id"] == body["run_id"]
    assert runs_response.json()[0]["duration_seconds"] == body["duration_seconds"]


def test_complete_assessment_rejects_openai_without_confirmation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        complete_assessment,
        "_chat_model",
        lambda *args, **kwargs: WorkflowFakeChatModel(),
    )
    settings = Settings(
        vector_backend="memory",
        graph_backend="memory",
        keyword_index_path=str(tmp_path / "keyword" / "chunks.jsonl"),
        run_store_path=str(tmp_path / "runs"),
    )
    pipeline = GraphRagPipeline(
        settings,
        embedding_client=HashEmbeddingClient(dimensions=32),
        dense_store=MemoryDenseStore(),
        graph_store=MemoryGraphStore(),
        chat_model=WorkflowFakeChatModel(),
    )
    client = TestClient(create_app(pipeline))

    response = client.post(
        "/api/workflows/complete-assessment/run",
        json={
            "input_source": {
                "adapter": "foundation_packet_v1",
                "payload": sample_foundation_packet().model_dump(mode="json"),
            },
            "model": {"provider": "openai", "model": "gpt-5.4-mini"},
        },
    )

    assert response.status_code == 400
    assert "confirm_external_call" in response.json()["detail"]


def test_complete_assessment_preflight_endpoint_estimates_before_model_calls(
    tmp_path: Path,
) -> None:
    pipeline = _memory_pipeline(tmp_path, WorkflowFakeChatModel())
    client = TestClient(create_app(pipeline))

    response = client.post(
        "/api/workflows/complete-assessment/preflight",
        json={
            "input_source": {
                "adapter": "foundation_packet_v1",
                "payload": sample_foundation_packet().model_dump(mode="json"),
            },
            "model": {"provider": "ollama", "model": "qwen3:14b"},
            "top_k": 8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["weakness_count"] == 2
    assert body["llm_call_count"] == 3
    assert body["estimate_policy"] == "conservative_workflow_reserve"
    assert body["estimated_input_tokens"] > 24_000
    assert body["max_estimated_input_tokens"] == 60_000
    assert body["will_exceed_guard"] is False
    assert body["estimate_breakdown"]["retrieved_chunk_reserve"] > 0
    assert body["token_budget_tolerance_percent"] == 10
    assert body["allowed_total_tokens"] == math.ceil(body["estimated_total_tokens"] * 1.1)
    assert body["estimated_cost_usd"] == 0
    assert body["estimated_cost_eur"] == 0
    assert body["note"].startswith("Preflight does not query")


def test_complete_assessment_openai_preflight_returns_usd_and_eur_cost(
    tmp_path: Path,
) -> None:
    pipeline = _memory_pipeline(tmp_path, WorkflowFakeChatModel())
    client = TestClient(create_app(pipeline))

    response = client.post(
        "/api/workflows/complete-assessment/preflight",
        json={
            "input_source": {
                "adapter": "foundation_packet_v1",
                "payload": sample_foundation_packet().model_dump(mode="json"),
            },
            "model": {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "confirm_external_call": True,
            },
            "top_k": 8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["estimated_total_tokens"] == (
        body["estimated_input_tokens"] + body["estimated_output_tokens"]
    )
    assert body["estimated_cost_usd"] > 0
    assert body["estimated_cost_eur"] > 0
    assert body["usd_to_eur_rate"] > 0
    assert body["price_per_million_tokens"]["input"] > 0


def test_complete_assessment_token_budget_can_stop_oversized_model_output(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        complete_assessment,
        "_chat_model",
        lambda *args, **kwargs: HugeWorkflowFakeChatModel(),
    )
    pipeline = _memory_pipeline(tmp_path, HugeWorkflowFakeChatModel())
    pipeline.ingest(_standards_fixture(tmp_path), chunk_size=300)
    client = TestClient(create_app(pipeline))

    response = client.post(
        "/api/workflows/complete-assessment/run",
        json={
            "input_source": {
                "adapter": "foundation_packet_v1",
                "payload": sample_foundation_packet().model_dump(mode="json"),
            },
            "model": {
                "provider": "ollama",
                "model": "qwen3:14b",
                "token_budget_tolerance_percent": 0,
            },
            "top_k": 5,
        },
    )

    assert response.status_code == 400
    assert "Token budget guard" in response.json()["detail"]


def test_complete_assessment_job_endpoint_runs_to_completion(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        complete_assessment,
        "_chat_model",
        lambda *args, **kwargs: WorkflowFakeChatModel(),
    )
    pipeline = _memory_pipeline(tmp_path, WorkflowFakeChatModel())
    pipeline.ingest(_standards_fixture(tmp_path), chunk_size=300)
    client = TestClient(create_app(pipeline))

    start = client.post(
        "/api/workflows/complete-assessment/jobs",
        json={
            "input_source": {
                "adapter": "foundation_packet_v1",
                "payload": sample_foundation_packet().model_dump(mode="json"),
            },
            "model": {"provider": "ollama", "model": "qwen3:14b"},
            "top_k": 5,
        },
    )

    assert start.status_code == 200
    job_id = start.json()["job_id"]
    final = _wait_for_job(client, job_id)
    assert final["status"] == "completed"
    assert final["current_step"] == "Finished"
    assert final["partial_steps"]
    assert final["result"]["steps"][0]["name"] == "Read assessment input"


def test_complete_assessment_job_cancel_requests_ollama_stop(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    stopped: list[str] = []
    monkeypatch.setattr(
        complete_assessment,
        "_chat_model",
        lambda *args, **kwargs: SlowWorkflowFakeChatModel(),
    )
    monkeypatch.setattr(job_manager, "_stop_ollama_model", stopped.append)
    pipeline = _memory_pipeline(tmp_path, SlowWorkflowFakeChatModel())
    pipeline.ingest(_standards_fixture(tmp_path), chunk_size=300)
    client = TestClient(create_app(pipeline))
    start = client.post(
        "/api/workflows/complete-assessment/jobs",
        json={
            "input_source": {
                "adapter": "foundation_packet_v1",
                "payload": sample_foundation_packet().model_dump(mode="json"),
            },
            "model": {"provider": "ollama", "model": "qwen3:14b"},
            "top_k": 5,
        },
    )
    job_id = start.json()["job_id"]

    cancel = client.post(f"/api/workflows/complete-assessment/jobs/{job_id}/cancel")

    assert cancel.status_code == 200
    assert cancel.json()["status"] in {"cancelling", "cancelled"}
    assert stopped == ["qwen3:14b"]
    final = _wait_for_job(client, job_id)
    assert final["status"] == "cancelled"


def test_compact_rag_evidence_removes_large_debug_payloads() -> None:
    compact = complete_assessment._compact_rag_evidence(
        [
            {
                "answer": {
                    "executive_summary": "Summary",
                    "recommended_controls": ["CIS Safeguard 10.1"],
                    "source_citations": [{"metadata": {"huge": "x" * 20_000}}],
                },
                "sources": [
                    {
                        "id": "S1",
                        "source": "standards/cis.pdf",
                        "score": 0.9,
                        "retrieval_method": "keyword",
                        "metadata": {
                            "filename": "cis.pdf",
                            "source_path": "standards/cis.pdf",
                            "unused": "x" * 20_000,
                        },
                    }
                ],
                "debug": {
                    "prompt_messages": [{"content": "x" * 20_000}],
                    "retrieved_chunks": [
                        {
                            "score": 0.9,
                            "source": "standards/cis.pdf",
                            "retrieval_method": "keyword",
                            "chunk": {
                                "text": "A" * 2_000,
                                "metadata": {
                                    "filename": "cis.pdf",
                                    "source_path": "standards/cis.pdf",
                                    "unused": "x" * 20_000,
                                },
                            },
                        }
                    ],
                },
            }
        ]
    )

    assert "source_citations" not in compact[0]["answer"]
    assert "unused" not in compact[0]["sources"][0]["metadata"]
    assert len(compact[0]["retrieved_chunks"][0]["text"]) == 350


def _memory_pipeline(tmp_path: Path, chat_model: object) -> GraphRagPipeline:
    settings = Settings(
        vector_backend="memory",
        graph_backend="memory",
        debug=True,
        keyword_index_path=str(tmp_path / "keyword" / "chunks.jsonl"),
        run_store_path=str(tmp_path / "runs"),
    )
    return GraphRagPipeline(
        settings,
        embedding_client=HashEmbeddingClient(dimensions=32),
        dense_store=MemoryDenseStore(),
        graph_store=MemoryGraphStore(),
        chat_model=chat_model,
    )


def _standards_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "test-standards.md"
    source.write_text(
        "CIS 10.1 requires deploying and maintaining anti-malware protection. "
        "NIST CSF recovery planning supports ransomware recovery and restoration.",
        encoding="utf-8",
    )
    return source


def _wait_for_job(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(50):
        response = client.get(f"/api/workflows/complete-assessment/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")
