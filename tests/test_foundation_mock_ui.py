from fastapi.testclient import TestClient

import app.main as app_main
from app.main import create_app


def test_foundation_mock_ui_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/mock/foundation")

    assert response.status_code == 200
    assert "Complete GraphRAG Assessment Workflow" in response.text
    assert "/api/workflows/complete-assessment/run" in response.text
    assert "gpt-5.4-mini" in response.text
    assert "qwen3:14b" in response.text
    assert "OpenAI API key (run only)" in response.text
    assert "Save key" in response.text
    assert "Forget key" in response.text
    assert "/api/local/openai-key" in response.text
    assert "External calls remain disabled" in response.text
    assert "Estimate cost" in response.text
    assert "Ask Standards Corpus" in response.text
    assert "/api/query" in response.text
    assert "askCorpus" in response.text
    assert "corpusQuestion" in response.text
    assert "Direct Corpus Query Result" in response.text
    assert "scrollbar-gutter: stable" in response.text
    assert "overflow-y: auto" in response.text
    assert "min-width: 0" in response.text
    assert "compact-config" in response.text
    assert 'onclick="estimateCost()"' in response.text
    assert "estimateCostQuietly" in response.text
    assert "/api/models/available" in response.text
    assert "minmax(380px, .95fr)" in response.text
    assert "@media (max-width: 860px)" in response.text
    assert "Token guard +/- %" in response.text
    assert "Enforce token guard" in response.text
    assert "Est. one-run price" in response.text
    assert "estimate-line" in response.text
    assert "Expected in/out" in response.text
    assert "Hard cap" in response.text
    assert "Total cost" in response.text
    assert "ETA" in response.text
    assert "No run history yet" in response.text
    assert "historicalDurationEstimate" in response.text
    assert "No completed runs to compare yet" in response.text
    assert "renderSavedRuns" in response.text
    assert "openSavedRun" in response.text
    assert "Saved run summary" in response.text
    assert "syncDirtyDbForm" in response.text
    assert "Codex review packet" in response.text
    assert "Result Summary And Saved Runs" in response.text
    assert "currentRunActions" in response.text
    assert "Final Result Contract" not in response.text
    assert "formatStorylineReport" in response.text
    assert "Saved run storyline" in response.text
    assert "Saved run evidence" in response.text
    assert "Evidence context actually available to the model" in response.text
    assert "evidenceInline" in response.text
    assert "showEvidenceInline" in response.text
    assert "closeEvidenceInline" in response.text
    assert "Graph/Neo4j" in response.text
    assert "Retrieved but not sent to the model" in response.text
    assert "formatEvidenceReport" in response.text
    assert "openEvidenceReport" in response.text
    assert "buildCodexReviewPacket" in response.text
    assert "Development-only independent quality review" in response.text
    assert "failureModal" in response.text
    assert "The model could not produce a complete risk matrix" in response.text
    assert "Why it probably happened" in response.text
    assert "Operator action" in response.text
    assert "System owner action" in response.text
    assert "Technical details" in response.text
    assert "Execution Workflow" in response.text
    assert "No run yet" in response.text
    assert "Simulated SQL Result JSON" in response.text
    assert "openPacketWindow" in response.text
    assert "packet-state" in response.text
    assert 'href="/mock/foundation/packet-editor"' in response.text
    assert '<textarea id="packet"></textarea>' not in response.text
    assert "Populate questionnaire" in response.text
    assert "/mock/foundation/questionnaire" in response.text
    assert 'target="foundationQuestionnaireEditor"' in response.text
    assert "foundationMock.savedQuestionnaire" in response.text
    assert "openQuestionnaireBuilder" in response.text
    assert "foundation-questionnaire-saved" in response.text
    assert "handleQuestionnaireMessage" in response.text
    assert "handleQuestionnaireStorage" in response.text
    assert "useSavedQuestionnairePacket" in response.text
    assert "openPreview" in response.text
    assert "Expand" in response.text


def test_available_models_endpoint_returns_safe_fallbacks() -> None:
    client = TestClient(create_app())

    response = client.post("/api/models/available", json={})

    assert response.status_code == 200
    body = response.json()
    assert "qwen3:14b" in body["ollama"]
    assert "gpt-5.4" in body["openai"]
    assert "openai_discovery_note" in body


def test_foundation_packet_editor_ui_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/mock/foundation/packet-editor")

    assert response.status_code == 200
    assert "Simulated SQL Result JSON" in response.text
    assert "Save for workflow" in response.text
    assert "Reset initial content" in response.text
    assert "foundationMock.savedDbScenario" in response.text


def test_foundation_questionnaire_editor_ui_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/mock/foundation/questionnaire-editor")

    assert response.status_code == 200
    assert "User Questionnaire Editor" in response.text
    assert ">Save<" in response.text
    assert ">Load<" in response.text
    assert ">Erase<" in response.text
    assert ">Default values<" in response.text
    assert "Add question" in response.text
    assert "Back to workflow" in response.text
    assert "Remove" in response.text
    assert "foundationMock.savedQuestionnaire" in response.text
    assert "foundation-questionnaire-saved" in response.text
    assert "The workflow page will use this questionnaire for the next run" in response.text

    alias_response = client.get("/mock/foundation/questionnaire")
    assert alias_response.status_code == 200
    assert "User Questionnaire Editor" in alias_response.text


def test_local_openai_key_cache_lifecycle(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        app_main,
        "_local_openai_key_path",
        lambda: tmp_path / "openai_api_key.txt",
    )
    client = TestClient(create_app())

    try:
        assert client.delete("/api/local/openai-key").json() == {"has_key": False}
        empty = client.get("/api/local/openai-key").json()
        assert empty == {"has_key": False, "api_key": ""}

        saved = client.post("/api/local/openai-key", json={"api_key": " sk-test-local "})
        assert saved.status_code == 200
        assert saved.json() == {"has_key": True}
        loaded = client.get("/api/local/openai-key").json()
        assert loaded == {"has_key": True, "api_key": "sk-test-local"}

        forgotten = client.delete("/api/local/openai-key")
        assert forgotten.status_code == 200
        assert forgotten.json() == {"has_key": False}
        assert client.get("/api/local/openai-key").json() == {"has_key": False, "api_key": ""}
    finally:
        client.delete("/api/local/openai-key")


def test_foundation_business_context_manifest_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/mock/foundation/business-context")

    assert response.status_code == 200
    assert "Tier Context" in response.text
    assert "Questionnaire Result Context" in response.text


def test_mock_foundation_summary_endpoint_runs_without_external_services() -> None:
    client = TestClient(create_app())
    packet = client.get("/api/mock/foundation-packet").json()

    response = client.post(
        "/api/mock/foundation-summary",
        json={"packet": packet, "debug": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["postgres_payload"]["vendor_id"] == "V-1"
    assert body["draft"]["weaknesses"]


def test_token_estimate_is_small_for_sample_packet() -> None:
    client = TestClient(create_app())
    packet = client.get("/api/mock/foundation-packet").json()

    response = client.post(
        "/api/assessments/foundation-summary/token-estimate",
        json={"packet": packet, "model": "gpt-4.1-mini"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["estimated_input_tokens"] < 6_000
    assert body["estimated_cost_usd"] < 0.02
    assert body["estimated_cost_eur"] < 0.02
    assert body["usd_to_eur_rate"] > 0
    assert "pricing_note" in body


def test_model_run_mock_returns_price_metadata() -> None:
    client = TestClient(create_app())
    packet = client.get("/api/mock/foundation-packet").json()

    response = client.post(
        "/api/assessments/foundation-summary/model-run",
        json={
            "packet": packet,
            "provider": "mock",
            "model": "gpt-5.4-mini",
            "debug": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_run"]["provider"] == "mock"
    assert body["model_run"]["model"] == "gpt-5.4-mini"
    assert body["model_run"]["token_estimate"]["estimated_cost_usd"] > 0


def test_model_run_blocks_openai_without_checkbox_confirmation() -> None:
    client = TestClient(create_app())
    packet = client.get("/api/mock/foundation-packet").json()

    response = client.post(
        "/api/assessments/foundation-summary/model-run",
        json={"packet": packet, "provider": "openai", "model": "gpt-5.4-mini"},
    )

    assert response.status_code == 400
    assert "External OpenAI call blocked" in response.json()["detail"]


def test_model_run_accepts_request_scoped_api_key_but_still_requires_confirmation() -> None:
    client = TestClient(create_app())
    packet = client.get("/api/mock/foundation-packet").json()

    response = client.post(
        "/api/assessments/foundation-summary/model-run",
        json={
            "packet": packet,
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "openai_api_key": "sk-test-not-used",
        },
    )

    assert response.status_code == 400
    assert "External OpenAI call blocked" in response.json()["detail"]


def test_openai_smoke_test_requires_explicit_confirmation() -> None:
    client = TestClient(create_app())
    packet = client.get("/api/mock/foundation-packet").json()

    response = client.post(
        "/api/assessments/foundation-summary/openai-smoke-test",
        json={"packet": packet, "model": "gpt-4.1-mini"},
    )

    assert response.status_code == 400
    assert "confirm_external_call" in response.json()["detail"]
