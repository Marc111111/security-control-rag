from fastapi.testclient import TestClient

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
    assert "Open Codex review packet" in response.text
    assert "Storyline report" in response.text
    assert "formatStorylineReport" in response.text
    assert "Saved run storyline" in response.text
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
    assert "openPacketEditorWindow" in response.text
    assert "openPacketWindow" in response.text
    assert "packet-state" in response.text
    assert "json-modal" in response.text
    assert "packetModalText" in response.text
    assert "Apply to workflow" in response.text
    assert '<textarea id="packet"></textarea>' not in response.text
    assert "Optional simulated DB input form" in response.text
    assert "business context manifest" in response.text
    assert "Apply form to JSON" in response.text
    assert "Save scenario" in response.text
    assert "openPreview" in response.text
    assert "Expand" in response.text


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
