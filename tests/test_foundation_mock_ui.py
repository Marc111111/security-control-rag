from fastapi.testclient import TestClient

from app.main import create_app


def test_foundation_mock_ui_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/mock/foundation")

    assert response.status_code == 200
    assert "Foundation Assessment Summary Mock" in response.text
    assert "/api/mock/foundation-summary" in response.text


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


def test_openai_smoke_test_requires_explicit_confirmation() -> None:
    client = TestClient(create_app())
    packet = client.get("/api/mock/foundation-packet").json()

    response = client.post(
        "/api/assessments/foundation-summary/openai-smoke-test",
        json={"packet": packet, "model": "gpt-4.1-mini"},
    )

    assert response.status_code == 400
    assert "confirm_external_call" in response.json()["detail"]
