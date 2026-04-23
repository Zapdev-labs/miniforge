"""Tests for Miniforge WebUI server."""

import pytest
from fastapi.testclient import TestClient

from miniforge.webui.server import create_app


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient for the WebUI app."""
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """Test health check returns expected shape."""
    response = client.get("/health")
    # May be 200 or 503 depending on whether a model is loaded in state
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_list_models(client: TestClient) -> None:
    """Test models list endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


def test_api_stats(client: TestClient) -> None:
    """Test stats endpoint returns memory info."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "uptime_seconds" in data
    assert "memory" in data
    assert "total_gb" in data["memory"]


def test_api_runtime(client: TestClient) -> None:
    """Test runtime endpoint returns resolved config surface."""
    response = client.get("/api/runtime")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "load_error" in data
    assert "config" in data


def test_metrics_endpoint(client: TestClient) -> None:
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    # Returns 501 if prometheus-client is missing, otherwise 200
    assert response.status_code in (200, 501)


def test_chat_completions_no_model(client: TestClient) -> None:
    """Test chat completions when model is not loaded."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "miniforge",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    # Should return 503 because model is not loaded in test state
    assert response.status_code == 503


def test_static_index_fallback(client: TestClient) -> None:
    """Test root returns at least a JSON response if static files missing."""
    response = client.get("/")
    assert response.status_code == 200
    # Could be HTML or JSON depending on whether static/index.html exists
    assert (
        response.headers["content-type"]
        in [
            "text/html; charset=utf-8",
            "application/json",
        ]
        or "text/html" in response.headers["content-type"]
    )
