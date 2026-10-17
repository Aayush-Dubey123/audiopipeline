"""
API and integration test suite for AI Voice Studio core endpoints.
"""

import pytest
from starlette.testclient import TestClient
from core.apis.api import app

client = TestClient(app)


def test_health_check_endpoint() -> None:
    """
    Verify GET /v1/health returns HTTP 200 with standard status and ISO timestamp.
    """
    response = client.get("/v1/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"
    assert "time" in payload
    assert len(payload["time"]) > 10


def test_security_headers_present() -> None:
    """
    Verify required hardened HTTP security headers are returned on all responses.
    """
    response = client.get("/v1/health")
    assert response.status_code == 200

    headers = response.headers
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert "Content-Security-Policy" in headers
    assert "Strict-Transport-Security" in headers


def test_not_found_returns_standard_error_envelope() -> None:
    """
    Verify 404 responses conform to the standard {"error": {"code", "message"}} format.
    """
    response = client.get("/v1/nonexistent-endpoint")
    assert response.status_code == 404

    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "404"
    assert "message" in payload["error"]


def test_forced_error_masks_internal_stack_trace() -> None:
    """
    Verify unhandled exceptions return a generic 500 message with no leaked stack trace.
    """
    error_client = TestClient(app, raise_server_exceptions=False)

    @app.get("/v1/test-crash-endpoint")
    def trigger_crash() -> None:
        raise RuntimeError("Simulated internal database crash with sensitive secret: SECRET123")

    response = error_client.get("/v1/test-crash-endpoint")
    assert response.status_code == 500

    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert payload["error"]["message"] == "An unexpected error occurred. Please try again later."
    assert "SECRET123" not in response.text
    assert "Traceback" not in response.text


def test_lifespan_initializes_database() -> None:
    """
    Verify application lifespan initializes the database and creates tables.
    """
    with TestClient(app) as test_client:
        response = test_client.get("/v1/health")
        assert response.status_code == 200

