"""Integration tests — FastAPI task endpoints."""

from fastapi.testclient import TestClient
from visionnav.api.app import create_app
from visionnav.settings import APISettings, Settings


def _client():
    return TestClient(
        create_app(
            Settings(
                env="test",
                api=APISettings(valid_keys=["test-key"]),
            )
        )
    )


def test_health():
    assert _client().get("/v1/health").status_code == 200


def test_no_auth_401():
    """Request with wrong API key must return 401."""
    import os

    # Force valid_keys via environment variable
    os.environ["VISIONNAV_API__VALID_KEYS"] = '["test-key-123"]'

    from visionnav.api.app import create_app
    from visionnav.settings import Settings

    # Clear lru_cache so new settings are loaded
    from visionnav.api.dependencies import get_cached_settings

    get_cached_settings.cache_clear()

    app = create_app(Settings())
    client = TestClient(app)

    # Send request with WRONG key
    resp = client.post(
        "/v1/tasks/",
        json={"instruction": "x"},
        headers={"authorization": "Bearer wrong-key-999"},
    )

    # Clean up env var after test
    del os.environ["VISIONNAV_API__VALID_KEYS"]
    get_cached_settings.cache_clear()

    assert resp.status_code == 401


def test_submit_202():
    client = TestClient(
        create_app(
            Settings(
                env="test",
                api=APISettings(valid_keys=["test-key"]),
            )
        )
    )
    resp = client.post(
        "/v1/tasks/",
        json={"instruction": "Open Chrome"},
        headers={"authorization": "Bearer test-key"},
    )
    assert resp.status_code == 202
    assert "task_id" in resp.json()


def test_unknown_task_404():
    client = TestClient(
        create_app(
            Settings(
                env="test",
                api=APISettings(valid_keys=["test-key"]),
            )
        )
    )
    resp = client.get(
        "/v1/tasks/none",
        headers={"authorization": "Bearer test-key"},
    )
    assert resp.status_code == 404
