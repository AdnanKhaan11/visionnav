"""Integration tests — FastAPI task endpoints."""
from fastapi.testclient import TestClient
from visionnav.api.app import create_app
from visionnav.settings import APISettings, Settings


def _client():
    return TestClient(create_app(Settings(env="test",
                                          api=APISettings(valid_keys=["test-key"]))))

def test_health():
    assert _client().get("/v1/health").status_code == 200

def test_no_auth_401():
    assert _client().post("/v1/tasks/", json={"instruction": "x"}).status_code == 401

def test_submit_202():
    r = _client().post("/v1/tasks/", json={"instruction": "Open Chrome"},
                       headers={"authorization": "Bearer test-key"})
    assert r.status_code == 202
    assert "task_id" in r.json()

def test_unknown_task_404():
    r = _client().get("/v1/tasks/none",
                      headers={"authorization": "Bearer test-key"})
    assert r.status_code == 404
