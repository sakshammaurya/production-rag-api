from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_rejects_empty_text():
    with TestClient(app) as client:
        response = client.post(
            "/v1/documents",
            json={"document_id": "doc-1", "title": "Empty", "text": "   "},
        )
    assert response.status_code == 422
