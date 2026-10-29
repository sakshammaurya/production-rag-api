import psycopg
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

pytestmark = pytest.mark.integration


def test_text_ingestion_persists_chunks_through_api(
    document_id, embedding_vector, monkeypatch
):
    monkeypatch.setattr(
        "app.main.embed_texts",
        lambda texts: [embedding_vector(1.0) for _ in texts],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/documents",
            json={
                "document_id": document_id,
                "title": "Integration handbook",
                "text": "Persist this text through the real PostgreSQL repository.",
            },
        )

    assert response.status_code == 201
    assert response.json() == {"document_id": document_id, "chunks_created": 1}
    with psycopg.connect(get_settings().database_url) as connection:
        row = connection.execute(
            """SELECT title, page_number, chunk_index, content, vector_dims(embedding)
               FROM document_chunks WHERE document_id = %s""",
            (document_id,),
        ).fetchone()
    assert row == (
        "Integration handbook",
        None,
        0,
        "Persist this text through the real PostgreSQL repository.",
        768,
    )
