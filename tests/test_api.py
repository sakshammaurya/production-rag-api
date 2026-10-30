import logging
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from app.pdf import PDFChunk, PDFPage


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(response.headers["X-Request-ID"]) == 32


def test_valid_request_id_is_preserved():
    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Request-ID": "test-run.123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-run.123"


def test_embedding_failure_is_logged_with_request_id_and_generic_error(monkeypatch, caplog):
    monkeypatch.setattr(
        "app.main.embed_texts",
        lambda texts: (_ for _ in ()).throw(RuntimeError("provider-secret-detail")),
    )

    with caplog.at_level(logging.ERROR, logger="uvicorn.error"), TestClient(app) as client:
        response = client.post(
            "/v1/ask",
            headers={"X-Request-ID": "test-failure-1"},
            json={"question": "What is the policy?", "top_k": 5},
        )

    assert response.status_code == 502
    assert "provider-secret-detail" not in response.text
    record = next(
        record for record in caplog.records if '"event":"rag_stage_failed"' in record.message
    )
    assert record.request_id == "test-failure-1"
    assert record.stage == "embedding"


def test_ingest_rejects_empty_text():
    with TestClient(app) as client:
        response = client.post(
            "/v1/documents",
            json={"document_id": "doc-1", "title": "Empty", "text": "   "},
        )
    assert response.status_code == 422


def test_pdf_ingestion_embeds_and_stores_page_chunks(monkeypatch):
    monkeypatch.setattr("app.main.extract_pdf_pages", lambda _: [PDFPage(2, "Known policy")])
    monkeypatch.setattr(
        "app.main.chunk_pdf_pages", lambda *args: [PDFChunk(0, 2, "Known policy")]
    )
    monkeypatch.setattr("app.main.embed_texts", lambda texts: [[0.1, 0.2]])
    captured = {}

    def fake_insert(document_id, title, chunks, embeddings):
        captured.update(
            document_id=document_id,
            title=title,
            chunks=chunks,
            embeddings=embeddings,
        )
        return 1

    monkeypatch.setattr("app.main.insert_chunks", fake_insert)

    with TestClient(app) as client:
        response = client.post(
            "/v1/documents/pdf",
            data={"document_id": "policy", "title": "Policy PDF"},
            files={"file": ("policy.pdf", b"%PDF-test", "application/pdf")},
        )

    assert response.status_code == 201
    assert response.json() == {"document_id": "policy", "chunks_created": 1}
    assert captured["chunks"][0].page_number == 2
    assert captured["embeddings"] == [[0.1, 0.2]]


def test_pdf_ingestion_rejects_wrong_media_type():
    with TestClient(app) as client:
        response = client.post(
            "/v1/documents/pdf",
            data={"document_id": "policy", "title": "Policy"},
            files={"file": ("policy.txt", b"text", "text/plain")},
        )

    assert response.status_code == 415


def test_pdf_ingestion_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(
        "app.main.get_settings",
        lambda: SimpleNamespace(max_pdf_upload_bytes=8),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/documents/pdf",
            data={"document_id": "policy", "title": "Policy"},
            files={"file": ("policy.pdf", b"%PDF-1234", "application/pdf")},
        )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "PDF exceeds the maximum upload size of 8 bytes"
    }


def test_pdf_ingestion_rejects_malformed_pdf():
    with TestClient(app) as client:
        response = client.post(
            "/v1/documents/pdf",
            data={"document_id": "broken", "title": "Broken PDF"},
            files={"file": ("broken.pdf", b"%PDF-not-valid", "application/pdf")},
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "PDF could not be read"}


def test_ask_abstains_when_retrieval_is_below_threshold(monkeypatch, caplog):
    monkeypatch.setattr("app.main.embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(
        "app.main.search_chunks",
        lambda embedding, top_k: [
            {
                "chunk_id": 1,
                "document_id": "unrelated",
                "title": "Unrelated",
                "page_number": 1,
                "content": "No relevant answer here.",
                "similarity": 0.59,
            }
        ],
    )
    monkeypatch.setattr(
        "app.main.generate_answer",
        lambda *args: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )

    with caplog.at_level(logging.INFO, logger="uvicorn.error"), TestClient(app) as client:
        response = client.post(
            "/v1/ask",
            headers={"X-Request-ID": "test-abstention-1"},
            json={"question": "What is not covered?", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "I don't have enough information in the indexed documents to answer that.",
        "abstained": True,
        "sources": [],
    }
    record = next(
        record
        for record in caplog.records
        if '"event":"rag_request_completed"' in record.message
    )
    assert record.request_id == "test-abstention-1"
    assert record.abstained is True
    assert record.generation_ms == 0.0
    assert record.evidence_count == 0


def test_ask_logs_stage_latency_metrics(monkeypatch, caplog):
    monkeypatch.setattr(
        "app.main.get_settings",
        lambda: SimpleNamespace(min_similarity=0.6),
    )
    monkeypatch.setattr("app.main.embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(
        "app.main.search_chunks",
        lambda embedding, top_k: [
            {
                "chunk_id": 7,
                "document_id": "policy",
                "title": "Policy",
                "page_number": 1,
                "content": "Remote work is allowed on Tuesday.",
                "similarity": 0.9,
            }
        ],
    )
    monkeypatch.setattr(
        "app.main.generate_answer",
        lambda question, evidence: "Remote work is allowed on Tuesday. [Source 7, Page 1]",
    )

    with caplog.at_level(logging.INFO, logger="uvicorn.error"), TestClient(app) as client:
        response = client.post(
            "/v1/ask",
            json={"question": "When is remote work allowed?", "top_k": 5},
        )

    assert response.status_code == 200
    record = next(
        record
        for record in caplog.records
        if '"event":"rag_request_completed"' in record.message
    )
    assert record.embedding_ms >= 0
    assert record.retrieval_ms >= 0
    assert record.generation_ms >= 0
    assert record.total_ms >= (
        record.embedding_ms + record.retrieval_ms + record.generation_ms
    )
    assert record.abstained is False
    assert record.evidence_count == 1
