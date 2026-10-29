import logging
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.chunking import chunk_text
from app.config import get_settings
from app.pdf import (
    PDFExtractionError,
    PDFUploadTooLargeError,
    chunk_pdf_pages,
    extract_pdf_pages,
    read_pdf_upload,
)
from app.providers import embed_texts, generate_answer
from app.repository import insert_chunks, search_chunks
from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse, Source

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Production RAG API",
    version="0.1.0",
    description="A small, testable RAG service with source-grounded answers.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/documents", response_model=IngestResponse, status_code=201)
def ingest_document(payload: IngestRequest) -> IngestResponse:
    settings = get_settings()
    chunks = chunk_text(payload.text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="Document contains no usable text")

    try:
        embeddings = embed_texts([chunk.content for chunk in chunks])
        count = insert_chunks(payload.document_id, payload.title, chunks, embeddings)
    except Exception as exc:
        # Keep provider and database details out of client responses.
        raise HTTPException(status_code=502, detail="Document ingestion failed") from exc

    return IngestResponse(document_id=payload.document_id, chunks_created=count)


@app.post("/v1/documents/pdf", response_model=IngestResponse, status_code=201)
def ingest_pdf(
    document_id: Annotated[str, Form(min_length=1, max_length=200)],
    title: Annotated[str, Form(min_length=1, max_length=300)],
    file: Annotated[UploadFile, File()],
) -> IngestResponse:
    document_id = document_id.strip()
    title = title.strip()
    if not document_id or not title:
        raise HTTPException(status_code=422, detail="Document metadata must not be blank")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported")

    settings = get_settings()
    try:
        content = read_pdf_upload(file.file, settings.max_pdf_upload_bytes)
        pages = extract_pdf_pages(content)
        chunks = chunk_pdf_pages(pages, settings.chunk_size, settings.chunk_overlap)
        embeddings = embed_texts([chunk.content for chunk in chunks])
        count = insert_chunks(document_id, title, chunks, embeddings)
    except PDFUploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except PDFExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="PDF ingestion failed") from exc

    return IngestResponse(document_id=document_id, chunks_created=count)


@app.post("/v1/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    request_started = perf_counter()
    settings = get_settings()

    embedding_started = perf_counter()
    try:
        query_embedding = embed_texts([payload.question])[0]
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Retrieval failed") from exc
    embedding_ms = (perf_counter() - embedding_started) * 1000

    retrieval_started = perf_counter()
    try:
        candidates = search_chunks(query_embedding, payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Retrieval failed") from exc
    retrieval_ms = (perf_counter() - retrieval_started) * 1000

    evidence = [item for item in candidates if item["similarity"] >= settings.min_similarity]
    if not evidence:
        logger.info(
            "rag_request_completed",
            extra={
                "embedding_ms": round(embedding_ms, 3),
                "retrieval_ms": round(retrieval_ms, 3),
                "generation_ms": 0.0,
                "total_ms": round((perf_counter() - request_started) * 1000, 3),
                "abstained": True,
                "evidence_count": 0,
            },
        )
        return AskResponse(
            answer="I don't have enough information in the indexed documents to answer that.",
            abstained=True,
            sources=[],
        )

    generation_started = perf_counter()
    try:
        answer = generate_answer(payload.question, evidence)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Answer generation failed") from exc
    generation_ms = (perf_counter() - generation_started) * 1000

    logger.info(
        "rag_request_completed",
        extra={
            "embedding_ms": round(embedding_ms, 3),
            "retrieval_ms": round(retrieval_ms, 3),
            "generation_ms": round(generation_ms, 3),
            "total_ms": round((perf_counter() - request_started) * 1000, 3),
            "abstained": False,
            "evidence_count": len(evidence),
        },
    )
    return AskResponse(
        answer=answer,
        abstained=False,
        sources=[Source(**item) for item in evidence],
    )
