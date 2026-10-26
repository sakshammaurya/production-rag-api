from fastapi import FastAPI, HTTPException

from app.chunking import chunk_text
from app.config import get_settings
from app.providers import embed_texts, generate_answer
from app.repository import insert_chunks, search_chunks
from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse, Source

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


@app.post("/v1/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    settings = get_settings()
    try:
        query_embedding = embed_texts([payload.question])[0]
        candidates = search_chunks(query_embedding, payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Retrieval failed") from exc

    evidence = [item for item in candidates if item["similarity"] >= settings.min_similarity]
    if not evidence:
        return AskResponse(
            answer="I don't have enough information in the indexed documents to answer that.",
            abstained=True,
            sources=[],
        )

    try:
        answer = generate_answer(payload.question, evidence)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Answer generation failed") from exc

    return AskResponse(
        answer=answer,
        abstained=False,
        sources=[Source(**item) for item in evidence],
    )
