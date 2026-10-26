# Production RAG API

A portfolio project demonstrating a **testable Retrieval-Augmented Generation (RAG) backend** with FastAPI, PostgreSQL/pgvector, OpenAI embeddings, and pytest.

> **Current milestone:** MVP scaffold. Do not describe this as production-ready until the database integration, automated checks, and retrieval evaluation have been run and documented.

## What this project demonstrates

- **Backend design:** typed API contracts, input validation, persistence boundaries, and environment-based configuration.
- **RAG fundamentals:** document chunking, embedding generation, cosine-distance retrieval, and source metadata.
- **Grounding behavior:** evidence thresholding and an explicit insufficient-evidence response.
- **Reliability basics:** bounded inputs, parameterized SQL, provider timeouts/retries, and no raw infrastructure details in API error responses.
- **Engineering discipline:** unit tests and a GitHub Actions workflow.

## Architecture

```text
POST /v1/documents
  validate → chunk text → embed chunks → PostgreSQL + pgvector

POST /v1/ask
  embed question → vector search → similarity threshold
                                  ├─ no evidence → abstain
                                  └─ evidence → LLM answer + source references
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `POST /v1/documents` | Chunk, embed, and store a text document |
| `POST /v1/ask` | Retrieve evidence and generate a grounded answer |

### Example: ingest a document

```json
{
  "document_id": "handbook-1",
  "title": "Team handbook",
  "text": "Vacation requests should be submitted two weeks in advance."
}
```

Send that body to `POST /v1/documents`.

### Example: ask a question

```json
{
  "question": "How early should I submit a vacation request?",
  "top_k": 5
}
```

The answer response includes `answer`, `abstained`, and `sources` with document IDs, chunk IDs, source text, and similarity scores.

## Local setup

Requires Python 3.11+ and PostgreSQL with the pgvector extension.

1. Create and activate a virtual environment.

   **Windows PowerShell**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the package and development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

3. Create a PostgreSQL database, enable pgvector, and apply the schema:

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

   ```bash
   psql "$DATABASE_URL" -f sql/schema.sql
   ```

4. Copy `.env.example` to `.env` and set `DATABASE_URL` and `OPENAI_API_KEY`. Never commit real credentials.

5. Start the service:

   ```bash
   uvicorn app.main:app --reload
   ```

   Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | Required | PostgreSQL connection string |
| `OPENAI_API_KEY` | Empty | Provider key; required for model calls |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (schema currently expects 1536 dimensions) |
| `CHAT_MODEL` | `gpt-4o-mini` | Answer-generation model |
| `TOP_K` | `5` | Default retrieval count |
| `MIN_SIMILARITY` | `0.25` | Initial cosine-similarity threshold; calibrate with evaluation data |
| `CHUNK_SIZE` | `1200` | Character window size |
| `CHUNK_OVERLAP` | `150` | Character overlap |

## Tests and quality checks

```bash
ruff check app tests
pytest
```

The initial tests cover chunking boundaries, request validation, and basic API behavior. They do **not** yet prove database persistence, provider integration, retrieval relevance, or answer faithfulness. Those are planned next milestones.

## Roadmap

- [ ] Make CI pass and record the result.
- [ ] Add a low-memory local PostgreSQL/pgvector setup.
- [ ] Add integration tests for ingestion, document replacement, and retrieval ordering.
- [ ] Create a small, versioned question/answer evaluation dataset.
- [ ] Measure retrieval Recall@k, abstention behavior, and grounded-answer quality.
- [ ] Add structured request IDs, logs, metrics, and explicit latency/error measurements.
- [ ] Document prompt-injection boundaries and tenant isolation before multi-user deployment.

## Known limitations

- Character-based chunking is intentionally simple; compare against token-aware and structure-aware chunking later.
- The similarity threshold is a starting point, not a universal guarantee of relevance.
- The first version is a single knowledge collection and has no authentication or tenant isolation. Do not expose it as a multi-user service or ingest confidential data.
- API handlers currently call provider/database functions synchronously. Evaluate async execution, connection pooling, and background ingestion only after measuring the MVP.
- The database schema uses 1536-dimensional vectors; changing the embedding model may require a schema migration.
