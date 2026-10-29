# Production RAG API

A portfolio project demonstrating a **testable Retrieval-Augmented Generation (RAG) backend** with FastAPI, PostgreSQL/pgvector, an OpenAI-compatible model provider, and pytest.

> **Current milestone:** Local PDF RAG prototype with page-aware citations and automated PostgreSQL/pgvector integration coverage. Broader retrieval and answer-quality evaluation is still pending, so this is not production-ready.

## What this project demonstrates

- **Backend design:** typed API contracts, input validation, persistence boundaries, and environment-based configuration.
- **RAG fundamentals:** document chunking, embedding generation, cosine-distance retrieval, and source metadata.
- **Grounding behavior:** evidence thresholding and an explicit insufficient-evidence response.
- **Reliability basics:** bounded inputs, parameterized SQL, provider timeouts/retries, and no raw infrastructure details in API error responses.
- **Engineering discipline:** isolated unit tests, real PostgreSQL/pgvector integration tests, and a GitHub Actions workflow.

## Architecture

```text
POST /v1/documents
  validate → chunk text → embed chunks → PostgreSQL + pgvector

POST /v1/documents/pdf
  validate size/type → extract each page → page-scoped chunks → embed → PostgreSQL + pgvector

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
| `POST /v1/documents/pdf` | Extract, chunk, embed, and store a PDF with page numbers |
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

### Example: ingest a PDF

Upload a text-based PDF as multipart form data. Scanned/image-only PDFs require OCR and are
not supported in this milestone.

```bash
curl -X POST http://127.0.0.1:8000/v1/documents/pdf \
  -F "document_id=handbook-pdf" \
  -F "title=Team handbook" \
  -F "file=@handbook.pdf;type=application/pdf"
```

Each extracted page is chunked independently. Retrieval sources include `page_number`, and the
answer-generation prompt asks the model to preserve that page citation.

### Example: ask a question

```json
{
  "question": "How early should I submit a vacation request?",
  "top_k": 5
}
```

The answer response includes `answer`, `abstained`, and `sources` with document IDs, chunk IDs, source text, and similarity scores. Every response includes an `X-Request-ID` header; a safe caller-supplied ID is preserved, otherwise the API generates one. Completed ask requests emit a `rag_request_completed` log record with `request_id`, `embedding_ms`, `retrieval_ms`, `generation_ms`, `total_ms`, `abstained`, and `evidence_count`. Failed provider/database stages emit `rag_stage_failed` or ingestion-specific error logs with the same request ID. Client responses remain generic and do not expose exception details.

## Docker Compose (recommended for a reproducible setup)

Prerequisites: Docker Desktop (or Docker Engine with the Compose plugin) and Ollama running. The default Compose configuration expects `nomic-embed-text` and `qwen3:4b` to be installed in Ollama on the host.

1. Create your local environment file:

   **Windows PowerShell**
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   The example configuration uses the local Ollama server through Docker Desktop's `host.docker.internal` address. Override the provider variables if you want to use another OpenAI-compatible service. Never commit real credentials.

2. Start the API and PostgreSQL/pgvector:

   ```bash
   docker compose up --build
   ```

   The database schema is applied automatically when the PostgreSQL data volume is first initialized. The API waits for PostgreSQL's health check before starting.

3. Open `http://127.0.0.1:8000/docs`. Check `GET /health` first.

4. Stop the services with `Ctrl+C`, then run:

   ```bash
   docker compose down
   ```

   Database data remains in the named Docker volume. To **permanently delete** that local database and its data, run `docker compose down -v`.

The default local Ollama configuration uses a placeholder API key because the OpenAI client requires a value, while Ollama ignores it. Docker Compose is a local development setup, not a production security configuration; its database password is intentionally simple and must not be reused in a public deployment.

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
| `OPENAI_API_KEY` | Empty | Provider key; use any non-empty value for local Ollama |
| `OPENAI_BASE_URL` | Provider default | OpenAI-compatible endpoint; Compose uses `http://host.docker.internal:11434/v1` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model; Compose uses `nomic-embed-text` |
| `CHAT_MODEL` | `gpt-4o-mini` | Answer-generation model; Compose uses `qwen3:4b` |
| `PROVIDER_TIMEOUT_SECONDS` | `600` | Per-request provider timeout; CPU-only local models may be slow |
| `PROVIDER_MAX_RETRIES` | `0` | Provider retry count |
| `TOP_K` | `5` | Default retrieval count |
| `MIN_SIMILARITY` | `0.60` | Locally measured cosine threshold; recalibrate with broader evaluation data |
| `CHUNK_SIZE` | `1200` | Character window size |
| `CHUNK_OVERLAP` | `150` | Character overlap |
| `MAX_PDF_UPLOAD_BYTES` | `10485760` | Maximum PDF upload size (10 MiB); larger uploads return HTTP 413 |

## Tests and quality checks

```bash
ruff check app tests
pytest -m "not integration"
```

Integration tests require a reachable PostgreSQL database with the pgvector extension and an
applied schema. With `DATABASE_URL` configured, run:

```bash
pytest -m integration
```

The integration suite exercises the real API/repository/database boundary. It verifies text
ingestion persistence, transactional document replacement, cosine-similarity ordering, and page
metadata. Provider calls are replaced with deterministic embeddings so CI does not depend on a
model server or network behavior. These tests prove storage and retrieval mechanics; they do
**not** measure semantic retrieval quality, answer correctness, citation support, or provider
latency. Those are measured separately by the versioned evaluation harness below.

## RAG evaluation dataset

`evals/v1` contains a versioned, reviewable benchmark with a deterministic four-page PDF, four
answerable questions, and two deliberately unsupported questions. It measures expected-page
retrieval, required facts in generated answers, model citations that map to returned structured
sources, abstention, and end-to-end question latency.

The [v1 evaluation notes](evals/v1/README.md) define each metric and record the first local Ollama
baseline, including the observed citation-format failures and hardware conditions.

Start the complete API stack with Ollama available, then run:

```bash
python -m evals.v1.evaluate
```

The evaluator re-ingests its namespaced document ID on every run and writes the detailed report to
`evals/v1/results/latest.json`. Generated reports are ignored by Git because results depend on the
machine, configured models, threshold, and runtime conditions. Record those conditions alongside
any published benchmark. Rebuild the committed PDF deterministically after editing
`evals/v1/corpus/pages.json` with:

```bash
python -m evals.v1.build_corpus
```

The expanded v2 benchmark covers two synthetic documents, paraphrased questions, related distractors, multi-page answers, and unsupported near-miss questions. Run it with `python -m evals.v2.evaluate` while the API and Ollama stack are running. Its expected facts and page labels were manually reviewed on 2026-10-29, and the first isolated local baseline is documented in `evals/v2/README.md`.

## Roadmap

- [x] Make CI pass and record the result (GitHub Actions lint and unit tests passed for the scaffold).
- [x] Verify the Docker Compose PostgreSQL/pgvector setup on a local machine.
- [x] Add integration tests for ingestion, document replacement, and retrieval ordering.
- [x] Add PDF upload, text extraction, and page-level source citations.
- [x] Create a small, versioned question/answer evaluation dataset.
- [x] Measure retrieval Recall@k, abstention behavior, and grounded-answer quality on dataset v1.
- [ ] Make request/stage metrics observable in container logs; request IDs and timing records exist in code, but the default Uvicorn logging configuration currently suppresses the application INFO records.
- [x] Document prompt-injection boundaries and the lack of tenant isolation before multi-user deployment.

## Security boundaries

- Retrieved document text is untrusted input. The generation prompt tells the model to treat evidence as data rather than instructions, but prompt wording is not a complete prompt-injection defense.
- Do not place secrets in indexed documents, and do not treat generated answers or citations as authorization decisions. Validate any future tool/action arguments independently.
- This version has no authentication, authorization, tenant isolation, or per-customer document filtering. All indexed chunks share one collection. Do not expose it as a multi-user service or ingest confidential/customer data.
- The included Docker Compose credentials are for local development only. A public deployment requires secret management, network restrictions, authentication, resource limits, and operational monitoring.

## Known limitations

- Character-based chunking is intentionally simple; compare against token-aware and structure-aware chunking later.
- The similarity threshold is a starting point, not a universal guarantee of relevance.
- The first version is a single knowledge collection and has no authentication or tenant isolation. Do not expose it as a multi-user service or ingest confidential data.
- API handlers currently call provider/database functions synchronously. Evaluate async execution, connection pooling, and background ingestion only after measuring the MVP.
- A collection must use embeddings with a consistent dimension so pgvector distance operations can compare them.
