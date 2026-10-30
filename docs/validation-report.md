# Validation Report

**Status date:** 30 October 2026  
**Validated merge:** `b92e0eea6d8f7157f96d0520766b2e4ff58e79cb` (PR #11)

This report records what has actually been implemented and tested. It intentionally separates
automated backend verification from local model-quality evaluation.

## Implemented scope

- FastAPI endpoints for health checks, text ingestion, PDF ingestion, and grounded questions.
- Page-by-page extraction for text-based PDFs, page-scoped chunking, and source-page metadata.
- Embeddings through an OpenAI-compatible provider; local Compose defaults to Ollama
  `nomic-embed-text`.
- PostgreSQL persistence and cosine retrieval through pgvector.
- Transactional document replacement: failed re-ingestion preserves the previous document.
- Evidence thresholding and explicit abstention when no retrieved chunk is sufficiently relevant.
- Grounded answer generation with structured sources.
- Citation normalization and validation against retrieved chunk IDs and page numbers.
- PDF type and 10 MiB upload limits, malformed-PDF handling, provider timeouts, and safe client
  errors that do not expose infrastructure details.
- Request IDs and stage timing records for embedding, retrieval, generation, and total duration.
- Docker Compose for the API and PostgreSQL/pgvector, plus GitHub Actions for linting, tests, and
  image construction.
- Versioned v1 and v2 RAG evaluation datasets with expected facts, pages, citations, abstention
  cases, and latency measurements.

## Automated verification

The merged code passed:

| Check | Result |
|---|---:|
| Ruff | Passed |
| Unit tests | 42 passed |
| PostgreSQL/pgvector integration tests | 5 passed |
| Docker image build | Passed |
| GitHub Actions on merge commit `b92e0ee` | Passed |

The integration tests use a real PostgreSQL/pgvector service while replacing model calls with
deterministic embeddings. They verify API persistence, successful replacement, rollback after a
failed replacement, vector ordering, `top_k=1`, and page metadata.

## Local Ollama evaluation

Both evaluations used local Ollama with `nomic-embed-text`, `qwen3:4b`, `top_k=5`, and a similarity
threshold of `0.60`. The v2 result used a dedicated disposable database containing only its two
declared documents, preventing unrelated records from changing retrieval or prompt size.

| Metric | v1 validation | Isolated v2 validation |
|---|---:|---:|
| Cases | 6 | 8 |
| Answerable cases | 4 | 6 |
| Unsupported cases | 2 | 2 |
| Expected-page retrieval | 100% | 100% |
| Expected-fact correctness | 100% | 100% |
| Citation accuracy | 100% | 100% |
| Abstention accuracy | 100% | 100% |
| Mean `/v1/ask` latency | 66.064 s | 63.839 s |
| Maximum `/v1/ask` latency | 153.369 s | 121.237 s |

The committed machine-readable baselines are:

- [`evals/v1/baselines/ollama-qwen3-4b-cpu.json`](../evals/v1/baselines/ollama-qwen3-4b-cpu.json)
- [`evals/v2/baselines/ollama-qwen3-4b-cpu.json`](../evals/v2/baselines/ollama-qwen3-4b-cpu.json)

The original v1 baseline recorded 25% strict citation accuracy before citation normalization. A
post-change rerun reached 100% on the same small dataset. This is regression evidence, not proof of
general RAG quality.

## Reproduce the checks

```bash
ruff check app evals tests
pytest -m "not integration"
pytest -m integration
python -m evals.v1.evaluate
python -m evals.v2.evaluate
```

Integration tests require PostgreSQL with pgvector. Live evaluations additionally require the API
and configured model provider. Run evaluation datasets in an isolated database: the API searches
the entire collection, so unrelated documents can change retrieval results, answers, and latency.

## Known limitations and unverified claims

- The evaluation datasets are small and synthetic. Their perfect scores must not be generalized to
  arbitrary business documents or user questions.
- CPU-only local generation is slow and variable. No production latency SLO or throughput claim is
  supported.
- Latency percentiles have not been measured over a statistically useful repeated sample.
- Stage timing records exist, but application INFO records are suppressed by the current default
  Uvicorn logging configuration. Container-level stage observability remains incomplete.
- Scanned/image-only PDFs require OCR and are not supported.
- Chunking is character-based rather than token- or structure-aware.
- The system has no authentication, authorization, tenant isolation, or per-user collections.
- API handlers and provider/database calls are synchronous; connection pooling and background
  ingestion have not been implemented.
- Local Ollama has no hosted token charge, but compute/energy cost has not been measured.

The next RAG engineering task is to make stage timings observable in deployed container logs and
then collect repeated latency samples before reporting percentiles.
