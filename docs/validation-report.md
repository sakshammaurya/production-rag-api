# Validation Report

**Status date:** 14 November 2026  
**Latest documented merged change:** PR #21, merged 5 November 2026

This report incorporates the native RAG implementation and the subsequent authentication,
observability, and local inference experiments through PR #21. It distinguishes historical
automated checks recorded in pull-request descriptions from a fresh CI run: the current CI status
was not independently verified as part of this documentation refresh.

This report records what has actually been implemented and tested. It intentionally separates
automated backend verification from local model-quality evaluation.

## Implemented scope

- FastAPI endpoints for health checks, text ingestion, PDF ingestion, and grounded questions.
- Page-by-page extraction for text-based PDFs, page-scoped chunking, and source-page metadata.
- Embeddings through an OpenAI-compatible provider; local Compose defaults to Ollama
  `nomic-embed-text`.
- PostgreSQL persistence and cosine retrieval through pgvector.
- API-key authentication with server-resolved tenant identity and tenant-scoped replacement and
  vector retrieval.
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

The original PR #11 validation recorded Ruff, 42 unit tests, 5 PostgreSQL/pgvector integration
tests, a Docker image build, and passing GitHub Actions on merge commit `b92e0ee`. Those counts are
historical and are not the current suite totals.

Later PR descriptions record the following focused verification:

| Change | Validation recorded in the PR description |
|---|---|
| PR #16 — API-key authentication and tenant isolation | Security and cross-tenant integration coverage |
| PR #18 — inference telemetry | Ruff, 55 unit tests, 6 PostgreSQL/pgvector integration tests, and a real Ollama smoke run |
| PR #19 — output-token ceiling screen | Ruff, 56 unit tests, 6 integration tests, and unchanged Docker/API behavior |
| PR #20 — CPU thread benchmark controls | Ruff, 57 unit tests, 6 integration tests, JSON/diff checks |
| PR #21 — Q8 quantization experiment | Ruff, 57 unit tests, 6 integration tests, machine-readable artifact validation, diff check |

These are results reported by the corresponding PR descriptions. They should not be represented as
a fresh run performed for this report. A current CI status check was not available during this
refresh.

The integration suite uses a real PostgreSQL/pgvector service while replacing model calls with
deterministic embeddings. It verifies API persistence, replacement/rollback behavior, vector
ordering, tenant boundaries, and page metadata. It does not measure semantic answer quality or
model latency.

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

The committed machine-readable evidence is:

- [v1 historical pre-normalization baseline](../evals/v1/baselines/ollama-qwen3-4b-cpu.json):
  25% citation accuracy, 55.301 s mean latency.
- [v1 post-normalization validation](../evals/v1/baselines/ollama-qwen3-4b-cpu-post-citation-normalization.json):
  100% citation accuracy, 66.064 s mean latency.
- [`evals/v2/baselines/ollama-qwen3-4b-cpu.json`](../evals/v2/baselines/ollama-qwen3-4b-cpu.json)

The original v1 baseline and post-change rerun are intentionally separate artifacts. This is
regression evidence for citation formatting on a small dataset, not proof of general RAG quality.

## Reproduce the checks

```bash
ruff check app evals tests
pytest -m "not integration"
pytest -m integration
python -m evals.v1.evaluate
python -m evals.v2.evaluate
```

Integration tests require PostgreSQL with pgvector. Live evaluations additionally require the API
and configured model provider. Run evaluation datasets in an isolated tenant or database: the API
searches the tenant's entire collection, so unrelated documents in that tenant can change retrieval
results, answers, and latency.

## Known limitations and unverified claims

- The evaluation datasets are small and synthetic. Their perfect scores must not be generalized to
  arbitrary business documents or user questions.
- The v2 expected-fact scorer is deterministic literal substring matching. It is not semantic
  entailment evaluation and can miss valid paraphrases or over-credit surface overlap. Treat its
  answer/fact scores as regression signals for the declared wording, not a complete measure of
  factual correctness.
- CPU-only local generation is slow and variable. No production latency SLO or throughput claim is
  supported.
- Latency percentiles have not been measured over a statistically useful repeated sample.
- Stage timing events are emitted as structured JSON through Uvicorn's active application logger.
  They are process-local logs rather than a durable metrics backend; aggregation, retention, and
  alerting remain deployment concerns.
- Scanned/image-only PDFs require OCR and are not supported.
- Chunking is character-based rather than token- or structure-aware.
- Authentication uses static tenant API keys rather than end-user identities, roles, OAuth/OIDC,
  rotation/revocation workflows, or PostgreSQL row-level security.
- API handlers and provider/database calls are synchronous; connection pooling and background
  ingestion have not been implemented.
- Local Ollama has no hosted token charge, but compute/energy cost has not been measured.

## Inference telemetry and experiment scope

PR #18 added reproducible Ollama inference telemetry alongside the existing request-stage timings.
The benchmark artifacts record model/runtime metadata, prompt and context measurements, generated
answer p50/p95/maximum latency, token counts, model-load timing, and generation throughput. The
captured-context chat replay uses the same production prompt builder so the local inference
experiments can be compared against the API's actual prompt construction.

The inference experiments through PR #21 are documented below and in
[`evals/inference/README.md`](../evals/inference/README.md). They are local CPU experiments on the
recorded host, not production load tests. The qwen3:4b Q4_K_M configuration remains the default;
the smaller model, token ceiling, thread override, and Q8 quantization candidates were not adopted.

## Repeated latency validation

A controlled local run used a dedicated disposable database, one excluded generated warm-up, then
ten sequential generated-answer requests and ten sequential abstention requests with `top_k=1`.
The run used local CPU-only inference with `nomic-embed-text` and `qwen3:4b`.

| Path | Total p50 | Total p95 | Generation p50 | Retrieval p50 |
|---|---:|---:|---:|---:|
| Generated answer | 54.868 s | 65.889 s | 54.658 s | 0.083 s |
| Abstention | 0.340 s | 0.835 s | 0.000 s | 0.126 s |

Generation consumed 99.62% of median generated-request server time, so the local CPU generation
model is the measured bottleneck. pgvector retrieval was not the dominant cost. The
[machine-readable baseline](../evals/latency/baselines/ollama-qwen3-4b-local-cpu.json) includes
minimum, p50, p95, and maximum values for every stage and client-observed latency.

These ten-sample percentiles are exploratory. The benchmark is sequential and does not establish a
production SLO, throughput, concurrency behavior, or latency under load.

## Model-size experiment

An isolated v2 run compared the default `qwen3:4b` with `qwen3:1.7b`. The smaller model reduced
mean end-to-end latency from 63.839 seconds to 20.457 seconds, but answer correctness dropped from
100% to 83.33% because one multi-page answer omitted a required fact. Retrieval, citation, and
abstention metrics remained at 100% on the eight-case dataset.

The smaller model was rejected as the default because it did not preserve the declared quality
gate. See the [candidate artifact](../evals/v2/baselines/ollama-qwen3-1.7b-cpu-candidate.json).

## Output-token ceiling experiment

After adding native Ollama telemetry, `num_predict=256` was screened once across all six v2
answerable cases, after one excluded warm-up. Every case consumed exactly 256 generated tokens.
Five cases returned no final answer because the ceiling was reached during Qwen's reasoning; the
remaining answer contained its expected fact but omitted its citation. Answer correctness and
mean fact recall were 16.67%, while critical-fact recall and citation accuracy were 0%.

The candidate failed the quality gate, so repeated latency measurements were deliberately skipped
and production defaults were not changed. The 63.194-second candidate mean is screening evidence,
not a performance improvement claim. See the
[rejected experiment artifact](../evals/inference/baselines/qwen3-4b-num-predict-256-rejected.json).

## CPU thread experiment

The local machine exposes two physical cores and four logical processors. With model,
quantization, prompts, evidence, and output budget fixed, explicit `num_thread=2` reduced mean
generation throughput to 3.805 tokens/second. Explicit `num_thread=4` preserved all quality gates
and reached 4.887 tokens/second in the six-case screen, while saturating all logical processors.

A focused three-pair alternating comparison used an identical 143-token prompt and 372-token
generation. Four threads averaged 4.857 tokens/second versus 4.729 for the runtime default, a small
2.71% exploratory difference; mean generation time fell 2.64%. Switching thread configurations
also forced model reloads, whose 35-61 second cost was analyzed separately from generation.

The runtime default remains unchanged because the small observed gain is not strong enough to
justify an override. See the
[thread experiment artifact](../evals/inference/baselines/qwen3-4b-cpu-thread-screen.json).

## Quantization experiment

The official Qwen3 4B thinking `Q8_0` variant was compared with the current `Q4_K_M` model on the
same CPU, prompts, evidence, and uncapped answer path. Q8 used 4.3 GB on disk and a 5.0 GB loaded
footprint, compared with 2.5 GB and 3.2 GB for Q4.

The Q8 full quality screen did not complete: four cases took 101.5-178.8 seconds, then the commuter
case exceeded the 600-second HTTP timeout. A separate stable expense case produced a correct cited
answer, but generation throughput fell from 4.817 tokens/second with Q4 to 3.326 with Q8. Q8 took
26.72% longer to generate despite producing fewer tokens.

Q8 was rejected before repeated benchmarking. `Q4_K_M` remains the production default because Q8
increased storage and memory requirements, reduced throughput by 30.95% on the stable comparison,
and failed to complete the quality gate. See the
[quantization artifact](../evals/inference/baselines/qwen3-4b-q8-rejected.json).
