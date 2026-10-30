# Repeated Latency Benchmark

This benchmark measures two sequential `/v1/ask` paths under controlled local conditions:

- a relevant question that performs embedding, retrieval, and generation;
- an unsupported question that performs embedding and retrieval, then abstains without generation.

Use a dedicated PostgreSQL database containing only the benchmark document. Run one generated
warm-up request, then at least ten measured iterations per request kind:

```bash
python -u -m evals.latency.benchmark \
  --run-id rag007-YYYYMMDD \
  --iterations 10 \
  --warmup 1
```

Summarize the matching structured container events:

```bash
docker compose logs --no-color api 2>&1 | \
  python -m evals.latency.summarize --run-id rag007-YYYYMMDD
```

The summarizer reports minimum, p50, p95, and maximum values for embedding, retrieval, generation,
and total server time. It uses linear interpolation with rank `(n - 1) * q`. Warm-up requests are
excluded. The client report and stage report are ignored by Git; publish a reviewed baseline with
the model, database isolation, execution order, hardware class, and sample count.

Ten samples per path provide an exploratory local baseline, not a production SLO or load test. The
benchmark is sequential and does not measure concurrency, throughput, saturation, or tail latency
under load.

## Recorded local baseline

The reviewed 30 October 2026 baseline used local CPU-only inference, `nomic-embed-text`,
`qwen3:4b`, `top_k=1`, and a dedicated disposable PostgreSQL/pgvector database. One generated
warm-up was excluded, followed by ten measured requests per path.

| Path | Server p50 | Server p95 | Dominant stage |
|---|---:|---:|---|
| Generated answer | 54.868 s | 65.889 s | Generation: 54.658 s p50 |
| Abstention | 0.340 s | 0.835 s | Embedding: 0.220 s p50 |

At the median, generation accounted for 99.62% of generated-request server time. This identifies
local CPU model generation—not pgvector retrieval—as the primary latency bottleneck for this
configuration. See the [machine-readable baseline](baselines/ollama-qwen3-4b-local-cpu.json).
