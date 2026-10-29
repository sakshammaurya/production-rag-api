# RAG Evaluation Dataset v2

Version 2 expands the small v1 smoke benchmark to two synthetic documents and eight cases. It is designed to expose retrieval and grounding regressions, not to establish production-grade RAG quality.

## Coverage

- Two documents with seven pages total.
- Paraphrased questions for remote work, expenses, and paid time off.
- A multi-page security-incident question whose expected facts span pages 3 and 4.
- A benefits FAQ containing semantically related commuter and home-office information.
- A near-miss unsupported question about parking prices, plus an unsupported dental-provider question.
- Per-case expected document IDs, page sets, answer facts, and abstention labels.

The document/page mappings and expected facts were manually reviewed against the authored corpus on
2026-10-29. The current deterministic answer scorer checks literal expected-fact substrings; it
does not understand paraphrases, entailment, or partial correctness. Expected facts therefore use
the smallest wording needed to verify the question's answer rather than requiring unrelated details.

## Run

Start the API, PostgreSQL/pgvector, and the local Ollama provider as described in the root README. Then run:

```bash
python -m evals.v1.evaluate --dataset evals/v2/dataset.json --output evals/v2/results/latest.json
```

The evaluator builds the v2 PDF documents deterministically in memory, ingests them under stable namespaced document IDs, and queries the public API. It reports retrieval success across every expected page, expected-fact answer correctness, strict citations for every expected page, abstention accuracy, and end-to-end latency.

Run benchmarks against a dedicated database containing only the declared evaluation documents.
The API searches the whole collection, so unrelated or earlier evaluation documents can change the
retrieved context, generation latency, and answers. Stable document IDs replace matching documents
but do not remove other records.

## Reviewed local baseline

The expected pages and facts were manually checked against both authored documents. A clean run on
2026-10-29 used a dedicated disposable PostgreSQL database, Ollama 0.40.12,
`nomic-embed-text`, `qwen3:4b`, `top_k=5`, and similarity threshold `0.60`. Ollama ran on an Intel
Core i5-6200U CPU with no VRAM allocation.

| Metric | Result |
|---|---:|
| Cases | 8 (6 answerable, 2 unsupported) |
| Expected-page retrieval hit rate | 100% (6/6) |
| Expected-fact answer correctness | 100% (6/6) |
| Strict citation accuracy | 100% (6/6) |
| Abstention accuracy | 100% (8/8) |
| Mean question latency | 63.839 s |
| Maximum question latency | 121.237 s |

These perfect quality scores apply only to this small synthetic dataset. They do not establish
production-grade quality. The committed machine-readable baseline records per-case latency and
top similarity. Keep v1 as the original historical baseline rather than overwriting it.
