# RAG Evaluation Dataset v2

Version 2 expands the small v1 smoke benchmark to two synthetic documents and eight cases. It is designed to expose retrieval and grounding regressions, not to establish production-grade RAG quality.

## Coverage

- Two documents with seven pages total.
- Paraphrased questions for remote work, expenses, and paid time off.
- A multi-page security-incident question whose expected facts span pages 3 and 4.
- A benefits FAQ containing semantically related commuter and home-office information.
- A near-miss unsupported question about parking prices, plus an unsupported dental-provider question.
- Per-case expected document IDs, page sets, answer facts, and abstention labels.

The labels are authored as a starting point and should be reviewed by a human before treating the benchmark as a release gate. The current deterministic answer scorer checks literal expected-fact substrings; it does not understand paraphrases, entailment, or partial correctness.

## Run

Start the API, PostgreSQL/pgvector, and the local Ollama provider as described in the root README. Then run:

```bash
python -m evals.v1.evaluate --dataset evals/v2/dataset.json --output evals/v2/results/latest.json
```

The evaluator builds the v2 PDF documents deterministically in memory, ingests them under stable namespaced document IDs, and queries the public API. It reports retrieval success across every expected page, expected-fact answer correctness, strict citations for every expected page, abstention accuracy, and end-to-end latency.

No v2 performance result is committed yet. Run it on the target machine and preserve the report with the model, threshold, and hardware conditions before comparing versions. Keep v1 as the original baseline rather than overwriting its benchmark.
