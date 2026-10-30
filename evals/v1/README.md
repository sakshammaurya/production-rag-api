# RAG Evaluation Dataset v1

This dataset is intentionally small and transparent. It is a regression baseline, not evidence of
general RAG quality. The corpus contains one four-page synthetic employee handbook, four
answerable questions with expected facts and pages, and two unsupported questions that should
abstain.

## Metrics

- **Retrieval hit rate:** the expected document and page appear in the structured sources returned
  within the configured `top_k`. Because v1 labels one relevant page per answerable question, this
  is equivalent to page-level Recall@k for this dataset.
- **Answer correctness:** every case-sensitive-independent expected fact occurs in a non-abstained
  answer. This is deterministic but does not capture paraphrases outside the declared facts.
- **Citation accuracy:** at least one complete `[Source <id>, Page <number>]` label in the answer
  matches a returned structured source on the expected page.
- **Abstention accuracy:** answerable cases do not abstain; unsupported cases abstain and return no
  sources.
- **Latency:** wall-clock duration of each `/v1/ask` request. Ingestion time is excluded.

## Historical baseline: before citation normalization

Measured on 2026-10-29 with Ollama 0.40.12, `nomic-embed-text`, `qwen3:4b`, PostgreSQL/pgvector,
`top_k=5`, similarity threshold `0.60`, chunk size `1200`, and overlap `150`. The machine used an
Intel Core i5-6200U CPU and approximately 12 GB RAM; Ollama reported both models running without
VRAM allocation.

| Metric | Result |
|---|---:|
| Cases | 6 (4 answerable, 2 unsupported) |
| Retrieval hit rate / page Recall@5 | 100% (4/4) |
| Expected-fact answer correctness | 100% (4/4) |
| Strict citation accuracy | 25% (1/4) |
| Abstention accuracy | 100% (6/6) |
| Mean question latency | 55.301 s |
| Maximum question latency | 162.686 s |

Per-case latency was 162.686, 59.405, 58.818, 50.320, 0.270, and 0.310 seconds in dataset order.
The first generation was a cold-model outlier. Unsupported questions were fast because retrieval
scores remained below the evidence threshold and the API skipped answer generation.

Three correct answers referred to the correct source and page but did not reproduce the complete
bracketed label required by the API prompt. For example, the model emitted `Source 38, Page 1`
instead of `[Source 38, Page 1]`. The strict citation metric records these as failures. This is a
useful contract-compliance finding and should not be hidden by weakening the evaluator after seeing
the output.

Machine-readable evidence: [`baselines/ollama-qwen3-4b-cpu.json`](baselines/ollama-qwen3-4b-cpu.json).

## Validation after citation normalization

The same v1 dataset was rerun after PR #6 added citation normalization and validation. The later
run is stored separately rather than overwriting the historical baseline.

| Metric | Post-normalization result |
|---|---:|
| Cases | 6 (4 answerable, 2 unsupported) |
| Retrieval hit rate / page Recall@5 | 100% (4/4) |
| Expected-fact answer correctness | 100% (4/4) |
| Strict citation accuracy | 100% (4/4) |
| Abstention accuracy | 100% (6/6) |
| Mean question latency | 66.064 s |
| Maximum question latency | 153.369 s |

Machine-readable evidence:
[`baselines/ollama-qwen3-4b-cpu-post-citation-normalization.json`](baselines/ollama-qwen3-4b-cpu-post-citation-normalization.json).

This comparison provides evidence that citation contract compliance improved on v1. The latency
difference is not treated as a performance improvement or regression because it is a single,
CPU-only run with substantial runtime variability.

Do not generalize the perfect retrieval and answer scores beyond this synthetic six-case dataset.
Future versions should add multiple documents, semantically similar distractors, paraphrases,
multi-page answers, and human-reviewed answer-support labels.
