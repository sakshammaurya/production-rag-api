# RAG Evaluation Dataset v2

Version 2 expands the small v1 smoke benchmark to two synthetic documents and eight cases. It is designed to expose retrieval and grounding regressions, not to establish production-grade RAG quality.

## Coverage

- Two documents with seven pages total.
- Paraphrased questions for remote work, expenses, and paid time off.
- A multi-page security-incident question whose expected facts span pages 3 and 4.
- A benefits FAQ containing semantically related commuter and home-office information.
- A near-miss unsupported question about parking prices, plus an unsupported dental-provider question.
- Per-case expected document IDs, page sets, answer facts, and abstention labels.
- Manually approved fact variants and explicit forbidden phrases for selected cases.

The document/page mappings and expected facts were manually reviewed against the authored corpus on
2026-10-29. The deterministic scorer accepts a canonical expected-fact phrase or a manually
approved phrase from `expected_fact_variants`. Selected `forbidden_phrases` explicitly fail answer
correctness when present. For example:

```json
{
  "expected_facts": ["not investigate"],
  "expected_fact_variants": {
    "not investigate": ["do not investigate"]
  },
  "forbidden_phrases": [
    "employees should investigate the incident themselves"
  ]
}
```

Each fact result reports the canonical fact, whether it matched, the exact approved phrase that
matched, and whether the fact is critical. Each case reports detected forbidden phrases and its
contradiction-check result. Datasets that omit both optional fields retain exact-match behavior.

This remains a deterministic lexical evaluator, not semantic understanding or entailment checking.
It recognizes only curated phrases and can miss valid unlisted paraphrases or contradictions with
different wording. Expected facts therefore use the smallest wording needed to verify the answer
without requiring unrelated details.

Substring matching can also produce false positives. For example, `"10 days"` matches inside
`"110 days"`. Word-boundary-aware matching may be evaluated in a future change; this version keeps
the historical case-insensitive substring behavior for compatibility.

## Run

Start the API, PostgreSQL/pgvector, and the local Ollama provider as described in the root README.
Set `RAG_API_KEY` to a configured tenant key, then run:

```bash
python -m evals.v1.evaluate --dataset evals/v2/dataset.json --output evals/v2/results/latest.json
```

The evaluator builds the v2 PDF documents deterministically in memory, ingests them under stable
namespaced document IDs, and queries the public API. It reports retrieval success across every
expected page, atomic expected-fact recall, critical-fact recall, evidence coverage, context size,
strict citations, abstention accuracy, overall latency, and generated-answer p50/p95 latency.
Historical baseline artifacts are preserved and are not rescored or overwritten automatically.

## Retrieval ranking evaluation

The v2 dataset evaluates API-visible retrieval ranking at `K = 1, 3, 5`. These values are declared
as `evaluation.ranking_k_values` and cannot exceed the API request's `top_k`. Existing datasets
without this field continue to report ranking metrics at their configured `top_k` only.

The metrics use binary relevance at **document-page level**. An item is relevant only when both its
`document_id` and `page_number` match a declared expected page for the case.

- **Precision@K:** relevant unique pages in the first K ranked page items divided by K. The
  denominator remains K when the API returns fewer than K items, so thresholded or missing results
  are visible in the score.
- **Recall@K:** relevant unique pages in the first K divided by all declared relevant pages.
- **MRR@K:** the mean, across eligible cases, of the reciprocal rank of the first relevant page in
  the first K items. A case with no relevant result contributes zero.
- **nDCG@K:** binary discounted cumulative gain divided by the ideal ranking for the declared
  relevant pages.

Multiple chunks from the same document page are collapsed at their first occurrence before ranks
are calculated. This prevents chunk overlap from giving one page repeated relevance credit. A
page-less chunk remains a distinct irrelevant ranked item because it cannot satisfy a page label.
Cases with no expected pages, such as unsupported questions, are excluded from ranking averages;
the summary reports eligible and excluded case counts separately.

Example summary shape:

```json
{
  "retrieval_ranking": {
    "relevance_granularity": "document_page",
    "eligible_case_count": 6,
    "excluded_case_count": 2,
    "by_k": {
      "3": {
        "precision": 0.3333,
        "recall": 1.0,
        "mrr": 1.0,
        "ndcg": 1.0
      }
    }
  }
}
```

### Ranking assumptions and trade-offs

PostgreSQL orders candidate chunks by ascending cosine distance. The repository, similarity
threshold filter, and response-model construction preserve that order, so `sources` is the ranked
evidence exposed by the public API. The evaluator intentionally measures this post-threshold list,
not hidden candidates rejected by `min_similarity`; this reflects generation-visible retrieval but
cannot separately diagnose vector ranking versus threshold selection.

Page-level labels match the dataset's reviewed page citations and avoid treating overlapping chunks
as independent relevant answers. The trade-off is that the benchmark cannot distinguish two chunks
of different quality on the same page. The API also applies `top_k` to chunks before the evaluator
deduplicates pages. Duplicate chunks can therefore leave fewer than K unique pages available, and
the evaluator cannot recover a page whose chunk ranked below the API cutoff. Binary relevance cannot
express that one relevant page is more useful than another. Equal-distance database results have no
explicit secondary sort key, so tied items may change order between database executions. Production
retrieval is not modified to remove these limitations in this evaluation-only change.

These ranking metrics measure placement of manually labelled pages. They do not establish semantic
answer correctness, factual faithfulness, citation support, or performance on arbitrary documents.

The security-incident case marks the prohibition `not investigate` as a critical fact. This makes
the previously observed smaller-model failure visible separately from aggregate answer accuracy.
Exact Ollama prompt/generation token and duration metrics are collected by the
[offline inference benchmark](../inference/README.md) using source contexts captured in this
end-to-end report.

Run benchmarks against a dedicated database containing only the declared evaluation documents.
The API searches the configured tenant's whole collection, so unrelated or earlier evaluation
documents in that tenant can change the retrieved context, generation latency, and answers. Stable
document IDs replace matching documents but do not remove other records.

## Smaller-model latency experiment

On 4 November 2026, the same v2 dataset was run through the merged authenticated API with
`qwen3:1.7b`, an isolated tenant, `nomic-embed-text`, `top_k=5`, and the unchanged `0.60`
similarity threshold.

| Metric | `qwen3:4b` baseline | `qwen3:1.7b` candidate |
|---|---:|---:|
| Retrieval hit rate | 100% | 100% |
| Answer correctness | 100% | 83.33% |
| Citation accuracy | 100% | 100% |
| Abstention accuracy | 100% | 100% |
| Mean latency | 63.839 s | 20.457 s |
| Maximum latency | 121.237 s | 32.119 s |

The candidate reduced mean latency by 67.96%, but it omitted the explicit “must not investigate”
instruction from the multi-page security-incident answer. Because the experiment traded away a
declared expected fact, `qwen3:4b` remains the default. This is a rejected optimization with
measured evidence, not a performance improvement claim.

The reviewed candidate artifact is
[`baselines/ollama-qwen3-1.7b-cpu-candidate.json`](baselines/ollama-qwen3-1.7b-cpu-candidate.json).

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
