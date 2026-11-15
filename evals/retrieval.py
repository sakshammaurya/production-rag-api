"""Deterministic page-level retrieval-ranking metrics."""

from math import log2


def _ranked_page_items(sources: list[dict]) -> list[tuple[str, int | None]]:
    """Preserve source order while collapsing duplicate document-page results."""
    ranked_items = []
    seen_pages: set[tuple[str, int]] = set()
    for index, source in enumerate(sources):
        document_id = source.get("document_id", "")
        page_number = source.get("page_number")
        if page_number is None:
            # Page-less chunks are distinct irrelevant items in a page-level evaluation.
            ranked_items.append((f"{document_id}#chunk-{source.get('chunk_id', index)}", None))
            continue
        page = (document_id, page_number)
        if page not in seen_pages:
            seen_pages.add(page)
            ranked_items.append(page)
    return ranked_items


def score_page_ranking(
    expected_document_id: str | None,
    expected_pages: list[int],
    sources: list[dict],
    k_values: list[int],
) -> dict:
    """Score ordered API sources against binary document-page relevance labels."""
    relevant_pages = {
        (expected_document_id, page_number) for page_number in expected_pages
    }
    if not relevant_pages:
        return {"eligible": False, "by_k": {}}

    ranked_items = _ranked_page_items(sources)
    by_k = {}
    for k in k_values:
        top_items = ranked_items[:k]
        relevance = [item in relevant_pages for item in top_items]
        relevant_retrieved = sum(relevance)
        first_relevant_rank = next(
            (rank for rank, is_relevant in enumerate(relevance, start=1) if is_relevant),
            None,
        )
        dcg = sum(
            1 / log2(rank + 1)
            for rank, is_relevant in enumerate(relevance, start=1)
            if is_relevant
        )
        ideal_relevant_count = min(len(relevant_pages), k)
        ideal_dcg = sum(
            1 / log2(rank + 1) for rank in range(1, ideal_relevant_count + 1)
        )
        by_k[str(k)] = {
            "precision": round(relevant_retrieved / k, 4),
            "recall": round(relevant_retrieved / len(relevant_pages), 4),
            "reciprocal_rank": (
                round(1 / first_relevant_rank, 4) if first_relevant_rank else 0.0
            ),
            "ndcg": round(dcg / ideal_dcg, 4),
        }

    return {"eligible": True, "by_k": by_k}


def summarize_page_ranking(results: list[dict], k_values: list[int]) -> dict:
    """Macro-average ranking metrics over cases with at least one relevant page."""
    eligible = [result["retrieval_ranking"] for result in results if result["retrieval_ranking"]["eligible"]]
    excluded_count = len(results) - len(eligible)
    by_k = {}
    for k in k_values:
        key = str(k)
        by_k[key] = {}
        for output_metric, case_metric in (
            ("precision", "precision"),
            ("recall", "recall"),
            ("mrr", "reciprocal_rank"),
            ("ndcg", "ndcg"),
        ):
            by_k[key][output_metric] = (
                round(
                    sum(result["by_k"][key][case_metric] for result in eligible)
                    / len(eligible),
                    4,
                )
                if eligible
                else None
            )
    return {
        "relevance_granularity": "document_page",
        "eligible_case_count": len(eligible),
        "excluded_case_count": excluded_count,
        "by_k": by_k,
    }
