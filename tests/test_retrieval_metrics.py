import pytest

from evals.retrieval import score_page_ranking, summarize_page_ranking


def source(document_id: str, page_number: int | None, chunk_id: int) -> dict:
    return {
        "document_id": document_id,
        "page_number": page_number,
        "chunk_id": chunk_id,
    }


def metrics(expected_pages: list[int], sources: list[dict], k: int = 3) -> dict:
    return score_page_ranking("handbook", expected_pages, sources, [k])["by_k"][str(k)]


def test_first_result_relevant():
    result = metrics(
        [2],
        [source("handbook", 2, 1), source("handbook", 1, 2)],
    )

    assert result == {
        "precision": 0.3333,
        "recall": 1.0,
        "reciprocal_rank": 1.0,
        "ndcg": 1.0,
    }


def test_relevant_result_at_rank_two():
    result = metrics(
        [2],
        [source("handbook", 1, 1), source("handbook", 2, 2)],
    )

    assert result == {
        "precision": 0.3333,
        "recall": 1.0,
        "reciprocal_rank": 0.5,
        "ndcg": 0.6309,
    }


@pytest.mark.parametrize("sources", [[], [source("other", 2, 1)]])
def test_no_relevant_result_scores_zero(sources):
    assert metrics([2], sources) == {
        "precision": 0.0,
        "recall": 0.0,
        "reciprocal_rank": 0.0,
        "ndcg": 0.0,
    }


def test_case_without_expected_pages_is_excluded():
    result = score_page_ranking("handbook", [], [source("handbook", 1, 1)], [1, 3])

    assert result == {"eligible": False, "by_k": {}}


def test_duplicate_chunks_from_same_page_count_once():
    result = metrics(
        [2],
        [
            source("handbook", 2, 1),
            source("handbook", 2, 2),
            source("handbook", 1, 3),
        ],
        k=2,
    )

    assert result["precision"] == 0.5
    assert result["recall"] == 1.0


def test_fewer_results_than_k_uses_k_as_precision_denominator():
    result = metrics([2], [source("handbook", 2, 1)], k=5)

    assert result["precision"] == 0.2
    assert result["recall"] == 1.0


def test_multiple_relevant_pages_use_binary_page_relevance():
    result = metrics(
        [2, 4],
        [
            source("handbook", 2, 1),
            source("handbook", 1, 2),
            source("handbook", 4, 3),
        ],
    )

    assert result == {
        "precision": 0.6667,
        "recall": 1.0,
        "reciprocal_rank": 1.0,
        "ndcg": 0.9197,
    }


def test_configurable_k_values_are_reported_separately():
    result = score_page_ranking(
        "handbook",
        [2],
        [source("handbook", 1, 1), source("handbook", 2, 2)],
        [1, 3],
    )

    assert result["by_k"]["1"]["recall"] == 0.0
    assert result["by_k"]["3"]["recall"] == 1.0


def test_ndcg_is_normalized_against_ideal_binary_ranking():
    result = metrics(
        [2, 4],
        [
            source("handbook", 1, 1),
            source("handbook", 2, 2),
            source("handbook", 4, 3),
        ],
    )

    assert result["ndcg"] == 0.6934


def test_summary_macro_averages_eligible_cases_and_counts_exclusions():
    eligible = score_page_ranking(
        "handbook", [2], [source("handbook", 2, 1)], [1]
    )
    excluded = score_page_ranking("handbook", [], [], [1])

    summary = summarize_page_ranking(
        [
            {"retrieval_ranking": eligible},
            {"retrieval_ranking": excluded},
        ],
        [1],
    )

    assert summary == {
        "relevance_granularity": "document_page",
        "eligible_case_count": 1,
        "excluded_case_count": 1,
        "by_k": {
            "1": {
                "precision": 1.0,
                "recall": 1.0,
                "mrr": 1.0,
                "ndcg": 1.0,
            }
        },
    }
