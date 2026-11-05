import json
from pathlib import Path

import pytest

from evals.v1.evaluate import load_dataset, score_case

ROOT = Path(__file__).parents[1]


def test_v1_dataset_remains_loadable():
    dataset = load_dataset(ROOT / "evals" / "v1" / "dataset.json")

    assert dataset["version"] == "1.0.0"
    assert len(dataset["cases"]) == 6


def test_v2_dataset_has_multiple_documents_and_multi_page_case():
    dataset = load_dataset(ROOT / "evals" / "v2" / "dataset.json")

    assert dataset["version"] == "2.0.0"
    assert len(dataset["documents"]) == 2
    multi_page = next(
        case for case in dataset["cases"] if case["id"] == "security-incident-multi-page"
    )
    assert multi_page["expected_pages"] == [3, 4]
    assert multi_page["critical_facts"] == ["not investigate"]


def test_dataset_rejects_critical_fact_without_matching_expected_fact(tmp_path):
    dataset = {
        "version": "test",
        "evaluation": {"top_k": 1},
        "document": {"document_id": "handbook", "title": "Handbook", "pages": ["Text"]},
        "cases": [
            {
                "id": "case",
                "question": "What is required?",
                "expected_page": 1,
                "expected_facts": ["report"],
                "critical_facts": ["do not investigate"],
                "expected_abstain": False,
            }
        ],
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="critical_facts must be a subset"):
        load_dataset(path)


def test_score_case_requires_all_expected_pages_and_citations():
    case = {
        "id": "security-incident-multi-page",
        "question": "What should an employee do after a suspected incident?",
        "document_id": "handbook",
        "expected_pages": [3, 4],
        "expected_facts": ["one hour", "must not investigate"],
        "expected_abstain": False,
    }
    response = {
        "answer": "Report within one hour. Employees must not investigate. "
        "[Source 11, Page 3] [Source 12, Page 4]",
        "abstained": False,
        "sources": [
            {"chunk_id": 11, "document_id": "handbook", "page_number": 3},
            {"chunk_id": 12, "document_id": "handbook", "page_number": 4},
        ],
    }

    result = score_case(case, response, 1.25)

    assert result["retrieval_hit"] is True
    assert result["answer_correct"] is True
    assert result["citation_correct"] is True


def test_score_case_reports_atomic_fact_and_evidence_coverage():
    case = {
        "id": "security-incident",
        "question": "What should I do?",
        "document_id": "handbook",
        "expected_pages": [3, 4],
        "expected_facts": ["one hour", "not investigate"],
        "critical_facts": ["not investigate"],
        "expected_abstain": False,
    }
    response = {
        "answer": "Report it within one hour. [Source 11, Page 3] [Source 12, Page 4]",
        "abstained": False,
        "sources": [
            {
                "chunk_id": 11,
                "document_id": "handbook",
                "page_number": 3,
                "content": "Report a suspected incident within one hour.",
            },
            {
                "chunk_id": 12,
                "document_id": "handbook",
                "page_number": 4,
                "content": "Employees must not investigate it themselves.",
            },
        ],
    }

    result = score_case(case, response, 2.0)

    assert result["fact_recall"] == 0.5
    assert result["critical_fact_recall"] == 0.0
    assert result["evidence_coverage"] == 1.0
    assert result["context_chunk_count"] == 2
    assert result["fact_results"] == [
        {"fact": "one hour", "present_in_answer": True, "critical": False},
        {"fact": "not investigate", "present_in_answer": False, "critical": True},
    ]


def test_score_case_fails_when_one_expected_page_is_missing():
    case = {
        "id": "security-incident-multi-page",
        "question": "What should an employee do after a suspected incident?",
        "document_id": "handbook",
        "expected_pages": [3, 4],
        "expected_facts": ["one hour", "must not investigate"],
        "expected_abstain": False,
    }
    response = {
        "answer": "Report within one hour. Employees must not investigate. [Source 11, Page 3]",
        "abstained": False,
        "sources": [
            {"chunk_id": 11, "document_id": "handbook", "page_number": 3},
            {"chunk_id": 12, "document_id": "handbook", "page_number": 4},
        ],
    }

    result = score_case(case, response, 1.25)

    assert result["retrieval_hit"] is True
    assert result["answer_correct"] is True
    assert result["citation_correct"] is False


def test_score_case_marks_near_miss_abstention_correct_only_without_sources():
    case = {
        "id": "unsupported-parking-price-near-miss",
        "question": "What is the monthly price of a parking permit?",
        "document_id": None,
        "expected_pages": [],
        "expected_facts": [],
        "expected_abstain": True,
    }

    abstained = score_case(
        case,
        {"answer": "I do not have enough information.", "abstained": True, "sources": []},
        0.2,
    )
    not_abstained = score_case(
        case,
        {
            "answer": "The commuter benefit is 75 dollars.",
            "abstained": False,
            "sources": [{"document_id": "benefits"}],
        },
        0.2,
    )

    assert abstained["abstention_correct"] is True
    assert not_abstained["abstention_correct"] is False
