from pathlib import Path

from app.pdf import extract_pdf_pages
from evals.v1.evaluate import failed_case, load_dataset, score_case, summarize


def test_score_case_checks_retrieval_answer_and_verifiable_citation():
    case = {
        "id": "policy",
        "question": "What is the policy?",
        "document_id": "handbook",
        "expected_page": 2,
        "expected_facts": ["10 days"],
        "expected_abstain": False,
    }
    response = {
        "answer": "Submit it within 10 days [Source 42, Page 2].",
        "abstained": False,
        "sources": [
            {
                "chunk_id": 42,
                "document_id": "handbook",
                "page_number": 2,
            }
        ],
    }

    result = score_case(case, response, 1.25)

    assert result["retrieval_hit"] is True
    assert result["answer_correct"] is True
    assert result["citation_correct"] is True
    assert result["abstention_correct"] is True


def test_score_case_rejects_citation_not_present_in_structured_sources():
    case = {
        "id": "policy",
        "question": "What is the policy?",
        "document_id": "handbook",
        "expected_page": 2,
        "expected_facts": ["10 days"],
        "expected_abstain": False,
    }
    response = {
        "answer": "Submit it within 10 days [Source 99, Page 2].",
        "abstained": False,
        "sources": [{"chunk_id": 42, "document_id": "handbook", "page_number": 2}],
    }

    assert score_case(case, response, 1.0)["citation_correct"] is False


def test_abstention_requires_no_returned_sources():
    case = {
        "id": "unknown",
        "question": "Unknown?",
        "document_id": "handbook",
        "expected_page": None,
        "expected_facts": [],
        "expected_abstain": True,
    }

    correct = score_case(
        case,
        {"answer": "Not enough information", "abstained": True, "sources": []},
        0.2,
    )
    incorrect = score_case(
        case,
        {
            "answer": "Not enough information",
            "abstained": True,
            "sources": [{"chunk_id": 1}],
        },
        0.2,
    )

    assert correct["abstention_correct"] is True
    assert incorrect["abstention_correct"] is False


def test_summarize_excludes_not_applicable_values():
    results = [
        {
            "retrieval_hit": True,
            "answer_correct": False,
            "citation_correct": True,
            "abstention_correct": True,
            "latency_seconds": 2.0,
        },
        {
            "retrieval_hit": None,
            "answer_correct": None,
            "citation_correct": None,
            "abstention_correct": True,
            "latency_seconds": 1.0,
        },
    ]

    assert summarize(results) == {
        "case_count": 2,
        "retrieval_k": 5,
        "retrieval_hit_rate": 1.0,
        "answer_correctness_rate": 0.0,
        "citation_accuracy": 1.0,
        "abstention_accuracy": 1.0,
        "mean_latency_seconds": 1.5,
        "max_latency_seconds": 2.0,
    }


def test_failed_answerable_case_counts_as_failed_metrics():
    case = {
        "id": "policy",
        "question": "What is the policy?",
        "expected_page": 2,
        "expected_abstain": False,
    }

    result = failed_case(case, 3.0, "502 Bad Gateway")

    assert result["retrieval_hit"] is False
    assert result["answer_correct"] is False
    assert result["citation_correct"] is False
    assert result["abstention_correct"] is False
    assert result["error"] == "502 Bad Gateway"


def test_versioned_dataset_matches_committed_pdf():
    dataset_path = Path("evals/v1/dataset.json")
    dataset = load_dataset(dataset_path)
    pdf_path = dataset_path.parent / dataset["document"]["pdf_path"]
    pages = extract_pdf_pages(pdf_path.read_bytes())

    assert dataset["version"] == "1.0.0"
    assert len(dataset["cases"]) == 6
    assert [page.page_number for page in pages] == [1, 2, 3, 4]
    assert "Tuesday and Thursday" in pages[0].text
    assert "10 calendar days" in pages[1].text
    assert "within one hour" in pages[2].text
    assert "15 days" in pages[3].text
