import json
from copy import deepcopy
from pathlib import Path

import pytest

from evals.semantic.calibrate import (
    ANNOTATION_SCHEMA_VERSION,
    build_annotation_template,
    build_review_packet,
    compare_calibration,
    validate_human_labels,
)
from evals.semantic.evaluate import RUBRIC_VERSION

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def dataset() -> dict:
    return {
        "version": "2.0.0",
        "cases": [
            {"id": "answerable", "question": "What is the policy?"},
            {"id": "unsupported", "question": "What is not documented?"},
        ],
    }


def rag_report() -> dict:
    return {
        "results": [
            {
                "id": "answerable",
                "answer_correct": True,
                "response": {
                    "answer": "Submit within ten days.",
                    "abstained": False,
                    "sources": [
                        {
                            "document_id": "handbook",
                            "page_number": 2,
                            "content": "Submit within ten days.",
                        }
                    ],
                },
            },
            {
                "id": "unsupported",
                "abstention_correct": True,
                "response": {
                    "answer": "I do not have enough information.",
                    "abstained": True,
                    "sources": [],
                },
            },
        ]
    }


def template() -> dict:
    packet = build_review_packet(dataset(), rag_report(), "rag-report.json")
    return build_annotation_template(packet, "review-packet.json")


def review_case(
    case: dict,
    *,
    faithfulness: int | None,
    answer_relevance: int,
    context_relevance: int | None,
    unsupported: bool = False,
) -> None:
    case["review_status"] = "reviewed"
    case["scores"]["faithfulness"]["value"] = faithfulness
    case["scores"]["answer_relevance"]["value"] = answer_relevance
    case["scores"]["context_relevance"]["value"] = context_relevance
    case["unsupported_claims_present"] = unsupported
    case["rationale"] = "Reviewed against the supplied evidence."


def completed_labels() -> dict:
    labels = template()
    labels["reviewer"] = {
        "reviewer_id": "reviewer-1",
        "reviewed_at": "2026-11-16T10:00:00+05:30",
    }
    review_case(
        labels["cases"][0],
        faithfulness=4,
        answer_relevance=3,
        context_relevance=2,
    )
    review_case(
        labels["cases"][1],
        faithfulness=None,
        answer_relevance=4,
        context_relevance=None,
    )
    return labels


def judge_report() -> dict:
    return {
        "rubric_version": RUBRIC_VERSION,
        "dataset_version": "2.0.0",
        "judge": {"pass_threshold": 3},
        "results": [
            {
                "id": "answerable",
                "semantic_pass": True,
                "judgment": {
                    "faithfulness_score": 4,
                    "answer_relevance_score": 4,
                    "context_relevance_score": 3,
                    "unsupported_claims": [],
                    "reasons": {
                        "faithfulness": "Supported by the evidence.",
                        "answer_relevance": "Directly answers the question.",
                        "context_relevance": "The evidence is relevant.",
                    },
                },
            },
            {
                "id": "unsupported",
                "semantic_pass": False,
                "judgment": {
                    "faithfulness_score": None,
                    "answer_relevance_score": 2,
                    "context_relevance_score": None,
                    "unsupported_claims": ["Invented provider"],
                    "reasons": {
                        "faithfulness": "Not applicable to the abstention.",
                        "answer_relevance": "The response is not appropriate.",
                        "context_relevance": "No evidence was returned.",
                    },
                },
            },
        ],
    }


def test_prepare_creates_blinded_packet_and_unreviewed_template():
    packet = build_review_packet(dataset(), rag_report(), "rag-report.json")
    labels = build_annotation_template(packet, "review-packet.json")

    assert packet["blinding"] == {
        "judge_scores_included": False,
        "deterministic_labels_included": False,
        "expected_facts_included": False,
    }
    assert "answer_correct" not in packet["cases"][0]
    assert labels["schema_version"] == ANNOTATION_SCHEMA_VERSION
    assert all(case["review_status"] == "unreviewed" for case in labels["cases"])
    assert labels["cases"][1]["scores"]["faithfulness"]["applicable"] is False


def test_prepare_rejects_unknown_missing_and_duplicate_cases():
    missing = rag_report()
    missing["results"].pop()
    with pytest.raises(ValueError, match="case mismatch"):
        build_review_packet(dataset(), missing, "report.json")

    duplicate = rag_report()
    duplicate["results"].append(duplicate["results"][0])
    with pytest.raises(ValueError, match="duplicate case ID"):
        build_review_packet(dataset(), duplicate, "report.json")


def test_complete_and_incomplete_annotations_are_valid():
    assert validate_human_labels(template())["cases"][0]["review_status"] == "unreviewed"
    assert validate_human_labels(completed_labels())["cases"][0]["review_status"] == "reviewed"


def test_reviewed_annotations_require_reviewer_identity():
    labels = completed_labels()
    labels["reviewer"] = {"reviewer_id": "", "reviewed_at": None}

    with pytest.raises(ValueError, match="require reviewer_id"):
        validate_human_labels(labels)


@pytest.mark.parametrize("invalid_score", [0, 5, "4", True])
def test_invalid_score_ranges_and_types_are_rejected(invalid_score):
    labels = completed_labels()
    labels["cases"][0]["scores"]["faithfulness"]["value"] = invalid_score

    with pytest.raises(ValueError, match="score must be from 1 to 4"):
        validate_human_labels(labels)


def test_missing_and_non_applicable_scores_are_handled_explicitly():
    labels = completed_labels()
    labels["cases"][1]["scores"]["faithfulness"]["value"] = 3
    with pytest.raises(ValueError, match="non-applicable faithfulness must be null"):
        validate_human_labels(labels)

    labels = completed_labels()
    labels["cases"][0]["scores"]["faithfulness"]["value"] = None
    with pytest.raises(ValueError, match="score must be from 1 to 4"):
        validate_human_labels(labels)


def test_duplicate_human_and_judge_case_ids_are_rejected():
    labels = completed_labels()
    labels["cases"].append(deepcopy(labels["cases"][0]))
    with pytest.raises(ValueError, match="duplicate case IDs"):
        validate_human_labels(labels)

    report = judge_report()
    report["results"].append(deepcopy(report["results"][0]))
    with pytest.raises(ValueError, match="duplicate case IDs"):
        compare_calibration(completed_labels(), report)


def test_calibration_calculates_agreement_error_and_disagreements():
    comparison = compare_calibration(completed_labels(), judge_report())

    assert comparison["reviewed_case_count"] == 2
    assert comparison["comparable_case_count"] == 2
    assert comparison["dimension_agreement"]["faithfulness"] == {
        "comparable_case_count": 1,
        "excluded_case_ids": ["unsupported"],
        "exact_agreement": 1.0,
        "within_one_agreement": 1.0,
        "mean_absolute_error": 0.0,
    }
    assert comparison["dimension_agreement"]["answer_relevance"] == {
        "comparable_case_count": 2,
        "excluded_case_ids": [],
        "exact_agreement": 0.0,
        "within_one_agreement": 0.5,
        "mean_absolute_error": 1.5,
    }
    assert comparison["pass_fail_agreement"] == 0.0
    assert comparison["unsupported_claim_agreement"] == 0.5
    assert [case["id"] for case in comparison["disagreements"]] == [
        "answerable",
        "unsupported",
    ]


def test_case_mismatches_are_reported_without_becoming_zero_scores():
    labels = completed_labels()
    labels["cases"].append(
        {
            "id": "human-only",
            "abstained": False,
            "evidence_present": True,
            "review_status": "unreviewed",
            "scores": {
                "faithfulness": {"applicable": True, "value": None},
                "answer_relevance": {"applicable": True, "value": None},
                "context_relevance": {"applicable": True, "value": None},
            },
            "unsupported_claims_present": None,
            "rationale": None,
        }
    )
    report = judge_report()
    report["results"].append(
        {"id": "judge-only", "semantic_pass": None, "judgment": None}
    )

    comparison = compare_calibration(labels, report)

    assert comparison["human_only_case_ids"] == ["human-only"]
    assert comparison["judge_only_case_ids"] == ["judge-only"]


def test_empty_reviewed_dataset_returns_no_fabricated_metrics():
    comparison = compare_calibration(template(), judge_report())

    assert comparison["reviewed_case_count"] == 0
    assert comparison["comparable_case_count"] == 0
    assert comparison["pass_fail_agreement"] is None
    assert comparison["unsupported_claim_agreement"] is None
    assert comparison["dimension_agreement"]["faithfulness"]["mean_absolute_error"] is None


def test_rubric_dataset_and_threshold_must_match():
    for field, value, error in (
        ("rubric_version", "2.0.0", "rubric versions"),
        ("dataset_version", "3.0.0", "dataset versions"),
    ):
        report = judge_report()
        report[field] = value
        with pytest.raises(ValueError, match=error):
            compare_calibration(completed_labels(), report)

    report = judge_report()
    report["judge"]["pass_threshold"] = 4
    with pytest.raises(ValueError, match="pass thresholds"):
        compare_calibration(completed_labels(), report)


def test_committed_v2_human_labels_are_complete_and_valid():
    labels_path = (
        REPOSITORY_ROOT
        / "evals"
        / "semantic"
        / "annotations"
        / "v2-2026-10-29-human-labels-reviewed.json"
    )
    labels = validate_human_labels(json.loads(labels_path.read_text(encoding="utf-8")))

    assert labels["reviewer"]["reviewer_id"] != ""
    assert labels["reviewer"]["reviewed_at"] is not None
    assert len(labels["cases"]) == 8
    assert all(case["review_status"] == "reviewed" for case in labels["cases"])
