import json
from pathlib import Path

from evals.failures.controlled import (
    ABSTENTION,
    CASES,
    generate,
    is_abstention,
)
from evals.v1.evaluate import load_dataset


def test_natural_failure_dataset_is_valid():
    dataset = load_dataset(Path("evals/failures/natural-dataset.json"))

    assert dataset["evaluation"]["top_k"] == 5
    assert len(dataset["documents"]) == 2
    assert [case["id"] for case in dataset["cases"]] == [
        "N1-manager-approved-remote-exception",
        "N2-expense-deadline-and-approval",
        "N3-security-reporting-and-handling",
        "N4-commuter-parking-distinction",
        "N5-unsupported-parking-price",
    ]


def test_controlled_fixtures_keep_synthetic_conflict_out_of_baselines():
    security_case = next(case for case in CASES if case["id"].startswith("C4-"))

    assert all(
        item["document_id"] != "synthetic-adversarial-fixture"
        for item in security_case["baseline"]
    )
    assert any(
        item["document_id"] == "synthetic-adversarial-fixture"
        for item in security_case["perturbed"]
    )


def test_semantic_abstention_phrases_and_empty_evidence_short_circuit():
    assert is_abstention(ABSTENTION)
    assert is_abstention("The evidence does not contain information about parking prices.")
    assert not is_abstention("The parking price is $75.")

    result = generate(None, "unused", "Unsupported question?", [])
    assert result == {
        "answer": ABSTENTION,
        "abstained": True,
        "latency_seconds": 0.0,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


def test_committed_failure_evidence_matches_current_scoring_rules():
    results_dir = Path("evals/failures/results")
    controlled = json.loads(
        (results_dir / "controlled-2026-11-20.json").read_text(encoding="utf-8")
    )
    cases = {case["id"]: case for case in controlled["results"]}

    home_office = cases["C2-home-office-with-commuter-distractor"]
    assert home_office["expected_facts"] == ["100"]
    assert home_office["variants"]["baseline"]["expected_facts_present"] is True

    parking = cases["C5-parking-with-commuter-near-miss"]
    assert parking["variants"]["perturbed"]["abstained"] is True

    retry = json.loads(
        (results_dir / "n1-retry-2026-11-20.json").read_text(encoding="utf-8")
    )
    assert retry["status_code"] == 200
    assert retry["interpretation"]["expected_page_rank"] == 1
    assert retry["interpretation"]["retrieval_failure"] is False
