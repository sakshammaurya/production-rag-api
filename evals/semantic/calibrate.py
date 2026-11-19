"""Prepare blinded human review files and compare them with a semantic judge report."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from evals.semantic.evaluate import RUBRIC_VERSION, parse_judge_output
from evals.v1.evaluate import load_dataset

ANNOTATION_SCHEMA_VERSION = "1.0.0"
DIMENSIONS = ("faithfulness", "answer_relevance", "context_relevance")


def _raise_validation_error(message: str) -> NoReturn:
    """Expose one exception type for invalid JSON report and annotation content."""
    raise ValueError(message)


def _indexed_results(rag_report: dict) -> dict[str, dict]:
    results = rag_report.get("results")
    if not isinstance(results, list):
        _raise_validation_error("RAG report results must be a list")
    indexed = {}
    for result in results:
        case_id = result.get("id") if isinstance(result, dict) else None
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every RAG result must define a non-empty case ID")
        if case_id in indexed:
            raise ValueError(f"RAG report contains duplicate case ID: {case_id}")
        indexed[case_id] = result
    return indexed


def build_review_packet(dataset: dict, rag_report: dict, source_rag_report: str) -> dict:
    results = _indexed_results(rag_report)
    dataset_ids = [case["id"] for case in dataset["cases"]]
    unknown_ids = sorted(set(results) - set(dataset_ids))
    missing_ids = sorted(set(dataset_ids) - set(results))
    if unknown_ids or missing_ids:
        raise ValueError(
            f"RAG report case mismatch; unknown={unknown_ids}, missing={missing_ids}"
        )

    cases = []
    for case in dataset["cases"]:
        response = results[case["id"]].get("response")
        if not isinstance(response, dict):
            _raise_validation_error(
                f"Case {case['id']} has no structured response to review"
            )
        sources = response.get("sources", [])
        if not isinstance(sources, list):
            _raise_validation_error(
                f"Case {case['id']} response sources must be a list"
            )
        cases.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answer": response.get("answer", ""),
                "abstained": bool(response.get("abstained", False)),
                "evidence": [
                    {
                        "document_id": source.get("document_id"),
                        "page_number": source.get("page_number"),
                        "content": source.get("content", ""),
                    }
                    for source in sources
                ],
            }
        )
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "dataset_version": dataset["version"],
        "source_rag_report": source_rag_report,
        "blinding": {
            "judge_scores_included": False,
            "deterministic_labels_included": False,
            "expected_facts_included": False,
        },
        "cases": cases,
    }


def build_annotation_template(review_packet: dict, source_review_packet: str) -> dict:
    cases = []
    for case in review_packet["cases"]:
        cases.append(
            {
                "id": case["id"],
                "abstained": case["abstained"],
                "evidence_present": bool(case["evidence"]),
                "review_status": "unreviewed",
                "scores": {
                    "faithfulness": {
                        "applicable": not case["abstained"],
                        "value": None,
                    },
                    "answer_relevance": {"applicable": True, "value": None},
                    "context_relevance": {
                        "applicable": bool(case["evidence"]),
                        "value": None,
                    },
                },
                "unsupported_claims_present": None,
                "rationale": None,
            }
        )
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "rubric_version": review_packet["rubric_version"],
        "dataset_version": review_packet["dataset_version"],
        "source_review_packet": source_review_packet,
        "pass_threshold": 3,
        "reviewer": {"reviewer_id": "", "reviewed_at": None},
        "cases": cases,
    }


def validate_human_labels(payload: dict) -> dict:
    if payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported annotation schema: {payload.get('schema_version')}")
    if payload.get("rubric_version") != RUBRIC_VERSION:
        raise ValueError(f"Unsupported rubric version: {payload.get('rubric_version')}")
    threshold = payload.get("pass_threshold")
    if not isinstance(threshold, int) or isinstance(threshold, bool) or not 1 <= threshold <= 4:
        raise ValueError("pass_threshold must be an integer from 1 to 4")
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, dict):
        _raise_validation_error("reviewer must be an object")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        _raise_validation_error("cases must be a list")

    case_ids = []
    reviewed_count = 0
    for case in cases:
        if not isinstance(case, dict):
            _raise_validation_error("Every annotation case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every annotation case must define a non-empty ID")
        case_ids.append(case_id)
        status = case.get("review_status")
        if status not in {"unreviewed", "reviewed"}:
            raise ValueError(f"Case {case_id} has invalid review_status")
        abstained = case.get("abstained")
        evidence_present = case.get("evidence_present")
        if not isinstance(abstained, bool) or not isinstance(evidence_present, bool):
            _raise_validation_error(
                f"Case {case_id} must preserve boolean abstained and evidence_present fields"
            )
        scores = case.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
            raise ValueError(f"Case {case_id} must define all score dimensions")
        for dimension in DIMENSIONS:
            score = scores[dimension]
            if not isinstance(score, dict) or set(score) != {"applicable", "value"}:
                raise ValueError(f"Case {case_id} has invalid {dimension} annotation")
            applicable = score["applicable"]
            value = score["value"]
            if not isinstance(applicable, bool):
                _raise_validation_error(
                    f"Case {case_id} {dimension}.applicable must be boolean"
                )
            if status == "unreviewed" and value is not None:
                raise ValueError(f"Unreviewed case {case_id} cannot contain scores")
            if status == "reviewed" and applicable and (
                not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 4
            ):
                raise ValueError(f"Case {case_id} {dimension} score must be from 1 to 4")
            if not applicable and value is not None:
                raise ValueError(f"Case {case_id} non-applicable {dimension} must be null")
        if scores["answer_relevance"]["applicable"] is not True:
            raise ValueError(f"Case {case_id} answer_relevance must be applicable")
        if scores["faithfulness"]["applicable"] != (not abstained):
            raise ValueError(f"Case {case_id} faithfulness applicability is inconsistent")
        if scores["context_relevance"]["applicable"] != evidence_present:
            raise ValueError(f"Case {case_id} context relevance applicability is inconsistent")
        unsupported = case.get("unsupported_claims_present")
        rationale = case.get("rationale")
        if status == "unreviewed":
            if unsupported is not None or rationale is not None:
                raise ValueError(f"Unreviewed case {case_id} must not contain labels")
        else:
            reviewed_count += 1
            if not isinstance(unsupported, bool):
                raise ValueError(
                    f"Reviewed case {case_id} unsupported_claims_present must be boolean"
                )
            if rationale is not None and (
                not isinstance(rationale, str) or not rationale.strip()
            ):
                raise ValueError(f"Case {case_id} rationale must be null or non-empty text")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Human annotations contain duplicate case IDs")
    if reviewed_count and (
        not isinstance(reviewer.get("reviewer_id"), str)
        or not reviewer["reviewer_id"].strip()
        or not isinstance(reviewer.get("reviewed_at"), str)
        or not reviewer["reviewed_at"].strip()
    ):
        raise ValueError("Reviewed annotations require reviewer_id and reviewed_at")
    return payload


def _human_pass(case: dict, threshold: int) -> bool:
    applicable_scores = [
        case["scores"][dimension]["value"]
        for dimension in DIMENSIONS
        if case["scores"][dimension]["applicable"]
    ]
    return (
        all(score >= threshold for score in applicable_scores)
        and not case["unsupported_claims_present"]
    )


def _judge_results(judge_report: dict) -> dict[str, dict]:
    results = judge_report.get("results")
    if not isinstance(results, list):
        _raise_validation_error("Judge report results must be a list")
    indexed = {}
    for result in results:
        case_id = result.get("id") if isinstance(result, dict) else None
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every judge result must define a non-empty case ID")
        if case_id in indexed:
            raise ValueError("Judge report contains duplicate case IDs")
        judgment = result.get("judgment")
        semantic_pass = result.get("semantic_pass")
        if judgment is not None:
            judgment = parse_judge_output(json.dumps(judgment))
            if not isinstance(semantic_pass, bool):
                raise ValueError(
                    f"Judge result {case_id} must define boolean semantic_pass"
                )
        elif semantic_pass is not None:
            raise ValueError(
                f"Failed judge result {case_id} must have null semantic_pass"
            )
        indexed[case_id] = {**result, "judgment": judgment}
    return indexed


def _dimension_comparison(
    reviewed: dict[str, dict], judge_results: dict[str, dict], dimension: str
) -> dict:
    judge_field = f"{dimension}_score"
    differences = []
    excluded_case_ids = []
    for case_id, human_case in reviewed.items():
        human_score = human_case["scores"][dimension]
        judge = judge_results.get(case_id, {}).get("judgment")
        judge_score = judge.get(judge_field) if isinstance(judge, dict) else None
        if not human_score["applicable"] or judge_score is None:
            excluded_case_ids.append(case_id)
            continue
        differences.append(abs(human_score["value"] - judge_score))
    count = len(differences)
    return {
        "comparable_case_count": count,
        "excluded_case_ids": excluded_case_ids,
        "exact_agreement": (
            round(sum(difference == 0 for difference in differences) / count, 4)
            if count
            else None
        ),
        "within_one_agreement": (
            round(sum(difference <= 1 for difference in differences) / count, 4)
            if count
            else None
        ),
        "mean_absolute_error": (
            round(sum(differences) / count, 4) if count else None
        ),
    }


def compare_calibration(human_labels: dict, judge_report: dict) -> dict:
    validate_human_labels(human_labels)
    if judge_report.get("rubric_version") != human_labels["rubric_version"]:
        raise ValueError("Human labels and judge report use different rubric versions")
    if judge_report.get("dataset_version") != human_labels.get("dataset_version"):
        raise ValueError("Human labels and judge report use different dataset versions")
    judge_threshold = judge_report.get("judge", {}).get("pass_threshold")
    if judge_threshold != human_labels["pass_threshold"]:
        raise ValueError("Human labels and judge report use different pass thresholds")

    human_cases = {case["id"]: case for case in human_labels["cases"]}
    reviewed = {
        case_id: case
        for case_id, case in human_cases.items()
        if case["review_status"] == "reviewed"
    }
    judge_results = _judge_results(judge_report)
    human_only = sorted(set(human_cases) - set(judge_results))
    judge_only = sorted(set(judge_results) - set(human_cases))

    pass_matches = []
    unsupported_matches = []
    disagreements = []
    comparable_case_ids = []
    for case_id, human_case in reviewed.items():
        judge_result = judge_results.get(case_id)
        judgment = judge_result.get("judgment") if judge_result else None
        if not isinstance(judgment, dict):
            continue
        comparable_case_ids.append(case_id)
        human_pass = _human_pass(human_case, human_labels["pass_threshold"])
        judge_pass = judge_result.get("semantic_pass")
        pass_match = (
            human_pass == judge_pass if isinstance(judge_pass, bool) else None
        )
        judge_unsupported = bool(judgment.get("unsupported_claims"))
        unsupported_match = (
            human_case["unsupported_claims_present"] == judge_unsupported
        )
        if pass_match is not None:
            pass_matches.append(pass_match)
        unsupported_matches.append(unsupported_match)

        differing_dimensions = []
        for dimension in DIMENSIONS:
            human_score = human_case["scores"][dimension]
            judge_score = judgment.get(f"{dimension}_score")
            if human_score["applicable"] != (judge_score is not None) or (
                human_score["applicable"] and human_score["value"] != judge_score
            ):
                differing_dimensions.append(dimension)
        if differing_dimensions or pass_match is False or not unsupported_match:
            disagreements.append(
                {
                    "id": case_id,
                    "differing_dimensions": differing_dimensions,
                    "human_scores": {
                        dimension: human_case["scores"][dimension]["value"]
                        for dimension in DIMENSIONS
                    },
                    "judge_scores": {
                        dimension: judgment.get(f"{dimension}_score")
                        for dimension in DIMENSIONS
                    },
                    "human_unsupported_claims_present": human_case[
                        "unsupported_claims_present"
                    ],
                    "judge_unsupported_claims_present": judge_unsupported,
                    "human_pass": human_pass,
                    "judge_pass": judge_pass,
                }
            )

    return {
        "annotation_schema_version": human_labels["schema_version"],
        "rubric_version": human_labels["rubric_version"],
        "pass_threshold": human_labels["pass_threshold"],
        "reviewed_case_count": len(reviewed),
        "comparable_case_count": len(comparable_case_ids),
        "unreviewed_case_count": len(human_cases) - len(reviewed),
        "human_only_case_ids": human_only,
        "judge_only_case_ids": judge_only,
        "dimension_agreement": {
            dimension: _dimension_comparison(reviewed, judge_results, dimension)
            for dimension in DIMENSIONS
        },
        "pass_fail_agreement": (
            round(sum(pass_matches) / len(pass_matches), 4) if pass_matches else None
        ),
        "unsupported_claim_agreement": (
            round(sum(unsupported_matches) / len(unsupported_matches), 4)
            if unsupported_matches
            else None
        ),
        "disagreements": disagreements,
    }


def _write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _prepare(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset)
    rag_report = json.loads(args.rag_report.read_text(encoding="utf-8"))
    packet = build_review_packet(dataset, rag_report, args.rag_report.as_posix())
    template = build_annotation_template(packet, args.packet_output.as_posix())
    _write_new(args.packet_output, packet)
    _write_new(args.labels_output, template)
    print(f"Review packet written to {args.packet_output}")
    print(f"Unreviewed label template written to {args.labels_output}")


def _compare(args: argparse.Namespace) -> None:
    human_labels = json.loads(args.human_labels.read_text(encoding="utf-8"))
    judge_report = json.loads(args.judge_report.read_text(encoding="utf-8"))
    comparison = {
        "calibration": "human-versus-semantic-judge",
        "compared_at": datetime.now(UTC).isoformat(),
        "human_labels": args.human_labels.as_posix(),
        "judge_report": args.judge_report.as_posix(),
        **compare_calibration(human_labels, judge_report),
    }
    _write_new(args.output, comparison)
    print(json.dumps(comparison, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Create a blinded packet and label template")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--rag-report", type=Path, required=True)
    prepare.add_argument("--packet-output", type=Path, required=True)
    prepare.add_argument("--labels-output", type=Path, required=True)
    prepare.set_defaults(handler=_prepare)

    compare = subparsers.add_parser("compare", help="Compare human labels with judge output")
    compare.add_argument("--human-labels", type=Path, required=True)
    compare.add_argument("--judge-report", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(handler=_compare)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
