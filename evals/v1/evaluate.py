"""Run versioned RAG evaluation cases through the public HTTP API."""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

ROOT = Path(__file__).parent
CITATION_PATTERN = re.compile(r"\[Source (\d+), Page (\d+)\]")


def load_dataset(path: Path) -> dict:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    required = {"version", "evaluation", "document", "cases"}
    missing = required - dataset.keys()
    if missing:
        raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")
    if not dataset["cases"]:
        raise ValueError("Dataset must contain at least one evaluation case")
    case_fields = {
        "id",
        "question",
        "expected_page",
        "expected_facts",
        "expected_abstain",
    }
    case_ids = []
    for case in dataset["cases"]:
        missing_case_fields = case_fields - case.keys()
        if missing_case_fields:
            raise ValueError(
                f"Case is missing required fields: {sorted(missing_case_fields)}"
            )
        case_ids.append(case["id"])
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case IDs must be unique")
    return dataset


def score_case(case: dict, response: dict, latency_seconds: float) -> dict:
    sources = response.get("sources", [])
    expected_page = case["expected_page"]
    expected_document = case["document_id"]
    answer = response.get("answer", "")
    normalized_answer = answer.casefold()
    expected_abstain = case["expected_abstain"]

    retrieval_hit = None
    answer_correct = None
    citation_correct = None
    if not expected_abstain:
        retrieval_hit = any(
            source["document_id"] == expected_document
            and source["page_number"] == expected_page
            for source in sources
        )
        answer_correct = not response.get("abstained", False) and all(
            fact.casefold() in normalized_answer for fact in case["expected_facts"]
        )
        returned_citations = {
            (int(source["chunk_id"]), source["page_number"])
            for source in sources
            if source["page_number"] is not None
        }
        answer_citations = {
            (int(chunk_id), int(page_number))
            for chunk_id, page_number in CITATION_PATTERN.findall(answer)
        }
        citation_correct = bool(answer_citations) and any(
            citation in returned_citations and citation[1] == expected_page
            for citation in answer_citations
        )

    abstention_correct = response.get("abstained") is expected_abstain
    if expected_abstain:
        abstention_correct = abstention_correct and not sources

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_abstain": expected_abstain,
        "expected_page": expected_page,
        "latency_seconds": round(latency_seconds, 3),
        "retrieval_hit": retrieval_hit,
        "answer_correct": answer_correct,
        "citation_correct": citation_correct,
        "abstention_correct": abstention_correct,
        "response": response,
    }


def rate(results: list[dict], field: str) -> float | None:
    values = [result[field] for result in results if result[field] is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarize(results: list[dict], top_k: int = 5) -> dict:
    latencies = [result["latency_seconds"] for result in results]
    return {
        "case_count": len(results),
        "retrieval_k": top_k,
        "retrieval_hit_rate": rate(results, "retrieval_hit"),
        "answer_correctness_rate": rate(results, "answer_correct"),
        "citation_accuracy": rate(results, "citation_correct"),
        "abstention_accuracy": rate(results, "abstention_correct"),
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 3),
        "max_latency_seconds": max(latencies),
    }


def failed_case(case: dict, latency_seconds: float, error: str) -> dict:
    expected_abstain = case["expected_abstain"]
    return {
        "id": case["id"],
        "question": case["question"],
        "expected_abstain": expected_abstain,
        "expected_page": case["expected_page"],
        "latency_seconds": round(latency_seconds, 3),
        "retrieval_hit": None if expected_abstain else False,
        "answer_correct": None if expected_abstain else False,
        "citation_correct": None if expected_abstain else False,
        "abstention_correct": False,
        "response": None,
        "error": error,
    }


def run(dataset_path: Path, base_url: str, timeout: float) -> dict:
    dataset = load_dataset(dataset_path)
    document = dataset["document"]
    top_k = dataset["evaluation"]["top_k"]
    pdf_path = dataset_path.parent / document["pdf_path"]
    cases = [case | {"document_id": document["document_id"]} for case in dataset["cases"]]

    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        with pdf_path.open("rb") as pdf_file:
            ingestion = client.post(
                "/v1/documents/pdf",
                data={"document_id": document["document_id"], "title": document["title"]},
                files={"file": (pdf_path.name, pdf_file, "application/pdf")},
            )
        ingestion.raise_for_status()

        results = []
        for case in cases:
            started = perf_counter()
            try:
                response = client.post(
                    "/v1/ask", json={"question": case["question"], "top_k": top_k}
                )
                response.raise_for_status()
                result = score_case(case, response.json(), perf_counter() - started)
            except (httpx.HTTPError, ValueError) as exc:
                result = failed_case(case, perf_counter() - started, str(exc))
            results.append(result)
            print(f"{case['id']}: {result['latency_seconds']}s")

    return {
        "dataset_version": dataset["version"],
        "evaluated_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "evaluation": dataset["evaluation"],
        "ingestion": ingestion.json(),
        "summary": summarize(results, top_k),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "latest.json")
    args = parser.parse_args()

    report = run(args.dataset, args.base_url.rstrip("/"), args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
