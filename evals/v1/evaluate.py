"""Run a versioned RAG evaluation dataset through the public HTTP API."""

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

from evals.latency.summarize import percentile
from evals.v1.build_corpus import build_text_pdf

ROOT = Path(__file__).parent
CITATION_PATTERN = re.compile(r"\[Source (\d+), Page (\d+)\]")


def load_dataset(path: Path) -> dict:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    required = {"version", "evaluation", "cases"}
    missing = required - dataset.keys()
    if missing:
        raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")
    if "document" not in dataset and "documents" not in dataset:
        raise ValueError("Dataset must define 'document' or 'documents'")
    documents = (
        dataset["documents"] if "documents" in dataset else [dataset["document"]]
    )
    if not documents:
        raise ValueError("Dataset must contain at least one document")
    document_ids = [document.get("document_id") for document in documents]
    if any(not document_id for document_id in document_ids):
        raise ValueError("Every document must define a document_id")
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Document IDs must be unique")
    if not dataset["cases"]:
        raise ValueError("Dataset must contain at least one evaluation case")

    case_fields = {"id", "question", "expected_facts", "expected_abstain"}
    case_ids = []
    known_document_ids = set(document_ids)
    for case in dataset["cases"]:
        missing_case_fields = case_fields - case.keys()
        if missing_case_fields:
            raise ValueError(
                f"Case is missing required fields: {sorted(missing_case_fields)}"
            )
        if "expected_page" not in case and "expected_pages" not in case:
            raise ValueError("Each case must define expected_page or expected_pages")
        unknown_critical_facts = set(case.get("critical_facts", [])) - set(
            case["expected_facts"]
        )
        if unknown_critical_facts:
            raise ValueError(
                "critical_facts must be a subset of expected_facts: "
                f"{sorted(unknown_critical_facts)}"
            )
        if not case["expected_abstain"]:
            case_document_id = case.get("document_id")
            if not case_document_id and len(documents) == 1:
                case_document_id = document_ids[0]
            if case_document_id not in known_document_ids:
                raise ValueError(f"Case references unknown document_id: {case_document_id}")
        case_ids.append(case["id"])
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case IDs must be unique")
    return dataset


def _expected_pages(case: dict) -> list[int]:
    if "expected_pages" in case:
        return [page for page in case["expected_pages"] if page is not None]
    return [case["expected_page"]] if case["expected_page"] is not None else []


def score_case(case: dict, response: dict, latency_seconds: float) -> dict:
    sources = response.get("sources", [])
    expected_pages = _expected_pages(case)
    expected_page = expected_pages[0] if len(expected_pages) == 1 else None
    expected_document = case.get("document_id")
    answer = response.get("answer", "")
    normalized_answer = answer.casefold()
    expected_abstain = case["expected_abstain"]
    expected_facts = case["expected_facts"]
    critical_facts = case.get("critical_facts", [])
    fact_results = [
        {
            "fact": fact,
            "present_in_answer": fact.casefold() in normalized_answer,
            "critical": fact in critical_facts,
        }
        for fact in expected_facts
    ]
    evidence_text = "\n".join(
        source.get("content", "")
        for source in sources
        if source.get("document_id") == expected_document
    ).casefold()
    evidence_fact_results = [
        {"fact": fact, "present_in_evidence": fact.casefold() in evidence_text}
        for fact in expected_facts
    ]

    retrieval_hit = None
    answer_correct = None
    citation_correct = None
    if not expected_abstain:
        relevant_pages = {
            source["page_number"]
            for source in sources
            if source["document_id"] == expected_document
            and source["page_number"] is not None
        }
        retrieval_hit = bool(expected_pages) and all(
            page in relevant_pages for page in expected_pages
        )
        answer_correct = not response.get("abstained", False) and all(
            result["present_in_answer"] for result in fact_results
        )
        returned_citations = {
            (int(source["chunk_id"]), source["page_number"], source["document_id"])
            for source in sources
            if source["page_number"] is not None
        }
        answer_citations = {
            (int(chunk_id), int(page_number))
            for chunk_id, page_number in CITATION_PATTERN.findall(answer)
        }
        citation_correct = bool(expected_pages) and all(
            any(
                source_chunk_id == cited_chunk_id
                and source_page == expected_page
                and cited_page == expected_page
                and source_document == expected_document
                for source_chunk_id, source_page, source_document in returned_citations
                for cited_chunk_id, cited_page in answer_citations
            )
            for expected_page in expected_pages
        )

    abstention_correct = response.get("abstained") is expected_abstain
    if expected_abstain:
        abstention_correct = abstention_correct and not sources

    return {
        "id": case["id"],
        "question": case["question"],
        "document_id": expected_document,
        "expected_abstain": expected_abstain,
        "expected_page": expected_page,
        "expected_pages": expected_pages,
        "latency_seconds": round(latency_seconds, 3),
        "retrieval_hit": retrieval_hit,
        "answer_correct": answer_correct,
        "citation_correct": citation_correct,
        "abstention_correct": abstention_correct,
        "fact_results": fact_results,
        "fact_recall": (
            round(sum(item["present_in_answer"] for item in fact_results) / len(fact_results), 4)
            if fact_results
            else None
        ),
        "critical_fact_recall": (
            round(
                sum(item["present_in_answer"] for item in fact_results if item["critical"])
                / len(critical_facts),
                4,
            )
            if critical_facts
            else None
        ),
        "evidence_fact_results": evidence_fact_results,
        "evidence_coverage": (
            round(
                sum(item["present_in_evidence"] for item in evidence_fact_results)
                / len(evidence_fact_results),
                4,
            )
            if evidence_fact_results
            else None
        ),
        "context_chunk_count": len(sources),
        "context_character_count": sum(len(source.get("content", "")) for source in sources),
        "response": response,
    }


def rate(results: list[dict], field: str) -> float | None:
    values = [result[field] for result in results if result[field] is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarize(results: list[dict], top_k: int = 5) -> dict:
    latencies = [result["latency_seconds"] for result in results]
    generated_latencies = [
        result["latency_seconds"]
        for result in results
        if not result.get("expected_abstain", result.get("answer_correct") is None)
        and result.get("response") is not None
    ]
    return {
        "case_count": len(results),
        "retrieval_k": top_k,
        "retrieval_hit_rate": rate(results, "retrieval_hit"),
        "answer_correctness_rate": rate(results, "answer_correct"),
        "citation_accuracy": rate(results, "citation_correct"),
        "abstention_accuracy": rate(results, "abstention_correct"),
        "mean_fact_recall": rate(results, "fact_recall"),
        "critical_fact_recall": rate(results, "critical_fact_recall"),
        "mean_evidence_coverage": rate(results, "evidence_coverage"),
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 3),
        "max_latency_seconds": max(latencies),
        "generated_latency_seconds": (
            {
                "sample_count": len(generated_latencies),
                "mean": round(sum(generated_latencies) / len(generated_latencies), 3),
                "p50": round(percentile(generated_latencies, 0.50), 3),
                "p95": round(percentile(generated_latencies, 0.95), 3),
                "max": max(generated_latencies),
            }
            if generated_latencies
            else None
        ),
    }


def failed_case(case: dict, latency_seconds: float, error: str) -> dict:
    expected_abstain = case["expected_abstain"]
    expected_pages = _expected_pages(case)
    return {
        "id": case["id"],
        "question": case["question"],
        "document_id": case.get("document_id"),
        "expected_abstain": expected_abstain,
        "expected_page": expected_pages[0] if len(expected_pages) == 1 else None,
        "expected_pages": expected_pages,
        "latency_seconds": round(latency_seconds, 3),
        "retrieval_hit": None if expected_abstain else False,
        "answer_correct": None if expected_abstain else False,
        "citation_correct": None if expected_abstain else False,
        "abstention_correct": False,
        "fact_results": [],
        "fact_recall": None if expected_abstain else 0.0,
        "critical_fact_recall": None,
        "evidence_fact_results": [],
        "evidence_coverage": None if expected_abstain else 0.0,
        "context_chunk_count": 0,
        "context_character_count": 0,
        "response": None,
        "error": error,
    }


def _ingest_document(client: httpx.Client, dataset_path: Path, document: dict) -> dict:
    if "pages" in document:
        pdf_content = build_text_pdf(document["pages"])
        response = client.post(
            "/v1/documents/pdf",
            data={"document_id": document["document_id"], "title": document["title"]},
            files={
                "file": (
                    f"{document['document_id']}.pdf",
                    pdf_content,
                    "application/pdf",
                )
            },
        )
    else:
        pdf_path = dataset_path.parent / document["pdf_path"]
        with pdf_path.open("rb") as pdf_file:
            response = client.post(
                "/v1/documents/pdf",
                data={"document_id": document["document_id"], "title": document["title"]},
                files={"file": (pdf_path.name, pdf_file, "application/pdf")},
            )
    response.raise_for_status()
    return response.json()


def run(dataset_path: Path, base_url: str, timeout: float, api_key: str) -> dict:
    dataset = load_dataset(dataset_path)
    documents = dataset.get("documents") or [dataset["document"]]
    top_k = dataset["evaluation"]["top_k"]
    default_document_id = documents[0]["document_id"]
    cases = [
        case
        | {
            "document_id": case.get("document_id", default_document_id),
            "expected_pages": _expected_pages(case),
        }
        for case in dataset["cases"]
    ]

    with httpx.Client(
        base_url=base_url,
        timeout=timeout,
        headers={"X-API-Key": api_key},
    ) as client:
        ingestions = [
            _ingest_document(client, dataset_path, document) for document in documents
        ]
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
        "ingestions": ingestions,
        "ingestion": ingestions[0],
        "summary": summarize(results, top_k),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--api-key", default=os.environ.get("RAG_API_KEY"))
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "latest.json")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key or RAG_API_KEY is required")
    report = run(args.dataset, args.base_url.rstrip("/"), args.timeout, args.api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
