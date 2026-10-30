"""Run repeatable generated and abstained requests through the public RAG API."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

ROOT = Path(__file__).parent
DOCUMENT_ID = "latency-benchmark-policy"
DOCUMENT_TEXT = (
    "Remote Work Policy. Employees may work remotely on Tuesday and Thursday. "
    "Monday, Wednesday, and Friday are office days."
)
GENERATED_QUESTION = "Which days may employees work remotely?"
ABSTAINED_QUESTION = "Which company provides employee dental insurance?"


def request_case(
    client: httpx.Client,
    *,
    run_id: str,
    kind: str,
    iteration: int,
    question: str,
    expected_abstained: bool,
) -> dict:
    request_id = f"{run_id}-{kind}-{iteration:02d}"
    started = perf_counter()
    response = client.post(
        "/v1/ask",
        headers={"X-Request-ID": request_id},
        json={"question": question, "top_k": 1},
    )
    client_ms = (perf_counter() - started) * 1000
    response.raise_for_status()
    payload = response.json()
    if payload["abstained"] is not expected_abstained:
        raise RuntimeError(
            f"{request_id} returned abstained={payload['abstained']}, "
            f"expected {expected_abstained}"
        )
    return {
        "request_id": request_id,
        "kind": kind,
        "iteration": iteration,
        "client_ms": round(client_ms, 3),
        "abstained": payload["abstained"],
        "source_count": len(payload["sources"]),
    }


def run(
    *,
    base_url: str,
    run_id: str,
    iterations: int,
    warmup: int,
    timeout: float,
) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must not be negative")

    started_at = datetime.now(UTC).isoformat()
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        ingestion = client.post(
            "/v1/documents",
            headers={"X-Request-ID": f"{run_id}-ingestion"},
            json={
                "document_id": DOCUMENT_ID,
                "title": "Latency Benchmark Policy",
                "text": DOCUMENT_TEXT,
            },
        )
        ingestion.raise_for_status()

        warmups = [
            request_case(
                client,
                run_id=run_id,
                kind="warmup",
                iteration=index,
                question=GENERATED_QUESTION,
                expected_abstained=False,
            )
            for index in range(1, warmup + 1)
        ]

        measurements = []
        for iteration in range(1, iterations + 1):
            for kind, question, expected_abstained in (
                ("generated", GENERATED_QUESTION, False),
                ("abstained", ABSTAINED_QUESTION, True),
            ):
                result = request_case(
                    client,
                    run_id=run_id,
                    kind=kind,
                    iteration=iteration,
                    question=question,
                    expected_abstained=expected_abstained,
                )
                measurements.append(result)
                print(f"{result['request_id']}: {result['client_ms']} ms", flush=True)

    return {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "configuration": {
            "iterations_per_kind": iterations,
            "warmup_generated_requests": warmup,
            "top_k": 1,
            "execution": "sequential",
        },
        "ingestion": ingestion.json(),
        "warmups": warmups,
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "requests.json")
    args = parser.parse_args()

    report = run(
        base_url=args.base_url.rstrip("/"),
        run_id=args.run_id,
        iterations=args.iterations,
        warmup=args.warmup,
        timeout=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Request report written to {args.output}")


if __name__ == "__main__":
    main()
