"""Benchmark Ollama generation telemetry using contexts captured by a RAG evaluation run."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.citations import normalize_citations
from app.providers import build_answer_messages
from evals.latency.summarize import percentile
from evals.v1.evaluate import load_dataset, score_case


def _seconds(nanoseconds: float | None) -> float | None:
    return round(nanoseconds / 1_000_000_000, 6) if nanoseconds is not None else None


def _tokens_per_second(count: int | None, duration_ns: int | None) -> float | None:
    if not count or not duration_ns:
        return None
    return round(count / (duration_ns / 1_000_000_000), 3)


def extract_telemetry(payload: dict) -> dict:
    return {
        "total_seconds": _seconds(payload.get("total_duration")),
        "load_seconds": _seconds(payload.get("load_duration")),
        "prompt_tokens": payload.get("prompt_eval_count"),
        "prompt_evaluation_seconds": _seconds(payload.get("prompt_eval_duration")),
        "prompt_tokens_per_second": _tokens_per_second(
            payload.get("prompt_eval_count"), payload.get("prompt_eval_duration")
        ),
        "generated_tokens": payload.get("eval_count"),
        "generation_seconds": _seconds(payload.get("eval_duration")),
        "generation_tokens_per_second": _tokens_per_second(
            payload.get("eval_count"), payload.get("eval_duration")
        ),
    }


def _metric_summary(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "sample_count": len(values),
        "mean": round(sum(values) / len(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def build_options(*, num_predict: int | None = None) -> dict:
    if num_predict is not None and num_predict < 1:
        raise ValueError("num_predict must be at least 1")
    options = {"temperature": 0}
    if num_predict is not None:
        options["num_predict"] = num_predict
    return options


def summarize(measurements: list[dict]) -> dict:
    telemetry_fields = (
        "total_seconds",
        "load_seconds",
        "prompt_tokens",
        "prompt_evaluation_seconds",
        "prompt_tokens_per_second",
        "generated_tokens",
        "generation_seconds",
        "generation_tokens_per_second",
    )
    return {
        "sample_count": len(measurements),
        "answer_correctness": round(
            sum(item["answer_correct"] for item in measurements) / len(measurements), 4
        ),
        "mean_fact_recall": round(
            sum(item["fact_recall"] for item in measurements) / len(measurements), 4
        ),
        "critical_fact_recall": (
            round(
                sum(
                    item["critical_fact_recall"]
                    for item in measurements
                    if item["critical_fact_recall"] is not None
                )
                / sum(item["critical_fact_recall"] is not None for item in measurements),
                4,
            )
            if any(item["critical_fact_recall"] is not None for item in measurements)
            else None
        ),
        "citation_accuracy": round(
            sum(item["citation_correct"] for item in measurements) / len(measurements), 4
        ),
        "token_limit_hit_rate": round(
            sum(item.get("hit_token_limit", False) for item in measurements)
            / len(measurements),
            4,
        ),
        "metrics": {
            field: _metric_summary(
                [
                    float(item["telemetry"][field])
                    for item in measurements
                    if item["telemetry"].get(field) is not None
                ]
            )
            for field in telemetry_fields
        },
    }


def run(
    *,
    dataset_path: Path,
    rag_report_path: Path,
    ollama_url: str,
    model: str,
    iterations: int,
    warmup: int,
    timeout: float,
    host_description: str | None = None,
    num_predict: int | None = None,
) -> dict:
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be at least 1 and warmup must not be negative")
    dataset = load_dataset(dataset_path)
    options = build_options(num_predict=num_predict)
    cases = {case["id"]: case for case in dataset["cases"]}
    rag_report = json.loads(rag_report_path.read_text(encoding="utf-8"))
    contexts = []
    for result in rag_report["results"]:
        case = cases[result["id"]]
        if case["expected_abstain"] or not result.get("response"):
            continue
        sources = result["response"]["sources"]
        contexts.append((case, sources))
    if not contexts:
        raise ValueError("RAG report contains no generated-answer contexts")

    with httpx.Client(base_url=ollama_url, timeout=timeout) as client:
        version = client.get("/api/version").json()
        model_info = client.post("/api/show", json={"model": model}).json()
        warmups = []
        first_case, first_sources = contexts[0]
        for index in range(warmup):
            response = client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": build_answer_messages(first_case["question"], first_sources),
                    "stream": False,
                    "options": options,
                },
            )
            response.raise_for_status()
            warmups.append({"iteration": index + 1, **extract_telemetry(response.json())})

        measurements = []
        for iteration in range(1, iterations + 1):
            for case, sources in contexts:
                response = client.post(
                    "/api/chat",
                    json={
                        "model": model,
                        "messages": build_answer_messages(case["question"], sources),
                        "stream": False,
                        "options": options,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                raw_answer = payload.get("message", {}).get("content", "").strip()
                answer = normalize_citations(raw_answer, sources)
                scored = score_case(
                    case,
                    {"answer": answer, "abstained": False, "sources": sources},
                    _seconds(payload.get("total_duration")) or 0.0,
                )
                measurements.append(
                    {
                        "case_id": case["id"],
                        "iteration": iteration,
                        "raw_answer": raw_answer,
                        "answer": answer,
                        "answer_correct": scored["answer_correct"],
                        "fact_results": scored["fact_results"],
                        "fact_recall": scored["fact_recall"],
                        "critical_fact_recall": scored["critical_fact_recall"],
                        "citation_correct": scored["citation_correct"],
                        "hit_token_limit": (
                            num_predict is not None
                            and payload.get("eval_count") == num_predict
                        ),
                        "context_chunk_count": scored["context_chunk_count"],
                        "context_character_count": scored["context_character_count"],
                        "telemetry": extract_telemetry(payload),
                    }
                )
                print(
                    f"{case['id']} iteration {iteration}: "
                    f"{measurements[-1]['telemetry']['total_seconds']}s",
                    flush=True,
                )

    details = model_info.get("details", {})
    return {
        "benchmark": "ollama-native-inference",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "dataset_version": dataset["version"],
        "source_rag_report": str(rag_report_path),
        "environment": {
            "ollama_version": version.get("version"),
            "model": model,
            "model_family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "host_description": host_description,
        },
        "methodology": {
            "iterations_per_answerable_case": iterations,
            "warmups_excluded": warmup,
            "execution": "sequential by iteration using contexts captured from the RAG API",
            "temperature": 0,
            "num_predict": num_predict,
            "duration_unit_from_ollama": "nanoseconds",
        },
        "warmups": warmups,
        "summary": summarize(measurements),
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rag-report", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--host-description",
        help="Human-readable CPU/RAM description recorded for reproducibility",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        help="Optional Ollama output-token ceiling; omit to use the model default",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        dataset_path=args.dataset,
        rag_report_path=args.rag_report,
        ollama_url=args.ollama_url.rstrip("/"),
        model=args.model,
        iterations=args.iterations,
        warmup=args.warmup,
        timeout=args.timeout,
        host_description=args.host_description,
        num_predict=args.num_predict,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
