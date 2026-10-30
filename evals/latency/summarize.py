"""Summarize RAG stage timings from structured container log events."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
STAGE_FIELDS = ("embedding_ms", "retrieval_ms", "generation_ms", "total_ms")


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile using rank (n - 1) * quantile."""
    if not values:
        raise ValueError("cannot calculate a percentile from no values")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def extract_event(line: str) -> dict | None:
    start = line.find("{")
    if start == -1:
        return None
    try:
        event = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    return event if event.get("event") == "rag_request_completed" else None


def summarize(lines: list[str], run_id: str) -> dict:
    groups = defaultdict(list)
    for line in lines:
        event = extract_event(line)
        if not event:
            continue
        request_id = event.get("request_id", "")
        for kind in ("generated", "abstained"):
            if request_id.startswith(f"{run_id}-{kind}-"):
                groups[kind].append(event)
                break

    result = {"run_id": run_id, "percentile_method": "linear rank=(n-1)*q", "groups": {}}
    for kind in ("generated", "abstained"):
        events = groups[kind]
        if not events:
            raise ValueError(f"no {kind} events found for run ID {run_id}")
        metrics = {}
        for field in STAGE_FIELDS:
            values = [float(event[field]) for event in events]
            metrics[field] = {
                "min": round(min(values), 3),
                "p50": round(percentile(values, 0.50), 3),
                "p95": round(percentile(values, 0.95), 3),
                "max": round(max(values), 3),
            }
        result["groups"][kind] = {"sample_count": len(events), "metrics": metrics}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stages.json")
    args = parser.parse_args()

    report = summarize(list(sys.stdin), args.run_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Stage report written to {args.output}")


if __name__ == "__main__":
    main()
