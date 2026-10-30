import json

import pytest

from evals.latency.summarize import extract_event, percentile, summarize


def _log_line(request_id: str, **overrides) -> str:
    event = {
        "event": "rag_request_completed",
        "request_id": request_id,
        "embedding_ms": 10.0,
        "retrieval_ms": 2.0,
        "generation_ms": 50.0,
        "total_ms": 62.0,
        **overrides,
    }
    return f"INFO:     {json.dumps(event)}\n"


def test_percentile_uses_documented_linear_interpolation():
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.50) == 25.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.95) == pytest.approx(38.5)


def test_extract_event_ignores_non_json_and_other_events():
    assert extract_event("INFO: server started") is None
    assert extract_event('INFO: {"event":"other"}') is None


def test_summarize_filters_run_and_warmup_events():
    lines = [
        _log_line("run-1-warmup-01"),
        _log_line("run-1-generated-01", total_ms=100.0),
        _log_line("run-1-generated-02", total_ms=200.0),
        _log_line("run-1-abstained-01", generation_ms=0.0, total_ms=12.0),
        _log_line("run-1-abstained-02", generation_ms=0.0, total_ms=14.0),
        _log_line("another-run-generated-01", total_ms=999.0),
    ]

    report = summarize(lines, "run-1")

    assert report["groups"]["generated"]["sample_count"] == 2
    assert report["groups"]["generated"]["metrics"]["total_ms"] == {
        "min": 100.0,
        "p50": 150.0,
        "p95": 195.0,
        "max": 200.0,
    }
    assert report["groups"]["abstained"]["metrics"]["generation_ms"]["max"] == 0.0


def test_summarize_requires_both_request_kinds():
    with pytest.raises(ValueError, match="no abstained events"):
        summarize([_log_line("run-1-generated-01")], "run-1")
