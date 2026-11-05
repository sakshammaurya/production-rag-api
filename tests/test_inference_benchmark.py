import pytest

from app.providers import build_answer_messages
from evals.inference.benchmark_ollama import (
    build_options,
    extract_telemetry,
    select_contexts,
    summarize,
)


def test_build_answer_messages_preserves_production_prompt_and_source_labels():
    messages = build_answer_messages(
        "What happened?",
        [{"chunk_id": 7, "page_number": 2, "content": "Report within one hour."}],
    )

    assert messages[0]["role"] == "system"
    assert "using only the supplied evidence" in messages[0]["content"]
    assert "[Source 7, Page 2] Report within one hour." in messages[1]["content"]


def test_extract_telemetry_converts_ollama_nanoseconds_and_rates():
    result = extract_telemetry(
        {
            "total_duration": 3_000_000_000,
            "load_duration": 100_000_000,
            "prompt_eval_count": 20,
            "prompt_eval_duration": 2_000_000_000,
            "eval_count": 5,
            "eval_duration": 500_000_000,
        }
    )

    assert result == {
        "total_seconds": 3.0,
        "load_seconds": 0.1,
        "prompt_tokens": 20,
        "prompt_evaluation_seconds": 2.0,
        "prompt_tokens_per_second": 10.0,
        "generated_tokens": 5,
        "generation_seconds": 0.5,
        "generation_tokens_per_second": 10.0,
    }


def test_build_options_changes_only_requested_output_ceiling():
    assert build_options() == {"temperature": 0}
    assert build_options(num_predict=256) == {"temperature": 0, "num_predict": 256}
    assert build_options(num_thread=2) == {"temperature": 0, "num_thread": 2}
    assert build_options(num_predict=512, num_thread=4) == {
        "temperature": 0,
        "num_predict": 512,
        "num_thread": 4,
    }

    with pytest.raises(ValueError, match="num_predict must be at least 1"):
        build_options(num_predict=0)
    with pytest.raises(ValueError, match="num_thread must be at least 1"):
        build_options(num_thread=0)


def test_select_contexts_supports_repeatable_focused_measurement():
    dataset = {
        "cases": [
            {"id": "answerable", "expected_abstain": False},
            {"id": "unsupported", "expected_abstain": True},
        ]
    }
    report = {
        "results": [
            {"id": "answerable", "response": {"sources": [{"chunk_id": 1}]}},
            {"id": "unsupported", "response": {"sources": []}},
        ]
    }

    contexts = select_contexts(dataset, report, ["answerable"])

    assert contexts == [
        (dataset["cases"][0], [{"chunk_id": 1}]),
    ]
    with pytest.raises(ValueError, match="Unknown case IDs"):
        select_contexts(dataset, report, ["missing"])


def test_inference_summary_keeps_quality_and_latency_separate():
    telemetry = {
        "total_seconds": 2.0,
        "load_seconds": 0.1,
        "prompt_tokens": 100,
        "prompt_evaluation_seconds": 0.5,
        "prompt_tokens_per_second": 200.0,
        "generated_tokens": 20,
        "generation_seconds": 1.0,
        "generation_tokens_per_second": 20.0,
    }
    measurements = [
        {
            "answer_correct": True,
            "fact_recall": 1.0,
            "critical_fact_recall": 1.0,
            "citation_correct": True,
            "hit_token_limit": True,
            "telemetry": telemetry,
        },
        {
            "answer_correct": False,
            "fact_recall": 0.5,
            "critical_fact_recall": 0.0,
            "citation_correct": True,
            "hit_token_limit": False,
            "telemetry": telemetry | {"total_seconds": 4.0},
        },
    ]

    result = summarize(measurements)

    assert result["answer_correctness"] == 0.5
    assert result["mean_fact_recall"] == 0.75
    assert result["critical_fact_recall"] == 0.5
    assert result["citation_accuracy"] == 1.0
    assert result["token_limit_hit_rate"] == 0.5
    assert result["metrics"]["total_seconds"] == {
        "sample_count": 2,
        "mean": 3.0,
        "p50": 3.0,
        "p95": 3.9,
        "max": 4.0,
    }
