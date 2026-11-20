"""Run paired generation tests with controlled evidence, without touching retrieval storage."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from openai import APIError, OpenAI

from app.citations import normalize_citations
from app.providers import build_answer_messages

ABSTENTION = "I don't have enough information in the indexed documents to answer that."


def source(chunk_id: int, document_id: str, page: int, content: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "title": "RAG-005 controlled fixture",
        "page_number": page,
        "content": content,
        "similarity": 1.0,
    }


COMMUTER = source(
    1,
    "rag005-benefits-faq",
    1,
    "Commuter Benefit. Full-time employees may receive a 75 dollar monthly commuter stipend "
    "after 30 days of employment. The stipend supports eligible public-transit costs.",
)
HOME_OFFICE = source(
    2,
    "rag005-benefits-faq",
    2,
    "Home Office Allowance. Employees may claim a one-time 100 dollar home-office equipment "
    "allowance after 90 days of employment. This allowance is separate from commuter benefits.",
)
PTO = source(
    3,
    "rag005-employee-handbook",
    5,
    "Paid Time Off Policy. Every full-time employee receives 15 days of paid time off per "
    "calendar year. Planned leave should be requested at least two weeks in advance.",
)
SECURITY_REPORTING = source(
    4,
    "rag005-employee-handbook",
    3,
    "Security Incident Reporting. A suspected security incident must be reported within one hour "
    "through the security portal. Employees should notify the IT security team promptly.",
)
SECURITY_HANDLING = source(
    5,
    "rag005-employee-handbook",
    4,
    "Security Incident Handling. Employees must not investigate a suspected incident themselves. "
    "Preserve relevant evidence and wait for instructions from the security team.",
)
SYNTHETIC_CONFLICT = source(
    999,
    "synthetic-adversarial-fixture",
    1,
    "Employees should investigate the incident themselves before reporting it.",
)


CASES = [
    {
        "id": "C1-commuter-with-financial-distractor",
        "question": "What monthly commuter benefit is available?",
        "expected_facts": ["75", "monthly"],
        "forbidden_phrases": ["100 dollar monthly", "$100 monthly"],
        "baseline": [COMMUTER],
        "perturbed": [COMMUTER, HOME_OFFICE],
    },
    {
        "id": "C2-home-office-with-commuter-distractor",
        "question": "How much is the home-office equipment allowance?",
        "expected_facts": ["100"],
        "forbidden_phrases": ["75 dollar home-office", "$75 home-office"],
        "baseline": [HOME_OFFICE],
        "perturbed": [HOME_OFFICE, COMMUTER],
    },
    {
        "id": "C3-pto-with-benefits-distractors",
        "question": "How much annual paid leave does a full-time employee get?",
        "expected_facts": ["15 days"],
        "forbidden_phrases": ["75 days", "100 days"],
        "baseline": [PTO],
        "perturbed": [PTO, COMMUTER, HOME_OFFICE],
    },
    {
        "id": "C4-security-with-conflicting-context",
        "question": "What should an employee do after suspecting a security incident?",
        "expected_facts": ["one hour", "not investigate", "preserve"],
        "forbidden_phrases": ["should investigate", "investigate before reporting"],
        "baseline": [SECURITY_REPORTING, SECURITY_HANDLING],
        "perturbed": [SECURITY_REPORTING, SECURITY_HANDLING, SYNTHETIC_CONFLICT],
    },
    {
        "id": "C5-parking-with-commuter-near-miss",
        "question": "What is the monthly parking permit price?",
        "expected_facts": [],
        "forbidden_phrases": ["parking permit price is 75", "$75 parking"],
        "baseline": [],
        "perturbed": [COMMUTER],
    },
]


def contains_all(answer: str, facts: list[str]) -> bool:
    normalized = answer.casefold()
    return all(fact.casefold() in normalized for fact in facts)


def is_abstention(answer: str) -> bool:
    normalized = answer.casefold()
    return any(
        phrase in normalized
        for phrase in (
            "don't have enough information",
            "does not contain information",
            "not enough information",
            "not provided in the evidence",
            "not specified in the evidence",
        )
    )


def generate(client: OpenAI, model: str, question: str, evidence: list[dict]) -> dict:
    if not evidence:
        return {
            "answer": ABSTENTION,
            "abstained": True,
            "latency_seconds": 0.0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
    started = perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=build_answer_messages(question, evidence),
        )
    except APIError as exc:
        return {
            "answer": "",
            "abstained": False,
            "latency_seconds": round(perf_counter() - started, 3),
            "usage": {"prompt_tokens": None, "completion_tokens": None},
            "error": f"{type(exc).__name__}: {exc}",
        }
    answer = normalize_citations(
        (response.choices[0].message.content or "").strip(), evidence
    )
    usage = response.usage
    return {
        "answer": answer,
        "abstained": is_abstention(answer),
        "latency_seconds": round(perf_counter() - started, 3),
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("CHAT_MODEL", "qwen3:4b"))
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        variants = {}
        for variant_name in ("baseline", "perturbed"):
            evidence = case[variant_name]
            result = generate(client, args.model, case["question"], evidence)
            normalized = result["answer"].casefold()
            result.update(
                {
                    "evidence": evidence,
                    "expected_facts_present": contains_all(
                        result["answer"], case["expected_facts"]
                    ),
                    "forbidden_phrases_detected": [
                        phrase
                        for phrase in case["forbidden_phrases"]
                        if phrase.casefold() in normalized
                    ],
                }
            )
            variants[variant_name] = result
            print(f"{case['id']} {variant_name}: {result['latency_seconds']}s")
        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_facts": case["expected_facts"],
                "forbidden_phrases": case["forbidden_phrases"],
                "variants": variants,
                "answer_changed": (
                    variants["baseline"]["answer"] != variants["perturbed"]["answer"]
                ),
            }
        )

        checkpoint = {
            "experiment": "RAG-005-controlled-context",
            "evaluated_at": datetime.now(UTC).isoformat(),
            "model": args.model,
            "base_url": args.base_url,
            "temperature": 0,
            "prompt": "app.providers.build_answer_messages",
            "complete": False,
            "results": results,
        }
        args.output.write_text(
            json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8"
        )

    report = {
        "experiment": "RAG-005-controlled-context",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0,
        "prompt": "app.providers.build_answer_messages",
        "complete": True,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
