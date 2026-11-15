# Semantic RAG evaluation

This optional offline evaluator uses an OpenAI-compatible chat model as a structured judge over a
previously saved RAG evaluation report. It does not run inside the production request path and does
not replace the deterministic expected-fact, contradiction, citation, abstention, or retrieval
metrics.

## What it measures

Rubric version `1.0.0` scores each applicable dimension from 1 to 4:

- **Faithfulness:** whether claims in the answer are supported by retrieved evidence. Explicit
  abstentions receive `null` because they make no substantive evidence-backed claim.
- **Answer relevance:** whether the answer directly addresses the question, including whether an
  abstention is appropriate when evidence is absent.
- **Context relevance:** whether the retrieved evidence helps answer the question. Cases without
  evidence receive `null`.
- **Unsupported claims:** concise answer claims that the evidence does not support.

The judge supplies scores and brief reasons. Application code—not the judge—computes
`semantic_pass`: every applicable score must meet the configured threshold (default 3), and the
unsupported-claims list must be empty. The report compares this result with the existing
deterministic answer/abstention result but keeps both measurements separate.

## Run

First create a normal RAG report using an isolated evaluation tenant/database:

```bash
python -m evals.v2.evaluate \
  --api-key "$RAG_API_KEY" \
  --output evals/v2/results/control.json
```

Then judge the saved answers and retrieved contexts. Local Ollama exposes an OpenAI-compatible API:

```bash
python -m evals.semantic.evaluate \
  --dataset evals/v2/dataset.json \
  --rag-report evals/v2/results/control.json \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen3:4b \
  --max-tokens 512 \
  --output evals/semantic/results/qwen3-4b-judge.json
```

For a hosted judge, set `JUDGE_API_KEY` in the environment instead of putting a secret in the
command line. The report records the model name and base URL but never records the API key.

The endpoint must support OpenAI-compatible `/chat/completions` and JSON-object response format.
The evaluator uses temperature 0, validates the response schema strictly, records per-case failures,
and reports prompt/completion token counts when the provider supplies them. The default 512-token
judge ceiling bounds runaway generation for this deliberately small JSON schema. It is a safety
limit, not a quality optimum: a response ending because of the ceiling fails validation and the
limit should be reviewed when the rubric changes.

## Report structure

```json
{
  "rubric_version": "1.0.0",
  "judge": {
    "model": "qwen3:4b",
    "temperature": 0,
    "max_tokens": 512,
    "pass_threshold": 3
  },
  "summary": {
    "judged_case_count": 8,
    "judge_failure_count": 0,
    "mean_faithfulness_score": 3.8,
    "semantic_pass_rate": 0.875,
    "unsupported_claim_case_rate": 0.125,
    "deterministic_agreement_rate": 0.875,
    "prompt_tokens": 4200,
    "completion_tokens": 900
  }
}
```

These numbers illustrate the schema only; they are not measured project results.

## Engineering decisions and trade-offs

- **Saved-report input:** retrieval and answer generation are not repeated, reducing cost and making
  judge comparisons use identical answers/context. The trade-off is that the report must contain
  full structured responses; compact historical baseline summaries are insufficient.
- **No evaluation framework:** the implementation uses existing `httpx` and a small versioned
  rubric. This keeps behavior inspectable and dependencies small, but it omits framework-provided
  datasets, dashboards, and prebuilt metrics.
- **Strict JSON contract:** malformed or incomplete judge output is recorded as a failure instead of
  being guessed or silently coerced. This improves auditability but may expose compatibility issues
  with providers that do not implement JSON mode reliably.
- **Code-computed pass/fail:** the judge cannot override the threshold or hide an unsupported claim
  behind an optimistic verdict. Threshold 3 is a policy choice and requires calibration.
- **One aggregate context score:** context relevance covers the returned evidence set, not each
  individual chunk. Per-chunk judging would diagnose noise more precisely but increases calls,
  tokens, latency, and cost.
- **Prompt-injection boundary:** question, answer, and evidence are serialized as untrusted JSON, and
  the system prompt says not to follow embedded instructions. This reduces but cannot eliminate
  judge prompt-injection risk.

## Limitations and responsible interpretation

An LLM judge is not ground truth. Temperature 0 does not guarantee identical results across model or
runtime versions. Scores are sensitive to rubric wording, model family, quantization, context order,
and provider behavior. A judge may prefer its own writing style, miss subtle contradictions, or
incorrectly accept persuasive unsupported text.

Using the same model for answer generation and judging is convenient locally but creates correlated
self-evaluation bias. For stronger evidence, pin the judge model/version, use a capable judge that is
different from the generator, repeat a reviewed sample, and compare it with blinded human labels.
Report disagreements rather than tuning the rubric solely to reproduce deterministic scores.

Retrieved documents and generated answers are sent to the configured judge provider. Do not use a
hosted provider for sensitive documents without an approved privacy and retention policy. Local
Ollama avoids sending content to a third party but does not remove judge-quality limitations.
