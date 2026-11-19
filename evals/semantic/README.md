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

## Human calibration workflow

RAG-004 adds a separate human-review layer for rubric `1.0.0`. The committed
[`v2 review packet`](annotations/v2-2026-10-29-review-packet.json) contains the questions, generated
answers, abstention flags, and retrieved evidence from the isolated 29 October v2 run. It deliberately
excludes expected facts, deterministic labels, judge scores, and pass/fail results. The accompanying
[`human-label template`](annotations/v2-2026-10-29-human-labels-template.json) is explicitly
unreviewed and is not calibration evidence.

The committed
[`reviewed v2 labels`](annotations/v2-2026-10-29-human-labels-reviewed.json) contain one completed
blinded review of all eight cases. At threshold 3, six of eight cases pass every applicable human
dimension. Mean human scores are 4.0 for faithfulness (six applicable cases), 4.0 for answer
relevance (eight cases), and 3.17 for context relevance (six applicable cases); no case is marked as
containing an unsupported claim. These are human-only descriptive results, not judge-calibration
results. Agreement metrics remain pending until a compatible semantic-judge report is produced.

### Prepare a packet from another full RAG report

```bash
python -m evals.semantic.calibrate prepare \
  --dataset evals/v2/dataset.json \
  --rag-report evals/v2/results/control.json \
  --packet-output evals/semantic/annotations/control-review-packet.json \
  --labels-output evals/semantic/annotations/control-human-labels-template.json
```

The command refuses to overwrite existing files and rejects missing, unknown, or duplicate cases.
Keep the source RAG report and packet fixed after review begins.

### Annotation instructions

1. Copy the unreviewed template to a new file; preserve the original template.
2. Fill `reviewer.reviewer_id` with a stable pseudonym and `reviewed_at` with an ISO-8601 timestamp.
3. Review only the blinded packet. Do not inspect deterministic results or the LLM judge report
   until every human label is complete.
4. For each case, change `review_status` to `reviewed`, fill every applicable score, set
   `unsupported_claims_present` to `true` or `false`, and optionally add a concise rationale.
5. Leave non-applicable score values as `null`. Applicability is explicit: faithfulness is N/A for
   an abstention, context relevance is N/A without evidence, and answer relevance is always scored.

Use the same anchors as judge rubric `1.0.0`:

| Score | Human interpretation |
|---:|---|
| 4 | Fully satisfies the criterion; no material issue. |
| 3 | Mostly satisfies it; only a minor issue. |
| 2 | Partially satisfies it; at least one material issue. |
| 1 | Does not satisfy it or directly conflicts with the evidence. |

Judge faithfulness using only the supplied evidence, not outside knowledge. Mark unsupported claims
when the answer contains at least one material claim that the evidence does not support. Do not mark
a claim unsupported merely because citation formatting is imperfect; citation correctness remains a
separate deterministic metric.

### Compare completed labels with a judge report

```bash
python -m evals.semantic.calibrate compare \
  --human-labels evals/semantic/annotations/v2-2026-10-29-human-labels-reviewed.json \
  --judge-report evals/semantic/results/qwen3-judge.json \
  --output evals/semantic/results/qwen3-human-calibration.json
```

The human labels and judge report must use the same dataset version, rubric version, and pass
threshold. The comparison reports:

- reviewed, comparable, and unreviewed case counts;
- exact score agreement for each dimension;
- agreement within one score point for each dimension;
- mean absolute score error for each dimension;
- human-versus-judge pass/fail agreement;
- unsupported-claim presence agreement;
- human-only and judge-only case IDs; and
- case-level disagreements with both score sets.

Missing or non-applicable scores are excluded and listed; they are never converted to zero. Human,
semantic-judge, deterministic, and retrieval metrics remain separate rather than being combined into
one quality score.

The following is a **synthetic test-fixture example**, not a measured project result:

```json
{
  "reviewed_case_count": 2,
  "comparable_case_count": 2,
  "dimension_agreement": {
    "answer_relevance": {
      "exact_agreement": 0.0,
      "within_one_agreement": 0.5,
      "mean_absolute_error": 1.5
    }
  },
  "pass_fail_agreement": 0.0,
  "unsupported_claim_agreement": 0.5
}
```

### Interpreting calibration results

Eight synthetic cases are enough to exercise the workflow but far too few to establish general judge
validity. Report raw comparable counts beside every agreement rate. A high agreement rate can reflect
easy or homogeneous cases, and a low rate can reflect ambiguous rubric wording, human inconsistency,
or judge failure. Do not tune the rubric merely to match this sample; change it only with a new rubric
version and renewed labels.

The strongest next validation adds a second blinded reviewer, measures human-human agreement, and
adjudicates disagreements before treating either human labels or judge scores as a reference. The
single completed review is useful calibration input but is not a ground-truth quality baseline.
