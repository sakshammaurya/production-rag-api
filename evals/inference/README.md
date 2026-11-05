# Ollama inference benchmark

This benchmark measures generation separately from retrieval. It replays the exact source chunks
captured by a completed end-to-end RAG evaluation through Ollama's native `/api/chat` endpoint.
The production prompt builder is reused, but the production API contract and provider abstraction
remain unchanged.

First create an authenticated RAG evaluation report:

```bash
python -m evals.v1.evaluate \
  --dataset evals/v2/dataset.json \
  --output evals/v2/results/control.json
```

Then run one excluded warm-up followed by repeated measurements of every answerable case:

```bash
python -m evals.inference.benchmark_ollama \
  --dataset evals/v2/dataset.json \
  --rag-report evals/v2/results/control.json \
  --model qwen3:4b \
  --iterations 10 \
  --warmup 1 \
  --host-description "Intel i5-6200U, 12 GB RAM, CPU-only" \
  --output evals/inference/results/qwen3-4b-control.json
```

The report records Ollama and model metadata, prompt and generated token counts, load time,
prompt-evaluation time, generation time, tokens per second, and p50/p95/max summaries. It also
scores atomic expected-fact recall, critical-fact recall, answer correctness, and citations for
every repetition.

The benchmark deliberately excludes unsupported questions. Their fast API responses measure the
retrieval abstention path rather than model inference and must not reduce the reported generation
latency. Use the existing end-to-end evaluator and latency benchmark for retrieval and abstention
measurements.

For comparisons, keep the dataset, captured contexts, hardware, Ollama version, model
quantization, warm-up count, iteration count, temperature, and execution order fixed. Change one
independent variable at a time. Results from this small synthetic dataset are regression evidence,
not a production SLO or a general model-quality claim.

## Controlled `num_predict` experiment

The optional `--num-predict` flag changes only Ollama's output-token ceiling. Omit it for the
control. Screen a candidate across every answerable case before spending time on repeated runs:

```bash
python -m evals.inference.benchmark_ollama \
  --dataset evals/v2/dataset.json \
  --rag-report evals/v2/results/control.json \
  --model qwen3:4b \
  --iterations 1 \
  --warmup 1 \
  --num-predict 256 \
  --output evals/inference/results/qwen3-4b-num-predict-256-screen.json
```

Reject the candidate if answer correctness, critical-fact recall, or citation accuracy regresses.
Run repeated control and candidate measurements only after the screen passes.

## CPU thread experiment

Use `--num-thread` to test CPU scheduling without changing the model, quantization, prompt,
retrieved context, or output budget. On a CPU with two physical cores and four logical processors,
compare the omitted runtime default with explicit values of `2` and `4`. Keep the model loaded,
exclude a warm-up, and use identical captured contexts and iteration counts. Promote a setting only
when it preserves all quality gates and improves repeated generation tokens per second and latency.

After the full quality screen passes, `--case-id` can limit a repeated throughput comparison to a
stable representative case. Alternate control and candidate runs to reduce model-temperature,
caching, and execution-order bias. This focused comparison supplements the full quality screen; it
does not replace it.

The reviewed local thread experiment found that two threads were materially slower and four
threads produced only a small exploratory gain over the runtime default while saturating the CPU.
The default therefore remains unchanged. See the
[`qwen3-4b-cpu-thread-screen.json`](baselines/qwen3-4b-cpu-thread-screen.json) artifact for the full
quality screen, utilization observations, and focused paired comparison.

## Quantization experiment

The current model uses `Q4_K_M`; quantization applies to model weights and does not impose a token
limit. The official matching `Q8_0` variant was screened with the same prompts, contexts, and CPU.
It required more disk and loaded memory, reduced stable-case generation throughput by 30.95%, and
timed out before completing the full quality screen. The candidate was rejected and the Q4 default
was retained. See
[`qwen3-4b-q8-rejected.json`](baselines/qwen3-4b-q8-rejected.json) for the measurements and scope.

```bash
python -m evals.inference.benchmark_ollama \
  --dataset evals/v2/dataset.json \
  --rag-report evals/v2/results/control.json \
  --model qwen3:4b \
  --iterations 1 \
  --warmup 1 \
  --num-thread 2 \
  --output evals/inference/results/qwen3-4b-threads-2-screen.json
```
