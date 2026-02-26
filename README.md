# debate-v-majority

`debate-v-majority` is an experiment harness for comparing LLM inference strategies on reasoning benchmarks, with emphasis on reproducible runs and auditable outputs.

## What It Runs

- `single`: one generation per question
- `majority`: sample multiple generations and pick the majority answer
- `debate`: multi-agent, multi-round debate with a final judge step

Supported datasets: GSM8K, AIME25, GPQA.

## Repository Layout

```text
src/debate_v_majority/cli/      CLI entry and run orchestration
src/debate_v_majority/engines/  vLLM backend and runtime config
src/debate_v_majority/shared/   parsing, normalization, and utilities
src/debate_v_majority/datasets/ dataset loaders/adapters
src/debate_v_majority/tools/    rejudge + analysis tooling
scripts/                        script entrypoints
tests/                          unit tests
data/                           dataset files
results/                        run outputs
```

## Setup

Python 3.10+ and a vLLM-compatible GPU environment are expected.

```bash
pip install -e .
pip install -e .[dev]
```

If CUDA shared libraries are not found, add your CUDA runtime path to `LD_LIBRARY_PATH`.

## Quick Usage

```bash
# majority vote on a small gsm8k subset
scripts/debate-v-majority \
  --dataset gsm8k \
  --mode majority \
  --subset_n 5 \
  --model_name Qwen/Qwen3-8B

# debate mode on gpqa
scripts/debate-v-majority \
  --dataset gpqa \
  --mode debate \
  --n_rounds 3 \
  --model_name Qwen/Qwen3-8B
```

## Utilities

```bash
# targeted judge repair
scripts/rejudge-targets --help

# aggregate analysis
scripts/analyze-results --results-dir results --out-dir _autogen
```

## Testing

```bash
pytest -q
```
