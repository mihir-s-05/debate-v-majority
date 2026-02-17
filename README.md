# debug_majority_debate

Evaluation harness for comparing LLM inference strategies on math and reasoning benchmarks.

## Modes

- **single** – one completion per question
- **majority** – generate N samples, return most frequent answer
- **debate** – multi-agent debate where models critique and refine answers over multiple rounds

## Datasets Supported

GSM8K, AIME25, GPQA

## Structure

```
cli.py       # CLI interface, evaluation loop
engines.py   # vLLM inference backend
shared.py    # answer extraction, voting, prompt utilities
gsm8k.py / aime25.py / gpqa.py  # dataset loaders + answer checking
data/        # dataset files (auto-downloaded if missing)
```

## Setup

Requires Python 3.10+ and a vLLM-compatible GPU environment.

```bash
pip install -r requirements.txt   # or: pip install vllm==0.13.0
```

**CUDA note:** The vLLM pip wheel bundles a CUDA 12 runtime (`libcudart.so.12`), but it isn't on the default library search path. If you see `ImportError: libcudart.so.12: cannot open shared object file`, add the bundled library to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$HOME/venvs/ma/lib/python3.10/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Add this to your `~/.bashrc` to make it permanent.

## Usage

```bash
python -m debug_majority_debate --dataset gsm8k --mode majority --n 5

python -m debug_majority_debate --dataset gpqa --mode debate --rounds 3
```

## Targeted Judge Repair

Use `rejudge_targets.py` to fix specific debate rows (by `orig_id`) in existing JSONL result files.

```bash
python rejudge_targets.py \
  --dataset gpqa \
  --target "../results/gpqa_quick/debate_gpqa_agents3_r3_n448_seed1105751790_all_20260207_211326_Qwen_Qwen3-32B.jsonl:115" \
  --target "../results/gpqa_quick/debate_gpqa_agents3_r3_n448_seed1105751790_all_20260207_211326_Qwen_Qwen3-32B.jsonl:281" \
  --dry_run
```

Behavior:
- reparses existing `judge_trace` outputs first
- if still invalid, rebuilds judge context from `agent_responses` and reruns judge
- uses the same retry prompt/sampling fallback as normal debate retry logic
- rewrites only targeted rows in-place (creates `.bak.<timestamp>` backups unless `--no_backup`)
