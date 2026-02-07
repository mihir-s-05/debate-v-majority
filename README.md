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
# Run majority voting with 5 samples on GSM8K
python -m debug_majority_debate --dataset gsm8k --mode majority --n 5

# Run 3-round debate on GPQA
python -m debug_majority_debate --dataset gpqa --mode debate --rounds 3
```
