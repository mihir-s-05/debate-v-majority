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

## Usage

```bash
# Run majority voting with 5 samples on GSM8K
python -m debug_majority_debate --dataset gsm8k --mode majority --n 5

# Run 3-round debate on GPQA
python -m debug_majority_debate --dataset gpqa --mode debate --rounds 3
```
