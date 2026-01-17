# debug_majority_debate

Small, modular CLI for running `single`, `majority`, and `debate` inference on GSM8K / AIME25 / GPQA using vLLM.

## Quick start

From the repo root:

```bash
python -m debug_majority_debate --model_name <hf-model-id> --dataset gsm8k --gpus 0,1 --mode single,majority,debate
```

## CLI flags

### Required
- `--model_name`: HuggingFace model id (passed to vLLM with `trust_remote_code=True`).

### GPUs
- `--gpus`: Comma-separated GPU ids (sets `CUDA_VISIBLE_DEVICES`). Tensor-parallel size is inferred from this list.

### Dataset selection
- `--dataset`: One of `gsm8k`, `aime25`, `gpqa`.

If the dataset JSONL isn’t present under `data/<dataset>/test.jsonl`, the CLI tries to download it via `datasets`.
`gpqa` is gated on HuggingFace and may require access + `huggingface-cli login`.

### Subset selection (pick one)
- `--subset_n`: Random subset size (default `20`).
- `--subset_ids`: Comma-separated original indices (e.g. `--subset_ids 0,5,9`).
- `--subset_range`: Range like `0:10` (end exclusive) or `0-9` (end inclusive).
- `--subset_seed`: Seed for random subset sampling (default: random per run).

### Modes
- `--mode`: Comma-separated list of `single`, `majority`, `debate` (default: `single,debate`).

Mode-specific:
- `--majority_samples`: Number of samples per question for `majority` (default `5`).
- `--n_agents`: Number of agents for `debate` (default `3`).
- `--n_rounds`: Number of debate rounds (default `3`).

### vLLM / performance
- `--batch_size`: Cap number of prompts queued into one `engine.generate_batch` call. If omitted, an automatic value based on GPU count is used. The engine will back off batch size on OOM.
- `--gpu_memory_utilization`: vLLM `gpu_memory_utilization` (default `0.9`).
- `--context_len`: Fixed vLLM context length (applies to both main generation and judge). If omitted, the engine uses an adaptive context length that can grow (up to a cap) when it detects long prompts.
- `--enable_yarn`: Enable YaRN RoPE scaling when `--context_len` exceeds the model’s native context (also sets `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`).
- `--enforce_eager`: Run vLLM in eager mode (disables torch.compile/cudagraph). Useful for long-context stability.
- `--quiet` / `--silent`: Silence vLLM/CUDA/etc logs; keep only progress bars, output paths, and the final summary.

### Output
- `--out_dir`: Output directory (default: `/home/ubuntu/multi-agent-attack/results/<dataset>_quick`).
- `--tag`: Optional run tag used in filenames.

## Output filenames (matches `quick_gsm8k_vllm.py`)

Outputs are written as JSONL under `--out_dir`:
- `single_{dataset_tag}_{run_tag}.jsonl`
- `majority_{dataset_tag}_{run_tag}_{model_tag}.jsonl`
- `debate_{dataset_tag}_agents{N}_r{R}_{run_tag}_{model_tag}.jsonl`

Where:
- `dataset_tag` is `aime` for `aime25`, otherwise the dataset name.
- `model_tag` is a sanitized version of `--model_name`.
- `run_tag` is `--tag` if provided, otherwise `n{subset_size}_seed{subset_seed}`, then:
  - `_{ids...}` if `--subset_ids` was used, or `_{range...}` if `--subset_range` was used
  - `_{timestamp}` (format `YYYYMMDD_HHMMSS`)

## How it works (high level)

- **single**: one completion per question; parse the answer and score it.
- **majority**: `--majority_samples` completions per question; parse each, then take majority vote.
- **debate**:
  - Maintains `n_agents` chat contexts per question across `n_rounds`.
  - Each round, agents see other agents’ last-round answers and respond again (all contexts are flattened and run in a single batched vLLM call).
  - After the final round, a **judge** is run (same model/engine) with deterministic sampling (`temperature=0`, bounded `max_tokens`) and an adaptive “largest window that fits” transcript selection based on token counting.

Internally, vLLM runs in tensor-parallel across all GPUs in `--gpus`. The engine defaults to throughput-oriented settings (large batching; prefix caching + chunked prefill when supported by the installed vLLM; FP8 KV cache is auto-selected on Ada/Hopper-class GPUs when available, with fallback to `auto` if rejected).
