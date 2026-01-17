"""
CLI for debug_majority_debate package.

Provides simplified command-line interface for running inference/evaluation
on GSM8K, AIME25, and GPQA datasets with single, majority, or debate modes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, TextIO, cast

from tqdm import tqdm

from . import DatasetName, Mode


class _DoubleCtrlCHandler:
    """
    Require two Ctrl+C presses within a time window to actually exit.
    Prevents accidental termination of long-running evaluations.
    """

    def __init__(self, timeout: float = 2.0, output_file: TextIO | None = None) -> None:
        self.timeout = timeout
        self.output_file = output_file or sys.stderr
        self._last_sigint_time: float | None = None
        self._original_handler = None

    def _handler(self, signum: int, frame) -> None:
        now = time.monotonic()
        if self._last_sigint_time is not None and (now - self._last_sigint_time) < self.timeout:
            # Second Ctrl+C within timeout - actually exit
            print("\nInterrupted.", file=self.output_file, flush=True)
            sys.exit(130)  # Standard exit code for SIGINT
        else:
            # First Ctrl+C - warn and wait for confirmation
            self._last_sigint_time = now
            print(
                f"\nPress Ctrl+C again within {self.timeout:.0f}s to cancel...",
                file=self.output_file,
                flush=True,
            )

    def __enter__(self) -> "_DoubleCtrlCHandler":
        self._last_sigint_time = None
        try:
            self._original_handler = signal.signal(signal.SIGINT, self._handler)
        except ValueError:
            # Likely not running in the main thread; leave default handling intact.
            self._original_handler = None
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
        return False


from .shared import (
    most_frequent_answer,
    render_agent_assistant_rounds,
    PromptTokenCounter,
    is_prompt_too_long,
    strip_thinking_content,
    strip_thinking_from_messages,
    round_block_start,
    PrevJudgeInfo,
    format_prev_judge_full,
    format_prev_judge_short,
)
from .engines import (
    InferenceEngine,
    build_sampling_config,
    set_sampling_config,
    create_inference_engine,
    infer_native_context_len,
)

class _QuietOutput:
    """
    Silence stdout/stderr noise (e.g., vLLM/CUDA logs) while still allowing
    explicitly chosen output (progress bars, summary) to be shown.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None
        self.keep_stdout: TextIO = sys.stdout

    def __enter__(self) -> "_QuietOutput":
        if not self.enabled:
            self.keep_stdout = sys.stdout
            return self

        for s in (sys.stdout, sys.stderr):
            try:
                s.flush()
            except Exception:
                pass

        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        self.keep_stdout = os.fdopen(
            self._saved_stdout_fd,
            "w",
            buffering=1,
            encoding="utf-8",
            errors="replace",
        )

        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
        finally:
            try:
                os.close(devnull_fd)
            except Exception:
                pass

        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled:
            return False

        # Restore before propagating exceptions so tracebacks are visible.
        if self._saved_stdout_fd is not None:
            try:
                os.dup2(self._saved_stdout_fd, 1)
            except Exception:
                pass
        if self._saved_stderr_fd is not None:
            try:
                os.dup2(self._saved_stderr_fd, 2)
            except Exception:
                pass

        try:
            self.keep_stdout.flush()
        except Exception:
            pass
        try:
            self.keep_stdout.close()
        except Exception:
            pass
        if self._saved_stderr_fd is not None:
            try:
                os.close(self._saved_stderr_fd)
            except Exception:
                pass

        return False


# =============================================================================
# Dataset imports (lazy to avoid circular imports)
# =============================================================================


def _get_dataset_module(dataset: DatasetName):
    """Lazily import dataset-specific module."""
    if dataset == "gsm8k":
        from . import gsm8k
        return gsm8k
    elif dataset == "gpqa":
        from . import gpqa
        return gpqa
    elif dataset == "aime25":
        from . import aime25
        return aime25
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _parse_question_answer(dataset: DatasetName, sample: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Parse a dataset sample."""
    mod = _get_dataset_module(dataset)
    return mod.parse_question_answer(sample)


def _parse_answer(dataset: DatasetName, text: str, raw_task: dict[str, Any]) -> str | None:
    """Parse an answer from model response."""
    mod = _get_dataset_module(dataset)
    return mod.parse_answer(text, raw_task)


def _check_answer_correctness(dataset: DatasetName, answer: Any, gt: Any) -> int:
    """Check if answer is correct."""
    mod = _get_dataset_module(dataset)
    return mod.check_answer_correctness(answer, gt)


def _construct_debate_message(dataset: DatasetName, other_agent_answers: list[str]) -> dict[str, str]:
    """Construct debate prompt."""
    mod = _get_dataset_module(dataset)
    return mod.construct_debate_message(other_agent_answers)


def _get_judge_prompt(dataset: DatasetName) -> dict[str, str]:
    """Get judge prompt configuration."""
    mod = _get_dataset_module(dataset)
    return mod.JUDGE_PROMPT


# =============================================================================
# Data utilities
# =============================================================================


@dataclass(frozen=True)
class SubsetItem:
    """A single item in the evaluation subset."""
    subset_id: int
    orig_id: int
    raw_task: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_dataset_test_path(dataset: DatasetName) -> Path:
    """Get default test dataset path."""
    return Path(__file__).resolve().parent / "data" / dataset / "test.jsonl"


def _ensure_dataset_test_jsonl(dataset: DatasetName, test_path: Path) -> None:
    """Download dataset if not present."""
    if test_path.exists():
        return

    try:
        import datasets
    except Exception as e:
        raise FileNotFoundError(
            f"Missing dataset file at {test_path} and `datasets` is unavailable: {e}"
        ) from e

    print(f"[data] Downloading {dataset} from HuggingFace -> {test_path}", file=sys.stderr)

    if dataset == "gsm8k":
        ds = datasets.load_dataset("gsm8k", name="main", split="test")
    elif dataset == "aime25":
        ds = datasets.load_dataset("math-ai/aime25", split="test")
    elif dataset == "gpqa":
        # GPQA is gated - try to load it
        try:
            cfgs = datasets.get_dataset_config_names("Idavidrein/gpqa")
            config = next((c for c in cfgs if "main" in c.lower() and "diamond" not in c.lower()), cfgs[0])
            ds = datasets.load_dataset("Idavidrein/gpqa", config, split="test")
        except Exception:
            raise FileNotFoundError(
                "Could not download GPQA (gated dataset). Visit https://huggingface.co/datasets/Idavidrein/gpqa "
                "and request access, then run `huggingface-cli login`."
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    test_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(test_path, (dict(ex) for ex in ds))


def _make_dataset_subset(
    *,
    dataset: DatasetName,
    test_path: Path,
    n: int,
    seed: int,
    ids: list[int] | None,
    range_str: str | None,
) -> tuple[list[SubsetItem], dict[str, Any]]:
    """Create a subset of the dataset."""
    _ensure_dataset_test_jsonl(dataset, test_path)
    all_rows = _read_jsonl(test_path)
    total = len(all_rows)

    # Choose indices
    if ids:
        chosen = list(dict.fromkeys(ids))
    elif range_str:
        s = str(range_str).strip().lower()
        if s in ("all", "*"):
            chosen = list(range(total))
        elif ":" in range_str:
            a, b = range_str.split(":", 1)
            start = int(a) if a else 0
            end = int(b) if b else total
            chosen = list(range(start, min(end, total)))
        else:
            a, b = range_str.split("-", 1)
            start = int(a)
            end = int(b) + 1
            chosen = list(range(start, min(end, total)))
    else:
        n = min(n, total)
        rng = random.Random(seed)
        chosen = rng.sample(range(total), n)

    items = [
        SubsetItem(subset_id=i, orig_id=orig_id, raw_task=all_rows[orig_id])
        for i, orig_id in enumerate(chosen)
    ]
    meta = {
        "dataset": dataset,
        "total_available": total,
        "subset_size": len(items),
        "seed": seed,
        "orig_ids": chosen,
        "source_path": str(test_path),
    }
    return items, meta


# =============================================================================
# Judge utilities
# =============================================================================


JUDGE_SYSTEM_PROMPT = (
    "You are a judge agent. You are tasked with evaluating some responses from different agents to a given "
    "question and decide which one is the correct answer to the question. Read their answers carefully and "
    "select the answer from the agent that you think is the most accurate. Provide the final answer as "
    "prompted in the question.\n\n"
)


def _build_judge_context(
    *,
    dataset: DatasetName,
    question: str,
    responses: list[str],
    previous_judge: str | None = None,
) -> list[dict[str, str]]:
    """Build judge context messages."""
    parts = [f"Question: {question}"]
    if previous_judge:
        parts.append(f"Previous judge output (from earlier rounds):\n{previous_judge}")

    for response in responses:
        parts.append(f"One agent solution: {response}")

    judge_prompt = _get_judge_prompt(dataset)
    user_prompt = "\n\n".join(parts) + judge_prompt["user_prompt_suffix"]

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _parse_csv_ints(s: str | None) -> list[int]:
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _parse_subset_n_arg(v: str) -> int | str:
    """
    Parse --subset_n which normally is an int, but also supports "all"/"*"
    to mean "use the full dataset" (equivalent to --all).
    """
    s = str(v).strip().lower()
    if s in ("all", "*"):
        return "all"
    return int(s)


def _parse_judge_rounds(rounds_str: str | None, n_rounds: int) -> list[int]:
    """Parse comma-separated round numbers and validate against max rounds."""
    if not rounds_str:
        return [n_rounds]
    rounds = _parse_csv_ints(rounds_str)
    if not rounds:
        return [n_rounds]
    invalid = [r for r in rounds if r < 1 or r > n_rounds]
    if invalid:
        raise ValueError(f"Invalid judge rounds {invalid}: must be between 1 and {n_rounds}")
    return sorted(set(rounds))


def _select_adaptive_judge_window(
    *,
    dataset: DatasetName,
    question: str,
    agent_contexts: list[list[dict[str, str]]],
    end_round: int,
    prev: PrevJudgeInfo | None,
    counter: PromptTokenCounter,
    context_len_tokens: int,
    max_new_tokens: int,
) -> tuple[int, str | None]:
    """
    Choose the earliest start_round such that the judge prompt fits within
    context_len_tokens, reserving max_new_tokens for generation.
    """
    if end_round <= 0 or context_len_tokens <= 0:
        return 1, None

    # Reserve space for generation + safety margin for tokenizer overhead
    budget = max(1, int(context_len_tokens) - max(0, int(max_new_tokens)) - 256)

    prev_options: list[str | None] = [None]
    if prev is not None:
        prev_options = [format_prev_judge_full(prev), format_prev_judge_short(prev), None]

    for prev_text in prev_options:
        start_round = 1
        while start_round <= end_round:
            responses = [
                render_agent_assistant_rounds(agent_conv=ctx, start_round=start_round, end_round=end_round)
                for ctx in agent_contexts
            ]
            msgs = _build_judge_context(
                dataset=dataset, question=question, responses=responses, previous_judge=prev_text
            )
            try:
                n_tokens = counter.count_chat_tokens(msgs)
            except Exception:
                n_tokens = int(counter.estimate_prompt_tokens(msgs, exact_if_large=1) * 1.1)
            if n_tokens <= budget:
                return start_round, prev_text
            start_round += 1

    raise RuntimeError(f"Judge prompt does not fit within context_len_tokens={context_len_tokens}")


# =============================================================================
# Execution modes
# =============================================================================


def run_sampled(
    *,
    dataset: DatasetName,
    items: list[SubsetItem],
    engine: InferenceEngine,
    n_samples: int,
    batch_size: int | None,
    mode_label: Mode,
    progress_file: TextIO = sys.stdout,
) -> list[dict[str, Any]]:
    """Run single or majority voting mode."""
    parsed_inputs: list[tuple[SubsetItem, str, Any, dict[str, Any]]] = []
    contexts_flat: list[list[dict[str, str]]] = []

    for item in items:
        question, gt_answer, raw_task = _parse_question_answer(dataset, item.raw_task)
        parsed_inputs.append((item, question, gt_answer, raw_task))
        contexts_flat.extend([[{"role": "user", "content": question}] for _ in range(n_samples)])

    pbar = tqdm(total=len(contexts_flat), desc=mode_label, unit="call", file=progress_file)
    completions_flat = engine.generate_batch(contexts_flat, batch_size=batch_size)
    pbar.update(len(contexts_flat))
    pbar.close()

    records: list[dict[str, Any]] = []
    for item_idx, (item, question, gt_answer, raw_task) in enumerate(parsed_inputs):
        start = item_idx * n_samples
        end = start + n_samples
        sample_completions = completions_flat[start:end]
        sample_parsed = [_parse_answer(dataset, c, raw_task) for c in sample_completions]
        final_answer = sample_parsed[0] if n_samples == 1 else most_frequent_answer(sample_parsed)
        final_correct = _check_answer_correctness(dataset, final_answer, gt_answer)
        records.append({
            "mode": mode_label,
            "subset_id": item.subset_id,
            "orig_id": item.orig_id,
            "question": question,
            "answer": gt_answer,
            "raw_task": raw_task,
            "n_samples": n_samples,
            "sample_completions": sample_completions,
            "sample_parsed_answers": sample_parsed,
            "final_answer": final_answer,
            "final_correct": final_correct,
        })

    return records


def run_debate(
    *,
    dataset: DatasetName,
    items: list[SubsetItem],
    engine: InferenceEngine,
    n_agents: int,
    n_rounds: int,
    judge_rounds: list[int],
    batch_size: int | None,
    judge_block_size: int | None = None,
    judge_sampling_kwargs: dict[str, Any] | None = None,
    judge_engine: InferenceEngine | None = None,
    progress_file: TextIO = sys.stdout,
) -> dict[int, list[dict[str, Any]]]:
    """Run multi-agent debate mode (matching quick_gsm8k_vllm.py semantics)."""
    # Pre-parse all items
    parsed_items: list[tuple[SubsetItem, str, Any, dict[str, Any]]] = []
    for item in items:
        question, gt_answer, raw_task = _parse_question_answer(dataset, item.raw_task)
        parsed_items.append((item, question, gt_answer, raw_task))

    # Initialize contexts: all_contexts[question_idx][agent_idx]
    all_contexts: list[list[list[dict[str, str]]]] = [
        [[{"role": "user", "content": question}] for _ in range(n_agents)]
        for item, question, gt_answer, raw_task in parsed_items
    ]

    judge_engine = judge_engine or engine
    token_counter = PromptTokenCounter(getattr(judge_engine, "model_name", engine.model_name))
    prev_judge_by_q: list[PrevJudgeInfo | None] = [None for _ in parsed_items]
    results_by_round: dict[int, list[dict[str, Any]]] = {r: [] for r in judge_rounds}

    for round_idx in tqdm(range(n_rounds), desc="debate rounds", total=n_rounds, file=progress_file):
        # Add debate prompts for rounds > 0
        if round_idx > 0:
            for q_idx in range(len(parsed_items)):
                agent_contexts = all_contexts[q_idx]
                last_answers = [ctx[-1]["content"] for ctx in agent_contexts]
                for agent_idx, agent_ctx in enumerate(agent_contexts):
                    other_answers = [a for j, a in enumerate(last_answers) if j != agent_idx]
                    agent_ctx.append(_construct_debate_message(dataset, other_answers))

        # Flatten all contexts for batch generation
        flat_contexts = [
            all_contexts[q_idx][agent_idx]
            for q_idx in range(len(parsed_items))
            for agent_idx in range(n_agents)
        ]

        # Generate all completions
        flat_completions = engine.generate_batch(flat_contexts, batch_size=batch_size)

        # Unflatten and append to contexts
        for q_idx in range(len(parsed_items)):
            for agent_idx in range(n_agents):
                flat_idx = q_idx * n_agents + agent_idx
                all_contexts[q_idx][agent_idx].append({"role": "assistant", "content": flat_completions[flat_idx]})

        # If the engine has flagged thinking stripping (due to context pressure),
        # strip thinking from all canonical contexts so subsequent rounds use stripped versions
        if hasattr(engine, "thinking_stripped") and engine.thinking_stripped:
            for q_idx in range(len(parsed_items)):
                for agent_idx in range(n_agents):
                    stripped_ctx, changed = strip_thinking_from_messages(all_contexts[q_idx][agent_idx])
                    if changed:
                        all_contexts[q_idx][agent_idx] = stripped_ctx

        current_round_num = round_idx + 1
        if current_round_num not in judge_rounds:
            continue

        block_end = current_round_num
        used_start_rounds: list[int] = [1 for _ in parsed_items]
        used_prev_texts: list[str | None] = [None for _ in parsed_items]

        # Determine judge context length (adaptive engines may upgrade mid-run).
        ctx_len = int(getattr(judge_engine, "context_len_tokens", 0) or 0) or (
            infer_native_context_len(getattr(judge_engine, "model_name", engine.model_name)) or 32768
        )

        # Reserve space for judge generation based on actual max_tokens that will be used.
        # This prevents position overflow when the judge generates a long response.
        if judge_sampling_kwargs and "max_tokens" in judge_sampling_kwargs:
            judge_max_new_tokens = int(judge_sampling_kwargs["max_tokens"])
        else:
            # Fall back to engine's default sampling params
            from .engines import get_sampling_config
            sampling_cfg = get_sampling_config()
            judge_max_new_tokens = sampling_cfg.max_tokens or 4096

        def _build_contexts_for_round() -> list[list[dict[str, str]]]:
            judge_contexts: list[list[dict[str, str]]] = []
            for q_idx, (item, question, gt_answer, raw_task) in enumerate(parsed_items):
                agent_contexts = all_contexts[q_idx]
                if judge_block_size is not None and int(judge_block_size) > 0:
                    block_start = round_block_start(block_end, int(judge_block_size))
                    prev_text = format_prev_judge_full(prev_judge_by_q[q_idx]) if prev_judge_by_q[q_idx] else None
                else:
                    block_start, prev_text = _select_adaptive_judge_window(
                        dataset=dataset,
                        question=question,
                        agent_contexts=agent_contexts,
                        end_round=block_end,
                        prev=prev_judge_by_q[q_idx],
                        counter=token_counter,
                        context_len_tokens=ctx_len,
                        max_new_tokens=judge_max_new_tokens,
                    )

                used_start_rounds[q_idx] = int(block_start)
                used_prev_texts[q_idx] = prev_text
                transcripts = [
                    render_agent_assistant_rounds(agent_conv=ctx, start_round=block_start, end_round=block_end)
                    for ctx in agent_contexts
                ]
                judge_contexts.append(
                    _build_judge_context(dataset=dataset, question=question, responses=transcripts, previous_judge=prev_text)
                )
            return judge_contexts

        judge_contexts = _build_contexts_for_round()
        try:
            judge_raw_outputs = judge_engine.generate_batch(
                judge_contexts, batch_size=batch_size, sampling_kwargs=judge_sampling_kwargs
            )
        except Exception as e:
            if not is_prompt_too_long(e):
                raise
            # Fallback: judge per-question (batching can exceed limits even if each prompt fits).
            print(
                "[warn] Judge prompt too long in batched evaluation; falling back to per-question judging.",
                file=sys.stderr,
            )
            judge_raw_outputs = []
            judge_contexts = []
            for q_idx, (item, question, gt_answer, raw_task) in enumerate(parsed_items):
                agent_contexts = all_contexts[q_idx]
                if judge_block_size is not None and int(judge_block_size) > 0:
                    block_start = round_block_start(block_end, int(judge_block_size))
                    prev_text = format_prev_judge_full(prev_judge_by_q[q_idx]) if prev_judge_by_q[q_idx] else None
                else:
                    block_start, prev_text = _select_adaptive_judge_window(
                        dataset=dataset,
                        question=question,
                        agent_contexts=agent_contexts,
                        end_round=block_end,
                        prev=prev_judge_by_q[q_idx],
                        counter=token_counter,
                        context_len_tokens=ctx_len,
                        max_new_tokens=judge_max_new_tokens,
                    )
                used_start_rounds[q_idx] = int(block_start)
                used_prev_texts[q_idx] = prev_text
                transcripts = [
                    render_agent_assistant_rounds(agent_conv=ctx, start_round=block_start, end_round=block_end)
                    for ctx in agent_contexts
                ]
                ctx_msgs = _build_judge_context(dataset=dataset, question=question, responses=transcripts, previous_judge=prev_text)
                out = judge_engine.generate_batch([ctx_msgs], batch_size=1, sampling_kwargs=judge_sampling_kwargs)[0]
                judge_contexts.append(ctx_msgs)
                judge_raw_outputs.append(out)

        # If the judge output is unparsable, do NOT hard-crash the whole eval.
        # Instead: try stripping <think> blocks, retry judge once with a strict-format nudge,
        # and if it still fails, keep the judge answer as None (counts as wrong).
        judged_answers: list[str | None] = []
        judge_retry_raw_outputs: list[str | None] = [None for _ in parsed_items]
        judge_parse_failed: list[bool] = [False for _ in parsed_items]
        judge_used_fallback: list[bool] = [False for _ in parsed_items]

        for q_idx, (raw_out, (item, question, gt_answer, raw_task)) in enumerate(zip(judge_raw_outputs, parsed_items)):
            used_fallback = False
            judged = _parse_answer(dataset, raw_out, raw_task)
            if judged is None:
                judged = _parse_answer(dataset, strip_thinking_content(str(raw_out)), raw_task)
                if judged is not None:
                    used_fallback = True

            if judged is None:
                # Retry once with an explicit instruction to output ONLY the final boxed answer.
                used_fallback = True
                try:
                    retry_ctx = list(judge_contexts[q_idx]) + [
                        {
                            "role": "user",
                            "content": (
                                "Your previous output was unparsable.\n"
                                "Reply again and output ONLY the final answer in the required format (e.g., \\boxed{...}).\n"
                                "Do not include any other text."
                            ),
                        }
                    ]
                    retry_sampling = dict(judge_sampling_kwargs or {})
                    retry_sampling.setdefault("temperature", 0.0)
                    retry_sampling.setdefault("top_p", 1.0)
                    retry_sampling.setdefault("max_tokens", min(int(judge_max_new_tokens), 512))
                    retry_out = judge_engine.generate_batch([retry_ctx], batch_size=1, sampling_kwargs=retry_sampling)[0]
                    judge_retry_raw_outputs[q_idx] = str(retry_out)
                    judged = _parse_answer(dataset, retry_out, raw_task)
                    if judged is None:
                        judged = _parse_answer(dataset, strip_thinking_content(str(retry_out)), raw_task)
                except Exception as e:
                    # Any retry failure should not kill the eval either.
                    judge_retry_raw_outputs[q_idx] = f"[retry_error] {type(e).__name__}: {e}"
                    judged = None

            if judged is None:
                judge_parse_failed[q_idx] = True
                # Keep stderr noise low: one concise warning per failure.
                try:
                    print(
                        f"[warn] Unparsable judge output after retry; marking wrong (judge_answer=None) for subset_id={item.subset_id} orig_id={item.orig_id} round={current_round_num}",
                        file=sys.stderr,
                    )
                except Exception:
                    pass

            judge_used_fallback[q_idx] = used_fallback
            judged_answers.append(judged)

        # Cache this block's judge output to condition the next block's judge.
        for q_idx, (raw_out, judged) in enumerate(zip(judge_raw_outputs, judged_answers)):
            # Only cache if we actually have a parsed judge answer (avoid poisoning future rounds).
            if judged is None or judge_used_fallback[q_idx]:
                continue
            prev_judge_by_q[q_idx] = PrevJudgeInfo(
                start_round=int(used_start_rounds[q_idx]),
                end_round=block_end,
                parsed_answer=str(judged),
                raw_output=str(raw_out),
            )

        for q_idx, (item, question, gt_answer, raw_task) in enumerate(parsed_items):
            agent_contexts = all_contexts[q_idx]
            judged = judged_answers[q_idx]
            final_turn_parsed = [_parse_answer(dataset, ctx[-1]["content"], raw_task) for ctx in agent_contexts]
            final_majority_answer = most_frequent_answer(final_turn_parsed)
            final_majority_correct = _check_answer_correctness(dataset, final_majority_answer, gt_answer)

            final_judge_answer = judged
            final_judge_correct = _check_answer_correctness(dataset, final_judge_answer, gt_answer)
            judge_trace = {
                "judge_backend": "vllm",
                "judge_model": getattr(judge_engine, "model_name", engine.model_name),
                "judge_context": judge_contexts[q_idx],
                "judge_raw_response": judge_raw_outputs[q_idx],
                "judge_retry_raw_response": judge_retry_raw_outputs[q_idx],
                "judge_parsed_answer": judged,
                "judge_parse_failed": bool(judge_parse_failed[q_idx]),
                "judge_used_fallback": bool(judge_used_fallback[q_idx]),
                "judge_correct": final_judge_correct,
            }

            results_by_round[current_round_num].append(
                {
                    "mode": "debate",
                    "subset_id": item.subset_id,
                    "orig_id": item.orig_id,
                    "question": question,
                    "answer": gt_answer,
                    "raw_task": raw_task,
                    "n_agents": n_agents,
                    "n_rounds": current_round_num,
                    "agent_responses": [ctx[:] for ctx in agent_contexts],
                    "final_majority_answer": final_majority_answer,
                    "final_majority_correct": final_majority_correct,
                    "judge_trace": judge_trace,
                    "final_judge_answer": final_judge_answer,
                    "final_judge_correct": final_judge_correct,
                    "final_answer": final_judge_answer,
                    "final_correct": final_judge_correct,
                }
            )

    return results_by_round


# =============================================================================
# Output utilities
# =============================================================================


def _accuracy(rows: list[dict[str, Any]]) -> float:
    """Calculate accuracy."""
    if not rows:
        return 0.0
    return sum(int(r["final_correct"]) for r in rows) / len(rows)


def _timestamp_tag() -> str:
    """Generate timestamp tag."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _model_tag(model_name: str) -> str:
    """Generate model tag for filenames."""
    import re
    s = (model_name or "").strip()
    if not s:
        return "model"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if s else "model"


def _dataset_tag(dataset: DatasetName) -> str:
    # Match quick_gsm8k_vllm.py naming.
    return "aime" if dataset == "aime25" else dataset


def _ids_tag(ids: list[int], *, max_ids: int = 5) -> str:
    if not ids:
        return ""
    if len(ids) <= max_ids:
        return "ids" + "-".join(str(i) for i in ids)
    head = "-".join(str(i) for i in ids[:max_ids])
    return f"ids{head}-plus{len(ids) - max_ids}"


def _range_tag(range_str: str | None) -> str:
    import re
    s = (range_str or "").strip()
    if not s:
        return ""
    return "range" + re.sub(r"[^0-9]+", "_", s).strip("_")


def _build_run_tag(*, tag: str | None, meta: dict[str, Any], subset_spec_tag: str, timestamp_tag: str) -> str:
    base = tag or f"n{meta['subset_size']}_seed{meta['seed']}"
    if subset_spec_tag:
        base = f"{base}_{subset_spec_tag}"
    return f"{base}_{timestamp_tag}"


def _default_out_dir(dataset: DatasetName) -> Path:
    return Path("/home/ubuntu/multi-agent-attack/results") / f"{dataset}_quick"


# =============================================================================
# Main CLI
# =============================================================================


def main() -> None:
    """Main CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Run multi-agent debate, majority voting, or single-response inference on GSM8K/AIME25/GPQA."
    )

    # Essential arguments
    ap.add_argument("--model_name", type=str, required=True, help="HuggingFace model ID.")
    ap.add_argument("--gpus", type=str, default="0", help="GPU IDs (comma-separated). Sets CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "aime25", "gpqa"], help="Dataset to evaluate on.")

    # Subset selection
    ap.add_argument(
        "--all",
        action="store_true",
        help="Run on the full dataset (overrides --subset_n/--subset_ids/--subset_range).",
    )
    ap.add_argument("--subset_n", type=_parse_subset_n_arg, default=20, help="Number of random samples (or 'all').")
    ap.add_argument("--subset_ids", type=str, default=None, help="Comma-separated specific indices.")
    ap.add_argument("--subset_range", type=str, default=None, help="Range like '0:10' or '0-9'.")
    ap.add_argument("--subset_seed", type=int, default=None, help="Random seed for subset sampling.")

    # Mode selection
    ap.add_argument("--mode", type=str, default="single,debate", help="Modes to run: single, majority, debate (comma-separated).")

    # Debate configuration
    ap.add_argument("--n_agents", type=int, default=3, help="Number of agents for debate.")
    ap.add_argument("--n_rounds", type=int, default=3, help="Number of debate rounds.")
    ap.add_argument("--majority_samples", type=int, default=5, help="Samples for majority voting.")
    ap.add_argument(
        "--debate_judge_rounds",
        type=str,
        default=None,
        help="Comma-separated round numbers to judge (e.g., '1,2,3'). If not specified, only judges final round.",
    )
    ap.add_argument(
        "--judge_block_size",
        type=int,
        default=None,
        help="Force a fixed judge block size (N rounds per judge prompt). If omitted, auto-selects the largest window that fits.",
    )
    # Judge sampling overrides (optional). If unset, inherits main sampling params.
    ap.add_argument("--judge_max_tokens", type=int, default=None, help="Optional max new tokens for judge output.")
    ap.add_argument("--judge_temperature", type=float, default=None, help="Optional judge temperature.")
    ap.add_argument("--judge_top_p", type=float, default=None, help="Optional judge top_p.")
    ap.add_argument("--judge_top_k", type=int, default=None, help="Optional judge top_k.")

    # vLLM configuration
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization (0.0-1.0).")
    ap.add_argument(
        "--enable_yarn",
        action="store_true",
        help="Enable YaRN RoPE scaling when --context_len exceeds the model's native context (also sets VLLM_ALLOW_LONG_MAX_MODEL_LEN=1).",
    )
    ap.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Run vLLM in eager mode (disables torch.compile/cudagraph). Useful for long-context stability.",
    )
    ap.add_argument(
        "--context_len",
        type=int,
        default=None,
        help="Fixed context length (applies to both main generation and judge). If omitted, uses adaptive context.",
    )
    ap.add_argument(
        "--max_model_len",
        dest="context_len",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # backwards-compatible alias
    )
    ap.add_argument("--batch_size", type=int, default=None, help="Max batch size for inference (default: auto).")

    ap.add_argument(
        "--quiet",
        "--silent",
        action="store_true",
        help="Silence vLLM/CUDA/etc logs; keep only progress bars, output paths, and final summary.",
    )

    # Output
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory.")
    ap.add_argument("--tag", type=str, default=None, help="Optional tag for output files.")

    args = ap.parse_args()

    with _QuietOutput(bool(args.quiet)) as q:
        progress_file: TextIO = q.keep_stdout if args.quiet else sys.stdout
        status_file: TextIO = q.keep_stdout if args.quiet else sys.stderr

        if args.quiet:
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

        # Set up environment
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        gpu_count = len([g.strip() for g in str(args.gpus).split(",") if g.strip()]) or 1

        def _auto_batch_size() -> int:
            # Tuned for "many prompts queued" workloads; OOM backoff will reduce if needed.
            if gpu_count >= 8:
                return 1024
            if gpu_count >= 4:
                return 512
            if gpu_count >= 2:
                return 256
            return 128

        batch_size = int(args.batch_size) if args.batch_size is not None else _auto_batch_size()

        # Parse modes
        modes: list[Mode] = []
        for m in args.mode.split(","):
            m = m.strip().lower()
            if m in ("single", "majority", "debate"):
                if m not in modes:
                    modes.append(cast(Mode, m))
        if not modes:
            print("No valid modes specified", file=status_file)
            sys.exit(1)

        dataset: DatasetName = cast(DatasetName, args.dataset)
        subset_seed = args.subset_seed if args.subset_seed is not None else random.SystemRandom().randint(0, 2**32 - 1)

        # Parse subset IDs if provided
        ids = None
        subset_range = args.subset_range
        if args.subset_ids:
            sids = str(args.subset_ids).strip()
            if sids.lower() in ("all", "*"):
                subset_range = "all"
            else:
                ids = [int(x.strip()) for x in sids.split(",") if x.strip()]

        # Allow "all" in --subset_range too (and a dedicated --all flag)
        if args.all or args.subset_n == "all":
            subset_range = "all"
            ids = None
        elif subset_range and str(subset_range).strip().lower() in ("all", "*"):
            subset_range = "all"
            ids = None

        # Get test path
        test_path = _default_dataset_test_path(dataset)

        # Create subset
        items, meta = _make_dataset_subset(
            dataset=dataset,
            test_path=test_path,
            n=0 if args.subset_n == "all" else int(args.subset_n),
            seed=subset_seed,
            ids=ids,
            range_str=subset_range,
        )
        if not args.quiet:
            print(f"[data] Subset: {len(items)} items from {dataset}", file=sys.stderr)

        # Build sampling config from model
        sampling_config = build_sampling_config(args.model_name)
        set_sampling_config(sampling_config)

        engine: InferenceEngine | None = None
        results: dict[str, list[dict[str, Any]]] = {}

        # Enable double Ctrl+C to cancel (prevents accidental termination)
        with _DoubleCtrlCHandler(timeout=2.0, output_file=status_file):
            try:
                # Create inference engine
                if not args.quiet:
                    print(f"[engine] Creating inference engine for {args.model_name}...", file=sys.stderr)
                engine = create_inference_engine(
                    model_name=args.model_name,
                    gpus=args.gpus,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.context_len,
                    enable_yarn=bool(args.enable_yarn),
                    enforce_eager=bool(args.enforce_eager),
                )

                # Output directory
                out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(dataset)
                out_dir.mkdir(parents=True, exist_ok=True)

                # Generate tags for output files (match quick_gsm8k_vllm.py)
                ts = _timestamp_tag()
                model_tag = _model_tag(args.model_name)
                dataset_tag = _dataset_tag(dataset)
                if subset_range and str(subset_range).strip().lower() == "all":
                    subset_spec_tag = "all"
                else:
                    subset_spec_tag = _ids_tag(ids) if ids else _range_tag(subset_range)
                run_tag = _build_run_tag(tag=args.tag, meta=meta, subset_spec_tag=subset_spec_tag, timestamp_tag=ts)

                for mode in modes:
                    results_by_round: dict[int, list[dict[str, Any]]] | None = None
                    judge_rounds: list[int] | None = None
                    if not args.quiet:
                        print(f"\n[run] Running {mode} mode...", file=sys.stderr)

                    if mode == "single":
                        records = run_sampled(
                            dataset=dataset,
                            items=items,
                            engine=engine,
                            n_samples=1,
                            batch_size=batch_size,
                            mode_label="single",
                            progress_file=progress_file,
                        )
                    elif mode == "majority":
                        records = run_sampled(
                            dataset=dataset,
                            items=items,
                            engine=engine,
                            n_samples=args.majority_samples,
                            batch_size=batch_size,
                            mode_label="majority",
                            progress_file=progress_file,
                        )
                    elif mode == "debate":
                        judge_rounds = _parse_judge_rounds(args.debate_judge_rounds, args.n_rounds)

                        judge_sampling_kwargs: dict[str, Any] | None = None
                        if (
                            args.judge_max_tokens is not None
                            or args.judge_temperature is not None
                            or args.judge_top_p is not None
                            or args.judge_top_k is not None
                        ):
                            main_sampling = sampling_config
                            judge_sampling_kwargs = {
                                "max_tokens": int(args.judge_max_tokens) if args.judge_max_tokens is not None else main_sampling.max_tokens,
                                "temperature": float(args.judge_temperature) if args.judge_temperature is not None else float(main_sampling.temperature),
                                "top_p": float(args.judge_top_p) if args.judge_top_p is not None else float(main_sampling.top_p),
                            }
                            jk = int(args.judge_top_k) if args.judge_top_k is not None else int(main_sampling.top_k)
                            if jk > 0:
                                judge_sampling_kwargs["top_k"] = int(jk)

                        results_by_round = run_debate(
                            dataset=dataset,
                            items=items,
                            engine=engine,
                            n_agents=args.n_agents,
                            n_rounds=args.n_rounds,
                            judge_rounds=judge_rounds,
                            batch_size=batch_size,
                            judge_block_size=args.judge_block_size,
                            judge_sampling_kwargs=judge_sampling_kwargs,
                            progress_file=progress_file,
                        )
                        max_round = max(judge_rounds) if judge_rounds else args.n_rounds
                        records = results_by_round[max_round]
                    else:
                        continue

                    results[mode] = records
                    if not args.quiet:
                        acc = _accuracy(records)
                        print(
                            f"[result] {mode}: {acc*100:.1f}% ({sum(r['final_correct'] for r in records)}/{len(records)})",
                            file=sys.stderr,
                        )

                    # Write results
                    if mode == "single":
                        out_path = out_dir / f"single_{dataset_tag}_{run_tag}_{model_tag}.jsonl"
                    elif mode == "majority":
                        out_path = out_dir / f"majority_{dataset_tag}_{run_tag}_{model_tag}.jsonl"
                    else:
                        assert results_by_round is not None
                        assert judge_rounds is not None
                        # If multiple judge rounds, write each round's file like quick_gsm8k_vllm.py.
                        if args.debate_judge_rounds is not None or (args.judge_block_size is not None and args.judge_block_size > 0):
                            for r in sorted(judge_rounds):
                                out_path_r = out_dir / f"debate_{dataset_tag}_agents{args.n_agents}_r{r}_{run_tag}_{model_tag}.jsonl"
                                _write_jsonl(out_path_r, results_by_round[r])
                                print(f"[output] Written to {out_path_r}", file=status_file)
                        # Always write the final judged round too (and keep out_path pointing at it).
                        final_r = args.n_rounds if args.debate_judge_rounds is None else max(judge_rounds)
                        out_path = out_dir / f"debate_{dataset_tag}_agents{args.n_agents}_r{final_r}_{run_tag}_{model_tag}.jsonl"

                    _write_jsonl(out_path, records)
                    print(f"[output] Written to {out_path}", file=status_file)

            finally:
                if engine is not None:
                    engine.shutdown()

        # Summary
        print("\n=== Summary ===", file=status_file)
        for mode, records in results.items():
            acc = _accuracy(records)
            print(f"{mode}: {acc*100:.1f}%", file=status_file)


if __name__ == "__main__":
    main()
