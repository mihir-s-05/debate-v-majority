"""
Repair targeted debate rows by reparsing / rerunning judge outputs.

This utility is designed for post-hoc fixes on specific JSONL rows identified by
`<path>:<orig_id>` target specs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .. import DatasetName
from ..cli import (
    JudgeParseResult,
    JUDGE_RETRY_NUDGE,
    _build_judge_context,
    _check_answer_correctness,
    _parse_judge_output,
)
from ..engines import (
    SamplingConfig,
    build_sampling_config,
    create_inference_engine,
    set_sampling_config,
)
from ..shared import (
    PromptTokenCounter,
    render_agent_assistant_rounds,
    strip_thinking_content,
)

_JUDGE_EXTRACT_FINAL_NUDGE_BY_DATASET: dict[str, str] = {
    "gpqa": (
        "You are given prior judge output that may be verbose or truncated.\n"
        "Output ONLY one final choice in this exact format: \\boxed{A} (one of A, B, C, D).\n"
        "Do not output any other text."
    ),
    "aime25": (
        "You are given prior judge output that may be verbose or truncated.\n"
        "Output ONLY one final integer answer in this exact format: \\boxed{N} where N is an integer from 0 to 999.\n"
        "N must be within 0..999. If a computed value is outside this range, convert it modulo 1000 first.\n"
        "Do not output any other text."
    ),
    "gsm8k": (
        "You are given prior judge output that may be verbose or truncated.\n"
        "Output ONLY one final numeric answer in this exact format: \\boxed{N} where N is a number.\n"
        "Do not output any other text."
    ),
}
JUDGE_EXTRACT_FINAL_NUDGE = _JUDGE_EXTRACT_FINAL_NUDGE_BY_DATASET["gpqa"]


def _get_extract_nudge(dataset: DatasetName) -> str:
    return _JUDGE_EXTRACT_FINAL_NUDGE_BY_DATASET.get(str(dataset), JUDGE_EXTRACT_FINAL_NUDGE)


_RETRY_STRONGER_NUDGE_BY_DATASET: dict[str, str] = {
    "gpqa": (
        "\nReturn exactly one final choice in this format only: \\boxed{A}."
        " Output only that boxed choice and nothing else."
    ),
    "aime25": (
        "\nReturn exactly one final integer answer in this format only: \\boxed{N} where N is 0..999."
        " Output only that boxed integer and nothing else."
        " If your computed value is outside 0..999, reduce it modulo 1000 first."
        " Example: if it is 1524, output \\boxed{524}."
    ),
    "gsm8k": "\nReturn exactly one final numeric answer in this format only: \\boxed{N}.",
}

RAW_ERROR_PREFIX = "raw_error"
RETRY_ERROR_PREFIX = "retry_error"
EXTRACT_ERROR_PREFIX = "extract_error"
EXTRACT_SAMPLING_KWARGS = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512}


def _get_retry_stronger_nudge(dataset: DatasetName) -> str:
    return _RETRY_STRONGER_NUDGE_BY_DATASET.get(
        str(dataset), _RETRY_STRONGER_NUDGE_BY_DATASET["gpqa"]
    )


@dataclass(frozen=True)
class TargetSpec:
    path: Path
    orig_id: int


@dataclass
class JudgeAttempt:
    judged_answer: str | None
    raw_output: str
    retry_output: str | None
    parse_failed: bool
    used_fallback: bool
    retry_used: bool
    parse_mode: str
    parse_source: str
    retry_reason: str | None
    finish_state: str
    raw_had_strict_final: bool
    retry_had_strict_final: bool
    judge_context: list[dict[str, str]]


@dataclass
class RowUpdateResult:
    path: Path
    orig_id: int
    action: str
    before: Any
    after: Any
    changed: bool
    details: str = ""


@dataclass
class RerunRequest:
    path: Path
    orig_id: int
    row_idx: int
    row: dict[str, Any]
    before: Any
    model_name: str


@dataclass
class PreparedRerunMeta:
    req: RerunRequest
    raw_task: dict[str, Any]
    judge_context: list[dict[str, str]]


@dataclass
class RowPlan:
    immediate_result: RowUpdateResult | None
    rerun_request: RerunRequest | None


class EngineManager:
    def __init__(
        self,
        *,
        gpus: str,
        gpu_memory_utilization: float,
        context_len: int | None,
        enable_yarn: bool,
        enforce_eager: bool,
        judge_overrides: dict[str, Any] | None,
    ) -> None:
        self._gpus = gpus
        self._gpu_memory_utilization = gpu_memory_utilization
        self._context_len = context_len
        self._enable_yarn = enable_yarn
        self._enforce_eager = enforce_eager
        self._judge_overrides = judge_overrides
        self._engines: dict[str, Any] = {}
        self._sampling_cfg_by_model: dict[str, SamplingConfig] = {}

    def close(self) -> None:
        for eng in self._engines.values():
            try:
                eng.shutdown()
            except Exception:
                pass
        self._engines.clear()

    def _get_sampling_cfg(self, model_name: str) -> SamplingConfig:
        cfg = self._sampling_cfg_by_model.get(model_name)
        if cfg is not None:
            return cfg
        cfg = build_sampling_config(model_name)
        self._sampling_cfg_by_model[model_name] = cfg
        return cfg

    def get_engine(self, model_name: str):
        eng = self._engines.get(model_name)
        if eng is not None:
            return eng
        cfg = self._get_sampling_cfg(model_name)
        set_sampling_config(cfg)
        eng = create_inference_engine(
            model_name=model_name,
            gpus=self._gpus,
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_model_len=self._context_len,
            enable_yarn=self._enable_yarn,
            enforce_eager=self._enforce_eager,
        )
        self._engines[model_name] = eng
        return eng

    def judge_sampling_kwargs(self, model_name: str) -> dict[str, Any] | None:
        if self._judge_overrides is None:
            return None
        cfg = self._get_sampling_cfg(model_name)
        out: dict[str, Any] = {
            "max_tokens": int(self._judge_overrides["max_tokens"])
            if self._judge_overrides.get("max_tokens") is not None
            else cfg.max_tokens,
            "temperature": float(self._judge_overrides["temperature"])
            if self._judge_overrides.get("temperature") is not None
            else float(cfg.temperature),
            "top_p": float(self._judge_overrides["top_p"])
            if self._judge_overrides.get("top_p") is not None
            else float(cfg.top_p),
        }
        tk = (
            int(self._judge_overrides["top_k"])
            if self._judge_overrides.get("top_k") is not None
            else int(cfg.top_k)
        )
        if tk > 0:
            out["top_k"] = int(tk)
        return out

    def judge_max_new_tokens(self, model_name: str, sampling_kwargs: dict[str, Any] | None) -> int:
        if sampling_kwargs and sampling_kwargs.get("max_tokens") is not None:
            return int(sampling_kwargs["max_tokens"])
        cfg = self._get_sampling_cfg(model_name)
        return int(cfg.max_tokens or 4096)

    def context_len_tokens(self, model_name: str) -> int | None:
        eng = self._engines.get(model_name)
        if eng is not None and hasattr(eng, "context_len_tokens"):
            return int(eng.context_len_tokens)
        return self._context_len

    def get_token_counter(self, model_name: str) -> PromptTokenCounter:
        return PromptTokenCounter(model_name)


def _parse_target(value: str) -> TargetSpec:
    s = str(value).strip()
    m = re.match(r"^(.*):(-?\d+)$", s)
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid target {value!r}. Expected format '<path>:<orig_id>'."
        )
    path_s, id_s = m.group(1).strip(), m.group(2).strip()
    if not path_s:
        raise argparse.ArgumentTypeError(f"Invalid target {value!r}: empty path.")
    p = Path(path_s).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return TargetSpec(path=p, orig_id=int(id_s))


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_row_has_fields(row: dict[str, Any]) -> None:
    for key in ("raw_task", "answer", "question", "agent_responses"):
        if key not in row:
            raise ValueError(f"Row missing required key: {key}")
    if "judge_trace" not in row or not isinstance(row["judge_trace"], dict):
        raise ValueError("Row missing required dict key: judge_trace")


def _is_empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _parse_judge_text(
    *,
    dataset: DatasetName,
    text: Any,
    raw_task: dict[str, Any],
    source_prefix: str,
    strict_enabled: bool = True,
    recovery_enabled: bool = True,
):
    parsed = _parse_judge_output(
        dataset=dataset,
        text=text,
        raw_task=raw_task,
        source_prefix=source_prefix,
        strict_enabled=strict_enabled,
        recovery_enabled=recovery_enabled,
    )
    if parsed.answer is not None or not recovery_enabled or dataset != "gpqa":
        return parsed

    recovered = _recover_gpqa_reasoning_choice(text)
    if recovered is None:
        return parsed
    return JudgeParseResult(
        answer=recovered,
        mode="recover",
        source=f"{source_prefix}_reasoning_recovery",
        strict_success=False,
    )


def _recover_gpqa_reasoning_choice(text: Any) -> str | None:
    t = str(text or "")
    if not t.strip():
        return None

    pats = [
        re.compile(
            r"(?i)\b(?:closest(?:\s+option)?(?:\s+is)?|go\s+with|choose|pick|select)\b"
            r"[^A-D]{0,36}(?:option\s*)?\(?\s*([ABCD])\s*\)?"
        ),
        re.compile(
            r"(?i)\b(?:matches?|corresponds?\s+to)\b[^A-D]{0,36}(?:option\s*)?\(?\s*([ABCD])\s*\)?"
        ),
    ]
    for pat in pats:
        m = None
        for m in pat.finditer(t):
            pass
        if m:
            return m.group(1).upper()
    return None


def _stored_final_is_valid(row: dict[str, Any], *, dataset: DatasetName) -> bool:
    raw_task = cast(dict[str, Any], row.get("raw_task") or {})
    jt = cast(dict[str, Any], row.get("judge_trace") or {})
    final_judge = row.get("final_judge_answer")
    if _is_empty(final_judge):
        return False

    mode = str(jt.get("judge_parse_mode") or "").strip().lower()
    parse_failed = bool(jt.get("judge_parse_failed"))
    if mode == "strict" and not parse_failed:
        parsed_answer = jt.get("judge_parsed_answer")
        if not _is_empty(parsed_answer) and str(parsed_answer) == str(final_judge):
            return True

    raw_out = jt.get("judge_raw_response")
    retry_out = jt.get("judge_retry_raw_response")
    for prefix, txt in (("raw", raw_out), ("retry", retry_out)):
        parsed = _parse_judge_text(
            dataset=dataset,
            text=txt,
            raw_task=raw_task,
            source_prefix=prefix,
            strict_enabled=True,
            recovery_enabled=False,
        )
        if parsed.answer is not None and str(parsed.answer) == str(final_judge):
            return True
    return False


def _try_reparse_existing(
    row: dict[str, Any], *, dataset: DatasetName
) -> tuple[str | None, str, str, bool, bool]:
    raw_task = cast(dict[str, Any], row["raw_task"])
    jt = cast(dict[str, Any], row.get("judge_trace") or {})
    raw_out = jt.get("judge_raw_response")
    retry_out = jt.get("judge_retry_raw_response")

    raw_strict = _parse_judge_text(
        dataset=dataset,
        text=raw_out,
        raw_task=raw_task,
        source_prefix="raw",
        strict_enabled=True,
        recovery_enabled=False,
    )
    raw_had_strict = bool(raw_strict.strict_success)
    if raw_strict.answer is not None:
        return raw_strict.answer, raw_strict.mode, raw_strict.source, raw_had_strict, False

    retry_had_strict = False
    if retry_out is not None:
        retry_strict = _parse_judge_text(
            dataset=dataset,
            text=retry_out,
            raw_task=raw_task,
            source_prefix="retry",
            strict_enabled=True,
            recovery_enabled=False,
        )
        retry_had_strict = bool(retry_strict.strict_success)
        if retry_strict.answer is not None:
            return retry_strict.answer, retry_strict.mode, retry_strict.source, raw_had_strict, retry_had_strict

    return None, "none", "none", raw_had_strict, retry_had_strict


def _needs_retry(row: dict[str, Any], *, dataset: DatasetName) -> bool:
    jt = cast(dict[str, Any], row.get("judge_trace") or {})
    if bool(jt.get("judge_parse_failed")):
        return True
    if str(jt.get("judge_parse_mode") or "").strip().lower() == "recover":
        return True
    return not _stored_final_is_valid(row, dataset=dataset)


def _strip_thinking_from_agent_responses(
    agent_responses: list[list[dict[str, str]]],
) -> list[list[dict[str, str]]]:
    stripped: list[list[dict[str, str]]] = []
    for ctx in agent_responses:
        new_ctx: list[dict[str, str]] = []
        for msg in ctx:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                new_content = strip_thinking_content(content)
                new_ctx.append({**msg, "content": new_content})
            else:
                new_ctx.append(msg)
        stripped.append(new_ctx)
    return stripped


def _rebuild_judge_context(
    row: dict[str, Any],
    *,
    dataset: DatasetName,
    context_len_tokens: int | None = None,
    max_new_tokens: int | None = None,
    counter: PromptTokenCounter | None = None,
) -> list[dict[str, str]]:
    question = str(row["question"])
    agent_responses = row["agent_responses"]
    if not isinstance(agent_responses, list) or not agent_responses:
        raise ValueError("Missing/invalid agent_responses; cannot rebuild judge context.")

    n_rounds = int(row.get("n_rounds") or 0)
    if n_rounds <= 0:
        n_rounds = 1000

    transcripts = [
        render_agent_assistant_rounds(agent_conv=ctx, start_round=1, end_round=n_rounds)
        for ctx in agent_responses
    ]
    msgs = _build_judge_context(
        dataset=dataset, question=question, responses=transcripts, previous_judge=None,
    )

    if context_len_tokens and max_new_tokens and counter:
        budget = max(1, int(context_len_tokens) - int(max_new_tokens) - 256)
        try:
            n_tokens = counter.count_chat_tokens(msgs)
        except Exception:
            n_tokens = int(counter.estimate_prompt_tokens(msgs, exact_if_large=1) * 1.1)

        if n_tokens > budget:
            stripped_responses = _strip_thinking_from_agent_responses(agent_responses)
            transcripts = [
                render_agent_assistant_rounds(agent_conv=ctx, start_round=1, end_round=n_rounds)
                for ctx in stripped_responses
            ]
            msgs = _build_judge_context(
                dataset=dataset, question=question, responses=transcripts, previous_judge=None,
            )
            try:
                n_tokens_after = counter.count_chat_tokens(msgs)
            except Exception:
                n_tokens_after = int(counter.estimate_prompt_tokens(msgs, exact_if_large=1) * 1.1)
            print(
                f"  [adaptive] stripped thinking blocks from agent transcripts "
                f"({n_tokens} -> {n_tokens_after} tokens, budget {budget})",
                file=sys.stderr,
            )

    return msgs


def _build_judge_attempt(
    *,
    judged_answer: str | None,
    raw_output: str,
    retry_output: str | None,
    parse_failed: bool,
    used_fallback: bool,
    retry_used: bool,
    parse_mode: str,
    parse_source: str,
    retry_reason: str | None,
    finish_state: str,
    raw_had_strict_final: bool,
    retry_had_strict_final: bool,
    judge_context: list[dict[str, str]],
) -> JudgeAttempt:
    return JudgeAttempt(
        judged_answer=judged_answer,
        raw_output=raw_output,
        retry_output=retry_output,
        parse_failed=parse_failed,
        used_fallback=used_fallback,
        retry_used=retry_used,
        parse_mode=parse_mode,
        parse_source=parse_source,
        retry_reason=retry_reason,
        finish_state=finish_state,
        raw_had_strict_final=raw_had_strict_final,
        retry_had_strict_final=retry_had_strict_final,
        judge_context=judge_context,
    )


def _prepare_rerun_metas(
    *,
    dataset: DatasetName,
    model_name: str,
    model_reqs: list[RerunRequest],
    engine_mgr: EngineManager,
    judge_max_new_tokens: int,
) -> list[PreparedRerunMeta]:
    ctx_len = engine_mgr.context_len_tokens(model_name)
    token_counter = engine_mgr.get_token_counter(model_name)
    metas: list[PreparedRerunMeta] = []

    for req in model_reqs:
        raw_task = cast(dict[str, Any], req.row["raw_task"])
        try:
            judge_context = _rebuild_judge_context(
                req.row,
                dataset=dataset,
                context_len_tokens=ctx_len,
                max_new_tokens=judge_max_new_tokens,
                counter=token_counter,
            )
        except Exception:
            jt = cast(dict[str, Any], req.row.get("judge_trace") or {})
            stored_ctx = jt.get("judge_context")
            if not isinstance(stored_ctx, list) or not stored_ctx:
                raise
            judge_context = cast(list[dict[str, str]], stored_ctx)
        metas.append(PreparedRerunMeta(req=req, raw_task=raw_task, judge_context=judge_context))

    return metas


def _build_retry_sampling_kwargs(
    *,
    sampling_kwargs: dict[str, Any] | None,
    judge_max_new_tokens: int,
    judge_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retry_sampling = dict(sampling_kwargs or {})
    temp_overridden = bool(judge_overrides and judge_overrides.get("temperature") is not None)
    top_p_overridden = bool(judge_overrides and judge_overrides.get("top_p") is not None)
    if not temp_overridden:
        retry_sampling["temperature"] = 0.0
    if not top_p_overridden:
        retry_sampling["top_p"] = 1.0
    retry_sampling["max_tokens"] = int(judge_max_new_tokens)
    return retry_sampling


def _run_raw_pass_for_model(
    *,
    dataset: DatasetName,
    metas: list[PreparedRerunMeta],
    raw_outputs: list[str],
    attempts: dict[tuple[Path, int], JudgeAttempt],
) -> tuple[list[int], list[bool]]:
    retry_pending: list[int] = []
    retry_raw_had_strict: list[bool] = [False for _ in metas]

    for i, (meta, raw_output_s) in enumerate(zip(metas, raw_outputs)):
        req = meta.req
        key = (req.path, req.orig_id)
        if raw_output_s.startswith(f"[{RAW_ERROR_PREFIX}]"):
            attempts[key] = _build_judge_attempt(
                judged_answer=None,
                raw_output=raw_output_s,
                retry_output=None,
                parse_failed=True,
                used_fallback=False,
                retry_used=False,
                parse_mode="none",
                parse_source="none",
                retry_reason="raw_generation_error",
                finish_state="raw_error",
                raw_had_strict_final=False,
                retry_had_strict_final=False,
                judge_context=meta.judge_context,
            )
            continue

        raw_strict = _parse_judge_text(
            dataset=dataset,
            text=raw_output_s,
            raw_task=meta.raw_task,
            source_prefix="raw",
            strict_enabled=True,
            recovery_enabled=False,
        )
        raw_had_strict_final = bool(raw_strict.strict_success)
        retry_raw_had_strict[i] = raw_had_strict_final
        parsed_raw = _parse_judge_text(
            dataset=dataset,
            text=raw_output_s,
            raw_task=meta.raw_task,
            source_prefix="raw",
            strict_enabled=True,
            recovery_enabled=True,
        )
        if parsed_raw.answer is not None and parsed_raw.mode != "recover":
            attempts[key] = _build_judge_attempt(
                judged_answer=parsed_raw.answer,
                raw_output=raw_output_s,
                retry_output=None,
                parse_failed=False,
                used_fallback=False,
                retry_used=False,
                parse_mode=parsed_raw.mode,
                parse_source=parsed_raw.source,
                retry_reason=None,
                finish_state="raw_parsed",
                raw_had_strict_final=raw_had_strict_final,
                retry_had_strict_final=False,
                judge_context=meta.judge_context,
            )
            continue

        retry_pending.append(i)

    return retry_pending, retry_raw_had_strict


def _run_retry_passes_for_model(
    *,
    dataset: DatasetName,
    engine: Any,
    metas: list[PreparedRerunMeta],
    raw_outputs: list[str],
    retry_pending: list[int],
    batch_size: int | None,
    retry_sampling: dict[str, Any],
) -> dict[int, str]:
    retry_ctx1: list[list[dict[str, str]]] = []
    idx1: list[int] = []
    for i in retry_pending:
        retry_ctx1.append(
            list(metas[i].judge_context)
            + [
                {"role": "assistant", "content": raw_outputs[i]},
                {"role": "user", "content": JUDGE_RETRY_NUDGE},
            ]
        )
        idx1.append(i)

    retry_out1 = _generate_batch_texts(
        engine=engine,
        contexts=retry_ctx1,
        batch_size=batch_size,
        sampling_kwargs=retry_sampling,
        error_prefix=RETRY_ERROR_PREFIX,
    )

    retry_pending2: list[int] = []
    retry_output_by_meta: dict[int, str] = {}
    for i, out_s in zip(idx1, retry_out1):
        retry_output_by_meta[i] = out_s
        if out_s.startswith(f"[{RETRY_ERROR_PREFIX}]"):
            continue
        parsed_probe = _parse_judge_text(
            dataset=dataset,
            text=out_s,
            raw_task=metas[i].raw_task,
            source_prefix="retry",
            strict_enabled=True,
            recovery_enabled=False,
        )
        if parsed_probe.answer is None:
            retry_pending2.append(i)

    retry_ctx2: list[list[dict[str, str]]] = []
    idx2: list[int] = []
    for i in retry_pending2:
        retry_ctx2.append(
            list(metas[i].judge_context)
            + [
                {"role": "assistant", "content": raw_outputs[i]},
                {
                    "role": "user",
                    "content": JUDGE_RETRY_NUDGE + _get_retry_stronger_nudge(dataset),
                }
            ]
        )
        idx2.append(i)

    retry_out2 = _generate_batch_texts(
        engine=engine,
        contexts=retry_ctx2,
        batch_size=batch_size,
        sampling_kwargs=retry_sampling,
        error_prefix=RETRY_ERROR_PREFIX,
    )
    for i, out_s in zip(idx2, retry_out2):
        retry_output_by_meta[i] = out_s
    return retry_output_by_meta


def _finalize_retry_pass_for_model(
    *,
    dataset: DatasetName,
    metas: list[PreparedRerunMeta],
    raw_outputs: list[str],
    retry_pending: list[int],
    retry_output_by_meta: dict[int, str],
    retry_raw_had_strict: list[bool],
    attempts: dict[tuple[Path, int], JudgeAttempt],
) -> list[int]:
    extract_pending: list[int] = []
    for i in retry_pending:
        meta = metas[i]
        req = meta.req
        key = (req.path, req.orig_id)
        raw_output_s = raw_outputs[i]
        retry_output = retry_output_by_meta.get(i) or f"[{RETRY_ERROR_PREFIX}] unknown"

        if retry_output.startswith(f"[{RETRY_ERROR_PREFIX}]"):
            attempts[key] = _build_judge_attempt(
                judged_answer=None,
                raw_output=raw_output_s,
                retry_output=retry_output,
                parse_failed=True,
                used_fallback=False,
                retry_used=True,
                parse_mode="none",
                parse_source="none",
                retry_reason="retry_generation_error",
                finish_state="retry_error",
                raw_had_strict_final=retry_raw_had_strict[i],
                retry_had_strict_final=False,
                judge_context=meta.judge_context,
            )
            continue

        retry_strict = _parse_judge_text(
            dataset=dataset,
            text=retry_output,
            raw_task=meta.raw_task,
            source_prefix="retry",
            strict_enabled=True,
            recovery_enabled=False,
        )
        retry_had_strict_final = bool(retry_strict.strict_success)
        parsed_retry = _parse_judge_text(
            dataset=dataset,
            text=retry_output,
            raw_task=meta.raw_task,
            source_prefix="retry",
            strict_enabled=True,
            recovery_enabled=True,
        )
        judged = parsed_retry.answer
        if judged is None or parsed_retry.mode == "recover":
            extract_pending.append(i)
            continue

        attempts[key] = _build_judge_attempt(
            judged_answer=judged,
            raw_output=raw_output_s,
            retry_output=retry_output,
            parse_failed=False,
            used_fallback=False,
            retry_used=True,
            parse_mode=parsed_retry.mode,
            parse_source=parsed_retry.source,
            retry_reason="parse_none",
            finish_state="retry_parsed",
            raw_had_strict_final=retry_raw_had_strict[i],
            retry_had_strict_final=retry_had_strict_final,
            judge_context=meta.judge_context,
        )

    return extract_pending


def _finalize_extract_pass_for_model(
    *,
    dataset: DatasetName,
    engine: Any,
    metas: list[PreparedRerunMeta],
    raw_outputs: list[str],
    extract_pending: list[int],
    retry_output_by_meta: dict[int, str],
    retry_raw_had_strict: list[bool],
    batch_size: int | None,
    attempts: dict[tuple[Path, int], JudgeAttempt],
) -> None:
    if not extract_pending:
        return

    retry_only_ctxs: list[list[dict[str, str]]] = []
    retry_only_idx: list[int] = []
    unresolved: list[int] = []
    for i in extract_pending:
        retry_output = retry_output_by_meta.get(i) or ""
        if retry_output:
            retry_only_ctxs.append(
                _build_extract_retry_only_context(retry_output=retry_output, dataset=dataset)
            )
            retry_only_idx.append(i)
        else:
            unresolved.append(i)

    retry_only_out = _generate_batch_texts(
        engine=engine,
        contexts=retry_only_ctxs,
        batch_size=batch_size,
        sampling_kwargs=dict(EXTRACT_SAMPLING_KWARGS),
        error_prefix=EXTRACT_ERROR_PREFIX,
    )

    for i, out_s in zip(retry_only_idx, retry_only_out):
        meta = metas[i]
        req = meta.req
        key = (req.path, req.orig_id)
        raw_output_s = raw_outputs[i]
        retry_output = retry_output_by_meta.get(i) or ""
        if out_s.startswith(f"[{EXTRACT_ERROR_PREFIX}]"):
            unresolved.append(i)
            continue

        parsed_extract = _parse_judge_text(
            dataset=dataset,
            text=out_s,
            raw_task=meta.raw_task,
            source_prefix="extract_retry_only",
            strict_enabled=True,
            recovery_enabled=False,
        )
        if parsed_extract.answer is None:
            unresolved.append(i)
            continue

        attempts[key] = _build_judge_attempt(
            judged_answer=parsed_extract.answer,
            raw_output=raw_output_s,
            retry_output=retry_output,
            parse_failed=False,
            used_fallback=False,
            retry_used=True,
            parse_mode=parsed_extract.mode,
            parse_source=parsed_extract.source,
            retry_reason="parse_none_then_extract_retry_only",
            finish_state="extract_parsed",
            raw_had_strict_final=retry_raw_had_strict[i],
            retry_had_strict_final=False,
            judge_context=meta.judge_context,
        )

    extract_ctxs: list[list[dict[str, str]]] = []
    extract_idx: list[int] = []
    for i in unresolved:
        raw_output_s = raw_outputs[i]
        retry_output = retry_output_by_meta.get(i) or ""
        extract_ctxs.append(
            _build_extract_context(raw_output=raw_output_s, retry_output=retry_output, dataset=dataset)
        )
        extract_idx.append(i)

    extract_out = _generate_batch_texts(
        engine=engine,
        contexts=extract_ctxs,
        batch_size=batch_size,
        sampling_kwargs=dict(EXTRACT_SAMPLING_KWARGS),
        error_prefix=EXTRACT_ERROR_PREFIX,
    )

    for i, out_s in zip(extract_idx, extract_out):
        meta = metas[i]
        req = meta.req
        key = (req.path, req.orig_id)
        raw_output_s = raw_outputs[i]
        retry_output = retry_output_by_meta.get(i) or ""
        if out_s.startswith(f"[{EXTRACT_ERROR_PREFIX}]"):
            attempts[key] = _build_judge_attempt(
                judged_answer=None,
                raw_output=raw_output_s,
                retry_output=retry_output,
                parse_failed=True,
                used_fallback=False,
                retry_used=True,
                parse_mode="none",
                parse_source="none",
                retry_reason="parse_none",
                finish_state="retry_unparsed",
                raw_had_strict_final=retry_raw_had_strict[i],
                retry_had_strict_final=False,
                judge_context=meta.judge_context,
            )
            continue

        parsed_extract = _parse_judge_text(
            dataset=dataset,
            text=out_s,
            raw_task=meta.raw_task,
            source_prefix="extract",
            strict_enabled=True,
            recovery_enabled=False,
        )
        if parsed_extract.answer is None:
            attempts[key] = _build_judge_attempt(
                judged_answer=None,
                raw_output=raw_output_s,
                retry_output=retry_output,
                parse_failed=True,
                used_fallback=False,
                retry_used=True,
                parse_mode="none",
                parse_source="none",
                retry_reason="parse_none",
                finish_state="retry_unparsed",
                raw_had_strict_final=retry_raw_had_strict[i],
                retry_had_strict_final=False,
                judge_context=meta.judge_context,
            )
            continue

        attempts[key] = _build_judge_attempt(
            judged_answer=parsed_extract.answer,
            raw_output=raw_output_s,
            retry_output=retry_output,
            parse_failed=False,
            used_fallback=False,
            retry_used=True,
            parse_mode=parsed_extract.mode,
            parse_source=parsed_extract.source,
            retry_reason="parse_none_then_extract",
            finish_state="extract_parsed",
            raw_had_strict_final=retry_raw_had_strict[i],
            retry_had_strict_final=False,
            judge_context=meta.judge_context,
        )


def _build_extract_retry_only_context(
    *, retry_output: str, dataset: DatasetName | None = None,
) -> list[dict[str, str]]:
    nudge = _get_extract_nudge(dataset) if dataset else JUDGE_EXTRACT_FINAL_NUDGE
    system_desc = (
        "You extract a final answer from prior judge output."
        if dataset and str(dataset) != "gpqa"
        else "You extract a final multiple-choice answer from prior judge output."
    )
    return [
        {"role": "system", "content": system_desc},
        {"role": "user", "content": nudge + "\n\nPrior judge output:\n" + str(retry_output or "")},
    ]


def _build_extract_context(
    *, raw_output: str, retry_output: str | None, dataset: DatasetName | None = None,
) -> list[dict[str, str]]:
    chunks = [str(raw_output or "")]
    if retry_output:
        chunks.append(str(retry_output))
    payload = "\n\n---\n\n".join(chunks)
    nudge = _get_extract_nudge(dataset) if dataset else JUDGE_EXTRACT_FINAL_NUDGE
    system_desc = (
        "You extract a final answer."
        if dataset and str(dataset) != "gpqa"
        else "You extract a final multiple-choice answer."
    )
    return [
        {"role": "system", "content": system_desc},
        {"role": "user", "content": nudge + "\n\nPrior judge output:\n" + payload},
    ]


def _generate_batch_texts(
    *,
    engine: Any,
    contexts: list[list[dict[str, str]]],
    batch_size: int | None,
    sampling_kwargs: dict[str, Any] | None,
    error_prefix: str,
) -> list[str]:
    if not contexts:
        return []
    try:
        return [str(x) for x in engine.generate_batch(contexts, batch_size=batch_size, sampling_kwargs=sampling_kwargs)]
    except Exception as e:
        err = f"[{error_prefix}] {type(e).__name__}: {e}"
        out: list[str] = []
        for ctx in contexts:
            try:
                txt = engine.generate_batch(
                    [ctx],
                    batch_size=1,
                    sampling_kwargs=sampling_kwargs,
                )[0]
                out.append(str(txt))
            except Exception as e2:
                out.append(f"[{error_prefix}] {type(e2).__name__}: {e2}")
        if not out:
            out.append(err)
        return out


def _run_judge_with_retry_batched(
    *,
    dataset: DatasetName,
    requests: list[RerunRequest],
    engine_mgr: EngineManager,
    batch_size: int | None,
) -> dict[tuple[Path, int], JudgeAttempt]:
    attempts: dict[tuple[Path, int], JudgeAttempt] = {}
    by_model: dict[str, list[RerunRequest]] = {}
    for req in requests:
        by_model.setdefault(req.model_name, []).append(req)

    for model_name, model_reqs in by_model.items():
        engine = engine_mgr.get_engine(model_name)
        sampling_kwargs = engine_mgr.judge_sampling_kwargs(model_name)
        judge_max_new_tokens = engine_mgr.judge_max_new_tokens(model_name, sampling_kwargs)
        metas = _prepare_rerun_metas(
            dataset=dataset,
            model_name=model_name,
            model_reqs=model_reqs,
            engine_mgr=engine_mgr,
            judge_max_new_tokens=judge_max_new_tokens,
        )
        raw_contexts = [m.judge_context for m in metas]
        raw_outputs = _generate_batch_texts(
            engine=engine,
            contexts=raw_contexts,
            batch_size=batch_size,
            sampling_kwargs=sampling_kwargs,
            error_prefix=RAW_ERROR_PREFIX,
        )
        retry_sampling = _build_retry_sampling_kwargs(
            sampling_kwargs=sampling_kwargs,
            judge_max_new_tokens=judge_max_new_tokens,
            judge_overrides=getattr(engine_mgr, "_judge_overrides", None),
        )
        retry_pending, retry_raw_had_strict = _run_raw_pass_for_model(
            dataset=dataset,
            metas=metas,
            raw_outputs=raw_outputs,
            attempts=attempts,
        )
        retry_output_by_meta = _run_retry_passes_for_model(
            dataset=dataset,
            engine=engine,
            metas=metas,
            raw_outputs=raw_outputs,
            retry_pending=retry_pending,
            batch_size=batch_size,
            retry_sampling=retry_sampling,
        )
        extract_pending = _finalize_retry_pass_for_model(
            dataset=dataset,
            metas=metas,
            raw_outputs=raw_outputs,
            retry_pending=retry_pending,
            retry_output_by_meta=retry_output_by_meta,
            retry_raw_had_strict=retry_raw_had_strict,
            attempts=attempts,
        )
        _finalize_extract_pass_for_model(
            dataset=dataset,
            engine=engine,
            metas=metas,
            raw_outputs=raw_outputs,
            extract_pending=extract_pending,
            retry_output_by_meta=retry_output_by_meta,
            retry_raw_had_strict=retry_raw_had_strict,
            batch_size=batch_size,
            attempts=attempts,
        )

    return attempts


def _apply_answer_update(
    *,
    dataset: DatasetName,
    row: dict[str, Any],
    judged_answer: str | None,
    parse_failed: bool,
    used_fallback: bool,
    parse_mode: str,
    parse_source: str,
    retry_reason: str | None,
    finish_state: str,
    raw_had_strict_final: bool,
    retry_had_strict_final: bool,
    judge_raw_output: str | None,
    judge_retry_output: str | None,
    judge_context: list[dict[str, str]] | None,
    rerun: bool,
) -> None:
    jt = cast(dict[str, Any], row.setdefault("judge_trace", {}))

    if rerun:
        if judge_context is not None:
            jt["judge_context"] = judge_context
        if judge_raw_output is not None:
            jt["judge_raw_response"] = judge_raw_output
        jt["judge_retry_raw_response"] = judge_retry_output

    gt = row.get("answer")
    final_judge_correct = _check_answer_correctness(dataset, judged_answer, gt)

    jt["judge_parsed_answer"] = judged_answer
    jt["judge_parse_failed"] = bool(parse_failed)
    jt["judge_used_fallback"] = bool(used_fallback)
    jt["judge_parse_mode"] = str(parse_mode or "none")
    jt["judge_parse_source"] = str(parse_source or "none")
    jt["judge_retry_reason"] = retry_reason
    jt["judge_finish_state"] = str(finish_state or "none")
    jt["judge_raw_had_strict_final"] = bool(raw_had_strict_final)
    jt["judge_retry_had_strict_final"] = bool(retry_had_strict_final)
    jt["judge_correct"] = int(final_judge_correct)

    row["final_judge_answer"] = judged_answer
    row["final_judge_correct"] = int(final_judge_correct)
    row["final_answer"] = judged_answer
    row["final_correct"] = int(final_judge_correct)


def _read_jsonl_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _write_jsonl_lines_atomic(path: Path, lines: list[str]) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for ln in lines:
            f.write(ln)
            f.write("\n")
    tmp.replace(path)


def _parse_judge_override_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if (
        args.judge_max_tokens is None
        and args.judge_temperature is None
        and args.judge_top_p is None
        and args.judge_top_k is None
    ):
        return None
    return {
        "max_tokens": args.judge_max_tokens,
        "temperature": args.judge_temperature,
        "top_p": args.judge_top_p,
        "top_k": args.judge_top_k,
    }


def _extract_only_existing_result(
    *,
    dataset: DatasetName,
    path: Path,
    row: dict[str, Any],
    orig_id: int,
    engine_mgr: EngineManager,
) -> RowUpdateResult:
    before = row.get("final_judge_answer")
    jt = cast(dict[str, Any], row.get("judge_trace") or {})
    model_name = str(jt.get("judge_model") or "").strip()
    if not model_name:
        return RowUpdateResult(
            path=path,
            orig_id=orig_id,
            action="extract_only_failed",
            before=before,
            after=before,
            changed=False,
            details="Missing judge_trace.judge_model; cannot run extraction-only pass.",
        )

    raw_output = str(jt.get("judge_raw_response") or "")
    retry_output = str(jt.get("judge_retry_raw_response") or "")
    if not raw_output and not retry_output:
        return RowUpdateResult(
            path=path,
            orig_id=orig_id,
            action="extract_only_failed",
            before=before,
            after=before,
            changed=False,
            details="No stored judge outputs available for extraction-only pass.",
        )

    raw_task = cast(dict[str, Any], row["raw_task"])
    engine = engine_mgr.get_engine(model_name)
    extract_ctx = _build_extract_context(raw_output=raw_output, retry_output=retry_output, dataset=dataset)
    try:
        extract_output = str(
            engine.generate_batch(
                [extract_ctx],
                batch_size=1,
                sampling_kwargs=dict(EXTRACT_SAMPLING_KWARGS),
            )[0]
        )
    except Exception as e:
        return RowUpdateResult(
            path=path,
            orig_id=orig_id,
            action="extract_only_failed",
            before=before,
            after=before,
            changed=False,
            details=f"Extraction generation error: {type(e).__name__}: {e}",
        )

    parsed = _parse_judge_text(
        dataset=dataset,
        text=extract_output,
        raw_task=raw_task,
        source_prefix="extract",
        strict_enabled=True,
        recovery_enabled=False,
    )
    if parsed.answer is None:
        return RowUpdateResult(
            path=path,
            orig_id=orig_id,
            action="extract_only_failed",
            before=before,
            after=before,
            changed=False,
            details="Extraction output still unparsable.",
        )

    row_before_update = json.dumps(row, ensure_ascii=False, sort_keys=True)
    _apply_answer_update(
        dataset=dataset,
        row=row,
        judged_answer=parsed.answer,
        parse_failed=False,
        used_fallback=False,
        parse_mode=parsed.mode,
        parse_source=parsed.source,
        retry_reason="parse_none_then_extract_only",
        finish_state="extract_only_parsed",
        raw_had_strict_final=bool(jt.get("judge_raw_had_strict_final")),
        retry_had_strict_final=bool(jt.get("judge_retry_had_strict_final")),
        judge_raw_output=None,
        judge_retry_output=None,
        judge_context=None,
        rerun=False,
    )
    row_after_update = json.dumps(row, ensure_ascii=False, sort_keys=True)
    after = row.get("final_judge_answer")
    return RowUpdateResult(
        path=path,
        orig_id=orig_id,
        action="extract_only",
        before=before,
        after=after,
        changed=(row_before_update != row_after_update),
        details=f"Extracted final choice from stored judge outputs. Parse mode={parsed.mode} source={parsed.source}.",
    )


def _apply_reparse_result(
    *,
    dataset: DatasetName,
    path: Path,
    row: dict[str, Any],
    orig_id: int,
    before: Any,
    reparsed: str,
    parse_mode: str,
    parse_source: str,
    raw_had_strict_final: bool,
    retry_had_strict_final: bool,
) -> RowUpdateResult:
    row_before_update = json.dumps(row, ensure_ascii=False, sort_keys=True)
    _apply_answer_update(
        dataset=dataset,
        row=row,
        judged_answer=reparsed,
        parse_failed=False,
        used_fallback=(parse_mode == "recover"),
        parse_mode=parse_mode,
        parse_source=parse_source,
        retry_reason=None,
        finish_state="reparse_existing",
        raw_had_strict_final=raw_had_strict_final,
        retry_had_strict_final=retry_had_strict_final,
        judge_raw_output=None,
        judge_retry_output=None,
        judge_context=None,
        rerun=False,
    )
    row_after_update = json.dumps(row, ensure_ascii=False, sort_keys=True)
    after = row.get("final_judge_answer")
    return RowUpdateResult(
        path=path,
        orig_id=orig_id,
        action="reparsed",
        before=before,
        after=after,
        changed=(row_before_update != row_after_update),
        details=f"Recovered via existing judge output parse mode={parse_mode} source={parse_source}.",
    )


def _rerun_action(attempt: JudgeAttempt) -> str:
    if attempt.parse_failed:
        return "rerun_failed"
    if attempt.retry_used:
        return "rerun_retry"
    return "rerun"


def _rerun_details(attempt: JudgeAttempt) -> str:
    return (
        "Reran judge with rebuilt context."
        + (" Retry was used." if attempt.retry_used else "")
        + f" Parse mode={attempt.parse_mode} source={attempt.parse_source}."
        + (" Still unparsable after retry." if attempt.parse_failed else "")
    )


def _apply_rerun_attempt_result(
    *,
    dataset: DatasetName,
    req: RerunRequest,
    attempt: JudgeAttempt,
) -> RowUpdateResult:
    row_before_update = json.dumps(req.row, ensure_ascii=False, sort_keys=True)
    _apply_answer_update(
        dataset=dataset,
        row=req.row,
        judged_answer=attempt.judged_answer,
        parse_failed=attempt.parse_failed,
        used_fallback=attempt.used_fallback,
        parse_mode=attempt.parse_mode,
        parse_source=attempt.parse_source,
        retry_reason=attempt.retry_reason,
        finish_state=attempt.finish_state,
        raw_had_strict_final=attempt.raw_had_strict_final,
        retry_had_strict_final=attempt.retry_had_strict_final,
        judge_raw_output=attempt.raw_output,
        judge_retry_output=attempt.retry_output,
        judge_context=attempt.judge_context,
        rerun=True,
    )
    row_after_update = json.dumps(req.row, ensure_ascii=False, sort_keys=True)
    after = req.row.get("final_judge_answer")
    return RowUpdateResult(
        path=req.path,
        orig_id=req.orig_id,
        action=_rerun_action(attempt),
        before=req.before,
        after=after,
        changed=(row_before_update != row_after_update),
        details=_rerun_details(attempt),
    )


def _plan_target_row(
    *,
    dataset: DatasetName,
    path: Path,
    row: dict[str, Any],
    row_idx: int,
    orig_id: int,
    extract_only_existing: bool,
    engine_mgr: EngineManager,
) -> RowPlan:
    _ensure_row_has_fields(row)
    before = row.get("final_judge_answer")

    if not _needs_retry(row, dataset=dataset):
        return RowPlan(
            immediate_result=RowUpdateResult(
                path=path,
                orig_id=orig_id,
                action="skip_valid",
                before=before,
                after=before,
                changed=False,
                details="Stored final_judge_answer is already strict-valid; no retry needed.",
            ),
            rerun_request=None,
        )

    reparsed, parse_mode, parse_source, raw_had_strict_final, retry_had_strict_final = _try_reparse_existing(
        row, dataset=dataset
    )
    if reparsed is not None:
        return RowPlan(
            immediate_result=_apply_reparse_result(
                dataset=dataset,
                path=path,
                row=row,
                orig_id=orig_id,
                before=before,
                reparsed=reparsed,
                parse_mode=parse_mode,
                parse_source=parse_source,
                raw_had_strict_final=raw_had_strict_final,
                retry_had_strict_final=retry_had_strict_final,
            ),
            rerun_request=None,
        )

    if extract_only_existing:
        return RowPlan(
            immediate_result=_extract_only_existing_result(
                dataset=dataset,
                path=path,
                row=row,
                orig_id=orig_id,
                engine_mgr=engine_mgr,
            ),
            rerun_request=None,
        )

    jt = cast(dict[str, Any], row.get("judge_trace") or {})
    model_name = str(jt.get("judge_model") or "").strip()
    if not model_name:
        raise ValueError(f"{path}: orig_id={orig_id} missing judge_trace.judge_model; cannot rerun judge.")

    return RowPlan(
        immediate_result=None,
        rerun_request=RerunRequest(
            path=path,
            orig_id=orig_id,
            row_idx=row_idx,
            row=row,
            before=before,
            model_name=model_name,
        ),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Repair targeted debate rows by reparsing existing judge outputs and rerunning judge as needed."
    )
    ap.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gsm8k", "aime25", "gpqa"],
        help="Dataset parser/checker to use.",
    )
    ap.add_argument(
        "--target",
        type=_parse_target,
        action="append",
        required=True,
        help="Target in format '<path>:<orig_id>'. Repeat for multiple targets.",
    )
    ap.add_argument("--gpus", type=str, default="0", help="GPU IDs (comma-separated).")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization (0.0-1.0).")
    ap.add_argument(
        "--context_len",
        type=int,
        default=None,
        help="Fixed context length for judge reruns. If omitted, uses adaptive context.",
    )
    ap.add_argument("--enable_yarn", action="store_true", help="Enable YaRN RoPE scaling for long context.")
    ap.add_argument("--enforce_eager", action="store_true", help="Run vLLM in eager mode.")
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for judge rerun generation (applies to batched raw/retry passes).",
    )

    ap.add_argument(
        "--judge_max_tokens",
        type=int,
        default=None,
        help="Optional judge max new tokens override.",
    )
    ap.add_argument("--judge_temperature", type=float, default=None, help="Optional judge temperature override.")
    ap.add_argument("--judge_top_p", type=float, default=None, help="Optional judge top_p override.")
    ap.add_argument("--judge_top_k", type=int, default=None, help="Optional judge top_k override.")

    ap.add_argument("--dry_run", action="store_true", help="Report planned changes without writing files.")
    ap.add_argument("--no_backup", action="store_true", help="Do not create .bak.* files before rewriting.")
    ap.add_argument(
        "--extract_only_existing",
        action="store_true",
        help="Do not rerun judge generation; run only extraction pass from stored judge outputs.",
    )
    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()

    dataset = cast(DatasetName, args.dataset)
    targets = cast(list[TargetSpec], args.target)
    if not targets:
        print("No targets provided.", file=sys.stderr)
        sys.exit(1)

    targets_by_file: dict[Path, list[int]] = {}
    for t in targets:
        targets_by_file.setdefault(t.path, []).append(t.orig_id)

    for p in targets_by_file:
        if not p.exists():
            raise FileNotFoundError(f"Target file not found: {p}")

    judge_overrides = _parse_judge_override_args(args)
    manager = EngineManager(
        gpus=str(args.gpus),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        context_len=args.context_len,
        enable_yarn=bool(args.enable_yarn),
        enforce_eager=bool(args.enforce_eager),
        judge_overrides=judge_overrides,
    )

    all_results: list[RowUpdateResult] = []
    rewritten_files: set[Path] = set()

    try:
        for path, orig_ids in targets_by_file.items():
            lines = _read_jsonl_lines(path)
            row_idx_by_orig: dict[int, list[int]] = {}
            idx_to_obj: dict[int, dict[str, Any]] = {}

            for idx, ln in enumerate(lines):
                if not ln.strip():
                    continue
                obj = cast(dict[str, Any], json.loads(ln))
                idx_to_obj[idx] = obj
                try:
                    oid = int(obj.get("orig_id"))
                except Exception:
                    continue
                row_idx_by_orig.setdefault(oid, []).append(idx)

            file_changed = False
            rerun_requests: list[RerunRequest] = []
            for orig_id in orig_ids:
                idxs = row_idx_by_orig.get(int(orig_id), [])
                if not idxs:
                    raise ValueError(f"{path}: orig_id={orig_id} not found.")
                if len(idxs) > 1:
                    raise ValueError(
                        f"{path}: orig_id={orig_id} matched multiple rows ({len(idxs)})."
                    )
                row_idx = idxs[0]
                row = idx_to_obj[row_idx]
                plan = _plan_target_row(
                    dataset=dataset,
                    path=path,
                    row=row,
                    row_idx=row_idx,
                    orig_id=orig_id,
                    extract_only_existing=bool(args.extract_only_existing),
                    engine_mgr=manager,
                )
                if plan.immediate_result is not None:
                    res = plan.immediate_result
                    all_results.append(res)
                    file_changed = file_changed or bool(res.changed)
                    if res.action != "skip_valid":
                        lines[row_idx] = json.dumps(row, ensure_ascii=False)
                    continue
                if plan.rerun_request is not None:
                    rerun_requests.append(plan.rerun_request)
                    continue
                raise RuntimeError(f"{path}: orig_id={orig_id} produced no row plan.")

            if rerun_requests:
                attempts_by_target = _run_judge_with_retry_batched(
                    dataset=dataset,
                    requests=rerun_requests,
                    engine_mgr=manager,
                    batch_size=int(args.batch_size) if args.batch_size is not None else None,
                )
                for req in rerun_requests:
                    key = (req.path, req.orig_id)
                    attempt = attempts_by_target.get(key)
                    if attempt is None:
                        raise RuntimeError(f"Missing batched rerun attempt for {req.path}: orig_id={req.orig_id}")
                    res = _apply_rerun_attempt_result(
                        dataset=dataset,
                        req=req,
                        attempt=attempt,
                    )
                    all_results.append(res)
                    file_changed = file_changed or bool(res.changed)
                    lines[req.row_idx] = json.dumps(req.row, ensure_ascii=False)

            if not args.dry_run and file_changed:
                if not args.no_backup:
                    backup = path.with_name(f"{path.name}.bak.{_now_tag()}")
                    shutil.copy2(path, backup)
                    print(f"[backup] {backup}", file=sys.stderr)
                _write_jsonl_lines_atomic(path, lines)
                rewritten_files.add(path)
                print(f"[write] {path}", file=sys.stderr)

        counts: dict[str, int] = {}
        for r in all_results:
            counts[r.action] = counts.get(r.action, 0) + 1

        print("\n=== Rejudge Repair Summary ===")
        print(f"Targets processed: {len(all_results)}")
        print(f"Files rewritten: {len(rewritten_files)}")
        print(f"Dry run: {bool(args.dry_run)}")
        for k in sorted(counts):
            print(f"- {k}: {counts[k]}")
        print("\nPer-target results:")
        for r in all_results:
            print(
                f"- {r.path} orig_id={r.orig_id}: {r.action} "
                f"(before={r.before!r} -> after={r.after!r})"
            )
            if r.details:
                print(f"  details: {r.details}")

    finally:
        manager.close()


if __name__ == "__main__":
    main()
