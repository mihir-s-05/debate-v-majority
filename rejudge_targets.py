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
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

if __package__ in (None, ""):
    import importlib.util

    _PKG_NAME = "debug_majority_debate"
    _PKG_DIR = Path(__file__).resolve().parent
    _INIT = _PKG_DIR / "__init__.py"
    if _PKG_NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _PKG_NAME,
            str(_INIT),
            submodule_search_locations=[str(_PKG_DIR)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create package spec for {_PKG_DIR}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_PKG_NAME] = mod
        spec.loader.exec_module(mod)
    __package__ = _PKG_NAME

from . import DatasetName
from .cli import (
    JUDGE_RETRY_NUDGE,
    _build_judge_context,
    _check_answer_correctness,
    _parse_judge_output,
)
from .engines import (
    SamplingConfig,
    build_sampling_config,
    create_inference_engine,
    set_sampling_config,
)
from .shared import render_agent_assistant_rounds


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
        if sampling_kwargs and "max_tokens" in sampling_kwargs:
            return int(sampling_kwargs["max_tokens"])
        cfg = self._get_sampling_cfg(model_name)
        return int(cfg.max_tokens or 4096)


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
    return _parse_judge_output(
        dataset=dataset,
        text=text,
        raw_task=raw_task,
        source_prefix=source_prefix,
        strict_enabled=strict_enabled,
        recovery_enabled=recovery_enabled,
    )


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
    parsed_raw = _parse_judge_text(
        dataset=dataset,
        text=raw_out,
        raw_task=raw_task,
        source_prefix="raw",
        strict_enabled=True,
        recovery_enabled=True,
    )
    if parsed_raw.answer is not None:
        return parsed_raw.answer, parsed_raw.mode, parsed_raw.source, raw_had_strict, False

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
        parsed_retry = _parse_judge_text(
            dataset=dataset,
            text=retry_out,
            raw_task=raw_task,
            source_prefix="retry",
            strict_enabled=True,
            recovery_enabled=True,
        )
        if parsed_retry.answer is not None:
            return parsed_retry.answer, parsed_retry.mode, parsed_retry.source, raw_had_strict, retry_had_strict

    return None, "none", "none", raw_had_strict, retry_had_strict


def _needs_retry(row: dict[str, Any], *, dataset: DatasetName) -> bool:
    jt = cast(dict[str, Any], row.get("judge_trace") or {})
    if bool(jt.get("judge_parse_failed")):
        return True
    if str(jt.get("judge_parse_mode") or "").strip().lower() == "recover":
        return True
    return not _stored_final_is_valid(row, dataset=dataset)


def _rebuild_judge_context(row: dict[str, Any], *, dataset: DatasetName) -> list[dict[str, str]]:
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
    return _build_judge_context(
        dataset=dataset,
        question=question,
        responses=transcripts,
        previous_judge=None,
    )


def _run_judge_with_retry(
    *,
    dataset: DatasetName,
    row: dict[str, Any],
    engine_mgr: EngineManager,
    model_name: str,
    batch_size: int | None,
) -> JudgeAttempt:
    raw_task = cast(dict[str, Any], row["raw_task"])
    try:
        judge_context = _rebuild_judge_context(row, dataset=dataset)
    except Exception:
        jt = cast(dict[str, Any], row.get("judge_trace") or {})
        stored_ctx = jt.get("judge_context")
        if not isinstance(stored_ctx, list) or not stored_ctx:
            raise
        judge_context = cast(list[dict[str, str]], stored_ctx)

    engine = engine_mgr.get_engine(model_name)
    sampling_kwargs = engine_mgr.judge_sampling_kwargs(model_name)
    judge_max_new_tokens = engine_mgr.judge_max_new_tokens(model_name, sampling_kwargs)

    raw_output = engine.generate_batch(
        [judge_context],
        batch_size=batch_size,
        sampling_kwargs=sampling_kwargs,
    )[0]
    raw_output_s = str(raw_output)

    raw_strict = _parse_judge_text(
        dataset=dataset,
        text=raw_output_s,
        raw_task=raw_task,
        source_prefix="raw",
        strict_enabled=True,
        recovery_enabled=False,
    )
    raw_had_strict_final = bool(raw_strict.strict_success)
    parsed_raw = _parse_judge_text(
        dataset=dataset,
        text=raw_output_s,
        raw_task=raw_task,
        source_prefix="raw",
        strict_enabled=True,
        recovery_enabled=True,
    )
    if parsed_raw.answer is not None:
        return JudgeAttempt(
            judged_answer=parsed_raw.answer,
            raw_output=raw_output_s,
            retry_output=None,
            parse_failed=False,
            used_fallback=(parsed_raw.mode == "recover"),
            retry_used=False,
            parse_mode=parsed_raw.mode,
            parse_source=parsed_raw.source,
            retry_reason=None,
            finish_state="raw_parsed",
            raw_had_strict_final=raw_had_strict_final,
            retry_had_strict_final=False,
            judge_context=judge_context,
        )

    retry_sampling = dict(sampling_kwargs or {})
    retry_sampling["temperature"] = 0.0
    retry_sampling["top_p"] = 1.0
    retry_sampling["max_tokens"] = int(judge_max_new_tokens)

    retry_ctx = list(judge_context) + [
        {"role": "assistant", "content": raw_output_s},
        {"role": "user", "content": JUDGE_RETRY_NUDGE},
    ]
    try:
        retry_output = str(
            engine.generate_batch(
                [retry_ctx],
                batch_size=batch_size,
                sampling_kwargs=retry_sampling,
            )[0]
        )
    except Exception as e:
        retry_output = f"[retry_error] {type(e).__name__}: {e}"

    if retry_output.startswith("[retry_error]"):
        return JudgeAttempt(
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
            raw_had_strict_final=raw_had_strict_final,
            retry_had_strict_final=False,
            judge_context=judge_context,
        )

    retry_strict = _parse_judge_text(
        dataset=dataset,
        text=retry_output,
        raw_task=raw_task,
        source_prefix="retry",
        strict_enabled=True,
        recovery_enabled=False,
    )
    retry_had_strict_final = bool(retry_strict.strict_success)
    parsed_retry = _parse_judge_text(
        dataset=dataset,
        text=retry_output,
        raw_task=raw_task,
        source_prefix="retry",
        strict_enabled=True,
        recovery_enabled=True,
    )
    judged = parsed_retry.answer
    return JudgeAttempt(
        judged_answer=judged,
        raw_output=raw_output_s,
        retry_output=retry_output,
        parse_failed=(judged is None),
        used_fallback=(parsed_retry.mode == "recover"),
        retry_used=True,
        parse_mode=parsed_retry.mode,
        parse_source=parsed_retry.source,
        retry_reason="parse_none",
        finish_state="retry_parsed" if judged is not None else "retry_unparsed",
        raw_had_strict_final=raw_had_strict_final,
        retry_had_strict_final=retry_had_strict_final,
        judge_context=judge_context,
    )


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


def _process_target_row(
    *,
    dataset: DatasetName,
    path: Path,
    row: dict[str, Any],
    orig_id: int,
    engine_mgr: EngineManager,
    batch_size: int | None,
) -> RowUpdateResult:
    _ensure_row_has_fields(row)
    before = row.get("final_judge_answer")
    jt = cast(dict[str, Any], row.get("judge_trace") or {})

    stored_invalid = _needs_retry(row, dataset=dataset)
    if not stored_invalid:
        return RowUpdateResult(
            path=path,
            orig_id=orig_id,
            action="skip_valid",
            before=before,
            after=before,
            changed=False,
            details="Stored final_judge_answer is already strict-valid; no retry needed.",
        )

    reparsed, parse_mode, parse_source, raw_had_strict_final, retry_had_strict_final = _try_reparse_existing(
        row, dataset=dataset
    )
    if reparsed is not None:
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

    model_name = str(jt.get("judge_model") or "").strip()
    if not model_name:
        raise ValueError("Row missing judge_trace.judge_model; cannot rerun judge.")

    attempt = _run_judge_with_retry(
        dataset=dataset,
        row=row,
        engine_mgr=engine_mgr,
        model_name=model_name,
        batch_size=batch_size,
    )

    row_before_update = json.dumps(row, ensure_ascii=False, sort_keys=True)
    _apply_answer_update(
        dataset=dataset,
        row=row,
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
    row_after_update = json.dumps(row, ensure_ascii=False, sort_keys=True)
    after = row.get("final_judge_answer")
    action = "rerun_retry" if attempt.retry_used else "rerun"
    if attempt.parse_failed:
        action = "rerun_failed"
    return RowUpdateResult(
        path=path,
        orig_id=orig_id,
        action=action,
        before=before,
        after=after,
        changed=(row_before_update != row_after_update),
        details=(
            "Reran judge with rebuilt context."
            + (" Retry was used." if attempt.retry_used else "")
            + f" Parse mode={attempt.parse_mode} source={attempt.parse_source}."
            + (" Still unparsable after retry." if attempt.parse_failed else "")
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
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size for judge rerun generation.")

    ap.add_argument("--judge_max_tokens", type=int, default=None, help="Optional judge max new tokens override.")
    ap.add_argument("--judge_temperature", type=float, default=None, help="Optional judge temperature override.")
    ap.add_argument("--judge_top_p", type=float, default=None, help="Optional judge top_p override.")
    ap.add_argument("--judge_top_k", type=int, default=None, help="Optional judge top_k override.")

    ap.add_argument("--dry_run", action="store_true", help="Report planned changes without writing files.")
    ap.add_argument("--no_backup", action="store_true", help="Do not create .bak.* files before rewriting.")
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
                res = _process_target_row(
                    dataset=dataset,
                    path=path,
                    row=row,
                    orig_id=orig_id,
                    engine_mgr=manager,
                    batch_size=int(args.batch_size) if args.batch_size is not None else None,
                )
                all_results.append(res)
                file_changed = file_changed or bool(res.changed)
                if row_idx in idx_to_obj:
                    lines[row_idx] = json.dumps(row, ensure_ascii=False)

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
