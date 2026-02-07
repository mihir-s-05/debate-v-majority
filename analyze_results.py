#!/usr/bin/env python3
"""
Analyze multi-agent debate vs majority voting outputs for AIME25 + GPQA.

Reads jsonl runs from:
  - multiagent_debate/results/aime25_quick/
  - multiagent_debate/results/gpqa_quick/

Writes:
  - ~/debug_majority_debate/_autogen/summary.json
  - ~/debug_majority_debate/_autogen/tables.md
  - Appends incremental findings to ~/debug_majority_debate/FINDINGS_LOG.md
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HOME = Path.home()
DEFAULT_RESULTS_DIR = Path("/home/ubuntu/multi-agent-attack/multiagent_debate/results")
DEFAULT_OUT_DIR = HOME / "debug_majority_debate" / "_autogen"
FINDINGS_LOG = HOME / "debug_majority_debate" / "FINDINGS_LOG.md"
TARGET_MODEL_TAG = os.environ.get("TARGET_MODEL_TAG", "Qwen/Qwen3-8B")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_findings_md(md: str) -> None:
    FINDINGS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FINDINGS_LOG, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"## {_now_iso()}\n\n")
        f.write(md.rstrip() + "\n")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (max(0.0, center - half), min(1.0, center + half))


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def fmt_ci(k: int, n: int) -> str:
    lo, hi = wilson_ci(k, n)
    return f"{fmt_pct(k/n if n else 0.0)} [{fmt_pct(lo)}, {fmt_pct(hi)}]"


def entropy_from_counts(counts: Counter[str | None]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p, 2)
    return h


def strict_majority_vote(answers: list[str | None]) -> str | None:
    """
    Return a strict-majority answer (> n/2), else None.
    Ignores None by treating it as a normal category (consistent with stored runs).
    """
    n = len(answers)
    if n == 0:
        return None
    counts = Counter(answers)
    ans, cnt = counts.most_common(1)[0]
    if cnt > n / 2:
        return ans
    return None


def plurality_vote(answers: list[str | None]) -> str | None:
    """Return max-count answer if unique max, else None."""
    if not answers:
        return None
    counts = Counter(answers)
    [(a1, c1), *rest] = counts.most_common()
    if not rest:
        return a1
    a2, c2 = rest[0]
    if c1 == c2:
        return None
    return a1


def plurality_vote_ignore_none(answers: list[str | None]) -> str | None:
    return plurality_vote([a for a in answers if a is not None])


@dataclass(frozen=True)
class RunMeta:
    path: str
    dataset: str  # "aime25" | "gpqa"
    mode: str  # "majority" | "debate" | "single"
    n: int | None
    seed: int | None
    n_samples: int | None
    n_agents: int | None
    n_rounds: int | None
    tag_all: bool
    model_tag: str | None
    ts: str | None


@dataclass
class RunSummary:
    meta: RunMeta
    n_questions: int
    final_correct: int
    final_incorrect: int
    final_none: int
    # Debate-only
    judge_correct: int | None = None
    judge_incorrect: int | None = None
    judge_none: int | None = None
    majority_correct: int | None = None
    majority_incorrect: int | None = None
    majority_none: int | None = None


_FILENAME_RE = re.compile(
    r"^(?P<mode>debate|majority|single)_(?P<dataset>aime|gpqa)"
    r"(?:_agents(?P<agents>\d+)_r(?P<rounds>\d+))?"
    r"(?:_samples(?P<samples>\d+))?"
    r"_n(?P<n>\d+)"
    r"_seed(?P<seed>\d+)"
    r"(?P<all>_all)?"
    r"_(?P<ts>\d{8}_\d{6})"
    r"(?:_(?P<org>[^_]+)_(?P<model>.+?))?"
    r"\.jsonl$"
)


def parse_run_meta(path: Path) -> RunMeta:
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized filename format: {path.name}")
    mode = m.group("mode")
    dataset_short = m.group("dataset")
    dataset = "aime25" if dataset_short == "aime" else "gpqa"
    n_agents = int(m.group("agents")) if m.group("agents") else None
    n_rounds = int(m.group("rounds")) if m.group("rounds") else None
    n_samples = int(m.group("samples")) if m.group("samples") else None
    n = int(m.group("n")) if m.group("n") else None
    seed = int(m.group("seed")) if m.group("seed") else None
    tag_all = bool(m.group("all"))
    if m.group("org") and m.group("model"):
        model_tag = f"{m.group('org')}/{m.group('model')}".replace("__", "_")
    else:
        model_tag = None
    ts = m.group("ts") if m.group("ts") else None
    return RunMeta(
        path=str(path),
        dataset=dataset,
        mode=mode,
        n=n,
        seed=seed,
        n_samples=n_samples,
        n_agents=n_agents,
        n_rounds=n_rounds,
        tag_all=tag_all,
        model_tag=model_tag,
        ts=ts,
    )


def infer_model_tag_from_siblings(path: Path) -> str | None:
    """
    Older runs may omit the *_Org_Model suffix. To avoid mixing models, attempt to infer
    the model tag by looking for sibling files with the same seed+timestamp.
    """
    try:
        meta = parse_run_meta(path)
    except Exception:
        return None
    if meta.model_tag:
        return meta.model_tag
    if meta.seed is None or meta.ts is None:
        return None

    sibling_model_tags: set[str] = set()
    for sib in path.parent.glob(f"*seed{meta.seed}*_{meta.ts}_*.jsonl"):
        if sib == path:
            continue
        try:
            sib_meta = parse_run_meta(sib)
        except Exception:
            continue
        if sib_meta.model_tag:
            sibling_model_tags.add(sib_meta.model_tag)
    if len(sibling_model_tags) == 1:
        return next(iter(sibling_model_tags))
    return None


def should_include_path(path: Path, *, target_model_tag: str) -> tuple[bool, str | None]:
    """
    Decide whether to include a run file for analysis.
    Returns (include?, effective_model_tag).
    """
    effective_model_tag = infer_model_tag_from_siblings(path)
    if effective_model_tag is None:
        return (False, None)
    return (effective_model_tag == target_model_tag, effective_model_tag)


def load_parsers() -> tuple[Any, Any]:
    """
    Import parsers from ~/debug_majority_debate as a package.
    """
    import sys

    # Allow `import debug_majority_debate.*`
    if str(HOME) not in sys.path:
        sys.path.insert(0, str(HOME))
    import debug_majority_debate.aime25 as aime25
    import debug_majority_debate.gpqa as gpqa

    return aime25, gpqa


def extract_round_answers_debate(
    rec: dict[str, Any],
    *,
    dataset: str,
    aime25_mod: Any,
    gpqa_mod: Any,
) -> list[list[str | None]]:
    """
    Returns answers[agent_idx][round_idx] (parsed).
    """
    raw_task = rec.get("raw_task") or {}
    n_agents = int(rec["n_agents"])
    n_rounds = int(rec["n_rounds"])
    agent_responses = rec["agent_responses"]
    out: list[list[str | None]] = []
    for a in range(n_agents):
        msgs = agent_responses[a]
        # Convention in these runs: [user, assistant] repeated per round
        answers_a: list[str | None] = []
        for r in range(n_rounds):
            idx = 2 * r + 1
            if idx >= len(msgs):
                answers_a.append(None)
                continue
            text = msgs[idx].get("content", "")
            if dataset == "aime25":
                ans = aime25_mod.parse_answer(text, raw_task)
            elif dataset == "gpqa":
                ans = gpqa_mod.parse_answer(text, raw_task)
            else:
                raise ValueError(f"unknown dataset {dataset}")
            answers_a.append(ans)
        out.append(answers_a)
    return out


def count_none(xs: Iterable[Any]) -> int:
    return sum(1 for x in xs if x is None)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def analyze(results_dir: Path, out_dir: Path) -> None:
    aime25_mod, gpqa_mod = load_parsers()

    out_dir.mkdir(parents=True, exist_ok=True)
    all_paths: list[Path] = []
    for sub in ("aime25_quick", "gpqa_quick"):
        all_paths.extend(sorted((results_dir / sub).glob("*.jsonl")))

    # Model filtering: keep only runs attributable to TARGET_MODEL_TAG.
    paths: list[Path] = []
    excluded: Counter[str] = Counter()
    for p in all_paths:
        include, eff = should_include_path(p, target_model_tag=TARGET_MODEL_TAG)
        if include:
            paths.append(p)
        else:
            excluded[eff or "unknown_model"] += 1

    # Inventory: count files by dataset/mode.
    inv = defaultdict(int)
    for p in paths:
        meta = parse_run_meta(p)
        inv[(meta.dataset, meta.mode)] += 1

    inv_rows = [
        [ds, mode, str(cnt)]
        for (ds, mode), cnt in sorted(inv.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]
    append_findings_md(
        "### Inventory\n\n"
        + f"- Target model tag: `{TARGET_MODEL_TAG}`\n"
        + f"- Included files: {len(paths)} / {len(all_paths)}\n"
        + (
            "- Excluded by inferred model tag:\n"
            + _md_table(["Inferred model tag", "Count"], [[k, str(v)] for k, v in excluded.items()])
            + "\n\n"
            if excluded
            else "\n\n"
        )
        + _md_table(["Dataset", "Mode", "Runs"], inv_rows)
        + "\n\nNotes:\n"
        + "- This analysis includes all `debate_*.jsonl` and `majority_*.jsonl` runs, plus `single_*.jsonl` baselines when present.\n"
        + "- Some GPQA runs use `n=448` with `_all_` in the filename (full dataset), others use `n=50` (subset).\n"
    )

    # Aggregators
    run_summaries: list[RunSummary] = []

    # Overall pooled (dataset, mode, cfg_key) -> counts
    pooled = defaultdict(lambda: Counter())

    # Majority voting internals (dataset -> counts)
    maj_patterns = defaultdict(lambda: Counter())
    # Formatting / truncation indicators for single+majority baselines
    baseline_format = defaultdict(lambda: Counter())  # dataset -> counts

    # Debate dynamics
    debate_dyn = defaultdict(lambda: Counter())  # (dataset, cfg) -> counters
    debate_round_agree = defaultdict(lambda: defaultdict(int))  # (dataset,cfg) -> round->unanimous_count
    debate_round_total = defaultdict(lambda: defaultdict(int))  # (dataset,cfg) -> round->total
    debate_round_entropy_sum = defaultdict(lambda: defaultdict(float))
    debate_round_entropy_n = defaultdict(lambda: defaultdict(int))

    # Belief transitions
    trans_correctness = defaultdict(lambda: Counter())  # (dataset,cfg) -> (prev_state->next_state)
    trans_choice_gpqa = defaultdict(lambda: Counter())  # (cfg) -> (A->B etc)
    change_toward_prev_other_majority = defaultdict(lambda: Counter())  # (dataset,cfg) -> counts

    # Judge vs majority within debate
    judge_matrix = defaultdict(lambda: Counter())  # (dataset,cfg) -> (maj_correct/judge_correct combos)
    judge_override = defaultdict(lambda: Counter())  # (dataset,cfg) -> stats

    # Per-question pooled by method (dataset, orig_id, method_key) -> correct/total
    per_question = defaultdict(lambda: Counter())

    # Qualitative example selection (keep small excerpts)
    examples: dict[str, list[dict[str, Any]]] = {
        "aime25": [],
        "gpqa": [],
        "aime25_judge_harm": [],
        "gpqa_judge_harm": [],
        "aime25_unanimous_wrong": [],
        "gpqa_unanimous_wrong": [],
        "aime25_lost_correct": [],
        "gpqa_lost_correct": [],
    }

    def cfg_key(meta: RunMeta) -> str:
        if meta.mode != "debate":
            return "-"
        return f"{meta.n_agents}a-{meta.n_rounds}r"

    def completion_has_boxed_or_choice(text: Any, *, dataset: str) -> bool:
        if not isinstance(text, str):
            return False
        if "\\boxed" in text or "\\fbox" in text:
            return True
        if dataset == "gpqa":
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            tail_lines = lines[-8:] if len(lines) > 8 else lines
            tail = "\n".join(tail_lines)
            if re.search(r"(?i)\b(?:final\s+answer|answer)\b[^A-D]*\(?\s*([ABCD])\s*\)?\b", tail):
                return True
            last = tail_lines[-1] if tail_lines else ""
            if re.fullmatch(r"(?i)\(?\s*([ABCD])\s*\)?", last):
                return True
        return False

    def short_tail(text: Any, n: int = 360) -> str:
        if not isinstance(text, str):
            return ""
        s = text.strip()
        if len(s) <= n:
            return s
        return s[-n:]

    def check_correct(answer: Any, *, dataset: str, gt: Any) -> int:
        if dataset == "aime25":
            return int(aime25_mod.check_answer_correctness(answer, gt))
        if dataset == "gpqa":
            return int(gpqa_mod.check_answer_correctness(answer, gt))
        raise ValueError(f"Unknown dataset: {dataset}")

    for path in paths:
        meta = parse_run_meta(path)
        eff = infer_model_tag_from_siblings(path)
        if meta.model_tag is None and eff is not None:
            meta = RunMeta(**{**asdict(meta), "model_tag": eff})

        n_q = 0
        final_correct = 0
        final_none = 0

        judge_correct = 0
        judge_none = 0
        maj_correct_in_debate = 0
        maj_none_in_debate = 0

        # Per-file n_samples may appear in records; capture mode-wise max
        observed_n_samples: set[int] = set()

        for rec in read_jsonl(path):
            n_q += 1
            gt = rec.get("answer")
            orig_id = rec.get("orig_id")

            if meta.mode == "majority":
                observed_n_samples.add(int(rec.get("n_samples") or 0))
                final_ans = rec.get("final_answer")
                final_is_none = final_ans is None
                final_is_correct = int(rec.get("final_correct") or 0)

                final_correct += final_is_correct
                final_none += int(final_is_none)

                # majority pattern analysis
                samples = rec.get("sample_parsed_answers") or []
                completions = rec.get("sample_completions") or []

                maj_strict = strict_majority_vote(list(samples))
                uniq = len(set(samples))
                maj_patterns[meta.dataset][f"n_samples={len(samples)}"] += 1
                maj_patterns[meta.dataset][f"final_none={int(final_is_none)}"] += 1
                maj_patterns[meta.dataset][f"strict_majority_exists={int(maj_strict is not None)}"] += 1
                maj_patterns[meta.dataset][f"unique_answers={uniq}"] += 1
                maj_patterns[meta.dataset][f"any_none_in_samples={int(any(x is None for x in samples))}"] += 1
                maj_patterns[meta.dataset][
                    f"all_none_in_samples={int(all(x is None for x in samples) if samples else 0)}"
                ] += 1
                maj_patterns[meta.dataset][f"n_non_none={sum(x is not None for x in samples)}"] += 1

                # "oracle" baselines: was the correct answer present among parsed samples?
                any_sample_correct = any(check_correct(x, dataset=meta.dataset, gt=gt) == 1 for x in samples)
                first_non_none = next((x for x in samples if x is not None), None)
                maj_patterns[meta.dataset]["any_sample_correct"] += int(any_sample_correct)
                maj_patterns[meta.dataset]["first_non_none_correct"] += int(
                    check_correct(first_non_none, dataset=meta.dataset, gt=gt) == 1
                )

                # Formatting/truncation indicators
                baseline_format[meta.dataset]["total_records"] += 1
                baseline_format[meta.dataset]["total_completions"] += len(completions)
                for comp in completions:
                    if completion_has_boxed_or_choice(comp, dataset=meta.dataset):
                        baseline_format[meta.dataset]["completions_with_boxed_or_choice"] += 1
                    if isinstance(comp, str):
                        baseline_format[meta.dataset]["completion_chars_sum"] += len(comp)
                        baseline_format[meta.dataset]["completion_chars_ge_2000"] += int(len(comp) >= 2000)

                # per-question pooled
                per_question[(meta.dataset, orig_id, "majority")]["total"] += 1
                per_question[(meta.dataset, orig_id, "majority")]["correct"] += final_is_correct

            elif meta.mode == "single":
                final_ans = rec.get("final_answer")
                final_is_none = final_ans is None
                final_is_correct = int(rec.get("final_correct") or 0)

                final_correct += final_is_correct
                final_none += int(final_is_none)

                completions = rec.get("sample_completions") or []
                per_question[(meta.dataset, orig_id, "single")]["total"] += 1
                per_question[(meta.dataset, orig_id, "single")]["correct"] += final_is_correct

                baseline_format[meta.dataset]["total_records"] += 1
                baseline_format[meta.dataset]["total_completions"] += len(completions)
                for comp in completions:
                    if completion_has_boxed_or_choice(comp, dataset=meta.dataset):
                        baseline_format[meta.dataset]["completions_with_boxed_or_choice"] += 1
                    if isinstance(comp, str):
                        baseline_format[meta.dataset]["completion_chars_sum"] += len(comp)
                        baseline_format[meta.dataset]["completion_chars_ge_2000"] += int(len(comp) >= 2000)

            elif meta.mode == "debate":
                cfg = cfg_key(meta)
                maj_ans = rec.get("final_majority_answer")
                maj_is_correct = int(rec.get("final_majority_correct") or 0)
                maj_correct_in_debate += maj_is_correct
                maj_none_in_debate += int(maj_ans is None)

                judge_ans = rec.get("final_judge_answer")
                judge_is_correct = int(rec.get("final_judge_correct") or 0)
                judge_correct += judge_is_correct
                judge_none += int(judge_ans is None)

                # For debate outcomes, treat the run's judge decision as authoritative.
                final_correct += judge_is_correct
                final_none += int(judge_ans is None)

                # Extract round answers (prefer stored parse from the run; fall back to transcript parsing).
                answers = rec.get("agent_round_parsed_answers")
                if not answers:
                    answers = extract_round_answers_debate(
                        rec, dataset=meta.dataset, aime25_mod=aime25_mod, gpqa_mod=gpqa_mod
                    )

                n_agents = len(answers)
                n_rounds = len(answers[0]) if answers else 0

                final_round_answers = [answers[a][n_rounds - 1] for a in range(n_agents)] if n_rounds else []

                # per-question pooled
                per_question[(meta.dataset, orig_id, f"debate:{cfg}:judge")]["total"] += 1
                per_question[(meta.dataset, orig_id, f"debate:{cfg}:judge")]["correct"] += judge_is_correct
                per_question[(meta.dataset, orig_id, f"debate:{cfg}:majority")]["total"] += 1
                per_question[(meta.dataset, orig_id, f"debate:{cfg}:majority")]["correct"] += maj_is_correct

                # Judge conditioning context.
                # - "repeat-winner": repo_vote result (unique max with count>1, ignoring None)
                # - "plurality": unique max among non-None, even if count==1
                final_round_non_none = [a for a in final_round_answers if a is not None]
                final_counts_non_none = Counter(final_round_non_none)
                final_max_count = max(final_counts_non_none.values()) if final_counts_non_none else 0
                final_plurality = plurality_vote_ignore_none(final_round_answers)
                repeat_winner_exists = maj_ans is not None

                # Judge-vs-majority matrix
                judge_matrix[(meta.dataset, cfg)][f"maj={maj_is_correct},judge={judge_is_correct}"] += 1
                jo = judge_override[(meta.dataset, cfg)]
                jo["n_questions"] += 1
                jo[f"final_max_count={final_max_count}"] += 1
                jo["final_repeat_winner_exists"] += int(repeat_winner_exists)
                jo["final_repeat_winner_absent"] += int(not repeat_winner_exists)

                jo["judge_equals_majority"] += int(judge_ans == maj_ans)
                jo["judge_differs_majority"] += int(judge_ans != maj_ans)
                jo["judge_equals_majority_when_repeat_winner_exists"] += int(repeat_winner_exists and judge_ans == maj_ans)
                jo["judge_differs_majority_when_repeat_winner_exists"] += int(repeat_winner_exists and judge_ans != maj_ans)
                jo["judge_equals_majority_when_repeat_winner_absent"] += int((not repeat_winner_exists) and judge_ans == maj_ans)
                jo["judge_differs_majority_when_repeat_winner_absent"] += int((not repeat_winner_exists) and judge_ans != maj_ans)

                jo["final_plurality_exists"] += int(final_plurality is not None)
                jo["judge_equals_plurality"] += int(judge_ans == final_plurality)
                jo["judge_differs_plurality"] += int(judge_ans != final_plurality)
                jo["judge_equals_plurality_when_repeat_winner_absent"] += int(
                    (not repeat_winner_exists) and (judge_ans == final_plurality)
                )

                jo["judge_correct_when_repeat_winner_exists"] += int(repeat_winner_exists and judge_is_correct == 1)
                jo["judge_total_when_repeat_winner_exists"] += int(repeat_winner_exists)
                jo["judge_correct_when_repeat_winner_absent"] += int((not repeat_winner_exists) and judge_is_correct == 1)
                jo["judge_total_when_repeat_winner_absent"] += int(not repeat_winner_exists)

                jo["judge_rescue_from_repeat_winner_wrong"] += int(
                    repeat_winner_exists and (maj_is_correct == 0) and (judge_is_correct == 1)
                )
                jo["judge_rescue_from_repeat_winner_absent"] += int(
                    (not repeat_winner_exists) and (judge_is_correct == 1)
                )
                jo["judge_harm_over_repeat_winner_correct"] += int(
                    repeat_winner_exists and (maj_is_correct == 1) and (judge_is_correct == 0)
                )

                # Round-level agreement/entropy
                for r in range(n_rounds):
                    round_answers = [answers[a][r] for a in range(n_agents)]
                    counts = Counter(round_answers)
                    unanimous = int(len(counts) == 1)
                    debate_round_agree[(meta.dataset, cfg)][r] += unanimous
                    debate_round_total[(meta.dataset, cfg)][r] += 1
                    debate_round_entropy_sum[(meta.dataset, cfg)][r] += entropy_from_counts(counts)
                    debate_round_entropy_n[(meta.dataset, cfg)][r] += 1

                # Judge answer provenance: was it in any final-round agent answer?
                if judge_ans not in set(final_round_answers):
                    jo["judge_not_in_final_round_answers"] += 1
                    jo["judge_correct_when_not_in_final_round_answers"] += int(judge_is_correct == 1)
                    jo["judge_total_when_not_in_final_round_answers"] += 1
                else:
                    jo["judge_correct_when_in_final_round_answers"] += int(judge_is_correct == 1)
                    jo["judge_total_when_in_final_round_answers"] += 1
                if judge_ans not in {x for row in answers for x in row}:
                    jo["judge_not_in_any_round_answers"] += 1

                # Ever-correct / lost-correct
                ever_correct = False
                final_round_has_correct = False
                for a in range(n_agents):
                    for r in range(n_rounds):
                        ans = answers[a][r]
                        if ans is None:
                            continue
                        if check_correct(ans, dataset=meta.dataset, gt=gt) == 1:
                            ever_correct = True
                            if r == n_rounds - 1:
                                final_round_has_correct = True
                debate_dyn[(meta.dataset, cfg)]["ever_correct"] += int(ever_correct)
                debate_dyn[(meta.dataset, cfg)]["final_round_has_correct"] += int(final_round_has_correct)
                debate_dyn[(meta.dataset, cfg)]["lost_correct"] += int(ever_correct and not final_round_has_correct)
                debate_dyn[(meta.dataset, cfg)]["judge_wrong_with_final_correct_present"] += int(
                    (judge_is_correct == 0) and final_round_has_correct
                )
                debate_dyn[(meta.dataset, cfg)]["judge_wrong_with_ever_correct_present"] += int(
                    (judge_is_correct == 0) and ever_correct
                )
                debate_dyn[(meta.dataset, cfg)]["judge_correct"] += judge_is_correct
                debate_dyn[(meta.dataset, cfg)]["majority_correct"] += maj_is_correct
                debate_dyn[(meta.dataset, cfg)]["n_questions"] += 1

                # Mirror some correctness-availability context into judge_override for easier reporting.
                jo["final_round_has_correct"] += int(final_round_has_correct)
                jo["ever_correct"] += int(ever_correct)
                jo["judge_wrong_with_final_correct_present"] += int((judge_is_correct == 0) and final_round_has_correct)
                jo["judge_wrong_with_ever_correct_present"] += int((judge_is_correct == 0) and ever_correct)

                # Belief changes: correctness-state transitions + conformity proxy
                for a in range(n_agents):
                    seq = answers[a]
                    for r in range(1, n_rounds):
                        prev = seq[r - 1]
                        cur = seq[r]
                        prev_state = (
                            "none"
                            if prev is None
                            else ("correct" if check_correct(prev, dataset=meta.dataset, gt=gt) == 1 else "wrong")
                        )
                        cur_state = (
                            "none"
                            if cur is None
                            else ("correct" if check_correct(cur, dataset=meta.dataset, gt=gt) == 1 else "wrong")
                        )
                        trans_correctness[(meta.dataset, cfg)][f"{prev_state}->{cur_state}"] += 1

                        if prev != cur:
                            debate_dyn[(meta.dataset, cfg)]["answer_changes"] += 1
                        debate_dyn[(meta.dataset, cfg)]["answer_steps"] += 1

                        # GPQA letter-to-letter transitions (excluding None)
                        if meta.dataset == "gpqa" and prev is not None and cur is not None:
                            trans_choice_gpqa[cfg][f"{prev}->{cur}"] += 1

                        # Conformity proxy: did the agent move *toward* prev-round plurality of other agents?
                        others_prev = [answers[aa][r - 1] for aa in range(n_agents) if aa != a]
                        other_plural = plurality_vote(others_prev)
                        if prev != cur:
                            change_toward_prev_other_majority[(meta.dataset, cfg)]["changed"] += 1
                            if cur == other_plural:
                                change_toward_prev_other_majority[(meta.dataset, cfg)]["changed_to_other_plurality"] += 1
                            if prev == other_plural:
                                change_toward_prev_other_majority[(meta.dataset, cfg)]["changed_away_from_other_plurality"] += 1

                # Keep a few examples (short tails only to avoid bloat)
                if judge_is_correct == 1 and maj_is_correct == 0:
                    # Judge rescue: judge correct, majority wrong/None.
                    ex = {
                        "run": path.name,
                        "cfg": cfg,
                        "orig_id": orig_id,
                        "gt": gt,
                        "final_majority": maj_ans,
                        "final_judge": judge_ans,
                        "agent_tail": "",
                        "judge_tail": "",
                    }
                    for a in range(n_agents):
                        for r in range(n_rounds):
                            if check_correct(answers[a][r], dataset=meta.dataset, gt=gt) == 1:
                                msg = rec["agent_responses"][a][2 * r + 1].get("content", "")
                                ex["agent_tail"] = short_tail(msg)
                                break
                        if ex["agent_tail"]:
                            break
                    ex["judge_tail"] = short_tail(str((rec.get("judge_trace") or {}).get("judge_raw_response", "")))
                    examples[meta.dataset].append(ex)

                if maj_is_correct == 1 and judge_is_correct == 0:
                    # Judge harm: judge flips a correct majority to wrong.
                    examples[f"{meta.dataset}_judge_harm"].append(
                        {
                            "run": path.name,
                            "cfg": cfg,
                            "orig_id": orig_id,
                            "gt": gt,
                            "final_majority": maj_ans,
                            "final_judge": judge_ans,
                            "judge_tail": short_tail(
                                str((rec.get("judge_trace") or {}).get("judge_raw_response", ""))
                            ),
                        }
                    )

                if n_rounds and final_round_answers:
                    final_counts = Counter(final_round_answers)
                    if len(final_counts) == 1:
                        unanimous_ans = next(iter(final_counts.keys()))
                        if unanimous_ans is not None and str(unanimous_ans) != str(gt):
                            examples[f"{meta.dataset}_unanimous_wrong"].append(
                                {
                                    "run": path.name,
                                    "cfg": cfg,
                                    "orig_id": orig_id,
                                    "gt": gt,
                                    "unanimous_final": unanimous_ans,
                                    "judge": judge_ans,
                                    "agent_tail": short_tail(
                                        rec["agent_responses"][0][2 * (n_rounds - 1) + 1].get("content", "")
                                    ),
                                }
                            )

                if ever_correct and not final_round_has_correct:
                    examples[f"{meta.dataset}_lost_correct"].append(
                        {
                            "run": path.name,
                            "cfg": cfg,
                            "orig_id": orig_id,
                            "gt": gt,
                            "final_majority": maj_ans,
                            "final_judge": judge_ans,
                        }
                    )

            else:
                raise ValueError(f"Unknown mode: {meta.mode}")

        final_incorrect = n_q - final_correct - final_none
        meta_final = meta
        if observed_n_samples:
            meta_final = RunMeta(**{**asdict(meta), "n_samples": max(observed_n_samples)})

        rs = RunSummary(
            meta=meta_final,
            n_questions=n_q,
            final_correct=final_correct,
            final_incorrect=final_incorrect,
            final_none=final_none,
        )
        if meta.mode == "debate":
            rs.judge_correct = judge_correct
            rs.judge_none = judge_none
            rs.judge_incorrect = n_q - judge_correct - judge_none
            rs.majority_correct = maj_correct_in_debate
            rs.majority_none = maj_none_in_debate
            rs.majority_incorrect = n_q - maj_correct_in_debate - maj_none_in_debate
        run_summaries.append(rs)

        # pooled stats
        cfg = cfg_key(meta_final)
        pooled[(meta_final.dataset, meta_final.mode, cfg)]["total"] += n_q
        pooled[(meta_final.dataset, meta_final.mode, cfg)]["correct"] += final_correct
        pooled[(meta_final.dataset, meta_final.mode, cfg)]["none"] += final_none

    # Write summary.json for downstream report generation.
    out_summary = {
        "generated_at": _now_iso(),
        "results_dir": str(results_dir),
        "target_model_tag": TARGET_MODEL_TAG,
        "run_summaries": [
            {
                "meta": asdict(r.meta),
                "n_questions": r.n_questions,
                "final_correct": r.final_correct,
                "final_incorrect": r.final_incorrect,
                "final_none": r.final_none,
                "judge_correct": r.judge_correct,
                "judge_incorrect": r.judge_incorrect,
                "judge_none": r.judge_none,
                "majority_correct": r.majority_correct,
                "majority_incorrect": r.majority_incorrect,
                "majority_none": r.majority_none,
            }
            for r in run_summaries
        ],
        "pooled": {str(k): dict(v) for k, v in pooled.items()},
        "majority_patterns": {ds: dict(c) for ds, c in maj_patterns.items()},
        "baseline_format": {ds: dict(c) for ds, c in baseline_format.items()},
        "debate_dynamics": {str(k): dict(v) for k, v in debate_dyn.items()},
        "judge_matrix": {str(k): dict(v) for k, v in judge_matrix.items()},
        "judge_override": {str(k): dict(v) for k, v in judge_override.items()},
        "trans_correctness": {str(k): dict(v) for k, v in trans_correctness.items()},
        "change_toward_prev_other_plurality": {
            str(k): dict(v) for k, v in change_toward_prev_other_majority.items()
        },
        "trans_choice_gpqa": {str(k): dict(v) for k, v in trans_choice_gpqa.items()},
        "per_question": {str(k): dict(v) for k, v in per_question.items()},
        "examples": examples,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2, sort_keys=True)

    # Build human-readable tables.md
    lines: list[str] = []
    lines.append("# Auto-Generated Tables\n")
    lines.append(f"- Generated at: `{out_summary['generated_at']}`")
    lines.append(f"- Results dir: `{results_dir}`\n")

    # Run summary table
    rows: list[list[str]] = []
    for r in sorted(run_summaries, key=lambda x: (x.meta.dataset, x.meta.mode, x.meta.path)):
        m = r.meta
        cfg = cfg_key(m)
        acc = fmt_ci(r.final_correct, r.n_questions)
        none_rate = fmt_pct(r.final_none / r.n_questions) if r.n_questions else "0.0%"
        extra = ""
        if m.mode == "debate":
            extra = (
                f"judge={fmt_pct((r.judge_correct or 0)/r.n_questions)} "
                f"maj={fmt_pct((r.majority_correct or 0)/r.n_questions)} "
            )
        rows.append(
            [
                m.dataset,
                m.mode,
                cfg,
                str(m.n or ""),
                str(m.seed or ""),
                acc,
                none_rate,
                extra,
                Path(m.path).name,
            ]
        )
    lines.append("## Per-Run Summary\n")
    lines.append(
        _md_table(
            [
                "Dataset",
                "Mode",
                "Cfg",
                "n",
                "Seed",
                "Final Acc",
                "Final None",
                "Debate extras",
                "File",
            ],
            rows,
        )
    )
    lines.append("")

    # Pooled stats table
    pooled_rows: list[list[str]] = []
    for (ds, mode, cfg), c in sorted(pooled.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        k_new = int(c["correct"])
        n = int(c["total"])
        none_new = int(c["none"])
        pooled_rows.append(
            [
                ds,
                mode,
                cfg,
                str(n),
                fmt_ci(k_new, n),
                fmt_pct(none_new / n if n else 0.0),
            ]
        )
    lines.append("## Pooled Accuracy by Dataset/Mode/Cfg\n")
    lines.append(
        _md_table(
            ["Dataset", "Mode", "Cfg", "Total Q", "Acc", "None"],
            pooled_rows,
        )
    )
    lines.append("")

    # Debate round dynamics summary
    lines.append("## Debate Round Dynamics\n")
    for (ds, cfg), totals in sorted(debate_round_total.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rows2: list[list[str]] = []
        for r in sorted(totals.keys()):
            tot = debate_round_total[(ds, cfg)][r]
            un = debate_round_agree[(ds, cfg)][r]
            ent_sum = debate_round_entropy_sum[(ds, cfg)][r]
            ent_n = debate_round_entropy_n[(ds, cfg)][r]
            ent = ent_sum / ent_n if ent_n else 0.0
            rows2.append([str(r + 1), str(tot), fmt_pct(un / tot if tot else 0.0), f"{ent:.3f}"])
        lines.append(f"### {ds} {cfg}\n")
        lines.append(_md_table(["Round", "Questions", "Unanimous", "Mean entropy (bits)"], rows2))
        lines.append("")

    # Judge conditioning (debate only)
    judge_cond_rows: list[list[str]] = []
    for (ds, cfg), c in sorted(judge_override.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = int(c.get("n_questions", 0))
        if n <= 0:
            continue
        rep = int(c.get("final_repeat_winner_exists", 0))
        rep_abs = int(c.get("final_repeat_winner_absent", 0))
        judge_rep_k = int(c.get("judge_correct_when_repeat_winner_exists", 0))
        judge_abs_k = int(c.get("judge_correct_when_repeat_winner_absent", 0))
        judge_in_final_n = int(c.get("judge_total_when_in_final_round_answers", 0))
        judge_in_final_k = int(c.get("judge_correct_when_in_final_round_answers", 0))
        judge_not_in_final_n = int(c.get("judge_total_when_not_in_final_round_answers", 0))
        judge_not_in_final_k = int(c.get("judge_correct_when_not_in_final_round_answers", 0))

        judge_cond_rows.append(
            [
                ds,
                cfg,
                str(n),
                fmt_pct(rep / n),
                fmt_pct(judge_rep_k / rep if rep else 0.0),
                fmt_pct(judge_abs_k / rep_abs if rep_abs else 0.0),
                fmt_pct(judge_in_final_k / judge_in_final_n if judge_in_final_n else 0.0),
                fmt_pct(judge_not_in_final_k / judge_not_in_final_n if judge_not_in_final_n else 0.0),
            ]
        )
    if judge_cond_rows:
        lines.append("## Judge Conditional Breakdown (Debate)\n")
        lines.append(
            _md_table(
                [
                    "Dataset",
                    "Cfg",
                    "Q",
                    "Repeat-winner exists",
                    "Judge acc (repeat-winner exists)",
                    "Judge acc (repeat-winner absent)",
                    "Judge acc (judge in final-round answers)",
                    "Judge acc (judge not in final-round answers)",
                ],
                judge_cond_rows,
            )
        )
        lines.append("")

    with open(out_dir / "tables.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    # Append key high-level findings from pooled stats
    pooled_md_rows: list[list[str]] = []
    for (ds, mode, cfg), c in sorted(pooled.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        k = int(c["correct"])
        n = int(c["total"])
        pooled_md_rows.append([ds, mode, cfg, f"{k}/{n}", fmt_ci(k, n)])
    append_findings_md(
        "### Pooled Accuracy (All Included Runs)\n\n"
        + _md_table(
            ["Dataset", "Mode", "Cfg", "k/n", "Acc (Wilson 95%)"],
            pooled_md_rows,
        )
        + "\n"
    )

    # Majority failure patterns (per dataset)
    for ds, c in maj_patterns.items():
        total_runs = sum(v for k, v in c.items() if k.startswith("n_samples="))
        # Extract unique-answers distribution
        uniq = sorted(
            [(k, v) for k, v in c.items() if k.startswith("unique_answers=")],
            key=lambda kv: int(kv[0].split("=")[1]),
        )
        uniq_table = _md_table(
            ["Unique answers among samples", "Count"],
            [[k.split("=")[1], str(v)] for k, v in uniq],
        )

        none_ct = int(c.get("final_none=1", 0))
        strict = int(c.get("strict_majority_exists=1", 0))
        append_findings_md(
            f"### Majority Voting Disagreement Patterns ({ds})\n\n"
            + f"- Records analyzed: {total_runs}\n"
            + f"- `final_answer=None`: {none_ct}/{total_runs} ({fmt_pct(none_ct/total_runs if total_runs else 0.0)})\n"
            + f"- Strict-majority exists (parsed samples): {strict}/{total_runs} ({fmt_pct(strict/total_runs if total_runs else 0.0)})\n\n"
            + uniq_table
            + "\n"
        )

        # Oracle-style upper bounds using the existing parsed samples:
        # - any_sample_correct: at least one parsed sample equals GT
        # - first_non_none_correct: always pick first parseable sample
        any_correct = int(c.get("any_sample_correct", 0))
        first_non_none_correct = int(c.get("first_non_none_correct", 0))
        append_findings_md(
            f"### Majority: Oracle-Style Bounds ({ds})\n\n"
            + _md_table(
                ["Metric", "Value"],
                [
                    ["Records", str(total_runs)],
                    ["Any sample correct", f"{any_correct}/{total_runs} ({fmt_pct(any_correct/total_runs if total_runs else 0.0)})"],
                    ["First non-None correct", f"{first_non_none_correct}/{total_runs} ({fmt_pct(first_non_none_correct/total_runs if total_runs else 0.0)})"],
                ],
            )
            + "\n"
        )

    # Baseline formatting/truncation signals
    fmt_rows: list[list[str]] = []
    for ds, c in sorted(baseline_format.items(), key=lambda kv: kv[0]):
        total_comp = int(c.get("total_completions", 0))
        total_rec = int(c.get("total_records", 0))
        with_box = int(c.get("completions_with_boxed_or_choice", 0))
        chars_sum = int(c.get("completion_chars_sum", 0))
        avg_len = (chars_sum / total_comp) if total_comp else 0.0
        fmt_rows.append(
            [
                ds,
                str(total_rec),
                str(total_comp),
                fmt_pct(with_box / total_comp if total_comp else 0.0),
                f"{avg_len:.0f}",
            ]
        )
    if fmt_rows:
        append_findings_md(
            "### Baseline Formatting/Truncation Signals (Single+Majority)\n\n"
            + _md_table(
                ["Dataset", "Records", "Completions", "Has \\boxed/explicit choice", "Avg chars/completion"],
                fmt_rows,
            )
            + "\n"
        )

    # Debate dynamics + judge behavior
    dyn_rows: list[list[str]] = []
    for (ds, cfg), c in sorted(debate_dyn.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n_q = int(c.get("n_questions", 0))
        if n_q <= 0:
            continue
        ever = int(c.get("ever_correct", 0))
        lost = int(c.get("lost_correct", 0))
        final_has = int(c.get("final_round_has_correct", 0))
        jc = int(c.get("judge_correct", 0))
        mc = int(c.get("majority_correct", 0))
        changes = int(c.get("answer_changes", 0))
        steps = int(c.get("answer_steps", 0))
        change_rate = changes / steps if steps else 0.0
        dyn_rows.append(
            [
                ds,
                cfg,
                str(n_q),
                fmt_pct(ever / n_q),
                fmt_pct(final_has / n_q),
                fmt_pct(lost / n_q),
                fmt_pct(jc / n_q),
                fmt_pct(mc / n_q),
                f"{100.0*change_rate:.1f}%",
            ]
        )
    append_findings_md(
        "### Debate Dynamics (Belief Change + GT Retention)\n\n"
        + _md_table(
            [
                "Dataset",
                "Cfg",
                "Q",
                "Ever correct (any agent/round)",
                "Correct in final round",
                "Correct lost (ever but not final)",
                "Judge acc",
                "Final-round majority acc",
                "Per-step answer change rate",
            ],
            dyn_rows,
        )
        + "\n"
    )

    # Judge behavior summary: overrides + rescue/harm counts.
    judge_rows: list[list[str]] = []
    for (ds, cfg), ov in sorted(judge_override.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        total = int(debate_dyn.get((ds, cfg), {}).get("n_questions", 0))
        if total <= 0:
            continue
        eq = int(ov.get("judge_equals_majority", 0))
        diff = int(ov.get("judge_differs_majority", 0))
        not_final = int(ov.get("judge_not_in_final_round_answers", 0))
        not_any = int(ov.get("judge_not_in_any_round_answers", 0))
        jm = judge_matrix.get((ds, cfg), {})
        rescue = int(jm.get("maj=0,judge=1", 0))
        harm = int(jm.get("maj=1,judge=0", 0))
        wrong_with_final_correct = int(debate_dyn[(ds, cfg)].get("judge_wrong_with_final_correct_present", 0))
        wrong_with_ever_correct = int(debate_dyn[(ds, cfg)].get("judge_wrong_with_ever_correct_present", 0))
        judge_rows.append(
            [
                ds,
                cfg,
                str(total),
                fmt_pct(diff / total),
                str(rescue),
                str(harm),
                fmt_pct(not_final / total),
                fmt_pct(not_any / total),
                fmt_pct(wrong_with_final_correct / total),
                fmt_pct(wrong_with_ever_correct / total),
            ]
        )
    if judge_rows:
        append_findings_md(
            "### Judge Behavior (Overrides, Rescues, Harms)\n\n"
            + _md_table(
                [
                    "Dataset",
                    "Cfg",
                    "Q",
                    "Override rate (judge != maj)",
                    "Rescues (maj wrong -> judge right)",
                    "Harms (maj right -> judge wrong)",
                    "Judge not in final-round answers",
                    "Judge not in any-round answers",
                    "Judge wrong despite final correct present",
                    "Judge wrong despite ever-correct present",
                ],
                judge_rows,
            )
            + "\n"
        )

    # Judge conditional breakdown: how the judge behaves when agents do/don't produce a repeat-winner.
    judge_cond_find_rows: list[list[str]] = []
    for (ds, cfg), c in sorted(judge_override.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = int(c.get("n_questions", 0))
        if n <= 0:
            continue
        rep = int(c.get("final_repeat_winner_exists", 0))
        rep_abs = int(c.get("final_repeat_winner_absent", 0))
        judge_rep_k = int(c.get("judge_correct_when_repeat_winner_exists", 0))
        judge_abs_k = int(c.get("judge_correct_when_repeat_winner_absent", 0))
        harm = int(c.get("judge_harm_over_repeat_winner_correct", 0))
        rescue_none = int(c.get("judge_rescue_from_repeat_winner_absent", 0))
        rescue_wrong = int(c.get("judge_rescue_from_repeat_winner_wrong", 0))
        judge_not_in_final = int(c.get("judge_not_in_final_round_answers", 0))
        judge_cond_find_rows.append(
            [
                ds,
                cfg,
                str(n),
                f"{rep}/{n}",
                f"{judge_rep_k}/{rep}" if rep else "0/0",
                f"{judge_abs_k}/{rep_abs}" if rep_abs else "0/0",
                str(harm),
                str(rescue_none),
                str(rescue_wrong),
                fmt_pct(judge_not_in_final / n),
            ]
        )
    if judge_cond_find_rows:
        append_findings_md(
            "### Judge Conditional Breakdown (Repeat-Winner vs No Repeat-Winner)\n\n"
            + _md_table(
                [
                    "Dataset",
                    "Cfg",
                    "Q",
                    "Repeat-winner exists",
                    "Judge correct (repeat-winner exists)",
                    "Judge correct (repeat-winner absent)",
                    "Harms over correct repeat-winner",
                    "Rescues from no repeat-winner",
                    "Rescues from wrong repeat-winner",
                    "Judge not in final-round answers",
                ],
                judge_cond_find_rows,
            )
            + "\n"
        )

    # Conformity proxy results
    conf_rows: list[list[str]] = []
    for (ds, cfg), c in sorted(change_toward_prev_other_majority.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        changed = int(c.get("changed", 0))
        to_other = int(c.get("changed_to_other_plurality", 0))
        away_other = int(c.get("changed_away_from_other_plurality", 0))
        conf_rows.append(
            [
                ds,
                cfg,
                str(changed),
                fmt_pct(to_other / changed if changed else 0.0),
                fmt_pct(away_other / changed if changed else 0.0),
            ]
        )
    append_findings_md(
        "### Conformity Proxy: Changes Toward Other-Agent Plurality\n\n"
        + _md_table(
            ["Dataset", "Cfg", "Total changes", "Changed to other plurality", "Changed away from other plurality"],
            conf_rows,
        )
        + "\n"
    )

    # Save examples (trim to small number per dataset)
    for ds in list(examples.keys()):
        # Keep top 12 only to avoid bloat.
        examples[ds] = examples[ds][:12]
    with open(out_dir / "examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, sort_keys=True)

    # Generate a lightweight case-studies markdown file by re-loading the referenced records.
    def load_record_from_run(run_name: str, orig_id: int | None) -> dict[str, Any] | None:
        # Determine dataset dir from filename prefix.
        if "_aime_" in run_name:
            p = results_dir / "aime25_quick" / run_name
        elif "_gpqa_" in run_name:
            p = results_dir / "gpqa_quick" / run_name
        else:
            return None
        if not p.exists() or orig_id is None:
            return None
        try:
            for rec in read_jsonl(p):
                if rec.get("orig_id") == orig_id:
                    return rec
        except Exception:
            return None
        return None

    def q_excerpt(rec: dict[str, Any] | None, n: int = 420) -> str:
        if not rec:
            return ""
        q = rec.get("question") or ""
        if not isinstance(q, str):
            q = str(q)
        q = q.strip().replace("\n", " ")
        if len(q) <= n:
            return q
        return q[:n] + "…"

    def md_case(title: str, items: list[dict[str, Any]]) -> str:
        out: list[str] = []
        out.append(f"## {title}\n")
        for ex in items:
            run = ex.get("run", "")
            cfg = ex.get("cfg", "")
            oid = ex.get("orig_id", "")
            gt = ex.get("gt", "")
            rec = load_record_from_run(str(run), int(oid) if isinstance(oid, int) else None)
            out.append(f"### {run} (cfg={cfg}, orig_id={oid})\n")
            out.append(f"- Ground truth: `{gt}`")
            if "final_majority" in ex:
                out.append(f"- Final majority: `{ex.get('final_majority')}`")
            if "final_judge" in ex:
                out.append(f"- Final judge: `{ex.get('final_judge')}`")
            if "unanimous_final" in ex:
                out.append(f"- Unanimous final: `{ex.get('unanimous_final')}`")
            if "judge" in ex:
                out.append(f"- Judge: `{ex.get('judge')}`")
            qe = q_excerpt(rec)
            if qe:
                out.append(f"- Question excerpt: {qe}")
            if ex.get("agent_tail"):
                out.append("\n**Agent tail**\n")
                out.append("```text")
                out.append(str(ex.get("agent_tail", "")).strip())
                out.append("```\n")
            if ex.get("judge_tail"):
                out.append("\n**Judge tail**\n")
                out.append("```text")
                out.append(str(ex.get("judge_tail", "")).strip())
                out.append("```\n")
        return "\n".join(out).rstrip() + "\n"

    case_md: list[str] = []
    case_md.append("# Auto-Generated Case Studies\n")
    case_md.append(f"- Generated at: `{out_summary['generated_at']}`")
    case_md.append(f"- Results dir: `{results_dir}`\n")
    case_md.append(
        "This file contains **short excerpt-based case studies** automatically selected from the runs.\n"
        "It is meant to support qualitative analysis without pasting full transcripts.\n"
    )
    case_md.append("")
    case_md.append(md_case("AIME25: Judge Rescues", examples.get("aime25", [])[:6]))
    case_md.append(md_case("AIME25: Judge Harms", examples.get("aime25_judge_harm", [])[:6]))
    case_md.append(md_case("AIME25: Unanimous-Wrong Finals", examples.get("aime25_unanimous_wrong", [])[:6]))
    case_md.append(md_case("AIME25: Lost-Correct (Correct Appears Then Disappears)", examples.get("aime25_lost_correct", [])[:6]))
    case_md.append(md_case("GPQA: Judge Rescues", examples.get("gpqa", [])[:6]))
    case_md.append(md_case("GPQA: Judge Harms", examples.get("gpqa_judge_harm", [])[:6]))
    case_md.append(md_case("GPQA: Unanimous-Wrong Finals", examples.get("gpqa_unanimous_wrong", [])[:6]))
    case_md.append(md_case("GPQA: Lost-Correct (Correct Appears Then Disappears)", examples.get("gpqa_lost_correct", [])[:6]))
    with open(out_dir / "case_studies.md", "w", encoding="utf-8") as f:
        f.write("\n".join(case_md).rstrip() + "\n")

    append_findings_md(
        "### Qualitative Case Studies Generated\n\n"
        + f"- Wrote: `{out_dir / 'case_studies.md'}`\n"
        + f"- Wrote: `{out_dir / 'examples.json'}`\n"
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    analyze(Path(args.results_dir), Path(args.out_dir))
    print(f"Wrote: {DEFAULT_OUT_DIR / 'summary.json'}")
    print(f"Wrote: {DEFAULT_OUT_DIR / 'tables.md'}")
    print(f"Appended findings: {FINDINGS_LOG}")


if __name__ == "__main__":
    main()
