#!/usr/bin/env python3
"""
Generate expanded, analysis-friendly case studies from debate/single/majority jsonl outputs.

Inputs:
  - ~/debug_majority_debate/_autogen/examples.json (produced by analyze_results.py)
  - /home/ubuntu/multi-agent-attack/multiagent_debate/results/{aime25_quick,gpqa_quick}/*.jsonl

Outputs:
  - ~/debug_majority_debate/_autogen/case_studies_expanded.md

This file is meant to be a *source* for writing narrative in ANALYSIS_REPORT.md:
it includes per-round parsed answer matrices and short excerpts, but avoids dumping full transcripts.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HOME = Path.home()
DEFAULT_RESULTS_DIR = Path("/home/ubuntu/multi-agent-attack/multiagent_debate/results")
DEFAULT_OUT_DIR = HOME / "debug_majority_debate" / "_autogen"
TARGET_MODEL_SUBSTR = "Qwen_Qwen3-8B"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def short(s: Any, n: int = 360) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def tail(s: Any, n: int = 360) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip()
    if len(s) <= n:
        return s
    return "…" + s[-(n - 1) :]


def dataset_from_run(run_name: str) -> str:
    if "_aime_" in run_name:
        return "aime25"
    if "_gpqa_" in run_name:
        return "gpqa"
    raise ValueError(f"Cannot infer dataset from run name: {run_name}")


def should_include_run_name(run_name: str) -> bool:
    return TARGET_MODEL_SUBSTR in str(run_name)


def load_record(results_dir: Path, run_name: str, orig_id: int) -> tuple[Path, dict[str, Any]] | None:
    dataset = dataset_from_run(run_name)
    sub = "aime25_quick" if dataset == "aime25" else "gpqa_quick"
    path = results_dir / sub / run_name
    if not path.exists():
        return None
    for rec in read_jsonl(path):
        if rec.get("orig_id") == orig_id:
            return path, rec
    return None


def load_parsers():
    import sys

    if str(HOME) not in sys.path:
        sys.path.insert(0, str(HOME))
    import debug_majority_debate.aime25 as aime25
    import debug_majority_debate.gpqa as gpqa

    return aime25, gpqa


def parse_debate_round_answers(rec: dict[str, Any], dataset: str, *, aime25_mod: Any, gpqa_mod: Any) -> list[list[str | None]]:
    stored = rec.get("agent_round_parsed_answers")
    if isinstance(stored, list) and stored:
        return stored

    raw_task = rec.get("raw_task") or {}
    n_agents = int(rec["n_agents"])
    n_rounds = int(rec["n_rounds"])
    agent_responses = rec["agent_responses"]

    out: list[list[str | None]] = []
    for a in range(n_agents):
        msgs = agent_responses[a]
        seq: list[str | None] = []
        for r in range(n_rounds):
            idx = 2 * r + 1
            text = msgs[idx].get("content", "") if idx < len(msgs) else ""
            if dataset == "aime25":
                seq.append(aime25_mod.parse_answer(text, raw_task))
            else:
                seq.append(gpqa_mod.parse_answer(text, raw_task))
        out.append(seq)
    return out


def correctness_marker(ans: str | None, gt: str) -> str:
    if ans is None:
        return "∅"
    if str(ans) == str(gt):
        return f"{ans}✓"
    return f"{ans}✗"


def round_summary(answers: list[list[str | None]], gt: str) -> list[dict[str, Any]]:
    if not answers:
        return []
    n_rounds = len(answers[0])
    out: list[dict[str, Any]] = []
    for r in range(n_rounds):
        col = [row[r] for row in answers]
        counts = Counter(col)
        uniq = len(counts)
        top = counts.most_common(1)[0]
        any_correct = any(a is not None and str(a) == str(gt) for a in col)
        out.append(
            {
                "round": r + 1,
                "unique": uniq,
                "mode_answer": top[0],
                "mode_count": top[1],
                "any_correct": any_correct,
            }
        )
    return out


def completion_has_boxed_or_choice(text: str) -> bool:
    if "\\boxed" in text or "\\fbox" in text:
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tail_lines = lines[-8:] if len(lines) > 8 else lines
    tail_txt = "\n".join(tail_lines)
    if re.search(r"(?i)\b(?:final\s+answer|answer)\b[^A-D]*\(?\s*([ABCD])\s*\)?\b", tail_txt):
        return True
    last = tail_lines[-1] if tail_lines else ""
    if re.fullmatch(r"(?i)\(?\s*([ABCD])\s*\)?", last):
        return True
    return False


def collect_truncation_cases(results_dir: Path, *, max_cases: int = 6) -> list[dict[str, Any]]:
    """
    Select GPQA baseline cases (single/majority) where parsing returned None
    and completion lacks boxed/explicit choice.
    """
    cases: list[dict[str, Any]] = []
    for sub in ["gpqa_quick"]:
        for path in sorted((results_dir / sub).glob("single_*.jsonl")):
            if not should_include_run_name(path.name):
                continue
            for rec in read_jsonl(path):
                if rec.get("final_answer") is not None:
                    continue
                comps = rec.get("sample_completions") or []
                if not comps:
                    continue
                comp0 = str(comps[0])
                if completion_has_boxed_or_choice(comp0):
                    continue
                cases.append(
                    {
                        "run": path.name,
                        "orig_id": rec.get("orig_id"),
                        "gt": rec.get("answer"),
                        "question_excerpt": short(rec.get("question", ""), 360),
                        "completion_tail": tail(comp0, 420),
                    }
                )
                if len(cases) >= max_cases:
                    return cases
    return cases


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument(
        "--target-model-substr",
        default=None,
        help=(
            "Only include run filenames containing this substring "
            "(default: env TARGET_MODEL_SUBSTR or 'Qwen_Qwen3-8B')."
        ),
    )
    ap.add_argument("--max-per-category", type=int, default=8)
    ap.add_argument("--max-truncation", type=int, default=6)
    args = ap.parse_args()

    global TARGET_MODEL_SUBSTR
    TARGET_MODEL_SUBSTR = (
        args.target_model_substr
        or str(__import__("os").environ.get("TARGET_MODEL_SUBSTR") or "").strip()
        or TARGET_MODEL_SUBSTR
    )

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples_path = out_dir / "examples.json"
    examples = json.load(open(examples_path, "r", encoding="utf-8"))

    aime25_mod, gpqa_mod = load_parsers()

    md: list[str] = []
    md.append("# Expanded Case Studies (Auto-Generated)\n")
    md.append(f"- Generated at: `{now_iso()}`")
    md.append(f"- Results dir: `{results_dir}`")
    md.append(f"- Source examples: `{examples_path}`\n")
    md.append(
        "Legend:\n"
        "- `∅` means the parser extracted no final answer for that agent/round.\n"
        "- `X✓` means parsed answer `X` equals ground truth.\n"
        "- `X✗` means parsed answer `X` does not equal ground truth.\n"
    )
    md.append("")

    ordered_keys = [
        "aime25",
        "aime25_judge_harm",
        "aime25_unanimous_wrong",
        "aime25_lost_correct",
        "gpqa",
        "gpqa_judge_harm",
        "gpqa_unanimous_wrong",
        "gpqa_lost_correct",
    ]

    for key in ordered_keys:
        items = examples.get(key, [])
        if not isinstance(items, list) or not items:
            continue
        md.append(f"## {key}\n")
        for ex in items[: args.max_per_category]:
            run = ex.get("run")
            cfg = ex.get("cfg", "")
            orig_id = ex.get("orig_id")
            gt = ex.get("gt")
            if run is None or orig_id is None or gt is None:
                continue
            if not should_include_run_name(str(run)):
                continue

            loaded = load_record(results_dir, str(run), int(orig_id))
            if loaded is None:
                continue
            path, rec = loaded
            dataset = dataset_from_run(str(run))

            md.append(f"### {path.name} (cfg={cfg}, orig_id={orig_id})\n")
            md.append(f"- Dataset: `{dataset}`")
            md.append(f"- Ground truth: `{gt}`")

            # Debate-specific info
            if rec.get("mode") == "debate":
                raw_task = rec.get("raw_task") or {}
                stored_maj = rec.get("final_majority_answer")
                stored_maj_correct = rec.get("final_majority_correct")
                stored_judge = rec.get("final_judge_answer")
                stored_judge_correct = rec.get("final_judge_correct")
                md.append(f"- Final majority (stored): `{stored_maj}` (correct={stored_maj_correct})")
                md.append(f"- Final judge (stored): `{stored_judge}` (correct={stored_judge_correct})")

                answers = parse_debate_round_answers(rec, dataset, aime25_mod=aime25_mod, gpqa_mod=gpqa_mod)
                n_agents = int(rec["n_agents"])
                n_rounds = int(rec["n_rounds"])

                # Agent x round matrix
                headers = ["Agent"] + [f"R{r}" for r in range(1, n_rounds + 1)]
                rows: list[list[str]] = []
                for a in range(n_agents):
                    rows.append([str(a)] + [correctness_marker(answers[a][r], str(gt)) for r in range(n_rounds)])
                md.append("\n**Parsed answers by agent/round**\n")
                md.append(md_table(headers, rows))

                rs = round_summary(answers, str(gt))
                rs_rows = [
                    [
                        str(x["round"]),
                        str(x["unique"]),
                        str(x["mode_answer"]),
                        str(x["mode_count"]),
                        "yes" if x["any_correct"] else "no",
                    ]
                    for x in rs
                ]
                md.append("\n**Round summary**\n")
                md.append(md_table(["Round", "Unique answers", "Plurality", "Plurality count", "Any correct present"], rs_rows))

                md.append("\n**Agent excerpt (tail)**\n")
                md.append("```text")
                md.append(tail(ex.get("agent_tail", ""), 520))
                md.append("```\n")
                jraw = (rec.get("judge_trace") or {}).get("judge_raw_response", "")
                md.append("\n**Judge excerpt (tail)**\n")
                md.append("```text")
                md.append(tail(ex.get("judge_tail", jraw), 520))
                md.append("```\n")

            else:
                # baseline records: show sample completion tail and parsed answer
                md.append(f"- Mode: `{rec.get('mode')}`")
                md.append(f"- final_answer (stored): `{rec.get('final_answer')}` (correct={rec.get('final_correct')})")
                comp0 = ""
                comps = rec.get("sample_completions") or []
                if comps:
                    comp0 = str(comps[0])
                md.append("**Completion (tail)**\n")
                md.append("```text")
                md.append(tail(comp0, 520))
                md.append("```\n")

    # Add GPQA truncation / missing-final-answer cases.
    trunc = collect_truncation_cases(results_dir, max_cases=args.max_truncation)
    if trunc:
        md.append("## gpqa_truncation_missing_final_answer\n")
        for tcase in trunc:
            md.append(f"### {tcase['run']} (orig_id={tcase['orig_id']})\n")
            md.append(f"- Ground truth: `{tcase['gt']}`")
            md.append(f"- Question excerpt: {tcase['question_excerpt']}\n")
            md.append("**Completion tail (no boxed / no explicit choice)**\n")
            md.append("```text")
            md.append(tcase["completion_tail"])
            md.append("```\n")

    out_path = out_dir / "case_studies_expanded.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md).rstrip() + "\n")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
