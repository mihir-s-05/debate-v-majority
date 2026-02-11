#!/usr/bin/env python3
"""
Interactive Dashboard for Model Run Results.

A fully customizable Streamlit dashboard that loads JSONL result files from
aime25/ and gpqa/ subfolders and provides:
  - Overview statistics across all runs
  - Run comparison by model, seed, mode
  - Transcript viewer with full conversation display
  - Debate analysis with round-by-round behavior
  - Per-model statistics with subset filtering

Usage:
    streamlit run dashboard.py -- --results-dir /path/to/results
    python -m streamlit run dashboard.py -- --results-dir /path/to/results
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Filename pattern: {mode}_{dataset}_{config}_{run_tag}_{model_tag}.jsonl
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


@dataclass
class RunMeta:
    path: str
    filename: str
    dataset: str
    mode: str
    n: int | None = None
    seed: int | None = None
    n_samples: int | None = None
    n_agents: int | None = None
    n_rounds: int | None = None
    tag_all: bool = False
    model_tag: str | None = None
    ts: str | None = None


@dataclass
class RunData:
    meta: RunMeta
    records: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_run_meta(path: Path) -> RunMeta | None:
    """Parse filename into RunMeta; returns None if unrecognised."""
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
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
        filename=path.name,
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


def infer_model_tag_from_siblings(path: Path, meta: RunMeta) -> str | None:
    """Infer model tag from sibling files with matching seed+timestamp."""
    if meta.model_tag:
        return meta.model_tag
    if meta.seed is None or meta.ts is None:
        return None
    tags: set[str] = set()
    for sib in path.parent.glob(f"*seed{meta.seed}*_{meta.ts}_*.jsonl"):
        if sib == path:
            continue
        sib_meta = parse_run_meta(sib)
        if sib_meta and sib_meta.model_tag:
            tags.add(sib_meta.model_tag)
    return next(iter(tags)) if len(tags) == 1 else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading result files...")
def load_all_runs(results_dir: str) -> list[RunData]:
    """Discover and load all .jsonl result files under results_dir."""
    root = Path(results_dir)
    runs: list[RunData] = []
    jsonl_files: list[Path] = []

    # Search in expected subfolders and root
    search_dirs = [root]
    for sub in ("aime25_quick", "gpqa_quick", "aime25", "gpqa"):
        d = root / sub
        if d.is_dir():
            search_dirs.append(d)

    seen: set[str] = set()
    for d in search_dirs:
        for f in sorted(d.glob("*.jsonl")):
            real = str(f.resolve())
            if real not in seen:
                seen.add(real)
                jsonl_files.append(f)

    for f in jsonl_files:
        meta = parse_run_meta(f)
        if meta is None:
            continue
        if meta.model_tag is None:
            inferred = infer_model_tag_from_siblings(f, meta)
            if inferred:
                meta.model_tag = inferred
        try:
            records = read_jsonl(f)
        except Exception:
            continue
        if records:
            runs.append(RunData(meta=meta, records=records))
    return runs


def runs_to_summary_df(runs: list[RunData]) -> pd.DataFrame:
    """Build a summary DataFrame with one row per run."""
    rows = []
    for rd in runs:
        m = rd.meta
        n_q = len(rd.records)
        correct = sum(1 for r in rd.records if r.get("final_correct"))
        acc = correct / n_q if n_q else 0.0
        row = {
            "filename": m.filename,
            "dataset": m.dataset,
            "mode": m.mode,
            "model": m.model_tag or "unknown",
            "seed": m.seed,
            "n_questions": n_q,
            "n_correct": correct,
            "accuracy": acc,
            "n_agents": m.n_agents,
            "n_rounds": m.n_rounds,
            "n_samples": m.n_samples,
            "subset_n": m.n,
            "all": m.tag_all,
            "timestamp": m.ts,
        }
        if m.mode == "debate":
            judge_correct = sum(1 for r in rd.records if r.get("final_judge_correct"))
            maj_correct = sum(1 for r in rd.records if r.get("final_majority_correct"))
            row["judge_accuracy"] = judge_correct / n_q if n_q else 0.0
            row["majority_accuracy"] = maj_correct / n_q if n_q else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def records_to_question_df(records: list[dict[str, Any]], meta: RunMeta) -> pd.DataFrame:
    """Build a per-question DataFrame from records of a single run."""
    rows = []
    for rec in records:
        row = {
            "subset_id": rec.get("subset_id"),
            "orig_id": rec.get("orig_id"),
            "question": (rec.get("question") or "")[:200],
            "ground_truth": rec.get("answer"),
            "final_answer": rec.get("final_answer"),
            "correct": bool(rec.get("final_correct")),
        }
        if meta.mode == "debate":
            row["judge_answer"] = rec.get("final_judge_answer")
            row["judge_correct"] = bool(rec.get("final_judge_correct"))
            row["majority_answer"] = rec.get("final_majority_answer")
            row["majority_correct"] = bool(rec.get("final_majority_correct"))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


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


def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------


def page_overview(runs: list[RunData], summary_df: pd.DataFrame):
    st.header("Overview")

    if summary_df.empty:
        st.warning("No runs loaded.")
        return

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs", len(runs))
    col2.metric("Datasets", summary_df["dataset"].nunique())
    col3.metric("Models", summary_df["model"].nunique())
    col4.metric("Total Questions Evaluated", int(summary_df["n_questions"].sum()))

    st.subheader("Accuracy by Dataset and Mode")

    # Grouped bar chart
    grouped = (
        summary_df.groupby(["dataset", "mode"])
        .agg(
            total_q=("n_questions", "sum"),
            total_correct=("n_correct", "sum"),
            n_runs=("filename", "count"),
        )
        .reset_index()
    )
    grouped["accuracy"] = grouped["total_correct"] / grouped["total_q"]
    grouped["accuracy_pct"] = grouped["accuracy"].apply(lambda x: f"{x*100:.1f}%")

    fig = px.bar(
        grouped,
        x="dataset",
        y="accuracy",
        color="mode",
        barmode="group",
        text="accuracy_pct",
        title="Pooled Accuracy by Dataset & Mode",
        labels={"accuracy": "Accuracy", "dataset": "Dataset"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy by model
    st.subheader("Accuracy by Model")
    model_grouped = (
        summary_df.groupby(["model", "mode"])
        .agg(total_q=("n_questions", "sum"), total_correct=("n_correct", "sum"))
        .reset_index()
    )
    model_grouped["accuracy"] = model_grouped["total_correct"] / model_grouped["total_q"]
    model_grouped["accuracy_pct"] = model_grouped["accuracy"].apply(lambda x: f"{x*100:.1f}%")

    fig2 = px.bar(
        model_grouped,
        x="model",
        y="accuracy",
        color="mode",
        barmode="group",
        text="accuracy_pct",
        title="Pooled Accuracy by Model & Mode",
        labels={"accuracy": "Accuracy", "model": "Model"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig2.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

    # Run table
    st.subheader("All Runs")
    display_cols = [
        "filename", "dataset", "mode", "model", "seed", "n_questions",
        "n_correct", "accuracy", "n_agents", "n_rounds", "n_samples",
        "subset_n", "timestamp",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    st.dataframe(
        summary_df[display_cols].style.format({"accuracy": "{:.1%}"}),
        use_container_width=True,
        height=400,
    )


# ---------------------------------------------------------------------------
# Page: Run Comparison
# ---------------------------------------------------------------------------


def page_run_comparison(runs: list[RunData], summary_df: pd.DataFrame):
    st.header("Run Comparison")

    if summary_df.empty:
        st.warning("No runs loaded.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_models = st.multiselect(
            "Models", sorted(summary_df["model"].unique()), default=sorted(summary_df["model"].unique())
        )
    with col2:
        sel_datasets = st.multiselect(
            "Datasets", sorted(summary_df["dataset"].unique()), default=sorted(summary_df["dataset"].unique())
        )
    with col3:
        sel_modes = st.multiselect(
            "Modes", sorted(summary_df["mode"].unique()), default=sorted(summary_df["mode"].unique())
        )

    filtered = summary_df[
        (summary_df["model"].isin(sel_models))
        & (summary_df["dataset"].isin(sel_datasets))
        & (summary_df["mode"].isin(sel_modes))
    ]

    if filtered.empty:
        st.info("No runs match the selected filters.")
        return

    # Comparison axis
    compare_by = st.selectbox("Compare by", ["seed", "model", "mode", "subset_n", "timestamp"])

    st.subheader("Accuracy Comparison")

    # Build comparison chart
    if compare_by in filtered.columns:
        filtered_plot = filtered.copy()
        filtered_plot[compare_by] = filtered_plot[compare_by].astype(str)
        filtered_plot["label"] = (
            filtered_plot["dataset"] + " / " + filtered_plot["mode"]
            + ((" / " + filtered_plot["model"]) if compare_by != "model" else "")
        )
        fig = px.bar(
            filtered_plot,
            x=compare_by,
            y="accuracy",
            color="label",
            barmode="group",
            title=f"Accuracy by {compare_by}",
            labels={"accuracy": "Accuracy"},
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # Seed-based comparison for same model
    st.subheader("Seed Stability")
    st.markdown("Compare how accuracy varies across seeds for the same model/mode/dataset.")

    seed_data = filtered[filtered["seed"].notna()].copy()
    if not seed_data.empty:
        seed_data["config_label"] = seed_data["dataset"] + " / " + seed_data["mode"] + " / " + seed_data["model"]
        seed_data["seed_str"] = seed_data["seed"].astype(str)
        fig_seed = px.strip(
            seed_data,
            x="config_label",
            y="accuracy",
            color="seed_str",
            title="Accuracy Distribution by Seed",
            labels={"accuracy": "Accuracy", "config_label": "Config", "seed_str": "Seed"},
        )
        fig_seed.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_seed, use_container_width=True)

    # Debate: Judge vs Majority comparison
    debate_runs = filtered[filtered["mode"] == "debate"]
    if not debate_runs.empty and "judge_accuracy" in debate_runs.columns:
        st.subheader("Debate: Judge vs Majority Accuracy")
        debate_melted = debate_runs.melt(
            id_vars=["filename", "dataset", "model", "seed"],
            value_vars=["judge_accuracy", "majority_accuracy"],
            var_name="method",
            value_name="acc",
        )
        debate_melted["method"] = debate_melted["method"].str.replace("_accuracy", "")
        debate_melted["label"] = debate_melted["dataset"] + " / " + debate_melted["model"]
        fig_jm = px.bar(
            debate_melted,
            x="label",
            y="acc",
            color="method",
            barmode="group",
            title="Judge vs Majority Accuracy",
            color_discrete_map={"judge": "#636EFA", "majority": "#EF553B"},
        )
        fig_jm.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_jm, use_container_width=True)

    # Detailed table
    st.subheader("Filtered Runs Table")
    cols = ["filename", "dataset", "mode", "model", "seed", "n_questions", "accuracy"]
    if "judge_accuracy" in filtered.columns:
        cols += ["judge_accuracy", "majority_accuracy"]
    cols = [c for c in cols if c in filtered.columns]
    fmt = {c: "{:.1%}" for c in cols if "accuracy" in c}
    st.dataframe(filtered[cols].style.format(fmt), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Transcript Viewer
# ---------------------------------------------------------------------------


def page_transcript_viewer(runs: list[RunData], summary_df: pd.DataFrame):
    st.header("Transcript Viewer")

    if not runs:
        st.warning("No runs loaded.")
        return

    # Run selection
    run_labels = {
        f"{rd.meta.filename} ({rd.meta.mode} | {rd.meta.dataset} | {rd.meta.model_tag or 'unknown'})": i
        for i, rd in enumerate(runs)
    }
    selected_label = st.selectbox("Select Run", list(run_labels.keys()))
    if selected_label is None:
        return
    rd = runs[run_labels[selected_label]]
    meta = rd.meta

    # Question selection
    q_options = {
        f"Q{rec.get('subset_id', i)} (orig={rec.get('orig_id', '?')}) | GT={rec.get('answer', '?')} | Correct={rec.get('final_correct', '?')}": i
        for i, rec in enumerate(rd.records)
    }
    selected_q_label = st.selectbox("Select Question", list(q_options.keys()))
    if selected_q_label is None:
        return
    rec = rd.records[q_options[selected_q_label]]

    # Display question
    st.subheader("Question")
    st.text_area("", rec.get("question", ""), height=200, disabled=True, key="q_text")

    col1, col2, col3 = st.columns(3)
    col1.metric("Ground Truth", str(rec.get("answer", "N/A")))
    col2.metric("Final Answer", str(rec.get("final_answer", "N/A")))
    correct = rec.get("final_correct", False)
    col3.metric("Correct", "Yes" if correct else "No")

    if meta.mode == "debate":
        _render_debate_transcript(rec, meta)
    elif meta.mode in ("single", "majority"):
        _render_sampled_transcript(rec, meta)


def _render_sampled_transcript(rec: dict[str, Any], meta: RunMeta):
    """Render single/majority transcript."""
    completions = rec.get("sample_completions") or []
    parsed_answers = rec.get("sample_parsed_answers") or []

    st.subheader(f"Completions ({len(completions)} samples)")
    for i, comp in enumerate(completions):
        parsed = parsed_answers[i] if i < len(parsed_answers) else None
        with st.expander(f"Sample {i+1} | Parsed Answer: {parsed}", expanded=(i == 0)):
            st.markdown(f"```\n{comp}\n```")


def _render_debate_transcript(rec: dict[str, Any], meta: RunMeta):
    """Render debate transcript with round-by-round view."""
    agent_responses = rec.get("agent_responses") or []
    n_agents = rec.get("n_agents", len(agent_responses))
    n_rounds = rec.get("n_rounds", 0)
    agent_round_answers = rec.get("agent_round_parsed_answers") or []

    # Debate-specific metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Judge Answer", str(rec.get("final_judge_answer", "N/A")))
        judge_correct = rec.get("final_judge_correct", False)
        st.metric("Judge Correct", "Yes" if judge_correct else "No")
    with col2:
        st.metric("Majority Answer", str(rec.get("final_majority_answer", "N/A")))
        maj_correct = rec.get("final_majority_correct", False)
        st.metric("Majority Correct", "Yes" if maj_correct else "No")

    # Answer evolution table
    if agent_round_answers:
        st.subheader("Answer Evolution by Round")
        gt = rec.get("answer")
        evo_data = []
        for a_idx, agent_answers in enumerate(agent_round_answers):
            for r_idx, ans in enumerate(agent_answers):
                evo_data.append({
                    "Agent": f"Agent {a_idx+1}",
                    "Round": r_idx + 1,
                    "Answer": str(ans) if ans is not None else "None",
                    "Correct": str(ans) == str(gt) if ans is not None and gt is not None else False,
                })
        if evo_data:
            evo_df = pd.DataFrame(evo_data)
            # Pivot for nice display
            pivot = evo_df.pivot(index="Agent", columns="Round", values="Answer")
            st.dataframe(pivot, use_container_width=True)

            # Color-coded heatmap
            correct_pivot = evo_df.pivot(index="Agent", columns="Round", values="Correct")
            fig_heat = go.Figure(data=go.Heatmap(
                z=correct_pivot.values.astype(int),
                x=[f"Round {c}" for c in correct_pivot.columns],
                y=correct_pivot.index.tolist(),
                colorscale=[[0, "#ff6b6b"], [1, "#51cf66"]],
                showscale=False,
                text=pivot.values,
                texttemplate="%{text}",
                textfont={"size": 14},
            ))
            fig_heat.update_layout(
                title="Answer Correctness by Agent & Round",
                xaxis_title="Round",
                yaxis_title="Agent",
                height=max(200, 60 * n_agents + 100),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # Agent transcripts
    st.subheader("Agent Transcripts")
    view_mode = st.radio("View mode", ["By Round", "By Agent"], horizontal=True, key="debate_view_mode")

    if view_mode == "By Round":
        for r in range(n_rounds):
            with st.expander(f"Round {r+1}", expanded=(r == n_rounds - 1)):
                for a_idx, agent_ctx in enumerate(agent_responses):
                    assistant_idx = 2 * r + 1
                    if assistant_idx < len(agent_ctx):
                        content = agent_ctx[assistant_idx].get("content", "")
                        parsed = None
                        if a_idx < len(agent_round_answers) and r < len(agent_round_answers[a_idx]):
                            parsed = agent_round_answers[a_idx][r]
                        st.markdown(f"**Agent {a_idx+1}** | Answer: `{parsed}`")
                        st.text_area("", content, height=200, disabled=True,
                                     key=f"agent_{a_idx}_round_{r}")
    else:
        for a_idx, agent_ctx in enumerate(agent_responses):
            with st.expander(f"Agent {a_idx+1}", expanded=(a_idx == 0)):
                for r in range(n_rounds):
                    assistant_idx = 2 * r + 1
                    if assistant_idx < len(agent_ctx):
                        content = agent_ctx[assistant_idx].get("content", "")
                        parsed = None
                        if a_idx < len(agent_round_answers) and r < len(agent_round_answers[a_idx]):
                            parsed = agent_round_answers[a_idx][r]
                        st.markdown(f"**Round {r+1}** | Answer: `{parsed}`")
                        st.text_area("", content, height=200, disabled=True,
                                     key=f"agent_{a_idx}_round_{r}_byagent")

    # Judge trace
    judge_trace = rec.get("judge_trace") or {}
    if judge_trace:
        with st.expander("Judge Trace", expanded=False):
            st.markdown(f"**Judge Model**: {judge_trace.get('judge_model', 'N/A')}")
            st.markdown(f"**Parsed Answer**: `{judge_trace.get('judge_parsed_answer', 'N/A')}`")
            st.markdown(f"**Parse Failed**: {judge_trace.get('judge_parse_failed', False)}")
            st.markdown(f"**Used Fallback**: {judge_trace.get('judge_used_fallback', False)}")
            raw_resp = judge_trace.get("judge_raw_response", "")
            if raw_resp:
                st.text_area("Judge Raw Response", raw_resp, height=300, disabled=True, key="judge_raw")
            retry_resp = judge_trace.get("judge_retry_raw_response")
            if retry_resp:
                st.text_area("Judge Retry Response", retry_resp, height=200, disabled=True, key="judge_retry")


# ---------------------------------------------------------------------------
# Page: Debate Analysis
# ---------------------------------------------------------------------------


def page_debate_analysis(runs: list[RunData], summary_df: pd.DataFrame):
    st.header("Debate Analysis")

    debate_runs = [rd for rd in runs if rd.meta.mode == "debate"]
    if not debate_runs:
        st.warning("No debate runs found.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        datasets = sorted(set(rd.meta.dataset for rd in debate_runs))
        sel_ds = st.multiselect("Datasets", datasets, default=datasets, key="da_ds")
    with col2:
        models = sorted(set(rd.meta.model_tag or "unknown" for rd in debate_runs))
        sel_models = st.multiselect("Models", models, default=models, key="da_models")

    filtered_runs = [
        rd for rd in debate_runs
        if rd.meta.dataset in sel_ds
        and (rd.meta.model_tag or "unknown") in sel_models
    ]

    if not filtered_runs:
        st.info("No debate runs match the filters.")
        return

    # Round-by-round agreement analysis
    st.subheader("Round-by-Round Agreement")

    round_data = []
    for rd in filtered_runs:
        for rec in rd.records:
            answers = rec.get("agent_round_parsed_answers") or []
            if not answers:
                continue
            gt = rec.get("answer")
            n_agents = len(answers)
            n_rounds = len(answers[0]) if answers else 0
            for r in range(n_rounds):
                round_answers = [answers[a][r] for a in range(n_agents) if r < len(answers[a])]
                counts = Counter(round_answers)
                unanimous = len(counts) == 1
                has_correct = any(str(a) == str(gt) for a in round_answers if a is not None)
                ent = entropy_from_counts(counts)
                round_data.append({
                    "dataset": rd.meta.dataset,
                    "model": rd.meta.model_tag or "unknown",
                    "seed": rd.meta.seed,
                    "round": r + 1,
                    "unanimous": int(unanimous),
                    "has_correct": int(has_correct),
                    "entropy": ent,
                    "n_unique": len(set(round_answers)),
                    "n_none": sum(1 for a in round_answers if a is None),
                })

    if round_data:
        rdf = pd.DataFrame(round_data)

        # Agreement rate by round
        round_agg = rdf.groupby(["dataset", "model", "round"]).agg(
            unanimous_rate=("unanimous", "mean"),
            correct_rate=("has_correct", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_unique=("n_unique", "mean"),
        ).reset_index()

        round_agg["label"] = round_agg["dataset"] + " / " + round_agg["model"]

        fig_agree = px.line(
            round_agg, x="round", y="unanimous_rate", color="label",
            markers=True, title="Unanimous Agreement Rate by Round",
            labels={"unanimous_rate": "Unanimous Rate", "round": "Round"},
        )
        fig_agree.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_agree, use_container_width=True)

        fig_correct = px.line(
            round_agg, x="round", y="correct_rate", color="label",
            markers=True, title="Any Agent Has Correct Answer by Round",
            labels={"correct_rate": "Has Correct Rate", "round": "Round"},
        )
        fig_correct.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_correct, use_container_width=True)

        fig_ent = px.line(
            round_agg, x="round", y="mean_entropy", color="label",
            markers=True, title="Mean Answer Entropy by Round",
            labels={"mean_entropy": "Entropy (bits)", "round": "Round"},
        )
        st.plotly_chart(fig_ent, use_container_width=True)

    # Belief transitions
    st.subheader("Belief Transitions")

    trans_data = []
    for rd in filtered_runs:
        for rec in rd.records:
            answers = rec.get("agent_round_parsed_answers") or []
            gt = rec.get("answer")
            n_agents = len(answers)
            for a in range(n_agents):
                seq = answers[a]
                for r in range(1, len(seq)):
                    prev = seq[r - 1]
                    cur = seq[r]
                    prev_correct = str(prev) == str(gt) if prev is not None and gt is not None else False
                    cur_correct = str(cur) == str(gt) if cur is not None and gt is not None else False
                    changed = prev != cur
                    prev_state = "none" if prev is None else ("correct" if prev_correct else "wrong")
                    cur_state = "none" if cur is None else ("correct" if cur_correct else "wrong")
                    trans_data.append({
                        "dataset": rd.meta.dataset,
                        "model": rd.meta.model_tag or "unknown",
                        "transition": f"{prev_state} -> {cur_state}",
                        "changed": int(changed),
                        "from_round": r,
                        "to_round": r + 1,
                    })

    if trans_data:
        tdf = pd.DataFrame(trans_data)

        # Transition counts
        trans_counts = tdf.groupby(["dataset", "model", "transition"]).size().reset_index(name="count")
        fig_trans = px.bar(
            trans_counts, x="transition", y="count",
            color="dataset", barmode="group",
            facet_col="model",
            title="Answer State Transitions",
            labels={"count": "Count", "transition": "Transition"},
        )
        fig_trans.update_xaxes(tickangle=45)
        st.plotly_chart(fig_trans, use_container_width=True)

        # Change rate by round
        change_by_round = tdf.groupby(["dataset", "model", "from_round"]).agg(
            change_rate=("changed", "mean"),
            n=("changed", "count"),
        ).reset_index()
        change_by_round["label"] = change_by_round["dataset"] + " / " + change_by_round["model"]
        fig_change = px.line(
            change_by_round, x="from_round", y="change_rate", color="label",
            markers=True, title="Answer Change Rate by Round",
            labels={"change_rate": "Change Rate", "from_round": "From Round"},
        )
        fig_change.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_change, use_container_width=True)

    # Judge vs Majority analysis
    st.subheader("Judge vs Majority Breakdown")
    jm_data = []
    for rd in filtered_runs:
        for rec in rd.records:
            jc = bool(rec.get("final_judge_correct"))
            mc = bool(rec.get("final_majority_correct"))
            jm_data.append({
                "dataset": rd.meta.dataset,
                "model": rd.meta.model_tag or "unknown",
                "judge_correct": jc,
                "majority_correct": mc,
                "both_correct": jc and mc,
                "judge_only": jc and not mc,
                "majority_only": not jc and mc,
                "both_wrong": not jc and not mc,
            })

    if jm_data:
        jmdf = pd.DataFrame(jm_data)
        jm_summary = jmdf.groupby(["dataset", "model"]).agg(
            total=("judge_correct", "count"),
            both_correct=("both_correct", "sum"),
            judge_only=("judge_only", "sum"),
            majority_only=("majority_only", "sum"),
            both_wrong=("both_wrong", "sum"),
        ).reset_index()

        st.dataframe(jm_summary, use_container_width=True)

        # Stacked bar
        jm_melted = jm_summary.melt(
            id_vars=["dataset", "model", "total"],
            value_vars=["both_correct", "judge_only", "majority_only", "both_wrong"],
            var_name="category",
            value_name="count",
        )
        jm_melted["label"] = jm_melted["dataset"] + " / " + jm_melted["model"]
        fig_jm = px.bar(
            jm_melted, x="label", y="count", color="category",
            title="Judge vs Majority Outcome Distribution",
            color_discrete_map={
                "both_correct": "#51cf66",
                "judge_only": "#339af0",
                "majority_only": "#fcc419",
                "both_wrong": "#ff6b6b",
            },
        )
        st.plotly_chart(fig_jm, use_container_width=True)

    # Lost correct analysis
    st.subheader("Lost Correct Analysis")
    st.markdown("Questions where the correct answer appeared in some round but was lost by the final round.")
    lost_data = []
    for rd in filtered_runs:
        for rec in rd.records:
            answers = rec.get("agent_round_parsed_answers") or []
            gt = rec.get("answer")
            n_agents = len(answers)
            n_rounds = len(answers[0]) if answers else 0
            ever_correct = False
            final_has_correct = False
            for a in range(n_agents):
                for r in range(n_rounds):
                    if r < len(answers[a]):
                        ans = answers[a][r]
                        if ans is not None and str(ans) == str(gt):
                            ever_correct = True
                            if r == n_rounds - 1:
                                final_has_correct = True
            lost_data.append({
                "dataset": rd.meta.dataset,
                "model": rd.meta.model_tag or "unknown",
                "ever_correct": ever_correct,
                "final_has_correct": final_has_correct,
                "lost": ever_correct and not final_has_correct,
            })

    if lost_data:
        ldf = pd.DataFrame(lost_data)
        lost_summary = ldf.groupby(["dataset", "model"]).agg(
            total=("lost", "count"),
            ever_correct=("ever_correct", "sum"),
            final_correct=("final_has_correct", "sum"),
            lost=("lost", "sum"),
        ).reset_index()
        lost_summary["ever_correct_pct"] = (lost_summary["ever_correct"] / lost_summary["total"]).apply(fmt_pct)
        lost_summary["lost_pct"] = (lost_summary["lost"] / lost_summary["total"]).apply(fmt_pct)
        st.dataframe(lost_summary, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model Statistics
# ---------------------------------------------------------------------------


def page_model_statistics(runs: list[RunData], summary_df: pd.DataFrame):
    st.header("Model Statistics")

    if not runs:
        st.warning("No runs loaded.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        datasets = sorted(summary_df["dataset"].unique())
        sel_ds = st.multiselect("Datasets", datasets, default=datasets, key="ms_ds")
    with col2:
        models = sorted(summary_df["model"].unique())
        sel_models = st.multiselect("Models", models, default=models, key="ms_models")
    with col3:
        modes = sorted(summary_df["mode"].unique())
        sel_modes = st.multiselect("Modes", modes, default=modes, key="ms_modes")

    # Subset size filter
    subset_sizes = sorted(summary_df["subset_n"].dropna().unique())
    if subset_sizes:
        sel_subset = st.multiselect(
            "Subset sizes (n questions)",
            [int(s) for s in subset_sizes],
            default=[int(s) for s in subset_sizes],
            key="ms_subset",
        )
    else:
        sel_subset = []

    filtered = summary_df[
        (summary_df["dataset"].isin(sel_ds))
        & (summary_df["model"].isin(sel_models))
        & (summary_df["mode"].isin(sel_modes))
    ]
    if sel_subset:
        filtered = filtered[filtered["subset_n"].isin(sel_subset)]

    if filtered.empty:
        st.info("No runs match the filters.")
        return

    # Aggregated statistics
    st.subheader("Aggregated Statistics")

    agg = filtered.groupby(["dataset", "model", "mode"]).agg(
        n_runs=("filename", "count"),
        total_q=("n_questions", "sum"),
        total_correct=("n_correct", "sum"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        min_accuracy=("accuracy", "min"),
        max_accuracy=("accuracy", "max"),
    ).reset_index()
    agg["pooled_accuracy"] = agg["total_correct"] / agg["total_q"]

    # Wilson CI
    ci_data = []
    for _, row in agg.iterrows():
        lo, hi = wilson_ci(int(row["total_correct"]), int(row["total_q"]))
        ci_data.append({"ci_low": lo, "ci_high": hi})
    ci_df = pd.DataFrame(ci_data)
    agg = pd.concat([agg.reset_index(drop=True), ci_df], axis=1)

    st.dataframe(
        agg.style.format({
            "mean_accuracy": "{:.1%}",
            "std_accuracy": "{:.3f}",
            "min_accuracy": "{:.1%}",
            "max_accuracy": "{:.1%}",
            "pooled_accuracy": "{:.1%}",
            "ci_low": "{:.1%}",
            "ci_high": "{:.1%}",
        }),
        use_container_width=True,
    )

    # Visualization
    st.subheader("Accuracy with Confidence Intervals")
    agg["label"] = agg["dataset"] + " / " + agg["mode"] + " / " + agg["model"]
    fig = go.Figure()
    for _, row in agg.iterrows():
        fig.add_trace(go.Bar(
            x=[row["label"]],
            y=[row["pooled_accuracy"]],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[row["ci_high"] - row["pooled_accuracy"]],
                arrayminus=[row["pooled_accuracy"] - row["ci_low"]],
            ),
            name=row["label"],
            text=f"{row['pooled_accuracy']:.1%}",
            textposition="outside",
        ))
    fig.update_layout(
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
        showlegend=False,
        title="Pooled Accuracy with 95% Wilson CI",
        xaxis_tickangle=45,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-question analysis
    st.subheader("Per-Question Analysis")

    # Build per-question data across runs
    q_data = defaultdict(lambda: defaultdict(list))
    filtered_runs = [rd for rd in runs
                     if rd.meta.dataset in sel_ds
                     and (rd.meta.model_tag or "unknown") in sel_models
                     and rd.meta.mode in sel_modes
                     and (not sel_subset or rd.meta.n in sel_subset)]

    for rd in filtered_runs:
        for rec in rd.records:
            key = (rd.meta.dataset, rec.get("orig_id"))
            method = f"{rd.meta.mode}/{rd.meta.model_tag or 'unknown'}"
            q_data[key][method].append(bool(rec.get("final_correct")))

    if q_data:
        q_rows = []
        for (ds, orig_id), methods in q_data.items():
            row = {"dataset": ds, "orig_id": orig_id}
            for method, results in methods.items():
                row[f"{method}_correct_rate"] = sum(results) / len(results) if results else 0
                row[f"{method}_n_runs"] = len(results)
            q_rows.append(row)

        q_df = pd.DataFrame(q_rows)
        if not q_df.empty:
            # Filter to show "hardest" questions
            correct_cols = [c for c in q_df.columns if c.endswith("_correct_rate")]
            if correct_cols:
                q_df["avg_correct_rate"] = q_df[correct_cols].mean(axis=1)
                q_df_sorted = q_df.sort_values("avg_correct_rate")

                n_show = st.slider("Number of questions to show", 5, min(100, len(q_df)), 20, key="ms_nshow")
                show_which = st.radio(
                    "Show", ["Hardest", "Easiest", "Most Variable"], horizontal=True, key="ms_show_which"
                )

                if show_which == "Hardest":
                    q_display = q_df_sorted.head(n_show)
                elif show_which == "Easiest":
                    q_display = q_df_sorted.tail(n_show)
                else:
                    if len(correct_cols) > 1:
                        q_df["variance"] = q_df[correct_cols].var(axis=1)
                        q_display = q_df.sort_values("variance", ascending=False).head(n_show)
                    else:
                        q_display = q_df_sorted.head(n_show)

                fmt_dict = {c: "{:.0%}" for c in q_display.columns if "correct_rate" in c}
                st.dataframe(q_display.style.format(fmt_dict), use_container_width=True)

    # Subset analysis
    st.subheader("Subset Size Impact")
    subset_analysis = filtered.groupby(["dataset", "model", "mode", "subset_n"]).agg(
        n_runs=("filename", "count"),
        mean_accuracy=("accuracy", "mean"),
    ).reset_index()

    if not subset_analysis.empty and subset_analysis["subset_n"].nunique() > 1:
        subset_analysis["label"] = (
            subset_analysis["dataset"] + " / "
            + subset_analysis["mode"] + " / "
            + subset_analysis["model"]
        )
        fig_sub = px.line(
            subset_analysis,
            x="subset_n",
            y="mean_accuracy",
            color="label",
            markers=True,
            title="Accuracy by Subset Size",
            labels={"mean_accuracy": "Mean Accuracy", "subset_n": "Subset Size (n)"},
        )
        fig_sub.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_sub, use_container_width=True)
    else:
        st.info("Only one subset size available; subset comparison not applicable.")


# ---------------------------------------------------------------------------
# Page: Per-Question Deep Dive
# ---------------------------------------------------------------------------


def page_question_deep_dive(runs: list[RunData], summary_df: pd.DataFrame):
    st.header("Per-Question Deep Dive")

    if not runs:
        st.warning("No runs loaded.")
        return

    # Let user pick a dataset and question
    col1, col2 = st.columns(2)
    with col1:
        datasets = sorted(set(rd.meta.dataset for rd in runs))
        sel_ds = st.selectbox("Dataset", datasets, key="qdd_ds")

    # Find all orig_ids for the dataset
    ds_runs = [rd for rd in runs if rd.meta.dataset == sel_ds]
    all_orig_ids = sorted(set(
        rec.get("orig_id") for rd in ds_runs for rec in rd.records
        if rec.get("orig_id") is not None
    ))

    with col2:
        sel_qid = st.selectbox("Question (orig_id)", all_orig_ids, key="qdd_qid")

    if sel_qid is None:
        return

    # Collect all records for this question across all runs
    q_records = []
    for rd in ds_runs:
        for rec in rd.records:
            if rec.get("orig_id") == sel_qid:
                q_records.append((rd, rec))

    if not q_records:
        st.info("No records found for this question.")
        return

    # Display question text from first record
    first_rec = q_records[0][1]
    st.subheader("Question")
    st.text_area("", first_rec.get("question", ""), height=150, disabled=True, key="qdd_qtext")
    st.markdown(f"**Ground Truth**: `{first_rec.get('answer', 'N/A')}`")

    # Summary across runs
    st.subheader(f"Results Across {len(q_records)} Runs")

    q_summary_rows = []
    for rd, rec in q_records:
        row = {
            "Run": rd.meta.filename,
            "Mode": rd.meta.mode,
            "Model": rd.meta.model_tag or "unknown",
            "Seed": rd.meta.seed,
            "Final Answer": rec.get("final_answer"),
            "Correct": bool(rec.get("final_correct")),
        }
        if rd.meta.mode == "debate":
            row["Judge Answer"] = rec.get("final_judge_answer")
            row["Judge Correct"] = bool(rec.get("final_judge_correct"))
            row["Majority Answer"] = rec.get("final_majority_answer")
            row["Majority Correct"] = bool(rec.get("final_majority_correct"))
        q_summary_rows.append(row)

    q_summary_df = pd.DataFrame(q_summary_rows)
    st.dataframe(q_summary_df, use_container_width=True)

    # If debate runs, show answer evolution comparison
    debate_q_records = [(rd, rec) for rd, rec in q_records if rd.meta.mode == "debate"]
    if debate_q_records:
        st.subheader("Debate Answer Evolution Comparison")

        for rd, rec in debate_q_records:
            answers = rec.get("agent_round_parsed_answers") or []
            gt = rec.get("answer")
            with st.expander(
                f"{rd.meta.filename} (seed={rd.meta.seed}, model={rd.meta.model_tag})",
                expanded=True,
            ):
                if answers:
                    n_agents = len(answers)
                    n_rounds = max(len(a) for a in answers) if answers else 0
                    evo_data = []
                    for a_idx in range(n_agents):
                        for r_idx in range(n_rounds):
                            ans = answers[a_idx][r_idx] if r_idx < len(answers[a_idx]) else None
                            evo_data.append({
                                "Agent": f"Agent {a_idx+1}",
                                "Round": r_idx + 1,
                                "Answer": str(ans) if ans is not None else "None",
                                "Correct": str(ans) == str(gt) if ans is not None and gt is not None else False,
                            })
                    evo_df = pd.DataFrame(evo_data)
                    pivot = evo_df.pivot(index="Agent", columns="Round", values="Answer")
                    correct_pivot = evo_df.pivot(index="Agent", columns="Round", values="Correct")

                    fig_heat = go.Figure(data=go.Heatmap(
                        z=correct_pivot.values.astype(int),
                        x=[f"R{c}" for c in correct_pivot.columns],
                        y=correct_pivot.index.tolist(),
                        colorscale=[[0, "#ff6b6b"], [1, "#51cf66"]],
                        showscale=False,
                        text=pivot.values,
                        texttemplate="%{text}",
                        textfont={"size": 14},
                    ))
                    fig_heat.update_layout(
                        height=max(180, 50 * n_agents + 80),
                        margin=dict(t=20, b=20),
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                col_j, col_m = st.columns(2)
                col_j.markdown(f"**Judge**: `{rec.get('final_judge_answer')}` ({'correct' if rec.get('final_judge_correct') else 'wrong'})")
                col_m.markdown(f"**Majority**: `{rec.get('final_majority_answer')}` ({'correct' if rec.get('final_majority_correct') else 'wrong'})")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def get_results_dir() -> str:
    """Get results directory from command-line args or default."""
    # Streamlit passes extra args after --
    if "--results-dir" in sys.argv:
        idx = sys.argv.index("--results-dir")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    if "--results_dir" in sys.argv:
        idx = sys.argv.index("--results_dir")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return ""


def main():
    st.set_page_config(
        page_title="Model Run Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Model Run Dashboard")

    # Sidebar: results directory
    with st.sidebar:
        st.header("Configuration")
        default_dir = get_results_dir()
        results_dir = st.text_input(
            "Results Directory",
            value=default_dir,
            help="Path to folder containing aime25/ and gpqa/ subfolders with .jsonl result files",
        )

        if not results_dir:
            st.info("Enter a results directory path to load data.")
            st.stop()

        results_path = Path(results_dir)
        if not results_path.exists():
            st.error(f"Directory does not exist: {results_dir}")
            st.stop()

        # Load data
        runs = load_all_runs(results_dir)

        if not runs:
            st.warning(f"No valid .jsonl result files found in {results_dir}")
            st.stop()

        summary_df = runs_to_summary_df(runs)
        st.success(f"Loaded {len(runs)} runs")

        # Navigation
        st.header("Navigation")
        page = st.radio(
            "Page",
            [
                "Overview",
                "Run Comparison",
                "Transcript Viewer",
                "Debate Analysis",
                "Model Statistics",
                "Per-Question Deep Dive",
            ],
            key="nav_page",
        )

        # Quick stats in sidebar
        st.header("Quick Stats")
        st.markdown(f"**Runs**: {len(runs)}")
        st.markdown(f"**Datasets**: {', '.join(sorted(summary_df['dataset'].unique()))}")
        st.markdown(f"**Models**: {', '.join(sorted(summary_df['model'].unique()))}")
        st.markdown(f"**Modes**: {', '.join(sorted(summary_df['mode'].unique()))}")

        # Refresh button
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # Render selected page
    if page == "Overview":
        page_overview(runs, summary_df)
    elif page == "Run Comparison":
        page_run_comparison(runs, summary_df)
    elif page == "Transcript Viewer":
        page_transcript_viewer(runs, summary_df)
    elif page == "Debate Analysis":
        page_debate_analysis(runs, summary_df)
    elif page == "Model Statistics":
        page_model_statistics(runs, summary_df)
    elif page == "Per-Question Deep Dive":
        page_question_deep_dive(runs, summary_df)


if __name__ == "__main__":
    main()
