#!/usr/bin/env python3
"""
Interactive Dashboard for Model Run Results.

A fully customizable Streamlit dashboard that loads JSONL result files from
aime25/ and gpqa/ subfolders and provides:
  - Overview statistics across all runs
  - Run comparison by model, seed, mode (single / majority / debate)
  - Transcript viewer with full conversation display
  - Debate analysis with round-by-round behaviour, conformity tracking,
    GPQA letter-transition analysis
  - Per-model statistics with subset filtering and confidence intervals
  - Per-question deep dive across multiple runs
  - Raw data viewer & CSV export

Usage:
    streamlit run dashboard.py -- --results-dir /path/to/results
    python launch_dashboard.py /path/to/results
"""
from __future__ import annotations

import io
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

    # Search in expected subfolders and root
    search_dirs = [root]
    for sub in ("aime25_quick", "gpqa_quick", "aime25", "gpqa"):
        d = root / sub
        if d.is_dir():
            search_dirs.append(d)

    seen: set[str] = set()
    jsonl_files: list[Path] = []
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
        row: dict[str, Any] = {
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
        row: dict[str, Any] = {
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


def plurality_vote(answers: list[Any]) -> Any:
    """Return the most common answer (ignoring None). None if tied."""
    filtered = [a for a in answers if a is not None]
    if not filtered:
        return None
    counts = Counter(filtered)
    top_count = max(counts.values())
    top = [a for a, c in counts.items() if c == top_count]
    return top[0] if len(top) == 1 else None


# ---------------------------------------------------------------------------
# Global sidebar filter state (shared across pages)
# ---------------------------------------------------------------------------


def apply_global_filters(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar global filters to summary_df."""
    if summary_df.empty:
        return summary_df
    with st.sidebar:
        st.header("Global Filters")
        sel_datasets = st.multiselect(
            "Datasets",
            sorted(summary_df["dataset"].unique()),
            default=sorted(summary_df["dataset"].unique()),
            key="gf_ds",
        )
        sel_models = st.multiselect(
            "Models",
            sorted(summary_df["model"].unique()),
            default=sorted(summary_df["model"].unique()),
            key="gf_models",
        )
        sel_modes = st.multiselect(
            "Modes",
            sorted(summary_df["mode"].unique()),
            default=sorted(summary_df["mode"].unique()),
            key="gf_modes",
        )
        # Subset sizes
        subset_sizes = sorted(summary_df["subset_n"].dropna().unique())
        if subset_sizes:
            sel_subsets = st.multiselect(
                "Subset sizes (n)",
                [int(s) for s in subset_sizes],
                default=[int(s) for s in subset_sizes],
                key="gf_subset",
            )
        else:
            sel_subsets = []

    mask = (
        summary_df["dataset"].isin(sel_datasets)
        & summary_df["model"].isin(sel_models)
        & summary_df["mode"].isin(sel_modes)
    )
    if sel_subsets:
        mask = mask & summary_df["subset_n"].isin(sel_subsets)
    return summary_df[mask]


def filter_runs(runs: list[RunData], filtered_df: pd.DataFrame) -> list[RunData]:
    """Return runs matching the filtered summary DataFrame."""
    filenames = set(filtered_df["filename"])
    return [rd for rd in runs if rd.meta.filename in filenames]


# ---------------------------------------------------------------------------
# CSV export helper
# ---------------------------------------------------------------------------


def download_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    """Offer a CSV download button for a DataFrame."""
    csv = df.to_csv(index=False)
    st.download_button(label, csv, file_name=filename, mime="text/csv")


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------


def page_overview(runs: list[RunData], filtered_df: pd.DataFrame):
    st.header("Overview")

    if filtered_df.empty:
        st.warning("No runs match the current filters.")
        return

    filt_runs = filter_runs(runs, filtered_df)

    # Top-level metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Runs", len(filt_runs))
    c2.metric("Datasets", filtered_df["dataset"].nunique())
    c3.metric("Models", filtered_df["model"].nunique())
    c4.metric("Questions", int(filtered_df["n_questions"].sum()))
    overall_acc = (
        filtered_df["n_correct"].sum() / filtered_df["n_questions"].sum()
        if filtered_df["n_questions"].sum() > 0
        else 0
    )
    c5.metric("Overall Accuracy", fmt_pct(overall_acc))

    # --- Accuracy by dataset & mode ---
    st.subheader("Accuracy by Dataset & Mode")
    grouped = (
        filtered_df.groupby(["dataset", "mode"])
        .agg(total_q=("n_questions", "sum"), total_correct=("n_correct", "sum"), n_runs=("filename", "count"))
        .reset_index()
    )
    grouped["accuracy"] = grouped["total_correct"] / grouped["total_q"]
    grouped["ci_lo"] = grouped.apply(lambda r: wilson_ci(int(r["total_correct"]), int(r["total_q"]))[0], axis=1)
    grouped["ci_hi"] = grouped.apply(lambda r: wilson_ci(int(r["total_correct"]), int(r["total_q"]))[1], axis=1)
    grouped["label"] = grouped["accuracy"].apply(lambda x: f"{x*100:.1f}%")

    fig = px.bar(
        grouped, x="dataset", y="accuracy", color="mode", barmode="group",
        text="label", title="Pooled Accuracy by Dataset & Mode",
        error_y=grouped["ci_hi"] - grouped["accuracy"],
        labels={"accuracy": "Accuracy"}, color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # --- Accuracy by model ---
    st.subheader("Accuracy by Model")
    mg = (
        filtered_df.groupby(["model", "mode"])
        .agg(total_q=("n_questions", "sum"), total_correct=("n_correct", "sum"))
        .reset_index()
    )
    mg["accuracy"] = mg["total_correct"] / mg["total_q"]
    mg["label"] = mg["accuracy"].apply(lambda x: f"{x*100:.1f}%")
    fig2 = px.bar(
        mg, x="model", y="accuracy", color="mode", barmode="group",
        text="label", title="Pooled Accuracy by Model & Mode",
        labels={"accuracy": "Accuracy"}, color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig2.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

    # --- Per-run table ---
    st.subheader("All Runs")
    display_cols = [
        "filename", "dataset", "mode", "model", "seed", "n_questions",
        "n_correct", "accuracy", "n_agents", "n_rounds", "n_samples", "subset_n", "timestamp",
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(
        filtered_df[display_cols].style.format({"accuracy": "{:.1%}"}),
        use_container_width=True, height=400,
    )
    download_csv(filtered_df[display_cols], "overview_runs.csv")


# ---------------------------------------------------------------------------
# Page: Run Comparison
# ---------------------------------------------------------------------------


def page_run_comparison(runs: list[RunData], filtered_df: pd.DataFrame):
    st.header("Run Comparison")

    if filtered_df.empty:
        st.warning("No runs match the current filters.")
        return

    # Comparison axis
    compare_by = st.selectbox("Compare by", ["seed", "model", "mode", "subset_n", "timestamp"], key="rc_compare")

    st.subheader("Accuracy Comparison")
    if compare_by in filtered_df.columns:
        plot_df = filtered_df.copy()
        plot_df[compare_by] = plot_df[compare_by].astype(str)
        plot_df["label"] = (
            plot_df["dataset"] + " / " + plot_df["mode"]
            + ((" / " + plot_df["model"]) if compare_by != "model" else "")
        )
        fig = px.bar(
            plot_df, x=compare_by, y="accuracy", color="label", barmode="group",
            title=f"Accuracy by {compare_by}",
            labels={"accuracy": "Accuracy"}, color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # --- Seed stability ---
    st.subheader("Seed Stability")
    st.markdown("How accuracy varies across random seeds for the same model/mode/dataset.")
    seed_data = filtered_df[filtered_df["seed"].notna()].copy()
    if not seed_data.empty:
        seed_data["config"] = seed_data["dataset"] + " / " + seed_data["mode"] + " / " + seed_data["model"]
        seed_data["seed_str"] = seed_data["seed"].astype(str)
        fig_s = px.strip(
            seed_data, x="config", y="accuracy", color="seed_str",
            title="Accuracy Distribution by Seed",
            labels={"accuracy": "Accuracy", "config": "", "seed_str": "Seed"},
        )
        fig_s.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_s, use_container_width=True)

        # Box plot of accuracy per config
        fig_box = px.box(
            seed_data, x="config", y="accuracy",
            title="Accuracy Spread (Box Plot)",
            labels={"accuracy": "Accuracy", "config": ""},
        )
        fig_box.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_box, use_container_width=True)

    # --- Debate: Judge vs Majority ---
    debate_df = filtered_df[filtered_df["mode"] == "debate"]
    if not debate_df.empty and "judge_accuracy" in debate_df.columns:
        st.subheader("Debate: Judge vs Majority Accuracy")
        melted = debate_df.melt(
            id_vars=["filename", "dataset", "model", "seed"],
            value_vars=["judge_accuracy", "majority_accuracy"],
            var_name="method", value_name="acc",
        )
        melted["method"] = melted["method"].str.replace("_accuracy", "")
        melted["label"] = melted["dataset"] + " / " + melted["model"]
        fig_jm = px.bar(
            melted, x="label", y="acc", color="method", barmode="group",
            title="Judge vs Majority Accuracy",
            color_discrete_map={"judge": "#636EFA", "majority": "#EF553B"},
        )
        fig_jm.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_jm, use_container_width=True)

    # --- Side-by-side mode comparison for same model ---
    st.subheader("Mode Comparison (Same Model & Dataset)")
    st.markdown("For each model+dataset, compare single/majority/debate accuracy side-by-side.")
    mode_cmp = (
        filtered_df.groupby(["dataset", "model", "mode"])
        .agg(mean_acc=("accuracy", "mean"), n_runs=("filename", "count"))
        .reset_index()
    )
    if len(mode_cmp) > 1:
        mode_cmp["label"] = mode_cmp["dataset"] + " / " + mode_cmp["model"]
        fig_mc = px.bar(
            mode_cmp, x="label", y="mean_acc", color="mode", barmode="group",
            text=mode_cmp["mean_acc"].apply(lambda x: f"{x*100:.1f}%"),
            title="Mean Accuracy by Mode (grouped by model+dataset)",
        )
        fig_mc.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_mc, use_container_width=True)

    # Table
    st.subheader("Filtered Runs")
    cols = ["filename", "dataset", "mode", "model", "seed", "n_questions", "accuracy"]
    if "judge_accuracy" in filtered_df.columns:
        cols += ["judge_accuracy", "majority_accuracy"]
    cols = [c for c in cols if c in filtered_df.columns]
    fmt = {c: "{:.1%}" for c in cols if "accuracy" in c}
    st.dataframe(filtered_df[cols].style.format(fmt), use_container_width=True)
    download_csv(filtered_df[cols], "run_comparison.csv")


# ---------------------------------------------------------------------------
# Page: Transcript Viewer
# ---------------------------------------------------------------------------


def page_transcript_viewer(runs: list[RunData], filtered_df: pd.DataFrame):
    st.header("Transcript Viewer")

    filt_runs = filter_runs(runs, filtered_df)
    if not filt_runs:
        st.warning("No runs match the current filters.")
        return

    # Run selection
    run_map = {
        f"{rd.meta.filename} ({rd.meta.mode} | {rd.meta.dataset} | {rd.meta.model_tag or 'unknown'})": i
        for i, rd in enumerate(filt_runs)
    }
    sel = st.selectbox("Select Run", list(run_map.keys()), key="tv_run")
    if sel is None:
        return
    rd = filt_runs[run_map[sel]]
    meta = rd.meta

    # Question filtering
    filter_mode = st.radio(
        "Show questions", ["All", "Correct only", "Incorrect only"], horizontal=True, key="tv_filter"
    )
    records = rd.records
    if filter_mode == "Correct only":
        records = [r for r in records if r.get("final_correct")]
    elif filter_mode == "Incorrect only":
        records = [r for r in records if not r.get("final_correct")]

    if not records:
        st.info("No questions match this filter.")
        return

    q_map = {
        f"Q{rec.get('subset_id', i)} (orig={rec.get('orig_id','?')}) | GT={rec.get('answer','?')} | {'CORRECT' if rec.get('final_correct') else 'WRONG'}": i
        for i, rec in enumerate(records)
    }
    sel_q = st.selectbox("Select Question", list(q_map.keys()), key="tv_q")
    if sel_q is None:
        return
    rec = records[q_map[sel_q]]

    # Question display
    st.subheader("Question")
    st.text_area("", rec.get("question", ""), height=180, disabled=True, key="tv_qtext")

    c1, c2, c3 = st.columns(3)
    c1.metric("Ground Truth", str(rec.get("answer", "N/A")))
    c2.metric("Final Answer", str(rec.get("final_answer", "N/A")))
    correct = rec.get("final_correct", False)
    c3.metric("Correct", "Yes" if correct else "No")

    if meta.mode == "debate":
        _render_debate_transcript(rec, meta)
    elif meta.mode in ("single", "majority"):
        _render_sampled_transcript(rec, meta)

    # Raw JSON viewer
    with st.expander("Raw JSON Record", expanded=False):
        # Make a copy without huge fields for readability
        raw_display = {k: v for k, v in rec.items() if k not in ("raw_task",)}
        st.json(raw_display)


def _render_sampled_transcript(rec: dict[str, Any], meta: RunMeta):
    completions = rec.get("sample_completions") or []
    parsed = rec.get("sample_parsed_answers") or []

    st.subheader(f"Completions ({len(completions)} samples)")

    # Summary of parsed answers
    if parsed:
        counts = Counter(parsed)
        st.markdown("**Parsed answer distribution**: " + ", ".join(f"`{a}`: {c}" for a, c in counts.most_common()))

    for i, comp in enumerate(completions):
        p = parsed[i] if i < len(parsed) else None
        with st.expander(f"Sample {i+1} | Answer: `{p}`", expanded=(i == 0)):
            st.code(comp, language=None)


def _render_debate_transcript(rec: dict[str, Any], meta: RunMeta):
    agent_responses = rec.get("agent_responses") or []
    n_agents = rec.get("n_agents", len(agent_responses))
    n_rounds = rec.get("n_rounds", 0)
    agent_round_answers = rec.get("agent_round_parsed_answers") or []
    gt = rec.get("answer")

    # Metrics
    col_j, col_m = st.columns(2)
    with col_j:
        ja = rec.get("final_judge_answer", "N/A")
        jc = rec.get("final_judge_correct", False)
        st.metric("Judge Answer", str(ja))
        st.metric("Judge Correct", "Yes" if jc else "No")
    with col_m:
        ma = rec.get("final_majority_answer", "N/A")
        mc = rec.get("final_majority_correct", False)
        st.metric("Majority Answer", str(ma))
        st.metric("Majority Correct", "Yes" if mc else "No")

    # Answer evolution table + heatmap
    if agent_round_answers:
        st.subheader("Answer Evolution by Round")
        evo_data = []
        for a_idx, answers in enumerate(agent_round_answers):
            for r_idx, ans in enumerate(answers):
                is_correct = (str(ans) == str(gt)) if ans is not None and gt is not None else False
                evo_data.append({
                    "Agent": f"Agent {a_idx+1}",
                    "Round": r_idx + 1,
                    "Answer": str(ans) if ans is not None else "None",
                    "Correct": is_correct,
                })
        evo_df = pd.DataFrame(evo_data)
        pivot = evo_df.pivot(index="Agent", columns="Round", values="Answer")
        correct_pivot = evo_df.pivot(index="Agent", columns="Round", values="Correct")

        # Heatmap
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
            height=max(200, 60 * n_agents + 100),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Plain table too
        st.dataframe(pivot, use_container_width=True)

    # Transcript display
    st.subheader("Agent Transcripts")
    view_mode = st.radio("View", ["By Round", "By Agent"], horizontal=True, key="dt_view")

    # Round range selector
    if n_rounds > 1:
        round_range = st.slider(
            "Round range to display", 1, n_rounds, (1, n_rounds), key="dt_rr"
        )
    else:
        round_range = (1, n_rounds)

    if view_mode == "By Round":
        for r in range(round_range[0] - 1, round_range[1]):
            with st.expander(f"Round {r+1}", expanded=(r == n_rounds - 1)):
                for a_idx, ctx in enumerate(agent_responses):
                    idx = 2 * r + 1
                    if idx < len(ctx):
                        content = ctx[idx].get("content", "")
                        p = None
                        if a_idx < len(agent_round_answers) and r < len(agent_round_answers[a_idx]):
                            p = agent_round_answers[a_idx][r]
                        st.markdown(f"**Agent {a_idx+1}** | Answer: `{p}`")
                        st.code(content, language=None)
    else:
        for a_idx, ctx in enumerate(agent_responses):
            with st.expander(f"Agent {a_idx+1}", expanded=(a_idx == 0)):
                for r in range(round_range[0] - 1, round_range[1]):
                    idx = 2 * r + 1
                    if idx < len(ctx):
                        content = ctx[idx].get("content", "")
                        p = None
                        if a_idx < len(agent_round_answers) and r < len(agent_round_answers[a_idx]):
                            p = agent_round_answers[a_idx][r]
                        st.markdown(f"**Round {r+1}** | Answer: `{p}`")
                        st.code(content, language=None)

    # Debate messages (user prompts showing other agents' answers)
    if n_rounds > 1:
        with st.expander("Debate Prompts (what each agent sees)", expanded=False):
            for a_idx, ctx in enumerate(agent_responses):
                st.markdown(f"#### Agent {a_idx+1}")
                for r in range(1, n_rounds):
                    user_idx = 2 * r
                    if user_idx < len(ctx):
                        st.markdown(f"**Round {r+1} prompt:**")
                        st.code(ctx[user_idx].get("content", "")[:2000], language=None)

    # Judge trace
    jt = rec.get("judge_trace") or {}
    if jt:
        with st.expander("Judge Trace", expanded=False):
            st.markdown(f"**Model**: {jt.get('judge_model', 'N/A')}")
            st.markdown(f"**Parsed**: `{jt.get('judge_parsed_answer', 'N/A')}`")
            st.markdown(f"**Parse Failed**: {jt.get('judge_parse_failed', False)}")
            st.markdown(f"**Used Fallback**: {jt.get('judge_used_fallback', False)}")
            raw_resp = jt.get("judge_raw_response", "")
            if raw_resp:
                st.code(str(raw_resp), language=None)
            retry = jt.get("judge_retry_raw_response")
            if retry:
                st.markdown("**Retry Response:**")
                st.code(str(retry), language=None)


# ---------------------------------------------------------------------------
# Page: Debate Analysis
# ---------------------------------------------------------------------------


def page_debate_analysis(runs: list[RunData], filtered_df: pd.DataFrame):
    st.header("Debate Analysis")

    debate_df = filtered_df[filtered_df["mode"] == "debate"]
    if debate_df.empty:
        st.warning("No debate runs match the current filters.")
        return

    debate_runs = filter_runs(runs, debate_df)

    # --- Round-by-round agreement ---
    st.subheader("Round-by-Round Agreement & Correctness")

    round_data = []
    for rd in debate_runs:
        for rec in rd.records:
            answers = rec.get("agent_round_parsed_answers") or []
            if not answers:
                continue
            gt = rec.get("answer")
            n_a = len(answers)
            n_r = len(answers[0]) if answers else 0
            for r in range(n_r):
                ra = [answers[a][r] for a in range(n_a) if r < len(answers[a])]
                counts = Counter(ra)
                unanimous = len(set(ra)) == 1 and len(ra) > 0
                has_correct = any(str(a) == str(gt) for a in ra if a is not None)
                all_correct = all(str(a) == str(gt) for a in ra if a is not None) and any(a is not None for a in ra)
                round_data.append({
                    "dataset": rd.meta.dataset,
                    "model": rd.meta.model_tag or "unknown",
                    "seed": rd.meta.seed,
                    "round": r + 1,
                    "unanimous": int(unanimous),
                    "has_correct": int(has_correct),
                    "all_correct": int(all_correct),
                    "entropy": entropy_from_counts(counts),
                    "n_unique": len(set(ra)),
                    "n_none": sum(1 for a in ra if a is None),
                })

    if round_data:
        rdf = pd.DataFrame(round_data)
        ragg = rdf.groupby(["dataset", "model", "round"]).agg(
            unanimous_rate=("unanimous", "mean"),
            any_correct_rate=("has_correct", "mean"),
            all_correct_rate=("all_correct", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_unique=("n_unique", "mean"),
            mean_none=("n_none", "mean"),
        ).reset_index()
        ragg["label"] = ragg["dataset"] + " / " + ragg["model"]

        tab1, tab2, tab3, tab4 = st.tabs(["Agreement", "Correctness", "Entropy", "Diversity"])

        with tab1:
            fig = px.line(
                ragg, x="round", y="unanimous_rate", color="label",
                markers=True, title="Unanimous Agreement Rate by Round",
            )
            fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig_c = go.Figure()
            for lbl in ragg["label"].unique():
                sub = ragg[ragg["label"] == lbl]
                fig_c.add_trace(go.Scatter(
                    x=sub["round"], y=sub["any_correct_rate"],
                    mode="lines+markers", name=f"{lbl} (any correct)",
                    line=dict(dash="solid"),
                ))
                fig_c.add_trace(go.Scatter(
                    x=sub["round"], y=sub["all_correct_rate"],
                    mode="lines+markers", name=f"{lbl} (all correct)",
                    line=dict(dash="dash"),
                ))
            fig_c.update_layout(
                title="Correct Agent Presence by Round",
                yaxis_tickformat=".0%", yaxis_range=[0, 1],
                xaxis_title="Round", yaxis_title="Rate",
            )
            st.plotly_chart(fig_c, use_container_width=True)

        with tab3:
            fig_e = px.line(
                ragg, x="round", y="mean_entropy", color="label",
                markers=True, title="Mean Answer Entropy (bits) by Round",
            )
            st.plotly_chart(fig_e, use_container_width=True)

        with tab4:
            fig_d = px.line(
                ragg, x="round", y="mean_unique", color="label",
                markers=True, title="Mean Unique Answers per Question by Round",
            )
            st.plotly_chart(fig_d, use_container_width=True)

    # --- Belief transitions ---
    st.subheader("Belief Transitions")

    trans_data = []
    for rd in debate_runs:
        for rec in rd.records:
            answers = rec.get("agent_round_parsed_answers") or []
            gt = rec.get("answer")
            for a in range(len(answers)):
                seq = answers[a]
                for r in range(1, len(seq)):
                    prev, cur = seq[r - 1], seq[r]
                    pc = str(prev) == str(gt) if prev is not None and gt is not None else False
                    cc = str(cur) == str(gt) if cur is not None and gt is not None else False
                    ps = "none" if prev is None else ("correct" if pc else "wrong")
                    cs = "none" if cur is None else ("correct" if cc else "wrong")
                    changed = prev != cur
                    trans_data.append({
                        "dataset": rd.meta.dataset,
                        "model": rd.meta.model_tag or "unknown",
                        "transition": f"{ps} -> {cs}",
                        "changed": int(changed),
                        "from_round": r,
                        "to_round": r + 1,
                        "prev_answer": str(prev),
                        "cur_answer": str(cur),
                    })

    if trans_data:
        tdf = pd.DataFrame(trans_data)

        col_a, col_b = st.columns(2)
        with col_a:
            tc = tdf.groupby(["dataset", "model", "transition"]).size().reset_index(name="count")
            fig_t = px.bar(
                tc, x="transition", y="count", color="dataset", barmode="group",
                title="Correctness State Transitions",
            )
            fig_t.update_xaxes(tickangle=45)
            st.plotly_chart(fig_t, use_container_width=True)

        with col_b:
            cbr = tdf.groupby(["dataset", "model", "from_round"]).agg(
                change_rate=("changed", "mean"),
            ).reset_index()
            cbr["label"] = cbr["dataset"] + " / " + cbr["model"]
            fig_cr = px.line(
                cbr, x="from_round", y="change_rate", color="label",
                markers=True, title="Answer Change Rate by Round",
            )
            fig_cr.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
            st.plotly_chart(fig_cr, use_container_width=True)

    # --- Conformity proxy ---
    st.subheader("Conformity Proxy")
    st.markdown(
        "When an agent changes its answer, how often does it move **toward** "
        "the plurality of the other agents from the previous round?"
    )

    conf_data = []
    for rd in debate_runs:
        for rec in rd.records:
            answers = rec.get("agent_round_parsed_answers") or []
            n_a = len(answers)
            for a in range(n_a):
                seq = answers[a]
                for r in range(1, len(seq)):
                    prev, cur = seq[r - 1], seq[r]
                    if prev == cur:
                        continue  # no change
                    others_prev = [answers[aa][r - 1] for aa in range(n_a) if aa != a and r - 1 < len(answers[aa])]
                    other_plur = plurality_vote(others_prev)
                    conf_data.append({
                        "dataset": rd.meta.dataset,
                        "model": rd.meta.model_tag or "unknown",
                        "toward_plurality": int(cur == other_plur) if other_plur is not None else 0,
                        "away_from_plurality": int(prev == other_plur) if other_plur is not None else 0,
                        "round": r + 1,
                    })

    if conf_data:
        cdf = pd.DataFrame(conf_data)
        cagg = cdf.groupby(["dataset", "model"]).agg(
            total_changes=("toward_plurality", "count"),
            toward=("toward_plurality", "sum"),
            away=("away_from_plurality", "sum"),
        ).reset_index()
        cagg["toward_pct"] = (cagg["toward"] / cagg["total_changes"]).apply(fmt_pct)
        cagg["away_pct"] = (cagg["away"] / cagg["total_changes"]).apply(fmt_pct)
        st.dataframe(cagg, use_container_width=True)

        # By round
        cagg_r = cdf.groupby(["dataset", "model", "round"]).agg(
            toward_rate=("toward_plurality", "mean"),
        ).reset_index()
        cagg_r["label"] = cagg_r["dataset"] + " / " + cagg_r["model"]
        fig_conf = px.line(
            cagg_r, x="round", y="toward_rate", color="label",
            markers=True, title="Conformity Rate by Round (fraction of changes toward other-agent plurality)",
        )
        fig_conf.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("No answer changes detected (agents never changed their answers).")

    # --- GPQA letter transitions ---
    gpqa_debate_runs = [rd for rd in debate_runs if rd.meta.dataset == "gpqa"]
    if gpqa_debate_runs:
        st.subheader("GPQA: Letter Choice Transitions")
        letter_trans: list[dict[str, Any]] = []
        for rd in gpqa_debate_runs:
            for rec in rd.records:
                answers = rec.get("agent_round_parsed_answers") or []
                for a in range(len(answers)):
                    seq = answers[a]
                    for r in range(1, len(seq)):
                        prev, cur = seq[r - 1], seq[r]
                        if prev is not None and cur is not None and prev in "ABCD" and cur in "ABCD":
                            letter_trans.append({
                                "model": rd.meta.model_tag or "unknown",
                                "from": prev,
                                "to": cur,
                            })
        if letter_trans:
            ltdf = pd.DataFrame(letter_trans)
            # Transition matrix
            for model in ltdf["model"].unique():
                sub = ltdf[ltdf["model"] == model]
                matrix = pd.crosstab(sub["from"], sub["to"], margins=True)
                st.markdown(f"**{model}** — Letter transition matrix (from rows, to columns):")
                st.dataframe(matrix, use_container_width=True)

                # Sankey-style visualization
                from_labels = [f"From {l}" for l in "ABCD"]
                to_labels = [f"To {l}" for l in "ABCD"]
                all_labels = from_labels + to_labels
                source, target, value = [], [], []
                for fi, fl in enumerate("ABCD"):
                    for ti, tl in enumerate("ABCD"):
                        ct = len(sub[(sub["from"] == fl) & (sub["to"] == tl)])
                        if ct > 0:
                            source.append(fi)
                            target.append(4 + ti)
                            value.append(ct)
                if value:
                    fig_sk = go.Figure(data=[go.Sankey(
                        node=dict(label=all_labels, pad=15, thickness=20),
                        link=dict(source=source, target=target, value=value),
                    )])
                    fig_sk.update_layout(title=f"{model}: Letter Choice Flow", height=400)
                    st.plotly_chart(fig_sk, use_container_width=True)

    # --- Judge vs Majority detailed ---
    st.subheader("Judge vs Majority Outcome Matrix")
    jm_data = []
    for rd in debate_runs:
        for rec in rd.records:
            jc = bool(rec.get("final_judge_correct"))
            mc = bool(rec.get("final_majority_correct"))
            jm_data.append({
                "dataset": rd.meta.dataset,
                "model": rd.meta.model_tag or "unknown",
                "both_correct": jc and mc,
                "judge_only": jc and not mc,
                "majority_only": not jc and mc,
                "both_wrong": not jc and not mc,
            })
    if jm_data:
        jmdf = pd.DataFrame(jm_data)
        jm_agg = jmdf.groupby(["dataset", "model"]).agg(
            total=("both_correct", "count"),
            both_correct=("both_correct", "sum"),
            judge_rescues=("judge_only", "sum"),
            judge_harms=("majority_only", "sum"),
            both_wrong=("both_wrong", "sum"),
        ).reset_index()
        jm_agg["judge_rescue_pct"] = (jm_agg["judge_rescues"] / jm_agg["total"]).apply(fmt_pct)
        jm_agg["judge_harm_pct"] = (jm_agg["judge_harms"] / jm_agg["total"]).apply(fmt_pct)
        st.dataframe(jm_agg, use_container_width=True)

        jm_melted = jm_agg.melt(
            id_vars=["dataset", "model", "total"],
            value_vars=["both_correct", "judge_rescues", "judge_harms", "both_wrong"],
            var_name="category", value_name="count",
        )
        jm_melted["label"] = jm_melted["dataset"] + " / " + jm_melted["model"]
        fig_jm = px.bar(
            jm_melted, x="label", y="count", color="category",
            title="Judge vs Majority Outcome Distribution",
            color_discrete_map={
                "both_correct": "#51cf66", "judge_rescues": "#339af0",
                "judge_harms": "#fcc419", "both_wrong": "#ff6b6b",
            },
        )
        st.plotly_chart(fig_jm, use_container_width=True)

    # --- Lost correct ---
    st.subheader("Lost Correct Analysis")
    st.markdown("Questions where a correct answer appeared in some agent/round but was lost by the final round.")
    lost_data = []
    for rd in debate_runs:
        for rec in rd.records:
            answers = rec.get("agent_round_parsed_answers") or []
            gt = rec.get("answer")
            n_a = len(answers)
            n_r = len(answers[0]) if answers else 0
            ever, final_has = False, False
            first_correct_round = None
            for a in range(n_a):
                for r in range(n_r):
                    if r < len(answers[a]) and answers[a][r] is not None and str(answers[a][r]) == str(gt):
                        ever = True
                        if first_correct_round is None:
                            first_correct_round = r + 1
                        if r == n_r - 1:
                            final_has = True
            lost_data.append({
                "dataset": rd.meta.dataset,
                "model": rd.meta.model_tag or "unknown",
                "orig_id": rec.get("orig_id"),
                "ever_correct": ever,
                "final_has_correct": final_has,
                "lost": ever and not final_has,
                "first_correct_round": first_correct_round,
            })
    if lost_data:
        ldf = pd.DataFrame(lost_data)
        lagg = ldf.groupby(["dataset", "model"]).agg(
            total=("lost", "count"),
            ever_correct=("ever_correct", "sum"),
            final_correct=("final_has_correct", "sum"),
            lost=("lost", "sum"),
        ).reset_index()
        lagg["ever_pct"] = (lagg["ever_correct"] / lagg["total"]).apply(fmt_pct)
        lagg["lost_pct"] = (lagg["lost"] / lagg["total"]).apply(fmt_pct)
        st.dataframe(lagg, use_container_width=True)

        # Which questions lost correct?
        lost_questions = ldf[ldf["lost"]].sort_values(["dataset", "model", "orig_id"])
        if not lost_questions.empty:
            with st.expander(f"Questions that lost correct ({len(lost_questions)} total)", expanded=False):
                st.dataframe(lost_questions[["dataset", "model", "orig_id", "first_correct_round"]], use_container_width=True)

    # --- Round-level accuracy convergence (for runs with multiple judged rounds) ---
    st.subheader("Debate Round Effectiveness")
    st.markdown(
        "If multiple debate rounds were judged, this shows how judge and majority "
        "accuracy evolve with each additional round of debate."
    )

    # Collect runs that have different round numbers
    round_acc_data = []
    for rd in debate_runs:
        for rec in rd.records:
            answers = rec.get("agent_round_parsed_answers") or []
            gt = rec.get("answer")
            n_a = len(answers)
            n_r = len(answers[0]) if answers else 0
            for r in range(n_r):
                # Majority of agents at round r
                ra = [answers[a][r] for a in range(n_a) if r < len(answers[a])]
                non_none = [a for a in ra if a is not None]
                if non_none:
                    counts = Counter(non_none)
                    top_count = max(counts.values())
                    top = [a for a, c in counts.items() if c == top_count]
                    maj_ans = top[0] if len(top) == 1 else None
                else:
                    maj_ans = None
                maj_correct = int(str(maj_ans) == str(gt)) if maj_ans is not None and gt is not None else 0
                any_correct = int(any(str(a) == str(gt) for a in ra if a is not None))
                round_acc_data.append({
                    "dataset": rd.meta.dataset,
                    "model": rd.meta.model_tag or "unknown",
                    "round": r + 1,
                    "majority_correct": maj_correct,
                    "any_agent_correct": any_correct,
                })

    if round_acc_data:
        radf = pd.DataFrame(round_acc_data)
        ragg = radf.groupby(["dataset", "model", "round"]).agg(
            majority_acc=("majority_correct", "mean"),
            any_correct_rate=("any_agent_correct", "mean"),
        ).reset_index()
        ragg["label"] = ragg["dataset"] + " / " + ragg["model"]

        fig_conv = go.Figure()
        for lbl in ragg["label"].unique():
            sub = ragg[ragg["label"] == lbl]
            fig_conv.add_trace(go.Scatter(
                x=sub["round"], y=sub["majority_acc"],
                mode="lines+markers", name=f"{lbl} (majority)",
                line=dict(dash="solid"),
            ))
            fig_conv.add_trace(go.Scatter(
                x=sub["round"], y=sub["any_correct_rate"],
                mode="lines+markers", name=f"{lbl} (any agent correct)",
                line=dict(dash="dot"),
            ))
        fig_conv.update_layout(
            title="Per-Round Majority Accuracy & Correct Agent Presence",
            xaxis_title="Round", yaxis_title="Rate",
            yaxis_tickformat=".0%", yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    download_csv(debate_df, "debate_analysis_runs.csv")


# ---------------------------------------------------------------------------
# Page: Model Statistics
# ---------------------------------------------------------------------------


def page_model_statistics(runs: list[RunData], filtered_df: pd.DataFrame):
    st.header("Model Statistics")

    if filtered_df.empty:
        st.warning("No runs match the current filters.")
        return

    filt_runs = filter_runs(runs, filtered_df)

    # Aggregated
    st.subheader("Aggregated Statistics")
    agg = filtered_df.groupby(["dataset", "model", "mode"]).agg(
        n_runs=("filename", "count"),
        total_q=("n_questions", "sum"),
        total_correct=("n_correct", "sum"),
        mean_acc=("accuracy", "mean"),
        std_acc=("accuracy", "std"),
        min_acc=("accuracy", "min"),
        max_acc=("accuracy", "max"),
    ).reset_index()
    agg["pooled_acc"] = agg["total_correct"] / agg["total_q"]
    agg["ci_lo"] = agg.apply(lambda r: wilson_ci(int(r["total_correct"]), int(r["total_q"]))[0], axis=1)
    agg["ci_hi"] = agg.apply(lambda r: wilson_ci(int(r["total_correct"]), int(r["total_q"]))[1], axis=1)

    st.dataframe(
        agg.style.format({
            "mean_acc": "{:.1%}", "std_acc": "{:.3f}", "min_acc": "{:.1%}",
            "max_acc": "{:.1%}", "pooled_acc": "{:.1%}", "ci_lo": "{:.1%}", "ci_hi": "{:.1%}",
        }),
        use_container_width=True,
    )
    download_csv(agg, "model_statistics.csv")

    # CI plot
    st.subheader("Accuracy with 95% Wilson CI")
    agg["label"] = agg["dataset"] + " / " + agg["mode"] + " / " + agg["model"]
    fig = go.Figure()
    for _, row in agg.iterrows():
        fig.add_trace(go.Bar(
            x=[row["label"]], y=[row["pooled_acc"]],
            error_y=dict(type="data", symmetric=False,
                         array=[row["ci_hi"] - row["pooled_acc"]],
                         arrayminus=[row["pooled_acc"] - row["ci_lo"]]),
            name=row["label"],
            text=f"{row['pooled_acc']:.1%}", textposition="outside",
        ))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], showlegend=False, xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Subset size impact
    st.subheader("Subset Size Impact")
    sa = filtered_df.groupby(["dataset", "model", "mode", "subset_n"]).agg(
        n_runs=("filename", "count"), mean_acc=("accuracy", "mean"),
    ).reset_index()
    if not sa.empty and sa["subset_n"].nunique() > 1:
        sa["label"] = sa["dataset"] + " / " + sa["mode"] + " / " + sa["model"]
        fig_s = px.line(
            sa, x="subset_n", y="mean_acc", color="label",
            markers=True, title="Accuracy by Subset Size",
        )
        fig_s.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.info("Only one subset size available.")

    # Per-question analysis
    st.subheader("Per-Question Difficulty Analysis")
    q_data: dict[tuple, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for rd in filt_runs:
        for rec in rd.records:
            key = (rd.meta.dataset, rec.get("orig_id"))
            method = f"{rd.meta.mode}/{rd.meta.model_tag or 'unknown'}"
            q_data[key][method].append(bool(rec.get("final_correct")))

    if q_data:
        q_rows = []
        for (ds, oid), methods in q_data.items():
            row: dict[str, Any] = {"dataset": ds, "orig_id": oid}
            all_results: list[bool] = []
            for method, results in methods.items():
                row[f"{method}_rate"] = sum(results) / len(results)
                row[f"{method}_n"] = len(results)
                all_results.extend(results)
            row["overall_rate"] = sum(all_results) / len(all_results) if all_results else 0
            q_rows.append(row)
        qdf = pd.DataFrame(q_rows)
        if not qdf.empty:
            n_show = st.slider("Questions to show", 5, min(100, len(qdf)), 20, key="ms_n")
            view = st.radio("Show", ["Hardest", "Easiest", "All"], horizontal=True, key="ms_view")
            if view == "Hardest":
                qdf_show = qdf.sort_values("overall_rate").head(n_show)
            elif view == "Easiest":
                qdf_show = qdf.sort_values("overall_rate", ascending=False).head(n_show)
            else:
                qdf_show = qdf.head(n_show)
            rate_cols = [c for c in qdf_show.columns if c.endswith("_rate")]
            fmt = {c: "{:.0%}" for c in rate_cols}
            st.dataframe(qdf_show.style.format(fmt), use_container_width=True)
            download_csv(qdf, "per_question_analysis.csv")


# ---------------------------------------------------------------------------
# Page: Per-Question Deep Dive
# ---------------------------------------------------------------------------


def page_question_deep_dive(runs: list[RunData], filtered_df: pd.DataFrame):
    st.header("Per-Question Deep Dive")

    filt_runs = filter_runs(runs, filtered_df)
    if not filt_runs:
        st.warning("No runs match the current filters.")
        return

    col1, col2 = st.columns(2)
    with col1:
        datasets = sorted(set(rd.meta.dataset for rd in filt_runs))
        sel_ds = st.selectbox("Dataset", datasets, key="qdd_ds")

    ds_runs = [rd for rd in filt_runs if rd.meta.dataset == sel_ds]
    all_oids = sorted(set(
        rec.get("orig_id") for rd in ds_runs for rec in rd.records if rec.get("orig_id") is not None
    ))

    with col2:
        sel_qid = st.selectbox("Question (orig_id)", all_oids, key="qdd_qid")
    if sel_qid is None:
        return

    q_recs = [(rd, rec) for rd in ds_runs for rec in rd.records if rec.get("orig_id") == sel_qid]
    if not q_recs:
        st.info("No records found.")
        return

    first = q_recs[0][1]
    st.subheader("Question")
    st.text_area("", first.get("question", ""), height=150, disabled=True, key="qdd_qt")
    st.markdown(f"**Ground Truth**: `{first.get('answer', 'N/A')}`")

    # Summary table
    st.subheader(f"Results Across {len(q_recs)} Runs")
    rows = []
    for rd, rec in q_recs:
        row: dict[str, Any] = {
            "Run": rd.meta.filename,
            "Mode": rd.meta.mode,
            "Model": rd.meta.model_tag or "unknown",
            "Seed": rd.meta.seed,
            "Answer": rec.get("final_answer"),
            "Correct": bool(rec.get("final_correct")),
        }
        if rd.meta.mode == "debate":
            row["Judge"] = rec.get("final_judge_answer")
            row["Judge OK"] = bool(rec.get("final_judge_correct"))
            row["Majority"] = rec.get("final_majority_answer")
            row["Maj OK"] = bool(rec.get("final_majority_correct"))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Debate evolutions
    debate_recs = [(rd, rec) for rd, rec in q_recs if rd.meta.mode == "debate"]
    if debate_recs:
        st.subheader("Debate Answer Evolution Comparison")
        for rd, rec in debate_recs:
            answers = rec.get("agent_round_parsed_answers") or []
            gt = rec.get("answer")
            with st.expander(
                f"{rd.meta.filename} (seed={rd.meta.seed})",
                expanded=True,
            ):
                if answers:
                    n_a = len(answers)
                    n_r = max(len(a) for a in answers) if answers else 0
                    evo_data = []
                    for ai in range(n_a):
                        for ri in range(n_r):
                            ans = answers[ai][ri] if ri < len(answers[ai]) else None
                            evo_data.append({
                                "Agent": f"Ag{ai+1}",
                                "Round": ri + 1,
                                "Answer": str(ans) if ans is not None else "None",
                                "Correct": (str(ans) == str(gt)) if ans is not None and gt is not None else False,
                            })
                    edf = pd.DataFrame(evo_data)
                    piv = edf.pivot(index="Agent", columns="Round", values="Answer")
                    cpiv = edf.pivot(index="Agent", columns="Round", values="Correct")
                    fig = go.Figure(data=go.Heatmap(
                        z=cpiv.values.astype(int),
                        x=[f"R{c}" for c in cpiv.columns],
                        y=cpiv.index.tolist(),
                        colorscale=[[0, "#ff6b6b"], [1, "#51cf66"]],
                        showscale=False,
                        text=piv.values,
                        texttemplate="%{text}",
                        textfont={"size": 14},
                    ))
                    fig.update_layout(height=max(180, 50 * n_a + 80), margin=dict(t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                cj, cm = st.columns(2)
                cj.markdown(f"**Judge**: `{rec.get('final_judge_answer')}` ({'correct' if rec.get('final_judge_correct') else 'wrong'})")
                cm.markdown(f"**Majority**: `{rec.get('final_majority_answer')}` ({'correct' if rec.get('final_majority_correct') else 'wrong'})")


# ---------------------------------------------------------------------------
# Page: Raw Data & Export
# ---------------------------------------------------------------------------


def page_raw_data(runs: list[RunData], filtered_df: pd.DataFrame):
    st.header("Raw Data & Export")

    filt_runs = filter_runs(runs, filtered_df)
    if not filt_runs:
        st.warning("No runs match the current filters.")
        return

    st.subheader("Summary CSV Export")
    download_csv(filtered_df, "all_runs_summary.csv", "Download All Runs Summary")

    # Per-run export
    st.subheader("Per-Run Data Export")
    run_map = {
        f"{rd.meta.filename} ({rd.meta.mode} | {rd.meta.dataset})": i
        for i, rd in enumerate(filt_runs)
    }
    sel = st.selectbox("Select Run to Export", list(run_map.keys()), key="rd_sel")
    if sel:
        rd = filt_runs[run_map[sel]]
        qdf = records_to_question_df(rd.records, rd.meta)
        st.dataframe(qdf, use_container_width=True, height=400)
        download_csv(qdf, f"{rd.meta.filename.replace('.jsonl', '')}_questions.csv", "Download Questions CSV")

        # Full JSONL download
        buf = io.StringIO()
        for rec in rd.records:
            buf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        st.download_button(
            "Download Raw JSONL",
            buf.getvalue(),
            file_name=rd.meta.filename,
            mime="application/jsonl",
        )

    # Raw record viewer
    st.subheader("Raw Record Viewer")
    if sel:
        rd = filt_runs[run_map[sel]]
        rec_idx = st.number_input("Record index", 0, len(rd.records) - 1, 0, key="rd_idx")
        rec = rd.records[int(rec_idx)]
        # Omit very large fields for display
        display_rec = {}
        for k, v in rec.items():
            if k == "raw_task":
                display_rec[k] = "{...}"  # too large
            elif k == "agent_responses":
                display_rec[k] = f"[{len(v)} agents]"
            elif k == "sample_completions":
                display_rec[k] = f"[{len(v)} completions]"
            else:
                display_rec[k] = v
        st.json(display_rec)

        # Full record
        with st.expander("Full JSON (including large fields)", expanded=False):
            st.json(rec)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def get_results_dir() -> str:
    """Get results directory from command-line args."""
    for flag in ("--results-dir", "--results_dir"):
        if flag in sys.argv:
            idx = sys.argv.index(flag)
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

    # Sidebar: configuration
    with st.sidebar:
        st.header("Configuration")
        default_dir = get_results_dir()
        results_dir = st.text_input(
            "Results Directory",
            value=default_dir,
            help="Path to folder containing result .jsonl files (aime25/ and gpqa/ subfolders)",
        )

        if not results_dir:
            st.info("Enter a results directory path to load data.")
            st.stop()

        if not Path(results_dir).exists():
            st.error(f"Directory not found: {results_dir}")
            st.stop()

        runs = load_all_runs(results_dir)
        if not runs:
            st.warning(f"No valid result files found in {results_dir}")
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
                "Raw Data & Export",
            ],
            key="nav",
        )

        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # Apply global filters
    filtered_df = apply_global_filters(summary_df)

    # Render page
    pages = {
        "Overview": page_overview,
        "Run Comparison": page_run_comparison,
        "Transcript Viewer": page_transcript_viewer,
        "Debate Analysis": page_debate_analysis,
        "Model Statistics": page_model_statistics,
        "Per-Question Deep Dive": page_question_deep_dive,
        "Raw Data & Export": page_raw_data,
    }
    pages[page](runs, filtered_df)


if __name__ == "__main__":
    main()
