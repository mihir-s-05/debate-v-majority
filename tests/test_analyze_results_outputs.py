from __future__ import annotations

import json
from pathlib import Path

from debate_v_majority.tools import analyze_results as ar


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_analyze_results_writes_metrics_only_outputs(tmp_path: Path):
    results_dir = tmp_path / "results"
    out_dir = tmp_path / "out"
    gpqa_quick = results_dir / "gpqa_quick"

    run_name = "single_gpqa_n1_seed1_20260101_000000_Qwen_Qwen3-8B.jsonl"
    _write_jsonl(
        gpqa_quick / run_name,
        [
            {
                "orig_id": 1,
                "answer": "A",
                "final_answer": "A",
                "final_correct": 1,
                "sample_completions": ["\\boxed{A}"],
            }
        ],
    )

    ar.TARGET_MODEL_TAG = "Qwen/Qwen3-8B"
    ar.analyze(results_dir, out_dir)

    summary_path = out_dir / "summary.json"
    tables_path = out_dir / "tables.md"
    assert summary_path.exists()
    assert tables_path.exists()
    assert not (out_dir / "examples.json").exists()
    assert not (out_dir / "case_studies.md").exists()

    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "examples" not in payload
