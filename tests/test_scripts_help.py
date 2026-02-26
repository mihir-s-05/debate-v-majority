from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_help(script_name: str) -> subprocess.CompletedProcess[str]:
    script = ROOT / "scripts" / script_name
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )


def test_debate_script_help_runs():
    proc = _run_help("debate-v-majority")
    assert proc.returncode == 0
    assert "--dataset" in proc.stdout
    assert "--mode" in proc.stdout


def test_rejudge_script_help_runs():
    proc = _run_help("rejudge-targets")
    assert proc.returncode == 0
    assert "--dataset" in proc.stdout
    assert "--target" in proc.stdout


def test_analyze_script_help_runs():
    proc = _run_help("analyze-results")
    assert proc.returncode == 0
    assert "--results-dir" in proc.stdout
    assert "--out-dir" in proc.stdout
