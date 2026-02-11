#!/usr/bin/env python3
"""
Convenience launcher for the Model Run Dashboard.

Usage:
    python launch_dashboard.py /path/to/results
    python launch_dashboard.py --results-dir /path/to/results --port 8501
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Launch the interactive Model Run Dashboard (Streamlit)."
    )
    ap.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Path to the folder containing result .jsonl files (with aime25/ and gpqa/ subfolders).",
    )
    ap.add_argument(
        "--results-dir",
        dest="results_dir_flag",
        default=None,
        help="Path to the folder containing result .jsonl files (alternative to positional arg).",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to serve the dashboard on (default: 8501).",
    )
    ap.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0).",
    )
    args = ap.parse_args()

    results_dir = args.results_dir or args.results_dir_flag or ""

    dashboard_path = Path(__file__).resolve().parent / "dashboard.py"
    if not dashboard_path.exists():
        print(f"Error: dashboard.py not found at {dashboard_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true",
    ]

    if results_dir:
        cmd += ["--", "--results-dir", results_dir]

    print(f"Launching dashboard at http://{args.host}:{args.port}")
    if results_dir:
        print(f"Results directory: {results_dir}")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
