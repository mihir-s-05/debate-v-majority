"""
Debug Majority Debate Package

Run multi-agent debate, majority voting, or single-response inference
on GSM8K, AIME25, and GPQA datasets using vLLM backend.
"""
from __future__ import annotations

from typing import Literal

# Type definitions
Mode = Literal["single", "majority", "debate"]
DatasetName = Literal["gsm8k", "aime25", "gpqa"]
Backend = Literal["vllm"]
Parallelism = Literal["auto", "tp", "dp", "hybrid"]

__all__ = ["Mode", "DatasetName", "Backend", "Parallelism"]
