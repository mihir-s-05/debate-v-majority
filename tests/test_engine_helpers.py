from __future__ import annotations

from debate_v_majority.engines.runtime_env import (
    _is_kv_cache_dtype_unsupported_error,
    _maybe_disable_fp8_kv_for_backend,
    _resolve_kv_cache_dtype,
)


def test_resolve_kv_cache_dtype_defaults_to_auto():
    dtype, reason = _resolve_kv_cache_dtype(None)
    assert dtype == "auto"
    assert reason is None


def test_maybe_disable_fp8_kv_for_flash_backend():
    dtype, reason = _maybe_disable_fp8_kv_for_backend("fp8_e5m2", attn_backend="FLASH_ATTN")
    assert dtype == "auto"
    assert "rejected fp8 KV cache" in (reason or "")


def test_maybe_disable_fp8_kv_for_non_flash_backend():
    dtype, reason = _maybe_disable_fp8_kv_for_backend("fp8_e5m2", attn_backend="XFORMERS")
    assert dtype == "fp8_e5m2"
    assert reason is None


def test_is_kv_cache_dtype_unsupported_error_matches_message():
    err = RuntimeError("kv_cache_dtype fp8 is unsupported on this backend")
    assert _is_kv_cache_dtype_unsupported_error(err) is True
