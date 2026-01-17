"""
vLLM engine infrastructure for inference.

Contains:
- SamplingConfig for generation parameters
- VLLMInferenceEngine for single-instance inference
- Helper functions for context length, RoPE scaling, etc.
"""
from __future__ import annotations

import ctypes
import gc
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .shared import (
    is_cuda_oom,
    is_cuda_device_side_assert,
    is_vllm_engine_dead,
    is_vllm_oom_like,
    is_prompt_too_long,
    is_flash_attn_import_error,
    extract_prompt_length_tokens,
    PromptTokenCounter,
    truncate_chat_messages_to_fit,
    strip_thinking_from_contexts,
    THINKING_STRIP_THRESHOLD,
)


# =============================================================================
# Sampling configuration
# =============================================================================


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1  # -1 means disabled
    max_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
        }


# Global sampling config
_SAMPLING_CONFIG: SamplingConfig = SamplingConfig()


def load_generation_config(model_name: str) -> dict[str, Any]:
    """Load generation_config.json from a HuggingFace model."""
    try:
        from transformers import GenerationConfig
    except ImportError:
        return {}

    try:
        gen_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
        result: dict[str, Any] = {}

        if hasattr(gen_config, "temperature") and gen_config.temperature is not None:
            result["temperature"] = float(gen_config.temperature)
        if hasattr(gen_config, "top_p") and gen_config.top_p is not None:
            result["top_p"] = float(gen_config.top_p)
        if hasattr(gen_config, "top_k") and gen_config.top_k is not None:
            result["top_k"] = int(gen_config.top_k)
        if hasattr(gen_config, "max_new_tokens") and gen_config.max_new_tokens is not None:
            result["max_new_tokens"] = int(gen_config.max_new_tokens)

        return result
    except Exception as e:
        print(f"[config] Could not load generation_config for {model_name}: {e}", file=sys.stderr)
        return {}


def build_sampling_config(model_name: str) -> SamplingConfig:
    """Build a SamplingConfig from model's generation_config.json."""
    config = SamplingConfig()
    model_gen_config = load_generation_config(model_name)

    if model_gen_config:
        print(f"[config] Loaded generation_config from {model_name}: {model_gen_config}", file=sys.stderr)
        if "temperature" in model_gen_config:
            config.temperature = model_gen_config["temperature"]
        if "top_p" in model_gen_config:
            config.top_p = model_gen_config["top_p"]
        if "top_k" in model_gen_config:
            config.top_k = model_gen_config["top_k"]
        if "max_new_tokens" in model_gen_config:
            config.max_tokens = model_gen_config["max_new_tokens"]

    return config


def get_sampling_config() -> SamplingConfig:
    """Get the global sampling configuration."""
    return _SAMPLING_CONFIG


def set_sampling_config(config: SamplingConfig) -> None:
    """Set the global sampling configuration."""
    global _SAMPLING_CONFIG
    _SAMPLING_CONFIG = config


# =============================================================================
# Context length and model configuration
# =============================================================================


@lru_cache(maxsize=32)
def _load_hf_config_cached(model_name: str) -> Any | None:
    """Load HuggingFace model config with caching."""
    try:
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None


def infer_native_context_len(model_name: str) -> int | None:
    """Infer the model's native max context length from config."""
    cfg = _load_hf_config_cached(model_name)
    if cfg is None:
        return None

    candidates: list[int] = []
    for attr in (
        "max_position_embeddings",
        "n_positions",
        "seq_length",
        "max_seq_len",
        "max_sequence_length",
    ):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and 0 < v <= 1_000_000:
            candidates.append(v)

    # Check for RoPE scaling
    rope = getattr(cfg, "rope_scaling", None)
    if isinstance(rope, dict):
        base = rope.get("original_max_position_embeddings")
        factor = rope.get("factor")
        if isinstance(base, int) and base > 0 and isinstance(factor, (int, float)) and factor > 0:
            try:
                derived = int(round(float(base) * float(factor)))
                if 0 < derived <= 1_000_000:
                    candidates.append(derived)
            except Exception:
                pass

    return max(candidates) if candidates else None


def _get_rope_scaling_type(cfg: Any) -> str | None:
    """Get the RoPE scaling type from config."""
    rope = getattr(cfg, "rope_scaling", None)
    if not isinstance(rope, dict):
        return None
    t = rope.get("type") or rope.get("rope_type") or rope.get("rope_scaling_type")
    if not isinstance(t, str):
        return None
    return t.strip().lower() or None


def model_supports_yarn(model_name: str) -> bool:
    """Check if model supports YaRN RoPE scaling."""
    cfg = _load_hf_config_cached(model_name)
    if cfg is None:
        return False
    return _get_rope_scaling_type(cfg) == "yarn"


def build_rope_scaling_overrides(
    model_name: str,
    *,
    target_len: int,
    force_yarn: bool = False,
) -> dict[str, Any] | None:
    """Build HF config overrides for extending context via RoPE scaling."""
    target_len = int(target_len)

    def _maybe_add_context_len_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
        """
        Some model implementations (and compiled kernels) size RoPE caches based on
        config context-length fields (e.g., max_position_embeddings). If we only
        set rope_scaling but leave these at the native value, long-context runs can
        hit CUDA device-side asserts due to index-out-of-bounds accesses.
        """
        cfg_local = _load_hf_config_cached(model_name)
        if cfg_local is None:
            return overrides

        # Common names across HF configs.
        for key in (
            "max_position_embeddings",
            "n_positions",
            "seq_length",
            "max_seq_len",
            "max_sequence_length",
        ):
            if not hasattr(cfg_local, key):
                continue
            cur = getattr(cfg_local, key, None)
            if cur is None or isinstance(cur, int):
                overrides[key] = target_len
        return overrides

    def _qwen3_yarn_base_len() -> int | None:
        """
        Qwen3 sets `max_position_embeddings=40960` by default, but the model card
        documents a native 32,768-token window and recommends YaRN factors based
        on 32,768 (e.g., 65,536 => factor 2.0).
        """
        cfg_local = _load_hf_config_cached(model_name)
        if cfg_local is None:
            return None
        if getattr(cfg_local, "model_type", None) != "qwen3":
            return None
        mpe = getattr(cfg_local, "max_position_embeddings", None)
        if mpe == 40960:
            return 32768
        return None

    native = infer_native_context_len(model_name)

    if force_yarn:
        base = _qwen3_yarn_base_len() or native
        if base is None or base <= 0:
            raise ValueError(
                f"YaRN requires inferred native context length, but it could not be determined for {model_name!r}."
            )
        if base >= target_len:
            return None
        factor = float(target_len) / float(base)
        overrides = {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": int(base),
            }
        }
        return _maybe_add_context_len_overrides(overrides)

    if native is None or native >= target_len or native <= 0:
        return None

    cfg = _load_hf_config_cached(model_name)
    if cfg is None:
        return None

    rope_type = _get_rope_scaling_type(cfg)
    if rope_type is None:
        return None

    rope = getattr(cfg, "rope_scaling", None)
    rope_original = None
    if isinstance(rope, dict) and isinstance(rope.get("original_max_position_embeddings"), int):
        rope_original = int(rope["original_max_position_embeddings"])

    base = rope_original if (rope_original is not None and rope_original > 0) else native
    factor = float(target_len) / float(base)

    if rope_type == "yarn":
        overrides = {"rope_scaling": {"rope_type": "yarn", "factor": factor, "original_max_position_embeddings": base}}
        return _maybe_add_context_len_overrides(overrides)
    if rope_type in {"dynamic", "linear", "ntk"}:
        overrides = {"rope_scaling": {"rope_type": rope_type, "factor": factor}}
        return _maybe_add_context_len_overrides(overrides)
    return None


# =============================================================================
# vLLM helpers
# =============================================================================


def _prepend_ld_library_path(path: str) -> None:
    p = (path or "").strip()
    if not p or not os.path.isdir(p):
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [x for x in cur.split(":") if x]
    if p in parts:
        return
    os.environ["LD_LIBRARY_PATH"] = p + (":" + cur if cur else "")


def _sanitize_flash_attn_env_for_device(*, verbose: bool = False) -> None:
    """Ensure FlashAttention version is compatible with GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        major, minor = torch.cuda.get_device_capability()
        # FA3 requires compute capability >= 9.0 (Hopper)
        if major < 9 and os.environ.get("VLLM_FLASH_ATTN_VERSION") == "3":
            if verbose:
                print(f"[flash] GPU compute capability {major}.{minor} < 9.0; forcing VLLM_FLASH_ATTN_VERSION=2", file=sys.stderr)
            os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
    except Exception:
        pass


def _resolve_kv_cache_dtype(requested: str | None) -> tuple[str, str | None]:
    """Resolve KV cache dtype with fallback for unsupported GPUs."""
    req = (requested or "").strip() or "auto"
    if not req.lower().startswith("fp8"):
        return req, None

    try:
        import torch
        if not torch.cuda.is_available():
            return "auto", "CUDA not available"
        major, minor = torch.cuda.get_device_capability()
        # FP8 KV cache needs compute capability >= 8.9 (Ada/Hopper)
        if major < 8 or (major == 8 and minor < 9):
            return "auto", f"GPU compute capability {major}.{minor} < 8.9; fp8 KV cache requires Ada/Hopper+"
    except Exception as e:
        return "auto", f"Could not check GPU capability: {e}"

    return req, None


def _prepare_vllm_runtime_linking() -> None:
    """
    Prepare environment for vLLM runtime.

    vLLM wheels can depend on shared libraries (e.g. libcudart.so.12, libtorch.so)
    that are not on the default loader path.
    """
    # Safer than fork with CUDA.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # Reduce fragmentation / improve allocator stability in long-running inference loops.
    os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ["PYTORCH_ALLOC_CONF"])

    # torch/lib (libtorch.so, libc10.so, etc.)
    try:
        import torch  # noqa: F401
        torch_lib = str(Path(torch.__file__).resolve().parent / "lib")
        _prepend_ld_library_path(torch_lib)
    except Exception:
        pass

    # pip CUDA runtime (libcudart.so.12) when installed as a Python package.
    try:
        import nvidia.cuda_runtime  # type: ignore
        cuda_rt_lib = str(Path(nvidia.cuda_runtime.__file__).resolve().parent / "lib")
        _prepend_ld_library_path(cuda_rt_lib)
    except Exception:
        pass

    def _load_global(path: str) -> bool:
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError:
            return False

    # Load libcudart by absolute path if available (LD_LIBRARY_PATH updates may not apply mid-process).
    try:
        import nvidia.cuda_runtime  # type: ignore
        cuda_rt_lib = Path(nvidia.cuda_runtime.__file__).resolve().parent / "lib"
        libcudart12 = cuda_rt_lib / "libcudart.so.12"
        if libcudart12.exists():
            _load_global(str(libcudart12))
    except Exception:
        pass

    # As a fallback, try generic loader search.
    _load_global("libcudart.so.12")

    # Ensure libtorch deps are globally visible for vLLM's native extensions.
    try:
        import torch
        torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
        for name in ("libc10.so", "libtorch_cpu.so", "libtorch_cuda.so", "libtorch.so"):
            p = torch_lib_dir / name
            if p.exists():
                _load_global(str(p))
    except Exception:
        pass


def _is_flash_attn_backend(attn_backend: str | None) -> bool:
    s = (attn_backend or "").strip().upper()
    return bool(s) and ("FLASH" in s) and ("ATTN" in s)


def _maybe_disable_fp8_kv_for_backend(kv_dtype: str, *, attn_backend: str | None) -> tuple[str, str | None]:
    """
    Some vLLM builds reject fp8 KV cache on specific attention backends.
    """
    if not (kv_dtype or "").lower().startswith("fp8"):
        return kv_dtype, None
    if _is_flash_attn_backend(attn_backend):
        return "auto", f"attention backend {attn_backend!r} rejected fp8 KV cache"
    return kv_dtype, None


def _is_kv_cache_dtype_unsupported_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return ("kv_cache_dtype" in msg or "kv cache dtype" in msg) and ("unsupported" in msg or "not supported" in msg)


def require_vllm_deps() -> None:
    """Check that vLLM is installed and CUDA is available."""
    _prepare_vllm_runtime_linking()
    try:
        import vllm  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing vLLM for local model inference. Install with: pip install vllm"
        ) from e

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("vLLM requires CUDA but torch.cuda.is_available() is False")
        if torch.cuda.device_count() <= 0:
            raise RuntimeError("No CUDA devices found")
    except Exception as e:
        raise RuntimeError(f"CUDA setup issue: {e}") from e


# =============================================================================
# Batch size cache for OOM recovery
# =============================================================================


class BatchSizeCache:
    """
    Cache for dynamically discovered *safe* batch sizes.

    Important: we only want to persist a batch size across calls when we have evidence
    that larger batches may OOM (i.e., after an actual OOM). A previous small successful
    batch (e.g., single mode with only 50 prompts) should NOT cap later calls (e.g., majority
    mode with 250 prompts and --batch_size 256).
    """
    _instance: "BatchSizeCache | None" = None
    PROBE_INTERVAL: int = 5

    def __init__(self) -> None:
        # Only set when we hit an OOM-like failure and back off. When None, we start each
        # call at the caller-provided cap (or full size).
        self.safe_batch_size: int | None = None
        self.success_count: int = 0

    @classmethod
    def get(cls) -> "BatchSizeCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record_success(self, bs: int) -> None:
        # Do not update safe_batch_size on success: success at a small batch may simply
        # reflect a small workload, not a hardware limit.
        self.success_count += 1

    def record_oom(self, new_bs: int) -> None:
        self.safe_batch_size = new_bs
        self.success_count = 0

    def should_probe(self) -> bool:
        return self.safe_batch_size is not None and self.success_count >= self.PROBE_INTERVAL

    def get_probe_bs(self, max_possible: int) -> int:
        if self.safe_batch_size is None:
            return max_possible
        return min(max_possible, self.safe_batch_size + max(1, self.safe_batch_size // 4))


# =============================================================================
# VLLMInferenceEngine
# =============================================================================


class VLLMInferenceEngine:
    """vLLM inference engine with OOM recovery."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        attention_backend: str | None = None,
        hf_overrides: dict[str, Any] | None = None,
        kv_cache_dtype: str = "auto",
        *,
        enforce_eager: bool = False,
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None
        self._sampling_params = None
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._attention_backend = attention_backend
        self._hf_overrides = hf_overrides
        self._kv_cache_dtype = kv_cache_dtype
        self._enforce_eager = enforce_eager
        # Optimized defaults for high-throughput, multi-GPU eval workloads.
        # Keep these internal (avoid proliferating CLI flags).
        self._enable_prefix_caching = True
        self._enable_chunked_prefill = True

    @property
    def context_len_tokens(self) -> int:
        return int(self._max_model_len)

    def initialize(self) -> None:
        """Initialize the vLLM engine."""
        if self._llm is not None:
            return

        _prepare_vllm_runtime_linking()
        if self._attention_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = self._attention_backend
        _sanitize_flash_attn_env_for_device(verbose=False)
        if self._hf_overrides and "rope_scaling" in self._hf_overrides:
            os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

        for attempt in range(2):
            try:
                from vllm import LLM, SamplingParams

                kv_dtype, kv_reason = _resolve_kv_cache_dtype(self._kv_cache_dtype)
                attn_backend = os.environ.get("VLLM_ATTENTION_BACKEND") or self._attention_backend
                kv_dtype2, kv_reason2 = _maybe_disable_fp8_kv_for_backend(kv_dtype, attn_backend=attn_backend)
                if kv_dtype2 != kv_dtype:
                    kv_dtype = kv_dtype2
                    kv_reason = kv_reason2 or kv_reason
                if kv_reason:
                    print(
                        f"[kv] Falling back kv_cache_dtype={kv_dtype!r} (requested {self._kv_cache_dtype!r}): {kv_reason}",
                        file=sys.stderr,
                    )
                else:
                    print(f"[kv] Using kv_cache_dtype={kv_dtype!r}", file=sys.stderr)

                llm_kwargs: dict[str, Any] = {
                    "enable_prefix_caching": bool(self._enable_prefix_caching),
                    "enable_chunked_prefill": bool(self._enable_chunked_prefill),
                }

                def _make_llm(kv_cache_dtype_final: str, extra: dict[str, Any]) -> Any:
                    return LLM(
                        model=self.model_name,
                        tensor_parallel_size=self.tensor_parallel_size,
                        max_model_len=self._max_model_len,
                        gpu_memory_utilization=self._gpu_memory_utilization,
                        trust_remote_code=True,
                        dtype="bfloat16",
                        hf_overrides=self._hf_overrides,
                        kv_cache_dtype=kv_cache_dtype_final,
                        enforce_eager=self._enforce_eager,
                        disable_log_stats=True,
                        **extra,
                    )

                try:
                    try:
                        self._llm = _make_llm(kv_dtype, llm_kwargs)
                    except TypeError:
                        # Older vLLM may not support these perf flags.
                        self._llm = _make_llm(kv_dtype, {})
                except Exception as e:
                    if kv_dtype.lower().startswith("fp8") and _is_kv_cache_dtype_unsupported_error(e):
                        self._llm = _make_llm("auto", llm_kwargs)
                    else:
                        raise

                # Use sampling config from model's generation_config
                sampling_cfg = get_sampling_config()
                sampling_kwargs: dict[str, Any] = {
                    "max_tokens": sampling_cfg.max_tokens or 4096,
                    "temperature": sampling_cfg.temperature,
                    "top_p": sampling_cfg.top_p,
                }
                if sampling_cfg.top_k > 0:
                    sampling_kwargs["top_k"] = sampling_cfg.top_k
                self._sampling_params = SamplingParams(**sampling_kwargs)
                return

            except Exception as e:
                if attempt == 0 and self._attention_backend is None and is_flash_attn_import_error(e):
                    print("[warn] vLLM failed to import flash-attn; falling back to TRITON_ATTN.", file=sys.stderr)
                    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
                    continue
                raise

    def generate_batch(
        self,
        contexts: list[list[dict[str, str]]],
        batch_size: int | None = None,
        *,
        sampling_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """Generate completions with OOM recovery."""
        if self._llm is None:
            self.initialize()
        if not contexts:
            return []

        return self._generate_with_oom_backoff(
            contexts=contexts,
            max_batch_size=batch_size,
            sampling_kwargs=sampling_kwargs,
        )

    def _generate_with_oom_backoff(
        self,
        contexts: list[list[dict[str, str]]],
        max_batch_size: int | None = None,
        sampling_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """Generate with dynamic batch sizing and OOM backoff."""
        import torch

        cache = BatchSizeCache.get()

        def _restart_with_safer_runtime(reason: str) -> None:
            """
            Best-effort recovery from unrecoverable CUDA failures (device-side asserts / illegal memory access)
            which can poison the CUDA context.
            """
            try:
                print(f"[recover] {reason}; restarting vLLM with safer settings.", file=sys.stderr)
            except Exception:
                pass
            # Avoid torch.compile / cudagraph after a device-side assert.
            # Do NOT force TORCH_SDPA here: some vLLM builds don't register it, and
            # forcing it can make recovery fail during engine startup.
            cur_backend = os.environ.get("VLLM_ATTENTION_BACKEND") or self._attention_backend
            if not cur_backend:
                os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
            self._enforce_eager = True
            # Reduce long-context instability at the cost of throughput.
            self._enable_prefix_caching = False
            # Some models (e.g., Qwen3) require chunked prefill; disabling it can crash.
            self._enable_chunked_prefill = True
            # fp8 KV cache can trigger backend-specific issues; fall back to auto.
            try:
                if str(getattr(self, "_kv_cache_dtype", "")).lower().startswith("fp8"):
                    self._kv_cache_dtype = "auto"
            except Exception:
                self._kv_cache_dtype = "auto"

            # Drop the current engine instance and reinitialize.
            try:
                if self._llm is not None:
                    del self._llm
            except Exception:
                pass
            self._llm = None
            self._sampling_params = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self.initialize()

        def _generate_chunk(chunk_contexts: list[list[dict[str, str]]]) -> list[str]:
            llm = self._llm
            sampling_params = self._sampling_params
            if sampling_kwargs is not None:
                from vllm import SamplingParams
                sampling_params = SamplingParams(**sampling_kwargs)
            try:
                outputs = llm.chat(messages=chunk_contexts, sampling_params=sampling_params)
            except TypeError:
                outputs = llm.chat(chunk_contexts, sampling_params=sampling_params)
            return [out.outputs[0].text for out in outputs]

        out: list[str] = []
        i = 0
        max_possible = len(contexts) if max_batch_size is None else min(max_batch_size, len(contexts))

        # Start at the caller-provided cap (or full size) unless we previously observed an OOM.
        current_bs = min(cache.safe_batch_size, max_possible) if cache.safe_batch_size else max_possible

        while i < len(contexts):
            remaining = len(contexts) - i
            current_bs = max(1, min(current_bs, remaining))

            batch_contexts = contexts[i : i + current_bs]
            try:
                out.extend(_generate_chunk(batch_contexts))
                cache.record_success(current_bs)
            except Exception as e:
                # Device-side asserts / illegal memory access can poison the CUDA context; restart vLLM.
                if is_cuda_device_side_assert(e) or is_vllm_engine_dead(e):
                    _restart_with_safer_runtime(f"{type(e).__name__}: {e}")
                    # Retry the same chunk at a smaller batch size to reduce stress.
                    current_bs = max(1, current_bs // 2)
                    continue
                if is_vllm_oom_like(e) and current_bs > 1:
                    gc.collect()
                    torch.cuda.empty_cache()
                    new_bs = max(1, current_bs // 2)
                    cache.record_oom(new_bs)
                    current_bs = new_bs
                    continue
                raise

            i += len(batch_contexts)

        return out

    def shutdown(self) -> None:
        """Release GPU memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._sampling_params = None
            gc.collect()


# =============================================================================
# AdaptiveVLLMEngine
# =============================================================================


class AdaptiveVLLMEngine:
    """
    Engine that starts with smaller context and auto-extends when needed.
    """

    def __init__(
        self,
        *,
        model_name: str,
        gpus: str = "0",
        gpu_memory_utilization: float = 0.9,
        start_max_model_len: int = 32768,
        max_model_len_cap: int = 131072,
        allow_long_max_model_len: bool = False,
        force_yarn: bool = False,
        kv_cache_dtype: str = "auto",
        enforce_eager: bool = False,
    ) -> None:
        self.model_name = model_name
        self._gpus = gpus
        self._gpu_memory_utilization = gpu_memory_utilization
        self._start_max_model_len = int(start_max_model_len)
        self._max_model_len_cap = int(max_model_len_cap)
        self._allow_long_max_model_len = bool(allow_long_max_model_len)
        self._force_yarn = bool(force_yarn)
        self._kv_cache_dtype = str(kv_cache_dtype)
        self._enforce_eager = bool(enforce_eager)
        # Keep perf knobs internal; engine defaults are tuned for multi-GPU throughput.
        self._enable_prefix_caching = True
        self._enable_chunked_prefill = True

        self._supports_yarn = model_supports_yarn(model_name)
        self._effective_allow_long = self._allow_long_max_model_len or self._supports_yarn or self._force_yarn

        self._native_max_model_len = infer_native_context_len(model_name)
        if self._effective_allow_long:
            if self._native_max_model_len is None:
                self._upgrade_cap = self._max_model_len_cap
            else:
                self._upgrade_cap = max(int(self._native_max_model_len), self._max_model_len_cap)
        else:
            if self._native_max_model_len is None:
                self._upgrade_cap = self._start_max_model_len
            else:
                self._upgrade_cap = int(self._native_max_model_len)

        self._current_max_model_len = min(self._start_max_model_len, self._upgrade_cap)
        self._engine: VLLMInferenceEngine | None = None
        self._counter = PromptTokenCounter(model_name)
        
        # Track whether thinking content should be stripped from contexts.
        # Once set to True, all subsequent contexts will have <think>...</think> stripped.
        self._thinking_stripped: bool = False

        native_s = str(self._native_max_model_len) if self._native_max_model_len else "unknown"
        print(f"[ctx] {model_name}: start={self._current_max_model_len} cap={self._upgrade_cap} native={native_s}", file=sys.stderr)

    @property
    def context_len_tokens(self) -> int:
        return int(self._current_max_model_len)

    @property
    def thinking_stripped(self) -> bool:
        """
        Whether thinking content is being stripped from contexts.
        
        Once True, all contexts passed to generate_batch will have
        <think>...</think> blocks removed. This is triggered automatically
        when context usage exceeds THINKING_STRIP_THRESHOLD.
        """
        return self._thinking_stripped

    def mark_thinking_stripped(self) -> None:
        """
        Mark that thinking content should be stripped from all contexts.
        
        This is called automatically when context pressure is detected,
        but can also be called manually to preemptively enable stripping.
        """
        if not self._thinking_stripped:
            print("[ctx] Enabling thinking content stripping for all contexts.", file=sys.stderr)
            self._thinking_stripped = True

    def _make_engine(self, *, max_model_len: int) -> VLLMInferenceEngine:
        """Create a new engine with the specified context length."""
        hf_overrides = None
        if self._effective_allow_long:
            hf_overrides = build_rope_scaling_overrides(
                self.model_name,
                target_len=max_model_len,
                force_yarn=self._force_yarn,
            )

        # Set CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = self._gpus
        gpu_count = len(self._gpus.split(","))

        eng = VLLMInferenceEngine(
            self.model_name,
            tensor_parallel_size=gpu_count,
            max_model_len=max_model_len,
            gpu_memory_utilization=self._gpu_memory_utilization,
            hf_overrides=hf_overrides,
            kv_cache_dtype=self._kv_cache_dtype,
            enforce_eager=self._enforce_eager,
        )
        eng.initialize()
        return eng

    def initialize(self) -> None:
        if self._engine is None:
            self._engine = self._make_engine(max_model_len=self._current_max_model_len)

    def shutdown(self) -> None:
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception:
                pass
            self._engine = None

    def _upgrade_to(self, new_len: int) -> None:
        """Restart with larger context."""
        new_len = min(int(new_len), self._upgrade_cap)
        if new_len <= self._current_max_model_len:
            return
        print(f"[ctx] Restarting vLLM with max_model_len={new_len} (was {self._current_max_model_len})", file=sys.stderr)
        self.shutdown()
        self._current_max_model_len = new_len
        self.initialize()

    def _maybe_upgrade_for_contexts(self, contexts: list[list[dict[str, str]]]) -> None:
        """Check if we need to upgrade context length."""
        if self._current_max_model_len >= self._upgrade_cap:
            return

        # Account for max_tokens when computing effective limit to avoid position overflow
        max_new_tokens = 4096  # default
        if self._engine is not None and self._engine._sampling_params is not None:
            max_new_tokens = getattr(self._engine._sampling_params, "max_tokens", 4096) or 4096
        effective_limit = max(1, self._current_max_model_len - max_new_tokens)
        near_limit = int(effective_limit * 0.92)
        # Use exact token counting when within 80% of effective limit to avoid underestimation
        exact_if_large = int(effective_limit * 0.80)

        max_prompt = 0
        for ctx in contexts:
            n = self._counter.estimate_prompt_tokens(ctx, exact_if_large=exact_if_large)
            if n > max_prompt:
                max_prompt = n
            if max_prompt >= near_limit:
                break

        if max_prompt >= near_limit:
            # Target should account for prompt + max_new_tokens
            target = max(self._current_max_model_len * 2, int((max_prompt + max_new_tokens) * 1.1))
            target = int((target + 255) // 256 * 256)  # Round to 256
            if target <= self._upgrade_cap:
                self._upgrade_to(target)

    def _maybe_strip_thinking_for_contexts(
        self, contexts: list[list[dict[str, str]]]
    ) -> tuple[list[list[dict[str, str]]], bool]:
        """
        Check if thinking content should be stripped and strip if needed.
        
        Thinking is stripped when:
        1. It hasn't been stripped yet, AND
        2. Any context exceeds THINKING_STRIP_THRESHOLD of the context limit
        
        Returns (contexts, was_stripped_this_call).
        """
        if not contexts:
            return contexts, False
        
        # If already stripped, just apply stripping to any new content
        if self._thinking_stripped:
            stripped, num_changed = strip_thinking_from_contexts(contexts)
            return stripped, False  # Not a new decision, just maintenance
        
        # Check if any context exceeds the threshold
        max_prompt_tokens = max(1, self._current_max_model_len - 128)
        strip_threshold = int(max_prompt_tokens * THINKING_STRIP_THRESHOLD)
        exact_if_large = int(strip_threshold * 0.75)
        
        needs_strip = False
        for ctx in contexts:
            n_est = self._counter.estimate_prompt_tokens(ctx, exact_if_large=exact_if_large)
            if n_est >= strip_threshold:
                needs_strip = True
                break
        
        if not needs_strip:
            return contexts, False
        
        # Strip thinking from all contexts
        stripped, num_changed = strip_thinking_from_contexts(contexts)
        if num_changed > 0:
            self.mark_thinking_stripped()
            print(
                f"[ctx] Stripped thinking from {num_changed}/{len(contexts)} contexts "
                f"(threshold: {strip_threshold} tokens).",
                file=sys.stderr,
            )
        return stripped, num_changed > 0

    def _truncate_contexts_to_fit(self, contexts: list[list[dict[str, str]]]) -> list[list[dict[str, str]]]:
        """Truncate contexts that exceed the current limit."""
        if not contexts:
            return contexts

        # First, try stripping thinking content if we're approaching the limit
        contexts, _ = self._maybe_strip_thinking_for_contexts(contexts)

        max_prompt_tokens = max(1, self._current_max_model_len - 128)
        near = int(max_prompt_tokens * 0.92)
        exact_if_large = int(near * 0.75)

        out: list[list[dict[str, str]]] = []
        changed = 0
        for ctx in contexts:
            n_est = self._counter.estimate_prompt_tokens(ctx, exact_if_large=exact_if_large)
            if n_est <= max_prompt_tokens:
                out.append(ctx)
                continue
            trimmed, did = truncate_chat_messages_to_fit(
                counter=self._counter,
                messages=ctx,
                max_prompt_tokens=max_prompt_tokens,
            )
            out.append(trimmed)
            if did:
                changed += 1

        if changed:
            print(f"[ctx] Truncated {changed}/{len(contexts)} prompts to fit max_model_len={self._current_max_model_len}.", file=sys.stderr)
        return out

    def generate_batch(
        self,
        contexts: list[list[dict[str, str]]],
        batch_size: int | None = None,
        *,
        sampling_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """Generate with thinking stripping and truncation (no context upgrades)."""
        self.initialize()
        if not contexts:
            return []

        # Strip thinking and truncate to fit - no context length upgrades
        contexts = self._truncate_contexts_to_fit(contexts)

        assert self._engine is not None
        try:
            return self._engine.generate_batch(contexts, batch_size=batch_size, sampling_kwargs=sampling_kwargs)
        except Exception as e:
            if is_prompt_too_long(e):
                # Try truncating again (more aggressively if needed)
                truncated = self._truncate_contexts_to_fit(contexts)
                if truncated != contexts:
                    return self._engine.generate_batch(truncated, batch_size=batch_size, sampling_kwargs=sampling_kwargs)
            raise


# Type alias for any inference engine
InferenceEngine = VLLMInferenceEngine | AdaptiveVLLMEngine


def create_inference_engine(
    *,
    model_name: str,
    gpus: str = "0",
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    allow_long_max_model_len: bool = False,
    enable_yarn: bool = False,
    kv_cache_dtype: str = "auto",
    enforce_eager: bool = False,
) -> InferenceEngine:
    """
    Factory to create an inference engine.
    If max_model_len is None, uses adaptive context.
    """
    require_vllm_deps()

    # Default to fp8 KV cache when the GPU supports it (2x KV memory savings on Ada/Hopper),
    # but keep a clean interface (no extra flags required).
    kv_req = (kv_cache_dtype or "").strip() or "auto"
    if kv_req.lower() == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                if major > 8 or (major == 8 and minor >= 9):
                    kv_req = "fp8_e5m2"
        except Exception:
            pass

    if max_model_len is None:
        # Use adaptive engine (defaults to tensor parallel across all visible GPUs).
        return AdaptiveVLLMEngine(
            model_name=model_name,
            gpus=gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            start_max_model_len=32768,
            max_model_len_cap=131072,
            allow_long_max_model_len=allow_long_max_model_len,
            force_yarn=enable_yarn,
            kv_cache_dtype=kv_req,
            enforce_eager=enforce_eager,
        )
    else:
        # Use fixed context engine
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        gpu_count = len(gpus.split(","))

        hf_overrides = None
        if enable_yarn or allow_long_max_model_len:
            hf_overrides = build_rope_scaling_overrides(
                model_name,
                target_len=max_model_len,
                force_yarn=enable_yarn,
            )
            if hf_overrides:
                os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

        engine = VLLMInferenceEngine(
            model_name,
            tensor_parallel_size=gpu_count,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            hf_overrides=hf_overrides,
            kv_cache_dtype=kv_req,
            enforce_eager=enforce_eager,
        )
        engine.initialize()
        return engine
