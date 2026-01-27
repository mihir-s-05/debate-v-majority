"""Shared utilities for the debug_majority_debate package."""
from __future__ import annotations

import os
import re
import signal
import hashlib
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Backend

THINKING_STRIP_THRESHOLD = 0.85  # Strip thinking when context reaches this fraction


# =============================================================================
# Thinking content stripping (for reasoning models like Qwen3, DeepSeek-R1)
# =============================================================================

# Pattern matches <think>...</think> blocks (case-insensitive, handles newlines)
_THINKING_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking_content(text: str) -> str:
    """Remove <think>...</think> blocks."""
    if not text:
        return text
    return _THINKING_PATTERN.sub("", text).strip()


def has_thinking_content(text: str) -> bool:
    """Check if text contains <think>...</think> blocks."""
    if not text:
        return False
    return bool(_THINKING_PATTERN.search(text))


def strip_thinking_from_message(msg: dict[str, str]) -> dict[str, str]:
    """Strip thinking content from a single assistant message."""
    if msg.get("role") != "assistant":
        return msg
    content = msg.get("content", "")
    stripped = strip_thinking_content(content)
    if stripped == content:
        return msg
    return {**msg, "content": stripped}


def strip_thinking_from_messages(
    messages: list[dict[str, str]]
) -> tuple[list[dict[str, str]], bool]:
    """Strip thinking content from all assistant messages."""
    result: list[dict[str, str]] = []
    changed = False
    for msg in messages:
        stripped_msg = strip_thinking_from_message(msg)
        if stripped_msg is not msg:
            changed = True
        result.append(stripped_msg)
    return result, changed


def strip_thinking_from_contexts(
    contexts: list[list[dict[str, str]]],
    *,
    parallel_threshold: int = 32,
) -> tuple[list[list[dict[str, str]]], int]:
    """Strip thinking content from all contexts."""
    if not contexts:
        return [], 0

    result: list[list[dict[str, str]]] = []
    num_changed = 0
    for ctx in contexts:
        stripped_ctx, changed = strip_thinking_from_messages(ctx)
        if changed:
            num_changed += 1
        result.append(stripped_ctx)
    return result, num_changed


# =============================================================================
# Math parsing
# =============================================================================


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{...} or \\fbox{...} from a string."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str | None:
    """
    Remove a \\boxed{...} or \\fbox{...} wrapper and return inner content.
    Tolerates whitespace between the command and the opening brace (e.g. \\boxed {A}).
    """
    if not isinstance(s, str):
        return None
    if not s.endswith("}"):
        return None
    m = re.match(r"^\\(?:boxed|fbox)\s*{", s)
    if not m:
        return None
    return s[m.end() : -1]


def parse_math(text: str) -> str | None:
    """Parse math answer from \\boxed{} format."""
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    return remove_boxed(boxed)


# =============================================================================
# String normalization
# =============================================================================


def normalize_numeric_string(s: str | None) -> str | None:
    """Normalize a numeric string (remove commas, leading zeros, etc.)."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    # Canonicalize integer forms with leading zeros.
    m = re.fullmatch(r"-?\d+", s)
    if m:
        sign = ""
        if s.startswith("-"):
            sign = "-"
            s = s[1:]
        s = s.lstrip("0") or "0"
        return sign + s
    return s


def normalize_freeform_string(s: str | None) -> str | None:
    """Normalize a freeform string for comparison."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = s.strip("$")
    s = " ".join(s.split())
    s = s.strip().strip(".")
    return s.lower()


# =============================================================================
# Majority voting
# =============================================================================


def most_frequent_answer(answers: list[str | None] | None) -> str | None:
    """Return the most frequent answer, or None if no clear majority."""
    if not answers:
        return None
    # Ignore unparsable outputs when voting. Treating `None` as a normal category
    # makes voting unnecessarily brittle (e.g., [None, None, "C"] would otherwise
    # return None even though the model produced a valid answer).
    filtered = [a for a in answers if a is not None]
    if not filtered:
        return None
    counts = Counter(filtered)
    top_count = max(counts.values(), default=0)
    if top_count <= 1:
        return None
    top = [ans for ans, count in counts.items() if count == top_count]
    return top[0] if len(top) == 1 else None


# =============================================================================
# Conversation/transcript utilities
# =============================================================================


def render_agent_transcript(agent_conv: list[dict[str, str]], include_system: bool = False) -> str:
    """Render a single agent's chat transcript into a string."""
    parts = []
    for msg in agent_conv:
        role = (msg.get("role") or "").strip()
        if role == "system" and not include_system:
            continue
        content = msg.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(parts)


def assistant_message_indexes(agent_conv: list[dict[str, str]]) -> list[int]:
    """Find all assistant message indexes in a conversation."""
    idxs: list[int] = []
    for i, msg in enumerate(agent_conv):
        role = (msg.get("role") or "").strip()
        if role == "assistant":
            idxs.append(i)
    return idxs


def slice_agent_conv_round_range(
    agent_conv: list[dict[str, str]], *, start_round: int, end_round: int
) -> list[dict[str, str]]:
    """
    Slice an agent conversation to include only assistant rounds [start_round, end_round] (1-indexed).
    Retains initial pre-round prefix (system/user question).
    """
    if not agent_conv:
        return []
    if start_round <= 0 or end_round <= 0:
        return agent_conv[:]
    if start_round > end_round:
        return agent_conv[:]

    assistant_idxs = assistant_message_indexes(agent_conv)
    if not assistant_idxs:
        return agent_conv[:]

    n_rounds = len(assistant_idxs)
    start_round = max(1, min(int(start_round), n_rounds))
    end_round = max(1, min(int(end_round), n_rounds))
    if start_round > end_round:
        return agent_conv[:]

    prefix_end = assistant_idxs[0]
    start_assistant = assistant_idxs[start_round - 1]
    end_assistant = assistant_idxs[end_round - 1]

    start = max(prefix_end, start_assistant - 1)
    end = end_assistant + 1
    return agent_conv[:prefix_end] + agent_conv[start:end]


def render_agent_assistant_rounds(
    agent_conv: list[dict[str, str]], *, start_round: int, end_round: int
) -> str:
    """Render only assistant messages for rounds [start_round, end_round] (1-indexed)."""
    assistant_idxs = assistant_message_indexes(agent_conv)
    if not assistant_idxs:
        return ""
    n_rounds = len(assistant_idxs)
    start_round = max(1, min(int(start_round), n_rounds))
    end_round = max(1, min(int(end_round), n_rounds))
    if start_round > end_round:
        return ""

    parts: list[str] = []
    for round_num in range(start_round, end_round + 1):
        msg_idx = assistant_idxs[round_num - 1]
        content = agent_conv[msg_idx].get("content", "")
        parts.append(f"ROUND {round_num}:\n{content}")
    return "\n\n".join(parts)


# =============================================================================
# Error detection
# =============================================================================


def exception_chain_contains(err: BaseException, needles: tuple[str, ...], max_depth: int = 5) -> bool:
    """Check if any exception in the chain contains any of the needle strings."""
    cur: BaseException | None = err
    for _ in range(max_depth):
        if cur is None:
            break
        msg = str(cur).lower()
        if any(n in msg for n in needles):
            return True
        cur = cur.__cause__
    return False


def is_cuda_oom(err: BaseException) -> bool:
    """Check if an exception is a CUDA OOM error."""
    try:
        import torch
        if isinstance(err, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    msg = str(err).lower()
    return ("out of memory" in msg) and ("cuda" in msg or "cublas" in msg)


def is_cuda_device_side_assert(err: BaseException) -> bool:
    """Detect CUDA device-side asserts (unrecoverable)."""
    needles = (
        "device-side assert",
        "device side assert",
        "cudaerrorassert",
        "illegal memory access",
        "an illegal memory access was encountered",
        "unspecified launch failure",
        "misaligned address",
        "warp illegal address",
    )
    return exception_chain_contains(err, needles)


def is_vllm_engine_dead(err: BaseException) -> bool:
    """Check if vLLM engine has died."""
    name = err.__class__.__name__
    if name == "EngineDeadError":
        return True
    mod = getattr(err.__class__, "__module__", "") or ""
    return ("vllm" in mod) and ("EngineDeadError" in name)


def is_vllm_oom_like(err: BaseException) -> bool:
    """Check if exception is a vLLM OOM or engine death."""
    if is_cuda_oom(err):
        return True
    if is_vllm_engine_dead(err):
        return True
    # Check chained exceptions
    cur: BaseException | None = err
    for _ in range(5):
        if cur is None:
            break
        if is_cuda_oom(cur):
            return True
        cur = cur.__cause__
    return False


def is_prompt_too_long(err: BaseException) -> bool:
    """Check if error is due to prompt exceeding context length."""
    msg = str(err).lower()
    return (
        ("longer than the maximum model length" in msg)
        or ("maximum model length" in msg and "prompt" in msg)
        or ("context length" in msg and "maximum" in msg)
        or ("exceeds the context window" in msg)
    )


def is_flash_attn_import_error(err: BaseException) -> bool:
    """Check if an exception is a FlashAttention import error."""
    msg = str(err)
    return (
        isinstance(err, ImportError)
        and ("flash_attn" in msg or "flash-attn" in msg)
        and ("undefined symbol" in msg or "flash_attn_2_cuda" in msg)
    )


def extract_prompt_length_tokens(err: BaseException) -> int | None:
    """Extract prompt length from error message."""
    msg = str(err)
    patterns = [
        r"\(length\s+(\d+)\)",
        r"prompt\s*\(length\s*(\d+)\)",
        r"prompt\s+length\s+(\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, msg, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


# =============================================================================
# Process management
# =============================================================================


def kill_process_tree(pid: int) -> None:
    """Best-effort kill of a process and all its children (Linux)."""
    try:
        children: list[int] = []
        try:
            with open(f"/proc/{pid}/task/{pid}/children", "r") as f:
                children = [int(c) for c in f.read().split()]
        except (FileNotFoundError, ProcessLookupError, ValueError):
            pass

        for child_pid in children:
            kill_process_tree(child_pid)

        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    except Exception:
        pass


# =============================================================================
# Token counting
# =============================================================================


def _hash_obj_for_cache(hasher: "hashlib._Hash", obj: Any) -> None:
    """
    Update `hasher` with a stable representation of `obj`.

    This avoids keeping large prompt strings alive in cache keys while still
    incorporating all message fields that may affect `apply_chat_template()`.
    """

    def _u(b: bytes) -> None:
        hasher.update(b)
        hasher.update(b"\0")

    def _u_int(n: int) -> None:
        # Use 64-bit packing when possible; fall back to decimal bytes.
        try:
            hasher.update(struct.pack("!q", int(n)))
            hasher.update(b"\0")
        except Exception:
            _u(str(n).encode("utf-8"))

    if obj is None:
        _u(b"none")
        return
    if obj is True:
        _u(b"true")
        return
    if obj is False:
        _u(b"false")
        return

    if isinstance(obj, int):
        _u(b"int")
        _u_int(obj)
        return
    if isinstance(obj, float):
        _u(b"float")
        _u(repr(obj).encode("utf-8"))
        return
    if isinstance(obj, str):
        _u(b"str")
        # Avoid keeping large strings alive in the cache key, and avoid
        # re-encoding big prompts on every cache lookup. Python string hashes
        # are cached per object, so this is typically O(1) after first use.
        _u_int(len(obj))
        _u_int(hash(obj))
        # Add small, stable-ish extra signal to reduce hash-collision risk.
        prefix = obj[:64]
        suffix = obj[-64:]
        _u_int(hash(prefix))
        _u_int(hash(suffix))
        return
    if isinstance(obj, (bytes, bytearray, memoryview)):
        b = bytes(obj)
        _u(b"bytes")
        _u_int(len(b))
        _u_int(hash(b))
        return

    if isinstance(obj, (list, tuple)):
        _u(b"list" if isinstance(obj, list) else b"tuple")
        _u_int(len(obj))
        for item in obj:
            _hash_obj_for_cache(hasher, item)
        return

    if isinstance(obj, dict):
        _u(b"dict")
        _u_int(len(obj))
        # Sort keys for determinism (messages are expected to be dict-like).
        items = list(obj.items())
        items.sort(key=lambda kv: (repr(kv[0]), type(kv[0]).__qualname__))
        for k, v in items:
            _hash_obj_for_cache(hasher, k)
            _hash_obj_for_cache(hasher, v)
        return

    # Fallback: incorporate type and a bounded repr to keep hashing safe.
    _u(b"obj")
    _u(f"{type(obj).__module__}.{type(obj).__qualname__}".encode("utf-8"))
    try:
        r = repr(obj)
    except Exception:
        r = f"<unreprable {type(obj).__qualname__}>"
    if len(r) > 2048:
        r = r[:2048] + "...(truncated)"
    _u(r.encode("utf-8", errors="surrogatepass"))


def _messages_cache_key(messages: list[dict[str, Any]]) -> tuple[int, int, str]:
    """
    Create a compact, stable cache key for chat messages.

    Returns a tuple containing a version number, message count, and a digest.
    """
    hasher = hashlib.blake2b(digest_size=32)
    hasher.update(b"messages_cache_key_v2\0")
    _hash_obj_for_cache(hasher, messages)
    return (2, len(messages), hasher.hexdigest())


class PromptTokenCounter:
    """Token counter for chat messages with lazy tokenizer loading and caching."""

    # Class-level cache shared across instances (keyed by model_name + messages)
    _token_cache: dict[tuple, int] = {}
    _cache_max_size: int = 4096

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True, use_fast=True
        )
        return self._tokenizer

    def _cache_key(self, messages: list[dict[str, Any]]) -> tuple:
        """Create a full cache key including model name."""
        # Keep cache keys small and include template options that affect encoding.
        add_generation_prompt = True
        return (self._model_name, add_generation_prompt, _messages_cache_key(messages))

    def _get_cached(self, key: tuple) -> int | None:
        """Get cached token count if available."""
        return self._token_cache.get(key)

    def _set_cached(self, key: tuple, count: int) -> None:
        """Cache token count with simple LRU-style eviction."""
        # Simple eviction: clear half the cache when full
        if len(self._token_cache) >= self._cache_max_size:
            keys_to_remove = list(self._token_cache.keys())[: self._cache_max_size // 2]
            for k in keys_to_remove:
                self._token_cache.pop(k, None)
        self._token_cache[key] = count

    def count_chat_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Exact token count for chat messages (cached)."""
        cache_key = self._cache_key(messages)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        tok = self._get_tokenizer()
        if hasattr(tok, "apply_chat_template"):
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            try:
                count = int(len(ids))
            except TypeError:
                count = int(ids.shape[-1])
        else:
            text = "\n".join((str(m.get("role", "")) + ": " + str(m.get("content", ""))) for m in messages)
            count = int(len(tok.encode(text)))

        self._set_cached(cache_key, count)
        return count

    def estimate_prompt_tokens(self, messages: list[dict[str, Any]], *, exact_if_large: int) -> int:
        """
        Estimated token count - uses heuristic for small prompts, tokenizer for large.
        
        When approx >= exact_if_large, switches to exact tokenization to avoid
        underestimation near context limits (which can cause CUDA device asserts).
        """
        text = "\n".join(
            (str(m.get("role", "")) + ": " + str(m.get("content", ""))) for m in messages
        )
        approx = max(1, len(text) // 4)
        if approx < exact_if_large:
            return approx

        # Check cache first before expensive tokenization
        cache_key = self._cache_key(messages)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Use exact token counting when near the limit to avoid underestimation
        # that could lead to position overflow and CUDA device-side asserts
        try:
            tok = self._get_tokenizer()
        except Exception:
            # If tokenizer fails, add a safety margin to the approximation
            return int(approx * 1.2)
        try:
            if hasattr(tok, "apply_chat_template"):
                ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                count = int(len(ids))
                self._set_cached(cache_key, count)
                return count
        except Exception:
            pass
        try:
            count = int(len(tok.encode(text)))
            self._set_cached(cache_key, count)
            return count
        except Exception:
            # If all exact methods fail, add safety margin
            return int(approx * 1.2)


def truncate_chat_messages_to_fit(
    *,
    counter: PromptTokenCounter,
    messages: list[dict[str, str]],
    max_prompt_tokens: int,
) -> tuple[list[dict[str, str]], bool]:
    """
    Truncate chat history to fit within token budget.
    Returns (truncated_messages, was_truncated).
    """
    if max_prompt_tokens <= 0 or not messages:
        return messages, False

    def _approx_tokens(msgs: list[dict[str, str]]) -> int:
        return counter.estimate_prompt_tokens(msgs, exact_if_large=max_prompt_tokens + 1)

    def _fits(candidate: list[dict[str, str]]) -> bool:
        try:
            return counter.count_chat_tokens(candidate) <= max_prompt_tokens
        except Exception:
            return _approx_tokens(candidate) <= max_prompt_tokens

    try:
        if _fits(messages):
            return messages, False
    except Exception:
        pass

    msgs = [dict(m) for m in messages]

    # Preserve system prefix
    sys_prefix: list[dict[str, str]] = []
    i = 0
    while i < len(msgs) and msgs[i].get("role") == "system":
        sys_prefix.append(msgs[i])
        i += 1
    tail = msgs[i:]

    # Drop oldest turns until fits
    while len(tail) > 1 and not _fits(sys_prefix + tail):
        tail = tail[1:]

    if _fits(sys_prefix + tail):
        return sys_prefix + tail, True

    if not tail:
        return sys_prefix, True

    # Truncate remaining message content
    msg = dict(tail[0])
    content = msg.get("content") or ""
    if not content:
        return sys_prefix + [msg], True

    try:
        tok = counter._get_tokenizer()
    except Exception:
        msg["content"] = content[-max(1, min(len(content), max_prompt_tokens * 4)) :]
        return sys_prefix + [msg], True

    try:
        content_ids = tok.encode(content)
    except Exception:
        msg["content"] = content[-max(1, min(len(content), max_prompt_tokens * 4)) :]
        return sys_prefix + [msg], True

    # Binary search for max content that fits
    lo, hi = 1, len(content_ids)
    best: str | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            truncated = tok.decode(content_ids[-mid:], skip_special_tokens=True)
        except Exception:
            truncated = content
        candidate = sys_prefix + [dict(msg, content=truncated)]
        if _fits(candidate):
            best = truncated
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        return sys_prefix, True

    msg["content"] = best
    return sys_prefix + [msg], True


# =============================================================================
# Judge context utilities
# =============================================================================


def round_block_start(round_num: int, block_size: int) -> int:
    """Calculate the start round for a block-based judge window."""
    if block_size <= 0:
        return 1
    round_num = max(1, int(round_num))
    block_size = int(block_size)
    return ((round_num - 1) // block_size) * block_size + 1


@dataclass
class PrevJudgeInfo:
    """Information about a previous judge decision."""
    start_round: int
    end_round: int
    parsed_answer: str
    raw_output: str


def format_prev_judge_full(prev: PrevJudgeInfo) -> str:
    """Format previous judge output (full version)."""
    return (
        f"Rounds {prev.start_round}-{prev.end_round} judge answer: {prev.parsed_answer}\n"
        f"Judge transcript:\n{prev.raw_output}"
    )


def format_prev_judge_short(prev: PrevJudgeInfo) -> str:
    """Format previous judge output (short version)."""
    return f"Rounds {prev.start_round}-{prev.end_round} judge answer: {prev.parsed_answer}"
