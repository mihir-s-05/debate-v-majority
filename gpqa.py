"""
GPQA dataset handling: prompts, loading, parsing, and answer evaluation.
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from .shared import parse_math, normalize_freeform_string


# =============================================================================
# Prompts
# =============================================================================

AGENT_PROMPT = {
    "question": "Answer the following multiple-choice question. Put your final choice in the form \\boxed{{A}} (one of A, B, C, D).\n\nQuestion:\n{question}\n\nChoices:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
    "debate": [
        "These are the solutions to the problem from other agents:",
        "\n\nCarefully evaluate each agent's reasoning and final answer. Consider their approaches critically, identifying any potential errors or superior logic. After this deep reflection, and referring to your historical answers, provide your updated solution and final answer. Put your final choice in the form \\boxed{{A}} (one of A, B, C, D) at the end of your response.",
    ],
}

JUDGE_PROMPT = {
    "user_prompt_suffix": "\n\nRead all the agents' responses and decide which one is the correct one. Put your final choice in the form \\boxed{{A}} (one of A, B, C, D) at the end of your response."
}


# =============================================================================
# Helpers
# =============================================================================


def _stable_shuffle(items: list[str], *, seed_text: str) -> list[str]:
    """Deterministic shuffle based on MD5 of seed text."""
    if len(items) <= 1:
        return items[:]
    h = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
    seed_int = int(h[:8], 16)
    rng = random.Random(seed_int)
    out = items[:]
    rng.shuffle(out)
    return out


def _first_nonempty_str(sample: dict[str, Any], keys: list[str]) -> str | None:
    """Get the first non-empty string value from a list of keys."""
    for k in keys:
        v = sample.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s
        else:
            try:
                s = str(v).strip()
                if s and s.lower() != "none":
                    return s
            except Exception:
                continue
    return None


def _extract_fields(sample: dict[str, Any]) -> tuple[str, str, list[str] | None]:
    """
    Extract (question, correct_answer, incorrect_answers) from various GPQA schemas.
    """
    question = _first_nonempty_str(
        sample,
        [
            "question",
            "problem",
            "query",
            "Question",
            "Extra Revised Question",
            "Pre-Revision Question",
        ],
    )
    if not question:
        raise KeyError(
            "GPQA sample missing question text (expected keys like question/Question/Extra Revised Question)"
        )

    correct = _first_nonempty_str(
        sample,
        [
            "correct_answer",
            "answer",
            "solution",
            "Correct Answer",
            "Extra Revised Correct Answer",
            "Pre-Revision Correct Answer",
        ],
    )
    if correct is None:
        raise KeyError(
            "GPQA sample missing correct answer (expected keys like correct_answer/Correct Answer)"
        )

    # Check for incorrect answers in various formats
    incorrect = (
        sample.get("incorrect_answers")
        or sample.get("wrong_answers")
        or sample.get("distractors")
    )
    if isinstance(incorrect, (list, tuple)):
        return str(question), str(correct), [str(x) for x in incorrect]

    # Try numbered keys
    numbered_keys_sets: list[list[str]] = [
        ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"],
        ["Extra Revised Incorrect Answer 1", "Extra Revised Incorrect Answer 2", "Extra Revised Incorrect Answer 3"],
        ["Pre-Revision Incorrect Answer 1", "Pre-Revision Incorrect Answer 2", "Pre-Revision Incorrect Answer 3"],
        ["incorrect_answer_1", "incorrect_answer_2", "incorrect_answer_3"],
    ]
    for keys in numbered_keys_sets:
        vals = [_first_nonempty_str(sample, [k]) for k in keys]
        if all(v is not None for v in vals):
            return str(question), str(correct), [str(v) for v in vals if v is not None]

    return str(question), str(correct), None


# =============================================================================
# Dataset loading
# =============================================================================


def parse_question_answer(sample: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """
    Parse a GPQA sample into (question_prompt, ground_truth_answer, raw_task).
    """
    question_raw, correct_text, incorrect_texts = _extract_fields(sample)
    raw_task = sample

    # If we have 3 distractors, format as MCQ
    if incorrect_texts is not None and len(incorrect_texts) >= 3:
        options = [correct_text] + incorrect_texts[:3]
        shuffled = _stable_shuffle(options, seed_text=question_raw)
        correct_idx = shuffled.index(correct_text)
        gt_letter = "ABCD"[correct_idx]
        prompt = AGENT_PROMPT["question"].format(
            question=question_raw,
            A=shuffled[0],
            B=shuffled[1],
            C=shuffled[2],
            D=shuffled[3],
        )
        return prompt, gt_letter, raw_task

    # Fallback: freeform exact string match
    prompt = "Solve the following question carefully and give your final answer in the form \\boxed{answer}.\nQuestion: {q}".format(
        q=question_raw
    )
    gt = normalize_freeform_string(correct_text)
    if gt is None:
        raise ValueError("GPQA ground-truth answer parsed as None")
    return prompt, gt, raw_task


# =============================================================================
# Answer parsing
# =============================================================================


def parse_answer(text: str, task_info: dict[str, Any]) -> str | None:
    """
    Parse a model's response to extract the answer (letter or freeform).
    """
    def _boxed_to_choice(s: str) -> str | None:
        """
        Map boxed content to A/B/C/D if it clearly encodes a choice.
        Handles variants like:
          - C
          - c
          - (C)
          - \\text{c}
          - \\mathrm{C}
          - \\mathbf{D}
        """
        s0 = str(s).strip()
        if not s0:
            return None
        # If we boxed another \\boxed{...}, unwrap once.
        m = re.fullmatch(r"(?is)\\(?:boxed|fbox)\s*{\s*(.*?)\s*}", s0)
        if m:
            s0 = m.group(1).strip()

        # Direct single-letter (optionally parenthesized)
        m = re.fullmatch(r"(?i)\(?\s*([ABCD])\s*\)?", s0)
        if m:
            return m.group(1).upper()
        # Common LaTeX wrappers
        m = re.fullmatch(r"(?is)\s*\\(?:text|mathrm|mathbf)\s*{\s*\(?\s*([ABCD])\s*\)?\s*}\s*", s0)
        if m:
            return m.group(1).upper()
        return None

    # Prefer explicit boxed answers.
    parsed = parse_math(text)
    if parsed is not None:
        boxed_choice = _boxed_to_choice(parsed)
        if boxed_choice is not None:
            return boxed_choice
        parsed_norm = normalize_freeform_string(parsed)
        if parsed_norm is not None and parsed_norm in ("a", "b", "c", "d"):
            return parsed_norm.upper()
        return parsed_norm

    # Fallback: try to extract a final choice from unboxed outputs.
    # We search cue-based patterns across the whole completion (less brittle when the model states an
    # answer and then continues), but still use a character tail for "last-line" heuristics.
    t = str(text or "")
    tail = t[-2400:] if len(t) > 2400 else t

    # Prefer explicit "answer/final" cues
    m = None
    cue_pat = re.compile(
        r"(?i)\b(?:final\s+answer|answer|final\s+choice|choice)\b[^A-D]{0,40}\(?\s*([ABCD])\s*\)?\b"
    )
    for m in cue_pat.finditer(t):
        pass
    if m:
        return m.group(1).upper()

    # "Option C" pattern
    m = None
    opt_pat = re.compile(r"(?i)\boption\s*([ABCD])\b")
    for m in opt_pat.finditer(tail):
        pass
    if m:
        return m.group(1).upper()

    # Markdown-ish endings like "**Final Answer**: C" or "Final: (D)"
    m = None
    final_line_pat = re.compile(r"(?i)\bfinal\b[^A-D]{0,40}\(?\s*([ABCD])\s*\)?")
    for m in final_line_pat.finditer(t):
        pass
    if m:
        return m.group(1).upper()

    # Or a standalone final line like "(C)" or "C".
    tail_lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    last = tail_lines[-1] if tail_lines else ""
    m = re.fullmatch(r"(?i)\(?\s*([ABCD])\s*\)?", last)
    if m:
        return m.group(1).upper()

    # Last resort: a standalone letter line in the last few lines of the tail.
    for ln in reversed(tail_lines[-12:]):
        m2 = re.fullmatch(r"(?i)\(?\s*([ABCD])\s*\)?", ln)
        if m2:
            return m2.group(1).upper()

    return None


# =============================================================================
# Answer evaluation
# =============================================================================


def check_answer_correctness(answer: Any, gt: Any) -> int:
    """
    Check if the parsed answer matches the ground truth.
    Returns 1 for correct, 0 for incorrect.
    """
    if answer is None:
        return 0
    # Prefer exact letter match (A/B/C/D) when available
    ans = str(answer).strip().upper()
    gt_s = str(gt).strip().upper()
    if gt_s in ("A", "B", "C", "D") or ans in ("A", "B", "C", "D"):
        return int(ans == gt_s)
    return int(normalize_freeform_string(str(answer)) == normalize_freeform_string(str(gt)))


# =============================================================================
# Debate message construction
# =============================================================================


def construct_debate_message(other_agent_answers: list[str]) -> dict[str, str]:
    """
    Construct a debate prompt showing other agents' answers.
    """
    prefix = AGENT_PROMPT["debate"][0]
    for answer in other_agent_answers:
        prefix += f"\n\n One agent solution: ```{answer}```"
    prefix += AGENT_PROMPT["debate"][1]
    return {"role": "user", "content": prefix}
