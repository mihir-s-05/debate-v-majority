"""
AIME25 dataset handling: prompts, loading, parsing, and answer evaluation.
"""
from __future__ import annotations

import re
from typing import Any

from .shared import parse_math, normalize_numeric_string


# =============================================================================
# Prompts
# =============================================================================

AGENT_PROMPT = {
    "question": "Solve the problem carefully and give the final answer in the form \\boxed{{answer}}.\nProblem: {}",
    "debate": [
        "These are the solutions to the problem from other agents:",
        "\n\nCarefully evaluate each agent's reasoning and final answer. Consider their approaches critically, identifying any potential errors or superior logic. After this deep reflection, and referring to your historical answers, provide your updated solution and final answer. Put your answer in the form \\boxed{{answer}} at the end of your response.",
    ],
}

JUDGE_PROMPT = {
    "user_prompt_suffix": "\n\nRead all the agents' responses and decide which one is the correct one. Put the answer in the form \\boxed{{answer}} at the end of your response."
}


# =============================================================================
# Dataset loading
# =============================================================================


def parse_question_answer(sample: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """
    Parse an AIME25 sample into (question_prompt, ground_truth_answer, raw_task).
    """
    question_raw = sample.get("problem") or sample.get("question")
    if question_raw is None:
        raise KeyError("AIME25 sample missing 'problem' field")
    answer_raw = sample.get("answer")
    if answer_raw is None:
        raise KeyError("AIME25 sample missing 'answer' field")
    raw_task = sample
    question = AGENT_PROMPT["question"].format(question_raw)
    gt = normalize_numeric_string(str(answer_raw))
    if gt is None:
        raise ValueError("AIME25 ground-truth answer parsed as None")
    return question, gt, raw_task


# =============================================================================
# Answer parsing
# =============================================================================


def parse_answer(text: str, task_info: dict[str, Any]) -> str | None:
    """
    Parse a model's response to extract the numeric answer.
    AIME answers are integers 0-999.
    """
    parsed = parse_math(text)
    if parsed is not None:
        norm = normalize_numeric_string(parsed)
        if norm is not None and re.fullmatch(r"-?\d+", norm):
            try:
                v = int(norm)
                if 0 <= v <= 999:
                    return norm
            except Exception:
                pass
        # If the boxed content isn't a valid AIME integer in [0,999], fall back to
        # tail heuristics rather than returning an out-of-range intermediate result.
    # If no boxed answer is present, avoid grabbing large intermediate values
    # (e.g. counts like 32768) by preferring 0-999 integers near the end.
    text = re.sub(r"(\d),(\d)", r"\1\2", str(text))
    tail = text[-4000:] if len(text) > 4000 else text
    # Avoid mis-reading decimals like 0.418 as the integer 0 or 418.
    int_token = r"(?<![\d.])-?\d{1,3}(?!\.\d)\b"

    # Prefer numbers near explicit "final/answer" cues.
    m = None
    for m in re.finditer(rf"(?i)\b(?:final\s+answer|answer|final)\b[^0-9]{{0,40}}({int_token})", tail):
        pass
    if m:
        try:
            v = int(m.group(1))
            if 0 <= v <= 999:
                return normalize_numeric_string(str(v))
        except Exception:
            pass

    # Last integer in [0,999] in the tail.
    ints = re.findall(int_token, tail)
    for s in reversed(ints):
        try:
            v = int(s)
        except Exception:
            continue
        if 0 <= v <= 999:
            return normalize_numeric_string(str(v))
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
    return int(normalize_numeric_string(str(answer)) == normalize_numeric_string(str(gt)))


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
