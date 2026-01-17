"""
GSM8K dataset handling: prompts, loading, parsing, and answer evaluation.
"""
from __future__ import annotations

import re
from typing import Any

from .shared import parse_math, normalize_numeric_string


# =============================================================================
# Prompts
# =============================================================================

AGENT_PROMPT = {
    "question": "Solve the math word problem step by step and give the final numeric answer in the form \\boxed{{answer}}.\nProblem: {}",
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
    Parse a GSM8K sample into (question_prompt, ground_truth_answer, raw_task).
    """
    question_raw = sample["question"]
    answer_raw = sample["answer"]
    # Remove commas between digits to handle thousand separators
    answer_raw_cleaned = re.sub(r"(\d),(\d)", r"\1\2", answer_raw)
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_raw_cleaned)
    if match:
        answer = match.group(1)
    else:
        fallback = re.findall(r"-?\d+(?:\.\d+)?", answer_raw_cleaned)
        answer = fallback[-1] if fallback else answer_raw
    raw_task = sample
    question = AGENT_PROMPT["question"].format(question_raw)
    return question, answer, raw_task


# =============================================================================
# Answer parsing
# =============================================================================


def parse_answer(text: str, task_info: dict[str, Any]) -> str | None:
    """
    Parse a model's response to extract the numeric answer.
    """
    parsed = parse_math(text)
    if parsed is not None:
        return parsed.replace(",", "")
    # Remove commas between digits to handle thousand separators
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else None


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
