from debate_v_majority.datasets import aime25, gpqa, gsm8k


def test_gsm8k_parse_answer_prefers_boxed_numeric():
    text = "analysis then \\boxed{1,234}"
    assert gsm8k.parse_answer(text, {}) == "1234"


def test_aime25_parse_answer_uses_tail_fallback_for_out_of_range_boxed_value():
    text = "candidate \\boxed{1524}. final answer is 524"
    assert aime25.parse_answer(text, {}) == "524"


def test_gpqa_parse_answer_extracts_boxed_choice():
    text = "reasoning...\n\\boxed{c}"
    assert gpqa.parse_answer(text, {}) == "C"


def test_gpqa_parse_question_answer_supports_numbered_incorrect_keys():
    sample = {
        "Question": "What is correct?",
        "Correct Answer": "blue",
        "Incorrect Answer 1": "red",
        "Incorrect Answer 2": "green",
        "Incorrect Answer 3": "yellow",
    }
    prompt, gt, raw = gpqa.parse_question_answer(sample)
    assert "Choices:" in prompt
    assert gt in {"A", "B", "C", "D"}
    assert raw is sample
