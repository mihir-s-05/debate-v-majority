from debate_v_majority.shared import (
    most_frequent_answer,
    normalize_freeform_string,
    normalize_numeric_string,
    parse_math,
    strip_thinking_content,
)


def test_strip_thinking_content_removes_closed_and_unclosed_blocks():
    text = "prefix <think>hidden</think> mid <think>tail"
    assert strip_thinking_content(text) == "prefix  mid"


def test_parse_math_extracts_last_boxed_value():
    text = "work \\boxed{3} more work \\boxed{42}"
    assert parse_math(text) == "42"


def test_normalize_numeric_string_handles_commas_and_leading_zeroes():
    assert normalize_numeric_string("001,234") == "1234"
    assert normalize_numeric_string("-0007") == "-7"


def test_normalize_freeform_string_canonicalizes_case_and_spacing():
    assert normalize_freeform_string("  $Hello   World.$ ") == "hello world"


def test_most_frequent_answer_ignores_none_and_rejects_ties():
    assert most_frequent_answer([None, "C", "C"]) == "C"
    assert most_frequent_answer(["A", "B", "A", "B"]) is None
