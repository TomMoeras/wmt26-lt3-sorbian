# Ported from the working repo's tests/data/synth/test_mr.py (import path changed only).
"""Tests for MR synthesis pure helpers (no GPU)."""
from training.generators.synth import mr


# --------------------------------------------------------------------------- #
# parse_seed
# --------------------------------------------------------------------------- #

def test_parse_gsm8k_keeps_integer_answer_and_strips_calc_annotations():
    ex = {
        "question": "Natalia sold clips to 48 of her friends in April, and then she "
                    "sold half as many clips in May. How many clips did Natalia sell "
                    "altogether in April and May?",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
                  "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
                  "#### 72",
    }
    out = mr.parse_seed(ex, "gsm8k")
    assert out is not None
    assert out["answer"] == "72"
    assert out["question_en"].startswith("Natalia sold clips")
    # rationale present, calculator <<...>> annotations removed, #### line dropped.
    assert "<<" not in out["rationale_en"] and ">>" not in out["rationale_en"]
    assert "####" not in out["rationale_en"]
    assert "24 clips in May" in out["rationale_en"]


def test_parse_gsm8k_with_comma_grouped_integer():
    ex = {"question": "How much profit?", "answer": "stuff\n#### 70,000"}
    out = mr.parse_seed(ex, "gsm8k")
    assert out["answer"] == "70000"


def test_parse_math_keeps_boxed_latex_verbatim():
    ex = {
        "problem": "What is the range of the function $y = \\frac{x^2 + 3x + 2}{x+1}$?",
        "solution": "We can factor ... the range is "
                    "$y \\in \\boxed{(-\\infty, 1)\\cup(1, \\infty)}.$",
        "level": "Level 5",
        "type": "Algebra",
    }
    out = mr.parse_seed(ex, "math")
    assert out is not None
    assert out["answer"] == "(-\\infty, 1)\\cup(1, \\infty)"
    assert out["question_en"].startswith("What is the range")
    assert out["rationale_en"]


def test_parse_math_boxed_with_frac_preserved():
    ex = {"problem": "Find x.", "solution": "Thus $x=\\boxed{\\frac{\\pi}{3}}$."}
    out = mr.parse_seed(ex, "math")
    assert out["answer"] == "\\frac{\\pi}{3}"


def test_parse_returns_none_when_no_extractable_answer():
    assert mr.parse_seed({"question": "Q", "answer": "no marker here"}, "gsm8k") is None
    assert mr.parse_seed({"problem": "P", "solution": "no box here"}, "math") is None


# --------------------------------------------------------------------------- #
# answer_is_safe
# --------------------------------------------------------------------------- #

def test_answer_is_safe_accepts_integers_and_simple_latex():
    assert mr.answer_is_safe("72")
    assert mr.answer_is_safe("-3")
    assert mr.answer_is_safe("70000")
    assert mr.answer_is_safe("\\frac{1}{2}")
    assert mr.answer_is_safe("\\frac{\\pi}{3}")
    assert mr.answer_is_safe("(-\\infty, 1)\\cup(1, \\infty)")
    assert mr.answer_is_safe("[2,3)")
    assert mr.answer_is_safe("3/4")


def test_answer_is_safe_rejects_prose_and_empty():
    assert not mr.answer_is_safe("")
    assert not mr.answer_is_safe("   ")
    assert not mr.answer_is_safe("the answer is forty two and a bit")
    # long free text (a sentence) is not a clean math answer
    assert not mr.answer_is_safe("We conclude that the value must be positive everywhere")


# --------------------------------------------------------------------------- #
# render_mr_training
# --------------------------------------------------------------------------- #

def test_render_mr_training_ends_exactly_in_answer_tag():
    row = mr.render_mr_training(
        question_sorbian="Kelko jejkow?",
        steps_sorbian="Wona ma 16 jejkow. 16 - 3 - 4 = 9. 9 * 2 = 18.",
        answer="18",
    )
    msgs = row["messages"]
    assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "assistant"
    assert "Kelko jejkow?" in msgs[0]["content"]
    assistant = msgs[-1]["content"]
    assert assistant.endswith("<answer> 18 </answer>")
    assert "16 - 3 - 4 = 9" in assistant


def test_render_mr_training_preserves_latex_answer_verbatim():
    row = mr.render_mr_training(
        question_sorbian="Namakaj wobwod.",
        steps_sorbian="Faktorizuj ...",
        answer="\\frac{\\pi}{3}",
    )
    assistant = row["messages"][-1]["content"]
    assert assistant.endswith("<answer> \\frac{\\pi}{3} </answer>")


def test_render_mr_training_empty_steps_still_valid():
    row = mr.render_mr_training("Q?", "", "5")
    assistant = row["messages"][-1]["content"]
    assert assistant.strip().endswith("<answer> 5 </answer>")
    assert "<answer>" in assistant
