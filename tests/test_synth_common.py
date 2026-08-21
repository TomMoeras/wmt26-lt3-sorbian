# Ported from the working repo's tests/data/synth/test_common.py (import path changed only;
# render_sc_row/render_gc_row/render_mr_row now go through lt3wmt26.generate's prompt builders
# instead of the organizer-harness src.eval.official.prompts -- see synth/common.py docstring).
from training.generators.synth import common


def test_length_filter_keeps_3_to_40_tokens():
    assert common.length_ok("a b c", lo=3, hi=40)
    assert not common.length_ok("a b", lo=3, hi=40)
    assert not common.length_ok(" ".join(["w"] * 41), lo=3, hi=40)


def test_dedup_normalizes_whitespace_and_case_insensitive_exact():
    rows = ["Dom je  wulki", "dom je wulki", "Něšto druhe"]
    out = common.dedup(rows)
    assert len(out) == 2


def test_dev_exclusion_drops_dev_sentences():
    pool = ["clean one here", "a dev sentence text", "another clean line"]
    dev = {"a dev sentence text"}
    kept = common.exclude_dev(pool, dev)
    assert "a dev sentence text" not in kept and len(kept) == 2


def test_render_sc_training_row_official_format():
    row = common.render_sc_row(input_sentence="To je sentenca.",
                               incorrect_word="sentenca", correct_word="sadnica",
                               exemplars=[])
    msgs = row["messages"]
    assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "assistant"
    assert "<wrong>" in msgs[0]["content"] and "<sentence> To je sentenca. </sentence>" in msgs[0]["content"]
    assert msgs[-1]["content"] == "<wrong> sentenca </wrong> <corrected> sadnica </corrected>"


def test_render_sc_clean_row_uses_CORRECT_sentinel():
    row = common.render_sc_row("Cista sadnica.", "CORRECT", "CORRECT", exemplars=[])
    assert row["messages"][-1]["content"] == "<wrong> CORRECT </wrong> <corrected> CORRECT </corrected>"


def test_seeded_sample_is_deterministic():
    a = common.seeded_sample(list(range(100)), 10, seed=42)
    b = common.seeded_sample(list(range(100)), 10, seed=42)
    assert a == b and len(a) == 10


def test_render_gc_row_assistant_format():
    row = common.render_gc_row(input_sentence="Wona je dobrej.",
                               incorrect_word="dobrej", correct_word="dobra",
                               exemplars=[])
    msgs = row["messages"]
    assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == "<wrong> dobrej </wrong> <corrected> dobra </corrected>"
    # gc kind: instruction mentions "grammatically"
    assert "grammatic" in msgs[0]["content"].lower()


def test_render_qa_row_format():
    row = common.render_qa_row(
        context="Serbski dom steji w Budyšinje.",
        question="Hdźe steji dom?",
        possible_answers={"1": "Budyšinje", "2": "Drježdźanach"},
        correct_answer_num=1,
    )
    msgs = row["messages"]
    assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "assistant"
    assert "Budyšinje" in msgs[0]["content"]
    assert "1:" in msgs[0]["content"]
    assert msgs[-1]["content"] == "1"


def test_render_mr_row_format():
    row = common.render_mr_row(
        question="Što je 2+2?",
        answer="4",
        steps="Přičitaj 2 a 2.",
        exemplars=[],
    )
    msgs = row["messages"]
    assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "assistant"
    assert "<answer> 4 </answer>" in msgs[-1]["content"]
