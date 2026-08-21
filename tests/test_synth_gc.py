# Ported from the working repo's tests/data/synth/test_gc.py (import path changed only).
import random
from training.generators.synth import gc


def test_suffix_swap_changes_ending_keeps_stem():
    rng = random.Random(0)
    w = gc.perturb_word("žona", "hsb", rng)  # a-stem
    assert w is not None and w != "žona" and w.startswith("žon")  # stem retained


def test_perturb_returns_none_when_no_rule_applies():
    rng = random.Random(0)
    assert gc.perturb_word("xyz", "hsb", rng) is None  # no matching suffix


def test_build_rows_50_50_and_single_token_change():
    rng = random.Random(1)
    sents = ["žona ma wulku chěžu a dobreho muža tu"] * 100
    rows = gc.build_rows(sents, lang="hsb", p_error=0.5, rng=rng)
    err = [r for r in rows if r["incorrect_word"] != "CORRECT"]
    assert 0.35 <= len(err) / len(rows) <= 0.65
    for r in err:
        diffs = [(a, b) for a, b in zip(r["input_sentence"].split(), r["original_sentence"].split()) if a != b]
        assert len(diffs) == 1
        assert r["incorrect_word"] in r["input_sentence"].split()


def test_dsb_suffix_rules_apply():
    rng = random.Random(0)
    # dsb has ow->am rule; try word ending in 'ow'
    w = gc.perturb_word("bratrow", "dsb", rng)
    assert w is not None and w != "bratrow"


def test_perturb_word_returns_different_token():
    """perturb_word must always return a token different from the input."""
    rng = random.Random(42)
    for word in ["žona", "woda", "bratrow", "domow"]:
        for lang in ("hsb", "dsb"):
            result = gc.perturb_word(word, lang, rng)
            if result is not None:
                assert result != word


def test_clean_rows_have_CORRECT_sentinel():
    rng = random.Random(99)
    sents = ["woda tece dołoj po rěce"] * 50
    rows = gc.build_rows(sents, lang="hsb", p_error=0.0, rng=rng)
    for r in rows:
        assert r["incorrect_word"] == "CORRECT"
        assert r["correct_word"] == "CORRECT"
        assert r["input_sentence"] == r["original_sentence"]
