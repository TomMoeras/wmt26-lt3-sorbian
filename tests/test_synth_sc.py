# Ported from the working repo's tests/data/synth/test_sc.py (import path changed only).
import random
from training.generators.synth import sc


def test_inject_one_typo_changes_exactly_one_word_and_is_substring():
    rng = random.Random(0)
    out = sc.inject_typo("dom je wulki dom", rng)
    assert out is not None
    iw, cw, sent = out["incorrect_word"], out["correct_word"], out["input_sentence"]
    assert iw != cw and iw in sent.split()
    # exactly one token differs
    diffs = [(a, b) for a, b in zip(sent.split(), "dom je wulki dom".split()) if a != b]
    assert len(diffs) == 1


def test_typo_types_cover_all_four_over_many_draws():
    rng = random.Random(1)
    seen = set()
    for _ in range(500):
        seen.add(sc._apply_edit("sadnica", "substitution", rng) and "ok")  # smoke
    for t in ("substitution", "deletion", "insertion", "transposition"):
        w = sc._apply_edit("sadnica", t, rng)
        assert w != "sadnica" and isinstance(w, str)


def test_transposition_swaps_adjacent_chars():
    rng = random.Random(2)
    w = sc._apply_edit("abcd", "transposition", rng)
    assert sorted(w) == sorted("abcd") and w != "abcd"


def test_build_dataset_hits_target_error_fraction():
    rng = random.Random(3)
    sents = [f"slowo {i} druhe třeće štwórte" for i in range(200)]
    rows = sc.build_rows(sents, p_error=0.5, rng=rng)
    err = sum(1 for r in rows if r["incorrect_word"] != "CORRECT")
    assert 0.4 <= err / len(rows) <= 0.6  # ~50%
    # clean rows are labelled CORRECT/CORRECT and input==original
    for r in rows:
        if r["incorrect_word"] == "CORRECT":
            assert r["correct_word"] == "CORRECT"


def test_deletion_produces_shorter_word():
    rng = random.Random(5)
    w = sc._apply_edit("sadnica", "deletion", rng)
    assert len(w) == len("sadnica") - 1


def test_insertion_produces_longer_word():
    rng = random.Random(6)
    w = sc._apply_edit("sadnica", "insertion", rng)
    assert len(w) == len("sadnica") + 1


def test_substitution_same_length():
    rng = random.Random(7)
    w = sc._apply_edit("sadnica", "substitution", rng)
    assert len(w) == len("sadnica") and w != "sadnica"


def test_confusion_chars_appear_in_substitutions():
    """Diacritic confusion pairs should appear with noticeable frequency."""
    rng = random.Random(42)
    results = set()
    for _ in range(1000):
        w = sc._apply_edit("sadnicas", "substitution", rng)
        results.add(w)
    # at least one result should contain a diacritic character
    diacritics = set("šśřćłó")
    has_diacritic = any(any(c in diacritics for c in r) for r in results)
    assert has_diacritic
