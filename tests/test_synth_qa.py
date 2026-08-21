# Ported from the working repo's tests/data/synth/test_qa.py (import path changed only).
import random
from training.generators.synth import qa


def test_cloze_masks_one_salient_token_and_answer_recoverable():
    rng = random.Random(0)
    item = qa.make_cloze("Serbski dom steji w Budyšinje za wiki.", vocab=["Lipsku", "Drježdźanach", "Choćebuzu"], rng=rng)
    assert item is not None
    # the answer string is one of the options and correct_answer_num indexes it (1-based)
    opts = item["possible_answers"]
    assert str(item["correct_answer_num"]) in opts
    assert item["answer_text"] == opts[str(item["correct_answer_num"])]


def test_distractors_count_and_no_duplicate_options():
    rng = random.Random(1)
    item = qa.make_cloze("Wona pije kofej kóžde ranje doma.", vocab=["wodu", "piwo", "mloko", "čaj"], rng=rng, n_options=4)
    vals = list(item["possible_answers"].values())
    assert len(vals) == len(set(vals)) == 4


def test_no_first_slot_bias_over_many():
    rng = random.Random(2)
    slots = []
    for i in range(200):
        it = qa.make_cloze(f"Token {i} steji how a tam pak", vocab=["aa", "bb", "cc", "dd"], rng=rng, n_options=4)
        if it:
            slots.append(it["correct_answer_num"])
    assert max(slots.count(s) for s in set(slots)) < 0.5 * len(slots)  # answer position spread


def test_possible_answers_are_1_based_string_keys():
    rng = random.Random(3)
    item = qa.make_cloze("Wulki swět je rjany a wšudźe.", vocab=["mały", "krasny", "płochy"], rng=rng, n_options=3)
    assert item is not None
    keys = list(item["possible_answers"].keys())
    assert sorted(keys) == [str(i) for i in range(1, len(keys) + 1)]


def test_correct_answer_num_is_1_based_int():
    rng = random.Random(4)
    item = qa.make_cloze("Słónco swěći jasnje a wótře.", vocab=["měsac", "wjedro", "wóheń"], rng=rng, n_options=3)
    assert item is not None
    assert isinstance(item["correct_answer_num"], int)
    assert 1 <= item["correct_answer_num"] <= len(item["possible_answers"])


def test_answer_text_in_possible_answers():
    rng = random.Random(5)
    item = qa.make_cloze("Rěka teče přez město a dale.", vocab=["jězor", "morjo", "bahno"], rng=rng, n_options=3)
    if item is not None:
        assert item["answer_text"] in item["possible_answers"].values()
