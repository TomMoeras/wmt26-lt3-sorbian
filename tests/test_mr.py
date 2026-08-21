from lt3wmt26.mr_pipeline import (
    tier_of, postformat, format_exemplars, retrieval_layer, retry_ladder,
)


def test_tier_of_medium_id():
    assert tier_of({"id": "medium-hsb-3"}) == "medium"


def test_tier_of_low_id_default():
    assert tier_of({"id": "low-hsb-3"}) == "low"
    assert tier_of({}) == "low"


def test_postformat_wraps_conforming_latex():
    assert postformat(r"\frac{\pi}{3}", "medium") == r"$\frac{\pi}{3}$"


def test_postformat_leaves_low_tier_ints():
    assert postformat("42", "low") == "42"


def test_postformat_leaves_non_latex_medium_untouched():
    assert postformat("42", "medium") == "42"


def test_postformat_idempotent_on_already_wrapped():
    assert postformat(r"$\frac{\pi}{3}$", "medium") == r"$\frac{\pi}{3}$"


def test_postformat_empty_answer():
    assert postformat("", "medium") == ""
    assert postformat(None, "medium") == ""


def test_retrieval_exact_hit():
    idx = [("kelko jajow ma jan", "7")]
    assert retrieval_layer("Kelko  jajow ma Jan", [(q.lower(), g) for q, g in idx], 0.95) is not None


def test_retrieval_below_threshold_abstains():
    assert retrieval_layer("completely different", [("kelko jajow", "7")], 0.95) is None


def test_retrieval_picks_best_of_several():
    idx = [("kelko jajow ma jan", "7"), ("kelko psow ma jan", "2")]
    assert retrieval_layer("kelko jajow ma jan", idx, 0.95) == "7"


def test_format_exemplars_filters_by_tier_and_caps_k(tmp_path):
    dev = tmp_path / "dev.jsonl"
    dev.write_text(
        '{"id": "low-hsb-0", "question": "q0", "answer": "1"}\n'
        '{"id": "medium-hsb-0", "question": "q1", "answer": "$1$"}\n'
        '{"id": "low-hsb-1", "question": "q2", "answer": "2"}\n'
        '{"id": "low-hsb-2", "question": "q3", "answer": "3"}\n',
        encoding="utf-8",
    )
    ex = format_exemplars(str(dev), "low", k=2)
    assert ex == [{"question": "q0", "answer": "1"}, {"question": "q2", "answer": "2"}]


def _extract(raw):
    # stand-in for the organizer's src.eval.official.extract.mr_answer: content after an
    # <answer> marker, or the whole raw string if there is no marker. A bare "<answer>" with
    # nothing following it (attempt-2's continuation prompt, un-answered) extracts empty.
    return raw.split("<answer>")[-1].strip()


def test_retry_ladder_fills_empty_on_first_attempt():
    raws = ["", "42"]
    texts = ["prompt-a", "prompt-b"]

    def run(sub_texts, max_new, sample=False, temp=0.0, min_new=0):
        assert sample is False and min_new == 24
        return ["7" for _ in sub_texts]

    new_raws, stage = retry_ladder(raws, texts, run, extract_fn=_extract, max_new=640)
    assert new_raws == ["7", "42"]
    assert stage == [1, 0]


def test_retry_ladder_stops_once_nothing_empty():
    calls = []

    def run(sub_texts, max_new, sample=False, temp=0.0, min_new=0):
        calls.append(len(sub_texts))
        return ["ok" for _ in sub_texts]

    new_raws, stage = retry_ladder(["already"], ["prompt"], run, extract_fn=_extract, max_new=640)
    assert new_raws == ["already"]
    assert stage == [0]
    assert calls == []


def test_retry_ladder_advances_through_attempts_when_still_empty():
    attempts_seen = []

    def run(sub_texts, max_new, sample=False, temp=0.0, min_new=0):
        attempts_seen.append((len(sub_texts), sample))
        return ["" for _ in sub_texts]  # still empty every attempt -> ladder exhausts

    new_raws, stage = retry_ladder([""], ["prompt"], run, extract_fn=_extract, max_new=640)
    assert new_raws == [""]
    assert stage == [0]
    assert len(attempts_seen) == 3            # all three ladder rungs tried
    assert attempts_seen[-1][1] is True        # stage-3 rung samples


# --- FIX-11 regression: MR dev-retrieval `pred` must serialize as str even when the organizer
# dev `answer` is a JSON number (it is for ~half the dev items). Without the str() coercion in
# generate.mr_dev_index, dev-retrieval hits wrote ints and the MR file differed from the shipped
# submission on JSON type (24/500 rows). See docs/REPRODUCIBILITY.md section 6.

def test_mr_dev_index_coerces_numeric_answer_to_str():
    from lt3wmt26.generate import mr_dev_index
    idx = mr_dev_index([{"question": "Kelko jajow ma Jan", "answer": 18},
                        {"question": "Kak wjele", "answer": "3"}])
    assert idx == [("kelko jajow ma jan", "18"), ("kak wjele", "3")]
    assert all(isinstance(a, str) for _, a in idx)


def test_mr_dev_retrieval_hit_is_str_for_numeric_answer():
    from lt3wmt26.generate import mr_dev_index
    from lt3wmt26.mr_pipeline import retrieval_layer
    idx = mr_dev_index([{"question": "Kelko jajow ma Jan", "answer": 70000}])
    hit = retrieval_layer("Kelko  jajow  ma  Jan", idx, 0.95)
    assert hit == "70000" and isinstance(hit, str)
