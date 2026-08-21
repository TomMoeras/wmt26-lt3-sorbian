import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from lt3wmt26.generate import (
    REQUIRED_TOP, validate_config, discover_test_files, Resources, resolve_weights,
    mt_translation, wrong_corrected, mr_answer, strip_thinking, qa_parse, qa_format_options,
    mt_prompt, sc_gc_prompt, mr_prompt, qa_prompt, run_sc_gc,
)


def _base_cfg(stack="composed"):
    cfg = {
        "system_name": "t", "weights": {"hf_repo": "r", "subfolder": "s"}, "stack": stack,
        "mt": {"decode": "greedy", "exemplar_k": 3, "pool": "deduped_realbt"},
        "qa": {"slot_scorer": stack == "composed"},
        "gc": {"union": stack == "composed", "paradigm_cap": 60,
               "witness": {"hsb": {"w": 0.1, "t": 0.42}, "dsb": {"w": 0.4, "t": 0.76}}},
        "sc": {"bktree": stack == "composed", "paradigm_cap": 30},
        "mr": {"retry_ladder": stack == "composed", "postformat": stack == "composed",
               "retrieval": {"enabled": stack == "composed", "sim_threshold": 0.95,
                             "corpus": "dev"}},
        "seed": 1234,
    }
    if stack == "composed":
        cfg["engine"] = {"batch": 96, "max_cand": 24}
    return cfg


def test_module_importable_without_torch_transformers_peft_faiss():
    # None of the heavy deps may be imported at module scope; only inside functions that
    # actually need them (Gen, resolve_weights, MiniLMRetriever, merge_ckpt.merge).
    out = subprocess.run([sys.executable, "-c", "import lt3wmt26.generate"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_validate_config_passes_on_complete_composed_config():
    validate_config(_base_cfg("composed"))


def test_validate_config_passes_on_complete_plain_config():
    validate_config(_base_cfg("plain"))


def test_validate_config_fails_on_missing_top_level_key():
    cfg = _base_cfg("composed")
    del cfg["gc"]
    with pytest.raises(AssertionError, match="gc"):
        validate_config(cfg)


def test_validate_config_fails_when_union_true_but_witness_missing():
    cfg = _base_cfg("composed")
    del cfg["gc"]["witness"]
    with pytest.raises(AssertionError, match="witness"):
        validate_config(cfg)


def test_validate_config_fails_when_retrieval_enabled_but_threshold_missing():
    cfg = _base_cfg("composed")
    del cfg["mr"]["retrieval"]["sim_threshold"]
    with pytest.raises(AssertionError, match="sim_threshold"):
        validate_config(cfg)


def test_validate_config_fails_when_composed_but_engine_missing():
    cfg = _base_cfg("composed")
    del cfg["engine"]
    with pytest.raises(AssertionError, match="engine"):
        validate_config(cfg)


def test_validate_config_allows_plain_without_engine_section():
    cfg = _base_cfg("plain")
    assert "engine" not in cfg
    validate_config(cfg)  # must not raise


def test_validate_config_fails_when_nonword_topup_enabled_but_tau_missing():
    cfg = _base_cfg("composed")
    cfg["sc"]["topups"] = {"nonword": {"enabled": True, "tau_hsb": 0.30}}  # tau_dsb missing
    with pytest.raises(AssertionError, match="tau_dsb"):
        validate_config(cfg)


def test_validate_config_passes_when_nonword_topup_disabled_without_tau():
    cfg = _base_cfg("composed")
    cfg["sc"]["topups"] = {"nonword": {"enabled": False}}
    validate_config(cfg)  # must not raise -- tau only required when the lever is on


def test_all_five_shipped_configs_validate(tmp_path):
    import glob
    for p in glob.glob("configs/*.yaml"):
        cfg = yaml.safe_load(open(p))
        validate_config(cfg, p)


def test_discover_test_files_finds_official_filenames(tmp_path):
    (tmp_path / "deu-hsb_mt_test.jsonl").write_text('{"a":1}\n', encoding="utf-8")
    (tmp_path / "hsb_qa_test.jsonl").write_text('{"a":1}\n', encoding="utf-8")
    (tmp_path / "dsb_sc_test.jsonl").write_text('{"a":1}\n', encoding="utf-8")
    found = discover_test_files(str(tmp_path))
    assert ("de", "hsb", str(tmp_path / "deu-hsb_mt_test.jsonl")) in found["mt"]
    assert ("hsb", str(tmp_path / "hsb_qa_test.jsonl")) in found["qa"]
    assert ("dsb", str(tmp_path / "dsb_sc_test.jsonl")) in found["sc"]
    assert found["gc"] == [] and found["mr"] == []


def test_resources_missing_required_key_raises(tmp_path):
    res = Resources({}, str(tmp_path))
    with pytest.raises(AssertionError, match="dict_dir"):
        res.dict_aff("hsb")


def test_resources_pool_override_wins_over_resources_yaml(tmp_path):
    res = Resources({"pool_dir": "/from/yaml"}, str(tmp_path), pool_override="/from/cli")
    assert res.pool_dir() == "/from/cli"


def test_resources_bktree_path_none_when_missing_file(tmp_path):
    res = Resources({"bktree_dir": str(tmp_path)}, str(tmp_path))
    assert res.bktree_path("hsb") is None


class _FakeEngine:
    """Stands in for lt3wmt26.sc_gc_engine.Engine so run_sc_gc can be exercised without torch/
    a real model -- only used to observe what Resources.bktree_path was called with."""

    def __init__(self, **kw):
        pass

    def constrained_correction(self, sent, w, c):
        return "CORRECT", "CORRECT"

    def run_sentence(self, sent):
        return "CORRECT", "CORRECT", 0.0, 0


def _sc_gc_run_fixture(tmp_path, monkeypatch, bktree_enabled):
    import lt3wmt26.sc_gc_engine as sc_gc_engine_mod

    monkeypatch.setattr(sc_gc_engine_mod, "Engine", _FakeEngine)
    monkeypatch.setattr(sc_gc_engine_mod, "unigram", lambda path: ({}, 1))

    dict_dir = tmp_path / "dicts"
    dict_dir.mkdir()
    (dict_dir / "hsb.aff").write_text("", encoding="utf-8")
    (dict_dir / "hsb.dic").write_text("", encoding="utf-8")

    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "hsb_sc_test.jsonl").write_text(
        json.dumps({"dataset_id": "d", "id": "0", "input_sentence": "test sentence"}) + "\n",
        encoding="utf-8")

    res_data = {"dict_dir": str(dict_dir)}
    if bktree_enabled:
        # A real (if fake-content) pickle must exist so Resources.bktree_path resolves to a
        # path instead of None -- run_sc_gc now asserts loudly on None when sc.bktree is true
        # (see test_run_sc_gc_raises_loudly_when_bktree_true_but_unresolvable below), and
        # _FakeEngine never actually unpickles it, so the content doesn't matter here.
        bktree_dir = tmp_path / "bktrees"
        bktree_dir.mkdir()
        (bktree_dir / "hsb.bktree.pkl").write_bytes(b"")
        res_data["bktree_dir"] = str(bktree_dir)
    res = Resources(res_data, str(test_dir))
    calls = []
    real_bktree_path = res.bktree_path

    def spy_bktree_path(lang):
        calls.append(lang)
        return real_bktree_path(lang)
    monkeypatch.setattr(res, "bktree_path", spy_bktree_path)

    cfg = _base_cfg("composed")
    cfg["sc"]["bktree"] = bktree_enabled
    cfg["gc"]["union"] = False

    args = SimpleNamespace(out_dir=str(tmp_path))
    gen = lambda msgs, max_new: [""] * len(msgs)

    run_sc_gc(cfg, res, args, gen, "fake-weights-path", "sc")
    return calls


def test_bktree_path_not_called_when_config_disables_it(tmp_path, monkeypatch):
    calls = _sc_gc_run_fixture(tmp_path, monkeypatch, bktree_enabled=False)
    assert calls == []


def test_bktree_path_called_when_config_enables_it(tmp_path, monkeypatch):
    calls = _sc_gc_run_fixture(tmp_path, monkeypatch, bktree_enabled=True)
    assert calls == ["hsb"]


def test_run_sc_gc_raises_loudly_when_bktree_true_but_unresolvable(tmp_path, monkeypatch):
    # C1 fix: sc.bktree: true with no usable bktree_dir/<lang>.bktree.pkl must fail loudly
    # (naming bktree_dir and scripts/setup.sh), not silently fall back to bktree_path=None.
    import lt3wmt26.sc_gc_engine as sc_gc_engine_mod

    monkeypatch.setattr(sc_gc_engine_mod, "Engine", _FakeEngine)
    monkeypatch.setattr(sc_gc_engine_mod, "unigram", lambda path: ({}, 1))

    dict_dir = tmp_path / "dicts"
    dict_dir.mkdir()
    (dict_dir / "hsb.aff").write_text("", encoding="utf-8")
    (dict_dir / "hsb.dic").write_text("", encoding="utf-8")

    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "hsb_sc_test.jsonl").write_text(
        json.dumps({"dataset_id": "d", "id": "0", "input_sentence": "test sentence"}) + "\n",
        encoding="utf-8")

    res = Resources({"dict_dir": str(dict_dir)}, str(test_dir))  # no bktree_dir at all

    cfg = _base_cfg("composed")
    cfg["sc"]["bktree"] = True
    cfg["gc"]["union"] = False

    args = SimpleNamespace(out_dir=str(tmp_path))
    gen = lambda msgs, max_new: [""] * len(msgs)

    with pytest.raises(AssertionError, match="bktree_dir"):
        run_sc_gc(cfg, res, args, gen, "fake-weights-path", "sc")


def test_resolve_weights_local_directory(tmp_path):
    (tmp_path / "LT3-DevTransfer").mkdir()
    p = resolve_weights({"hf_repo": str(tmp_path), "subfolder": "LT3-DevTransfer"})
    assert p == str(tmp_path / "LT3-DevTransfer")


# --------------------------- prompt / extraction round-trips ---------------------------

def test_mt_translation_extracts_tag():
    assert mt_translation("blah <hsb> hallo swět </hsb> junk", "hsb") == "hallo swět"


def test_mt_translation_no_tag_returns_stripped_text():
    assert mt_translation("  hallo swět  ", "hsb") == "hallo swět"


def test_wrong_corrected_extracts_both_tags():
    text = "<wrong> teh </wrong> <corrected> the </corrected>"
    assert wrong_corrected(text) == ("teh", "the")


def test_wrong_corrected_missing_tags_returns_empty():
    assert wrong_corrected("no tags here") == ("", "")


def test_mr_answer_extracts_tag_and_strips_thinking():
    text = "<think>scratch</think><answer> 42 </answer>"
    assert mr_answer(text) == "42"


def test_strip_thinking_removes_think_block():
    assert strip_thinking("<think>x</think>rest") == "rest"


def test_qa_parse_extracts_first_int():
    assert qa_parse("The answer is 3.") == 3


def test_qa_parse_no_digit_returns_minus_one():
    assert qa_parse("no number") == -1


def test_qa_format_options_sorted_numerically():
    opts = {"10": "j", "2": "b", "1": "a"}
    assert qa_format_options(opts) == "1. a\n2. b\n10. j"


def test_mt_prompt_includes_exemplars_and_source():
    msgs = mt_prompt("witaj", "Upper Sorbian", "German", [("ahoj", "hallo")])
    content = msgs[0]["content"]
    assert "Upper Sorbian: ahoj" in content and "German: hallo" in content
    assert "Upper Sorbian: witaj" in content


def test_sc_gc_prompt_wraps_sentence():
    msgs = sc_gc_prompt("test sentence", "sc")
    assert "<sentence> test sentence </sentence>" in msgs[0]["content"]


def test_mr_prompt_includes_question():
    msgs = mr_prompt("2+2=?")
    assert "2+2=?" in msgs[0]["content"]


def test_qa_prompt_includes_options_and_context():
    item = {"context": "ctx.", "question": "Q?", "possible_answers": {"1": "a", "2": "b"}}
    content = qa_prompt(item)[0]["content"]
    assert "ctx." in content and "Q?" in content and "1. a" in content and "2. b" in content
