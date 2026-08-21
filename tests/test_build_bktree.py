import pickle

from lt3wmt26.build_bktree import BKTree, build_bktree, levenshtein, load_dict_stems
from lt3wmt26.lexicon import Hunspell
from lt3wmt26.sc_gc_engine import Engine


def test_levenshtein_basic_distances():
    assert levenshtein("kniha", "kniha") == 0
    assert levenshtein("kniha", "knihu") == 1
    assert levenshtein("kniha", "xnihx") == 2


def _write_toy_dic(tmp_path):
    dic = tmp_path / "toy.dic"
    dic.write_text("3\nkniha/AB\nknihu\nautomobil\n", encoding="utf-8")
    return dic


def test_load_dict_stems_strips_flag_suffix_and_count_header(tmp_path):
    dic = _write_toy_dic(tmp_path)
    stems = load_dict_stems(str(dic))
    assert stems == ["kniha", "knihu", "automobil"]


def test_bktree_range_finds_edit_distance_neighbours():
    tree = BKTree(["kniha", "knihu", "automobil"])
    assert set(tree.range("xnihx", 2)) == {"kniha", "knihu"}
    assert tree.range("xnihx", 0) == []


def test_build_bktree_writes_loadable_pickle(tmp_path):
    dic = _write_toy_dic(tmp_path)
    out = tmp_path / "toy.bktree.pkl"
    tree, n = build_bktree(str(dic), str(out))
    assert n == 3
    assert out.exists()
    with open(out, "rb") as fh:
        loaded = pickle.load(fh)
    assert isinstance(loaded, BKTree)
    assert set(loaded.range("xnihx", 2)) == {"kniha", "knihu"}


def test_engine_candidate_enlargement_via_pickled_bktree(tmp_path):
    """End-to-end through the actual consumer path: `Engine.candidates()` calling
    `self.bk.range(word, k)` on a pickle built by `build_bktree` and loaded exactly as
    `Engine.__init__` loads it -- the field-for-field format C1 requires. `Engine.__init__`
    itself is skipped (it loads a real HF model) since only the candidate-generation path,
    which is pure Python, needs exercising here."""
    dic = _write_toy_dic(tmp_path)
    bktree_out = tmp_path / "toy.bktree.pkl"
    build_bktree(str(dic), str(bktree_out))

    aff = tmp_path / "toy.aff"
    aff.write_text("", encoding="utf-8")
    hs = Hunspell(str(aff), str(dic))

    eng = Engine.__new__(Engine)
    eng.lang, eng.task = "toy", "sc"
    eng.hs = hs
    eng.freq, eng.tot = {}, 1
    eng.gc_paradigm_cap, eng.sc_paradigm_cap = 60, 30
    eng.max_cand = 24
    with open(bktree_out, "rb") as fh:
        eng.bk = pickle.load(fh)  # exactly how Engine.__init__ loads bktree_path

    # "xnihx" is unknown to the lexicon and at edit distance 2 from both "kniha" and "knihu" --
    # the known edit-distance-2 neighbour candidate enlargement must surface.
    cands = eng.candidates("xnihx")
    assert "kniha" in cands or "knihu" in cands


def test_cli_built_pickle_loads_in_a_fresh_process(tmp_path):
    """scripts/setup.sh builds the pickles via `python -m lt3wmt26.build_bktree`. Run as a
    script, the module is `__main__`, and a naively pickled BKTree records
    `__main__.BKTree` -- unloadable by sc_gc_engine.Engine's bare `pickle.load` in any other
    process. Build through the real CLI in a subprocess and load the pickle here, exactly as
    the engine would."""
    import subprocess, sys
    dic = tmp_path / "hsb.dic"
    dic.write_text("3\nwoda/AB\nwina\nhora\n", encoding="utf-8")
    out = tmp_path / "bk"
    subprocess.run(
        [sys.executable, "-m", "lt3wmt26.build_bktree",
         "--dict-dir", str(tmp_path), "--out", str(out), "--langs", "hsb"],
        check=True, capture_output=True)
    with open(out / "hsb.bktree.pkl", "rb") as fh:
        tree = pickle.load(fh)  # raises AttributeError without the canonical-class fix
    assert sorted(tree.range("wodu", 1)) == ["woda"]
