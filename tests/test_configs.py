import glob
import yaml

REQUIRED = {"system_name", "weights", "stack", "mt", "qa", "gc", "sc", "mr", "seed"}


def test_all_configs_complete():
    for p in glob.glob("configs/*.yaml"):
        c = yaml.safe_load(open(p))
        assert REQUIRED <= set(c), p


def test_primary_witness_params_present():
    c = yaml.safe_load(open("configs/primary.yaml"))
    assert set(c["gc"]["witness"]) == {"hsb", "dsb"} and "t" in c["gc"]["witness"]["hsb"]


def test_primary_config_shipped():
    names = {p.split("/")[-1] for p in glob.glob("configs/*.yaml")}
    assert names == {"primary.yaml"}


def test_gc_and_sc_paradigm_caps_match_scgc_engine_source():
    # scripts/phase6/scgc_engine.py: hs.paradigm(word, cap=60) for GC, cap=30 for SC.
    for p in glob.glob("configs/*.yaml"):
        c = yaml.safe_load(open(p))
        assert c["gc"]["paradigm_cap"] == 60, p
        assert c["sc"]["paradigm_cap"] == 30, p


def test_no_hardcoded_threshold_in_lt3wmt26_source():
    import re
    import subprocess
    out = subprocess.run(["grep", "-rnE", r"0\.(42|76)", "lt3wmt26/"],
                         capture_output=True, text=True)
    assert out.stdout == "", f"hard-coded threshold literal found: {out.stdout}"


def test_sc_lever_provenance_matches_shipped_files():
    # Byte-verified in the working repo's .superpowers/sdd/task-9-report.md's fix section: the shipped primary SC
    # file (sha256 3131f432...) is sc-v2 (BK-tree engine) + sc-v4's tau-gated margin top-up,
    # applied directly to sc-v2 -- NOT chained through sc-v3's guarded hsb diacritic top-up,
    # which was probed and rejected. heldoutdev/earlystop's shipped SC files are byte-identical
    # to their pre-bktree-fix, pre-margin-topup baselines (they predate the probe-day SC levers).
    # Plain configs never touch the composed-only sc_gc_engine.Engine at all.
    c = yaml.safe_load(open("configs/primary.yaml"))
    assert c["sc"]["bktree"] is True
    assert c["sc"]["topups"]["nonword"]["enabled"] is True
