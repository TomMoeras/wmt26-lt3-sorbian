"""Repo-hygiene guardrails: nothing organizer-restricted, GPL, or internally-named ever
gets tracked into this companion repo. See NOTICE for the policy
these tests enforce.
"""
import subprocess
import glob
import os


def tracked():
    return subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True
    ).stdout.splitlines()


def test_no_forbidden_payloads():
    t = tracked()
    assert not [f for f in t if "silver" in f.lower()]
    assert not [f for f in t if f.endswith((".aff", ".dic", ".bktree.pkl"))]  # GPL dicts
    # Organizer raw test data: "test_mt"-style filenames are how the organizer ships blind
    # test sets, but our own pytest suite (tests/test_*.py) and submission manifests
    # (submissions/) legitimately contain similar substrings -- exclude both.
    assert not [
        f for f in t
        if "test_mt" in f and "submissions/" not in f and not f.startswith("tests/")
    ]
    # Organizer data / GPL dicts / retrieval pools are fetched at runtime (data/fetch_*.py),
    # never vendored -- see NOTICE.
    assert not [f for f in t if f.startswith(("organizer_data/", "dicts/", "pools/"))]


def test_docs_only_three_files():
    assert sorted(os.listdir("docs")) == [
        "NAME_MAPPING.md",
        "REPRODUCIBILITY.md",
    ]


def test_no_internal_jargon_in_module_names():
    # Repo-wide: the global constraint on internal phase-jargon filenames applies to every
    # tracked path, not just the two Python source trees. docs/NAME_MAPPING.md is exempt --
    # its whole job is recording the internal names in its mapping column -- and so is this
    # test file itself, since the jargon tokens appear here as data, not as a filename.
    exempt = {"docs/NAME_MAPPING.md", "tests/test_repo_hygiene.py"}
    jargon_hits = [
        f for f in tracked()
        if f not in exempt and ("phase" in f or "v10" in f)
    ]
    assert not jargon_hits


def test_resources_yaml_template_tracked_but_not_filled_copy():
    t = tracked()
    assert "resources.yaml.template" in t
    assert "resources.yaml" not in t


def test_no_content_level_codename_leaks():
    """Content-level check (the removed `tests/assemble_submissions.py` leaked internal
    OCELoT-tracking vocabulary and a private relative path into the public tree by CONTENT, not
    filename, so `test_no_internal_jargon_in_module_names` above -- which only scans tracked
    *paths* -- never caught it). Scans tracked text files for that vocabulary.

    Deliberately narrow: `configs/*.yaml` and `data/regeneration_recipe.json` cite internal
    working-repo paths (`results/submission/ocelot/...`, `scripts/phase6/...`) as provenance by
    design (accepted; see the final-review's M2 finding) and are not matched by these patterns --
    this test bans the SPECIFIC leaked vocabulary (a `../..`-relative path outside this repo, the
    OCELoT internal upload-directory naming convention, and bare OCELoT submission IDs), not
    internal filenames or paths in general.
    """
    exempt = {"docs/NAME_MAPPING.md", "tests/test_repo_hygiene.py"}
    forbidden = [
        "../../results/",   # assemble_submissions.py's SRC -- a private path outside this repo
        "__upload-to__",    # OCELoT internal upload-directory naming convention it hard-coded
    ] + [f"#{n}" for n in ("160", "161", "203", "163", "186")]  # its OCELoT submission IDs

    hits = []
    for f in tracked():
        if f in exempt or not os.path.isfile(f):
            continue
        try:
            text = open(f, encoding="utf-8").read()
        except UnicodeDecodeError:
            continue  # tracked binary file -- nothing to text-scan
        for pat in forbidden:
            if pat in text:
                hits.append((f, pat))
    assert not hits, hits
