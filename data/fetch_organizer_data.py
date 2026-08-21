#!/usr/bin/env python3
"""Fetch the WMT26 "Multitask LLMs with Limited Resources" organizer train/dev
distribution (never vendored, see NOTICE) and emit a filled-in `resources.yaml` at the
repo root pointing every `lt3wmt26.generate.Resources` key at a concrete path.

Source: https://github.com/TUM-NLP/llms-limited-resources2026 (public, organizer-run).
Layout used downstream (matches the working repo's own `vendor/llms-limited-resources2026`
mirror): `Sorbian/{MT,SC,GC,QA,MR}/...`, monolingual CSVs at
`Sorbian/MT/<lang>_monolingual_2026.csv`, MR dev at `Sorbian/MR/<lang>_mr_dev.jsonl`.

This is a shallow git clone (no API token, no scraping) -- the safest way to pin an
external git repository without vendoring its contents. `--revision` lets a caller pin
an exact commit; the default is the repository's current default branch tip, which is
NOT reproducible byte-for-byte across time (the organizers may amend their distribution)
-- record the resolved commit (printed at the end) alongside any run that depends on it.

Only training-time / synthesis-time resources are covered here: the `mono_dir`/`mr_dev_dir`
`lt3wmt26.generate.Resources` keys plus the training-generator dev/test exclusion sets under
`training/generators/`. Weights, the retrieval pool, and the BK-tree pickles are fetched/built
by other scripts (see `NOTICE` and `lt3wmt26/build_bktree.py`); this script writes the other
four `Resources` keys (`dict_dir`, `bktree_dir`, `pool_dir` at their `scripts/setup.sh`-fetched
paths, plus `mono_dir`/`mr_dev_dir` from the organizer distribution this script clones) into
`resources.yaml`.
"""
import argparse
import os
import subprocess
import sys

ORGANIZER_REPO_URL = "https://github.com/TUM-NLP/llms-limited-resources2026.git"

RESOURCES_TEMPLATE = """\
# Filled in by data/fetch_organizer_data.py on {timestamp}.
# See lt3wmt26/generate.py's `Resources` class for the full key list and semantics.
# organizer_data/ and this file are gitignored -- regenerate with:
#   python3 data/fetch_organizer_data.py --out organizer_data/
dict_dir: dicts
bktree_dir: bktrees
pool_dir: pools
mono_dir: {mono_dir}
mr_dev_dir: {mr_dev_dir}
"""


def _run(cmd, **kw):
    print(f"[fetch_organizer_data] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def clone_or_update(out_dir, revision=None):
    if os.path.isdir(os.path.join(out_dir, ".git")):
        _run(["git", "-C", out_dir, "fetch", "--depth", "1", "origin",
              revision or "HEAD"])
        _run(["git", "-C", out_dir, "checkout", "FETCH_HEAD"])
    else:
        os.makedirs(os.path.dirname(out_dir) or ".", exist_ok=True)
        clone_cmd = ["git", "clone", "--depth", "1"]
        if revision:
            clone_cmd += ["--branch", revision]
        clone_cmd += [ORGANIZER_REPO_URL, out_dir]
        _run(clone_cmd)
    rev = subprocess.run(["git", "-C", out_dir, "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True).stdout.strip()
    return rev


def main(argv=None):
    import datetime

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="organizer_data",
                     help="Directory to clone the organizer distribution into.")
    ap.add_argument("--revision", default=None,
                     help="Branch/tag/commit to pin (default: repo's current default branch "
                          "tip -- NOT reproducible across time; record the printed commit).")
    ap.add_argument("--resources-out", default="resources.yaml",
                     help="Where to write the filled-in resources.yaml.")
    ap.add_argument("--sorbian-subdir", default="Sorbian",
                     help="Subdirectory of the organizer clone holding the Sorbian track "
                          "(MT/SC/GC/QA/MR); override if the organizers restructure the repo.")
    args = ap.parse_args(argv)

    rev = clone_or_update(args.out, args.revision)

    sorbian_root = os.path.join(args.out, args.sorbian_subdir)
    mono_dir = os.path.join(sorbian_root, "MT")
    mr_dev_dir = os.path.join(sorbian_root, "MR")
    for label, p in (("mono_dir", mono_dir), ("mr_dev_dir", mr_dev_dir)):
        if not os.path.isdir(p):
            print(f"[fetch_organizer_data] WARNING: expected {label} at {p!r} but it does not "
                  f"exist -- the organizer layout may have changed; edit resources.yaml by hand.",
                  file=sys.stderr)

    with open(args.resources_out, "w", encoding="utf-8") as fh:
        fh.write(RESOURCES_TEMPLATE.format(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            mono_dir=mono_dir,
            mr_dev_dir=mr_dev_dir,
        ))

    print(f"[fetch_organizer_data] organizer distribution at {args.out} (commit {rev})")
    print(f"[fetch_organizer_data] wrote {args.resources_out}")


if __name__ == "__main__":
    main()
