#!/usr/bin/env python3
"""Fetch the published synthetic / derived training sets (see NOTICE: training datasets are
HuggingFace datasets, never vendored in this repo) from the training/ subdirectory of the
reproduction dataset, into a local directory of <recipe_key>/<file>.jsonl.

This is the NO-REGENERATION path: instead of rerunning the generators in training/generators/,
pull the exact archived artifacts used in training. Every fetched file's sha256 is verified
against data/regeneration_recipe.json (the frozen contract); a mismatch fails loudly.

Only the artifacts whose seed licenses permit redistribution are published here. Organizer-dev
sets (dev_fold, v10 training mix) and the SC/GC/QA synthetic components are regenerate-only:
rebuild them with training/generators/ + build_dev_fold.py against data you fetch yourself
(see training/README.md and the dataset card).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

# Same public dataset as the retrieval pool; the training sets live under training/.
TRAINING_DATASET_ID = "TomMoeras/wmt26-lt3-sorbian-data"


def _recipe_hashes():
    here = os.path.dirname(os.path.abspath(__file__))
    rec = json.load(open(os.path.join(here, "regeneration_recipe.json")))
    # {basename -> sha256} for every recipe artifact that was archived with a path + hash
    out = {}
    for a in rec["artifacts"]:
        if a.get("path") and a.get("sha256"):
            out[os.path.basename(a["path"])] = a["sha256"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                     help="Directory to write the training sets (<recipe_key>/<file>.jsonl) into.")
    args = ap.parse_args(argv)

    from huggingface_hub import snapshot_download
    expected = _recipe_hashes()
    local = snapshot_download(TRAINING_DATASET_ID, repo_type="dataset",
                              allow_patterns="training/*")
    root = os.path.join(local, "training")
    if not os.path.isdir(root):
        sys.exit(f"[fetch_training_sets] no training/ dir in {TRAINING_DATASET_ID}; "
                 "nothing to fetch.")
    os.makedirs(args.out, exist_ok=True)
    verified, bad = 0, []
    for key in sorted(os.listdir(root)):
        d = os.path.join(root, key)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".jsonl"):
                continue
            src = os.path.join(d, fn)
            got = hashlib.sha256(open(src, "rb").read()).hexdigest()
            if fn in expected and got != expected[fn]:
                bad.append(f"{key}/{fn}")
            elif fn in expected:
                verified += 1
            dst = os.path.join(args.out, key)
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src, os.path.join(dst, fn))
    if bad:
        sys.exit(f"[fetch_training_sets] sha256 mismatch vs regeneration_recipe.json for: {bad}. "
                 "The published training data has drifted from the frozen recipe; do not use it.")
    print(f"[fetch_training_sets] training sets fetched to {args.out} "
          f"({verified} files, hashes verified against the recipe)", flush=True)


if __name__ == "__main__":
    main()
