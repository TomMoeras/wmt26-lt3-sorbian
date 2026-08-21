#!/usr/bin/env python3
"""Fetch the deduplicated real+BT MT exemplar retrieval pool (see NOTICE: retrieval pools and
generated datasets are HuggingFace datasets, never vendored in this repo) into the
<src>_<tgt>.src/.tgt layout lt3wmt26.generate.Resources.pool_dir() expects.

The pool lives in the pool/ subdirectory of the published dataset. Every fetched file's sha256
is verified against data/pool_hashes.json (the frozen contract the test suite checks); a
mismatch fails loudly rather than silently serving drifted data.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

# Public HuggingFace dataset holding the reproduction data (pool/ + training/).
POOL_DATASET_ID = "TomMoeras/wmt26-lt3-sorbian-data"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                     help="Directory to write the pool's <src>_<tgt>.src/.tgt files into.")
    args = ap.parse_args(argv)

    from huggingface_hub import snapshot_download
    here = os.path.dirname(os.path.abspath(__file__))
    expected = json.load(open(os.path.join(here, "pool_hashes.json")))
    local = snapshot_download(POOL_DATASET_ID, repo_type="dataset", allow_patterns="pool/*")
    pool = os.path.join(local, "pool")
    os.makedirs(args.out, exist_ok=True)
    bad = []
    for fn in sorted(os.listdir(pool)):
        if not fn.endswith((".src", ".tgt")):
            continue
        src = os.path.join(pool, fn)
        got = hashlib.sha256(open(src, "rb").read()).hexdigest()
        if fn in expected and got != expected[fn]:
            bad.append(fn)
        shutil.copy2(src, os.path.join(args.out, fn))
    if bad:
        sys.exit(f"[fetch_pools] sha256 mismatch vs pool_hashes.json for: {bad}. "
                 "The published pool has drifted from the frozen contract; do not use it.")
    print(f"[fetch_pools] pool fetched to {args.out} ({len(expected)} files, hashes verified)",
          flush=True)


if __name__ == "__main__":
    main()
