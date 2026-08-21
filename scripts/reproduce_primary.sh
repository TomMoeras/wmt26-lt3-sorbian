#!/usr/bin/env bash
# Regenerates the LT3-FullStack-DevTransfer primary submission from the frozen config and
# weights, then byte-compares it against the shipped MANIFEST_sha256.json.
#
# generate.py writes one raw file per MT direction / per SC-GC-QA-MR language; scripts/
# assemble_ocelot.py bundles those into the 5-file official layout the manifest hashes were
# computed over (see that script's docstring for why the bundling step exists at all).
#
# Requires a resources.yaml at the repo root (dict_dir, bktree_dir, pool_dir, mono_dir, mr_dev_dir) --
# see scripts/setup.sh's trailing comment for what it must point at.
set -euo pipefail
cd "$(dirname "$0")/.."
source shims/activate.sh   # REQUIRED kernels shim on PYTHONPATH (see environment/ENVIRONMENT.md)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: reproduce_primary.sh [TEST_DIR]"
  echo "  TEST_DIR: directory holding all 14 official test files FLAT (6 MT <src>-<tgt>_mt_test.jsonl"
  echo "            + {hsb,dsb}_{qa,sc,gc,mr}_test.jsonl). Defaults to official_test_flat/, which"
  echo "            scripts/setup.sh builds from the organizer distribution."
  echo "  Regenerates the primary submission and diffs it against the shipped manifest."
  exit 0
fi
# Defaults to the flat test dir scripts/setup.sh builds from the organizer distribution.
TEST_DIR="${1:-official_test_flat}"
OUT=outputs/primary

python3 -m lt3wmt26.generate --config configs/primary.yaml --test-dir "$TEST_DIR" --out-dir "$OUT" --seed 1234
python3 scripts/assemble_ocelot.py --out-dir "$OUT"

python3 - << PY
import json, hashlib, os
man = json.load(open("submissions/LT3-FullStack-DevTransfer/MANIFEST_sha256.json"))
for fn, meta in man.items():
    ours = open(os.path.join("$OUT", fn), "rb").read()
    match = hashlib.sha256(ours).hexdigest() == meta["sha256"]
    ref = [l for l in open(os.path.join("submissions/LT3-FullStack-DevTransfer", fn))]
    new = [l for l in open(os.path.join("$OUT", fn))]
    diff = sum(1 for a, b in zip(ref, new) if a != b)
    print(f"{fn}: exact={match} differing_rows={diff}/{len(ref)}")
PY
