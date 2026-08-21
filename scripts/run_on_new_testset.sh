#!/usr/bin/env bash
# Runs the primary config on a NEW (non-WMT26) test set. By default the MT exemplar pool is
# first re-deduped against the new test set's own files (lt3wmt26.retrieval.pool --dedup-against)
# so no new-test-set sentence can leak into its own exemplars; pass --no-dedup to skip that and
# reuse pools/ as-is (e.g. when the new test set shares no content with the pool by construction).
#
# Requires a resources.yaml at the repo root (dict_dir, bktree_dir, pool_dir, mono_dir, mr_dev_dir) --
# see scripts/setup.sh's trailing comment for what it must point at.
set -euo pipefail
cd "$(dirname "$0")/.."
source shims/activate.sh   # REQUIRED kernels shim on PYTHONPATH (see environment/ENVIRONMENT.md)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: run_on_new_testset.sh /path/to/task/jsonls [--no-dedup]"
  echo "  Runs configs/primary.yaml on a new test set; official 5-file OCELoT output."
  exit 0
fi
TEST_DIR="${1:?usage: run_on_new_testset.sh /path/to/task/jsonls [--no-dedup]}"
OUT=outputs/new_testset

if [[ "${2:-}" != "--no-dedup" ]]; then
  python3 -m lt3wmt26.retrieval.pool --dedup-against "$TEST_DIR" --pool pools/ --out pools_deduped_new/
  POOL=pools_deduped_new/
else
  POOL=pools/
fi

python3 -m lt3wmt26.generate --config configs/primary.yaml --test-dir "$TEST_DIR" --out-dir "$OUT" --pool-override "$POOL" --seed 1234
python3 scripts/assemble_ocelot.py --out-dir "$OUT"
echo "outputs in $OUT (official 5-file OCELoT format)"
