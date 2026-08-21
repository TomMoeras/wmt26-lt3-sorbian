#!/usr/bin/env bash
# Sets up a clean checkout: installs the pinned environment, then fetches (never vendors, see
# NOTICE) the two GPL dictionaries, builds the SC/GC BK-tree pickles from them, fetches the
# deduped retrieval pool and the organizer train/dev distribution, verifying the pool against
# pinned hashes and writing a filled-in resources.yaml before declaring setup complete.
#
# After this script, resources.yaml already points dict_dir at dicts/, bktree_dir at bktrees/,
# pool_dir at pools/, and mono_dir / mr_dev_dir at organizer_data/ (written by
# data/fetch_organizer_data.py -- that data is not redistributed by this repo; see NOTICE). See
# lt3wmt26/generate.py's `Resources` class for the full key list, and resources.yaml.template
# for the committed shape. bktree_dir is REQUIRED to reproduce the primary submission
# (configs/primary.yaml sets sc.bktree: true) -- see resources.yaml.template's comment.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: setup.sh"
  echo "  Installs environment/requirements.txt, fetches dicts/, builds bktrees/, fetches"
  echo "  pools/ and organizer_data/, builds official_test_flat/, and writes resources.yaml."
  exit 0
fi

# Environment: requires Python 3.12.13 (see environment/ENVIRONMENT.md). Refuse to install the 220
# pinned packages (incl. torch/vllm) into a non-virtual interpreter -- create a venv first.
python3 -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)' || {
  echo "refusing to pip-install into a non-virtual environment. Create one first, e.g.:" >&2
  echo "  python3.12 -m venv .venv && source .venv/bin/activate    # Python 3.12.13" >&2
  exit 1; }

# HF cache isolation: HF_HUB_CACHE (and friends) OVERRIDE HF_HOME and are pre-set on many shared
# clusters, which would silently serve weights/pool from a foreign cache. Pin them to this checkout.
: "${HF_HOME:=$(pwd)/hf_cache}"; export HF_HOME
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE

pip install -r environment/requirements.txt
mkdir -p dicts bktrees pools

# GPL lexicons: fetched, never vendored (see NOTICE)
python3 data/fetch_dicts.py --out dicts/          # soblex + dsb-spell from source pages

# SC/GC BK-tree pickles, built from the just-fetched dictionaries' wordlists (required for the
# primary's sc.bktree: true lever -- see resources.yaml.template).
python3 -m lt3wmt26.build_bktree --dict-dir dicts/ --out bktrees/

python3 data/fetch_pools.py --out pools/          # deduped retrieval pool from HF datasets

python3 - << 'PY'
import json, hashlib
exp = json.load(open("data/pool_hashes.json"))
for f, h in exp.items():
    got = hashlib.sha256(open(f"pools/{f}", "rb").read()).hexdigest()
    assert got == (h["sha256"] if isinstance(h, dict) else h), f
print("pool hashes OK")
PY

# Organizer train/dev distribution (public, fetched, never vendored -- see NOTICE): writes
# resources.yaml pointing dict_dir/bktree_dir/pool_dir/mono_dir/mr_dev_dir at the paths this
# script just fetched/built.
python3 data/fetch_organizer_data.py --out organizer_data/

# Official blind test files: lt3wmt26/generate.py's discover_test_files() expects all five tasks'
# test files FLAT in ONE directory, while the organizer distribution nests them per task. Build
# that directory (official_test_flat/) so `reproduce_primary.sh` needs no argument. Symlinks only
# -- the organizer test data is fetched, never redistributed by this repo (see NOTICE); the dir is
# .gitignored for the same reason.
mkdir -p official_test_flat
n=0
for f in organizer_data/Sorbian/*/*_test.jsonl; do
  [ -e "$f" ] || continue
  ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "official_test_flat/$(basename "$f")"
  n=$((n+1))
done
echo "official_test_flat/: $n test files (pass it, or nothing, to reproduce_primary.sh)"

echo "setup complete"
