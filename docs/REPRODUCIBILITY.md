# Reproducibility -- LT3 at WMT26, Sorbian track

This file is the single entry point for reproducing every system we submitted, adapted from
the working repo's root `REPRODUCIBILITY.md` for this companion package. The test of this
document: an agent or reviewer with no project context, given only this repository and its two
public fetches (`scripts/setup.sh`'s dictionaries/pool and `data/fetch_organizer_data.py`'s
organizer distribution), must be able to regenerate our submitted prediction files up to the
semantic-equivalence bound documented below, and verify every hash this repository commits to.

## 1. What was submitted

The primary submission's prediction-file hashes are pinned in
`submissions/LT3-FullStack-DevTransfer/MANIFEST_sha256.json` (see `submissions/README.md`);
officially scored numbers are in the README. The primary designation is
`LT3-FullStack-DevTransfer` (weights `v10gc @3500`, composed stack + exemplars).

| family | weights identity | selection protocol |
|---|---|---|
| v5 | base multitask QLoRA, merged | gold dev (20k step optimum) |
| v10 | v5 mixture + dev fold | silver-proxy (masked) |
| P1/P2 | v10, steps 20000 / 32000 | dev-transfer / CE-early-stop (test-signal-free) |
| v5gc | v5 + GC-calibrated continuation | gold dev, step 3500 |
| v10gc | v5gc mixture + dev fold | transfer from v5gc's gold-dev optimum (step 3500) |

See `training/README.md` for how each weights family's training mixture was assembled.

## 2. Environment and weights

- Base model: Qwen 3.5 2B (task-fixed). QLoRA r256 adapters merged; parameter count unchanged.
- Full environment spec (packages, versions, CUDA, the required `kernels` shim):
  `environment/ENVIRONMENT.md`.
- Weights hosting: https://huggingface.co/TomMoeras/wmt26-lt3-sorbian (public); resolved by
  `lt3wmt26.generate.resolve_weights` from `configs/*.yaml`'s `weights.{hf_repo,subfolder}`.

## 3. Regenerating the submissions

This repository ships exactly two entry-point scripts (see the README quickstarts):

- **`scripts/reproduce_primary.sh [TEST_DIR]`** -- regenerates the primary submission
  (`configs/primary.yaml`, seed 1234) against the official WMT26 test files, then byte-compares
  the result against `submissions/LT3-FullStack-DevTransfer/MANIFEST_sha256.json` and prints a
  per-file exact-match / differing-row report. `TEST_DIR` must hold all **14 official test files
  FLAT in one directory** -- the 6 MT files `<src>-<tgt>_mt_test.jsonl` (deu-hsb, hsb-deu, deu-dsb,
  dsb-deu, hsb-dsb, dsb-hsb) plus `{hsb,dsb}_{qa,sc,gc,mr}_test.jsonl` -- because
  `lt3wmt26.generate.discover_test_files` matches those official filenames in one directory. The
  organizer distribution nests them per task, so `scripts/setup.sh` assembles the flat directory
  `official_test_flat/` from it; `reproduce_primary.sh` reads that by default when no `TEST_DIR`
  is given. (If the test-dir yields no files for the requested tasks, `generate` now exits with a
  clear message rather than silently producing an empty or partial bundle.)
- **`scripts/run_on_new_testset.sh TEST_DIR [--no-dedup]`** -- runs the same primary config on
  any other test set, first re-deduping the MT exemplar pool against that test set's own files
  (so no new-test-set sentence can leak into its own exemplars) unless `--no-dedup` is passed.

Both scripts call `lt3wmt26.generate` (one raw file per MT direction / per SC-GC-QA-MR
language) followed by `scripts/assemble_ocelot.py` (bundles those into the five-file official
OCELoT layout the manifests hash over -- see that script's docstring for why the bundling step
exists as a separate pass). Every fitted number `configs/primary.yaml` supplies is sourced
mechanically from the working repo's own delivery records; see the comment block at the top of
that file for the field-by-field provenance, including the corrected sc-v4 lineage (built
directly from sc-v2, not chained through the sc-v3 probe).

Both scripts require a filled-in `resources.yaml` at the repo root (`dict_dir`, `bktree_dir`,
`pool_dir`, `mono_dir`, `mr_dev_dir`); `scripts/setup.sh` fetches the dictionaries and pool,
builds the BK-tree pickles from the dictionaries (`python3 -m lt3wmt26.build_bktree`), and
writes `resources.yaml` for you via `data/fetch_organizer_data.py` (see
`resources.yaml.template` for the committed shape). `bktree_dir` is REQUIRED to reproduce the
primary: `configs/primary.yaml` sets `sc.bktree: true`, and `lt3wmt26/generate.py` now raises a
clear error naming `bktree_dir` and `scripts/setup.sh` if that lever is on and the pickle isn't
found, rather than silently running SC without the BK-tree candidates.

The deduped retrieval pool is fetched by `data/fetch_pools.py` from the published HuggingFace
dataset `TomMoeras/wmt26-lt3-sorbian-data` (its `pool/` subdirectory); every fetched file's
sha256 is verified against `data/pool_hashes.json` at fetch time and again by
`scripts/setup.sh`. The archived synthetic training sets live in the same dataset's
`training/` subdirectory (see `data/fetch_training_sets.py` and `data/README.md`).


## 4. Regenerating the training data and pools

- Every synthetic/BT-derived dataset: generator script + seed corpus + settings + row count +
  sha256 in `data/regeneration_recipe.json` (includes the failed T1 and QA-format datasets,
  kept for the negative-results section, and the calibrated GC-rescue data with its taxonomy
  learned on the even-id dev half). The generators themselves are ported under
  `training/generators/`; see `training/README.md` for the three-phase weights recipe that
  consumed them.
- Training vs inference retrieval (correction 2026-08-17): the two channels differ. MT
  fine-tuning rows were REAL parallel data only, with a seeded 50% of rows carrying ONE top-1
  fuzzy exemplar retrieved from the real-only pool (`training/generators/composition.py` +
  `build_training_set.py`). Only at inference does MT retrieve k=3 exemplars from the full
  real+BT pool described in section 3. Do not assume the training retrieval pool equals the
  inference pool.
- External data: every MANIFEST under `data/MANIFESTS/` carries license + URL (+ provenance
  labels).
- Inference retrieval pool: per-file sha256 in `data/pool_hashes.json`, verified by
  `scripts/setup.sh`; the full enumeration + the test-dedup proof (0 residual test material,
  11,359 pairs removed across all six directions) lives in the working repo's
  `docs/reproducibility/deduped_retrieval_pool.json` this file was derived from.
