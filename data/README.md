Provenance and fetch scripts for third-party data this repository never vendors (see NOTICE).

- `MANIFESTS/` -- per-artifact provenance manifests (source URL, pinned version/revision,
  sha256), mirroring the working repo's own manifest tree under the same relative paths
  (`shared/`, `sorbian/mt/...`, `uk/...`, `english-math-seeds/`). `MANIFEST_SCHEMA.json` at
  `data/` root is the schema every manifest here follows. See `docs/NAME_MAPPING.md` for which
  manifest backs which config value.
- `regeneration_recipe.json` -- copy of the working repo's
  `docs/reproducibility/regeneration_recipe.json`: every synthetic / BT-derived dataset used or
  trialled, its generator script (working-repo path; see `docs/NAME_MAPPING.md` for the
  `training/generators/` port of each), seed corpus, settings, row count and sha256. Ported
  generators must regenerate a dataset whose hash matches an entry here (see
  `training/README.md`).
- `pool_hashes.json` -- per-file sha256 for the deduped retrieval pool `fetch_pools.py` writes
  into `pools/` (the `<src>_<tgt>.src`/`.tgt` layout `lt3wmt26.generate.Resources.pool_dir()`
  expects). Derived from the working repo's `docs/reproducibility/deduped_retrieval_pool.json`
  (per-direction `sha256_src`/`sha256_tgt`, residual test material 0 in all six directions).
  Verified by `scripts/setup.sh` after `fetch_pools.py` runs.
- `fetch_dicts.py` -- downloads the two GPL-licensed hunspell dictionaries (soblex for hsb,
  dsb-spell for dsb) from their source pages, pinned by exact download URL + version + sha256
  (constants at the top of the file). Run via `scripts/setup.sh`.
- `fetch_pools.py` -- downloads the **compliance-trimmed** MT exemplar retrieval pool (60k
  pairs/direction, deduped against the test sources) from the `pool/` subdirectory of the public
  HuggingFace dataset `TomMoeras/wmt26-lt3-sorbian-data`, verifying every file's sha256 against
  `pool_hashes.json`. Run via `scripts/setup.sh`. **NB:** this published pool is effectively
  real-only -- the 60k cap (applied over a real-first-ordered source) drops all back-translation on
  the two German-source directions -- whereas the submission served the full real+BT translation
  memory (deu→hsb 600k, deu→dsb 234k). See the "Published pool ≠ submission-time pool" note in
  `docs/REPRODUCIBILITY.md` §3. The published pool's real parallel rows are sourced from the WMT22
  Unsupervised & Very-Low-Resource Supervised MT shared task via the organisers' public
  distribution (`github.com/mariondimarco/WMT22_UnsupVeryLowResMT_Data`; research use only); the
  full real+BT pool is not republished because its BT targets derive from the organisers'
  monolingual distribution. Per-source license + URL + provenance in `data/MANIFESTS/`.
- `fetch_training_sets.py` -- downloads the archived synthetic / derived training sets from the
  `training/` subdirectory of the same dataset (the no-regeneration path), verifying each file
  against `regeneration_recipe.json`. The organizer-dev sets (`dev_fold`, `v10 training mix`)
  and the SC/GC/QA synthetic components are regenerate-only, not published -- rebuild them with
  `training/generators/` (see `training/README.md`).
- `fetch_organizer_data.py` -- shallow-clones the WMT26 organizer train/dev distribution
  (https://github.com/TUM-NLP/llms-limited-resources2026, public) into `organizer_data/`
  (gitignored) and writes a filled-in `resources.yaml` at the repo root (see
  `resources.yaml.template` for the committed shape). Run via `scripts/setup.sh`.
