# Name mapping: companion module/config names -> internal working-repo sources

This file exists so a reader who spots a working-repo path in a paper draft, a review comment,
or an old note can find the corresponding file in this repository, and vice versa.

## 1. `lt3wmt26/` modules -> internal source(s)

| companion module | internal source(s) | what changed in the port |
|---|---|---|
| `lt3wmt26/generate.py` | `scripts/phase6/blind_generate.py` | config-driven (`configs/*.yaml` + `--resources`) instead of CLI-flag defaults; self-contained prompt/extraction functions instead of the organiser-harness import |
| `lt3wmt26/sc_gc_engine.py` | `scripts/phase6/scgc_engine.py` (candidate generation/scoring) + `scgc_correct_only.py` (`constrained_correction`) | lexicon and BK-tree path injected by the caller; loud failure on an unloadable BK-tree instead of a silent except |
| `lt3wmt26/sc_topups.py` | `scripts/phase6/apply_sc_diacritic.py` + `apply_sc_diacritic_hsb.py` (merged into `nonword_topup`, per-language `min_freq` now a `tau` parameter) + `apply_scv4_topup.py` (`margin_topup`) | two near-duplicate scripts merged into one parameterized function; guards passed explicitly, no hard-coded defaults |
| `lt3wmt26/gc_union.py` | `scripts/phase6/gc_union_arbitration.py` (`_3500` sibling is byte-identical, not separately ported) | pure function; `t`/`w` supplied by the caller instead of hard-coded |
| `lt3wmt26/witness.py` | `scripts/phase6/witness.py` | unchanged decision rule, parameterized |
| `lt3wmt26/qa_slot_scorer.py` | `scripts/phase6/qa_cloze_scorer.py` | unchanged scoring logic |
| `lt3wmt26/mr_pipeline.py` | `scripts/phase6/mr_run.py` (`gen()`/`score()` cluster CLI harness stripped) | pure, testable core kept: `tier_of`, `postformat`, `format_exemplars`, `retry_ladder`; GPU call + answer extractor injected by the caller |
| `lt3wmt26/package_ocelot.py` | `scripts/phase6/ocelot_repackage.py` (`main()`'s blindset-join + validation stripped; kept out of scope, see `scripts/assemble_ocelot.py` below) | pure per-task field-order re-shaping only |
| `lt3wmt26/lexicon.py` | `scripts/phase6/hunspell_paradigm.py` | unchanged parsing/paradigm logic |
| `lt3wmt26/merge_ckpt.py` | `scripts/phase5/retrain/merge_ckpt_v2.py` | `--base` has no default (the v1-script merge-base trap this version fixes) |
| `lt3wmt26/retrieval/pool.py` | `scripts/phase6/build_deduped_pool.py` | the test-file blocklist is a caller-supplied parameter instead of six hard-coded, vendored paths |
| `lt3wmt26/retrieval/exemplars.py` | `src/extract_fuzzy/char_matcher.py` (`CharSimRetriever`) + the `FuzzyMatcher` encoder path (`microsoft/Multilingual-MiniLM-L12-H384`) | kept verbatim (pure, no cluster paths in the original) |
| `scripts/assemble_ocelot.py` | new file, not a 1:1 port | bridges the gap `package_ocelot.py`'s own docstring places out of scope: joins `generate.py`'s per-direction/per-language files into the 5-file official OCELoT layout the submission manifests hash over (added by the companion-repo build; see `the working repo's .superpowers/sdd/task-10-report.md`) |
| `lt3wmt26/build_bktree.py` | `src/bets/b2_hybrid/candidates.py` (`BKTree`, `levenshtein`, `load_dict_stems`, `build_bktree_cached`) | algorithm body (`levenshtein`, `BKTree.add`/`.range`) unchanged; `build_bktree_cached`'s mtime-based cache check dropped in favor of an explicit CLI build step (`scripts/setup.sh`) since this repo never vendors a stale pickle to compare against |

## 1a. New orchestration files (not ports)

The three shell scripts and the `data/fetch_*.py` fetchers are new to this companion package
(the working repo's equivalent steps were manual/cluster-specific); there is no internal source
to map them to.

| companion file | role |
|---|---|
| `scripts/setup.sh` | installs the pinned environment, fetches dictionaries, builds the BK-tree pickles, fetches the retrieval pool and organizer distribution, writes `resources.yaml` |
| `scripts/reproduce_primary.sh` | regenerates the primary submission and byte-compares it against the shipped manifest |
| `scripts/run_on_new_testset.sh` | runs the primary config on an arbitrary new test set |
| `scripts/assemble_ocelot.py` | see row above (new file, not a port) |
| `data/fetch_dicts.py` | fetches the two GPL hunspell dictionaries |
| `data/fetch_pools.py` | fetches the deduped retrieval pool from the `pool/` subdirectory of the HF dataset `TomMoeras/wmt26-lt3-sorbian-data`, verifying sha256 against `data/pool_hashes.json` |
| `data/fetch_training_sets.py` | fetches the archived synthetic training sets from the same dataset's `training/` subdirectory, verifying sha256 against `data/regeneration_recipe.json` |
| `data/fetch_organizer_data.py` | fetches the organizer train/dev distribution and writes `resources.yaml` |
| `lt3wmt26/build_bktree.py` | see row above (ported algorithm, new CLI entry point) |

## 2. `training/generators/` -> internal source(s)

| companion module | internal source(s) |
|---|---|
| `training/generators/synth/{build,common,sc,gc,qa,mr}.py` | `src/data/synth/{build,common,sc,gc,qa,mr}.py` |
| `training/generators/gc_calibrated_generate.py` | `scripts/phase6/gc_learn_and_generate.py` (+ `hunspell_paradigm.py` and `src.eval.official.prompts.sc_messages`, now `lt3wmt26.lexicon` / `lt3wmt26.generate.sc_gc_prompt`) |
| `training/generators/composition.py` | `src/data/composition.py` (+ inlined `format_user_content`/`make_entry` from `src/data_prep/prepare_dataset.py`) |
| `training/generators/build_training_set.py` | `src/data/build_training_set.py` (retrieval backends injected; internally `src/extract_fuzzy/{char_matcher,fuzzy_matcher}.py`) |
| `training/configs/sc_gc_qa_synth_full.yaml` | `configs/data/phase2_synth_full.yaml` |

## 3. Config values -> originating gate records

Every fitted number in `configs/*.yaml` was extracted mechanically from the working repo's
delivery/decision records, not hand-typed; the field-by-field source is documented in the
comment block at the top of `configs/primary.yaml` (reproduced/summarised here for the other
four configs, which share the same sourcing):

| config field | value(s) | originating record |
|---|---|---|
| `weights.hf_repo` / `weights.subfolder` | e.g. `LT3-DevTransfer` | HF delivery README's per-candidate subfolder table, cross-checked against the candidate's own `weights_sha256` |
| `mt.exemplar_k` | `3` | delivery README's per-candidate flag table (`--mt-exemplars 3`) |
| `gc.witness.<lang>.w` / `.t` | hsb `(0.1, 0.42)`, dsb `(0.4, 0.76)` | delivery README's fitted-parameter table for the gc-line candidates -- **not** the base-v5 row in the same table (`t*=0.210`, hsb `(0.1,0.22)`, dsb `(0.8,0.74)`), which that table's own note marks as superseded for gc-line candidates. This is the GC union-arbitration threshold (`t`) and witness weight (`w`) `lt3wmt26/gc_union.py` consumes. |
| `gc.paradigm_cap` | `60` | `scripts/phase6/scgc_engine.py`'s inline `hs.paradigm(word, cap=60)` GC call site |
| `sc.paradigm_cap` | `30` | the same engine's SC call site, `cap=30` |
| `engine.batch` / `engine.max_cand` | `96` / `24` | `scripts/phase6/scgc_engine.py`'s `Engine.__init__(..., batch=96, max_cand=24)` |
| `sc.topups.diacritic.dsb.min_freq` | `5` | `scripts/phase6/apply_sc_diacritic.py`'s `MINF = {"dsb": 5}` (docstring: "SHIP SET: dsb only ... hsb is WITHHELD") |
| `sc.topups.nonword.tau_hsb` / `.tau_dsb` | `0.30` / `0.36` | `scripts/phase6/apply_scv4_topup.py`'s per-language `TAU`, described there as "full-fit on consistent dev" |
| `sc.bktree` | `true` | the re-enabled BK-tree edit-2 candidate generator (the sc-v2 fix) |
| `seed` | `1234` (submission configs) | `scripts/phase6/blind_generate.py`'s `--seed` default, added post-submission (see `docs/REPRODUCIBILITY.md` §5 -- shipped bundles were themselves unseeded; this seed is for future regenerations) |

The provenance of the shipped primary SC file's exact lever chain (bktree -> dsb diacritic
top-up -> sc-v4 margin top-up, built from sc-v2 directly rather than through the sc-v3 probe)
is documented in `configs/primary.yaml`'s comment block and corrected in
the working repo's disclosure records.
