# Training data and weights recipe

This directory ports the data-synthesis generators referenced by
`data/regeneration_recipe.json` (the working repo's per-artifact generator/seed/hash record for
every synthetic or back-translation-derived training dataset) and documents the three-phase
weights recipe that consumed them. It does not include a training loop -- QLoRA fine-tuning
itself used a standard SFT trainer over the mixtures described below; what is preserved here is
how those mixtures were produced and in what proportions.

**Hash-matching pointer**: if you regenerate any dataset below, its sha256 should match the
corresponding entry in `data/regeneration_recipe.json` (given the same seed corpora and the
determinism caveats in `docs/REPRODUCIBILITY.md`'s determinism section -- decoding/generation
steps are not bitwise-deterministic; the pure-Python corpus generators here are deterministic
from their `--seed`).

**No-regeneration path**: to skip regeneration, `python3 data/fetch_training_sets.py --out DIR`
downloads the archived training sets from the `training/` subdirectory of the public dataset
`TomMoeras/wmt26-lt3-sorbian-data`, and verifies each file's sha256 against
`data/regeneration_recipe.json`. The shipped MT training rows (compfz build, 0/1 uniform
scheme, authentic-only training retrieval) are published there as
`training/mt_fuzzy_compfz/messages_training.jsonl`, alongside the MR messages. The assembled
mixtures (v5, v5gc, v10gc) are not yet published; the organizer-dev sets (`dev_fold`) and the
SC/GC/QA synthetic components are regenerate-only -- rebuild those with the generators below.

## Generators (`training/generators/`)

| generator | ported from | covers |
|---|---|---|
| `generators/synth/` (`build.py`, `common.py`, `sc.py`, `gc.py`, `qa.py`, `mr.py`) | `src/data/synth/{build,common,sc,gc,qa,mr}.py` | typo injection (SC) + paradigm/suffix-swap (GC) + cloze MCQ (QA) + MR seed translation |
| `generators/gc_calibrated_generate.py` | `scripts/phase6/gc_learn_and_generate.py` | GC-calibrated learn+generate (taxonomy learned on even-id GC dev half) |
| `generators/composition.py` + `generators/build_training_set.py` | `src/data/composition.py` + `src/data/build_training_set.py` | the SHIPPED MT-row builder (June/Phase-1b compfz): 0/1 uniform scheme, seeded 50% carrier mask (seed 42), carriers get ONE top-1 fuzzy exemplar from the real-only pool |

`training/configs/sc_gc_qa_synth_full.yaml` is the config `generators/synth/build.py` reads for
the SC/GC/QA volumes (ported from `configs/data/phase2_synth_full.yaml`; see
`docs/NAME_MAPPING.md`).

## The three-phase weights recipe

Numbers in this section are cited mechanically from `data/regeneration_recipe.json` (mixture
shares, row counts, generator names) and from `docs/REPRODUCIBILITY.md` §1 (checkpoint/selection
identities); no number here is invented.

### Phase 1 -- base multitask QLoRA ("v5")

The base model is fine-tuned once on the **v5 training mix** (1,397,057 rows -- derived total,
cluster sha256 verification pending; see the recipe's corrected entries), assembled from:

| component | rows | generator | share of v5 mix |
|---|---|---|---|
| MT fuzzy training rows | 1,069,344 | June/Phase-1b compfz build (this repo: `generators/composition.py` + `generators/build_training_set.py`): REAL-ONLY parallel rows, 0/1 uniform scheme, seeded 50% carriers get ONE top-1 fuzzy exemplar from the real-only pool | MT 76.5% |
| MR training messages | 28,147 | `translate_mr_seeds.py` (this repo: `generators/synth/mr.py`; gsm8k + Hendrycks MATH seeds) | MR 7.0% |
| SC/GC/QA synthetic | -- (volumes: SC hsb 100k/dsb 15k, GC hsb 100k/dsb 15k, QA hsb 60k/dsb 12k) | `src.data.synth.build` (this repo: `generators/synth/build.py` + `training/configs/sc_gc_qa_synth_full.yaml`) | SC 6.3% / GC 6.3% / QA 3.9% |

At inference (unchanged across all systems) MT retrieves k=3 fuzzy exemplars from the full
authentic+BT pool -- the training and inference retrieval stages are deliberately different
(training: 0/1 uniform, authentic-only; inference: k=3, authentic+BT).


Checkpoint selection: gold dev, 20k-step optimum (`docs/REPRODUCIBILITY.md` §1). This
checkpoint is the weights identity `merged_newchamp_v5` in that table.

### Phase 2 -- GC-calibrated continuation ("v5gc")

A QLoRA continuation run on the merged v5 weights, using `generators/gc_calibrated_generate.py`
to generate distribution-matched GC corruption training data: the error taxonomy (a
distribution over correct-suffix -> corrupt-suffix transformations) is learned **only** on the
even-id half of the GC dev set, so the odd-id half stays an untouched primary gate
(circularity guard). Corruptions are deduplicated against dev + test + the frozen silver
inputs.

Checkpoint selection: gold dev, step **3500** (`docs/REPRODUCIBILITY.md` §1: "v5gc @3500 |
v5 + calibrated-GC continuation | gold dev"). Continuation mixture: 300k rows replayed from the base multitask mix plus 520k calibrated GC corruptions (GC share 63.4%, total 820k), with the dev-fold variant adding 31,614 rows (852k total). These ratios are recorded in the internal run log and are not independently hash-verified in this repository.

### Phase 3 -- optional dev fold ("v10" / "v10gc")

Both the base line and the GC-calibrated line have an optional dev-fold continuation: the
**v10 dev-fold rows** (`build_dev_fold.py`, 31,614 rows, "all five dev sets in OFFICIAL eval
formats so training matches inference byte-for-byte") are folded into the corresponding
mixture (1,428,671 rows total for the v5-line variant -- derived, cluster verification
pending).

Because the dev set is now inside training data, this line **cannot be gated on held-out gold
dev**. The v5-line dev-folded checkpoint ("v10") is instead selected by a masked silver-proxy
gate (`docs/REPRODUCIBILITY.md` §1: "silver-proxy (D1, masked)"); the GC-calibrated line's
dev-folded checkpoint ("v10gc") is not independently re-gated at all -- it transfers the SAME
step as v5gc's gold-dev optimum (step 3500), since v10-line weights cannot be gated on the
dev set they were trained on. This phase is optional: the earlier P1/P2 dev-transfer / CE-early-stop candidates
in `docs/REPRODUCIBILITY.md` §1 are test-signal-free alternatives that skip it.

## Porting status

Two builders of the recipe are still being ported from the working repo and will land in
`training/generators/`: `build_dev_fold.py` (renders the five organiser development sets into
the official evaluation formats, +31,614 rows, for the phase-3 mixture) and the phase-2
mixture assembler (300k phase-1 replay + 520k GC-calibrated corruptions = the 820k v5gc mix).
Until then, `scripts/retrain_primary.sh` consumes the archived phase-2/3 mixtures fetched by
`data/fetch_training_sets.py`.

## Trainer

`training/train_qlora.py` is a byte-for-byte port of the working repo's shipped SFT trainer
(TRL `SFTTrainer`, LoRA r256/alpha512, bf16, seed 42). `scripts/retrain_primary.sh` chains the
three phases with the shipped hyperparameters (phase 1: lr 2e-5; continuations: lr 1e-5).
