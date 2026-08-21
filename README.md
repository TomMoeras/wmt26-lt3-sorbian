# LT3 at WMT26: Multitask LLMs with Limited Resources (Sorbian)

Companion repository for the LT3 (Ghent University) system description paper. One
QLoRA-tuned Qwen 3.5 2B serves machine translation, QA, spell-checking, grammar-checking
and maths reasoning for Upper and Lower Sorbian, with a deterministic inference stack on top.

- Weights (public): https://huggingface.co/TomMoeras/wmt26-lt3-sorbian
- Paper: (link on publication)

## Results

The primary system was **joint winner of the official Sorbian track** (tied on ranking points
with team HeyBusan). Official final scores for the primary
([organizer results](https://github.com/TUM-NLP/llms-limited-resources2026/blob/main/results.md)):

| task | MT chrF++ | QA | SC | GC | MR |
|---|---|---|---|---|---|
| official score | 69.57 | 65.05 | 76.11 | 78.32 | 30.40 |
| track rank | 2nd | **1st** (+7.7) | 2nd | 2nd | **1st** (+1.4) |


## Quickstart: reproduce the primary submission

Requires **Python 3.12.13** (see `environment/ENVIRONMENT.md`) and one ≥80 GB CUDA-13 GPU. Create
and activate a virtual environment first (`setup.sh` refuses to install into a non-virtual
interpreter):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
bash scripts/setup.sh
bash scripts/reproduce_primary.sh          # defaults to official_test_flat/ built by setup.sh
```

`setup.sh` installs the pinned environment, creates the required `kernels` shim
(`shims/kernels/`, on `PYTHONPATH` via `shims/activate.sh` -- see `environment/ENVIRONMENT.md`),
fetches the two GPL dictionaries, builds the SC/GC BK-tree pickles from them (required for the
primary's `sc.bktree: true` lever), fetches the deduped retrieval pool and the organizer
train/dev/test distribution, and assembles the flat test directory `official_test_flat/` (all 14
official test files -- 6 MT `<src>-<tgt>_mt_test.jsonl` plus `{hsb,dsb}_{qa,sc,gc,mr}_test.jsonl`
-- which `reproduce_primary.sh` reads by default). It writes `resources.yaml`; none of that data
is vendored in this repository (see `NOTICE`). Pass an explicit directory to
`reproduce_primary.sh` only if your official test files live elsewhere (same flat layout).

The retrieval pool and the archived synthetic training sets are published as the HuggingFace
dataset [`TomMoeras/wmt26-lt3-sorbian-data`](https://huggingface.co/datasets/TomMoeras/wmt26-lt3-sorbian-data);
`setup.sh` fetches the pool from it and verifies every file's sha256 against
`data/pool_hashes.json`. The training sets are only needed for retraining from scratch (see
`data/fetch_training_sets.py` and `training/README.md`).

`reproduce_primary.sh` regenerates `LT3-FullStack-DevTransfer` and byte-compares it against
`submissions/LT3-FullStack-DevTransfer/MANIFEST_sha256.json`, printing one line per task file.
Example output line:

```
wmt26lowres-02-qa-sb.jsonl: exact=True differing_rows=0/415
```

## Quickstart: retrain the primary from scratch

```bash
bash scripts/setup.sh
bash scripts/retrain_primary.sh /path/to/data /path/to/weights
```

Fetches (or regenerates, see `training/README.md`) the training mixtures and runs the
three-phase QLoRA recipe with `training/train_qlora.py`, the ported shipped trainer, ending in
the primary's weights (phase 3 at step 3,500). Each phase is a single-GPU bf16 run.

## Quickstart: run on a new test set

```bash
bash scripts/setup.sh
bash scripts/run_on_new_testset.sh /path/to/your/task/jsonls
```

Runs the primary config (`configs/primary.yaml`) on any other test set in the official per-task
JSONL layout, first re-deduping the MT exemplar pool against that test set's own files so no
sentence can leak into its own exemplars (pass `--no-dedup` to skip this). Output lands in
`outputs/new_testset/` in the official five-file OCELoT format.

## Repository map

| path | contents |
|---|---|
| `lt3wmt26/` | the inference-stack package: generation driver, SC/GC engine + top-ups, GC union arbitration, witness, QA slot scorer, MR pipeline, OCELoT packaging, MT retrieval |
| `training/` | ported data-synthesis generators (`generators/`), the shipped QLoRA trainer (`train_qlora.py`), and the three-phase weights recipe (`README.md`) |
| `configs/` | the primary system's fitted-parameter YAML (`primary.yaml`), validated by `lt3wmt26.generate.validate_config` |
| `data/` | provenance manifests, the regeneration recipe, pool hashes, and the fetch scripts for every piece of third-party data (dictionaries, retrieval pool, organizer distribution) -- nothing here is vendored |
| `docs/` | `REPRODUCIBILITY.md`, `NAME_MAPPING.md` |
| `environment/` | pinned `requirements.txt` + `ENVIRONMENT.md` (hardware, required `kernels` shim) |
| `scripts/` | `setup.sh`, `reproduce_primary.sh`, `retrain_primary.sh`, `run_on_new_testset.sh`, `assemble_ocelot.py` |
| `submissions/` | the primary prediction-file bundle as submitted to OCELoT, with its `MANIFEST_sha256.json` |
| `tests/` | the test suite (`pytest`, no GPU required) |
