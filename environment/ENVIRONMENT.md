# Environment specification (Sofia-side)

Referenced by `REPRODUCIBILITY.md` §2. This is the exact environment every submitted prediction
file was generated in. The full pinned list is `environment/requirements.txt` (installed by
`scripts/setup.sh`); the curated table below is what actually matters for reproduction.

## Hardware

- **Cluster:** VSC Sofia (Tier-1, Rocky Linux 9.8 "Blue Onyx").
- **Compute nodes:** `zen4_h200` partition, 1× **NVIDIA H200** GPU per job.

## Software stack

- **Conda env:** `vllm_latest`.
- **Python:** 3.12.13. **CUDA runtime (torch):** 13.0 (`torch 2.11.0+cu130`); **cuDNN:** 9.19.0.

| package | version | role |
|---|---|---|
| torch | 2.11.0+cu130 | inference / QLoRA training |
| transformers | 5.9.0 | model + generation |
| peft | 0.19.1 | QLoRA adapters, merge |
| trl | 1.7.1 | SFTTrainer (training only) |
| vllm | 0.21.0 | (batch generation where used) |
| accelerate | 1.14.0 | training launch |
| datasets | 5.0.0 | data assembly |
| sacrebleu | 2.6.0 | MT chrF++ (word_order=2), scoring |
| sentence-transformers | 5.6.0 | MT fuzzy-exemplar embeddings (+ex candidates) |
| faiss-cpu | 1.14.3 | exemplar index |
| numpy | 2.3.5 | n/a |
| kernels | 0.16.0 | **must be shimmed, see below** |

Hunspell (GC/SC engine candidate generation) uses the system `hunspell` bindings against the
`.aff`/`.dic` dictionaries in `dicts/` and the `.bktree.pkl` pickles in `bktrees/`, both produced
by `scripts/setup.sh` (fetched/built, never vendored -- see NOTICE).

## REQUIRED workaround: the kernels shim

`kernels==0.16.0` is **binary-incompatible** with `transformers==5.9.0` in this env: importing
`transformers` (hence `peft`) fails at
`kernels/layer/layer.py … ValueError: Either a revision or a version must be specified.`
(from `transformers/integrations/hub_kernels.py` building a `LayerRepository`).

The fix is a job-local shim package that disables the hub-kernels integration. It is **committed
to this repo** at `shims/kernels/__init__.py`:

```
# shims/kernels/__init__.py
raise ImportError("kernels hub package disabled job-locally "
                  "(transformers 5.9.0 x kernels 0.16.0 LayerRepository incompat)")
```

Every job must prepend the shim dir to `PYTHONPATH` so `import kernels` resolves to the shim
(raising a *caught* ImportError that makes transformers fall back to its torch path) instead of
the broken real package. **Reproduction will fail at import time without this.** The run scripts
(`scripts/reproduce_primary.sh`, `run_on_new_testset.sh`, `retrain_primary.sh`) do this for you by
sourcing `shims/activate.sh`; to run `lt3wmt26.generate` directly, first
`source shims/activate.sh`.

## HuggingFace cache and offline flags

`resolve_weights` pulls the primary's weights from the public HF repo `TomMoeras/wmt26-lt3-sorbian`
at run time, and `scripts/setup.sh` fetches the retrieval pool from a HF dataset. Therefore:

- **Do NOT set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`.** Those flags describe the ORIGINAL
  cluster runs, where every artifact was already staged locally; on a fresh checkout they break the
  weights download the very next step performs.
- `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / `TRANSFORMERS_CACHE` / `HF_DATASETS_CACHE` **override
  `HF_HOME`** and are pre-set in many shared cluster environments, which would silently serve
  weights/pool from a foreign cache. `scripts/setup.sh` pins `HF_HOME` to `./hf_cache` and unsets
  those four; do the same for any direct `lt3wmt26.generate` invocation.

Any single ≥80 GB CUDA-13 GPU reproduces the stack (H200-class assumed for the runtimes quoted).
