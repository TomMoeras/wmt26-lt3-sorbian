#!/usr/bin/env bash
# Retrains the primary system (LT3-FullStack-DevTransfer weights, v10gc @3500) from scratch:
# fetch/generate the training data, then run the three-phase QLoRA recipe of
# training/README.md with training/train_qlora.py (the ported shipped trainer).
#
# Data paths: stage 0 FETCHES the archived shipped artifacts from the public dataset
# (data/fetch_training_sets.py -- hash-verified; the compfz MT rows are published, 0/1 uniform
# scheme with authentic-only training retrieval) or REGENERATES them with training/generators/
# (see data/regeneration_recipe.json). The assembled v5/v5gc/v10gc mixtures are not yet
# published, and the dev-fold builder and mixture assemblers are being ported from the working
# repo (training/README.md, "Porting status") -- until either lands, the mixture paths below
# are the target layout.
#
# Compute: each phase is a single-GPU QLoRA run (H200-class, bf16). Phase 1 is the long run
# (gold-dev optimum at step 20,000 on the shipped schedule); phases 2 and 3 are ~3.5k-step
# continuations at lr 1e-5. Selection follows docs/REPRODUCIBILITY.md section 1: the primary
# is the phase-3 (dev-folded) run at step 3,500, the step transferred from the phase-2
# sibling's gold-dev optimum.
set -euo pipefail
cd "$(dirname "$0")/.."
source shims/activate.sh   # REQUIRED kernels shim on PYTHONPATH (see environment/ENVIRONMENT.md)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: retrain_primary.sh DATA_DIR OUT_DIR"
  echo "  DATA_DIR  where training mixtures live (or are fetched to)"
  echo "  OUT_DIR   where merged weights are written per phase"
  exit 0
fi
DATA="${1:?usage: retrain_primary.sh DATA_DIR OUT_DIR}"
OUT="${2:?usage: retrain_primary.sh DATA_DIR OUT_DIR}"
BASE="Qwen/Qwen3.5-2B"

# --- stage 0: training data (fetch the archived artifacts; see header for regeneration) ---
python3 data/fetch_training_sets.py --out "$DATA"

# --- phase 1: base multitask QLoRA on the v5 mix (compfz MT + synthetic tasks) ---
python3 training/train_qlora.py \
  --messages "$DATA/v5_mix/messages_training.jsonl" \
  --base "$BASE" --out "$OUT/phase1_merged" \
  --max-steps 20000 --save-steps 1000 \
  --lr 2e-5 --lora-r 256 --lora-alpha 512 --batch-size 8 --grad-accum 2 --seq-len 1024

# --- phase 2: GC-calibrated continuation from the merged phase-1 weights ---
python3 training/train_qlora.py \
  --messages "$DATA/v5gc_mix/messages_training.jsonl" \
  --base "$OUT/phase1_merged" --out "$OUT/phase2_merged" \
  --max-steps 3500 --save-steps 500 \
  --lr 1e-5 --lora-r 256 --lora-alpha 512 --batch-size 8 --grad-accum 2 --seq-len 1024

# --- phase 3: phase-2 mixture + dev fold, stopped at the step transferred from phase 2 ---
python3 training/train_qlora.py \
  --messages "$DATA/v10gc_mix/messages_training.jsonl" \
  --base "$OUT/phase1_merged" --out "$OUT/phase3_merged" \
  --max-steps 3500 --save-steps 500 \
  --lr 1e-5 --lora-r 256 --lora-alpha 512 --batch-size 8 --grad-accum 2 --seq-len 1024

echo "primary weights: $OUT/phase3_merged (v10gc @3500)"
echo "verify: point configs/primary.yaml's weights at this directory and run"
echo "  bash scripts/reproduce_primary.sh /path/to/official/test/files"
