#!/usr/bin/env python3
"""Merge a LoRA adapter checkpoint into its base -- fixed version.

Ported from `scripts/phase5/retrain/merge_ckpt_v2.py`.

THE MERGE-BASE TRAP: `--base` has NO DEFAULT here, on purpose. v1 of this script defaulted to
`Qwen/Qwen3.5-2B`; every candidate whose true training base is a PREVIOUSLY MERGED model (the
gc-line continuation adapters -- v10gc/v5gc -- were trained on top of `merged_newchamp_v5`, not
the raw Qwen checkpoint) would then silently merge against the wrong base. The result loads and
generates plausible-looking text, so there is no downstream signal that anything went wrong --
the only defence is refusing to guess and forcing the caller to name the actual base every time.

THE EOS BUG (the reason this is "v2", not the original merger): loading a base with
`device_map="auto"` and merging on the accelerator degraded long-generation tasks (MT/MR) while
leaving short-output tasks (QA/SC/GC) looking fine -- that merge path is not numerically safe.
Separately, after `PeftModel.merge_and_unload()`, `generation_config` is rebuilt from
`config.json` and loses the base's `eos_token_id` LIST (`[248046, 248044]` collapses to
`248044`). TRL's SFTTrainer appends the chat-template turn-end token (`248046`) during training;
without it in the list the model never stops at the turn boundary and runs on into a new
'assistant' turn, which cost -14.5 chrF++ on MT silently (SC/MR extractors strip trailing junk,
so only MT/MR-length outputs showed it).

Fix: merge on CPU in float32 (numerically safe), cast to bf16 only after merging, then
reconstruct `eos_token_id` from the IN-MEMORY base config -- not `GenerationConfig.from_pretrained`,
which hits the network and can fail silently under `HF_HUB_OFFLINE=1` -- and ASSERT it before
saving. A merge that can't reconstruct `[248046, 248044]` refuses to write output rather than
ship a model that will run past the turn boundary.
"""
import argparse
import copy
import os
import shutil

EXPECTED_EOS = [248046, 248044]


def merge(base, ckpt, out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True,
                                             torch_dtype=torch.float32, device_map=None)
    base_gen_cfg = copy.deepcopy(m.generation_config)   # holds eos_token_id [248046, 248044]
    print("[base] generation_config.eos_token_id =", base_gen_cfg.eos_token_id)
    m = PeftModel.from_pretrained(m, ckpt, torch_dtype=torch.float32)
    m = m.merge_and_unload()
    m = m.to(torch.bfloat16)

    print("[pre-fix] merged eos_token_id =", m.generation_config.eos_token_id)
    base_eos = base_gen_cfg.eos_token_id
    base_eos = base_eos if isinstance(base_eos, list) else [base_eos]
    eos = [tok.eos_token_id] + [e for e in base_eos if e != tok.eos_token_id]
    m.generation_config.eos_token_id = eos
    m.generation_config.pad_token_id = base_eos[0]
    print("[fix] reconstructed eos_token_id =", m.generation_config.eos_token_id,
          "pad =", m.generation_config.pad_token_id)

    assert m.generation_config.eos_token_id == EXPECTED_EOS, (
        f"EOS RECONSTRUCTION FAILED: expected {EXPECTED_EOS}, got "
        f"{m.generation_config.eos_token_id}. Refusing to save -- this model would run past "
        f"the chat-template turn boundary during generation.")
    print(f"[assert] generation_config.eos_token_id == {EXPECTED_EOS} OK")

    m.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    # Resolve the base to its real on-disk location (v1 joined against the repo id -> no-op).
    base_dir = base
    if not os.path.isdir(base_dir):
        try:
            from huggingface_hub import snapshot_download
            base_dir = snapshot_download(base, local_files_only=True)
        except Exception as e:
            print("[warn] could not resolve base dir:", e)
            base_dir = None
    copied = []
    if base_dir:
        for f in ("chat_template.jinja",):   # generation_config is now set explicitly above
            s = os.path.join(base_dir, f)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(out, f))
                copied.append(f)
    print(f"merged -> {out} (copied: {copied or 'none'})")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="Base model directory or hub id -- NO DEFAULT (see the merge-base "
                         "trap in this module's docstring). For a continuation adapter "
                         "(v10gc/v5gc) this is a previously merged model directory, NOT the "
                         "raw Qwen checkpoint.")
    ap.add_argument("--ckpt", required=True, help="LoRA adapter checkpoint directory.")
    ap.add_argument("--out", required=True, help="Output directory for the merged model.")
    a = ap.parse_args()
    merge(a.base, a.ckpt, a.out)


if __name__ == "__main__":
    main()
