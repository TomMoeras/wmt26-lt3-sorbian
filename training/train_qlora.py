#!/usr/bin/env python3
"""QLoRA SFT trainer for the LT3 WMT26 system -- byte-for-byte port of the working repo's
`scripts/phase5/retrain/mr_probe_train.py`, the script that trained every shipped weights
family (phase-1 v5 base and the phase-2/3 continuations, with different --messages/--lr/--base
arguments; see training/README.md for the exact invocations). Trains a LoRA adapter on a
messages-format jsonl, merges it, and saves the merged bf16 model.

Shipped phase-1 invocation (train_v5.slurm): --lr 2e-5 --lora-r 256 --lora-alpha 512
--batch-size 8 --grad-accum 2 --seq-len 1024, base Qwen/Qwen3.5-2B."""
import argparse, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=0,
                    help="If >0, save an adapter checkpoint every N steps so the "
                         "step-count curve can be evaluated post-hoc on official dev "
                         "(training loss is a poor proxy for chrF++/exact-match).")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    rows = [json.loads(l) for l in open(a.messages, encoding="utf-8")]
    ds = Dataset.from_list(rows)
    lora = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    cfg = SFTConfig(output_dir=a.out + "_ckpt", max_steps=a.max_steps, warmup_steps=30,
                    learning_rate=a.lr, per_device_train_batch_size=a.batch_size,
                    gradient_accumulation_steps=a.grad_accum, max_length=a.seq_len,
                    bf16=True, seed=42, logging_steps=25,
                    save_strategy=("steps" if a.save_steps > 0 else "no"),
                    save_steps=(a.save_steps if a.save_steps > 0 else 500),
                    save_total_limit=None,
                    report_to="none", remove_unused_columns=False)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    print("[train] merging LoRA + saving ->", a.out, flush=True)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    # copy chat template + generation config from base so vLLM serves identically
    import shutil, os
    for f in ("chat_template.jinja", "generation_config.json"):
        src = os.path.join(a.base, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, f))
    print("[train] done", flush=True)


if __name__ == "__main__":
    main()
