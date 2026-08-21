#!/usr/bin/env python3
"""In-gap loglikelihood scorer for QA function-word cloze items.

Ported from `scripts/phase6/qa_cloze_scorer.py`.

Adjudication: the 0/12 `grammar_B*_2` blocks are FUNCTION-WORD CLOZE (16 shared
options such as `z / tohodla / waša / kak / wam`, 12 numbered gaps I..XII), not paradigm
choices -- so the GC paradigm engine cannot help. Instead: parse the gap markers, insert each
option INTO THE GAP, score the whole passage with the model, argmax.

Applied as a TARGETED OVERRIDE only where n_options is large enough; the generation path stays
everywhere else. Gold-gated on the dev grammar items (question-type distribution identical to
test, 0% dev/test overlap).

Two variants:
  independent -- argmax per gap (options may repeat across gaps)
VERIFIED on dev: a block is 12 items sharing ONE passage and ONE 16-option set, and the golds
are a PERMUTATION (e.g. hsb grammar_B1_2 golds = [1,2,3,4,6,7,8,9,10,12,14,15]: 12 distinct
options, no repeats). That is exactly the structure the assignment variant exploits.
  assignment  -- global NxM assignment (Hungarian), each option used at most once. Strictly
                 better if the blocks really are permutation-style; falls back to greedy if
                 scipy is unavailable.
If gap-marker parsing fails the lever is INERT (returns None) and the caller keeps the
generation-path prediction -- a safe failure mode.
"""
import re

GAP = re.compile(r"\(\s*([IVXLC]+)\s*\)")
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]


def target_gap(question):
    """Which gap does this item ask about? e.g. 'Wupjelnce mestno co. I z prawym slowom.'"""
    m = re.search(r"[čc]o\.?\s*([IVXLC]+)\b", question or "")
    if m:
        return m.group(1)
    m = re.search(r"\b([IVXLC]+)\b", question or "")
    return m.group(1) if m else None


def fill(context, gap_id, option):
    """Insert `option` at gap `gap_id`; strip the other gap markers so the passage reads
    naturally (leaving them in injects tokens the model never saw in training)."""
    def sub(m):
        return option if m.group(1) == gap_id else ""
    out = GAP.sub(sub, context or "")
    return re.sub(r"\s{2,}", " ", out).strip()


def solve_block(score_matrix):
    """Global assignment over a shared-option block: row i (a slot) -> chosen column
    (an option index), each column used at most once, via the Hungarian algorithm (scipy).

    scipy is a HARD dependency (it is pinned in environment/requirements.txt): the Hungarian
    assignment is part of the shipped primary QA path, and it strictly beats the greedy
    conflict resolution on permutation-style blocks. A missing scipy therefore raises loudly
    rather than silently downgrading to greedy -- a silent downgrade would be an undetected
    non-reproduction of the shipped submission (docs/relay Instruction 35).

    score_matrix: rows=slots, cols=options (same option set for every row)."""
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError(
            "solve_block requires scipy+numpy (the QA Hungarian assignment is part of the "
            "shipped primary; install environment/requirements.txt). Refusing to silently "
            "downgrade to greedy, which would be a quiet non-reproduction of the submission."
        ) from e
    mat = score_matrix
    M = np.array(mat)
    r, c = linear_sum_assignment(-M)
    picks = [None] * len(mat)
    for ri, ci in zip(r, c):
        picks[ri] = int(ci)
    return picks


class Scorer:
    def __init__(self, model, batch=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.m = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.batch = batch

    def score(self, texts):
        import torch
        out = []
        for s in range(0, len(texts), self.batch):
            b = texts[s:s + self.batch]
            enc = self.tok(b, return_tensors="pt", padding=True, truncation=True,
                           max_length=1024).to(self.m.device)
            with torch.no_grad():
                lg = self.m(**enc).logits.float()
            lp = torch.log_softmax(lg, dim=-1)
            ids, am = enc["input_ids"], enc["attention_mask"]
            tgt = lp[:, :-1, :].gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            msk = am[:, 1:].float()
            out += ((tgt * msk).sum(1) / msk.sum(1).clamp(min=1)).tolist()
        return out
