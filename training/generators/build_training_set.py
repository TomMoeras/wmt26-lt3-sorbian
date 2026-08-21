#!/usr/bin/env python3
"""Build the Phase-1 MT dataset in three composition variants (the SHIPPED MT-row builder).

Correction 2026-08-17 (see data/regeneration_recipe.json's corrected MT entry): the
submitted system's MT fine-tuning rows are the `compfz` output of this pipeline over
REAL-ONLY parallel data -- 0/1 uniform scheme, seeded 50% carrier mask (seed 42),
carriers get ONE top-1 fuzzy exemplar retrieved from the real-only pool. BT never enters
the FT rows or the train-time retrieval pool; at inference (see `lt3wmt26/retrieval/`)
MT retrieves k=3 exemplars from the full real+BT pool instead.

Ported from the working repo's `src/data/build_training_set.py`. The scheme-defining core
(carrier mask, exemplar selection, rendering, output layout) is verbatim; the retrieval
backends and corpus loaders it calls are injected instead of imported, because they live
in the working repo (`src/extract_fuzzy/char_matcher.CharSimRetriever` for
Sorbian-source char-similarity, `src/extract_fuzzy/fuzzy_matcher.FuzzyMatcher` for
German-source MiniLM+FAISS retrieval, `src/data/{sorbian_corpus,dedupe,balance,holdout}`
for corpus assembly). Their inference-side counterparts in this repo are
`lt3wmt26/retrieval/exemplars.py` (same CharSim/MiniLM split, same interfaces). Pass
`fuzzy_fn` / `random_fn` callables mapping `(direction, train_pairs, pool)` to a list of
per-row `(src, tgt) | None` exemplars; `_random_exemplars` below is the verbatim seeded
random selector.

Output: <out_dir>/{comp0,compfz,comprd}/messages_{training,validation}.jsonl
all sharing item order; only carrier exemplars differ between compfz/comprd.
"""
from __future__ import annotations

import json
import pathlib
import random

from training.generators.composition import carrier_mask, render_entry

COMPS = ("comp0", "compfz", "comprd")


def _random_exemplars(train_pairs, pool, seed):
    rng = random.Random(seed)
    out = []
    for s, _ in train_pairs:
        cand = pool[rng.randrange(len(pool))]
        if cand[0] == s and len(pool) > 1:
            cand = pool[(rng.randrange(len(pool)))]
        out.append(cand)
    return out


def build_from_pairs(direction_pairs, out_dir, seed, val_per_direction,
                     comps=COMPS, fuzzy_fn=None, random_fn=_random_exemplars):
    """direction_pairs: {direction: (pairs, src_lang_name, tgt_lang_name, pool)}.

    fuzzy_fn(direction, train_pairs, pool) -> list of (src, tgt) | None per row; required
    when "compfz" is in comps (top-1, CharSim for Sorbian-source / MiniLM for de-source).
    """
    writers = {c: {"train": [], "val": []} for c in comps}
    for direction, (pairs, src_lang, tgt_lang, pool) in direction_pairs.items():
        val = pairs[:val_per_direction]; train = pairs[val_per_direction:]
        mask = carrier_mask(len(train), seed)
        fz = fuzzy_fn(direction, train, pool) if "compfz" in comps else None
        rd = random_fn(train, pool, seed) if "comprd" in comps else None
        for i, (s, t) in enumerate(train):
            carrier = mask[i]
            if "comp0" in comps:
                writers["comp0"]["train"].append(render_entry(s, t, src_lang, tgt_lang, carrier, "comp0", None))
            if "compfz" in comps:
                writers["compfz"]["train"].append(render_entry(s, t, src_lang, tgt_lang, carrier, "compfz", fz[i]))
            if "comprd" in comps:
                writers["comprd"]["train"].append(render_entry(s, t, src_lang, tgt_lang, carrier, "comprd", rd[i]))
        for s, t in val:   # validation is plain 0-shot, identical across comps
            for c in comps:
                writers[c]["val"].append(render_entry(s, t, src_lang, tgt_lang, False, "comp0", None))
    for c in comps:
        d = pathlib.Path(out_dir) / c; d.mkdir(parents=True, exist_ok=True)
        splits = [("training", writers[c]["train"])]
        if val_per_direction > 0:
            splits.append(("validation", writers[c]["val"]))
        for split, rows in splits:
            with open(d / f"messages_{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
