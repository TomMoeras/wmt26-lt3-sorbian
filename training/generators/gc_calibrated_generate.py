#!/usr/bin/env python3
"""Instruction 8a -- learn the GC error taxonomy on the EVEN-id half of dev, then generate
distribution-matched corruption training data over the mono corpora. This is the
"GC-calibrated learn+generate" generator: the calibrated GC-rescue continuation data described
in docs/REPRODUCIBILITY.md §4 ("the calibrated GC-rescue data with its taxonomy learned on the
even-id dev half") and in `training/README.md`'s phase-2 mixture recipe.

Ported from `scripts/phase6/gc_learn_and_generate.py`. Changes from the original:
  - `hunspell_paradigm.Hunspell` is replaced by `lt3wmt26.lexicon.Hunspell` (same class,
    already ported once for the inference stack -- no need to vendor it twice).
  - `src.eval.official.prompts.sc_messages(s, [], "gc")` is replaced by
    `lt3wmt26.generate.sc_gc_prompt(s, "gc")`, which returns the identical one-element
    user-turn list (this repo's own self-contained port of the same prompt string).
  - The hard-coded `/sofia/projects/...` cluster paths (REPO/V/D/SILVER) are replaced by
    required CLI arguments; there is no default.

Run as a module (needed for the `lt3wmt26` import to resolve from the repo root):
    python3 -m training.generators.gc_calibrated_generate --vendor-root organizer_data/Sorbian \\
        --dict-dir dicts/ --lang hsb --n 520000 --out staged/gc_rescue/hsb.jsonl

Circularity guard: taxonomy is estimated ONLY on even-id dev items; odd-id dev is the primary
gate and is never seen here. Corruptions are made via the soblex/dsb-spell paradigm machinery
in the EXACT task format; the ~50% clean prior is matched with CORRECT/CORRECT examples.
Hygiene: dedup against dev + test + frozen-silver inputs; settings + sha256 logged.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re

from lt3wmt26.generate import sc_gc_prompt
from lt3wmt26.lexicon import Hunspell

WORD = re.compile(r"^[^\W\d_]+$", re.UNICODE)


def suffix_delta(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return (a[:i], a[i:], b[i:])   # (shared prefix, correct_suffix removed, corrupt_suffix added)


def learn(lang, vendor_root):
    rows = [json.loads(l) for l in
            open(os.path.join(vendor_root, "GC", f"{lang}_gc_dev.jsonl"), encoding="utf-8")
            if l.strip()]
    even = [r for r in rows if int(r["id"]) % 2 == 0]
    err = [r for r in even if r["incorrect_word"] != "CORRECT"]
    # taxonomy = distribution over (correct_suffix -> corrupt_suffix) transformations, learned
    # in the CORRECT->INCORRECT direction (that is what we must synthesize)
    trans = collections.Counter()
    for r in err:
        w, c = r["incorrect_word"], r["correct_word"]   # w corrupt, c correct
        pre, cs, ws = suffix_delta(c, w)                 # c's suffix -> w's suffix
        if len(pre) >= 2:                                # require a real shared stem
            trans[(cs, ws)] += 1
    clean_rate = sum(1 for r in even if r["incorrect_word"] == "CORRECT") / len(even)
    return trans, clean_rate, len(even)


def forbidden(lang, vendor_root, silver_dir):
    out = set()
    for sub, fld, files in (("SC", "input_sentence", [f"{lang}_sc_dev.jsonl", f"{lang}_sc_test.jsonl"]),
                            ("GC", "input_sentence", [f"{lang}_gc_dev.jsonl", f"{lang}_gc_test.jsonl"])):
        for f in files:
            p = os.path.join(vendor_root, sub, f)
            if os.path.exists(p):
                for l in open(p, encoding="utf-8"):
                    if l.strip():
                        r = json.loads(l)
                        out.add(" ".join(str(r.get(fld) or "").split()))
                        out.add(" ".join(str(r.get("original_sentence") or "").split()))
    if silver_dir and os.path.isdir(silver_dir):
        for f in os.listdir(silver_dir):
            if f.endswith(".jsonl"):
                for l in open(os.path.join(silver_dir, f), encoding="utf-8"):
                    if l.strip():
                        r = json.loads(l)
                        for k in ("input_sentence", "original_sentence", "context", "question"):
                            if r.get(k):
                                out.add(" ".join(str(r[k]).split()))
    out.discard("")
    return out


def mono(lang, vendor_root):
    rows = []
    with open(os.path.join(vendor_root, "MT", f"{lang}_monolingual_2026.csv"),
              encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r, None)
        for x in r:
            s = (x[-1] if x else "").strip().strip('"')
            if 5 <= len(s.split()) <= 40 and s[:1].isupper():
                rows.append(s)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor-root", required=True,
                    help="organizer-distribution Sorbian root (holds GC/, MT/)")
    ap.add_argument("--dict-dir", required=True,
                    help="directory with <lang>.aff/<lang>.dic (see data/fetch_dicts.py --out)")
    ap.add_argument("--silver-dir", default=None,
                    help="frozen silver-eval directory to exclude inputs from (optional)")
    ap.add_argument("--lang", required=True, choices=("hsb", "dsb"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    hs = Hunspell(os.path.join(a.dict_dir, f"{a.lang}.aff"), os.path.join(a.dict_dir, f"{a.lang}.dic"))
    trans, clean_rate, learn_n = learn(a.lang, a.vendor_root)
    bad = forbidden(a.lang, a.vendor_root, a.silver_dir)
    sents = [s for s in mono(a.lang, a.vendor_root) if " ".join(s.split()) not in bad]
    rng.shuffle(sents)
    # build a sampler over learned suffix-transformations
    tkeys = list(trans)
    print(f"[{a.lang}] taxonomy from {learn_n} even-id dev items: {len(tkeys)} suffix-transforms, "
          f"clean_rate={clean_rate:.3f}; mono usable={len(sents)}, forbidden={len(bad)}", flush=True)

    def corrupt(sent):
        toks = sent.split()
        idxs = [i for i, t in enumerate(toks)
                if WORD.match(t.strip(".,!?;:\"'()„“»«–—")) and len(t.strip(".,!?;:\"'()„“»«–—")) >= 4]
        rng.shuffle(idxs)
        for i in idxs:
            core = toks[i].strip(".,!?;:\"'()„“»«–—")
            pre = toks[i][:toks[i].index(core)]
            suf = toks[i][len(pre) + len(core):]
            # try a learned transform whose "remove" suffix matches this word and whose result is
            # (a) a real word-form and (b) genuinely different
            order = sorted(range(len(tkeys)), key=lambda _: rng.random())
            for j in order[:40]:
                cs, ws = tkeys[j]
                if cs and core.endswith(cs):
                    cand = core[:len(core) - len(cs)] + ws
                elif not cs:
                    cand = core + ws
                else:
                    continue
                if cand != core and hs.known(core) and hs.known(cand):
                    nt = toks[:]
                    nt[i] = pre + cand + suf
                    return " ".join(nt), cand, core     # sentence, wrong, correct
        return None

    rows, stats = [], collections.Counter()
    n_clean_target = int(a.n * clean_rate)
    tries = 0
    while len(rows) < a.n and tries < a.n * 50:
        tries += 1
        s = sents[tries % len(sents)]
        make_clean = (stats["clean"] < n_clean_target and rng.random() < clean_rate)
        if make_clean:
            m = sc_gc_prompt(s, "gc")
            tgt = "<wrong> CORRECT </wrong> <corrected> CORRECT </corrected>"
            rows.append({"messages": m + [{"role": "assistant", "content": tgt}]})
            stats["clean"] += 1
        else:
            c = corrupt(s)
            if not c:
                continue
            corr_sent, w, cw = c
            if " ".join(corr_sent.split()) in bad:
                continue
            m = sc_gc_prompt(corr_sent, "gc")
            tgt = f"<wrong> {w} </wrong> <corrected> {cw} </corrected>"
            rows.append({"messages": m + [{"role": "assistant", "content": tgt}]})
            stats["error"] += 1
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    meta = {"lang": a.lang, "n": len(rows), "clean": stats["clean"], "error": stats["error"],
            "clean_rate_learned": clean_rate, "learn_half": "even-id dev", "gate_half": "odd-id dev",
            "n_suffix_transforms": len(tkeys), "seed": a.seed, "sha256": h,
            "dedup": "dev+test GC/SC inputs + all frozen silver inputs"}
    json.dump(meta, open(a.out.replace(".jsonl", "_meta.json"), "w"), indent=2)
    print(f"[{a.lang}] wrote {len(rows)} ({stats['clean']} clean / {stats['error']} error) "
          f"-> {a.out}  sha256={h[:16]}", flush=True)


if __name__ == "__main__":
    main()
