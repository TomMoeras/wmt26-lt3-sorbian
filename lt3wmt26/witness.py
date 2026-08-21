#!/usr/bin/env python3
"""Corpus bigram attestation witness for the GC detection decision.

Ported from `scripts/phase6/witness.py`.

Step 0 showed unigram frequency does NOT separate gold from the chosen loser, but LOCAL BIGRAM
context does (detection-miss bucket: hsb 261:24, dsb 130:30, p<1e-4). The witness is therefore
used in the GC DETECTION decision:

    edit iff  margin + w * witness >= t,   witness = bigram(candidate) - bigram(observed)

The per-language weight `w` and threshold `t` are fitted on gold dev data; they are NOT
hard-coded here (fitted values live in configs only) -- callers supply them.
"""
import csv, collections, math


def load_bigrams(path):
    """Build a bigram-count table from a monolingual corpus CSV (last column = text)."""
    big = collections.Counter()
    with open(path, encoding="utf-8") as fh:
        rd = csv.reader(fh); next(rd, None)
        for row in rd:
            t = [x.strip(".,!?;:\"'()„“»«–—").lower() for x in (row[-1] if row else "").split()]
            t = [x for x in t if x]
            for a, b in zip(t, t[1:]):
                big[(a, b)] += 1
    return big


def neighbours(sent, word):
    tk = [x.strip(".,!?;:\"'()„“»«–—") for x in sent.split()]
    lw = [x.lower() for x in tk]
    try:
        i = lw.index(word.lower())
    except ValueError:
        return None, None
    return (lw[i - 1] if i > 0 else None), (lw[i + 1] if i + 1 < len(tk) else None)


def bg(big, left, w, right):
    s = 0.0
    if left:
        s += math.log(big.get((left, w.lower()), 0) + 0.5)
    if right:
        s += math.log(big.get((w.lower(), right), 0) + 0.5)
    return s


def witness(bigrams, sent, observed, candidate):
    """bigram(candidate) - bigram(observed form) in the same slot"""
    if not observed or not candidate or observed == "CORRECT":
        return 0.0
    left, right = neighbours(sent, observed)
    return bg(bigrams, left, candidate, right) - bg(bigrams, left, observed, right)
