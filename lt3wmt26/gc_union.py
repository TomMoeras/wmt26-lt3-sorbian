#!/usr/bin/env python3
"""GC UNION arbitration (pure function; zero new compute).

Ported from `scripts/phase6/gc_union_arbitration.py` (its `_3500` sibling is the same rule
over a different CSV bundle -- `gc_union_arbitration_3500.py`'s `arbitrate` is byte-identical
to the base version).

Per-item rule:
  if v5 flagged a word            -> keep v5's WORD, take the engine-constrained CORRECTION
                                     (fall back to correction-only if the engine disagrees on
                                     which token was flagged)
  elif margin + w*witness >= t    -> engine solo (word + correction)
  else                            -> CORRECT / CORRECT

Rationale: v5's detector is strong (it finds most errors) while the engine's value is in
choosing the replacement; the engine only gets to *introduce* an edit where it is confident.
The engine-solo branch folds in the bigram-attestation witness from `witness.py`, per that
module's detection decision `margin + w*witness >= t` -- `t` (the union threshold) and `w`
(the witness weight) are both fitted on gold dev data and are supplied by the caller (no
defaults: fitted values live in configs only).
"""


def arbitrate(row, t, w, witness_fn):
    """row keys: `v5_w`/`v5_c` (v5's flagged word/correction, "CORRECT" if none), `eng_w`/
    `eng_c` (the engine-solo word/correction) and `margin` (the engine's null-margin for that
    call), `co_w`/`co_c` (the correction-only fallback, used when the engine disagrees with
    v5 on which token was flagged).

    `witness_fn(row) -> float`: the bigram-attestation witness for this row, supplied by the
    caller (typically a closure over the loaded per-language bigram table)."""
    if row["v5_w"] and row["v5_w"] != "CORRECT":
        # v5 flagged: keep its word; prefer the engine's correction when it agrees on the token
        if row["eng_w"] == row["v5_w"] and row["eng_c"]:
            return row["v5_w"], row["eng_c"]
        return row["co_w"], row["co_c"]        # correction-only fallback
    if row["margin"] + w * witness_fn(row) >= t:
        return row["eng_w"], row["eng_c"]      # engine solo
    return "CORRECT", "CORRECT"
