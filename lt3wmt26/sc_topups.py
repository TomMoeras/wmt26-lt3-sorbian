#!/usr/bin/env python3
"""SC non-word top-ups (deterministic, no GPU): fire on items the primary pipeline abstained
on (`pred_incorrect == "CORRECT"`).

Ported from `scripts/phase6/apply_sc_diacritic.py` + `apply_sc_diacritic_hsb.py` (the two
scripts are identical except for which language's `min_freq` they hard-coded into a `MINF`
dict -- merged here into one function, `nonword_topup`, with that value taken as the `tau`
parameter) and `scripts/phase6/apply_scv4_topup.py` (the engine-margin top-up, ported as
`margin_topup`).

`nonword_topup`: on an item currently answered CORRECT, a token qualifies if it is NOT in the
dictionary and some hunspell-MAP diacritic variant of it IS in the dictionary and clears the
corpus-frequency floor `tau` (the source's `min_freq`); fire only when EXACTLY ONE token in
the sentence qualifies.

`margin_topup`: on an abstained item, adopt the engine's own solo correction when its
null-margin clears the deploy `tau` (the source's per-language `TAU`) and the guards pass.

Guards (both functions; caller-supplied via `guards`, no defaults -- fitted/tuned choices
live in configs only): `exactly_one`, `skip_caps` (proper nouns), `skip_quoted` (deliberately
quoted forms). Both source scripts found real over-firing on exactly these two categories
when the guards were dropped.
"""

STRIP = ".,!?;:\"'()„“»«–—"
QUOTES = ("'", '"', "„", "“", "»", "«")


def nonword_topup(sentence, lexicon, freq, tau, guards):
    """(word, correction) if exactly one non-word token in `sentence` has a frequent
    diacritic-restored dictionary form, else None."""
    cands = []
    for t in sentence.split():
        core = t.strip(STRIP)
        if len(core) < 3 or lexicon.known(core):
            continue
        if guards.get("skip_caps") and core[:1].isupper():
            continue   # proper nouns are not errors
        if guards.get("skip_quoted") and any(
                f"{q}{core}" in sentence or f"{core}{q}" in sentence for q in QUOTES):
            continue   # deliberately quoted forms are not errors
        fix = [v for v in lexicon.diacritic_variants(core)
               if lexicon.known(v) and freq.get(v.lower(), 0) >= tau]
        if fix:
            cands.append((core, max(fix, key=lambda v: freq.get(v.lower(), 0))))
    if guards.get("exactly_one", True) and len(cands) != 1:
        return None
    return cands[0] if cands else None


def margin_topup(word, corr, margin, sentence, tau, guards):
    """Adopt an abstained SC engine-solo correction (`word`, `corr`, `margin`) when `margin`
    clears the deploy `tau` and the guards pass against `sentence`; else None."""
    if not word or word == "CORRECT" or margin < tau:
        return None
    if guards.get("skip_caps"):
        toks = sentence.split()
        first = toks[0].strip(STRIP) if toks else ""
        if word[:1].isupper() and word != first:
            return None
    if guards.get("skip_quoted") and any(
            f"{q}{word}" in sentence or f"{word}{q}" in sentence for q in QUOTES):
        return None
    return word, corr
