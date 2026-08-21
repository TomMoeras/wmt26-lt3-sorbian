"""Phase 2 GC synthesis -- morphological suffix-substitution perturbation.

Ported from `src/data/synth/gc.py` verbatim (no cluster paths in the original).

Calibration (from the working repo's
docs/superpowers/notes/2026-06-25-phase2-synthesis-calibration.md §2):

90-93% of GC errors are clean same-stem SUFFIX SWAPS (shared prefix >= 60%).
Perturb ONE content word per sentence by swapping its inflectional suffix for a
wrong-but-real paradigm ending (same stem).

Top suffix pairs per language (correct_suffix -> wrong_suffixes):

HSB: a->ej, a->u, a->(empty), mi->ch, (empty)->om, e->(empty), u->om, a->om, u->(empty),
     a->eje, a->je, a->e, (empty)->a, eje->oh, u->oh, u->a, a->y, u->ej, om->a, a->i,
     u->eho, u->eje, om->u, i->aj
DSB: ow->am, a->u, ch->mi, ow->ami, u->e, mi->ch, u->(empty), ow->ach, u->a, u->om,
     a->eje, om->a, je->go, a->e, a->eju, a->y, u->eje, je->om, (empty)->om, ow->y, am->ow
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Suffix rules table
# Each entry: (correct_suffix, [wrong_suffixes])
# Ordered longest-first so we match the most specific suffix.
# Empty string suffix '' means bare stem (word with no inflectional suffix).
# ---------------------------------------------------------------------------

# HSB suffix rules
_HSB_RULES: List[Tuple[str, List[str]]] = [
    # Dual markers and adjective endings (longer suffixes first)
    ("eje",  ["oh", "e", "∅"]),      # eje->oh, eje->e (adjective agreement/dual)
    ("eho",  ["a", "u", "om"]),      # adjective genitive
    ("ami",  ["ach", "om", "mi"]),   # instrumental plural
    ("ach",  ["ami", "om", "mi"]),   # locative plural
    ("eje",  ["oh"]),                 # duplicate handled above; covered
    ("ej",   ["a", "u", "e"]),       # dative/dual feminine
    ("je",   ["a", "u", "om"]),      # soft-stem endings
    ("aj",   ["i", "e", "u"]),       # dual masculine
    ("om",   ["a", "u", "∅"]),       # instrumental singular
    ("mi",   ["ch", "ami", "ach"]),  # instrumental plural
    ("ch",   ["mi", "ami", "om"]),   # locative plural (hsb)
    ("oh",   ["eje", "u", "a"]),     # genitive adjective
    ("ow",   ["a", "e", "u"]),       # genitive plural
    ("ej",   ["u", "a"]),            # duplicate; covered
    ("a",    ["ej", "u", "∅", "om", "eje", "je", "e", "y", "i"]),
    ("e",    ["∅", "a", "u"]),
    ("u",    ["om", "∅", "a", "ej", "oh", "eje", "eho"]),
    ("i",    ["aj", "e", "a"]),
    ("y",    ["a", "e", "u"]),
]

# DSB suffix rules
_DSB_RULES: List[Tuple[str, List[str]]] = [
    # Longer suffixes first
    ("ami",  ["ach", "om", "ow"]),   # instrumental plural
    ("ach",  ["ami", "om", "ow"]),   # locative plural
    ("eju",  ["a", "u", "e"]),       # dual feminine DSB
    ("eje",  ["a", "u", "e"]),       # dual/adjective
    ("om",   ["a", "u", "∅"]),       # instrumental singular
    ("mi",   ["ch", "ami", "ach"]),  # instrumental plural
    ("ch",   ["mi", "ami", "om"]),   # locative plural
    ("am",   ["ow", "y", "u"]),      # dative plural
    ("go",   ["je", "om", "a"]),     # adjective genitive
    ("ow",   ["am", "ami", "ach", "y"]),  # genitive plural / possessive
    ("je",   ["go", "om", "u"]),     # soft-stem
    ("a",    ["u", "eje", "eju", "e", "y"]),
    ("e",    ["u", "a", "∅"]),
    ("u",    ["e", "∅", "a", "om", "eje"]),
    ("y",    ["a", "u", "ow"]),
]

# Combined dict for lookup
SUFFIX_RULES: Dict[str, List[Tuple[str, List[str]]]] = {
    "hsb": _HSB_RULES,
    "dsb": _DSB_RULES,
}

_EMPTY_SENTINEL = "∅"


def _apply_suffix_swap(word: str, correct_suffix: str, wrong_suffix: str) -> str:
    """Replace correct_suffix at end of word with wrong_suffix."""
    if correct_suffix == _EMPTY_SENTINEL or correct_suffix == "":
        # bare stem -- append wrong suffix
        ws = wrong_suffix if wrong_suffix != _EMPTY_SENTINEL else ""
        return word + ws
    # strip the correct suffix and attach wrong one
    stem = word[: len(word) - len(correct_suffix)]
    ws = wrong_suffix if wrong_suffix != _EMPTY_SENTINEL else ""
    return stem + ws


def perturb_word(word: str, lang: str, rng: random.Random) -> Optional[str]:
    """Apply a paradigm suffix swap to word in the given language.

    Finds the longest matching correct_suffix in the SUFFIX_RULES table,
    samples a wrong suffix that yields a different token, and returns the
    perturbed word.

    Returns None if no rule applies or no differing variant can be produced.
    """
    rules = SUFFIX_RULES.get(lang)
    if rules is None:
        return None

    # Try rules in order (longest-first by table ordering)
    for correct_suffix, wrong_suffixes in rules:
        cs = correct_suffix if correct_suffix != _EMPTY_SENTINEL else ""
        if not word.endswith(cs) or (cs == "" and len(word) == 0):
            continue
        # Shuffle wrong suffixes to randomise choice
        shuffled = list(wrong_suffixes)
        rng.shuffle(shuffled)
        for ws in shuffled:
            candidate = _apply_suffix_swap(word, cs, ws)
            if candidate != word and candidate != "":
                return candidate
    return None


def build_rows(
    sentences: List[str],
    lang: str,
    p_error: float = 0.5,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, str]]:
    """Build GC training rows with approximately p_error fraction containing a suffix error.

    Each row has keys:
        input_sentence, original_sentence, incorrect_word, correct_word

    Clean rows: incorrect_word == correct_word == 'CORRECT',
                input_sentence == original_sentence.
    Error rows: exactly one token replaced with a suffix-swapped variant;
                incorrect_word is in input_sentence.split().

    Args:
        sentences: list of clean source sentences.
        lang: 'hsb' or 'dsb'.
        p_error: target fraction of error rows.
        rng: seeded Random instance.
    Returns:
        List of row dicts (same length as sentences).
    """
    if rng is None:
        rng = random.Random(42)

    rows: List[Dict[str, str]] = []
    for sent in sentences:
        is_error = rng.random() < p_error
        if is_error:
            result = _inject_gc_error(sent, lang, rng)
            if result is not None:
                rows.append(result)
                continue
        # Clean row
        rows.append({
            "input_sentence": sent,
            "original_sentence": sent,
            "incorrect_word": "CORRECT",
            "correct_word": "CORRECT",
        })

    return rows


def _inject_gc_error(
    sentence: str,
    lang: str,
    rng: random.Random,
) -> Optional[Dict[str, str]]:
    """Try to inject a single GC error into sentence.

    Returns a row dict on success, None if no eligible token found.
    """
    tokens = sentence.split()
    # Shuffle token order to pick randomly
    indices = list(range(len(tokens)))
    rng.shuffle(indices)

    for idx in indices:
        tok = tokens[idx]
        # Eligible: length >= 3, alphabetic (skip punctuation/numbers)
        if len(tok) < 3 or not tok.isalpha():
            continue
        perturbed = perturb_word(tok, lang, rng)
        if perturbed is None or perturbed == tok:
            continue
        # Build new sentence
        new_tokens = tokens[:]
        new_tokens[idx] = perturbed
        input_sentence = " ".join(new_tokens)
        return {
            "input_sentence": input_sentence,
            "original_sentence": sentence,
            "incorrect_word": perturbed,
            "correct_word": tok,
        }

    return None
