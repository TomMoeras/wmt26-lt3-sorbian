"""Phase 2 SC synthesis -- character typo injection calibrated to dev.

Ported from `src/data/synth/sc.py` verbatim (no cluster paths in the original; pure
random-seeded string transforms over caller-supplied sentences).

Calibration (from the working repo's
docs/superpowers/notes/2026-06-25-phase2-synthesis-calibration.md §1):

Edit-type mix over single-edit bucket (68% of errors):
  substitution  35%  (of which ~35% involve diacritic/digraph confusion)
  transposition 20%  (fixed bucket, adjacent swap)
  insertion     16%
  deletion      16%
  multi-edit    13%  (implemented as double single-edit)

Within substitutions, diacritic/digraph confusion pairs (bidirectional; see CONFUSIONS below
for the literal characters -- Sorbian hacek/accent variants of s, l, r, o, c).

Position bias: word-END third (~44% hsb/46% dsb) > mid > start.

Typo'd word is an exact whitespace-token substring of input_sentence.
"""
from __future__ import annotations

import random
import string
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Diacritic / digraph confusion pairs  (bidirectional)
# Calibration §1: dominant axes are the s/hacek-s/accent-s/accent-r family, o/accent-o,
# l/barred-l -- see the literal pairs below.
# ---------------------------------------------------------------------------
_RAW_CONFUSIONS: List[Tuple[str, str]] = [
    ("s", "š"),
    ("s", "ś"),
    ("š", "ś"),
    ("ł", "w"),
    ("ř", "š"),
    ("ó", "o"),
    ("ć", "c"),
    ("ŕ", "r"),
]

# Build a bidirectional mapping: char -> list of confusion partners
CONFUSIONS: Dict[str, List[str]] = {}
for _a, _b in _RAW_CONFUSIONS:
    CONFUSIONS.setdefault(_a, []).append(_b)
    CONFUSIONS.setdefault(_b, []).append(_a)

# ---------------------------------------------------------------------------
# Edit-type probability table (calibrated, normalised from §1 single-edit mix)
# substitution .35 / transposition .20 / insertion .16 / deletion .16 / multi .13
# ---------------------------------------------------------------------------
_EDIT_TYPES = ["substitution", "transposition", "insertion", "deletion", "multi"]
_EDIT_WEIGHTS = [0.35, 0.20, 0.16, 0.16, 0.13]

# Sorbian alphabet supplement (for random insertions / substitutions)
_SORBIAN_CHARS = "aábcčćdeěéfghiíjklłmnńoóprřśštuúwyzžź"
_ALPHA_CHARS = string.ascii_lowercase + _SORBIAN_CHARS


def _bias_position(length: int, rng: random.Random) -> int:
    """Pick a character index biased toward the word-end third."""
    # end third: indices [2*length//3, length)
    # The calibration shows end ~44-46%, mid ~30%, start ~26%.
    # We model this with a triangular-ish weighting.
    weights = []
    for i in range(length):
        frac = i / max(length - 1, 1)
        if frac >= 2 / 3:
            weights.append(3.0)   # end third
        elif frac >= 1 / 3:
            weights.append(2.0)   # middle third
        else:
            weights.append(1.5)   # start third
    total = sum(weights)
    weights = [w / total for w in weights]
    # weighted choice
    r = rng.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r < cumulative:
            return i
    return length - 1


def _apply_edit(word: str, type: str, rng: random.Random) -> str:
    """Apply a single edit of the given type to word.

    Args:
        word: the word to corrupt.
        type: one of 'substitution', 'deletion', 'insertion', 'transposition', 'multi'.
    Returns:
        Corrupted string (guaranteed != word for reasonable inputs; for
        edge cases like 1-char words with deletion, may equal word).
    """
    if not word:
        return word

    if type == "transposition":
        if len(word) < 2:
            return word
        # Pick an adjacent pair to swap; bias toward end
        if len(word) == 2:
            i = 0
        else:
            i = _bias_position(len(word) - 1, rng)
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    if type == "deletion":
        if len(word) < 2:
            return word
        i = _bias_position(len(word), rng)
        return word[:i] + word[i + 1:]

    if type == "insertion":
        i = _bias_position(len(word), rng)
        # Insert a random character near the chosen position
        ch = rng.choice(_SORBIAN_CHARS)
        return word[:i] + ch + word[i:]

    if type == "substitution":
        i = _bias_position(len(word), rng)
        orig = word[i]
        # ~35% of substitutions -> diacritic confusion if the char has a partner
        if orig in CONFUSIONS and rng.random() < 0.35:
            replacement = rng.choice(CONFUSIONS[orig])
        else:
            # random substitution (not the same char)
            candidates = [c for c in _SORBIAN_CHARS if c != orig]
            replacement = rng.choice(candidates)
        return word[:i] + replacement + word[i + 1:]

    if type == "multi":
        # Apply two sequential single edits (not both transposition, variety)
        sub_types = ["substitution", "deletion", "insertion", "transposition"]
        t1 = rng.choice(sub_types)
        t2 = rng.choice(sub_types)
        w1 = _apply_edit(word, t1, rng)
        w2 = _apply_edit(w1, t2, rng)
        return w2 if w2 != word else w1

    return word


def inject_typo(
    sentence: str,
    rng: random.Random,
) -> Optional[Dict[str, str]]:
    """Inject exactly one typo into a randomly selected eligible word.

    Eligible: length >= 3, all alphabetic (unicode letters).

    Returns:
        Dict with keys input_sentence, incorrect_word, correct_word, or None if
        no eligible word found.
    """
    tokens = sentence.split()
    eligible = [
        (i, tok) for i, tok in enumerate(tokens)
        if len(tok) >= 3 and tok.isalpha()
    ]
    if not eligible:
        return None

    idx, correct_word = rng.choice(eligible)

    # Draw edit type from calibrated mix
    edit_type = rng.choices(_EDIT_TYPES, weights=_EDIT_WEIGHTS, k=1)[0]
    incorrect_word = _apply_edit(correct_word, edit_type, rng)

    # Safety: if the edit produced the same token, fall back to substitution
    attempts = 0
    while incorrect_word == correct_word and attempts < 10:
        incorrect_word = _apply_edit(correct_word, "substitution", rng)
        attempts += 1

    if incorrect_word == correct_word:
        return None

    # Build the new sentence (exactly one token replaced)
    new_tokens = tokens[:]
    new_tokens[idx] = incorrect_word
    input_sentence = " ".join(new_tokens)

    return {
        "input_sentence": input_sentence,
        "incorrect_word": incorrect_word,
        "correct_word": correct_word,
    }


def build_rows(
    sentences: List[str],
    p_error: float = 0.5,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, str]]:
    """Build SC training rows with approximately p_error fraction containing a typo.

    Each row has keys:
        input_sentence, original_sentence, incorrect_word, correct_word

    Clean rows: incorrect_word == correct_word == 'CORRECT', input_sentence == original_sentence.
    Error rows: exactly one token replaced; incorrect_word is in input_sentence.split().

    Args:
        sentences: list of clean source sentences.
        p_error: target fraction of error rows (default 0.5 = 50%).
        rng: seeded Random instance; if None, uses random.Random(42).
    Returns:
        List of row dicts (same length as sentences).
    """
    if rng is None:
        rng = random.Random(42)

    rows: List[Dict[str, str]] = []
    for sent in sentences:
        is_error = rng.random() < p_error
        if is_error:
            result = inject_typo(sent, rng)
            if result is not None:
                rows.append({
                    "input_sentence": result["input_sentence"],
                    "original_sentence": sent,
                    "incorrect_word": result["incorrect_word"],
                    "correct_word": result["correct_word"],
                })
                continue
        # Clean row (either p_clean or failed inject)
        rows.append({
            "input_sentence": sent,
            "original_sentence": sent,
            "incorrect_word": "CORRECT",
            "correct_word": "CORRECT",
        })

    return rows
