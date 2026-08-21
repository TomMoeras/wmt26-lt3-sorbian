"""Seeded 50% exemplar-carrier mask + render one training entry per composition.

This is the SHIPPED MT-row prompt scheme (June/Phase-1b compfz build; correction
2026-08-17 -- see data/regeneration_recipe.json's corrected MT entry): a seeded 50% of
MT training rows ("carriers") get exactly ONE top-1 fuzzy exemplar retrieved from the
REAL-ONLY parallel pool; the other 50% are zero-shot. Ported verbatim from the working
repo's `src/data/composition.py`, with its two prompt-format helpers
(`format_user_content`, `make_entry` from `src/data_prep/prepare_dataset.py`) inlined so
the port is self-contained.

comp0   : every item is plain 0-shot.
compfz  : carriers get a top-1 fuzzy exemplar (others 0-shot).   <- SHIPPED
comprd  : carriers get a random exemplar (others 0-shot).
The SAME items are carriers across compfz/comprd (only the exemplar source differs).
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple


def format_user_content(
    src_lang: str,
    tgt_lang: str,
    source: str,
    examples: Optional[List[Tuple[str, str]]] = None,
    include_instruction: bool = True,
) -> str:
    """Build the user-role content with an optional instruction, optional examples, and the source sentence."""
    lines: List[str] = []

    if examples:
        for ex_src, ex_tgt in examples:
            lines.append(f"{src_lang}: {ex_src}")
            lines.append(f"{tgt_lang}: {ex_tgt}")

    lines.append(f"{src_lang}: {source}")
    lines.append(f"{tgt_lang}:")

    body = "\n".join(lines)
    if include_instruction:
        return f"Translate the source text from {src_lang} to {tgt_lang}\n\n{body}"
    return body


def make_entry(user_content: str, assistant_content: str) -> dict:
    """Create a single messages-format entry."""
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def carrier_mask(n: int, seed: int) -> list[bool]:
    rng = random.Random(seed)
    idx = list(range(n)); rng.shuffle(idx)
    carriers = set(idx[: n // 2])
    return [i in carriers for i in range(n)]


def render_entry(src, tgt, src_lang, tgt_lang, carrier, comp, exemplar):
    examples = None
    if carrier and comp in ("compfz", "comprd") and exemplar is not None:
        examples = [exemplar]
    uc = format_user_content(src_lang, tgt_lang, src, examples)
    return make_entry(uc, tgt)
