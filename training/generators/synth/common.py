"""Phase 2 synthesis common utilities: pool filtering, mono loading, and official-format
training-row renderers.

Ported from `src/data/synth/common.py`. That module's renderers reused
`src.eval.official.prompts.sc_messages`/`mr_messages` (the organizer-harness prompt builders,
not part of this repo) so the training user-turn is byte-identical to the submission-time
prompt. This port reuses `lt3wmt26.generate.sc_gc_prompt`/`mr_prompt` instead -- those are the
companion repo's own self-contained port of the same prompt strings (see `lt3wmt26/generate.py`
"prompts / extraction" section), so the identity still holds without importing the
organizer-harness package.

Provides:
- length_ok, dedup, exclude_dev -- sentence pool filtering
- load_mono -- multi-source ranked CSV/gz loader
- seeded_sample -- deterministic sampling
- render_sc_row, render_gc_row, render_qa_row, render_mr_row -- official-format renderers
"""
from __future__ import annotations

import csv
import gzip
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from lt3wmt26.generate import sc_gc_prompt, mr_prompt


# ---------------------------------------------------------------------------
# Pool utilities
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Collapse whitespace and casefold for dedup/exclusion keys."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def length_ok(text: str, lo: int = 3, hi: int = 40) -> bool:
    """Return True iff whitespace-split token count is in [lo, hi]."""
    n = len(text.split())
    return lo <= n <= hi


def dedup(rows: List[str]) -> List[str]:
    """Return rows with duplicates removed (whitespace-normalised, casefold key, first-seen wins)."""
    seen: Set[str] = set()
    out: List[str] = []
    for row in rows:
        key = _normalise(row)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def exclude_dev(pool: List[str], dev_set: Set[str]) -> List[str]:
    """Drop pool sentences whose normalised form appears in dev_set (normalised)."""
    norm_dev = {_normalise(s) for s in dev_set}
    return [s for s in pool if _normalise(s) not in norm_dev]


def load_mono(
    lang: str,
    sources: List[str],
    cap: Optional[int] = None,
    seed: int = 42,
) -> List[str]:
    """Load monolingual sentences from a ranked list of file paths.

    Supports:
    - ``*.csv`` / ``*.csv.gz``: reads a column named after ``lang`` (e.g. "hsb") or
      falls back to the first column.
    - ``*.gz`` (plain text gzip): one sentence per line.
    - plain text files: one sentence per line.

    Applies length_ok filter (3-40 tokens) and dedup. Does NOT dev-exclude here;
    call exclude_dev separately after loading.

    Args:
        lang: language code (e.g. "hsb", "dsb") -- used as CSV column name.
        sources: ordered list of file paths (high to low quality).
        cap: if set, stop after collecting this many unique sentences total.
        seed: reserved for call-site shuffling; not applied here (sources are read
              in order).

    Returns:
        Deduplicated, length-filtered list of sentences.
    """
    accumulated: List[str] = []
    seen: Set[str] = set()

    for src in sources:
        p = Path(src)
        if not p.exists():
            continue

        name_lower = p.name.lower()
        is_csv = ".csv" in name_lower

        def _iter_lines():
            if name_lower.endswith(".gz"):
                with gzip.open(p, "rt", encoding="utf-8", errors="replace") as fh:
                    yield from fh
            else:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    yield from fh

        if is_csv:
            if name_lower.endswith(".gz"):
                with gzip.open(p, "rt", encoding="utf-8", errors="replace") as fh:
                    reader = csv.DictReader(fh)
                    col = lang if (reader.fieldnames and lang in reader.fieldnames) else None
                    for row in reader:
                        text = row.get(lang, "") if col else next(iter(row.values()), "")
                        text = text.strip()
                        if not text or not length_ok(text):
                            continue
                        key = _normalise(text)
                        if key not in seen:
                            seen.add(key)
                            accumulated.append(text)
                            if cap and len(accumulated) >= cap:
                                return accumulated
            else:
                with open(p, encoding="utf-8", errors="replace", newline="") as fh:
                    reader = csv.DictReader(fh)
                    col = lang if (reader.fieldnames and lang in reader.fieldnames) else None
                    for row in reader:
                        text = row.get(lang, "") if col else next(iter(row.values()), "")
                        text = text.strip()
                        if not text or not length_ok(text):
                            continue
                        key = _normalise(text)
                        if key not in seen:
                            seen.add(key)
                            accumulated.append(text)
                            if cap and len(accumulated) >= cap:
                                return accumulated
        else:
            for line in _iter_lines():
                text = line.strip()
                if not text or not length_ok(text):
                    continue
                key = _normalise(text)
                if key not in seen:
                    seen.add(key)
                    accumulated.append(text)
                    if cap and len(accumulated) >= cap:
                        return accumulated

    return accumulated


def seeded_sample(items: List, k: int, seed: int) -> List:
    """Return a deterministic sample of k items from items."""
    rng = random.Random(seed)
    return rng.sample(items, k)


# ---------------------------------------------------------------------------
# Official-format renderers
# ---------------------------------------------------------------------------

def render_sc_row(
    input_sentence: str,
    incorrect_word: str,
    correct_word: str,
    exemplars: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Render one SC training example in axolotl chat format.

    exemplars is accepted for interface parity with the working-repo generator but is unused:
    the ported `sc_gc_prompt` builds the exemplar-free instruction+sentence prompt (no
    generator in this recipe passed SC/GC exemplars -- see `data/regeneration_recipe.json`)."""
    msgs = sc_gc_prompt(input_sentence, "sc")
    assistant = f"<wrong> {incorrect_word} </wrong> <corrected> {correct_word} </corrected>"
    return {"messages": msgs + [{"role": "assistant", "content": assistant}]}


def render_gc_row(
    input_sentence: str,
    incorrect_word: str,
    correct_word: str,
    exemplars: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Render one GC training example in axolotl chat format. See render_sc_row re: exemplars."""
    msgs = sc_gc_prompt(input_sentence, "gc")
    assistant = f"<wrong> {incorrect_word} </wrong> <corrected> {correct_word} </corrected>"
    return {"messages": msgs + [{"role": "assistant", "content": assistant}]}


def render_qa_row(
    context: str,
    question: str,
    possible_answers: Dict[str, str],
    correct_answer_num: int,
    exemplars: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Render one QA training example mirroring the official QA YAML doc_to_text.

    User prompt template:
        {context}

        Question:
        {question}

        Possible answers:
        {num}: {text}   (for each option, sorted by key)
        ...

        Answer:

    Assistant = 1-based correct_answer_num as a string.
    """
    lines = [context, "", "Question:", question, "", "Possible answers:"]
    for key in sorted(possible_answers.keys(), key=lambda k: int(k)):
        lines.append(f"{key}: {possible_answers[key]}")
    lines.append("")
    lines.append("Answer:")
    content = "\n".join(lines)
    user_msg = {"role": "user", "content": content}
    assistant_msg = {"role": "assistant", "content": str(correct_answer_num)}
    return {"messages": [user_msg, assistant_msg]}


def render_mr_row(
    question: str,
    answer: str,
    steps: str = "",
    exemplars: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Render one MR training example.

    User turn via mr_prompt(question) (exemplars unused -- see render_sc_row).
    Assistant = clean step body + '<answer> {answer} </answer>'.
    """
    msgs = mr_prompt(question)
    body = steps.strip()
    assistant = (f"{body}\n<answer> {answer} </answer>").lstrip()
    return {"messages": msgs + [{"role": "assistant", "content": assistant}]}
