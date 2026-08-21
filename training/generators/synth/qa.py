"""Phase 2 QA synthesis -- templated cloze MCQ from monolingual passages.

Ported from `src/data/synth/qa.py` verbatim (no cluster paths in the original).

Calibration (from the working repo's
docs/superpowers/notes/2026-06-25-phase2-synthesis-calibration.md §3):

- possible_answers: dict with 1-based string-int keys ("1","2",...).
- correct_answer_num: 1-based integer, uniformly distributed (no first-slot bias).
- Option count: A1=2, A2 in {2,3}, B1/B2 mostly 3 (+ 10/11/16-option ordering items
  out of scope for cloze; default n_options=3).
- Context: the passage sentence(s); question: "Which word fits the gap?"
- Distractors from caller-supplied vocab list (morphologically/semantically nearby).

Template (mirrors the official QA YAML doc_to_text / common.render_qa_row):
    {context}

    Question:
    {question}

    Possible answers:
    {num}: {text}
    ...

    Answer:
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

# Sorbian stopwords to avoid masking function words
_STOPWORDS = frozenset({
    "a", "w", "z", "so", "je", "ja", "ty", "wón", "wona", "wono", "my", "wy", "woni",
    "to", "ta", "tón", "ale", "pak", "haj", "ně", "nic", "tule", "tu", "tam",
    "su", "bu", "je", "zo", "na", "po", "při", "wot", "do", "ze", "přez", "za",
    "pod", "nad", "před", "za", "mjez", "ma", "mam", "maš", "maja",
    "how", "pak", "tež", "hišće", "hač", "abo",
    # DSB
    "ga", "ten", "ta",
})


def _is_salient(token: str, min_len: int = 4) -> bool:
    """Return True iff token is a salient content word (long enough, alphabetic, not stopword)."""
    return (
        len(token) >= min_len
        and token.isalpha()
        and token.lower() not in _STOPWORDS
    )


def make_cloze(
    passage: str,
    vocab: List[str],
    rng: random.Random,
    n_options: int = 3,
) -> Optional[Dict[str, Any]]:
    """Create a cloze MCQ item from a passage sentence.

    Picks one salient content token as the answer, masks it in the passage to form
    the context, draws (n_options-1) distractors from vocab, shuffles options so the
    correct answer slot is uniformly distributed, and returns the item dict.

    Args:
        passage: a source sentence used as reading context.
        vocab: candidate distractor words (morphologically/semantically similar pool).
        rng: seeded Random instance.
        n_options: number of answer options (default 3).

    Returns:
        Dict with keys:
            context: passage with answer masked as ___
            question: "Kotre słowo słuša do pralizny?" (cloze frame)
            possible_answers: {"1": text, "2": text, ...} (1-based)
            correct_answer_num: int (1-based)
            answer_text: the correct answer string
        or None if no salient token found or not enough distractors.
    """
    tokens = passage.split()
    candidates = [
        (i, tok) for i, tok in enumerate(tokens)
        if _is_salient(tok)
    ]
    if not candidates:
        return None

    # Randomly pick the answer token
    answer_idx, answer_text = rng.choice(candidates)

    # Build context with answer masked
    masked_tokens = tokens[:]
    masked_tokens[answer_idx] = "___"
    context = " ".join(masked_tokens)

    # Cloze question frame (bilingual-friendly neutral phrasing)
    question = "Kotre słowo słuša do pralizny?"

    # Build distractors: sample from vocab, excluding the answer itself
    distractor_pool = [w for w in vocab if w != answer_text]
    # De-duplicate by casefold
    seen_cf = {answer_text.casefold()}
    unique_distractors = []
    for w in distractor_pool:
        cf = w.casefold()
        if cf not in seen_cf:
            seen_cf.add(cf)
            unique_distractors.append(w)

    n_distractors = n_options - 1
    if len(unique_distractors) < n_distractors:
        # Not enough distractors -- pad with generic fillers if needed
        fillers = ["wóno", "tute", "tamne", "druhe", "dalše", "wšě", "někajke"]
        for f in fillers:
            cf = f.casefold()
            if cf not in seen_cf:
                seen_cf.add(cf)
                unique_distractors.append(f)
            if len(unique_distractors) >= n_distractors:
                break

    if len(unique_distractors) < n_distractors:
        return None  # Still not enough after padding

    distractors = rng.sample(unique_distractors, n_distractors)

    # Build option list and shuffle so correct answer position is uniform
    all_options = [answer_text] + distractors
    rng.shuffle(all_options)

    # Find the (now random) position of the answer
    correct_pos = all_options.index(answer_text) + 1  # 1-based

    possible_answers = {str(i + 1): opt for i, opt in enumerate(all_options)}

    return {
        "context": context,
        "question": question,
        "possible_answers": possible_answers,
        "correct_answer_num": correct_pos,
        "answer_text": answer_text,
    }
