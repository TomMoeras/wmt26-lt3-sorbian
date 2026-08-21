"""Phase 2 MR (maths reasoning) synthesis -- MT-translate English seeds. This is the "MR
training messages (Sorbian Q -> English CoT)" generator described in
`data/regeneration_recipe.json` (there, a since-lost cluster CLI wrapper
`staged/phase5/mr_data/translate_mr_seeds.py`; the underlying translate/render logic below is
what that wrapper called).

Ported from `src/data/synth/mr.py`. Two changes from the original:
  1. The organizer-harness prompt/extraction imports
     (`src.eval.official.prompts.mt_messages` / `src.eval.official.extract.mt_translation`)
     are replaced with `lt3wmt26.generate.mt_prompt` / `mt_translation` -- this repo's own
     self-contained port of the same prompt/extraction logic (see `lt3wmt26/generate.py`).
  2. The hard-coded cluster `DATA_ROOT` default is removed (see `main()`); `--data-root` (or
     the `DATA_ROOT` env var) is required, no fallback.

Two layers:

1. Pure helpers (TDD, no GPU):
   - parse_seed(example, source)      -> {question_en, answer, rationale_en?} | None
   - answer_is_safe(answer)           -> bool  (accept int/fraction/short-LaTeX, reject prose)
   - render_mr_training(q_sorbian, steps_sorbian, answer) -> {"messages":[user, assistant]}

   The numeric / LaTeX answer is kept VERBATIM; the assistant turn ends EXACTLY in
   "<answer> {answer} </answer>" via synth.common.render_mr_row / lt3wmt26.generate.mr_prompt.

2. A GPU translation driver (load_seeds / translate_seeds / build) that loads the
   Track A vLLM checkpoint and translates the English *question* (and rationale where
   present) into Upper/Lower Sorbian with the trained MT prompt, leaving math / the
   answer untranslated. Best-effort, probe-gated.

   TRANSLATION PATH (decision, documented):
   Track A was trained de/hsb/dsb, so en->Sorbian is technically out-of-distribution.
   We default to DIRECT en->hsb / en->dsb via lt3wmt26.generate.mt_prompt(src_name="English",
   tgt_name="Upper Sorbian"/"Lower Sorbian") with greedy decoding, and let the later
   probe judge quality. The 50-seed smoke gates this: if direct output is visibly
   degenerate (empty / English echo), the operator falls back to prose-only translation
   (numbers/equations kept untranslated).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from training.generators.synth.common import render_mr_row

# ---------------------------------------------------------------------------
# Seed parsing
# ---------------------------------------------------------------------------

# GSM8K: answer text ends with "#### <int>" and contains <<calc>> annotations.
_GSM_FINAL = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_CALC_ANNOT = re.compile(r"<<[^>]*>>")
# Hendrycks MATH / MATH-Hard: final answer in \boxed{...} inside the solution.
_BOXED_KW = "\\boxed"


def _extract_boxed(text: str) -> Optional[str]:
    """Return the brace-balanced content of the LAST \\boxed{...} in text, or None."""
    idx = text.rfind(_BOXED_KW)
    if idx == -1:
        return None
    i = idx + len(_BOXED_KW)
    # Skip optional whitespace then require an opening brace.
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j].strip()
    return None  # unbalanced


def parse_seed(example: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    """Parse one English seed into {question_en, answer, rationale_en}.

    source in {"gsm8k", "math", "math_hard"}. Keeps the numeric / LaTeX answer
    verbatim. Returns None when no clean answer can be extracted.

    - gsm8k:  question -> question_en; integer after "####" -> answer (commas dropped);
              the answer body (calc <<...>> annotations + the #### line stripped) ->
              rationale_en.
    - math / math_hard:  problem -> question_en; LAST \\boxed{...} content -> answer
              (verbatim LaTeX); solution -> rationale_en.
    """
    if source == "gsm8k":
        q = (example.get("question") or "").strip()
        ans_text = example.get("answer") or ""
        m = _GSM_FINAL.search(ans_text)
        if not q or not m:
            return None
        raw = m.group(1).strip()
        # Integer answer: strip thousands separators / spaces, keep sign.
        cleaned = raw.replace(",", "").replace(" ", "")
        if not re.fullmatch(r"-?\d+", cleaned):
            return None
        answer = cleaned
        # Rationale: drop the #### line, strip <<...>> calculator annotations.
        body = _GSM_FINAL.sub("", ans_text).strip()
        rationale = _CALC_ANNOT.sub("", body).strip()
        return {"question_en": q, "answer": answer, "rationale_en": rationale}

    if source in ("math", "math_hard"):
        q = (example.get("problem") or "").strip()
        sol = example.get("solution") or ""
        boxed = _extract_boxed(sol)
        if not q or boxed is None or not boxed.strip():
            return None
        return {"question_en": q, "answer": boxed.strip(), "rationale_en": sol.strip()}

    raise ValueError(f"unknown MR seed source: {source}")


# ---------------------------------------------------------------------------
# Answer safety filter
# ---------------------------------------------------------------------------

# Allowed characters for a "clean" short math/LaTeX answer (no prose words).
_SAFE_CHARS = re.compile(r"^[0-9A-Za-z\\{}\[\]()+\-*/.,;:=<>^_|!~ \t°πφ√∞∪∩×÷±√$]+$")
# A run of >= 6 alphabetic prose-like words signals a sentence, not a math answer.
_WORD = re.compile(r"[A-Za-z]{2,}")


def answer_is_safe(answer: str) -> bool:
    """Accept integers, simple fractions, intervals/sets, and short LaTeX
    (\\boxed/\\frac/etc.); reject empty strings and free-text prose."""
    if answer is None:
        return False
    a = answer.strip()
    if not a:
        return False
    # Plain integer or simple decimal/fraction -> always safe.
    if re.fullmatch(r"-?\d[\d,]*(\.\d+)?", a):
        return True
    if re.fullmatch(r"-?\d+\s*/\s*-?\d+", a):
        return True
    # LaTeX / interval / set notation: bounded length and limited charset.
    if len(a) > 120:
        return False
    if not _SAFE_CHARS.match(a):
        return False
    # Reject prose: many bare alphabetic words with no math markers.
    has_math = any(ch in a for ch in "\\{}[]()/^_=+<>|") or any(
        ch in a for ch in "πφ√∞∪∩×÷±°"
    )
    words = _WORD.findall(a)
    if not has_math and len(words) >= 4:
        return False
    return True


# ---------------------------------------------------------------------------
# Training-row renderer
# ---------------------------------------------------------------------------

def render_mr_training(
    question_sorbian: str,
    steps_sorbian: str,
    answer: str,
    exemplars: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Render one MR training example in axolotl chat format.

    User turn = official MR prompt over the Sorbian question (via
    synth.common.render_mr_row / lt3wmt26.generate.mr_prompt); assistant = clean
    step-by-step body ending EXACTLY in "<answer> {answer} </answer>". The answer is
    inserted verbatim.
    """
    return render_mr_row(
        question=question_sorbian,
        answer=answer,
        steps=steps_sorbian or "",
        exemplars=exemplars or [],
    )


# ===========================================================================
# GPU translation driver (NOT exercised by the pure-helper tests)
# ===========================================================================

# Per-source seed file locations under <data_root>/sorbian/mr/english-seeds.
# gsm8k + math are parquet (read with pyarrow in the CPU export step); math-hard is
# pretty-printed JSON arrays. We never import pyarrow in the GPU env: a CPU prep step
# (load_seeds_to_jsonl) materialises parsed seeds to JSONL first.

_SEED_SUBDIR = "sorbian/mr/english-seeds"

_MATH_SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)


def _seeds_root(data_root: str) -> Path:
    return Path(data_root) / _SEED_SUBDIR


def load_seeds_to_jsonl(
    data_root: str,
    out_path: str,
    n_low: int,
    n_medium: int,
    seed: int = 42,
) -> Dict[str, int]:
    """CPU prep: parse English seeds -> parsed-seed JSONL (needs pyarrow for parquet).

    Writes rows: {"source","tier","question_en","answer","rationale_en"} where
    tier is "low" (GSM8K integer) or "medium" (MATH/MATH-Hard LaTeX). Samples a
    balanced n_low + n_medium with answer_is_safe filtering. Returns realised counts.
    """
    import random as _random

    import pyarrow.parquet as pq

    rng = _random.Random(seed)
    root = _seeds_root(data_root)

    # ---- low tier: GSM8K main train ----
    low: List[Dict[str, Any]] = []
    gsm_path = root / "gsm8k" / "main" / "train-00000-of-00001.parquet"
    for ex in pq.read_table(gsm_path).to_pylist():
        parsed = parse_seed(ex, "gsm8k")
        if parsed and answer_is_safe(parsed["answer"]):
            parsed.update(source="gsm8k", tier="low")
            low.append(parsed)

    # ---- medium tier: MATH (parquet) + MATH-Hard (json) ----
    medium: List[Dict[str, Any]] = []
    for subj in _MATH_SUBJECTS:
        p = root / "math" / subj / "train-00000-of-00001.parquet"
        if not p.exists():
            continue
        for ex in pq.read_table(p).to_pylist():
            parsed = parse_seed(ex, "math")
            if parsed and answer_is_safe(parsed["answer"]):
                parsed.update(source="math", tier="medium")
                medium.append(parsed)
    mh_dir = root / "math-hard" / "train"
    if mh_dir.exists():
        for jf in sorted(mh_dir.glob("*.jsonl")):
            for ex in json.loads(jf.read_text(encoding="utf-8")):
                parsed = parse_seed(ex, "math_hard")
                if parsed and answer_is_safe(parsed["answer"]):
                    parsed.update(source="math_hard", tier="medium")
                    medium.append(parsed)

    rng.shuffle(low)
    rng.shuffle(medium)
    low = low[:n_low]
    medium = medium[:n_medium]

    rows = low + medium
    rng.shuffle(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"low": len(low), "medium": len(medium), "total": len(rows)}


def _read_parsed_seeds(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Track A LANG_NAMES mirror (avoid importing the MT task module unnecessarily).
_TGT_NAME = {"hsb": "Upper Sorbian", "dsb": "Lower Sorbian"}


def _mt_prompt(text: str, tgt_lang: str) -> List[Dict[str, str]]:
    """Direct en->Sorbian MT prompt (Track A trained format, no exemplars)."""
    from lt3wmt26.generate import mt_prompt
    return mt_prompt(text, src_name="English", tgt_name=_TGT_NAME[tgt_lang], exemplars=[])


def translate_and_build(
    parsed_jsonl: str,
    out_dir: str,
    tgt_lang: str,
    model_path: str,
    max_model_len: int = 4096,
    translate_rationale: bool = True,
    prose_only: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """GPU: load Track A vLLM, translate question (and rationale) en->Sorbian, write
    MR training rows.

    Writes <out_dir>/{train.jsonl, pool.jsonl, meta.json}. The answer is never
    translated. prose_only is the documented fallback (kept for the operator; the
    default direct path is prose_only=False).

    Returns the meta dict.
    """
    from vllm import LLM, SamplingParams

    seeds = _read_parsed_seeds(parsed_jsonl)
    if limit is not None:
        seeds = seeds[:limit]

    # Build the batch of MT prompts: question for every seed, rationale optionally.
    q_prompts = [_mt_prompt(s["question_en"], tgt_lang) for s in seeds]

    llm = LLM(model=model_path, runner="generate", trust_remote_code=True,
              max_model_len=max_model_len, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1024)
    chat_kwargs = {"chat_template_kwargs": {"enable_thinking": False}}

    def _run(prompts):
        outs = llm.chat(prompts, sp, use_tqdm=True, **chat_kwargs)
        return [_clean_mt(o.outputs[0].text, tgt_lang) for o in outs]

    q_sorbian = _run(q_prompts)

    r_sorbian: List[str] = ["" for _ in seeds]
    if translate_rationale and not prose_only:
        # Translate rationales that are present and short enough to be worthwhile.
        idx, r_prompts = [], []
        for i, s in enumerate(seeds):
            rat = (s.get("rationale_en") or "").strip()
            if rat and len(rat) <= 2000:
                idx.append(i)
                r_prompts.append(_mt_prompt(rat, tgt_lang))
        if r_prompts:
            translated = _run(r_prompts)
            for i, t in zip(idx, translated):
                r_sorbian[i] = t

    # Render rows. A row is kept only if the Sorbian question is non-degenerate.
    rows: List[Dict[str, Any]] = []
    kept_meta: List[Dict[str, Any]] = []
    n_degenerate = 0
    for s, q_s, r_s in zip(seeds, q_sorbian, r_sorbian):
        if _is_degenerate(q_s, s["question_en"]):
            n_degenerate += 1
            continue
        steps = r_s if (r_s and not _is_degenerate(r_s, s.get("rationale_en", ""))) else ""
        row = render_mr_training(q_s, steps, s["answer"])
        rows.append(row)
        kept_meta.append({"source": s["source"], "tier": s["tier"], "answer": s["answer"]})

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 90/10 train/pool split (deterministic by input order).
    n_pool = max(1, len(rows) // 10) if rows else 0
    pool_rows, train_rows = rows[:n_pool], rows[n_pool:]
    _write_jsonl(out / "train.jsonl", train_rows)
    _write_jsonl(out / "pool.jsonl", pool_rows)

    meta = {
        "lang": tgt_lang,
        "translation_path": "prose_only" if prose_only else "direct_en_sorbian",
        "model_path": model_path,
        "translate_rationale": translate_rationale and not prose_only,
        "seeds_in": len(seeds),
        "rows_out": len(rows),
        "degenerate_dropped": n_degenerate,
        "train_rows": len(train_rows),
        "pool_rows": len(pool_rows),
        "tier_counts": _count_tiers(kept_meta),
    }
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


def _clean_mt(text: str, tgt_lang: str) -> str:
    """Apply the ported MT extractor (strips <think>, tolerates raw output)."""
    from lt3wmt26.generate import mt_translation
    return mt_translation(text, tgt_lang).strip()


def _is_degenerate(out_text: str, src_text: str) -> bool:
    """Heuristic degeneracy check: empty, or an exact echo of the English source."""
    o = (out_text or "").strip()
    if not o:
        return True
    if src_text and o.casefold() == src_text.strip().casefold():
        return True
    return False


def _count_tiers(metas: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {"low": 0, "medium": 0}
    for m in metas:
        out[m["tier"]] = out.get(m["tier"], 0) + 1
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 MR synthesis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("export-seeds",
                         help="CPU: parse English seeds -> balanced parsed-seed JSONL (needs pyarrow)")
    exp.add_argument("--data-root", required=True,
                     help="organizer-distribution root holding sorbian/mr/english-seeds/ "
                          "(no default -- see module docstring)")
    exp.add_argument("--out", required=True)
    exp.add_argument("--n-low", type=int, default=3000)
    exp.add_argument("--n-medium", type=int, default=3000)
    exp.add_argument("--seed", type=int, default=42)

    tr = sub.add_parser("translate",
                        help="GPU: translate parsed seeds en->Sorbian and write MR training rows")
    tr.add_argument("--parsed", required=True, help="parsed-seed JSONL from export-seeds")
    tr.add_argument("--out-dir", required=True)
    tr.add_argument("--lang", required=True, choices=("hsb", "dsb"))
    tr.add_argument("--model", required=True, help="Track A vLLM checkpoint dir")
    tr.add_argument("--max-model-len", type=int, default=4096)
    tr.add_argument("--no-rationale", action="store_true",
                    help="translate the question only (skip the step-by-step rationale)")
    tr.add_argument("--prose-only", action="store_true",
                    help="DOCUMENTED FALLBACK: question-only, no rationale (used if direct is degenerate)")
    tr.add_argument("--limit", type=int, default=None, help="cap #seeds (smoke)")

    args = parser.parse_args(argv)

    if args.cmd == "export-seeds":
        counts = load_seeds_to_jsonl(args.data_root, args.out, args.n_low, args.n_medium, args.seed)
        print(json.dumps(counts, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "translate":
        meta = translate_and_build(
            parsed_jsonl=args.parsed,
            out_dir=args.out_dir,
            tgt_lang=args.lang,
            model_path=args.model,
            max_model_len=args.max_model_len,
            translate_rationale=not args.no_rationale,
            prose_only=args.prose_only,
            limit=args.limit,
        )
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
