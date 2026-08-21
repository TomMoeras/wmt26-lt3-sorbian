"""Phase 2 synthesis driver -- generates the "SC/GC/QA synthetic (v5 mix)" artifact named in
`data/regeneration_recipe.json` (there: `python -m src.data.synth.build generate --task
{sc,gc,qa} --lang {hsb,dsb} --config configs/data/phase2_synth_full.yaml`).

Ported from `src/data/synth/build.py`. The only cluster-specific value in the original was a
hard-coded `DATA_ROOT` default (`/dodrio/scratch/projects/.../wmt26_fuzzy_all/data`); per the
port rules that default is removed here -- `--data-root` (or the `DATA_ROOT` env var) is
required, with no fallback, so a caller cannot silently synthesise against a path that happens
to exist on someone else's machine.

CLI:
    python -m training.generators.synth.build generate \\
        --task {sc,gc,qa} --lang {hsb,dsb} \\
        --config training/configs/sc_gc_qa_synth_full.yaml [--limit N] \\
        --data-root organizer_data/Sorbian

Loads monolingual sentences via common.load_mono (length-filtered + deduped),
dev-excludes against the task dev set + all MT dev src/tgt strings, runs the task
synthesiser (sc/gc/qa), renders rows via common.render_*, holds out a pool, and
writes under <out-root>/<task>/<lang>/:

    train.jsonl   {"messages":[...]} rows for training
    pool.jsonl    held-out rows usable as few-shot source
    meta.json     counts, seed, params, per-source line counts, #dev-excluded

Deterministic from the config seed. MR is OUT OF SCOPE (see synth/mr.py).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from training.generators.synth import common, sc, gc, qa

TASKS = ("sc", "gc", "qa")
LANGS = ("hsb", "dsb")


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    # training/generators/synth/build.py -> repo root is 3 parents up.
    return Path(__file__).resolve().parents[3]


def _expand(path: str, repo: Path, data_root: str) -> str:
    return (path.replace("${REPO}", str(repo))
                .replace("${DATA_ROOT}", data_root))


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_data_root(cli_value: Optional[str]) -> str:
    data_root = cli_value or os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit(
            "build.py: no --data-root given and DATA_ROOT is not set. This must point at the "
            "organizer distribution's Sorbian data directory (see data/fetch_organizer_data.py "
            "and resources.yaml.template's mono_dir) -- there is no cluster-path default."
        )
    return data_root


# ---------------------------------------------------------------------------
# Dev-exclusion set
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dev_set(cfg: Dict[str, Any], task: str, lang: str, repo: Path) -> Set[str]:
    """Build the dev-exclusion string set: task dev original/input sentences +
    all MT dev src/tgt strings (de/hsb/dsb)."""
    dev: Set[str] = set()

    # Task dev set for this lang: SC/GC original_sentence + input_sentence; QA context.
    dev_path = repo / cfg["dev_sets"][task][lang]
    if dev_path.exists():
        for row in _read_jsonl(dev_path):
            for key in ("original_sentence", "input_sentence", "context"):
                v = row.get(key)
                if isinstance(v, str) and v:
                    dev.add(v)

    # MT dev files: fold in every string value of the language fields.
    for mt_file in cfg.get("mt_dev_files", []):
        p = repo / mt_file
        if not p.exists():
            continue
        for row in _read_jsonl(p):
            for k in ("de", "hsb", "dsb"):
                v = row.get(k)
                if isinstance(v, str) and v:
                    dev.add(v)

    return dev


# ---------------------------------------------------------------------------
# QA distractor vocabulary (mono token frequencies)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def build_distractor_vocab(sentences: List[str], top_n: int) -> List[str]:
    """Most-frequent alphabetic content tokens (len>=4, not a stopword) from mono."""
    counter: Counter = Counter()
    for s in sentences:
        for tok in _TOKEN_RE.findall(s):
            if len(tok) >= 4 and tok.isalpha() and tok.lower() not in qa._STOPWORDS:
                counter[tok] += 1
    return [w for w, _ in counter.most_common(top_n)]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(task: str, lang: str, config_path: str,
             limit: Optional[int] = None,
             out_root: Optional[str] = None,
             data_root: Optional[str] = None) -> Dict[str, Any]:
    """Run synthesis for one task/lang. Returns the meta dict (also written to disk).

    out_root: directory under which `<task>/<lang>/` is written. Required (a caller
    running a smoke test MUST pass a tmp out_root so it never clobbers a real dataset --
    a fixture run once truncated hsb to 8 rows in the working repo).
    data_root: organizer-distribution root (see _resolve_data_root); required, no default."""
    repo = _repo_root()
    data_root = _resolve_data_root(data_root)
    cfg = _load_config(config_path)
    seed = int(cfg.get("seed", 42))
    pool_size = int(cfg.get("pool_size", 2000))

    # Resolve ranked mono source paths.
    raw_sources = cfg["mono_sources"][lang]
    sources = [_expand(s, repo, data_root) for s in raw_sources]
    source_present = [(s, Path(s).exists()) for s in sources]

    # Target train volume (per-task per-lang); under --limit, shrink to keep smoke cheap.
    task_cfg = cfg["tasks"][task]
    train_volume = int(task_cfg["volume"][lang])
    if limit is not None:
        train_volume = min(train_volume, limit)
        pool_size = min(pool_size, max(0, limit // 5))

    # How many mono sentences we need: enough for train + pool, with a synthesis-failure
    # margin (some sentences yield no error / no cloze). Cap loading to avoid reading
    # the entire corpus when we only need a fraction.
    needed = train_volume + pool_size
    load_cap = max(needed * 3, needed + 5000)

    # Load mono (length-filtered + deduped), then dev-exclude.
    raw_mono = common.load_mono(lang, sources, cap=load_cap, seed=seed)
    dev_set = build_dev_set(cfg, task, lang, repo)
    before_excl = len(raw_mono)
    mono = common.exclude_dev(raw_mono, dev_set)
    n_dev_excluded = before_excl - len(mono)

    # Deterministic shuffle of the mono pool so source order doesn't bias selection.
    rng = random.Random(seed)
    rng.shuffle(mono)

    # Synthesise rows (task-specific dicts), then render to messages.
    syn_rng = random.Random(seed)
    rendered_train: List[Dict[str, Any]] = []
    rendered_pool: List[Dict[str, Any]] = []
    raw_train: List[Dict[str, Any]] = []   # task dicts for pool few-shot fields
    raw_pool: List[Dict[str, Any]] = []
    clean_count = 0
    error_count = 0

    if task in ("sc", "gc"):
        p_error = float(task_cfg.get("p_error", 0.5))
        # build_rows consumes one sentence per row; give it enough sentences.
        n_sent = min(len(mono), needed)
        sents = mono[:n_sent]
        if task == "sc":
            rows = sc.build_rows(sents, p_error=p_error, rng=syn_rng)
            render = common.render_sc_row
        else:
            rows = gc.build_rows(sents, lang=lang, p_error=p_error, rng=syn_rng)
            render = common.render_gc_row

        for r in rows:
            if r["incorrect_word"] == "CORRECT":
                clean_count += 1
            else:
                error_count += 1

        # Split into train / pool.
        pool_rows = rows[:pool_size]
        train_rows = rows[pool_size:pool_size + train_volume]
        for r in train_rows:
            rendered_train.append(render(r["input_sentence"], r["incorrect_word"],
                                         r["correct_word"], exemplars=[]))
            raw_train.append(r)
        for r in pool_rows:
            rendered_pool.append(render(r["input_sentence"], r["incorrect_word"],
                                        r["correct_word"], exemplars=[]))
            raw_pool.append(r)

    elif task == "qa":
        vocab_n = int(task_cfg.get("distractor_vocab_size", 5000))
        vocab = build_distractor_vocab(mono, vocab_n)
        n_opts_dist = task_cfg.get("n_options_dist", {"3": 1.0})
        opt_vals = [int(k) for k in n_opts_dist.keys()]
        opt_wts = [float(v) for v in n_opts_dist.values()]

        target = train_volume + pool_size
        items: List[Dict[str, Any]] = []
        for passage in mono:
            if len(items) >= target:
                break
            n_options = syn_rng.choices(opt_vals, weights=opt_wts, k=1)[0]
            item = qa.make_cloze(passage, vocab, syn_rng, n_options=n_options)
            if item is not None:
                items.append(item)
                error_count += 1  # every QA item is a "question" (no clean rows)

        pool_items = items[:pool_size]
        train_items = items[pool_size:pool_size + train_volume]
        for it in train_items:
            rendered_train.append(common.render_qa_row(
                it["context"], it["question"], it["possible_answers"],
                it["correct_answer_num"]))
            raw_train.append(it)
        for it in pool_items:
            rendered_pool.append(common.render_qa_row(
                it["context"], it["question"], it["possible_answers"],
                it["correct_answer_num"]))
            raw_pool.append(it)
    else:
        raise ValueError(f"unsupported task: {task}")

    # Write outputs. out_root is required (no production-path default -- see docstring).
    if not out_root:
        raise SystemExit("build.py: --out-root is required (no default output path).")
    out_dir = (Path(out_root) / task / lang).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(out_dir / "train.jsonl", rendered_train)
    _write_jsonl(out_dir / "pool.jsonl", rendered_pool)
    # Also persist raw task dicts alongside (used by joint assembly for few-shot).
    _write_jsonl(out_dir / "pool_raw.jsonl", raw_pool)

    meta = {
        "task": task,
        "lang": lang,
        "seed": seed,
        "params": {
            k: v for k, v in task_cfg.items() if k != "volume"
        } | {"target_train_volume": train_volume, "pool_size": pool_size},
        "limit": limit,
        "mono_sources": [{"path": s, "present": present} for s, present in source_present],
        "mono_loaded": before_excl,
        "mono_load_cap": load_cap,
        "dev_excluded": n_dev_excluded,
        "mono_after_exclusion": len(mono),
        "train_rows": len(rendered_train),
        "pool_rows": len(rendered_pool),
        "clean_rows_total": clean_count,
        "error_rows_total": error_count,
        "clean_fraction_total": (clean_count / (clean_count + error_count)
                                 if (clean_count + error_count) else 0.0),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    return meta


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 synthesis driver")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="generate one task/lang dataset")
    gen.add_argument("--task", required=True, choices=TASKS)
    gen.add_argument("--lang", required=True, choices=LANGS)
    gen.add_argument("--config", required=True)
    gen.add_argument("--limit", type=int, default=None,
                     help="cap train volume for cheap smoke runs")
    gen.add_argument("--out-root", required=True,
                     help="write under <out-root>/<task>/<lang>/")
    gen.add_argument("--data-root", default=None,
                     help="organizer-distribution root (Sorbian data tree); falls back to "
                          "the DATA_ROOT env var, no other default")

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        meta = generate(args.task, args.lang, args.config, limit=args.limit,
                        out_root=args.out_root, data_root=args.data_root)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
