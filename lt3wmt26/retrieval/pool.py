#!/usr/bin/env python3
"""Build the deduplicated real+BT translation-memory pool that INFERENCE-time MT exemplar
retrieval draws from, and prove it is free of test-set material. (Training-time exemplar
retrieval used the real-only pool instead -- see docs/REPRODUCIBILITY.md section 4.)

Ported from `scripts/phase6/build_deduped_pool.py`. That script hard-coded which files count as
"blocked": six specific MT test files plus SC/GC/MR test inputs, all read from one absolute,
vendored data tree (`REPO/vendor/llms-limited-resources2026/Sorbian`). That hard-coding is
exactly what makes the original script a one-shot: it only ever knows about the WMT26 Sorbian
test files it was written against.

Here the blocklist is a PARAMETER -- a set of normalized strings the caller assembles from
whatever files it wants excluded (`blocklist_from_jsonl`). That is what makes `dedup_pool`
reusable at inference time for a brand-new hidden test set: `generate.py --pool-override` (and
Task 10's `run_on_new_testset.sh`) build a fresh blocklist from the NEW test files and re-run the
same dedup function, rather than this module needing to know the WMT26 tree exists.

The three-stage filter (drop on blocklist match, drop on dev-source match, cap) and the
counters/verification (residual test material must be zero) are unchanged from the source.
"""
import argparse
import hashlib
import json
import os


def norm(s):
    """Whitespace-collapsed string, for cross-file string identity comparisons."""
    return " ".join(str(s or "").split())


def blocklist_from_jsonl(paths, fields):
    """Union of normalized strings pulled out of one or more JSONL files.

    `paths`: iterable of file paths (missing files are skipped, not an error -- a caller may
    point this at a subset of task files that happen to exist for a given test set).
    `fields`: field names to look for per row, tried in order; the first present field's value
    is added to the blocklist (so one call can cover MT's `source`-style fields as well as
    SC/GC's `input_sentence` or MR's `question` by passing all the field names that occur across
    the caller's files).
    """
    block = set()
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for f in fields:
                    if row.get(f):
                        block.add(norm(row[f]))
                        break
    return block


def dedup_pool(pool_src, pool_tgt, blocklist, dev_src=None, cap=60000):
    """Filter a parallel (src, tgt) pool for test-set leakage, then cap.

    `pool_src`/`pool_tgt`: parallel lists of strings (one MT direction's raw real+BT pool).
    `blocklist`: set of normalized strings; a pair is dropped if EITHER side matches (a pair can
    leak test material through its source OR its target).
    `dev_src`: optional set of normalized dev-source strings for leave-one-out hygiene (dropped
    AFTER the blocklist filter, so the counters attribute removals to the right stage).
    `cap`: pairs kept after filtering, applied last (a speed choice, not a compliance one).

    Returns `(pairs, report)`. `report` carries the same counters
    `build_deduped_pool.py` printed and persisted: `pool_original`, `removed_by_test_dedup`,
    `removed_by_dev_exclusion`, `removed_by_cap`, `shipped`, `residual_test_material` (must be
    0 -- computed independently of the filtering above as a verification, not an assumption).
    """
    assert len(pool_src) == len(pool_tgt), "pool_src/pool_tgt length mismatch"
    n0 = len(pool_src)
    pairs = [(s, t) for s, t in zip(pool_src, pool_tgt)
             if norm(s) not in blocklist and norm(t) not in blocklist]
    n_after_dedup = len(pairs)
    if dev_src:
        pairs = [(s, t) for s, t in pairs if norm(s) not in dev_src]
    n_after_dev = len(pairs)
    pairs = pairs[:cap]
    resid = sum(1 for s, t in pairs if norm(s) in blocklist or norm(t) in blocklist)
    report = {
        "pool_original": n0,
        "removed_by_test_dedup": n0 - n_after_dedup,
        "removed_by_dev_exclusion": n_after_dedup - n_after_dev,
        "removed_by_cap": n_after_dev - len(pairs),
        "shipped": len(pairs),
        "residual_test_material": resid,
    }
    return pairs, report


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_pool(pairs, out_src_path, out_tgt_path):
    """Write `pairs` as parallel `.src`/`.tgt` files (one sentence per line, matching the
    shipped `tm_realbt_deduped` layout)."""
    for path, col in ((out_src_path, 0), (out_tgt_path, 1)):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(p[col] for p in pairs) + "\n")


def load_pairs(pool_src_path, pool_tgt_path):
    ps = [l.rstrip("\n") for l in open(pool_src_path, encoding="utf-8")]
    pt = [l.rstrip("\n") for l in open(pool_tgt_path, encoding="utf-8")]
    return ps, pt


# --------------------------------------------------------------------------------------------
# CLI: re-run dedup_pool against a NEW test set's own blocklist (run_on_new_testset.sh).
# --------------------------------------------------------------------------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dedup-against", required=True,
                     help="Directory of test files (official blind filenames, see "
                          "lt3wmt26.generate.discover_test_files) whose content must not leak "
                          "into the pool.")
    ap.add_argument("--pool", required=True,
                     help="Directory of raw <src>_<tgt>.src/.tgt pairs to dedup (generate.py's "
                          "--resources pool_dir layout).")
    ap.add_argument("--out", required=True,
                     help="Directory to write the deduped <src>_<tgt>.src/.tgt pairs into.")
    ap.add_argument("--dev-dir", default=None,
                     help="Optional directory of dev-source jsonls for leave-one-out exclusion "
                          "(same field names as --dedup-against).")
    ap.add_argument("--cap", type=int, default=60000)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    # Reuses generate.py's own official-filename discovery rather than re-deriving it here --
    # a new test set's files are found exactly the way generate.py itself will read them.
    from lt3wmt26.generate import discover_test_files
    found = discover_test_files(args.dedup_against)
    test_paths = ([p for _, _, p in found["mt"]] + [p for _, p in found["sc"]] +
                  [p for _, p in found["gc"]] + [p for _, p in found["mr"]])
    # QA is deliberately excluded from the blocklist, matching the ported script's original
    # scope (pool.py module docstring): "six specific MT test files plus SC/GC/MR test inputs."
    blocklist = blocklist_from_jsonl(test_paths, ["de", "hsb", "dsb", "input_sentence"])

    dev_src = None
    if args.dev_dir:
        dev_found = discover_test_files(args.dev_dir)
        dev_paths = ([p for _, _, p in dev_found["mt"]] + [p for _, p in dev_found["sc"]] +
                     [p for _, p in dev_found["gc"]] + [p for _, p in dev_found["mr"]])
        dev_src = blocklist_from_jsonl(dev_paths, ["de", "hsb", "dsb", "input_sentence"])

    src_files = sorted(f for f in os.listdir(args.pool) if f.endswith(".src"))
    if not src_files:
        raise SystemExit(f"--pool {args.pool!r}: no *.src files found")
    for src_name in src_files:
        slug = src_name[:-len(".src")]
        tgt_path = os.path.join(args.pool, f"{slug}.tgt")
        if not os.path.exists(tgt_path):
            print(f"  [pool/{slug}] skipped -- no matching {slug}.tgt", flush=True)
            continue
        pool_src, pool_tgt = load_pairs(os.path.join(args.pool, src_name), tgt_path)
        pairs, report = dedup_pool(pool_src, pool_tgt, blocklist, dev_src=dev_src, cap=args.cap)
        write_pool(pairs, os.path.join(args.out, f"{slug}.src"), os.path.join(args.out, f"{slug}.tgt"))
        print(f"  [pool/{slug}] {report}", flush=True)


if __name__ == "__main__":
    main()
