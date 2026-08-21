#!/usr/bin/env python3
"""Assemble generate.py's per-direction/per-language OCELoT files into the official 5-file
submission layout (wmt26lowres-{01-mt,02-qa,03-sc,04-gc,05-mr}-sb.jsonl).

generate.py (lt3wmt26/generate.py) writes one file per MT direction and per SC/GC/QA/MR
language -- up to 14 files for a full run (6 MT directions + 2 QA + 2 SC + 2 GC + 2 MR) -- via
`package_ocelot.write_task_file`. The organiser-facing submission format instead bundles each
task into a single file (see submissions/*/MANIFEST_sha256.json and
submissions/README.md). That join is explicitly OUT of package_ocelot.py's own scope (its
docstring: "that join + validation is cluster I/O orchestration and is not part of this
module") -- this script is that orchestration, kept in scripts/ rather than lt3wmt26/.

The concatenation order below is not a free choice: it is read off the row order already
present in the shipped submissions/LT3-FullStack-DevTransfer/wmt26lowres-*.jsonl files, and it
matches (by construction) the iteration order generate.py's own MT_DIRS list and
NONMT_PATTERN hsb-then-dsb loop already use -- so each per-file bucket below is simply
"whichever of these raw files exist, concatenated in this fixed order," not a re-derivation of
row content.

A task bucket is skipped (no merged file written) if none of its raw files exist -- this keeps
the script usable against a partial --out-dir, e.g. a new test set that only supplies MT files
(run_on_new_testset.sh)."""
import argparse
import os

# (raw generate.py filename, ...) -> merged official filename, in shipped-submission row order.
BUCKETS = [
    ("wmt26lowres-01-mt-sb.jsonl", [
        "deu-hsb_preds.jsonl", "hsb-deu_preds.jsonl",
        "deu-dsb_preds.jsonl", "dsb-deu_preds.jsonl",
        "hsb-dsb_preds.jsonl", "dsb-hsb_preds.jsonl",
    ]),
    ("wmt26lowres-02-qa-sb.jsonl", ["hsb_qa_preds.jsonl", "dsb_qa_preds.jsonl"]),
    ("wmt26lowres-03-sc-sb.jsonl", ["hsb-sc_preds.jsonl", "dsb-sc_preds.jsonl"]),
    ("wmt26lowres-04-gc-sb.jsonl", ["hsb-gc_preds.jsonl", "dsb-gc_preds.jsonl"]),
    ("wmt26lowres-05-mr-sb.jsonl", ["hsb-mr_preds.jsonl", "dsb-mr_preds.jsonl"]),
]


def assemble(out_dir):
    """Concatenate whichever raw per-direction/per-language files are present in `out_dir`
    into the official 5-file layout, written back into the same directory. Returns the list of
    merged filenames actually written."""
    written = []
    for merged_name, raw_names in BUCKETS:
        present = [n for n in raw_names if os.path.exists(os.path.join(out_dir, n))]
        if not present:
            continue
        with open(os.path.join(out_dir, merged_name), "w", encoding="utf-8") as out_fh:
            for n in present:
                with open(os.path.join(out_dir, n), encoding="utf-8") as in_fh:
                    out_fh.write(in_fh.read())
        written.append(merged_name)
        missing = [n for n in raw_names if n not in present]
        note = f" (missing {missing}, partial test set)" if missing else ""
        print(f"[assemble_ocelot] {merged_name} <- {present}{note}", flush=True)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True,
                     help="Directory generate.py wrote its raw per-direction/per-language "
                          "files into; the merged official files are written into the same "
                          "directory.")
    args = ap.parse_args(argv)
    written = assemble(args.out_dir)
    if not written:
        raise SystemExit(f"[assemble_ocelot] no known raw prediction files found in "
                          f"{args.out_dir!r} -- nothing to assemble")


if __name__ == "__main__":
    main()
