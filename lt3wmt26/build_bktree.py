#!/usr/bin/env python3
"""BK-tree builder for the SC engine's edit-distance candidate generator (`sc.bktree` in
configs/*.yaml; consumed by `lt3wmt26.sc_gc_engine.Engine.candidates`'s `self.bk.range(word, k)`
calls, unpickled from `Resources.bktree_path` in `lt3wmt26.generate`).

Ported from `src/bets/b2_hybrid/candidates.py` (`BKTree`, `levenshtein`, `load_dict_stems`,
`build_bktree_cached`): a Burkhard-Keller tree over the hunspell dictionary's STEM list, not the
affix-expanded surface forms -- the internal design note: soblex hsb's full affix expansion is
194M entries / 2.7 GB, too large to index, and `Hunspell.paradigm` (see `lt3wmt26.lexicon`)
already covers inflected-form candidates, so the BK-tree only needs to cover unknown-word
edit-distance neighbours among stems.

`levenshtein` and `BKTree.add`/`.range` are algorithmically unchanged from the internal source.
`load_dict_stems` reads the same hunspell `.dic` format `lt3wmt26.lexicon.Hunspell._parse_dic`
reads (count header + `stem[/flags]` lines), kept as a separate, minimal reader here since the
BK-tree only needs the bare stem, not the affix-expanded forms.

A pickled `BKTree` from this module is exactly what `lt3wmt26.sc_gc_engine.Engine.__init__`
unpickles into `self.bk` and calls `self.bk.range(word, k)` on: `range` returns a list of plain
strings, which the engine's `c[0] if isinstance(c, (tuple, list)) else c` handles directly.

CLI: builds one `<lang>.bktree.pkl` per language from `--dict-dir/<lang>.dic` (the layout
`data/fetch_dicts.py` writes), written to `--out/<lang>.bktree.pkl` -- the layout
`lt3wmt26.generate.Resources.bktree_path` expects. Run via `scripts/setup.sh`, after
`data/fetch_dicts.py`, since the tree is built from the fetched lexicon wordlists.
"""
import argparse
import os
import pickle


def levenshtein(a, b, cap=None):
    """Edit distance with optional early-exit `cap`.

    When `cap` is set, returns `cap + 1` as soon as any row's minimum exceeds `cap` -- much
    faster than the full DP for near-matches.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > (cap if cap is not None else float("inf")):
        return (cap + 1) if cap is not None else abs(len(a) - len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        row_min = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            if cur[j] < row_min:
                row_min = cur[j]
        if cap is not None and row_min > cap:
            return cap + 1
        prev = cur
    return prev[-1]


class BKTree:
    """Burkhard-Keller tree over strings under Levenshtein distance.

    Lets `range(query, k)` find every indexed word within edit distance `k` of `query` in
    O(k^2 . log n) time (empirically) instead of the O(n) brute force a ~100k-stem dictionary
    would otherwise need per flagged word.
    """

    __slots__ = ("_root",)

    def __init__(self, words=None):
        self._root = None
        if words:
            for w in words:
                self.add(w)

    def add(self, word):
        if self._root is None:
            self._root = (word, {})
            return
        node = self._root
        while True:
            d = levenshtein(word, node[0])
            if d == 0:
                return
            if d in node[1]:
                node = node[1][d]
            else:
                node[1][d] = (word, {})
                return

    def range(self, query, k):
        """All indexed words within Levenshtein <= k of `query`."""
        if self._root is None:
            return []
        out = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = levenshtein(query, node[0])
            if d <= k:
                out.append(node[0])
            lo, hi = max(0, d - k), d + k
            children = node[1]
            for key in range(lo, hi + 1):
                if key in children:
                    stack.append(children[key])
        return out


def load_dict_stems(dic_path, max_stems=None):
    """Load stems from a hunspell .dic file: count header, then one `stem[/flags]` entry per
    line. Only the stem part is kept -- affix expansion is `lt3wmt26.lexicon.Hunspell`'s job."""
    with open(dic_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    start = 1 if lines and lines[0].strip().isdigit() else 0
    stems = []
    for line in lines[start:]:
        s = line.strip()
        if not s:
            continue
        slash = s.find("/")
        stem = s if slash == -1 else s[:slash]
        stem = stem.split("\t")[0].strip()
        if stem:
            stems.append(stem)
        if max_stems is not None and len(stems) >= max_stems:
            break
    return stems


def build_bktree(dic_path, out_path, max_stems=None):
    """Build a BKTree over `dic_path`'s stems and pickle it to `out_path`.

    Returns `(tree, n_stems)`."""
    stems = load_dict_stems(dic_path, max_stems=max_stems)
    # Pickle against the canonical import path, never `__main__`. When this file runs as a
    # script (`python -m lt3wmt26.build_bktree`, as scripts/setup.sh does), the module-level
    # BKTree class lives in `__main__`, and a pickle of it cannot be loaded by any other
    # process -- sc_gc_engine.Engine's bare `pickle.load` would fail with
    # "Can't get attribute 'BKTree' on <module '__main__'>".
    from lt3wmt26.build_bktree import BKTree as _CanonicalBKTree
    tree = _CanonicalBKTree(stems)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(tree, fh)
    return tree, len(stems)


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dict-dir", required=True,
                     help="Directory of <lang>.dic files (data/fetch_dicts.py --out).")
    ap.add_argument("--out", required=True,
                     help="Directory to write <lang>.bktree.pkl into -- resources.yaml's "
                          "bktree_dir should point here.")
    ap.add_argument("--langs", default="hsb,dsb",
                     help="Comma-separated language codes to build (default: hsb,dsb).")
    ap.add_argument("--max-stems", type=int, default=None,
                     help="Optional cap on stems indexed per language (debugging only; the "
                          "shipped primary indexes the full dictionary).")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    for lang in args.langs.split(","):
        dic_path = os.path.join(args.dict_dir, f"{lang}.dic")
        out_path = os.path.join(args.out, f"{lang}.bktree.pkl")
        _tree, n_stems = build_bktree(dic_path, out_path, max_stems=args.max_stems)
        print(f"[build_bktree] {lang}: {n_stems} stems -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
