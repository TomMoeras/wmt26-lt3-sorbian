#!/usr/bin/env python3
"""Hunspell affix expansion: full-paradigm candidate generation for GC, and
MAP-based diacritic classes for SC.

Ported from `scripts/phase6/hunspell_paradigm.py`.

The audit found 98% of missed GC errors are VALID word-forms in the wrong inflection with
the correction inside the SAME lemma's paradigm (kricznami->kricznach, prioritami->
prioritach, najdlesi->najdlesa). So: expand every dictionary stem through its affix flags
into full surface paradigms, then for a surface word W offer W's co-paradigm forms as
correction candidates.

No `unmunch` binary is available, so the .aff/.dic pair is expanded here directly.
Sources (MANIFEST'd, public, licensed): soblex hsb 3.09.18, dsb_spell 1.5.0.
"""
import re, collections


class Hunspell:
    def __init__(self, aff_path, dic_path, max_forms_per_lemma=400):
        self.flag_mode = "char"
        self.sfx, self.pfx, self.maps = {}, {}, []
        self._parse_aff(aff_path)
        self.lemma_forms, self.form_lemmas = [], collections.defaultdict(set)
        self._parse_dic(dic_path, max_forms_per_lemma)

    def _parse_aff(self, path):
        cur = None
        for line in open(path, encoding="utf-8", errors="ignore"):
            t = line.rstrip("\n").split("#")[0].strip()
            if not t:
                continue
            p = t.split()
            if p[0] == "FLAG" and len(p) > 1:
                self.flag_mode = p[1]
            elif p[0] == "MAP" and len(p) == 2 and not p[1].isdigit():
                self.maps.append(p[1])
            elif p[0] in ("SFX", "PFX"):
                tbl = self.sfx if p[0] == "SFX" else self.pfx
                if len(p) >= 4 and p[2] in ("Y", "N"):        # header line
                    cur = (p[0], p[1]); tbl.setdefault(p[1], [])
                elif len(p) >= 4:                              # rule line
                    flag, strip, add = p[1], p[2], p[3]
                    cond = p[4] if len(p) > 4 else "."
                    strip = "" if strip == "0" else strip
                    add = "" if add == "0" else add
                    add = add.split("/")[0]                    # drop continuation flags
                    try:
                        rx = re.compile(cond + "$" if p[0] == "SFX" else "^" + cond)
                    except re.error:
                        rx = None
                    tbl.setdefault(flag, []).append((strip, add, rx))

    def _flags(self, s):
        if not s:
            return []
        return [f for f in s.split(",") if f] if self.flag_mode == "num" else list(s)

    def _parse_dic(self, path, cap):
        with open(path, encoding="utf-8", errors="ignore") as fh:
            lines = fh.read().split("\n")
        start = 1 if lines and lines[0].strip().isdigit() else 0
        for ln in lines[start:]:
            ln = ln.strip()
            if not ln:
                continue
            if "/" in ln:
                stem, fl = ln.rsplit("/", 1)
            else:
                stem, fl = ln, ""
            stem = stem.split("\t")[0].strip()
            if not stem:
                continue
            forms = {stem}
            for flag in self._flags(fl.split("\t")[0].strip()):
                for strip, add, rx in self.sfx.get(flag, []):
                    if rx and not rx.search(stem):
                        continue
                    if strip and not stem.endswith(strip):
                        continue
                    forms.add((stem[:-len(strip)] if strip else stem) + add)
                for strip, add, rx in self.pfx.get(flag, []):
                    if rx and not rx.search(stem):
                        continue
                    if strip and not stem.startswith(strip):
                        continue
                    forms.add(add + (stem[len(strip):] if strip else stem))
                if len(forms) > cap:
                    break
            idx = len(self.lemma_forms)
            fl_list = sorted(forms)[:cap]
            self.lemma_forms.append(fl_list)
            for f in fl_list:
                self.form_lemmas[f].add(idx)
                self.form_lemmas[f.lower()].add(idx)

    def paradigm(self, word, cap):
        """All co-paradigm surface forms of `word` (excluding itself).

        cap: maximum forms to return; task-specific value supplied by the caller's config (no default: this bounds the candidate space and was chosen per task internally)."""
        out = set()
        for idx in list(self.form_lemmas.get(word, set()) or
                        self.form_lemmas.get(word.lower(), set())):
            out.update(self.lemma_forms[idx])
            if len(out) > cap * 4:
                break
        out.discard(word)
        return sorted(out)[:cap]

    def known(self, word):
        return word in self.form_lemmas or word.lower() in self.form_lemmas

    def diacritic_variants(self, word):
        """MAP-driven candidates (soblex MAP: n/n, s/s, c/c, r/r, z/z, o/o, e/e, l/l ...).
        The audit: p2 missed ALL ~10 diacritic-only errors."""
        out = set()
        for i, ch in enumerate(word):
            for grp in self.maps:
                if ch in grp:
                    for alt in grp:
                        if alt != ch:
                            out.add(word[:i] + alt + word[i + 1:])
        out.discard(word)
        return sorted(out)
