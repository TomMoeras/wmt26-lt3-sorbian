#!/usr/bin/env python3
"""W3 -- the shared SC/GC engine: per-position candidate scoring + ONE calibrated null-margin.

Ported from `scripts/phase6/scgc_engine.py` (candidate generation + sentence scoring,
`Engine.candidates`/`score_batch`/`run_sentence`) and `scripts/phase6/scgc_correct_only.py`
(correction-constrained rescoring, `Engine.constrained_correction`).

For every content-word position we generate candidates, build the single-edit sentence, and
score it with the model's sentence loglikelihood (PLL-style, LM-Critic lineage). The NULL
hypothesis (the unedited sentence) is scored too; ONE dev-tuned margin decides edit-vs-CORRECT.

Generators:
  GC -- full-paradigm expansion (hunspell soblex/dsb-spell). Audit: 98% of missed GC errors
        are valid forms whose correction sits in the SAME lemma's paradigm.
  SC -- dictionary membership + hunspell MAP diacritic classes + BK-tree edit neighbours,
        ranked by a corpus-frequency noisy-channel prior (unigram from the mono corpora).

Margin is SWEPT on dev and the whole curve is reported, so the operating point is fitted,
not eyeballed (the ~50% clean prior makes an un-fitted threshold dangerous).

The lexicon (a `lt3wmt26.lexicon.Hunspell`) is injected by the caller -- this module never
constructs it from a hard-coded dict directory. The BK-tree path, the two paradigm caps
(GC's and SC's -- distinct fitted candidate-space bounds in the source), `batch`, and
`max_cand` are all caller-supplied for the same reason: no defaults here would silently
encode a task-fitted value.

CRITICAL: the original engine's BK-tree pickle load was wrapped in a silent try/except that
once cost 5.7 official points when the pickle silently failed to load. This port fails
loudly: if `bktree_path` is given and unloadable, the exception propagates. Passing
`bktree_path=None` is the only supported way to run without a BK-tree.
"""
import collections, math, pickle, re


WORD = re.compile(r"^[^\W\d_]+$", re.UNICODE)
STRIP = ".,!?;:\"'()„“»«–—"


def unigram(csv_path, cap=400000):
    """Corpus-frequency prior from a monolingual corpus CSV (last column = text)."""
    import csv as _csv
    c = collections.Counter()
    with open(csv_path, encoding="utf-8") as fh:
        r = _csv.reader(fh); next(r, None)
        for row in r:
            for t in (row[-1] if row else "").split():
                t = t.strip(STRIP).lower()
                if t and WORD.match(t):
                    c[t] += 1
    tot = sum(c.values()) or 1
    return c, tot


class Engine:
    def __init__(self, lang, task, model, lexicon, bktree_path, batch, max_cand,
                 gc_paradigm_cap, sc_paradigm_cap, freq=None):
        self.lang, self.task, self.batch, self.max_cand = lang, task, batch, max_cand
        self.gc_paradigm_cap, self.sc_paradigm_cap = gc_paradigm_cap, sc_paradigm_cap
        self.hs = lexicon
        self.freq, self.tot = freq if freq is not None else (collections.Counter(), 1)
        self.bk = None
        if bktree_path is not None:
            with open(bktree_path, "rb") as fh:
                self.bk = pickle.load(fh)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto").eval()

    def logp(self, w):
        return math.log((self.freq.get(w.lower(), 0) + 0.5) / self.tot)

    def candidates(self, word):
        """Ranked candidate corrections for one surface word."""
        if not WORD.match(word) or len(word) < 3:
            return []
        cands = set()
        if self.task == "gc":
            cands.update(self.hs.paradigm(word, cap=self.gc_paradigm_cap))    # THE GC bet
        else:
            known = self.hs.known(word)
            cands.update(self.hs.diacritic_variants(word))        # audit: all diacritic errors missed
            if self.bk is not None:
                try:
                    for c in (self.bk.range(word, 1) if known else self.bk.range(word, 2)):
                        cands.add(c[0] if isinstance(c, (tuple, list)) else c)
                except Exception:
                    pass
            if not known:
                cands.update(self.hs.paradigm(word, cap=self.sc_paradigm_cap))
        cands = {c for c in cands if c != word and self.hs.known(c)}
        # noisy channel: corpus prior, preferring frequent real words
        return sorted(cands, key=lambda c: -self.logp(c))[:self.max_cand]

    def score_batch(self, sents):
        """Mean per-token loglikelihood of each sentence."""
        import torch
        out = []
        for s in range(0, len(sents), self.batch):
            b = sents[s:s + self.batch]
            enc = self.tok(b, return_tensors="pt", padding=True, truncation=True,
                           max_length=256).to(self.model.device)
            with torch.no_grad():
                lg = self.model(**enc).logits.float()
            lp = torch.log_softmax(lg, dim=-1)
            ids, am = enc["input_ids"], enc["attention_mask"]
            tgt = lp[:, :-1, :].gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            m = am[:, 1:].float()
            out += ((tgt * m).sum(1) / m.sum(1).clamp(min=1)).tolist()
        return out

    def run_sentence(self, sent):
        """Per-position candidate scoring + null-margin decision for one sentence."""
        toks = sent.split()
        variants, meta = [sent], []
        for i, t in enumerate(toks):
            core = t.strip(STRIP)
            if not core:
                continue
            pre = t[:t.index(core)] if core in t else ""
            suf = t[len(pre) + len(core):]
            for c in self.candidates(core):
                nt = toks[:]; nt[i] = pre + c + suf
                variants.append(" ".join(nt)); meta.append((core, c))
        if not meta:
            return ("CORRECT", "CORRECT", 0.0, 0)
        sc = self.score_batch(variants)
        null, best_i = sc[0], max(range(len(meta)), key=lambda k: sc[k + 1])
        return (meta[best_i][0], meta[best_i][1], sc[best_i + 1] - null, len(meta))

    def constrained_correction(self, sent, w, c):
        """Keep v5's flagged word `w` verbatim; choose its replacement from a constrained
        candidate set (v5's own correction `c` plus this engine's paradigm/diacritic/BK-tree
        candidates for `w`), scored by single-edit sentence loglikelihood.

        Ported from `scgc_correct_only.py`'s per-item loop, including its W0.2 no-op rule
        (`w == c` -> already CORRECT/CORRECT)."""
        if w == c:
            return "CORRECT", "CORRECT"
        cands = list(dict.fromkeys(([c] if c and c != "CORRECT" else []) + self.candidates(w)))
        cands = [x for x in cands if x]
        if cands and w in sent:
            variants = [sent.replace(w, x, 1) for x in cands]
            sc = self.score_batch(variants)
            best = cands[max(range(len(cands)), key=lambda i: sc[i])]
            return w, best
        return w, c
