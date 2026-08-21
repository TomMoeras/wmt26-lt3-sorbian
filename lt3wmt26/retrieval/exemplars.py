#!/usr/bin/env python3
"""MT exemplar retrieval: CharSim for Sorbian-source directions (no MiniLM language coverage),
MiniLM/FAISS semantic retrieval for German-source directions -- matching phase4a's dispatch
(`src/eval/official`'s `_mt_exemplars` / `blind_generate.py`'s `mt-exemplars` branch: `if
src_code in ("hsb", "dsb"): CharSimRetriever ... else: FuzzyMatcher`).

Ported from `src/extract_fuzzy/char_matcher.py` (`CharSimRetriever`, kept verbatim -- pure
Python, no third-party dependency) and `src/extract_fuzzy/fuzzy_matcher.py` (`FuzzyMatcher`,
reduced to the reusable retrieval class: the source file's CLI harness, FAISS index
save/load-to-disk, and `process_real_dataset` batch-file mode are cluster I/O orchestration and
are not part of this module).

`MiniLMRetriever.top_k` mirrors `FuzzyMatcher.find_fuzzy_matches`'s exclusion rule (a query that
IS a pool entry, or matches one at near-zero distance, must not retrieve itself) and returns
`(src, tgt)` pairs like `CharSimRetriever`, so `generate.py`'s exemplar-building code can treat
both retrievers identically regardless of which one a direction dispatches to.

`faiss` / `sentence-transformers` are imported lazily, inside `MiniLMRetriever` methods, so this
module -- and `CharSimRetriever` -- stay usable on a machine that has neither installed.
"""
from typing import List, Tuple


class CharSimRetriever:
    """Lexical (character-set Jaccard) retriever -- used for Sorbian-source MT directions."""

    def __init__(self, pool_src: List[str], pool_tgt: List[str]):
        assert len(pool_src) == len(pool_tgt)
        self.pool_src = [s.strip() for s in pool_src]
        self.pool_tgt = [t.strip() for t in pool_tgt]
        # Pre-compute char-sets once so top_k is O(N) instead of O(N*M).
        self._char_sets = [set(s) for s in self.pool_src]

    def top_k(self, query: str, k: int) -> List[Tuple[str, str]]:
        if k <= 0:
            return []
        q = query.strip()
        sq = set(q)
        scored = []
        for i, cs in enumerate(self._char_sets):
            if self.pool_src[i] == q:            # exclude the query itself
                continue
            if not cs:                           # skip empty-string pool entries
                continue
            inter = len(sq & cs)
            if not inter:                        # skip non-overlapping entries
                continue
            sim = inter / max(len(sq), len(cs))
            scored.append((sim, i))
        scored.sort(key=lambda x: x[0], reverse=True)        # best (highest sim) first
        return [(self.pool_src[i], self.pool_tgt[i]) for _, i in scored[:k]]


class MiniLMRetriever:
    """Semantic (MiniLM + FAISS IVF) retriever -- used for German-source MT directions.

    Ported from `FuzzyMatcher`: same encoder (`microsoft/Multilingual-MiniLM-L12-H384`), same
    index shape (`IndexIVFFlat`, up to 4096 clusters capped at `pool_size // 4`, `nprobe=32`),
    same over-fetch-then-filter retrieval (`k*3` candidates, drop the query's own near-zero-
    distance match, keep the first `k` survivors). Deterministic given a fixed pool and query
    (no sampling anywhere in this path).
    """

    def __init__(self, pool_src: List[str], pool_tgt: List[str],
                 model_name: str = "microsoft/Multilingual-MiniLM-L12-H384"):
        assert len(pool_src) == len(pool_tgt)
        self.pool_src = [s.strip() for s in pool_src]
        self.pool_tgt = [t.strip() for t in pool_tgt]
        self.model_name = model_name
        self._model = None
        self._index = None

    def _build_index(self):
        import faiss
        from sentence_transformers import SentenceTransformer
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cuda" if faiss.get_num_gpus() > 0 else "cpu"
        self._model = SentenceTransformer(self.model_name, device=device)
        emb = self._model.encode(self.pool_src, show_progress_bar=False)
        dim = emb.shape[1]
        n_clusters = min(4096, max(1, len(emb) // 4))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
        index.train(emb)
        index.add(emb)
        index.nprobe = 32
        self._index = index

    def top_k(self, query: str, k: int) -> List[Tuple[str, str]]:
        if k <= 0:
            return []
        if self._index is None:
            self._build_index()
        q_emb = self._model.encode([query])
        distances, ids = self._index.search(q_emb, k * 3)      # over-fetch, then filter
        q_norm = query.strip().lower()
        out = []
        for d, idx in zip(distances[0], ids[0]):
            if idx < 0:
                continue
            src = self.pool_src[idx]
            if src.lower() == q_norm or d < 1e-18:              # exclude self / near-duplicate
                continue
            out.append((src, self.pool_tgt[idx]))
            if len(out) == k:
                break
        return out
