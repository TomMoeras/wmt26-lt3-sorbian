#!/usr/bin/env python3
"""Config-driven inference driver: dispatches all five tasks (MT/QA/SC/GC/MR) for one shipped
system, writing OCELoT-schema `*_preds.jsonl` files via `package_ocelot.write_task_file`.

Ported from `scripts/phase6/blind_generate.py`. That script hard-coded a vendored data tree, an
absolute cluster `sys.path` insert, and mixed engineering knobs (`--gc-union-t`,
`--witness-params`, `--sc-lambda`) with the composed-stack on/off switch (`--composed`) as CLI
defaults -- exactly the "hand-typed threshold" the port rules forbid. Here a config YAML (see
`configs/*.yaml`, values extracted mechanically in `the working repo's .superpowers/sdd/task-9-report.md` Step 1)
is the single source of truth for every fitted number; `validate_config` asserts every key an
enabled lever needs is present BEFORE any GPU work starts, rather than failing deep inside a
stage with a bare `KeyError`.

Filesystem locations that are NOT fitted parameters -- dictionary/bktree paths, monolingual
corpora, the MT exemplar pool, MR dev files -- live in a separate `--resources` YAML (see
`Resources` below), so `configs/*.yaml` stays exactly what `tests/test_configs.py` validates:
fitted parameters only.

Everything GPU/model-dependent (`Gen`, and the `lt3wmt26.sc_gc_engine.Engine` /
`lt3wmt26.qa_slot_scorer.Scorer` this module calls into) lazily imports torch/transformers
inside its own methods, per the convention already established across this package -- so this
module, up to the point `main()` actually starts generating, imports and runs cleanly on a
torch-less machine (`python3 -c "import lt3wmt26.generate"` is part of the test suite).
"""
import argparse
import collections
import glob
import json
import os
import random
import re

REQUIRED_TOP = {"system_name", "weights", "stack", "mt", "qa", "gc", "sc", "mr", "seed"}

# Official blind test-file naming (organiser README Sec Output format / dummy_submission).
# MT direction files use the 3-letter 'deu' stem; the other four tasks are
# '<lang>_<task>_test.jsonl'.
MT_DIRS = [("de", "hsb", "deu-hsb_mt_test.jsonl"), ("hsb", "de", "hsb-deu_mt_test.jsonl"),
           ("de", "dsb", "deu-dsb_mt_test.jsonl"), ("dsb", "de", "dsb-deu_mt_test.jsonl"),
           ("hsb", "dsb", "hsb-dsb_mt_test.jsonl"), ("dsb", "hsb", "dsb-hsb_mt_test.jsonl")]
NAME = {"de": "German", "hsb": "Upper Sorbian", "dsb": "Lower Sorbian"}
NONMT_PATTERN = {"qa": "{lang}_qa_test.jsonl", "sc": "{lang}_sc_test.jsonl",
                 "gc": "{lang}_gc_test.jsonl", "mr": "{lang}_mr_test.jsonl"}


# --------------------------------------------------------------------------------------------
# config: load / validate (assert-fail on any key an ENABLED lever needs, before GPU work)
# --------------------------------------------------------------------------------------------

def load_config(path):
    import yaml
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    validate_config(cfg, path)
    return cfg


def validate_config(cfg, path="<config>"):
    missing = REQUIRED_TOP - set(cfg)
    assert not missing, f"{path}: config missing required top-level keys: {sorted(missing)}"
    assert cfg["stack"] in ("composed", "plain"), \
        f"{path}: stack must be 'composed' or 'plain', got {cfg['stack']!r}"

    w = cfg["weights"]
    for k in ("hf_repo", "subfolder"):
        assert k in w, f"{path}: weights.{k} missing"

    for k in ("decode", "exemplar_k", "pool"):
        assert k in cfg["mt"], f"{path}: mt.{k} missing"

    assert "slot_scorer" in cfg["qa"], f"{path}: qa.slot_scorer missing"

    assert "union" in cfg["gc"], f"{path}: gc.union missing"
    assert "paradigm_cap" in cfg["gc"], f"{path}: gc.paradigm_cap missing"
    if cfg["gc"]["union"]:
        wit = cfg["gc"].get("witness") or {}
        for lang in ("hsb", "dsb"):
            assert lang in wit and "w" in wit[lang] and "t" in wit[lang], \
                f"{path}: gc.witness.{lang}.{{w,t}} missing (required because gc.union is true)"

    assert "bktree" in cfg["sc"], f"{path}: sc.bktree missing"
    assert "paradigm_cap" in cfg["sc"], f"{path}: sc.paradigm_cap missing"
    nonword = ((cfg["sc"].get("topups") or {}).get("nonword")) or {}
    if nonword.get("enabled"):
        for k in ("tau_hsb", "tau_dsb"):
            assert k in nonword, \
                f"{path}: sc.topups.nonword.{k} missing (required because " \
                f"sc.topups.nonword.enabled is true)"

    for k in ("retry_ladder", "postformat", "retrieval"):
        assert k in cfg["mr"], f"{path}: mr.{k} missing"
    if cfg["mr"]["retrieval"].get("enabled"):
        for k in ("sim_threshold", "corpus"):
            assert k in cfg["mr"]["retrieval"], \
                f"{path}: mr.retrieval.{k} missing (required because mr.retrieval.enabled)"

    if cfg["stack"] == "composed":
        assert "batch" in cfg.get("engine", {}) and "max_cand" in cfg.get("engine", {}), \
            f"{path}: engine.{{batch,max_cand}} missing (required for the composed stack's " \
            f"sc_gc_engine.Engine)"


# --------------------------------------------------------------------------------------------
# test-file discovery (official filename patterns) + deployment-path resources
# --------------------------------------------------------------------------------------------

def discover_test_files(test_dir):
    """Map task -> list of (lang[, lang], path) using the official blind filename patterns.
    A task with no matching files is simply an empty list -- a caller running only `--tasks mt`
    against an MT-only test-dir is not forced to also have SC/GC/MR files present."""
    found = {"mt": [], "qa": [], "sc": [], "gc": [], "mr": []}
    for s, t, fn in MT_DIRS:
        p = os.path.join(test_dir, fn)
        if os.path.exists(p):
            found["mt"].append((s, t, p))
    for task, pattern in NONMT_PATTERN.items():
        for lang in ("hsb", "dsb"):
            p = os.path.join(test_dir, pattern.format(lang=lang))
            if os.path.exists(p):
                found[task].append((lang, p))
    return found


class Resources:
    """Deployment-specific filesystem paths: dictionaries, bktree pickles, monolingual corpora,
    the MT exemplar pool, MR dev files. These are NOT fitted parameters (they don't change the
    system's behaviour, only where its inputs live), so they are kept out of `configs/*.yaml`
    -- which `tests/test_configs.py` validates as fitted-parameter-only -- and live in a
    separate `--resources` YAML instead. A path missing for a lever the config actually enables
    fails loudly at first use; a path missing for a disabled lever is never touched."""

    def __init__(self, data, test_dir, pool_override=None):
        self.data = data or {}
        self.test_dir = test_dir
        self.pool_override = pool_override
        self.test_files = discover_test_files(test_dir)

    def _req(self, key):
        v = self.data.get(key)
        assert v, (f"--resources: missing required key '{key}' for an enabled lever "
                   f"(have: {sorted(self.data)})")
        return v

    def dict_aff(self, lang):
        return os.path.join(self._req("dict_dir"), f"{lang}.aff")

    def dict_dic(self, lang):
        return os.path.join(self._req("dict_dir"), f"{lang}.dic")

    def bktree_path(self, lang):
        bd = self.data.get("bktree_dir")
        if not bd:
            return None
        p = os.path.join(bd, f"{lang}.bktree.pkl")
        return p if os.path.exists(p) else None

    def mono_path(self, lang):
        return os.path.join(self._req("mono_dir"), f"{lang}_monolingual_2026.csv")

    def pool_dir(self):
        return self.pool_override or self._req("pool_dir")

    def mr_dev_path(self, lang):
        return os.path.join(self._req("mr_dev_dir"), f"{lang}_mr_dev.jsonl")


def resolve_weights(weights_cfg):
    """Resolve {hf_repo, subfolder} to a local `from_pretrained`-compatible directory.

    `hf_repo` may already be a local directory laid out like the HF repo (subfolder per system);
    otherwise it is downloaded from the Hub, restricted to just that subfolder's files. An empty
    `subfolder` means the whole repo IS the model (e.g. a plain base model like Qwen/Qwen3.5-2B
    for the paper ladder's base-model rows), so no subfolder is joined."""
    hf_repo, subfolder = weights_cfg["hf_repo"], weights_cfg.get("subfolder", "")
    if os.path.isdir(hf_repo):
        p = os.path.join(hf_repo, subfolder) if subfolder else hf_repo
        return p if os.path.isdir(p) else hf_repo
    from huggingface_hub import snapshot_download
    if not subfolder:
        return snapshot_download(hf_repo)
    snap = snapshot_download(hf_repo, allow_patterns=[f"{subfolder}/*"])
    return os.path.join(snap, subfolder)


# --------------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------------

def rd(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def wr(rows, task, out_path):
    from lt3wmt26.package_ocelot import write_task_file
    write_task_file(rows, task, out_path)
    print(f"  -> {out_path} ({len(rows)})", flush=True)


# --------------------------------------------------------------------------------------------
# prompts / extraction -- self-contained (no organiser-harness dependency), matching the
# official tag/format conventions
# --------------------------------------------------------------------------------------------

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_WRONG = re.compile(r"<wrong>\s*(.*?)\s*</wrong>")
_CORRECTED = re.compile(r"<corrected>\s*(.*?)\s*</corrected>")
_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

_SC_INSTR = ("The following sentence contains at most one misspelled word. Respond with exactly "
             "these two tags and nothing else:\n<wrong> misspelled word </wrong> <corrected> "
             "correct spelling </corrected>\nIf the sentence has no error, use the literal string "
             "CORRECT in both tags:\n<wrong> CORRECT </wrong> <corrected> CORRECT </corrected>")
_GC_INSTR = ("The following sentence contains at most one grammatically incorrect word. Respond with "
             "exactly these two tags and nothing else:\n<wrong> incorrect word </wrong> <corrected> "
             "correct form </corrected>\nIf the sentence has no error, use the literal string "
             "CORRECT in both tags:\n<wrong> CORRECT </wrong> <corrected> CORRECT </corrected>")
_MR_INSTR = ("Answer this mathematical reasoning question. Place your final answer, using LaTeX when "
             "appropriate, in this format: <answer> answer </answer>")
_QA_INSTR = "Answer the multiple-choice question with the number of the correct option only."


def strip_thinking(text):
    return _THINK.sub("", text).strip()


def mt_translation(text, tgt_lang):
    text = strip_thinking(text)
    m = re.search(rf"<{tgt_lang}>\s*([\s\S]*?)\s*(?:</{tgt_lang}>|$)", text)
    return (m.group(1) if m else text).strip()


def wrong_corrected(text):
    text = strip_thinking(text)
    w = _WRONG.search(text)
    c = _CORRECTED.search(text)
    return (w.group(1).strip() if w else "", c.group(1).strip() if c else "")


def mr_answer(text):
    text = strip_thinking(text)
    m = _ANSWER.search(text)
    return m.group(1).strip() if m else ""


def mr_dev_index(dev):
    """Whitespace-normalized (question -> answer) index for MR dev-retrieval. The answer is coerced
    to `str`: the organizer dev `answer` is a JSON number for ~half the items, while both the
    generation path (mr_answer(o).strip()) and the shipped submission emit strings. Writing the raw
    numeric value would make dev-retrieval rows differ from the shipped MR file on JSON type (24/500
    rows). See tests/test_mr.py::test_mr_dev_index_* and docs/REPRODUCIBILITY.md section 6."""
    return [(" ".join(x["question"].split()).lower(), str(x["answer"])) for x in dev]


def mt_prompt(source, src_name, tgt_name, exemplars):
    lines = []
    for ex_src, ex_tgt in exemplars:
        lines.append(f"{src_name}: {ex_src}")
        lines.append(f"{tgt_name}: {ex_tgt}")
    lines.append(f"{src_name}: {source}")
    lines.append(f"{tgt_name}:")
    body = "\n".join(lines)
    content = f"Translate the source text from {src_name} to {tgt_name}\n\n{body}"
    return [{"role": "user", "content": content}]


def sc_gc_prompt(sentence, task):
    instr = _SC_INSTR if task == "sc" else _GC_INSTR
    return [{"role": "user", "content": f"{instr}\n<sentence> {sentence} </sentence>"}]


def mr_prompt(question):
    return [{"role": "user", "content": f"{_MR_INSTR} {question}"}]


def qa_format_options(possible):
    return "\n".join(f"{k}. {v}" for k, v in sorted(possible.items(), key=lambda kv: int(kv[0])))


def qa_prompt(item):
    ctx = (item.get("context") or "").strip()
    block = ((f"{ctx}\n" if ctx else "") + f"Question: {item['question']}\n"
             f"{qa_format_options(item['possible_answers'])}\nAnswer:")
    return [{"role": "user", "content": f"{_QA_INSTR}\n\n{block}"}]


def qa_parse(raw):
    m = re.search(r"-?\d+", raw)
    return int(m.group()) if m else -1


# --------------------------------------------------------------------------------------------
# Gen: batched chat-template generation over one HF model (lazy torch/transformers import)
# --------------------------------------------------------------------------------------------

class Gen:
    def __init__(self, model_path, batch=32):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map="auto").eval()
        self.batch = batch

    def render(self, msgs):
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                            enable_thinking=False)

    def generate_texts(self, texts, max_new, sample=False, temp=0.0, min_new=0):
        torch = self.torch
        out = []
        for s in range(0, len(texts), self.batch):
            b = texts[s:s + self.batch]
            enc = self.tok(b, return_tensors="pt", padding=True, truncation=True,
                           max_length=3072).to(self.model.device)
            kw = dict(max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id)
            if min_new:
                kw["min_new_tokens"] = min_new
            kw.update(dict(do_sample=True, temperature=temp, top_p=0.95) if sample
                      else dict(do_sample=False))
            with torch.no_grad():
                o = self.model.generate(**enc, **kw)
            out += [self.tok.decode(o[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    for j in range(len(b))]
        return out

    def __call__(self, msgs_list, max_new, sample=False, temp=0.0, min_new=0):
        return self.generate_texts([self.render(m) for m in msgs_list], max_new,
                                   sample=sample, temp=temp, min_new=min_new)


# --------------------------------------------------------------------------------------------
# per-task dispatch
# --------------------------------------------------------------------------------------------

def run_mt(cfg, resources, args, gen):
    from lt3wmt26.retrieval.exemplars import CharSimRetriever, MiniLMRetriever
    assert cfg["mt"]["decode"] == "greedy", (
        "only greedy MT decoding is a fitted, gated lever (MBR was rejected on gold dev: "
        f"mean -0.38); config asked for decode={cfg['mt']['decode']!r}")
    k = cfg["mt"]["exemplar_k"]
    for s_c, t_c, path in resources.test_files["mt"]:
        items = rd(path)
        ex = [[] for _ in items]
        if k > 0:
            pool_dir = resources.pool_dir()
            slug = f"{s_c}_{t_c}"
            ps = [l.rstrip("\n") for l in open(os.path.join(pool_dir, f"{slug}.src"), encoding="utf-8")]
            pt = [l.rstrip("\n") for l in open(os.path.join(pool_dir, f"{slug}.tgt"), encoding="utf-8")]
            retr = CharSimRetriever(ps, pt) if s_c in ("hsb", "dsb") else MiniLMRetriever(ps, pt)
            ex = [retr.top_k(d[s_c], k) for d in items]
            print(f"  [mt/{slug}] exemplars k={k} from pool ({len(ps)} pairs)", flush=True)
        msgs = [mt_prompt(d[s_c], NAME[s_c], NAME[t_c], e) for d, e in zip(items, ex)]
        outs = gen(msgs, 320)
        rows = [{"dataset_id": d.get("dataset_id"), "sent_id": d.get("sent_id"),
                 "source": d[s_c], "pred": mt_translation(o, t_c)}
                for d, o in zip(items, outs)]
        fname = f"{s_c if s_c != 'de' else 'deu'}-{t_c if t_c != 'de' else 'deu'}_preds.jsonl"
        wr(rows, "mt", os.path.join(args.out_dir, fname))


def run_qa(cfg, resources, args, gen, weights_path):
    from lt3wmt26.qa_slot_scorer import target_gap, fill, GAP, solve_block, Scorer
    for lang, path in resources.test_files["qa"]:
        items = rd(path)
        outs = gen([qa_prompt(d) for d in items], 16)
        rows = []
        for d, o in zip(items, outs):
            n_opt = len(d.get("possible_answers") or {})
            p = qa_parse(strip_thinking(o))
            p = p if (isinstance(p, int) and 1 <= p <= n_opt) else 1
            rows.append({"dataset_id": d.get("dataset_id"), "question_id": d["question_id"],
                         "question": d.get("question"), "pred": int(p)})

        if cfg["stack"] == "composed" and cfg["qa"]["slot_scorer"]:
            cs = Scorer(weights_path)
            blocks = collections.defaultdict(list)
            for d in items:
                gap_id = target_gap(d.get("question"))
                if gap_id and gap_id in GAP.findall(d.get("context") or ""):
                    blocks[(d.get("context") or "")[:400]].append((d, gap_id))
            override = {}
            for _, blk in blocks.items():
                mat, meta = [], []
                for d, gap_id in blk:
                    opts = [str(x) for x in sorted(d["possible_answers"].keys(), key=lambda x: int(x))]
                    sc_ = cs.score([fill(d.get("context") or "", gap_id, d["possible_answers"][o])
                                    for o in opts])
                    override[d["question_id"]] = int(opts[max(range(len(opts)), key=lambda i: sc_[i])])
                    mat.append(sc_)
                    meta.append((d, opts))
                shared = (len({tuple(sorted(d["possible_answers"].items())) for d, _ in meta}) == 1
                          and len(meta) > 1)
                if shared and mat:
                    picks = solve_block(mat)
                    for (d, opts), pick in zip(meta, picks):
                        if pick is not None:
                            override[d["question_id"]] = int(opts[pick])
            n_ov = 0
            for r in rows:
                if r["question_id"] in override:
                    r["pred"] = override[r["question_id"]]
                    n_ov += 1
            print(f"  [qa/{lang}] in-place slot override applied to {n_ov} items", flush=True)

        assert len(rows) == len(items), "FATAL: QA missing predictions"
        wr(rows, "qa", os.path.join(args.out_dir, f"{lang}_qa_preds.jsonl"))


def run_sc_gc(cfg, resources, args, gen, weights_path, task):
    from lt3wmt26.lexicon import Hunspell
    from lt3wmt26.sc_gc_engine import Engine, unigram
    from lt3wmt26.gc_union import arbitrate
    from lt3wmt26.sc_topups import nonword_topup, margin_topup
    from lt3wmt26 import witness as witness_mod

    for lang, path in resources.test_files[task]:
        items = rd(path)
        outs = gen([sc_gc_prompt(d["input_sentence"], task) for d in items], 64)
        raw = [wrong_corrected(o) for o in outs]

        if cfg["stack"] == "composed":
            hs = Hunspell(resources.dict_aff(lang), resources.dict_dic(lang))
            bktree_path = None
            if task == "sc" and cfg["sc"]["bktree"]:
                bktree_path = resources.bktree_path(lang)
                assert bktree_path is not None, (
                    f"sc.bktree is true in the config but no usable BK-tree pickle was found "
                    f"for lang={lang!r}. Set 'bktree_dir' in --resources (see "
                    f"resources.yaml.template) to a directory containing '{lang}.bktree.pkl', "
                    f"built with 'python3 -m lt3wmt26.build_bktree' (see scripts/setup.sh). "
                    f"Running the primary config with sc.bktree: true and no BK-tree silently "
                    f"changes the SC candidate set (see docs/REPRODUCIBILITY.md); that failure mode "
                    f"must not pass silently here."
                )
            eng = Engine(lang=lang, task=task, model=weights_path, lexicon=hs,
                        bktree_path=bktree_path,
                        batch=cfg["engine"]["batch"], max_cand=cfg["engine"]["max_cand"],
                        gc_paradigm_cap=cfg["gc"]["paradigm_cap"],
                        sc_paradigm_cap=cfg["sc"]["paradigm_cap"])
            union = task == "gc" and cfg["gc"]["union"]
            bigrams = None
            if union:
                w_wt = cfg["gc"]["witness"][lang]["w"]
                t_wt = cfg["gc"]["witness"][lang]["t"]
                bigrams = witness_mod.load_bigrams(resources.mono_path(lang))

            fixed = []
            for d, (w0, c0) in zip(items, raw):
                sent = d["input_sentence"]
                co_w, co_c = eng.constrained_correction(sent, w0 or "CORRECT", c0 or "CORRECT")
                if union:
                    ew, ec, margin, _ = eng.run_sentence(sent)
                    row = {"v5_w": w0 or "CORRECT", "v5_c": c0 or "CORRECT",
                           "eng_w": ew, "eng_c": ec, "margin": margin,
                           "co_w": co_w, "co_c": co_c}
                    wit_fn = lambda r, _s=sent: witness_mod.witness(bigrams, _s, r["eng_w"], r["eng_c"])
                    fw, fc = arbitrate(row, t_wt, w_wt, wit_fn)
                else:
                    fw, fc = co_w, co_c
                fixed.append((fw or "CORRECT", fc or "CORRECT"))
            raw = fixed

            if task == "sc":
                topcfg = ((cfg["sc"].get("topups") or {}).get("diacritic") or {}).get(lang, {})
                if topcfg.get("enabled"):
                    freq, _tot = unigram(resources.mono_path(lang))
                    guards = {"exactly_one": True, "skip_caps": True, "skip_quoted": True}
                    fired, new_raw = 0, []
                    for d, (w0, c0) in zip(items, raw):
                        if w0 == "CORRECT":
                            hit = nonword_topup(d["input_sentence"], hs, freq,
                                                topcfg["min_freq"], guards)
                            if hit:
                                w0, c0 = hit
                                fired += 1
                        new_raw.append((w0, c0))
                    raw = new_raw
                    print(f"  [{task}/{lang}] diacritic top-up fired on {fired}/{len(items)} "
                          f"(min_freq={topcfg['min_freq']})", flush=True)

                # tau-gated engine-margin top-up (apply_scv4_topup.py -- sc-v4, the shipped
                # primary's lever). Empirically it is chained AFTER the diacritic top-up above
                # (both gate on "still abstained", so order only matters where they'd compete
                # for the same item): the working repo's .superpowers/sdd/task-9-report.md fix section reconstructs
                # the shipped primary SC file byte-for-byte as sc-v2 (this engine, bktree-
                # enabled) -> diacritic top-up -> this margin top-up, with sc-v3's guarded hsb
                # diacritic top-up confirmed NOT part of that chain (a rejected sibling probe).
                topcfg_nw = (cfg["sc"].get("topups") or {}).get("nonword") or {}
                if topcfg_nw.get("enabled"):
                    tau = topcfg_nw[f"tau_{lang}"]
                    guards = {"skip_caps": True, "skip_quoted": True}
                    fired, new_raw = 0, []
                    for d, (w0, c0) in zip(items, raw):
                        if w0 == "CORRECT":
                            ew, ec, margin, _ = eng.run_sentence(d["input_sentence"])
                            hit = margin_topup(ew, ec, margin, d["input_sentence"], tau, guards)
                            if hit:
                                w0, c0 = hit
                                fired += 1
                        new_raw.append((w0, c0))
                    raw = new_raw
                    print(f"  [{task}/{lang}] nonword margin top-up fired on {fired}/{len(items)} "
                          f"(tau={tau})", flush=True)

        rows = [{"dataset_id": d.get("dataset_id"),
                 # SC/GC test ids are ints in the organizer files, but the frozen submission
                 # serialized them as strings ("0"); coerce so regenerated output matches.
                 "id": (str(d["id"]) if d.get("id") is not None else None),
                 "input_sentence": d["input_sentence"],
                 "pred_incorrect": w or "CORRECT", "pred_corrected": c or "CORRECT"}
                for d, (w, c) in zip(items, raw)]
        wr(rows, task, os.path.join(args.out_dir, f"{lang}-{task}_preds.jsonl"))


def run_mr(cfg, resources, args, gen):
    from lt3wmt26.mr_pipeline import tier_of, postformat, retrieval_layer, retry_ladder

    for lang, path in resources.test_files["mr"]:
        items = rd(path)
        msgs = [mr_prompt(d["question"]) for d in items]
        texts = [gen.render(m) for m in msgs]
        outs = gen.generate_texts(texts, 640)

        if cfg["stack"] == "composed" and cfg["mr"]["retry_ladder"]:
            def _run(sub_texts, max_new, sample=False, temp=0.0, min_new=0):
                return gen.generate_texts(sub_texts, max_new, sample=sample, temp=temp,
                                          min_new=min_new)
            outs, _stage = retry_ladder(
                outs, texts, _run, mr_answer, 640,
                log=lambda att, n: print(f"  [mr/{lang}] retry {att}: {n} empty", flush=True))

        rows = []
        for d, o in zip(items, outs):
            ans = mr_answer(o).strip()
            if cfg["stack"] == "composed" and cfg["mr"]["postformat"]:
                ans = postformat(ans, tier_of(d))
            rows.append({"dataset_id": d.get("dataset_id"), "id": d.get("id"),
                         "question": d.get("question"), "pred": ans})

        if cfg["stack"] == "composed" and cfg["mr"]["retrieval"].get("enabled"):
            dev = rd(resources.mr_dev_path(lang))
            dev_index = mr_dev_index(dev)   # coerces dev 'answer' to str (see mr_dev_index docstring)
            sim_t = cfg["mr"]["retrieval"]["sim_threshold"]
            n_hit = 0
            for r, d in zip(rows, items):
                hit = retrieval_layer(d["question"], dev_index, sim_t)
                if hit is not None:
                    r["pred"] = hit
                    n_hit += 1
            print(f"  [mr/{lang}] dev-retrieval hits: {n_hit}/{len(rows)}", flush=True)

        n_empty = sum(1 for r in rows if not r["pred"])
        print(f"  [mr/{lang}] empty={n_empty}/{len(rows)}", flush=True)
        wr(rows, "mr", os.path.join(args.out_dir, f"{lang}-mr_preds.jsonl"))


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="One of configs/*.yaml.")
    ap.add_argument("--test-dir", required=True, help="Directory of official blind test files.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="Overrides config['seed'] if given. Only the MR stage-3 sampled retry "
                         "is stochastic; everything else is greedy.")
    ap.add_argument("--pool-override", default=None,
                    help="Directory of <src>_<tgt>.src/.tgt pairs to use as the MT exemplar "
                         "pool instead of --resources' pool_dir -- run_on_new_testset.sh points "
                         "this at a freshly deduped pool built for the new test set's own "
                         "blocklist (see retrieval/pool.py).")
    ap.add_argument("--resources", default="resources.yaml",
                    help="YAML of deployment paths: dict_dir, bktree_dir, mono_dir, pool_dir, "
                         "mr_dev_dir. Filesystem locations, not fitted parameters -- kept "
                         "separate from --config.")
    ap.add_argument("--tasks", default="mt,qa,sc,gc,mr")
    ap.add_argument("--batch", type=int, default=32)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    random.seed(seed)

    res_data = {}
    if os.path.exists(args.resources):
        import yaml
        res_data = yaml.safe_load(open(args.resources, encoding="utf-8")) or {}
    resources = Resources(res_data, args.test_dir, pool_override=args.pool_override)

    os.makedirs(args.out_dir, exist_ok=True)
    weights_path = resolve_weights(cfg["weights"])

    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    want = set(args.tasks.split(","))
    # Fail loudly BEFORE the expensive model load if the test-dir yields no files for the requested
    # tasks. discover_test_files() otherwise returns empty lists silently, so a wrong --test-dir
    # would burn a full model load and emit an empty or partial bundle (the silent-failure class
    # the repo guards against elsewhere, e.g. bktree_dir).
    if not any(resources.test_files.get(t) for t in want):
        raise SystemExit(
            f"--test-dir {args.test_dir!r} contains no official test files for tasks {sorted(want)}. "
            f"Expected all test files FLAT in one directory: 6 MT <src>-<tgt>_mt_test.jsonl plus "
            f"{{hsb,dsb}}_{{qa,sc,gc,mr}}_test.jsonl. scripts/setup.sh builds this as official_test_flat/.")
    _missing = sorted(t for t in want if not resources.test_files.get(t))
    if _missing:
        print(f"[generate] WARNING: no test files for {_missing} in {args.test_dir!r} -- "
              f"the assembled bundle will be INCOMPLETE.", flush=True)

    gen = Gen(weights_path, batch=args.batch)
    print(f"[generate] {cfg['system_name']} stack={cfg['stack']} seed={seed}", flush=True)

    if "mt" in want and resources.test_files["mt"]:
        run_mt(cfg, resources, args, gen)
    if "qa" in want and resources.test_files["qa"]:
        run_qa(cfg, resources, args, gen, weights_path)
    for task in ("sc", "gc"):
        if task in want and resources.test_files[task]:
            run_sc_gc(cfg, resources, args, gen, weights_path, task)
    if "mr" in want and resources.test_files["mr"]:
        run_mr(cfg, resources, args, gen)


if __name__ == "__main__":
    main()
