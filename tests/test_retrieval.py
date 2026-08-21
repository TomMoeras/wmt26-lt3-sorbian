import json

from lt3wmt26.retrieval.pool import blocklist_from_jsonl, dedup_pool, norm, main as pool_main
from lt3wmt26.retrieval.exemplars import CharSimRetriever


def test_norm_collapses_whitespace():
    assert norm("  a   b\tc\n") == "a b c"


def test_blocklist_from_jsonl_reads_first_present_field(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"source": "hi tola"}\n{"input_sentence": "wona jo"}\n', encoding="utf-8")
    block = blocklist_from_jsonl([str(p)], ["source", "input_sentence"])
    assert block == {"hi tola", "wona jo"}


def test_blocklist_from_jsonl_skips_missing_files():
    assert blocklist_from_jsonl(["/no/such/file.jsonl"], ["source"]) == set()


def test_dedup_pool_drops_source_or_target_match():
    src = ["a", "b", "c"]
    tgt = ["x", "y", "z"]
    pairs, report = dedup_pool(src, tgt, blocklist={"a", "z"}, cap=100)
    assert pairs == [("b", "y")]
    assert report["pool_original"] == 3
    assert report["removed_by_test_dedup"] == 2
    assert report["residual_test_material"] == 0


def test_dedup_pool_dev_exclusion_then_cap():
    src = ["a", "b", "c", "d"]
    tgt = ["w", "x", "y", "z"]
    pairs, report = dedup_pool(src, tgt, blocklist=set(), dev_src={"b"}, cap=2)
    assert [s for s, _ in pairs] == ["a", "c"]
    assert report["removed_by_dev_exclusion"] == 1
    assert report["removed_by_cap"] == 1
    assert report["shipped"] == 2


def test_dedup_pool_residual_is_zero_when_clean():
    src, tgt = ["m", "n"], ["p", "q"]
    _, report = dedup_pool(src, tgt, blocklist={"zzz"}, cap=10)
    assert report["residual_test_material"] == 0


def test_char_sim_retriever_excludes_self_and_ranks_by_overlap():
    pool_src = ["kelko haus", "totally different text", "kelko jajow"]
    pool_tgt = ["t1", "t2", "t3"]
    retr = CharSimRetriever(pool_src, pool_tgt)
    top = retr.top_k("kelko haus", 2)
    assert ("kelko haus", "t1") not in top          # exact self-match excluded
    assert top[0] == ("kelko jajow", "t3")           # highest char overlap ranked first


def test_char_sim_retriever_k_zero_returns_empty():
    retr = CharSimRetriever(["a"], ["b"])
    assert retr.top_k("a", 0) == []


# --------------------------- pool.py --dedup-against CLI (run_on_new_testset.sh) ---------------

def _write_pool(pool_dir, slug, src_lines, tgt_lines):
    pool_dir.mkdir(exist_ok=True)
    (pool_dir / f"{slug}.src").write_text("\n".join(src_lines) + "\n", encoding="utf-8")
    (pool_dir / f"{slug}.tgt").write_text("\n".join(tgt_lines) + "\n", encoding="utf-8")


def test_cli_dedups_pool_against_new_mt_test_file(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "deu-hsb_mt_test.jsonl").write_text(
        json.dumps({"de": "blocked source"}) + "\n", encoding="utf-8")

    pool_dir = tmp_path / "pool"
    _write_pool(pool_dir, "de_hsb", ["blocked source", "keep this one"], ["t1", "t2"])
    out_dir = tmp_path / "out"

    pool_main(["--dedup-against", str(test_dir), "--pool", str(pool_dir), "--out", str(out_dir)])

    assert (out_dir / "de_hsb.src").read_text(encoding="utf-8").splitlines() == ["keep this one"]
    assert (out_dir / "de_hsb.tgt").read_text(encoding="utf-8").splitlines() == ["t2"]


def test_cli_excludes_qa_from_blocklist(tmp_path):
    # pool.py's module docstring: the ported blocklist is "six specific MT test files plus
    # SC/GC/MR test inputs" -- QA is deliberately not part of it.
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "hsb_qa_test.jsonl").write_text(
        json.dumps({"question": "keep this one"}) + "\n", encoding="utf-8")

    pool_dir = tmp_path / "pool"
    _write_pool(pool_dir, "de_hsb", ["keep this one"], ["t1"])
    out_dir = tmp_path / "out"

    pool_main(["--dedup-against", str(test_dir), "--pool", str(pool_dir), "--out", str(out_dir)])

    assert (out_dir / "de_hsb.src").read_text(encoding="utf-8").splitlines() == ["keep this one"]


def test_cli_skips_src_file_with_no_matching_tgt(tmp_path, capsys):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (pool_dir / "orphan.src").write_text("x\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    pool_main(["--dedup-against", str(test_dir), "--pool", str(pool_dir), "--out", str(out_dir)])

    assert not (out_dir / "orphan.tgt").exists()
    assert "skipped" in capsys.readouterr().out
