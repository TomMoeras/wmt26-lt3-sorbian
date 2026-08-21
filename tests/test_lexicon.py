from lt3wmt26.lexicon import Hunspell


def test_parses_minimal_dic(tmp_path):
    aff = tmp_path / "t.aff"
    dic = tmp_path / "t.dic"
    aff.write_text("SET UTF-8\n")
    dic.write_text("2\nwoda\nknihi\n")
    h = Hunspell(str(aff), str(dic))
    assert h.known("woda") and not h.known("xyzq")
