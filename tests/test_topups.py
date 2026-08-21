from lt3wmt26.sc_topups import nonword_topup, margin_topup


class FakeLex:
    def known(self, w): return w in {"dalšna", "woda", "Zdzary"}
    def diacritic_variants(self, w): return {"dalšna"} if w == "dalsna" else set()


FREQ = {"dalšna": 9}
G = {"exactly_one": True, "skip_caps": True, "skip_quoted": True}


def test_fires_on_single_nonword_with_frequent_variant():
    r = nonword_topup("to jo dalsna wěc", FakeLex(), FREQ, tau=5, guards=G)
    assert r == ("dalsna", "dalšna")


def test_cap_guard_blocks_place_name():
    assert nonword_topup("we wsi Zdzary jo", FakeLex(), FREQ, tau=5, guards=G) is None or \
           nonword_topup("we wsi Zdzary jo", FakeLex(), FREQ, tau=5, guards=G)[0] != "Zdzary"


def test_tau_floor_blocks_rare_variant():
    assert nonword_topup("to jo dalsna wěc", FakeLex(), {"dalšna": 1}, tau=5, guards=G) is None


# ---- margin_topup (apply_scv4_topup.py's tau-gated engine-margin top-up, sc-v4) ----

MG = {"skip_caps": True, "skip_quoted": True}


def test_margin_topup_fires_when_margin_clears_tau():
    assert margin_topup("njemože", "njemóže", 0.35, "to njemože być", tau=0.30, guards=MG) \
        == ("njemože", "njemóže")


def test_margin_topup_abstains_below_tau():
    assert margin_topup("njemože", "njemóže", 0.20, "to njemože być", tau=0.30, guards=MG) is None


def test_margin_topup_abstains_on_no_word_or_already_correct():
    assert margin_topup(None, None, 0.99, "to jo derje", tau=0.30, guards=MG) is None
    assert margin_topup("CORRECT", "CORRECT", 0.99, "to jo derje", tau=0.30, guards=MG) is None


def test_margin_topup_cap_guard_blocks_mid_sentence_proper_noun():
    assert margin_topup("Budyšin", "Budyšinu", 0.99, "wón bydli w Budyšin", tau=0.30, guards=MG) \
        is None


def test_margin_topup_cap_guard_allows_sentence_initial_capital():
    assert margin_topup("Budyšin", "Budyšinu", 0.99, "Budyšin je stolica", tau=0.30, guards=MG) \
        == ("Budyšin", "Budyšinu")


def test_margin_topup_quote_guard_blocks_quoted_form():
    assert margin_topup("swět", "swjat", 0.99, 'won groni "swět" cyle wěsće', tau=0.30,
                        guards=MG) is None
