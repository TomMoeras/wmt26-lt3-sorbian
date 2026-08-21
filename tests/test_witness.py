from lt3wmt26.witness import witness


def test_witness_prefers_attested_candidate():
    big = {("je", "dobra"): 5, ("dobra", "kniha"): 4, ("je", "dobru"): 0, ("dobru", "kniha"): 0}
    w = witness(big, "to je dobru kniha", "dobru", "dobra")
    assert w > 0  # candidate attested in context, observed not


def test_witness_symmetric_zero():
    assert witness({}, "a b c", "b", "b") == 0.0
