from lt3wmt26.gc_union import arbitrate


def test_v5_flag_takes_engine_correction():
    row = {"v5_w": "Europjanom", "v5_c": "Europjanow", "eng_w": "Europjanom",
           "eng_c": "Europjan", "margin": 0.19}
    assert arbitrate(row, t=0.42, w=0.1, witness_fn=lambda *a: 0.0) == ("Europjanom", "Europjan")


def test_abstain_below_threshold():
    row = {"v5_w": "CORRECT", "v5_c": "CORRECT", "eng_w": "x", "eng_c": "y", "margin": 0.05}
    assert arbitrate(row, t=0.42, w=0.1, witness_fn=lambda *a: 0.0) == ("CORRECT", "CORRECT")
