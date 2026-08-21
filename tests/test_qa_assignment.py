import pytest

from lt3wmt26.qa_slot_scorer import solve_block, fill


def test_assignment_resolves_conflict():
    # Hungarian assignment is a hard-scipy path (part of the shipped primary); skip cleanly
    # in a bare env rather than exercising a downgrade that no longer exists.
    pytest.importorskip("scipy")
    # two slots both prefer option 0, joint solve must split them optimally
    m = [[0.9, 0.8, 0.1], [0.85, 0.2, 0.1]]
    picks = solve_block(m)
    assert picks == [1, 0]                       # joint optimum; greedy conflict gives [0, 1]
    assert sum(m[i][p] for i, p in enumerate(picks)) > 0.9 + 0.2   # strictly beats greedy sum


def test_fill_substitutes_marker():
    assert "wam" in fill("kak móžu ( I ) dźakować", "I", "wam")
