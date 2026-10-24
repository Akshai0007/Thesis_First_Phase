"""fim_arrays: fixed-shape FIM training arrays from packed windows."""
from qwenlab.tasks.fim import fim_arrays

PRE, MID, SUF = 901, 902, 903
L = 64


def _windows(n, seed=0):
    import random
    r = random.Random(seed)
    return [[r.randrange(1, 800) for _ in range(L + 1)] for _ in range(n)]


def test_shapes_and_mask_cover_only_the_middle():
    xs, ys, ms = fim_arrays(_windows(40), L, PRE, MID, SUF, seed=1)
    assert all(len(x) == len(y) == len(m) == L for x, y, m in zip(xs, ys, ms))
    for x, y, m in zip(xs, ys, ms):
        full = x + [y[-1]]                                   # reconstruct the full id sequence
        mid_pos = full.index(MID)
        assert [i + 1 for i, v in enumerate(m) if v] == list(range(mid_pos + 1, L + 1))   # targets = tokens after <fim_middle>
        assert m[mid_pos - 1] is False                       # the <fim_middle> marker itself is never a loss target
        assert m[mid_pos] is True                            # ...but the position AT the marker predicts the first middle token


def test_all_three_markers_present_exactly_once_and_original_tokens_recoverable():
    ws = _windows(40, seed=2)
    xs, ys, ms = fim_arrays(ws, L, PRE, MID, SUF, seed=1)
    for w, x, y in zip(ws, xs, ys):
        full = x + [y[-1]]
        assert full.count(PRE) == full.count(MID) == full.count(SUF) == 1
        body = [t for t in full if t not in (PRE, MID, SUF)]
        # prefix + middle + suffix were rotated; the SAME tokens must all be there
        assert sorted(body) == sorted(w[:L + 1 - 3])


def test_deterministic_and_both_orderings_occur():
    ws = _windows(60, seed=3)
    a = fim_arrays(ws, L, PRE, MID, SUF, seed=1)
    assert a == fim_arrays(ws, L, PRE, MID, SUF, seed=1)
    firsts = {x[0] for x in a[0]}
    assert firsts == {PRE, SUF}                              # PSM and SPM both used
