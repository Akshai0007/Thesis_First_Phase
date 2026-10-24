from qwenlab.budget import compute_max_steps


def test_fits_inside_the_budget():
    # 3 s/step, 5 h budget, nothing done yet -> about 5*3600/(3*1.15) = 5217 steps
    assert abs(compute_max_steps(20000, 0, 0, 3.0, 5 * 3600) - 5217) <= 1


def test_never_exceeds_the_target_and_never_goes_below_the_floor():
    assert compute_max_steps(1000, 0, 0, 1.0, 10 ** 6) == 1000
    assert compute_max_steps(20000, 0, 10 ** 6, 3.0, 10) == 600


def test_counts_steps_already_done():
    assert compute_max_steps(20000, 4000, 12000, 3.0, 12000 + 3.45 * 1000) == 5000
