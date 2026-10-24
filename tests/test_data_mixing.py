"""Resume-safe batch feeding and task mixing (pure Python/numpy)."""
import numpy as np

from qwenlab.data.mixing import MixedBatcher, ResumableBatcher, StagePlan, make_cycler, uniform01

W = {"summarize": 0.25, "instruct": 0.25, "qa": 0.20, "correction": 0.15, "fim": 0.15}


def test_uniform01_is_deterministic_and_in_range():
    xs = [uniform01(b, 7) for b in range(2000)]
    assert xs == [uniform01(b, 7) for b in range(2000)]
    assert all(0.0 <= x < 1.0 for x in xs)
    assert abs(np.mean(xs) - 0.5) < 0.02


def test_salts_give_independent_streams():
    a = np.array([uniform01(b, 3, 1) for b in range(20000)])
    c = np.array([uniform01(b, 3, 2) for b in range(20000)])
    assert abs(np.corrcoef(a, c)[0, 1]) < 0.03
    # task choice must not depend on the "is it a task batch" draw: look only at batches that ARE task batches
    sel = a < 0.5
    assert abs(c[sel].mean() - 0.5) < 0.02


def test_stage_one_is_pure_lm_and_stage_two_follows_the_weights():
    plan = StagePlan(switch_b=1000, share=0.5, weights=W, seed=42)
    assert all(plan.choose(b) == "lm" for b in range(1000))
    counts = plan.count_before(101_000)
    stage2 = 100_000
    assert abs(counts["lm"] - 1000 - 0.5 * stage2) < 0.015 * stage2
    for name, w in W.items():
        assert abs(counts[name] - 0.5 * w * stage2) < 0.012 * stage2, (name, counts[name])


def test_switch_can_follow_a_changing_plan():
    state = {"switch": 10_000}
    plan = StagePlan(switch_b=lambda: state["switch"], share=1.0, weights=W, seed=1)
    assert plan.choose(500) == "lm"
    state["switch"] = 100                      # the plan was shortened (fitted to the time budget)
    assert plan.choose(500) != "lm"


def test_resumable_batcher_resumes_exactly():
    items = list(range(37))
    full = ResumableBatcher(items, 4, lambda xs: tuple(xs), seed=5)
    seq = [full() for _ in range(30)]          # crosses an epoch boundary (37 items, 4 per batch)
    resumed = ResumableBatcher(items, 4, lambda xs: tuple(xs), seed=5, start_batches=12)
    assert [resumed() for _ in range(18)] == seq[12:]


def test_mixed_batcher_resumes_exactly_across_all_streams():
    def build(start):
        plan = StagePlan(switch_b=50, share=0.6, weights=W, seed=9)
        counts = plan.count_before(start)
        fns = {n: ResumableBatcher(list(range(100 * i, 100 * i + 23)), 2, lambda xs, n=n: (n, tuple(xs)),
                                    seed=i, start_batches=counts[n])
               for i, n in enumerate(["lm"] + plan.names)}
        return MixedBatcher(fns, plan, start_batches=start)
    full = build(0)
    seq = [full() for _ in range(400)]
    for k in (0, 49, 50, 137, 333):
        resumed = build(k)
        assert [resumed() for _ in range(400 - k)] == seq[k:], f"resume at {k} differs"


def test_cycler_cycles():
    f = make_cycler([1, 2, 3])
    assert [f() for _ in range(7)] == [1, 2, 3, 1, 2, 3, 1]
