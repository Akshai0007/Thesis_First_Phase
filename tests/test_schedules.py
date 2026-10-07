import pytest
from qwenlab.config import TrainConfig
from qwenlab.training.schedules import lr_multiplier
from qwenlab.registry import SCHEDULES


@pytest.mark.parametrize("schedule", SCHEDULES.names())
def test_schedule_starts_near_zero_and_reaches_one_after_warmup(schedule):
    cfg = TrainConfig(schedule=schedule, max_steps=1000, warmup_steps=100, min_lr_ratio=0.1)
    m0 = lr_multiplier(0, cfg)
    m_end_warmup = lr_multiplier(cfg.warmup_steps - 1, cfg)
    assert 0.0 < m0 <= 1.0 / cfg.warmup_steps + 1e-9
    assert abs(m_end_warmup - 1.0) < 1e-6, f"{schedule}: expected ~1.0 right at warmup end, got {m_end_warmup}"


@pytest.mark.parametrize("schedule", SCHEDULES.names())
def test_schedule_never_exceeds_one_or_drops_below_floor(schedule):
    cfg = TrainConfig(schedule=schedule, max_steps=500, warmup_steps=50, min_lr_ratio=0.1)
    for step in range(0, cfg.max_steps, 7):
        m = lr_multiplier(step, cfg)
        assert -1e-9 <= m <= 1.0 + 1e-9, f"{schedule} step {step}: multiplier {m} out of [0,1]"
        if step >= cfg.warmup_steps:
            assert m >= cfg.min_lr_ratio - 1e-6, f"{schedule} step {step}: {m} below floor {cfg.min_lr_ratio}"


def test_warmup_is_monotonically_increasing_for_every_schedule():
    for name in SCHEDULES.names():
        cfg = TrainConfig(schedule=name, max_steps=1000, warmup_steps=100)
        vals = [lr_multiplier(s, cfg) for s in range(cfg.warmup_steps)]
        assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1)), \
            f"{name}: warmup not monotonic: {vals[:5]}...{vals[-5:]}"


def test_cosine_and_linear_decay_monotonically_after_warmup():
    for name in ("cosine", "linear"):
        cfg = TrainConfig(schedule=name, max_steps=1000, warmup_steps=100, min_lr_ratio=0.05)
        sample_steps = list(range(cfg.warmup_steps, cfg.max_steps, 10))
        vals = [lr_multiplier(s, cfg) for s in sample_steps]
        assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1)), \
            f"{name}: post-warmup decay not monotonic"
        # check the TRUE final step separately -- the strided sample above may
        # land short of max_steps-1 (e.g. range(100,1000,10) ends at 990, not
        # 999), which would still be mid-decay and correctly not yet at the floor
        final = lr_multiplier(cfg.max_steps - 1, cfg)
        assert abs(final - cfg.min_lr_ratio) < 0.01, f"{name}: step max_steps-1 should reach the floor, got {final}"


def test_wsd_has_a_flat_stable_phase_between_warmup_and_decay():
    """The property that actually distinguishes WSD from cosine: multiplier
    stays pinned at 1.0 for a real stretch of steps, not just momentarily."""
    cfg = TrainConfig(schedule="wsd", max_steps=1000, warmup_steps=100, decay_fraction=0.1, min_lr_ratio=0.1)
    stable_region = [lr_multiplier(s, cfg) for s in range(200, 850, 50)]
    assert all(abs(v - 1.0) < 1e-6 for v in stable_region), f"WSD stable phase not flat: {stable_region}"
    # but it must still decay by the very end, same floor as the others
    assert abs(lr_multiplier(cfg.max_steps - 1, cfg) - cfg.min_lr_ratio) < 0.01


def test_constant_schedule_stays_at_one_after_warmup_regardless_of_max_steps():
    cfg = TrainConfig(schedule="constant", max_steps=100, warmup_steps=10)
    assert lr_multiplier(10_000, cfg) == 1.0  # far beyond max_steps -- constant shouldn't care
