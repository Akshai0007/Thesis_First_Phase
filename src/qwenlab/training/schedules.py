"""Learning-rate schedules. Each is a pure function (step, config) -> multiplier
in [0, 1], applied as lr = base_lr * multiplier -- no optimizer/model coupling,
so these are trivially unit-testable without ever building a model.

Includes WSD (Warmup-Stable-Decay) alongside the more common cosine, since
WSD is what several recent small-model reproductions (e.g. MiniCPM, and
variants of the SmolLM2 recipe referenced during research for this project)
use specifically because it lets you extend a training run without having
pre-committed to a total step count the way a cosine schedule's decay curve
requires -- relevant here since Kaggle session limits make "how many steps
will we actually get" a real unknown, not just a hyperparameter choice.
"""
import math

from ..registry import SCHEDULES


@SCHEDULES.register("constant")
def constant(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / max(cfg.warmup_steps, 1)
    return 1.0


@SCHEDULES.register("linear")
def linear(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return max(cfg.min_lr_ratio, 1.0 - progress * (1.0 - cfg.min_lr_ratio))


@SCHEDULES.register("cosine")
def cosine(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos_decay


@SCHEDULES.register("wsd")
def wsd(step: int, cfg) -> float:
    """Warmup -> Stable (flat at 1.0) -> Decay (last `decay_fraction` of steps,
    cosine down to min_lr_ratio). The 'stable' phase is what makes this
    resumable/extendable in a way cosine isn't: stopping early just means a
    shorter stable phase, not a schedule that never reached its intended low
    point."""
    if step < cfg.warmup_steps:
        return (step + 1) / max(cfg.warmup_steps, 1)
    decay_start = cfg.max_steps * (1.0 - cfg.decay_fraction)
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / max(cfg.max_steps - decay_start, 1)
    progress = min(max(progress, 0.0), 1.0)
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos_decay


def lr_multiplier(step: int, train_cfg) -> float:
    fn = SCHEDULES.get(train_cfg.schedule)
    return fn(step, train_cfg)
