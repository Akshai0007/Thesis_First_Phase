"""Time budgeting for session-limited training (Kaggle)."""


def compute_max_steps(target: int, step_now: int, elapsed_s: float, s_per_step: float, budget_s: float,
                      overhead: float = 1.15, floor: int = 600) -> int:
    """Largest total step count that still finishes inside the time budget.
    `overhead` covers checkpoint writes, uploads and evaluations (they make real steps slower than the
    raw training speed). Never above `target`, never below `floor` (or the current step)."""
    extra = int(max(budget_s - elapsed_s, 0) / (s_per_step * overhead))
    return int(min(target, max(floor, step_now + extra)))
