"""Measures looping text. Used to test whether repetition comes from running past the training context or
from the model itself (see scripts/generate.py --repetition-check)."""


def repeated_ngram_fraction(ids, n: int = 4) -> float:
    """Fraction of n-grams that already occurred earlier in the same sequence. 0 = no loops, near 1 = stuck in a loop."""
    ids = list(ids)
    if len(ids) < n + 1:
        return 0.0
    seen, repeats, total = set(), 0, 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        total += 1
        if g in seen:
            repeats += 1
        seen.add(g)
    return repeats / total
