"""Resume-safe, deterministic batch feeding for multi-task training.

Moved out of the training notebook so the data side is a real, tested module: the model, tokenizer,
dataset and task can be swapped without touching the trainer. Pure Python + numpy (no torch).

Design rule: everything that decides WHAT the next batch is depends only on the global batch number
(and a seed). That is what makes an exact resume possible: after a restart we recompute how many
batches each stream has already supplied and continue from there.
"""
from collections import Counter
from typing import Callable, Dict, List, Union

import numpy as np


def _mix32(x: int) -> int:
    """32-bit integer hash (the 'lowbias32' avalanche mix): close inputs give unrelated outputs."""
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def uniform01(b: int, seed: int, salt: int = 0) -> float:
    """Deterministic pseudo-random number in [0, 1) for global batch number b.
    Different salts give independent-looking streams (used for 'is it a task batch?' and 'which task?')."""
    return _mix32(_mix32(b + seed * 0x9E3779B1) + salt * 0x85EBCA6B) / 4294967296.0


class ResumableBatcher:
    """Deterministic shuffled epochs over `items`. `start_batches` = batches already consumed (for resuming)."""

    def __init__(self, items, batch_size: int, make_batch: Callable, seed: int = 42, start_batches: int = 0):
        self.items, self.bs, self.make_batch, self.seed = items, batch_size, make_batch, seed
        self.n = len(items)
        assert self.n >= batch_size, "fewer items than one batch"
        self.pos = start_batches * batch_size
        self._epoch, self._perm = None, None

    def _perm_for(self, epoch: int):
        if self._epoch != epoch:
            self._perm = np.random.RandomState(self.seed + epoch).permutation(self.n)
            self._epoch = epoch
        return self._perm

    def __call__(self):
        idx = []
        for _ in range(self.bs):
            epoch, i = divmod(self.pos, self.n)
            idx.append(int(self._perm_for(epoch)[i]))
            self.pos += 1
        return self.make_batch([self.items[j] for j in idx])


class StagePlan:
    """Which stream supplies global batch number b?
       stage 1 (b < switch_b): always 'lm' (plain next-word prediction)
       stage 2: with probability `share` a task batch (task chosen by `weights`), otherwise 'lm'.
    switch_b may be an int or a callable (so it can follow a plan that is still being fitted to the time budget)."""

    def __init__(self, switch_b: Union[int, Callable[[], int]], share: float, weights: Dict[str, float], seed: int):
        assert 0.0 <= share <= 1.0 and weights and all(w > 0 for w in weights.values())
        self.switch_b, self.share, self.seed = switch_b, share, seed
        self.names = sorted(weights)
        total = float(sum(weights.values()))
        self.cum, c = [], 0.0
        for n in self.names:
            c += weights[n] / total
            self.cum.append(c)

    def _switch(self) -> int:
        return int(self.switch_b() if callable(self.switch_b) else self.switch_b)

    def choose(self, b: int) -> str:
        if b < self._switch():
            return "lm"
        if uniform01(b, self.seed, 1) >= self.share:
            return "lm"
        u = uniform01(b, self.seed, 2)
        for name, c in zip(self.names, self.cum):
            if u < c:
                return name
        return self.names[-1]

    def count_before(self, n_batches: int) -> Dict[str, int]:
        """how many batches each stream supplied among batches 0..n_batches-1 (used to resume exactly)"""
        counts = Counter(self.choose(b) for b in range(n_batches))
        return {n: counts.get(n, 0) for n in ["lm"] + self.names}


class MixedBatcher:
    """Calls the stream chosen by the StagePlan for each global batch. `fns` maps stream name -> batch function."""

    def __init__(self, fns: Dict[str, Callable], plan: StagePlan, start_batches: int = 0):
        missing = set(["lm"] + plan.names) - set(fns)
        assert not missing, f"no batch function for streams: {sorted(missing)}"
        self.fns, self.plan, self.b = fns, plan, start_batches

    def __call__(self):
        name = self.plan.choose(self.b)
        self.b += 1
        return self.fns[name]()


def make_cycler(batches: List):
    """returns a function that serves the given (fixed) batches round-robin: used for validation sets"""
    st = {"i": 0}

    def fn():
        b = batches[st["i"] % len(batches)]
        st["i"] += 1
        return b
    return fn
