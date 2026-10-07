"""Tiny generic registry. Every swappable component (model, init scheme, tokenizer,
data source, task, LR schedule, eval task) registers itself under a string name, and
configs refer to components ONLY by that name. That is what makes "change the model /
dataset / tokenizer without touching anything else" a config edit instead of a code edit.
"""
from typing import Any, Callable, Dict


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Any] = {}

    def register(self, name: str) -> Callable:
        def deco(obj):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name]

    def names(self):
        return sorted(self._items)

    def __contains__(self, name):
        return name in self._items


MODELS = Registry("model")
INIT_SCHEMES = Registry("init scheme")
TOKENIZERS = Registry("tokenizer")
SOURCES = Registry("data source")
TASKS = Registry("task")
SCHEDULES = Registry("lr schedule")
MC_TASKS = Registry("multiple-choice eval task")
GEN_TASKS = Registry("generation eval task")
