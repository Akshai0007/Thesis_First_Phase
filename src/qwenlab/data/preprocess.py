"""Cleaning for the next-word-prediction (causal LM) stream.

The stream must be CONTINUOUS TEXT: the model learns language structure from running prose. Question/answer
pairs, chat templates and fill-in-the-middle markers do not belong in it. `is_task_formatted` detects them and
the data builder DROPS such documents (and reports how many) instead of silently training on them.
The language-specific part (normalisation, script filter) is chosen in the config, not in code.
"""
import re

from . import normalizers
from .normalizers import script_ratio
from ..registry import NORMALIZERS

normalize = normalizers.default          # kept for callers that import normalize from here

# chat-template / FIM markers, and explicit "Question: ... Answer: ..." layouts
_MARKERS = ("<|im_start|>", "<|im_end|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>")
_QA_LAYOUT = re.compile(r"(^|\n)\s*(question|q)\s*[:.]\s.+\n+\s*(answer|a)\s*[:.]\s", re.IGNORECASE)


def is_task_formatted(text: str) -> bool:
    return any(m in text for m in _MARKERS) or bool(_QA_LAYOUT.search(text))


def clean_document(text: str, min_chars: int = 200, normalizer: str = "default",
                   script: str = "", min_script_ratio: float = 0.0, drop_qa_format: bool = True):
    """Returns (cleaned_text or None, reason or None); reason is 'short', 'qa_format' or 'wrong_script'."""
    text = NORMALIZERS.get(normalizer)(text)
    if len(text) < min_chars:
        return None, "short"
    if drop_qa_format and is_task_formatted(text):
        return None, "qa_format"
    if script and script_ratio(text, script) < min_script_ratio:
        return None, "wrong_script"
    return text, None
