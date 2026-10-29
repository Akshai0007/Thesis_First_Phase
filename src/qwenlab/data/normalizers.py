"""Text normalisers, chosen by name in the config (`data.normalizer`). Each is a function str -> str.

RULE: use the same normaliser for the tokenizer's training text and for the model's training text. If they differ
(e.g. old-style Malayalam chillu letters reach a tokenizer that never saw them) the text falls into byte-fallback
pieces: in a test, one short phrase went from 5 tokens to 12 (3 of them raw bytes).
"""
import re
import unicodedata

from ..registry import NORMALIZERS

_SPACES = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def _collapse(text: str) -> str:
    text = _SPACES.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


@NORMALIZERS.register("default")
def default(text: str) -> str:
    """Language-neutral: Unicode NFC, drop NUL, collapse spaces and blank lines."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\x00", "")
    return _collapse(text)


# Malayalam: consonant + virama (U+0D4D) + ZWJ (U+200D)  ->  atomic chillu letter
_CHILLU = {"\u0d23": "\u0d7a", "\u0d28": "\u0d7b", "\u0d30": "\u0d7c",
           "\u0d32": "\u0d7d", "\u0d33": "\u0d7e", "\u0d15": "\u0d7f"}
_CHILLU_RE = re.compile("([" + "".join(_CHILLU) + "])\u0d4d\u200d")
# control characters (except newline/tab), zero-width space, byte-order mark, soft hyphen
_JUNK = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b\ufeff\u00ad]")


@NORMALIZERS.register("malayalam")
def malayalam(text: str) -> str:
    """NFC (also composes two-part vowel signs), old-style chillu -> atomic chillu, remove junk characters.
    ZWJ / ZWNJ are otherwise KEPT: they carry meaning in Malayalam spelling.
    To use a custom normaliser instead, register it under this name, so that tokenizer and model data are cleaned
    by exactly the same code."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    text = _CHILLU_RE.sub(lambda m: _CHILLU[m.group(1)], text)
    text = _JUNK.sub("", text)
    return _collapse(text)


_SCRIPTS = {"malayalam": (0x0D00, 0x0D7F)}


def script_ratio(text: str, script: str) -> float:
    """Share of the letters in `text` that belong to `script`."""
    lo, hi = _SCRIPTS[script]
    letters = [c for c in text if c.isalpha()]
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters) if letters else 0.0
